"""Simulate immunotherapy on human LUAD tumors initialized from CyCIF data.

Uses KP-calibrated parameters (v6 posterior mean) but initializes the ABM
grid with cell compositions derived from Gaglia 2023 Dataset05 (14 human
LUAD patients, CyCIF spatial phenotyping).

Hypothesis H6: Human tumors respond better to ICB than KP mice due to more
favorable CD8:Treg ratio and higher immune activation potential.

Usage:
    python3.11 scripts/human_luad_sweep.py \
        --posterior outputs/bayesian_inference_v6/posterior_samples.npy \
        --seeds 20 --out outputs/human_luad_sweep
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "gaglia_abm"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runs.bayesian_inference import (
    PARAM_NAMES, FIXED_PARAMS, TICKS_PER_WEEK,
)
from luad.model import LUADModel, PresetConfig
from luad.metrics import MetricsTracker
from luad.agents import AgentType

# ---------- Constants ----------
TOTAL_WEEKS = 9
TOTAL_TICKS = TOTAL_WEEKS * TICKS_PER_WEEK
WK7_START = 7 * TICKS_PER_WEEK
WK8_END = 8 * TICKS_PER_WEEK

# Treatment arms: same protocol as sequential dosing (1-week pulse at wk 7)
Schedule = list[tuple[int, int, list[str]]]
ARMS: dict[str, Schedule] = {
    "untreated": [],
    "PD1_CTLA4": [
        (WK7_START, WK8_END, ["PD1", "CTLA4"]),
    ],
    "PD1_IL15": [
        (WK7_START, WK8_END, ["PD1_IL15"]),
    ],
}

# Grid size matches KP calibration
GRID_W, GRID_H = 100, 100
GRID_TOTAL = GRID_W * GRID_H


# ---------- Patient data extraction ----------
def load_patient_compositions(data_dir: Path) -> list[dict]:
    """Extract per-patient cell type fractions from Gaglia Dataset05.

    Returns list of dicts with keys: name, tumor, cd8, cd4, treg, macrophage
    (as fractions of total cells).
    """
    import h5py
    import scipy.io as sio

    quant = data_dir / "Quantification"

    # Sample index per cell
    f = h5py.File(quant / "Results_Aggr_20210619.mat", "r")
    sample_idx = f["AggrResults"]["Indexes"][0]  # (7.8M,)
    nuc_sign = f["AggrResults"]["MeanNucSign"]  # (36, 7.8M)

    # Marker channels (0-indexed): CD4=1, FOXP3=6, CD8a=7, TTF1=9,
    # CD45=14, CD163=17, CD68=18, CD3D=22, Keratin=30
    cd4 = nuc_sign[1, :]
    foxp3 = nuc_sign[6, :]
    cd8a = nuc_sign[7, :]
    ttf1 = nuc_sign[9, :]
    cd163 = nuc_sign[17, :]
    cd68 = nuc_sign[18, :]
    cd3d = nuc_sign[22, :]
    keratin = nuc_sign[30, :]

    def threshold(arr):
        nonzero = arr[arr > 0]
        if len(nonzero) == 0:
            return 1
        return np.percentile(nonzero, 75)

    # Binary phenotyping
    tumor = (keratin > threshold(keratin)) | (ttf1 > threshold(ttf1))
    t_cell = (cd3d > threshold(cd3d)) & ~tumor
    cd8_t = t_cell & (cd8a > threshold(cd8a))
    cd4_t = t_cell & (cd4 > threshold(cd4)) & ~(cd8a > threshold(cd8a))
    treg = cd4_t & (foxp3 > threshold(foxp3))
    cd4_helper = cd4_t & ~(foxp3 > threshold(foxp3))
    macrophage = ((cd68 > threshold(cd68)) | (cd163 > threshold(cd163))) & ~tumor & ~t_cell

    f.close()

    # Sample names
    settings = sio.loadmat(quant / "Results_Settings_20210619.mat")
    tissues = settings["filename"][0, 0]["tissues"]

    patients = []
    for s in range(1, 15):
        mask = sample_idx == s
        n = mask.sum()
        name = str(tissues[0, s - 1].flat[0])
        patients.append({
            "name": name,
            "n_cells": int(n),
            "tumor_frac": float(tumor[mask].sum() / n),
            "cd8_frac": float(cd8_t[mask].sum() / n),
            "cd4_frac": float(cd4_helper[mask].sum() / n),
            "treg_frac": float(treg[mask].sum() / n),
            "macrophage_frac": float(macrophage[mask].sum() / n),
        })

    return patients


def patient_to_initial_agents(patient: dict, grid_total: int) -> dict[str, int]:
    """Convert patient cell fractions to ABM initial agent counts.

    We fill the grid proportionally to the patient's composition.
    Cells not classified (stroma, other) become empty grid space.
    """
    classified_frac = (patient["tumor_frac"] + patient["cd8_frac"] +
                       patient["cd4_frac"] + patient["treg_frac"] +
                       patient["macrophage_frac"])
    # Scale so classified cells fill ~80% of grid (leave room for dynamics)
    scale = min(0.80 / classified_frac, 1.0) if classified_frac > 0 else 0.5

    return {
        "tumor": int(round(patient["tumor_frac"] * scale * grid_total)),
        "cd8": int(round(patient["cd8_frac"] * scale * grid_total)),
        "cd4": int(round(patient["cd4_frac"] * scale * grid_total)),
        "treg": int(round(patient["treg_frac"] * scale * grid_total)),
        "macrophage": int(round(patient["macrophage_frac"] * scale * grid_total)),
    }


# ---------- Simulation ----------
def run_single_sim(params_vec: np.ndarray, initial_agents: dict[str, int],
                   schedule: Schedule, seed: int,
                   extra_params: dict | None = None) -> dict:
    """Run one ABM sim with patient-specific initial composition."""
    p = dict(zip(PARAM_NAMES, params_vec))
    preset = PresetConfig(
        name="human_luad",
        grid_width=GRID_W,
        grid_height=GRID_H,
        initial_agents=initial_agents,
        cxcl9_10_mean=0.12,
        macrophage_bias=-0.1,
        macrophage_m2_fraction=0.45,
        suppression={
            "treg": float(p["treg_suppression"]),
            "macrophage": float(p["macrophage_suppr_base"]),
        },
        suppressive_background=float(p["suppressive_background"]),
    )

    tracker = MetricsTracker(distance_interval=999, interaction_interval=999,
                             grid_interval=999, capture_grids=False)
    model = LUADModel(preset=preset, interventions=[], seed=seed,
                      metrics_tracker=tracker)

    # Override with posterior parameters
    for key in PARAM_NAMES:
        if key not in ("treg_suppression", "treg_death_rate"):
            model.params[key] = float(p[key])
    for key, val in FIXED_PARAMS.items():
        model.params[key] = val
    model.params["treg_death_rate"] = float(p["treg_death_rate"])

    # Antigen-driven CD8 proliferation (set via extra_params)
    if extra_params:
        for k, v in extra_params.items():
            if k not in model.params:
                raise KeyError(
                    f"extra_params key '{k}' not in model.params — "
                    f"did you modify the wrong codebase? "
                    f"Scripts use gaglia_abm/, not luad_abm/"
                )
            model.params[k] = v

    # Pre-compute treatment events
    start_events, end_events = {}, {}
    for start_tick, end_tick, interventions in schedule:
        start_events.setdefault(start_tick, []).extend(interventions)
        end_events.setdefault(end_tick, []).extend(interventions)

    for tick in range(TOTAL_TICKS):
        if tick in start_events:
            model.apply_interventions(start_events[tick])
        if tick in end_events:
            model.remove_interventions(end_events[tick])
        model.step()

    def count(at):
        return sum(1 for a in model.agents if getattr(a, "agent_type", None) == at)

    return {
        "n_tumor": count(AgentType.TUMOR),
        "n_cd8": count(AgentType.CD8),
        "n_cd4": count(AgentType.CD4),
        "n_treg": count(AgentType.TREG),
        "n_macrophage": count(AgentType.MACROPHAGE),
        "seed": seed,
    }


def _run_seed(args):
    """Worker function for multiprocessing."""
    params_vec, init_agents, schedule, seed, extra_params = args
    return run_single_sim(params_vec, init_agents, schedule, seed, extra_params)


def run_all(params_vec: np.ndarray, patients: list[dict],
            n_seeds: int, out_dir: Path = None,
            n_workers: int = 1,
            extra_params: dict | None = None) -> dict:
    """Run all patients × arms × seeds, saving incrementally."""
    results = {}

    # Also run KP baseline for comparison
    kp_agents = {"tumor": 1200, "cd8": 421, "cd4": 936,
                 "treg": 127, "macrophage": 88}
    all_configs = [("KP_mouse", kp_agents)] + [
        (pt["name"], patient_to_initial_agents(pt, GRID_TOTAL))
        for pt in patients
    ]

    executor = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None

    for config_name, init_agents in all_configs:
        for arm_name, schedule in ARMS.items():
            label = f"{config_name}__{arm_name}"
            print(f"\n=== {label} ({n_seeds} seeds) ===", flush=True)
            print(f"    agents: {init_agents}", flush=True)

            seeds = [1000 + s for s in range(n_seeds)]

            if executor is not None:
                futures = {
                    executor.submit(_run_seed, (params_vec, init_agents, schedule, seed, extra_params)): seed
                    for seed in seeds
                }
                seed_results = []
                for future in as_completed(futures):
                    result = future.result()
                    seed_results.append(result)
                    print(f"  seed {result['seed']}: tumor={result['n_tumor']}, "
                          f"cd8={result['n_cd8']}, treg={result['n_treg']}", flush=True)
                results[label] = sorted(seed_results, key=lambda r: r["seed"])
            else:
                results[label] = []
                for seed in seeds:
                    result = run_single_sim(params_vec, init_agents, schedule, seed, extra_params)
                    results[label].append(result)
                    print(f"  seed {seed}: tumor={result['n_tumor']}, cd8={result['n_cd8']}, "
                          f"treg={result['n_treg']}", flush=True)

        # Save checkpoint after each config completes (all arms)
        if out_dir is not None:
            ckpt_path = out_dir / "human_luad_results_checkpoint.npz"
            np.savez(ckpt_path,
                     **{k: [r for r in v] for k, v in results.items()})
            print(f"  [checkpoint saved: {len(results)} conditions]", flush=True)

    if executor is not None:
        executor.shutdown(wait=False)

    return results


def make_figure(results: dict, patients: list[dict], n_seeds: int, out_dir: Path):
    """Generate comparison figure: KP vs human patients."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect per-config treatment response
    configs = ["KP_mouse"] + [pt["name"] for pt in patients]
    arm_names = list(ARMS.keys())

    # Compute tumor reduction % for each config × arm
    tumor_data = {}
    for config in configs:
        baseline_key = f"{config}__untreated"
        baseline_tumor = np.mean([r["n_tumor"] for r in results[baseline_key]])
        tumor_data[config] = {"baseline": baseline_tumor}
        for arm in arm_names:
            key = f"{config}__{arm}"
            vals = [r["n_tumor"] for r in results[key]]
            mean_t = np.mean(vals)
            std_t = np.std(vals)
            delta = (mean_t - baseline_tumor) / baseline_tumor * 100
            tumor_data[config][arm] = {
                "mean": mean_t, "std": std_t, "delta": delta,
            }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Human LUAD vs KP Mouse: Immunotherapy Response\n"
        f"KP-calibrated parameters ({len(PARAM_NAMES)}-param posterior mean), {n_seeds} seeds/condition, "
        f"1-week pulse at wk 7, measured wk 9",
        fontsize=12, fontweight="bold",
    )

    x = np.arange(len(configs))

    # Panel A: Tumor reduction (%) — PD1+CTLA4
    ax = axes[0, 0]
    deltas = [tumor_data[c]["PD1_CTLA4"]["delta"] for c in configs]
    colors = ["#e37777" if d > 0 else "#6baed6" for d in deltas]
    colors[0] = "#fd8d3c"  # KP in orange
    ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Tumor change vs untreated (%)")
    ax.set_title("A. PD1+CTLA4 combo efficacy")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(deltas[0], color="#fd8d3c", linestyle="--", alpha=0.5,
               label=f"KP: {deltas[0]:+.1f}%")
    ax.legend(fontsize=8)

    # Panel B: Tumor reduction (%) — PD1/IL-15
    ax = axes[0, 1]
    deltas_il15 = [tumor_data[c]["PD1_IL15"]["delta"] for c in configs]
    colors_il15 = ["#e37777" if d > 0 else "#2ca02c" for d in deltas_il15]
    colors_il15[0] = "#fd8d3c"
    ax.bar(x, deltas_il15, color=colors_il15, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Tumor change vs untreated (%)")
    ax.set_title("B. PD1/IL-15 fusion efficacy")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(deltas_il15[0], color="#fd8d3c", linestyle="--", alpha=0.5,
               label=f"KP: {deltas_il15[0]:+.1f}%")
    ax.legend(fontsize=8)

    # Panel C: Response vs CD8:Treg ratio
    ax = axes[1, 0]
    cd8_treg_ratios = [0.0] * len(configs)
    cd8_treg_ratios[0] = 421 / 127  # KP
    for i, pt in enumerate(patients):
        cd8_treg_ratios[i + 1] = pt["cd8_frac"] / (pt["treg_frac"] + 1e-9)
    # Plot PD1+CTLA4 response vs ratio
    ax.scatter(cd8_treg_ratios[1:], [deltas[i] for i in range(1, len(configs))],
               c="#6baed6", edgecolor="black", s=60, zorder=3, label="Human patients")
    ax.scatter([cd8_treg_ratios[0]], [deltas[0]],
               c="#fd8d3c", edgecolor="black", s=100, marker="D", zorder=4, label="KP mouse")
    # Add patient labels
    for i, name in enumerate(configs[1:], 1):
        ax.annotate(name, (cd8_treg_ratios[i], deltas[i]),
                    fontsize=6, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("Initial CD8:Treg ratio")
    ax.set_ylabel("Tumor change (%) — PD1+CTLA4")
    ax.set_title("C. CD8:Treg ratio predicts PD1+CTLA4 response")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8)

    # Panel D: Response vs CD8:Treg ratio — PD1/IL-15
    ax = axes[1, 1]
    ax.scatter(cd8_treg_ratios[1:], [deltas_il15[i] for i in range(1, len(configs))],
               c="#2ca02c", edgecolor="black", s=60, zorder=3, label="Human patients")
    ax.scatter([cd8_treg_ratios[0]], [deltas_il15[0]],
               c="#fd8d3c", edgecolor="black", s=100, marker="D", zorder=4, label="KP mouse")
    for i, name in enumerate(configs[1:], 1):
        ax.annotate(name, (cd8_treg_ratios[i], deltas_il15[i]),
                    fontsize=6, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("Initial CD8:Treg ratio")
    ax.set_ylabel("Tumor change (%) — PD1/IL-15")
    ax.set_title("D. CD8:Treg ratio predicts PD1/IL-15 response")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig_path = out_dir / "human_luad_sweep.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}", flush=True)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=str, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--data-dir", type=str,
                        default="data/gaglia_2023/Dataset05_Human_Lung_Adenocarcinoma")
    parser.add_argument("--workers", type=int,
                        default=max(1, os.cpu_count() - 1),
                        help="Number of parallel workers (default: ncpus-1)")
    parser.add_argument("--kill-prolif", type=float, default=0.0,
                        help="CD8 kill-triggered proliferation probability (0=off)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # Load posterior
    samples = np.load(args.posterior)
    params_mean = samples.mean(axis=0).astype(np.float32)
    print(f"Posterior mean ({len(params_mean)} params):")
    for name, val in zip(PARAM_NAMES, params_mean):
        print(f"  {name:30s} = {val:.4f}")

    # Load patient compositions
    print("\nLoading patient CyCIF data...")
    patients = load_patient_compositions(data_dir)
    print(f"\n{'Patient':<10} {'Total':>8} {'Tumor%':>8} {'CD8%':>6} {'Treg%':>6} "
          f"{'Mac%':>6} {'CD8:Treg':>9}")
    print("-" * 60)
    for pt in patients:
        ratio = pt["cd8_frac"] / (pt["treg_frac"] + 1e-9)
        print(f"{pt['name']:<10} {pt['n_cells']:>8,} {pt['tumor_frac']*100:>7.1f}% "
              f"{pt['cd8_frac']*100:>5.1f}% {pt['treg_frac']*100:>5.1f}% "
              f"{pt['macrophage_frac']*100:>5.1f}% {ratio:>8.2f}")

    # Show ABM initial agents for each patient
    print(f"\nABM initial agents (100×100 grid):")
    print(f"{'Patient':<10} {'Tumor':>6} {'CD8':>6} {'CD4':>6} {'Treg':>5} {'Mac':>5} {'Total':>6}")
    print("-" * 50)
    for pt in patients:
        agents = patient_to_initial_agents(pt, GRID_TOTAL)
        total = sum(agents.values())
        print(f"{pt['name']:<10} {agents['tumor']:>6} {agents['cd8']:>6} {agents['cd4']:>6} "
              f"{agents['treg']:>5} {agents['macrophage']:>5} {total:>6}")

    # Build extra params
    extra_params = {}
    if args.kill_prolif > 0:
        extra_params["cd8_kill_prolif_prob"] = args.kill_prolif
        print(f"\nCD8 kill-triggered proliferation: {args.kill_prolif}")

    # Run sweep
    print(f"\nUsing {args.workers} parallel workers", flush=True)
    results = run_all(params_mean, patients, args.seeds, out_dir=out_dir,
                      n_workers=args.workers,
                      extra_params=extra_params if extra_params else None)

    # Save raw results
    np.savez(out_dir / "human_luad_results.npz",
             **{k: [r for r in v] for k, v in results.items()})

    # Summary table
    print(f"\n{'='*100}")
    print("Human LUAD vs KP Mouse — Treatment Response Summary (week 9)")
    print(f"{'='*100}")
    configs = ["KP_mouse"] + [pt["name"] for pt in patients]
    print(f"{'Config':<12} {'Arm':<14} {'Tumor (mean±sd)':<18} {'CD8':<8} {'Treg':<8} "
          f"{'CD8:Treg':<10} {'ΔTumor%':<10}")
    print("-" * 100)
    for config in configs:
        baseline_key = f"{config}__untreated"
        baseline = np.mean([r["n_tumor"] for r in results[baseline_key]])
        for arm in ARMS:
            key = f"{config}__{arm}"
            runs = results[key]
            tumors = [r["n_tumor"] for r in runs]
            cd8s = [r["n_cd8"] for r in runs]
            tregs = [r["n_treg"] for r in runs]
            ratios = [c / (t + 1e-9) for c, t in zip(cd8s, tregs)]
            tm, ts = np.mean(tumors), np.std(tumors)
            delta = (tm - baseline) / baseline * 100 if arm != "untreated" else 0.0
            print(f"{config:<12} {arm:<14} {tm:7.0f} ± {ts:5.0f}   {np.mean(cd8s):6.0f}  "
                  f"{np.mean(tregs):6.0f}  {np.mean(ratios):8.2f}  {delta:+8.1f}%",
                  flush=True)
        print()

    # Figure
    try:
        make_figure(results, patients, args.seeds, out_dir)
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
