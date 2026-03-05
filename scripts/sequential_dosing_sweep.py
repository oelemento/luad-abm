"""Sequential dosing sweep: PD1 vs CTLA4 ordering in KP mice.

Tests 6 treatment arms with two CTLA4 modes (suppression-only vs ADCC).
All arms start treatment at week 7, measured at week 9.

Usage:
    python3.11 scripts/sequential_dosing_sweep.py \
        --posterior outputs/bayesian_inference_v6/posterior_samples.npy \
        --seeds 20 --out outputs/sequential_dosing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "gaglia_abm"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runs.bayesian_inference import (
    PARAM_NAMES, FIXED_PARAMS, make_preset, TICKS_PER_WEEK,
)
from luad.model import LUADModel
from luad.metrics import MetricsTracker
from luad.agents import AgentType


# Treatment schedule: list of (start_tick, end_tick, interventions_to_apply)
# Interventions are applied at start_tick and removed at end_tick.
Schedule = list[tuple[int, int, list[str]]]

TOTAL_WEEKS = 9
TOTAL_TICKS = TOTAL_WEEKS * TICKS_PER_WEEK
WK7_START = 7 * TICKS_PER_WEEK
WK8_START = 8 * TICKS_PER_WEEK
WK8_END = 8 * TICKS_PER_WEEK
WK9_END = 9 * TICKS_PER_WEEK


def make_arms(ctla4_key: str) -> dict[str, Schedule]:
    """Build treatment schedules for all arms.

    ctla4_key: "CTLA4" for suppression-only, "CTLA4_ADCC" for suppression + depletion.
    """
    return {
        "untreated": [],
        "PD1_only": [
            (WK7_START, WK8_START, ["PD1"]),
        ],
        f"{ctla4_key}_only": [
            (WK7_START, WK8_START, [ctla4_key]),
        ],
        f"combo_{ctla4_key}": [
            (WK7_START, WK8_START, ["PD1", ctla4_key]),
        ],
        f"PD1_then_{ctla4_key}": [
            (WK7_START, WK8_START, ["PD1"]),
            (WK8_START, WK9_END, [ctla4_key]),
        ],
        f"{ctla4_key}_then_PD1": [
            (WK7_START, WK8_START, [ctla4_key]),
            (WK8_START, WK9_END, ["PD1"]),
        ],
        # SAR445877: anti-PD1/IL-15 fusion protein
        "PD1_IL15": [
            (WK7_START, WK8_START, ["PD1_IL15"]),
        ],
        # PD1/IL-15 + CTLA4 combo
        f"PD1_IL15_plus_{ctla4_key}": [
            (WK7_START, WK8_START, ["PD1_IL15", ctla4_key]),
        ],
    }


def run_single_sim(params_vec: np.ndarray, schedule: Schedule,
                   seed: int) -> dict:
    """Run one ABM simulation with a treatment schedule."""
    preset = make_preset(params_vec)
    tracker = MetricsTracker(distance_interval=999, interaction_interval=999,
                             grid_interval=999, capture_grids=False)
    model = LUADModel(preset=preset, interventions=[],
                      seed=seed, metrics_tracker=tracker)
    p = dict(zip(PARAM_NAMES, params_vec))
    for key in PARAM_NAMES:
        if key not in ("treg_suppression", "treg_death_rate"):
            model.params[key] = float(p[key])
    for key, val in FIXED_PARAMS.items():
        model.params[key] = val
    model.params["treg_death_rate"] = float(p["treg_death_rate"])

    # Pre-compute start/end events
    start_events = {}  # tick -> list of intervention names to apply
    end_events = {}    # tick -> list of intervention names to remove
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
        return sum(1 for a in model.agents if getattr(a, 'agent_type', None) == at)

    return {
        "n_tumor": count(AgentType.TUMOR),
        "n_cd8": count(AgentType.CD8),
        "n_cd4": count(AgentType.CD4),
        "n_treg": count(AgentType.TREG),
        "n_macrophage": count(AgentType.MACROPHAGE),
        "seed": seed,
    }


def run_all(params_vec: np.ndarray, n_seeds: int) -> dict:
    """Run all conditions and return results dict."""
    results = {}

    for ctla4_key in ["CTLA4", "CTLA4_ADCC"]:
        mode_label = "suppression" if ctla4_key == "CTLA4" else "ADCC"
        arms = make_arms(ctla4_key)

        for arm_name, schedule in arms.items():
            # Skip shared arms for ADCC mode (identical to suppression mode)
            if ctla4_key == "CTLA4_ADCC" and arm_name in ("untreated", "PD1_only", "PD1_IL15"):
                continue

            label = f"{arm_name}" if ctla4_key == "CTLA4" else f"{arm_name}"
            # Prefix with mode for CTLA4-containing arms in ADCC mode
            if ctla4_key == "CTLA4_ADCC":
                label = arm_name  # already has CTLA4_ADCC in the name

            print(f"\n=== {label} [{mode_label}] ({n_seeds} seeds) ===", flush=True)
            results[label] = []

            for s in range(n_seeds):
                seed = 1000 + s
                result = run_single_sim(params_vec, schedule, seed)
                results[label].append(result)
                print(f"  seed {s}: tumor={result['n_tumor']}, cd8={result['n_cd8']}, "
                      f"treg={result['n_treg']}", flush=True)

    return results


def make_figure(results: dict, n_seeds: int, out_dir: Path):
    """Generate grouped bar chart: 6 arms × 2 modes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Define display order and labels
    # (suppression_key, adcc_key, display_label)
    arm_pairs = [
        ("untreated",           "untreated",                    "Untreated"),
        ("PD1_only",            "PD1_only",                     "PD1 only"),
        ("CTLA4_only",          "CTLA4_ADCC_only",              "CTLA4 only"),
        ("combo_CTLA4",         "combo_CTLA4_ADCC",             "PD1+CTLA4"),
        ("PD1_then_CTLA4",      "PD1_then_CTLA4_ADCC",          "PD1→CTLA4"),
        ("CTLA4_then_PD1",      "CTLA4_ADCC_then_PD1",          "CTLA4→PD1"),
        ("PD1_IL15",            "PD1_IL15",                     "PD1/IL-15"),
        ("PD1_IL15_plus_CTLA4", "PD1_IL15_plus_CTLA4_ADCC",     "PD1/IL-15\n+CTLA4"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Sequential & Combination Dosing: ICB ± IL-15\n"
        f"1-week pulses starting at week 7, measured at week 9 "
        f"(v6 posterior mean, {n_seeds} seeds/condition)",
        fontsize=12, fontweight="bold",
    )

    x = np.arange(len(arm_pairs))
    w = 0.35

    for ax, metric, ylabel, title in [
        (axes[0], "n_tumor", "Tumor cell count", "Tumor burden at week 9"),
        (axes[1], "n_treg", "Treg count", "Treg count at week 9"),
    ]:
        means_a, stds_a = [], []  # suppression-only
        means_b, stds_b = [], []  # ADCC

        for key_a, key_b, _ in arm_pairs:
            vals_a = [r[metric] for r in results[key_a]]
            means_a.append(np.mean(vals_a))
            stds_a.append(np.std(vals_a))

            if key_b in results:
                vals_b = [r[metric] for r in results[key_b]]
            else:
                vals_b = vals_a  # shared arm (untreated, PD1-only)
            means_b.append(np.mean(vals_b))
            stds_b.append(np.std(vals_b))

        ax.bar(x - w/2, means_a, w, yerr=stds_a, label="Suppression only",
               color="#6baed6", edgecolor="black", linewidth=0.5, capsize=3)
        ax.bar(x + w/2, means_b, w, yerr=stds_b, label="Suppression + ADCC",
               color="#fd8d3c", edgecolor="black", linewidth=0.5, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, _, lbl in arm_pairs], rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)

    # Add % change annotations on tumor panel
    ax = axes[0]
    baseline = np.mean([r["n_tumor"] for r in results["untreated"]])
    for i, (key_a, key_b, _) in enumerate(arm_pairs):
        if key_a == "untreated":
            continue
        mean_a = np.mean([r["n_tumor"] for r in results[key_a]])
        delta_a = (mean_a - baseline) / baseline * 100
        ax.text(i - w/2, mean_a + np.std([r["n_tumor"] for r in results[key_a]]) + 15,
                f"{delta_a:+.0f}%", ha="center", va="bottom", fontsize=7, color="#2171b5")

        vals_b = results.get(key_b, results[key_a])
        mean_b = np.mean([r["n_tumor"] for r in vals_b])
        delta_b = (mean_b - baseline) / baseline * 100
        ax.text(i + w/2, mean_b + np.std([r["n_tumor"] for r in vals_b]) + 15,
                f"{delta_b:+.0f}%", ha="center", va="bottom", fontsize=7, color="#d94701")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig_path = out_dir / "sequential_dosing.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}", flush=True)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=str, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = np.load(args.posterior)
    params_mean = samples.mean(axis=0).astype(np.float32)
    print(f"Posterior mean ({len(params_mean)} params):")
    for name, val in zip(PARAM_NAMES, params_mean):
        print(f"  {name:30s} = {val:.4f}")

    results = run_all(params_mean, args.seeds)

    # Save raw results
    np.savez(out_dir / "sequential_dosing_results.npz",
             **{k: [r for r in v] for k, v in results.items()})

    # Summary table
    print(f"\n{'='*90}")
    print(f"Sequential Dosing Summary (week 9 endpoint)")
    print(f"{'='*90}")
    baseline = np.mean([r["n_tumor"] for r in results["untreated"]])
    print(f"{'Condition':<28} {'Tumor (mean±sd)':<18} {'CD8':<10} {'Treg':<10} "
          f"{'CD8:Treg':<10} {'ΔTumor%':<10}")
    print("-" * 90)
    for label, runs in results.items():
        tumors = [r["n_tumor"] for r in runs]
        cd8s = [r["n_cd8"] for r in runs]
        tregs = [r["n_treg"] for r in runs]
        ratios = [c / (t + 1e-9) for c, t in zip(cd8s, tregs)]
        tm, ts = np.mean(tumors), np.std(tumors)
        delta = (tm - baseline) / baseline * 100
        print(f"{label:<28} {tm:7.0f} ± {ts:5.0f}   {np.mean(cd8s):6.0f}    "
              f"{np.mean(tregs):6.0f}    {np.mean(ratios):6.2f}    {delta:+6.1f}%",
              flush=True)

    # Figure
    try:
        make_figure(results, args.seeds, out_dir)
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
