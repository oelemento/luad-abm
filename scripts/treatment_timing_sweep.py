"""Treatment timing sweep using v6 posterior mean parameters.

Runs the ABM with treatment starting at different weeks to explore
how intervention timing affects tumor control. Uses the correct
Gaglia protocol (1-week treatment pulse) with v6 posterior mean.

Usage:
    python3.11 scripts/treatment_timing_sweep.py \
        --posterior outputs/bayesian_inference_v6/posterior_samples.npy \
        --seeds 10 --out outputs/treatment_timing_sweep
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
    PARAM_NAMES, PARAM_DEFS, FIXED_PARAMS, STAT_KEYS,
    make_preset, TICKS_PER_WEEK,
)
from luad.calibration import extract_summary_stats
from luad.model import LUADModel
from luad.metrics import MetricsTracker
from luad.agents import AgentType


def run_single_sim(params_vec: np.ndarray, interventions: list[str],
                   total_ticks: int, seed: int,
                   intervention_start_tick: int | None = None,
                   intervention_duration_ticks: int | None = None) -> dict:
    """Run one ABM simulation and return final agent counts.

    If intervention_start_tick is set, treatment turns on at that tick.
    If intervention_duration_ticks is also set, treatment turns OFF after
    that many ticks (1-week pulse = 24 ticks). Otherwise treatment stays on.
    """
    start_iv = [] if intervention_start_tick is not None else interventions
    preset = make_preset(params_vec)
    tracker = MetricsTracker(distance_interval=999, interaction_interval=999,
                             grid_interval=999, capture_grids=False)
    model = LUADModel(preset=preset, interventions=start_iv,
                      seed=seed, metrics_tracker=tracker)
    p = dict(zip(PARAM_NAMES, params_vec))
    for key in PARAM_NAMES:
        if key not in ("treg_suppression", "treg_death_rate"):
            model.params[key] = float(p[key])
    for key, val in FIXED_PARAMS.items():
        model.params[key] = val
    model.params["treg_death_rate"] = float(p["treg_death_rate"])

    end_tick = None
    if intervention_start_tick is not None and intervention_duration_ticks is not None:
        end_tick = intervention_start_tick + intervention_duration_ticks

    for tick in range(total_ticks):
        if intervention_start_tick is not None and tick == intervention_start_tick:
            model.apply_interventions(interventions)
        if end_tick is not None and tick == end_tick:
            model.remove_interventions(interventions)
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


def run_sweep(params_vec: np.ndarray, total_weeks: int, treat_start_weeks: list[int | None],
              n_seeds: int) -> dict:
    """Run ABM at each treatment start time with multiple seeds.

    treat_start_weeks: list of ints (week to start treatment) or None (no treatment).
    Returns dict mapping treat_start -> list of result dicts.
    """
    total_ticks = total_weeks * TICKS_PER_WEEK
    results = {}

    for treat_start in treat_start_weeks:
        label = f"week_{treat_start}" if treat_start is not None else "no_treatment"
        print(f"\n=== {label} (total {total_weeks}wk, {n_seeds} seeds) ===", flush=True)
        results[label] = []

        for s in range(n_seeds):
            seed = 1000 + s
            if treat_start is not None:
                start_tick = treat_start * TICKS_PER_WEEK
                interventions = ["PD1", "CTLA4"]
            else:
                start_tick = None
                interventions = []

            result = run_single_sim(
                params_vec, interventions, total_ticks, seed,
                intervention_start_tick=start_tick,
                intervention_duration_ticks=TICKS_PER_WEEK,  # 1-week pulse
            )
            results[label].append(result)
            print(f"  seed {s}: tumor={result['n_tumor']}, cd8={result['n_cd8']}, "
                  f"treg={result['n_treg']}, cd8:treg={result['n_cd8']/(result['n_treg']+1e-9):.2f}",
                  flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=str, required=True,
                        help="Path to posterior_samples.npy")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds per condition")
    parser.add_argument("--total-weeks", type=int, default=8,
                        help="Total simulation duration in weeks")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load posterior mean
    samples = np.load(args.posterior)
    params_mean = samples.mean(axis=0).astype(np.float32)
    print(f"Posterior mean ({len(params_mean)} params):")
    for name, val in zip(PARAM_NAMES, params_mean):
        print(f"  {name:30s} = {val:.4f}")

    # Treatment timing sweep: no treatment, then week 1 through week (N-1)
    treat_starts = [None] + list(range(1, args.total_weeks))
    # None = no treatment, 1 = start at week 1, ..., 7 = start at week 7 (Gaglia protocol)

    results = run_sweep(params_mean, args.total_weeks, treat_starts, args.seeds)

    # Save raw results
    np.savez(out_dir / "sweep_results.npz",
             **{k: [r for r in v] for k, v in results.items()})

    # --- Summary table ---
    print(f"\n{'='*80}")
    print(f"8-week endpoint summary")
    print(f"{'='*80}")
    print(f"{'Treatment pulse':<18} {'Tumor (mean±sd)':<18} {'CD8':<12} {'Treg':<12} "
          f"{'CD8:Treg':<12} {'Tumor Δ%':<12}")
    print("-" * 80)

    baseline = np.mean([r["n_tumor"] for r in results["no_treatment"]])
    for label in ["no_treatment"] + [f"week_{w}" for w in range(1, args.total_weeks)]:
        if label not in results:
            continue
        runs = results[label]
        tumors = [r["n_tumor"] for r in runs]
        cd8s = [r["n_cd8"] for r in runs]
        tregs = [r["n_treg"] for r in runs]
        ratios = [c / (t + 1e-9) for c, t in zip(cd8s, tregs)]
        tm, ts = np.mean(tumors), np.std(tumors)
        delta = (tm - baseline) / baseline * 100
        disp = label.replace("no_treatment", "No treatment").replace("week_", "Wk ")
        is_gaglia = (label == "week_7")
        marker = " *" if is_gaglia else ""
        print(f"{disp + marker:<18} {tm:7.0f} ± {ts:5.0f}   "
              f"{np.mean(cd8s):6.0f}      {np.mean(tregs):6.0f}      "
              f"{np.mean(ratios):6.2f}      {delta:+6.1f}%", flush=True)

    # --- Figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Treatment Timing Sweep — 1-week PD1+CTLA4 pulse\n"
                     f"(v6 posterior mean, {args.seeds} seeds/condition, 8-week endpoint)",
                     fontsize=13, fontweight="bold")

        labels_raw = ["no_treatment"] + [f"week_{w}" for w in range(1, args.total_weeks)]
        labels_display = ["None"] + [f"Wk {w}" for w in range(1, args.total_weeks)]
        GAGLIA_WEEK = "week_7"

        tumor_means, tumor_stds = [], []
        cd8_means, treg_means = [], []
        ratio_means, ratio_stds = [], []
        for label in labels_raw:
            runs = results[label]
            tumors = [r["n_tumor"] for r in runs]
            cd8s = [r["n_cd8"] for r in runs]
            tregs = [r["n_treg"] for r in runs]
            ratios = [r["n_cd8"] / (r["n_treg"] + 1e-9) for r in runs]
            tumor_means.append(np.mean(tumors))
            tumor_stds.append(np.std(tumors))
            cd8_means.append(np.mean(cd8s))
            treg_means.append(np.mean(tregs))
            ratio_means.append(np.mean(ratios))
            ratio_stds.append(np.std(ratios))

        baseline_tumor = tumor_means[0]
        x = np.arange(len(labels_raw))

        colors, edge_colors, edge_widths = [], [], []
        for label in labels_raw:
            if label == "no_treatment":
                colors.append("#888888"); edge_colors.append("black"); edge_widths.append(0.5)
            elif label == GAGLIA_WEEK:
                colors.append("#2ca02c"); edge_colors.append("#1a6b1a"); edge_widths.append(2.0)
            else:
                colors.append("#6baed6"); edge_colors.append("black"); edge_widths.append(0.5)

        # Panel 1: Tumor burden
        ax = axes[0, 0]
        ax.bar(x, tumor_means, yerr=tumor_stds, color=colors, capsize=3,
               edgecolor=edge_colors, linewidth=edge_widths)
        ax.set_xticks(x); ax.set_xticklabels(labels_display, rotation=45, ha="right")
        ax.set_ylabel("Tumor cell count"); ax.set_title("Tumor burden at endpoint")
        ax.axhline(baseline_tumor, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        for i, label in enumerate(labels_raw):
            if label == GAGLIA_WEEK:
                ax.annotate("Gaglia\n8wk expt",
                            xy=(i, tumor_means[i] + tumor_stds[i] + 30),
                            ha="center", fontsize=7.5, color="#1a6b1a", fontweight="bold")

        # Panel 2: CD8 and Treg
        ax = axes[0, 1]
        w = 0.35
        ax.bar(x - w/2, cd8_means, w, label="CD8", color="dodgerblue", edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, treg_means, w, label="Treg", color="salmon", edgecolor="black", linewidth=0.5)
        for i, label in enumerate(labels_raw):
            if label == GAGLIA_WEEK:
                ax.axvspan(i - 0.45, i + 0.45, alpha=0.1, color="green")
        ax.set_xticks(x); ax.set_xticklabels(labels_display, rotation=45, ha="right")
        ax.set_ylabel("Cell count"); ax.set_title("CD8 and Treg at endpoint")
        ax.legend(fontsize=9)

        # Panel 3: CD8:Treg ratio
        ax = axes[1, 0]
        ax.bar(x, ratio_means, yerr=ratio_stds, color=colors, capsize=3,
               edgecolor=edge_colors, linewidth=edge_widths)
        ax.set_xticks(x); ax.set_xticklabels(labels_display, rotation=45, ha="right")
        ax.set_ylabel("CD8:Treg ratio"); ax.set_title("CD8:Treg ratio at endpoint")

        # Panel 4: Tumor reduction %
        ax = axes[1, 1]
        deltas = [(t - baseline_tumor) / baseline_tumor * 100 for t in tumor_means]
        delta_colors = []
        for d, label in zip(deltas, labels_raw):
            if label == "no_treatment": delta_colors.append("#888888")
            elif label == GAGLIA_WEEK: delta_colors.append("#2ca02c")
            elif d < 0: delta_colors.append("#6baed6")
            else: delta_colors.append("#e37777")
        ax.bar(x, deltas, color=delta_colors, edgecolor=edge_colors, linewidth=edge_widths)
        ax.set_xticks(x); ax.set_xticklabels(labels_display, rotation=45, ha="right")
        ax.set_ylabel("Tumor change vs no treatment (%)"); ax.set_title("Treatment efficacy by timing")
        ax.axhline(0, color="black", linewidth=0.5)
        for i, label in enumerate(labels_raw):
            if label == GAGLIA_WEEK:
                y_pos = deltas[i] - 3 if deltas[i] < 0 else deltas[i] + 1
                ax.annotate("Gaglia 8wk expt", xy=(i, y_pos),
                            ha="center", fontsize=7.5, color="#1a6b1a", fontweight="bold")

        legend_elements = [
            Patch(facecolor="#888888", edgecolor="black", label="No treatment"),
            Patch(facecolor="#2ca02c", edgecolor="#1a6b1a", linewidth=2, label="Gaglia-validated"),
            Patch(facecolor="#6baed6", edgecolor="black", label="Counterfactual prediction"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=3,
                   fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_path = out_dir / "treatment_timing_sweep.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure: {fig_path}", flush=True)
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
