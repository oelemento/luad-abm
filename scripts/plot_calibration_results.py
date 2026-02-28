"""Generate multi-panel calibration figure for Sanofi proposal.

Usage:
    python scripts/plot_calibration_results.py \
        --obs data/gaglia_2023/gaglia_summary_stats.csv \
        --results data/calibration_results_baseline.csv \
        --out luad_abm/summary_plots/calibration_figure.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARAM_NAMES = {
    "cd8_base_kill": "CD8 kill rate",
    "cd8_exhaustion_rate": "Exhaustion rate",
    "pd_l1_penalty": "PD-L1 penalty",
    "macrophage_suppr_base": "MAC suppression",
    "suppressive_background": "Bg suppression",
    "tumor_proliferation_rate": "Tumor prolif.",
    "snapshot_tick": "Snapshot tick",
}

GROUP_LABELS = {1: "Ctrl", 2: "anti-PD1", 3: "anti-CTLA4", 4: "Combo"}
GROUP_COLORS = {1: "#4477AA", 2: "#EE6677", 3: "#228833", 4: "#CCBB44"}

# Shared cell types between ABM and Gaglia
SHARED_FRACS = {
    "frac_t_cytotox": "CD8+ T",
    "frac_t_helper": "CD4+ T",
    "frac_t_reg": "Treg",
    "frac_macrophage": "Macrophage",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-frac", type=float, default=0.1,
                        help="Fraction of best fits to show")
    args = parser.parse_args()

    obs = pd.read_csv(args.obs)
    res = pd.read_csv(args.results)

    # Use mean_distance if available (multi-group), else distance
    dist_col = "mean_distance" if "mean_distance" in res.columns else "distance"
    has_groups = "group" in res.columns

    # Top parameter sets by aggregate distance
    if has_groups:
        agg = res.groupby("sample_idx")[dist_col].first().reset_index()
        top_n = max(5, int(len(agg) * args.top_frac))
        top_idxs = agg.nsmallest(top_n, dist_col)["sample_idx"].values
        top = res[res["sample_idx"].isin(top_idxs)]
    else:
        top_n = max(5, int(len(res) * args.top_frac))
        top = res.nsmallest(top_n, dist_col)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ------------------------------------------------------------------
    # Panel A: Observed vs simulated fractions PER GROUP (top params)
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    frac_keys = list(SHARED_FRACS.keys())
    frac_labels = list(SHARED_FRACS.values())
    n_frac = len(frac_keys)
    n_groups = len(GROUP_LABELS)
    x = np.arange(n_frac)
    bar_w = 0.8 / (n_groups * 2)  # obs + sim per group

    for gi, (g, glabel) in enumerate(GROUP_LABELS.items()):
        color = GROUP_COLORS[g]
        # Observed
        obs_g = obs[obs["mouse_group"] == g]
        obs_means = [obs_g[k].mean() if k in obs_g.columns else 0 for k in frac_keys]
        obs_sems = [obs_g[k].std() / np.sqrt(len(obs_g)) if k in obs_g.columns else 0 for k in frac_keys]
        offset = gi * 2 * bar_w
        ax.bar(x + offset, obs_means, bar_w, yerr=obs_sems, capsize=2,
               color=color, alpha=0.9, edgecolor="white",
               label=f"{glabel} obs" if gi == 0 else f"{glabel} obs")

        # Simulated (top parameter sets for this group)
        if has_groups:
            sim_g = top[top["group"] == g]
        else:
            sim_g = top
        sim_means = [sim_g[f"sim_{k}"].mean() if f"sim_{k}" in sim_g.columns else 0 for k in frac_keys]
        sim_sems = [sim_g[f"sim_{k}"].std() / np.sqrt(len(sim_g)) if f"sim_{k}" in sim_g.columns else 0 for k in frac_keys]
        ax.bar(x + offset + bar_w, sim_means, bar_w, yerr=sim_sems, capsize=2,
               color=color, alpha=0.4, edgecolor=color, hatch="//",
               label=f"{glabel} sim" if gi == 0 else f"{glabel} sim")

    ax.set_xticks(x + bar_w * (n_groups - 0.5))
    ax.set_xticklabels(frac_labels, fontsize=9)
    ax.set_ylabel("Fraction (shared-type)")
    ax.set_title("A. Observed vs simulated composition (all groups)")
    # Compact legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:4], [GROUP_LABELS[g] for g in GROUP_LABELS],
              fontsize=7, title="solid=obs, hatch=sim", title_fontsize=7)

    # ------------------------------------------------------------------
    # Panel B: Inferred parameter distributions (violin plots)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    param_keys = list(PARAM_NAMES.keys())
    positions = np.arange(len(param_keys))
    # Use one row per sample_idx for parameter violins (deduplicate groups/seeds)
    if has_groups:
        param_df = top.drop_duplicates("sample_idx")
    else:
        param_df = top
    for i, pk in enumerate(param_keys):
        if pk not in param_df.columns:
            continue
        vals = param_df[pk].dropna().values
        if len(vals) < 2:
            continue
        parts = ax.violinplot([vals], positions=[i], showmeans=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4477AA")
            pc.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([PARAM_NAMES[k] for k in param_keys],
                       rotation=45, ha="right", fontsize=8)
    ax.set_title("B. Inferred parameters (top 10%)")

    # ------------------------------------------------------------------
    # Panel C: Per-group distance comparison
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    if has_groups:
        for gi, (g, glabel) in enumerate(GROUP_LABELS.items()):
            group_dists = res[res["group"] == g]["distance"].values
            top_group_dists = top[top["group"] == g]["distance"].values
            parts = ax.violinplot([group_dists], positions=[gi], showmeans=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(GROUP_COLORS[g])
                pc.set_alpha(0.5)
            ax.scatter([gi] * len(top_group_dists), top_group_dists,
                       c=GROUP_COLORS[g], s=10, alpha=0.7, zorder=3)
        ax.set_xticks(range(len(GROUP_LABELS)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_LABELS], fontsize=10)
        ax.set_ylabel("Distance score")
        ax.set_title("C. Per-group calibration quality")
    else:
        # Fallback: single histogram
        ax.hist(res["distance"].values, bins=30, color="#4477AA", alpha=0.7, edgecolor="white")
        ax.set_xlabel("Distance score")
        ax.set_ylabel("Count")
        ax.set_title("C. Distance score distribution")

    # ------------------------------------------------------------------
    # Panel D: Aggregate distance distribution + top threshold
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    if has_groups:
        agg_dists = res.groupby("sample_idx")[dist_col].first().values
    else:
        agg_dists = res[dist_col].values
    ax.hist(agg_dists, bins=30, color="#4477AA", alpha=0.7, edgecolor="white")
    threshold = np.percentile(agg_dists, args.top_frac * 100)
    ax.axvline(threshold, color="#EE6677", linestyle="--",
               label=f"Top {args.top_frac*100:.0f}% threshold")
    ax.set_xlabel("Mean distance score (across groups)")
    ax.set_ylabel("Count")
    ax.set_title("D. Parameter search scores")
    ax.legend(fontsize=8)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
