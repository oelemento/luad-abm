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
}

# Mapping from observed fraction column names to simulated fraction column names.
# Observed uses frac_tam; ABM uses sim_frac_macrophage.
# Observed has frac_b but ABM has no B cells, so we skip it.
OBS_TO_SIM_FRAC = {
    "frac_t_cytotox": "sim_frac_t_cytotox",
    "frac_t_helper": "sim_frac_t_helper",
    "frac_t_reg": "sim_frac_t_reg",
    "frac_tam": "sim_frac_macrophage",
    # frac_b is intentionally omitted (ABM has no B cells)
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

    top_n = max(5, int(len(res) * args.top_frac))
    top = res.nsmallest(top_n, "distance")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ------------------------------------------------------------------
    # Panel A: Observed infiltration profiles by treatment group
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    groups = sorted(obs["mouse_group"].unique())
    group_labels = {1: "Ctrl", 2: "anti-PD1", 3: "anti-CTLA4", 4: "Combo"}
    colors = {1: "#4477AA", 2: "#EE6677", 3: "#228833", 4: "#CCBB44"}
    cell_types = ["t_cytotox", "t_helper", "t_reg"]
    regions = ["inside", "cuff", "periphery"]
    x_pos = np.arange(len(cell_types) * len(regions))
    bar_width = 0.8 / len(groups)
    for gi, g in enumerate(groups):
        sub = obs[obs["mouse_group"] == g]
        vals = []
        for ct in cell_types:
            for reg in regions:
                col = f"infilt_{ct}_{reg}"
                vals.append(sub[col].mean() if col in sub.columns else 0)
        ax.bar(x_pos + gi * bar_width, vals, bar_width,
               label=group_labels.get(g, f"G{g}"),
               color=colors.get(g, "gray"), alpha=0.8)
    labels = [f"{ct.split('_')[-1]}\n{reg[:3]}" for ct in cell_types for reg in regions]
    ax.set_xticks(x_pos + bar_width * (len(groups) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Fraction of cell type")
    ax.set_title("A. Observed infiltration (Gaglia)")
    ax.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Panel B: Inferred parameter distributions (violin plots)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    param_keys = list(PARAM_NAMES.keys())
    positions = np.arange(len(param_keys))
    for i, pk in enumerate(param_keys):
        if pk not in top.columns:
            continue
        vals = top[pk].dropna().values
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
    # Panel C: Observed vs simulated cell fractions (control group)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    ctrl_obs = obs[obs["mouse_group"] == 1]

    # Use the explicit mapping to handle frac_tam -> sim_frac_macrophage
    valid_pairs = []
    for obs_col, sim_col in OBS_TO_SIM_FRAC.items():
        if obs_col in ctrl_obs.columns and sim_col in top.columns:
            valid_pairs.append((obs_col, sim_col))

    if valid_pairs:
        obs_cols, sim_cols = zip(*valid_pairs)
        obs_means = [ctrl_obs[c].mean() for c in obs_cols]
        sim_means = [top[c].mean() for c in sim_cols]
        # Error bars: std across mice (obs) or across top parameter sets (sim)
        obs_sems = [ctrl_obs[c].std() / np.sqrt(len(ctrl_obs)) for c in obs_cols]
        sim_sems = [top[c].std() / np.sqrt(len(top)) for c in sim_cols]
        x = np.arange(len(obs_cols))
        # Display names: clean up the observed column names
        display_names = {
            "frac_t_cytotox": "CD8+ T",
            "frac_t_helper": "CD4+ T",
            "frac_t_reg": "Treg",
            "frac_tam": "Macrophage",
        }
        ax.bar(x - 0.2, obs_means, 0.35, yerr=obs_sems, capsize=3,
               label="Observed (Gaglia)", color="#4477AA")
        ax.bar(x + 0.2, sim_means, 0.35, yerr=sim_sems, capsize=3,
               label="Simulated (ABM)", color="#EE6677")
        ax.set_xticks(x)
        ax.set_xticklabels([display_names.get(c, c.replace("frac_", "").replace("_", " "))
                            for c in obs_cols],
                           rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=8)
    ax.set_ylabel("Fraction")
    ax.set_title("C. Observed vs simulated composition")

    # ------------------------------------------------------------------
    # Panel D: Distance score distribution
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    ax.hist(res["distance"].values, bins=30, color="#4477AA",
            alpha=0.7, edgecolor="white")
    ax.axvline(top["distance"].max(), color="#EE6677", linestyle="--",
               label=f"Top {args.top_frac*100:.0f}% threshold")
    ax.set_xlabel("Distance score")
    ax.set_ylabel("Count")
    ax.set_title("D. Parameter search scores")
    ax.legend(fontsize=8)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
