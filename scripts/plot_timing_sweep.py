"""Plot treatment timing sweep — 8-week endpoint, 1-week PD1+CTLA4 pulse."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = Path("outputs/treatment_timing_sweep")
data = np.load(out_dir / "sweep_results.npz", allow_pickle=True)

labels_raw = ["no_treatment"] + [f"week_{w}" for w in range(1, 8)]
# Make x-axis labels unambiguous: these are WHEN treatment was given
labels_display = [
    "Untreated",
    "Treated\nat wk 1", "Treated\nat wk 2", "Treated\nat wk 3",
    "Treated\nat wk 4", "Treated\nat wk 5", "Treated\nat wk 6",
    "Treated\nat wk 7",
]

GAGLIA_WEEK = None  # no special highlighting

tumor_means, tumor_stds = [], []
cd8_means, treg_means = [], []
ratio_means, ratio_stds = [], []

for label in labels_raw:
    runs = list(data[label])
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

# Colors: gray=untreated, blue=treated
colors, edge_colors, edge_widths = [], [], []
for label in labels_raw:
    if label == "no_treatment":
        colors.append("#888888"); edge_colors.append("black"); edge_widths.append(0.5)
    else:
        colors.append("#6baed6"); edge_colors.append("black"); edge_widths.append(0.5)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Treatment Timing Sweep\n"
    "Protocol: 1-week PD1+CTLA4 pulse administered at different weeks, "
    "tumor measured at 8 weeks\n"
    "(v6 posterior mean, 20 seeds/condition)",
    fontsize=12, fontweight="bold",
)

# --- Panel A: Tumor burden ---
ax = axes[0, 0]
ax.bar(x, tumor_means, yerr=tumor_stds, color=colors, capsize=3,
       edgecolor=edge_colors, linewidth=edge_widths)
ax.set_xticks(x)
ax.set_xticklabels(labels_display, fontsize=8)
ax.set_ylabel("Tumor cell count at week 8")
ax.set_title("A. Tumor burden at 8-week endpoint")
ax.axhline(baseline_tumor, color="red", linestyle="--", alpha=0.4, linewidth=0.8,
           label="Untreated baseline")
# % change labels
for i, (tm, ts) in enumerate(zip(tumor_means, tumor_stds)):
    if labels_raw[i] != "no_treatment":
        delta = (tm - baseline_tumor) / baseline_tumor * 100
        ax.text(i, tm + ts + 15, f"{delta:+.1f}%",
                ha="center", va="bottom", fontsize=7.5, color="black")

# --- Panel B: CD8 and Treg counts ---
ax = axes[0, 1]
w = 0.35
ax.bar(x - w/2, cd8_means, w, label="CD8+ T cells", color="dodgerblue",
       edgecolor="black", linewidth=0.5)
ax.bar(x + w/2, treg_means, w, label="Tregs", color="salmon",
       edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels_display, fontsize=8)
ax.set_ylabel("Cell count at week 8")
ax.set_title("B. Immune cell counts at endpoint")
ax.legend(fontsize=9)

# --- Panel C: CD8:Treg ratio ---
ax = axes[1, 0]
ax.bar(x, ratio_means, yerr=ratio_stds, color=colors, capsize=3,
       edgecolor=edge_colors, linewidth=edge_widths)
ax.set_xticks(x)
ax.set_xticklabels(labels_display, fontsize=8)
ax.set_ylabel("CD8:Treg ratio at week 8")
ax.set_title("C. CD8:Treg ratio at endpoint")

# --- Panel D: Tumor reduction % ---
ax = axes[1, 1]
deltas = [(t - baseline_tumor) / baseline_tumor * 100 for t in tumor_means]
delta_colors = []
for d, label in zip(deltas, labels_raw):
    if label == "no_treatment":
        delta_colors.append("#888888")
    elif d < 0:
        delta_colors.append("#6baed6")
    else:
        delta_colors.append("#e37777")
ax.bar(x, deltas, color=delta_colors, edgecolor=edge_colors, linewidth=edge_widths)
ax.set_xticks(x)
ax.set_xticklabels(labels_display, fontsize=8)
ax.set_ylabel("Tumor change vs untreated (%)")
ax.set_title("D. Treatment efficacy by timing")
ax.axhline(0, color="black", linewidth=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
fig_path = out_dir / "treatment_timing_sweep.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path}")
plt.close()
