"""Side-by-side trajectory comparison: Baseline vs PD1 vs PD1+CTLA4."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

baseline = pd.read_csv("outputs/gaglia_turnover_test/timeseries.csv", index_col="tick")
pd1 = pd.read_csv("outputs/gaglia_pd1_test/timeseries.csv", index_col="tick")
combo = pd.read_csv("outputs/gaglia_pd1_ctla4_test/timeseries.csv", index_col="tick")

cell_types = [
    ("tumor_count", "Tumor", "#e86464"),
    ("cd8_count", "CD8", "#3399e6"),
    ("cd4_count", "CD4", "#4dcce6"),
    ("treg_count", "Treg", "#9933cc"),
    ("macrophage_count", "Macrophage", "#8c8c8c"),
]

datasets = [
    (baseline, "Baseline (no treatment)"),
    (pd1, "Anti-PD-1"),
    (combo, "Anti-PD-1 + Anti-CTLA-4"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, (df, title) in zip(axes, datasets):
    for col, label, color in cell_types:
        ax.plot(df.index, df[col], label=label, color=color, linewidth=2)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Tick")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 169)

axes[0].set_ylabel("Agent count")
axes[1].legend(loc="center right", fontsize=9)

fig.suptitle("Gaglia ABM: Treatment Comparison", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()

out = Path("outputs/gaglia_3way_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
