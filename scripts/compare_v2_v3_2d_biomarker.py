"""Compare v2 vs v3 ABM predictions in 2D biomarker space against Sorin clinical data.

Tests whether SBI v7 (with cd8_kill_prolif_prob) fixes the discrepancy in the
high-ratio/low-CD8 quadrant.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GAGLIA_ROOT = ROOT / "gaglia_abm"
if str(GAGLIA_ROOT) not in sys.path:
    sys.path.insert(0, str(GAGLIA_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.human_luad_sweep import load_patient_compositions, GRID_TOTAL

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_sweep_results(npz_path: Path) -> dict:
    """Load sweep results from npz file."""
    data = np.load(npz_path, allow_pickle=True)
    results = {}
    for key in data.files:
        results[key] = data[key].tolist()
    return results


def classify_response(results: dict, config: str, arm: str = "PD1_CTLA4",
                      threshold: float = -5.0) -> tuple[float, bool]:
    """Return (delta_tumor_pct, is_responder) for a config."""
    ut_key = f"{config}__untreated"
    tx_key = f"{config}__{arm}"
    if ut_key not in results or tx_key not in results:
        return 0.0, False
    baseline = np.mean([r["n_tumor"] for r in results[ut_key]])
    treated = np.mean([r["n_tumor"] for r in results[tx_key]])
    if baseline == 0:
        return 0.0, False  # can't evaluate — tumor already cleared
    delta = (treated - baseline) / baseline * 100
    return delta, delta < threshold


def main():
    data_dir = ROOT / "data/gaglia_2023/Dataset05_Human_Lung_Adenocarcinoma"
    patients = load_patient_compositions(data_dir)

    v2_path = ROOT / "outputs/human_luad_sweep_v2/human_luad_results.npz"
    v3_path = ROOT / "outputs/human_luad_sweep_v3/human_luad_results.npz"

    results_v2 = load_sweep_results(v2_path)
    results_v3 = load_sweep_results(v3_path)

    # Patient features
    names = [pt["name"] for pt in patients]
    cd8_fracs = [pt["cd8_frac"] for pt in patients]
    treg_fracs = [pt["treg_frac"] for pt in patients]
    ratios = [c / (t + 1e-9) for c, t in zip(cd8_fracs, treg_fracs)]

    # Medians for quadrant split
    med_ratio = np.median(ratios)
    med_cd8 = np.median(cd8_fracs)

    print(f"Median CD8:Treg ratio = {med_ratio:.2f}")
    print(f"Median CD8 fraction = {med_cd8:.4f}")
    print()

    # Classify each patient
    for version, results, label in [
        ("v2", results_v2, "v2 (17-param, no kill-prolif)"),
        ("v3", results_v3, "v3 (18-param, kill-prolif=0.085)"),
    ]:
        print(f"=== {label} ===")
        quadrants = {
            "High ratio + High CD8": {"respond": 0, "total": 0, "patients": []},
            "High ratio + Low CD8": {"respond": 0, "total": 0, "patients": []},
            "Low ratio + High CD8": {"respond": 0, "total": 0, "patients": []},
            "Low ratio + Low CD8": {"respond": 0, "total": 0, "patients": []},
        }

        for i, name in enumerate(names):
            delta, resp = classify_response(results, name)
            if delta == 0.0 and name in ("CASE14", "CASE19", "CASE21"):
                continue  # tumor cleared untreated — can't classify

            high_ratio = ratios[i] >= med_ratio
            high_cd8 = cd8_fracs[i] >= med_cd8

            if high_ratio and high_cd8:
                q = "High ratio + High CD8"
            elif high_ratio and not high_cd8:
                q = "High ratio + Low CD8"
            elif not high_ratio and high_cd8:
                q = "Low ratio + High CD8"
            else:
                q = "Low ratio + Low CD8"

            quadrants[q]["total"] += 1
            if resp:
                quadrants[q]["respond"] += 1
            quadrants[q]["patients"].append(f"{name}({delta:+.1f}%)")

        for q, data in quadrants.items():
            n = data["total"]
            r = data["respond"]
            pct = f"{r/n*100:.0f}%" if n > 0 else "N/A"
            print(f"  {q}: {r}/{n} ({pct})  — {', '.join(data['patients'])}")
        print()

    # Sorin clinical data for comparison
    print("=== Sorin et al. 2025 (stroma, clinical) ===")
    print("  High ratio + High CD8: 15/21 (71%)")
    print("  High ratio + Low CD8: 7/8 (88%)")
    print("  Low ratio + High CD8: 2/8 (25%)")
    print("  Low ratio + Low CD8: 12/21 (57%)")
    print()

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("2D Biomarker: ABM v2 vs v3 vs Sorin Clinical Data\n"
                 "Quadrants split at median CD8:Treg ratio and CD8 fraction",
                 fontsize=12, fontweight="bold")

    for ax_idx, (version, results, title) in enumerate([
        ("v2", results_v2, "ABM v2 (no kill-prolif)"),
        ("v3", results_v3, "ABM v3 (kill-prolif=0.085)"),
    ]):
        ax = axes[ax_idx]
        for i, name in enumerate(names):
            delta, resp = classify_response(results, name)
            if delta == 0.0 and name in ("CASE14", "CASE19", "CASE21"):
                ax.scatter(ratios[i], cd8_fracs[i], c="gray", edgecolor="black",
                           s=60, marker="x", alpha=0.5)
                continue
            color = "#2ca02c" if resp else "#d62728"
            ax.scatter(ratios[i], cd8_fracs[i], c=color, edgecolor="black", s=80)
            ax.annotate(f"{name}\n{delta:+.1f}%", (ratios[i], cd8_fracs[i]),
                        fontsize=6, ha="center", va="bottom", xytext=(0, 5),
                        textcoords="offset points")

        ax.axvline(med_ratio, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(med_cd8, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("CD8:Treg ratio")
        ax.set_ylabel("CD8 fraction")
        ax.set_title(title)

        # Add quadrant labels
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

    # Panel 3: Sorin clinical (schematic)
    ax = axes[2]
    sorin_quadrants = {
        "High ratio\nHigh CD8": (0.75, 0.75, "15/21\n71%"),
        "High ratio\nLow CD8": (0.75, 0.25, "7/8\n88%"),
        "Low ratio\nHigh CD8": (0.25, 0.75, "2/8\n25%"),
        "Low ratio\nLow CD8": (0.25, 0.25, "12/21\n57%"),
    }
    for label, (x, y, txt) in sorin_quadrants.items():
        # Color by response rate
        rate = int(txt.split("\n")[1].replace("%", "")) / 100
        color = plt.cm.RdYlGn(rate)
        ax.add_patch(plt.Rectangle((x - 0.2, y - 0.2), 0.4, 0.4,
                                    facecolor=color, alpha=0.3, edgecolor="black"))
        ax.text(x, y + 0.05, txt, ha="center", va="center", fontsize=14, fontweight="bold")
        ax.text(x, y - 0.12, label, ha="center", va="center", fontsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("CD8:Treg ratio →")
    ax.set_ylabel("CD8 fraction →")
    ax.set_title("Sorin et al. 2025 (clinical)")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = ROOT / "outputs/human_luad_sweep_v3/v2_v3_sorin_2d_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
