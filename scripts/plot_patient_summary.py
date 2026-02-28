#!/usr/bin/env python
"""Summarize patient simulations by plotting tumor trajectories.

Reads patient presets and outputs under `luad_abm/config/patients` and
`luad_abm/outputs/patients` and produces a PNG with tumor counts over time.

Example:
    .venv/bin/python scripts/plot_patient_summary.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LUAD_ROOT = ROOT / "luad_abm"
CONFIG_DIR = LUAD_ROOT / "config" / "patients"
DEFAULT_OUTPUT_DIR = LUAD_ROOT / "outputs" / "patients"
PLOT_DIR = LUAD_ROOT / "summary_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_PATH = PLOT_DIR / "patient_tumor_trajectories.png"

GROUP_COLORS = {
    "Group1": "#8dd3c7",
    "Group2": "#ffd92f",
    "Group3": "#80b1d3",
    "Group4": "#fb8072",
    "Unknown": "#cccccc",
}


def load_patient_groups(config_dir: Path) -> dict[str, str]:
    groups = {}
    for cfg_path in sorted(config_dir.glob("patient_*.json")):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        metadata = data.get("metadata", {})
        patient_id = cfg_path.stem.replace("patient_", "")
        groups[patient_id] = metadata.get("Group", "Unknown")
    return groups


def load_timeseries(output_dir: Path, patient_id: str) -> pd.DataFrame | None:
    ts_path = output_dir / patient_id / "timeseries.csv"
    if not ts_path.exists():
        return None
    df = pd.read_csv(ts_path)
    df["patient"] = patient_id
    return df[["tick", "tumor_count", "patient"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize patient tumor trajectories")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing patient simulation outputs",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="patient_tumor_trajectories.png",
        help="Filename for the resulting plot",
    )
    args = parser.parse_args()

    output_dir = args.outputs
    if not output_dir.exists():
        raise SystemExit(f"Output directory not found: {output_dir}")

    plot_path = PLOT_DIR / args.plot_name

    patient_groups = load_patient_groups(CONFIG_DIR)
    if not patient_groups:
        raise SystemExit("No patient configs found. Run scripts/run_patient_sims.py first.")

    records = []
    for patient_id in sorted(patient_groups.keys()):
        df = load_timeseries(output_dir, patient_id)
        if df is None:
            continue
        df["Group"] = patient_groups[patient_id]
        records.append(df)

    if not records:
        raise SystemExit("No patient timeseries found under luad_abm/outputs/patients.")

    traj = pd.concat(records, ignore_index=True)
    traj["Group"] = traj["Group"].fillna("Unknown")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Plot each patient trajectory
    for patient_id, df in traj.groupby("patient"):
        group = df["Group"].iloc[0]
        color = GROUP_COLORS.get(group, GROUP_COLORS["Unknown"])
        ax.plot(df["tick"], df["tumor_count"], color=color, linewidth=2.0, alpha=1.0)

    ax.set_title("Tumor trajectories per patient")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Tumor agent count")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Group legend
    group_handles = []
    group_labels = []
    for group, color in GROUP_COLORS.items():
        if group in traj["Group"].values:
            group_handles.append(plt.Line2D([0], [0], color=color, linewidth=2.5))
            group_labels.append(group)
    group_legend = ax.legend(group_handles, group_labels, title="Group", loc="upper left")
    ax.add_artist(group_legend)

    # Patient legend on right
    patient_handles = []
    patient_labels = []
    for patient_id in sorted(traj["patient"].unique()):
        group = patient_groups.get(patient_id, "Unknown")
        color = GROUP_COLORS.get(group, GROUP_COLORS["Unknown"])
        patient_handles.append(plt.Line2D([0], [0], color=color, linewidth=2.0))
        patient_labels.append(f"{patient_id} ({group})")
    ax.legend(
        patient_handles,
        patient_labels,
        title="Patient (Group)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
