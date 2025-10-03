#!/usr/bin/env python
"""Visualize initial agent compositions for simulated patients.

Creates a heatmap of initial counts (per agent class) for every patient
with a generated preset under `luad_abm/config/patients`.

Example:
    .venv/bin/python scripts/plot_patient_composition.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LUAD_ROOT = ROOT / "luad_abm"
CONFIG_DIR = LUAD_ROOT / "config" / "patients"
PLOT_DIR = LUAD_ROOT / "summary_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_PATH = PLOT_DIR / "patient_initial_composition_heatmap.png"

AGENT_ORDER = ["tumor", "caf", "macrophage", "cd8", "cd4", "treg", "tls"]


def load_patient_configs(config_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for cfg_path in sorted(config_dir.glob("patient_*.json")):
        patient_id = cfg_path.stem.replace("patient_", "")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        metadata = data.get("metadata", {})
        group = metadata.get("Group", "Unknown")
        counts = data.get("initial_agents", {})
        row = {"patient": patient_id, "Group": group}
        for agent in AGENT_ORDER:
            row[agent] = counts.get(agent, 0)
        rows.append(row)
    if not rows:
        raise SystemExit("No patient presets found. Run scripts/run_patient_sims.py first.")
    df = pd.DataFrame(rows).set_index("patient")
    return df


def main() -> None:
    df = load_patient_configs(CONFIG_DIR)

    # Normalize ordering by group and patient ID for readability
    df["Group"] = df["Group"].fillna("Unknown")
    df = df.sort_values(["Group", df.index.name])

    agent_counts = df[AGENT_ORDER]

    fig, ax = plt.subplots(figsize=(len(AGENT_ORDER) * 1.2, max(4, len(df) * 0.4)))

    im = ax.imshow(agent_counts.values, aspect="auto", cmap="viridis")

    # Tick labels
    ax.set_xticks(np.arange(len(AGENT_ORDER)))
    ax.set_xticklabels(AGENT_ORDER, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df.index)

    ax.set_title("Initial agent counts per patient")
    ax.set_xlabel("Agent type")
    ax.set_ylabel("Patient (sorted by group)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Agent count")

    # Annotate group on right
    groups = df["Group"].tolist()
    for idx, group in enumerate(groups):
        ax.text(len(AGENT_ORDER) + 0.3, idx, group, va="center", fontsize=9)
    ax.set_xlim(-0.5, len(AGENT_ORDER) + 1)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved {PLOT_PATH}")


if __name__ == "__main__":
    main()
