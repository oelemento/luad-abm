#!/usr/bin/env python
"""Generate patient-specific LUAD ABM configs and optionally run simulations.

The script expects two CSVs:
  1. patient_celltype_broad_density_group.csv (cell densities per mm^2)
  2. patient_celltype_broad_density_group.obs.csv (metadata including ROI_area)

Usage example:
    ../.venv/bin/python scripts/run_patient_sims.py \
        --cells data/patient_celltype_broad_density_group.csv \
        --meta data/patient_celltype_broad_density_group.obs.csv \
        --preset luad_abm/config/G4_fibrotic.json \
        --config-dir luad_abm/config/patients \
        --outputs-dir luad_abm/outputs/patients \
        --group Group4 \
        --run
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LUAD_ROOT = ROOT / "luad_abm"
RUN_SCRIPT = LUAD_ROOT / "runs" / "run.py"
GRID_WIDTH = 100
GRID_HEIGHT = 100
GRID_CAPACITY = GRID_WIDTH * GRID_HEIGHT

# Map CSV columns to agent classes used in the ABM
AGENT_COLUMN_MAP: Dict[str, Iterable[str]] = {
    "tumor": ("Epi.-like", "Epi. Prol.", "Mesen.-like"),
    "caf": ("Fib.",),
    "macrophage": ("Mac.",),
    "cd8": ("CD8 T",),
    "cd4": ("CD4 T",),
    "treg": ("T reg",),
}

# Default TLS count by group label if present
GROUP_TLS_MAP = {
    "Group1": 1,
    "Group2": 1,
    "Group3": 2,
    "Group4": 1,
}

GROUP_PRESET_MAP = {
    "Group1": LUAD_ROOT / "config" / "G1_in_situ.json",
    "Group2": LUAD_ROOT / "config" / "G2_macrophage.json",
    "Group3": LUAD_ROOT / "config" / "G3_inflammatory.json",
    "Group4": LUAD_ROOT / "config" / "G4_fibrotic.json",
}


def load_csv(paths: Tuple[Path, Path]) -> pd.DataFrame:
    cells_path, meta_path = paths
    cells = pd.read_csv(cells_path, index_col=0)
    meta = pd.read_csv(meta_path, index_col=0)
    merged = cells.join(meta, how="inner", lsuffix="", rsuffix="")
    if merged.empty:
        raise ValueError("No overlap between cell density and metadata tables.")
    return merged


_TEMPLATE_CACHE: Dict[Path, dict] = {}


def load_template(preset_path: Path) -> dict:
    preset_path = preset_path.resolve()
    if preset_path not in _TEMPLATE_CACHE:
        with open(preset_path, "r", encoding="utf-8") as fh:
            _TEMPLATE_CACHE[preset_path] = json.load(fh)
    return _TEMPLATE_CACHE[preset_path]


def density_to_count(row: pd.Series, columns: Iterable[str], area: float) -> float:
    value = 0.0
    for col in columns:
        if col in row and pd.notna(row[col]):
            value += float(row[col])
    return value * area


def compute_agent_counts(row: pd.Series, min_tumor: int = 100) -> Dict[str, int]:
    area = float(row.get("ROI_area", 1.0) or 1.0)
    counts = {}
    for agent_key, cols in AGENT_COLUMN_MAP.items():
        total_density = density_to_count(row, cols, area)
        count = max(0, int(round(total_density)))
        counts[agent_key] = count

    # Ensure at least some tumor cells
    counts["tumor"] = max(min_tumor, counts.get("tumor", 0))

    # Estimate TLS nodes
    group = row.get("Group")
    counts["tls"] = int(GROUP_TLS_MAP.get(group, 1))
    return counts


def scale_counts_if_needed(counts: Dict[str, int], min_tumor: int) -> Dict[str, int]:
    total = sum(max(0, v) for v in counts.values())
    if total <= GRID_CAPACITY:
        counts["tumor"] = max(counts.get("tumor", 0), min_tumor)
        return counts
    scale = GRID_CAPACITY / total
    scaled = {}
    residual = GRID_CAPACITY
    for key, value in counts.items():
        scaled_value = max(0, int(round(value * scale)))
        scaled[key] = scaled_value
        residual -= scaled_value
    if residual > 0:
        scaled["tumor"] = scaled.get("tumor", 0) + residual
    scaled["tumor"] = max(scaled.get("tumor", 0), min_tumor)
    total_scaled = sum(scaled.values())
    if total_scaled > GRID_CAPACITY:
        overflow = total_scaled - GRID_CAPACITY
        scaled["tumor"] = max(min_tumor, scaled["tumor"] - overflow)
    return scaled


def update_preset(base: dict, patient_id: str, counts: Dict[str, int], metadata: dict) -> dict:
    config = json.loads(json.dumps(base))  # deep copy
    config["name"] = patient_id
    config["description"] = f"Patient-derived preset for {patient_id}"
    config.setdefault("metadata", {}).update(metadata)
    config.setdefault("initial_agents", {})
    for key, value in counts.items():
        config["initial_agents"][key] = max(0, int(value))
    return config


def write_config(config: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)


def run_simulation(config_path: Path, outputs_dir: Path, ticks: int, extra_args: list[str] | None = None) -> None:
    preset_arg = str(config_path.relative_to(LUAD_ROOT)) if config_path.is_absolute() else str(config_path)
    out_arg = str(outputs_dir.relative_to(LUAD_ROOT)) if outputs_dir.is_absolute() else str(outputs_dir)
    cmd = [sys.executable, str(RUN_SCRIPT), "--preset", preset_arg, "--ticks", str(ticks), "--out", out_arg]
    if extra_args:
        cmd.extend(extra_args)
    print('Launching:', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=str(LUAD_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build patient-driven ABM configs and optionally run simulations.")
    parser.add_argument("--cells", required=True, type=Path, help="CSV with cell-type densities per ROI")
    parser.add_argument("--meta", required=True, type=Path, help="CSV with patient metadata (must include ROI_area)")
    parser.add_argument("--preset", default=LUAD_ROOT / "config" / "G4_fibrotic.json", type=Path, help="Template preset JSON")
    parser.add_argument("--config-dir", default=LUAD_ROOT / "config" / "patients", type=Path, help="Where to write patient JSON presets")
    parser.add_argument("--outputs-dir", default=LUAD_ROOT / "outputs" / "patients", type=Path, help="Where to store simulation outputs")
    parser.add_argument("--group", action="append", help="Filter patients by metadata Group (can repeat)")
    parser.add_argument("--patient", action="append", help="Run only selected patient IDs")
    parser.add_argument("--ticks", type=int, default=2000, help="Number of simulation ticks per patient")
    parser.add_argument("--min-tumor", type=int, default=100, help="Minimum tumor agent count")
    parser.add_argument("--run-args", type=str, help="Quoted string of extra arguments for run.py (e.g., '--movie-scale 8 --no-movie')")
    parser.add_argument("--run", action="store_true", help="Launch simulations after generating configs")
    args = parser.parse_args()

    df = load_csv((args.cells, args.meta))
    args.config_dir = args.config_dir.resolve()
    args.outputs_dir = args.outputs_dir.resolve()
    template = load_template(args.preset)

    def template_for_row(row: pd.Series) -> dict:
        group = row.get("Group")
        preset_path = GROUP_PRESET_MAP.get(group)
        if preset_path is None:
            return template
        return load_template(Path(preset_path))


    # Filter by patient IDs or groups if requested
    idx = df.index
    if args.group:
        df = df[df["Group"].isin(args.group)]
    if args.patient:
        df = df.loc[df.index.intersection(args.patient)]
    if df.empty:
        raise ValueError("No patients remain after filtering.")

    configs = []
    for patient_id, row in df.iterrows():
        base_template = template_for_row(row)
        counts = compute_agent_counts(row, min_tumor=args.min_tumor)
        counts = scale_counts_if_needed(counts, args.min_tumor)
        metadata = row.to_dict()
        config = update_preset(base_template, patient_id, counts, metadata)
        config_path = args.config_dir / f"patient_{patient_id}.json"
        write_config(config, config_path)
        configs.append((patient_id, config_path, counts, metadata))

    print(f"Generated {len(configs)} patient presets in {args.config_dir}")

    if args.run:
        import shlex
        extra = shlex.split(args.run_args) if args.run_args else []
        for patient_id, config_path, _, metadata in configs:
            output_dir = args.outputs_dir / patient_id
            print(f"Running simulation for {patient_id} ...")
            run_simulation(config_path, output_dir, args.ticks, extra_args=extra)


if __name__ == "__main__":
    main()
