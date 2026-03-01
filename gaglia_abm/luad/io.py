"""IO helpers for saving LUAD simulation outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import DistanceRecord, InteractionRecord


class OutputManager:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.frames_dir = self.output_dir / "frames"
        self.summary_dir = self.output_dir / "summary_plots"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

    def save_timeseries(self, model) -> Path:
        df = model.datacollector.get_model_vars_dataframe()
        path = self.output_dir / "timeseries.csv"
        df.to_csv(path, index_label="tick")
        return path

    def save_agent_counts(self, model) -> Path:
        df = model.datacollector.get_model_vars_dataframe()
        counts = df[[col for col in df.columns if col.endswith("_count")]]
        path = self.output_dir / "agent_counts.csv"
        counts.to_csv(path, index_label="tick")
        return path

    def save_distance_records(self, records: Iterable[DistanceRecord]) -> Path:
        if not records:
            return self.output_dir / "cd8_tumor_distances.csv"
        rows = []
        for record in records:
            rows.append({
                "tick": record.tick,
                "mean_distance": float(record.distances.mean()),
                "median_distance": float(np.median(record.distances)),
            })
        df = pd.DataFrame(rows)
        path = self.output_dir / "cd8_tumor_distances.csv"
        df.to_csv(path, index=False)
        hist_path = self.output_dir / "cd8_tumor_distance_hist.npz"
        np.savez(
            hist_path,
            ticks=np.array([r.tick for r in records]),
            bins=np.stack([r.hist_bins for r in records]),
            cdfs=np.stack([r.hist_cdf for r in records]),
        )
        return path

    def save_interactions(self, records: Iterable[InteractionRecord]) -> Path:
        if not records:
            return self.output_dir / "interactions.csv"
        rows = []
        for record in records:
            for (a, b), value in record.observed.items():
                expected = record.expected.get((a, b), np.nan)
                enrichment = value / (expected + 1e-6)
                rows.append({
                    "tick": record.tick,
                    "pair": f"{a}-{b}",
                    "observed": value,
                    "expected": expected,
                    "enrichment": enrichment,
                })
        df = pd.DataFrame(rows)
        path = self.output_dir / "interactions.csv"
        df.to_csv(path, index=False)
        return path

    def save_grid_snapshots(self, snapshots, interval: int) -> Path:
        if not snapshots:
            return self.output_dir / "grid_snapshots.npz"
        ticks = np.array([tick for tick, _ in snapshots], dtype=np.int32)
        grids = np.stack([grid for _, grid in snapshots])
        path = self.output_dir / "grid_snapshots.npz"
        np.savez(path, ticks=ticks, grids=grids, interval=interval)
        return path
