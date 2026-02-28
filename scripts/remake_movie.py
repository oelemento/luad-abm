#!/usr/bin/env python
"""Rebuild a LUAD simulation movie from stored grid snapshots."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1] / 'luad_abm'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luad import viz


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate movie.gif from grid_snapshots.npz")
    parser.add_argument("run_dir", help="Directory containing grid_snapshots.npz")
    parser.add_argument("--movie-scale", type=int, default=6, help="Upscaling factor (default=6)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output movie (default=10)")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Include every Nth snapshot when building the movie (default=1, i.e. keep all frames)",
    )
    parser.add_argument("--no-legend", action="store_true", help="Disable legend panel in the movie")
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    snapshot_path = run_path / "grid_snapshots.npz"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Could not find {snapshot_path}")

    data = np.load(snapshot_path, allow_pickle=True)
    ticks = data["ticks"]
    grids = data["grids"]
    snapshots = list(zip(ticks, grids))
    stride = max(1, args.stride)
    if stride > 1:
        snapshots = snapshots[::stride]
    viz.write_movie(
        snapshots,
        run_path,
        fps=max(1, args.fps),
        scale=max(1, args.movie_scale),
        legend=not args.no_legend,
    )
    print(f"movie.gif regenerated in {run_path}")


if __name__ == "__main__":
    main()
