#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'luad_abm'
if str(ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(ROOT))

from luad import viz  # noqa: E402

PALETTE = viz.plot_trajectory_palette()

def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate population trajectory plots with movie-aligned colors.")
    parser.add_argument("run_dirs", nargs="*", help="Output directories to update; defaults to all subdirs with timeseries.csv")
    args = parser.parse_args()

    if args.run_dirs:
        run_paths = [Path(p) for p in args.run_dirs]
    else:
        run_paths = [p for p in (ROOT / 'outputs').glob('*') if (p / 'timeseries.csv').exists()]

    for run_dir in run_paths:
        df = pd.read_csv(run_dir / 'timeseries.csv', index_col='tick')
        viz.plot_trajectories(df, run_dir, palette=PALETTE)
        print(f"Updated trajectories for {run_dir}")


if __name__ == "__main__":
    main()
