"""Command-line interface to run Gaglia-grounded LUAD Mesa simulations."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luad.model import LUADModel, load_preset
from luad.io import OutputManager
from luad.metrics import MetricsTracker
from luad import viz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaglia-grounded LUAD immune simulator")
    parser.add_argument("--preset", required=True, help="Path to preset JSON")
    parser.add_argument("--ticks", type=int, default=1000, help="Number of simulation ticks")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out", type=str, default="outputs/run", help="Output directory")
    parser.add_argument("--add", action="append", default=[], help="Intervention to apply (can repeat)")
    parser.add_argument("--no-movie", action="store_true", help="Skip GIF generation")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output")
    parser.add_argument("--no-legend", action="store_true", help="Omit color legend from movie")
    parser.add_argument("--movie-scale", type=int, default=6, help="Integer upscaling factor for movie frames")
    parser.add_argument("--distance-interval", type=int, default=20, help="Ticks between distance captures")
    parser.add_argument("--interaction-interval", type=int, default=20, help="Ticks between interaction captures")
    parser.add_argument("--grid-interval", type=int, default=5, help="Ticks between grid snapshots")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a model parameter (can repeat, e.g. --param cd8_base_kill=0.24)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset_path = Path(args.preset)
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset not found: {preset_path}")

    preset = load_preset(preset_path)
    metrics_tracker = MetricsTracker(
        distance_interval=args.distance_interval,
        interaction_interval=args.interaction_interval,
        grid_interval=args.grid_interval,
        capture_grids=not args.no_movie,
    )

    model = LUADModel(
        preset=preset,
        interventions=args.add,
        seed=args.seed,
        metrics_tracker=metrics_tracker,
    )

    # Apply parameter overrides
    for pstr in args.param:
        key, val = pstr.split("=", 1)
        model.params[key] = float(val)
        print(f"  Override: {key} = {val}")

    tick_iter = range(args.ticks)
    progress = None
    if not args.no_progress:
        tick_iter = tqdm(tick_iter, desc="Sim ticks", unit="tick")
        progress = tick_iter

    for _ in tick_iter:
        model.step()
        if progress is not None and model.step_count % 50 == 0:
            progress.set_postfix(tick=model.step_count, refresh=False)

    output_manager = OutputManager(Path(args.out))
    timeseries_path = output_manager.save_timeseries(model)
    output_manager.save_agent_counts(model)
    output_manager.save_distance_records(metrics_tracker.distance_records)
    output_manager.save_interactions(metrics_tracker.interaction_records)
    output_manager.save_grid_snapshots(metrics_tracker.grid_snapshots, metrics_tracker.grid_interval)

    timeseries_df = model.datacollector.get_model_vars_dataframe()
    summary_dir = output_manager.summary_dir
    viz.plot_trajectories(timeseries_df, summary_dir)
    viz.plot_distance_cdf(metrics_tracker.distance_records, summary_dir)
    viz.plot_interaction_heatmap(metrics_tracker.interaction_records, summary_dir)

    if not args.no_movie:
        viz.write_movie(
            metrics_tracker.grid_snapshots,
            output_manager.output_dir,
            scale=max(1, args.movie_scale),
            legend=not args.no_legend,
        )

    print(f"Simulation complete. Outputs saved to {output_manager.output_dir}")
    print(f"Timeseries: {timeseries_path}")


if __name__ == "__main__":
    main()
