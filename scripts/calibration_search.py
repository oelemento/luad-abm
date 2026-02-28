"""Latin Hypercube parameter search for ABM calibration against Gaglia data.

Usage:
    python scripts/calibration_search.py \
        --obs data/gaglia_2023/gaglia_summary_stats.csv \
        --preset luad_abm/config/G3_inflammatory.json \
        --n-samples 200 \
        --ticks 500 \
        --seeds 2 \
        --out data/calibration_results.csv \
        --workers 4
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pyDOE2 import lhs

ROOT = Path(__file__).resolve().parents[1] / "luad_abm"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luad.model import LUADModel, load_preset
from luad.calibration import extract_summary_stats, compute_distance
from luad.metrics import MetricsTracker

# Parameter search space
PARAM_SPACE = {
    "cd8_base_kill":          (0.05, 0.30),
    "cd8_exhaustion_rate":    (0.02, 0.12),
    "pd_l1_penalty":          (0.30, 0.90),
    "macrophage_suppr_base":  (0.10, 0.50),
    "suppressive_background": (0.02, 0.15),
    "tumor_proliferation_rate": (0.01, 0.06),
}

# Keys to use for distance computation
CALIBRATION_KEYS = [
    "frac_t_cytotox", "frac_t_helper", "frac_t_reg", "frac_macrophage",
    "infilt_t_cytotox_inside", "infilt_t_cytotox_cuff", "infilt_t_cytotox_periphery",
    "infilt_t_helper_inside", "infilt_t_helper_cuff", "infilt_t_helper_periphery",
    "infilt_t_reg_inside", "infilt_t_reg_cuff", "infilt_t_reg_periphery",
    "cd8_mean_dist_to_tumor", "cd8_frac_within_5",
    "final_tumor_count",
]


def generate_lhs_samples(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Generate Latin Hypercube samples scaled to parameter ranges."""
    names = list(PARAM_SPACE.keys())
    ranges = list(PARAM_SPACE.values())
    rng = np.random.default_rng(seed)
    raw = lhs(len(names), samples=n_samples, random_state=rng.integers(0, 2**31))
    scaled = np.zeros_like(raw)
    for i, (lo, hi) in enumerate(ranges):
        scaled[:, i] = lo + raw[:, i] * (hi - lo)
    return pd.DataFrame(scaled, columns=names)


def _run_one(preset_path: str, param_overrides: dict, ticks: int, seed: int,
             sample_idx: int, obs_mean: dict, obs_var: dict,
             calibration_keys: list) -> dict:
    """Run one ABM simulation and return summary stats with distance.

    This is a module-level function (not a closure) so it can be pickled
    by ProcessPoolExecutor.
    """
    preset = load_preset(Path(preset_path))
    tracker = MetricsTracker(capture_grids=False, distance_interval=ticks + 1,
                             interaction_interval=ticks + 1)
    model = LUADModel(preset=preset, interventions=[],
                      seed=seed, metrics_tracker=tracker)
    # Override params
    for k, v in param_overrides.items():
        model.params[k] = v
    for _ in range(ticks):
        model.step()

    stats = extract_summary_stats(model)
    dist = compute_distance(obs_mean, stats, obs_var, calibration_keys)
    return {
        **param_overrides,
        "seed": seed,
        "sample_idx": sample_idx,
        "distance": dist,
        **{f"sim_{k}": v for k, v in stats.items()},
    }


def compute_obs_targets(obs_df: pd.DataFrame, group: int) -> tuple[dict, dict]:
    """Compute mean and variance of observed stats for a mouse group.

    Maps observed column names to ABM summary stat names where they differ
    (e.g. frac_tam -> frac_macrophage).
    """
    # Column name mapping: observed CSV name -> ABM calibration key
    col_rename = {
        "frac_tam": "frac_macrophage",
    }

    sub = obs_df[obs_df["mouse_group"] == group]
    means = {}
    variances = {}
    for col in sub.columns:
        if col in ("mouse_num", "mouse_group", "n_cells"):
            continue
        vals = sub[col].dropna().values
        if len(vals) > 0:
            key = col_rename.get(col, col)
            means[key] = float(vals.mean())
            variances[key] = float(vals.var()) if len(vals) > 1 else float(vals.mean() ** 2 * 0.1 + 1e-6)
    return means, variances


def main():
    parser = argparse.ArgumentParser(
        description="Latin Hypercube parameter search for ABM calibration"
    )
    parser.add_argument("--obs", required=True, help="Gaglia summary stats CSV")
    parser.add_argument("--preset", required=True, help="Base preset JSON")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=2, help="Replicates per param set")
    parser.add_argument("--out", required=True, help="Output CSV")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ctrl-group", type=int, default=1, help="Mouse group for control")
    args = parser.parse_args()

    obs_df = pd.read_csv(args.obs)
    obs_mean, obs_var = compute_obs_targets(obs_df, args.ctrl_group)

    samples = generate_lhs_samples(args.n_samples)
    total = args.n_samples * args.seeds
    print(f"Running {args.n_samples} parameter sets x {args.seeds} seeds = {total} simulations")

    # Resolve preset path to absolute so workers can find it
    preset_path = str(Path(args.preset).resolve())

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for idx, row in samples.iterrows():
            overrides = row.to_dict()
            for s in range(args.seeds):
                seed = 1000 * idx + s
                fut = pool.submit(
                    _run_one,
                    preset_path,
                    overrides,
                    args.ticks,
                    seed,
                    idx,
                    obs_mean,
                    obs_var,
                    CALIBRATION_KEYS,
                )
                futures.append(fut)

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                if completed % 20 == 0 or completed == total:
                    print(f"  {completed}/{total} done")
            except Exception as e:
                print(f"  Error: {e}")
                completed += 1

    df = pd.DataFrame(results)
    df.sort_values("distance", inplace=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nResults saved to {args.out}")
    print(f"Best distance: {df['distance'].iloc[0]:.4f}")
    print(f"Best params:")
    for k in PARAM_SPACE:
        print(f"  {k}: {df[k].iloc[0]:.4f}")


if __name__ == "__main__":
    main()
