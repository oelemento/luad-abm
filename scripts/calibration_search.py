"""Latin Hypercube parameter search for ABM calibration against Gaglia data.

Calibrates against all 4 treatment groups simultaneously:
  G1=Control, G2=anti-PD1, G3=anti-CTLA4, G4=Combo (PD1+CTLA4)

Usage:
    python scripts/calibration_search.py \
        --obs data/gaglia_2023/gaglia_summary_stats.csv \
        --preset luad_abm/config/G3_inflammatory.json \
        --n-samples 200 \
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

# Parameter search space (7 params: 6 biological + snapshot_tick)
PARAM_SPACE = {
    "cd8_base_kill":          (0.05, 0.30),
    "cd8_exhaustion_rate":    (0.02, 0.12),
    "pd_l1_penalty":          (0.30, 0.90),
    "macrophage_suppr_base":  (0.10, 0.50),
    "suppressive_background": (0.02, 0.15),
    "tumor_proliferation_rate": (0.01, 0.06),
    "snapshot_tick":          (80, 200),
}

# Treatment groups: Gaglia mouse_group -> ABM interventions
TREATMENT_GROUPS = {
    1: [],                    # Control
    2: ["PD1"],               # anti-PD1
    3: ["CTLA4"],             # anti-CTLA4
    4: ["PD1", "CTLA4"],      # Combo
}

# Keys to use for distance computation (shared-type fractions + ratios + infiltration)
CALIBRATION_KEYS = [
    "frac_t_cytotox", "frac_t_helper", "frac_t_reg", "frac_macrophage",
    "ratio_cd8_cd4", "ratio_cd8_treg", "ratio_cd4_treg", "ratio_cd8_mac",
    "infilt_t_cytotox_inside", "infilt_t_cytotox_cuff", "infilt_t_cytotox_periphery",
    "infilt_t_helper_inside", "infilt_t_helper_cuff", "infilt_t_helper_periphery",
    "infilt_t_reg_inside", "infilt_t_reg_cuff", "infilt_t_reg_periphery",
    "cd8_mean_dist_to_tumor", "cd8_frac_within_5",
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


def _run_one_group(preset_path: str, param_overrides: dict, ticks: int,
                   interventions: list, seed: int, sample_idx: int,
                   group: int, obs_mean: dict, obs_var: dict,
                   calibration_keys: list) -> dict:
    """Run one ABM simulation for one treatment group and return stats + distance.

    Module-level function (not a closure) so it can be pickled by ProcessPoolExecutor.
    """
    preset = load_preset(Path(preset_path))
    tracker = MetricsTracker(capture_grids=False, distance_interval=ticks + 1,
                             interaction_interval=ticks + 1)
    model = LUADModel(preset=preset, interventions=interventions,
                      seed=seed, metrics_tracker=tracker)
    # Override biological params (not snapshot_tick)
    for k, v in param_overrides.items():
        if k != "snapshot_tick":
            model.params[k] = v
    for _ in range(ticks):
        model.step()

    stats = extract_summary_stats(model)
    dist = compute_distance(obs_mean, stats, obs_var, calibration_keys)
    return {
        **param_overrides,
        "snapshot_tick": ticks,
        "seed": seed,
        "sample_idx": sample_idx,
        "group": group,
        "interventions": ",".join(interventions) if interventions else "none",
        "distance": dist,
        **{f"sim_{k}": v for k, v in stats.items()},
    }


def compute_obs_targets(obs_df: pd.DataFrame, group: int) -> tuple[dict, dict]:
    """Compute mean and variance of observed stats for a mouse group."""
    sub = obs_df[obs_df["mouse_group"] == group]
    means = {}
    variances = {}
    for col in sub.columns:
        if col in ("mouse_num", "mouse_group", "n_cells"):
            continue
        if col.startswith("frac_all_"):
            continue  # skip full-immune fractions, use shared-type only
        vals = sub[col].dropna().values
        if len(vals) > 0:
            means[col] = float(vals.mean())
            variances[col] = float(vals.var()) if len(vals) > 1 else float(vals.mean() ** 2 * 0.1 + 1e-6)
    return means, variances


def main():
    parser = argparse.ArgumentParser(
        description="Multi-group LHS parameter search for ABM calibration"
    )
    parser.add_argument("--obs", required=True, help="Gaglia summary stats CSV")
    parser.add_argument("--preset", required=True, help="Base preset JSON")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=2, help="Replicates per param set")
    parser.add_argument("--out", required=True, help="Output CSV")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    obs_df = pd.read_csv(args.obs)

    # Compute observed targets for each treatment group
    obs_targets = {}
    for group in TREATMENT_GROUPS:
        obs_targets[group] = compute_obs_targets(obs_df, group)

    samples = generate_lhs_samples(args.n_samples)
    n_groups = len(TREATMENT_GROUPS)
    total = args.n_samples * args.seeds * n_groups
    print(f"Running {args.n_samples} param sets x {args.seeds} seeds x {n_groups} groups = {total} simulations")

    preset_path = str(Path(args.preset).resolve())

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for idx, row in samples.iterrows():
            overrides = row.to_dict()
            ticks = int(round(overrides["snapshot_tick"]))
            for s in range(args.seeds):
                for group, interventions in TREATMENT_GROUPS.items():
                    seed = 10000 * idx + 100 * s + group
                    obs_mean, obs_var = obs_targets[group]
                    fut = pool.submit(
                        _run_one_group,
                        preset_path,
                        overrides,
                        ticks,
                        interventions,
                        seed,
                        idx,
                        group,
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
                if completed % 50 == 0 or completed == total:
                    print(f"  {completed}/{total} done")
            except Exception as e:
                print(f"  Error: {e}")
                completed += 1

    df = pd.DataFrame(results)

    # Compute aggregate distance per parameter set (mean across groups and seeds)
    agg = df.groupby("sample_idx")["distance"].mean().reset_index()
    agg.columns = ["sample_idx", "mean_distance"]
    df = df.merge(agg, on="sample_idx")

    df.sort_values("mean_distance", inplace=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Report best
    best_idx = df["sample_idx"].iloc[0]
    best = df[df["sample_idx"] == best_idx].iloc[0]
    print(f"\nResults saved to {args.out}")
    print(f"Best mean distance: {best['mean_distance']:.4f}")
    print(f"Best params:")
    for k in PARAM_SPACE:
        print(f"  {k}: {best[k]:.4f}")
    print(f"\nPer-group distances for best set:")
    for _, r in df[df["sample_idx"] == best_idx].iterrows():
        print(f"  G{int(r['group'])} ({r['interventions']}): {r['distance']:.4f}")


if __name__ == "__main__":
    main()
