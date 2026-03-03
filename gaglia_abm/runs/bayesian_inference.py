"""Simulation-based Bayesian inference for Gaglia ABM parameters.

Uses SNPE (Sequential Neural Posterior Estimation) from the `sbi` package
to infer posterior distributions over mechanistic parameters given observed
CyCIF spatial statistics from Gaglia et al. 2023 Dataset03.

Joint inference across all 4 conditions (5wk control, 5wk treated,
8wk control, 8wk treated): the same biological parameters must explain
the temporal evolution and treatment response simultaneously.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luad.model import LUADModel, PresetConfig
from luad.calibration import extract_summary_stats
from luad.metrics import MetricsTracker

# ── Summary statistic keys (must match both ABM output and Gaglia CSV) ──────
STAT_KEYS = [
    "frac_t_cytotox", "frac_t_helper", "frac_t_reg", "frac_macrophage",
    "ratio_cd8_cd4", "ratio_cd8_treg",
    "infilt_t_cytotox_inside", "infilt_t_cytotox_periphery",
    "infilt_t_reg_inside", "infilt_t_reg_periphery",
    # Functional marker stats (CyCIF-mapped)
    "cd8_frac_pd1_pos", "cd8_frac_exhausted",
    "cd8_frac_grzb_pos", "cd8_frac_ki67_pos",
    "tumor_frac_b2m_pos",
]

# ── Parameters to infer with prior bounds ───────────────────────────────────
PARAM_DEFS = [
    # name,                       lo,    hi
    ("cd8_base_kill",             0.02,  0.60),   # CD8 killing efficiency
    # pd_l1_penalty dropped in v4 (non-identifiable in v2+v3, fixed at 0.5)
    ("cd8_exhaustion_rate",       0.001, 0.15),   # exhaustion accumulation (v3 floor hit → widened)
    ("cd8_activation_gain",       0.02,  0.50),   # activation boost per kill
    ("tumor_proliferation_rate",  0.005, 0.20),   # tumor division prob/tick
    ("macrophage_suppr_base",     0.05,  0.50),   # M2 macrophage suppression
    ("suppressive_background",    0.01,  0.15),   # baseline immunosuppression
    ("immune_base_death_rate",    0.0005, 0.025), # immune cell turnover (v3 floor hit → widened)
    ("recruitment_rate",          0.001, 0.05),   # immune cell recruitment
    ("treg_suppression",          0.08,  0.45),   # Treg suppression strength
    ("cd8_exhaustion_death_bonus", 0.001, 0.05),  # exhaustion-scaled CD8 death (v3 floor hit → widened)
    ("treg_prolif_rate",          0.005, 0.15),   # Treg proliferation near tumor
    ("treg_death_rate",           0.001, 0.03),   # Treg turnover/death rate (new: CD8:Treg ratio)
    ("mac_tumor_death_rate",      0.01,  0.15),   # macrophage tumor-proximity death
    ("mac_recruit_suppression",   0.5,   8.0),    # tumor burden suppresses mac recruitment
    ("recruit_exhaustion_priming", 0.05, 0.6),    # CD8 recruits arrive pre-exhausted
    ("mhc_i_induction_rate",     0.005, 0.10),    # IFNg → MHC-I upregulation on tumor
    ("mhc_i_decay_rate",         0.0005, 0.01),   # MHC-I decay (immune evasion)
]
# Fixed parameters (removed from inference due to non-identifiability)
FIXED_PARAMS = {
    "pd_l1_penalty": 0.5,  # fixed at v3 posterior mean
}
PARAM_NAMES = [d[0] for d in PARAM_DEFS]
PARAM_LO = torch.tensor([d[1] for d in PARAM_DEFS], dtype=torch.float32)
PARAM_HI = torch.tensor([d[2] for d in PARAM_DEFS], dtype=torch.float32)


def load_observed_stats(csv_path: str | Path) -> dict:
    """Load observed summary statistics from Gaglia CSV, grouped by condition."""
    df = pd.read_csv(csv_path)
    groups = {
        "5wk_treated": df[df.mouse_group == 1],   # PD1+CTLA4
        "5wk_control": df[df.mouse_group == 2],    # untreated
        "8wk_treated": df[df.mouse_group == 3],
        "8wk_control": df[df.mouse_group == 4],
    }
    result = {}
    for label, gdf in groups.items():
        means = [float(gdf[k].mean()) for k in STAT_KEYS]
        stds = [float(gdf[k].std()) for k in STAT_KEYS]
        result[label] = {"mean": np.array(means), "std": np.array(stds)}
    return result


def make_preset(params_vec: np.ndarray) -> PresetConfig:
    """Build a PresetConfig with the parameter vector mapped into model params."""
    p = dict(zip(PARAM_NAMES, params_vec))
    return PresetConfig(
        name="sbi_run",
        grid_width=100,
        grid_height=100,
        initial_agents={"tumor": 1200, "cd8": 421, "cd4": 936,
                        "treg": 127, "macrophage": 88},
        cxcl9_10_mean=0.12,
        macrophage_bias=-0.1,
        macrophage_m2_fraction=0.45,
        suppression={"treg": float(p["treg_suppression"]), "macrophage": float(p["macrophage_suppr_base"])},
        suppressive_background=float(p["suppressive_background"]),
    )


def run_single_condition(params_vec: np.ndarray, interventions: list[str],
                         ticks: int, seed: int,
                         intervention_start_tick: int | None = None) -> np.ndarray:
    """Run one ABM simulation and return summary statistics vector.

    If intervention_start_tick is set, the model runs untreated until that tick,
    then interventions are applied for the remaining ticks.  This matches the
    Gaglia protocol where treatment is a 1-week pulse at the end.
    """
    # Always start with NO interventions — apply them later if delayed
    start_interventions = [] if intervention_start_tick is not None else interventions
    preset = make_preset(params_vec)
    tracker = MetricsTracker(distance_interval=999, interaction_interval=999,
                             grid_interval=999, capture_grids=False)
    model = LUADModel(preset=preset, interventions=start_interventions,
                      seed=seed, metrics_tracker=tracker)

    # Apply all inferred parameters to model.params
    p = dict(zip(PARAM_NAMES, params_vec))
    for key in PARAM_NAMES:
        if key not in ("treg_suppression", "treg_death_rate"):  # handled elsewhere
            model.params[key] = float(p[key])
    # Apply fixed (non-inferred) parameters
    for key, val in FIXED_PARAMS.items():
        model.params[key] = val
    # Store treg_death_rate for use in Treg death logic
    model.params["treg_death_rate"] = float(p["treg_death_rate"])

    for tick in range(ticks):
        # Delayed intervention: turn on treatment at the specified tick
        if intervention_start_tick is not None and tick == intervention_start_tick:
            model.apply_interventions(interventions)
        model.step()

    stats = extract_summary_stats(model)
    return np.array([stats.get(k, 0.0) for k in STAT_KEYS], dtype=np.float32)


# Timing constants (24 ticks = 1 week)
TICKS_PER_WEEK = 24
TICKS_5WK = 5 * TICKS_PER_WEEK   # 120
TICKS_8WK = 8 * TICKS_PER_WEEK   # 192
# Treatment is a 1-week pulse at the end (Gaglia protocol: 3 doses over days 0,3,6)
TREATMENT_DURATION = 1 * TICKS_PER_WEEK  # 24 ticks = 1 week
TRT_START_5WK = TICKS_5WK - TREATMENT_DURATION  # tick 96 (week 4)
TRT_START_8WK = TICKS_8WK - TREATMENT_DURATION  # tick 168 (week 7)


def simulator(params_vec: np.ndarray, ticks: int = 120, seed: int | None = None) -> np.ndarray:
    """Run ABM for all 4 conditions, return concatenated summary statistics.

    Matches Gaglia protocol: treatment is a 1-week pulse at the end, not from
    the start.  Control runs have no treatment.  Treated runs grow untreated
    for (N-1) weeks, then receive PD1+CTLA4 for the final week.

    Returns a vector of length 4 * len(STAT_KEYS):
      [5wk_control..., 5wk_treated..., 8wk_control..., 8wk_treated...]
    """
    if seed is None:
        seed = np.random.randint(0, 2**31)

    ctrl_5 = run_single_condition(params_vec, [], TICKS_5WK, seed)
    trt_5 = run_single_condition(params_vec, ["PD1", "CTLA4"], TICKS_5WK, seed + 1,
                                  intervention_start_tick=TRT_START_5WK)
    ctrl_8 = run_single_condition(params_vec, [], TICKS_8WK, seed + 2)
    trt_8 = run_single_condition(params_vec, ["PD1", "CTLA4"], TICKS_8WK, seed + 3,
                                  intervention_start_tick=TRT_START_8WK)
    return np.concatenate([ctrl_5, trt_5, ctrl_8, trt_8])


def _worker(args):
    """Worker function for parallel simulation."""
    import gc
    idx, params_vec = args
    t0 = time.time()
    try:
        x = simulator(params_vec)
        dt = time.time() - t0
        return idx, x, dt
    except Exception as e:
        print(f"  Sim {idx} failed: {e}", flush=True)
        return idx, None, 0.0
    finally:
        gc.collect()


def generate_training_data(n_sims: int, ticks: int, n_workers: int,
                           save_path: Path, resume: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (theta, x) training pairs for SNPE, with checkpointing.

    Uses Pool(maxtasksperchild=1) so each worker process is killed and
    restarted after every simulation — prevents Mesa memory leaks.
    """
    theta_all = []
    x_all = []
    start_idx = 0

    if resume and save_path.exists():
        checkpoint = np.load(save_path)
        theta_all = list(checkpoint["theta"])
        x_all = list(checkpoint["x"])
        start_idx = len(theta_all)
        print(f"Resumed from checkpoint: {start_idx}/{n_sims} simulations done", flush=True)
        if start_idx >= n_sims:
            return (torch.tensor(np.array(theta_all), dtype=torch.float32),
                    torch.tensor(np.array(x_all), dtype=torch.float32))

    remaining = n_sims - start_idx
    rng = np.random.default_rng(42 + start_idx)
    theta_new = rng.uniform(
        PARAM_LO.numpy(), PARAM_HI.numpy(),
        size=(remaining, len(PARAM_NAMES))
    )

    # Small batches = frequent checkpoints + fresh Pool per batch
    batch_size = n_workers * 2
    completed = start_idx
    t_start = time.time()
    failed = 0

    for batch_start in range(0, remaining, batch_size):
        batch_end = min(batch_start + batch_size, remaining)
        batch_theta = theta_new[batch_start:batch_end]
        tasks = [(start_idx + batch_start + i, batch_theta[i])
                 for i in range(len(batch_theta))]

        if n_workers > 1:
            # Fresh pool per batch with maxtasksperchild=1:
            # each worker dies after 1 sim, freeing all Mesa memory
            with Pool(n_workers, maxtasksperchild=1) as pool:
                results = pool.map(_worker, tasks, chunksize=1)
        else:
            results = [_worker(t) for t in tasks]

        for idx, x, dt in results:
            if x is not None:
                theta_all.append(batch_theta[idx - start_idx - batch_start])
                x_all.append(x)
            else:
                failed += 1

        completed = len(theta_all)
        elapsed = time.time() - t_start
        rate_per_min = (completed - start_idx) / (elapsed / 60) if elapsed > 0 else 0
        eta_min = (n_sims - completed) / rate_per_min if rate_per_min > 0 else float("inf")
        print(f"  {completed}/{n_sims} sims | "
              f"{rate_per_min:.1f} sims/min | ETA {eta_min:.0f} min | "
              f"{failed} failed", flush=True)

        # Checkpoint after each batch
        np.savez(save_path,
                 theta=np.array(theta_all),
                 x=np.array(x_all))

    theta_tensor = torch.tensor(np.array(theta_all), dtype=torch.float32)
    x_tensor = torch.tensor(np.array(x_all), dtype=torch.float32)
    return theta_tensor, x_tensor


def zscore_normalize(x: torch.Tensor, x_observed: torch.Tensor):
    """Z-score normalize simulation outputs using training data statistics.

    Returns normalized x, normalized x_observed, and (mean, std) for reference.
    This ensures all summary statistics contribute equally to the SNPE loss
    regardless of their natural scale (e.g., ratios ~3.5 vs fractions ~0.05).
    """
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0)
    # Avoid division by zero for constant columns
    x_std = torch.clamp(x_std, min=1e-8)
    x_norm = (x - x_mean) / x_std
    x_obs_norm = (x_observed - x_mean) / x_std
    return x_norm, x_obs_norm, x_mean, x_std


def run_inference(theta: torch.Tensor, x: torch.Tensor,
                  x_observed: torch.Tensor, n_posterior_samples: int = 10000):
    """Train SNPE and sample from the posterior."""
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform

    # Z-score normalize so all stats contribute equally
    x_norm, x_obs_norm, x_mean, x_std = zscore_normalize(x, x_observed)
    print(f"Z-score normalized {x.shape[1]} summary statistics")

    prior = BoxUniform(low=PARAM_LO, high=PARAM_HI)

    inference = SNPE(prior=prior)
    inference.append_simulations(theta, x_norm)

    print("Training neural density estimator...")
    density_estimator = inference.train(show_train_summary=True)
    posterior = inference.build_posterior(density_estimator)

    print(f"Sampling {n_posterior_samples} posterior samples...")
    samples = posterior.sample((n_posterior_samples,), x=x_obs_norm)
    return posterior, samples


def plot_posterior(samples: torch.Tensor, observed_stats: dict, out_dir: Path) -> None:
    """Plot marginal posterior distributions and pairwise correlations."""
    import matplotlib.pyplot as plt

    samples_np = samples.numpy()
    n_params = len(PARAM_NAMES)

    # Marginal posteriors
    n_rows = (n_params + 4) // 5  # ceil division for 5 columns
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    for i, (name, lo, hi) in enumerate(PARAM_DEFS):
        ax = axes[i]
        ax.hist(samples_np[:, i], bins=50, density=True, color="#3399e6", alpha=0.7, edgecolor="white")
        ax.axvline(samples_np[:, i].mean(), color="red", linewidth=2, label="posterior mean")
        ax.set_xlim(lo, hi)
        ax.set_title(name.replace("_", " "), fontsize=10, fontweight="bold")
        ax.set_ylabel("density")
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Posterior distributions over mechanistic parameters", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "posterior_marginals.png", dpi=150, bbox_inches="tight")
    print(f"Saved posterior_marginals.png")

    # Summary table
    rows = []
    for i, name in enumerate(PARAM_NAMES):
        s = samples_np[:, i]
        rows.append({
            "parameter": name,
            "posterior_mean": f"{s.mean():.4f}",
            "posterior_std": f"{s.std():.4f}",
            "95%_CI_lo": f"{np.percentile(s, 2.5):.4f}",
            "95%_CI_hi": f"{np.percentile(s, 97.5):.4f}",
            "prior_lo": f"{PARAM_DEFS[i][1]:.4f}",
            "prior_hi": f"{PARAM_DEFS[i][2]:.4f}",
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "posterior_summary.csv", index=False)
    print(f"Saved posterior_summary.csv")

    # Posterior predictive check: run ABM with posterior mean params
    mean_params = samples_np.mean(axis=0)
    pred_stats = simulator(mean_params)
    cond_order = ["5wk_control", "5wk_treated", "8wk_control", "8wk_treated"]
    x_obs_np = np.concatenate([observed_stats[c]["mean"] for c in cond_order])

    labels = []
    for c in cond_order:
        short = c.replace("_control", "_C").replace("_treated", "_T")
        labels.extend([f"{short}_{k}" for k in STAT_KEYS])
    fig2, ax2 = plt.subplots(figsize=(18, 5))
    x_pos = np.arange(len(labels))
    ax2.bar(x_pos - 0.15, x_obs_np, 0.3, label="Observed (Gaglia)", color="#e86464", alpha=0.8)
    ax2.bar(x_pos + 0.15, pred_stats, 0.3, label="Predicted (posterior mean)", color="#3399e6", alpha=0.8)
    # Shade held-out 8wk region
    n_5wk = 2 * len(STAT_KEYS)
    ax2.axvspan(n_5wk - 0.5, len(labels) - 0.5, alpha=0.08, color="green")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax2.set_ylabel("Value")
    ax2.set_title("Posterior predictive check: 4 conditions (green = 8wk)", fontweight="bold")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "posterior_predictive.png", dpi=150, bbox_inches="tight")
    print(f"Saved posterior_predictive.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian inference for Gaglia ABM")
    parser.add_argument("--n-sims", type=int, default=2000, help="Number of training simulations")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers")
    parser.add_argument("--out", type=str, default="outputs/bayesian_inference", help="Output directory")
    parser.add_argument("--data", type=str, default="data/gaglia_2023/gaglia_summary_stats.csv")
    parser.add_argument("--train-only", action="store_true", help="Only generate training data, skip inference")
    parser.add_argument("--infer-only", action="store_true", help="Skip simulation, run inference on existing data")
    parser.add_argument("--n-posterior", type=int, default=10000, help="Number of posterior samples")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load observed data — all 4 conditions
    observed = load_observed_stats(args.data)
    cond_order = ["5wk_control", "5wk_treated", "8wk_control", "8wk_treated"]
    x_observed = torch.tensor(
        np.concatenate([observed[c]["mean"] for c in cond_order]),
        dtype=torch.float32
    )
    print(f"Observed summary stats (4 conditions): {x_observed.shape[0]} dimensions")
    print(f"Parameters to infer: {len(PARAM_NAMES)}")

    checkpoint_path = out_dir / "training_data_v5.npz"

    if not args.infer_only:
        print(f"\n=== Phase 1: Generating {args.n_sims} training simulations ===")
        print(f"  {args.workers} workers")
        print(f"  Each sim runs 4 conditions (5wk/8wk × control/treated)")
        t0 = time.time()
        theta, x = generate_training_data(
            args.n_sims, 0, args.workers, checkpoint_path
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed/60:.1f} min ({len(theta)} valid simulations)")

        if args.train_only:
            print("Training data saved. Exiting.")
            return
    else:
        print("Loading existing training data...")
        data = np.load(checkpoint_path)
        theta = torch.tensor(data["theta"], dtype=torch.float32)
        x = torch.tensor(data["x"], dtype=torch.float32)
        print(f"  Loaded {len(theta)} simulations")

    # Filter out any NaN/inf simulations
    valid = torch.isfinite(x).all(dim=1)
    theta = theta[valid]
    x = x[valid]
    print(f"  {len(theta)} valid simulations after filtering")

    print(f"\n=== Phase 2: Neural posterior estimation (SNPE) ===")
    posterior, samples = run_inference(theta, x, x_observed, args.n_posterior)

    # Save posterior samples
    np.save(out_dir / "posterior_samples.npy", samples.numpy())

    print(f"\n=== Phase 3: Visualization ===")
    plot_posterior(samples, observed, out_dir)

    # Print summary
    print("\n=== Posterior summary ===")
    s = samples.numpy()
    for i, name in enumerate(PARAM_NAMES):
        mean = s[:, i].mean()
        ci_lo = np.percentile(s[:, i], 2.5)
        ci_hi = np.percentile(s[:, i], 97.5)
        print(f"  {name:30s}  {mean:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
