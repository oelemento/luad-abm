"""Simulation-based Bayesian inference for Gaglia ABM parameters.

Uses SNPE (Sequential Neural Posterior Estimation) from the `sbi` package
to infer posterior distributions over mechanistic parameters given observed
CyCIF spatial statistics from Gaglia et al. 2023 Dataset03.

Joint inference: the same biological parameters must explain both the
untreated (Group 2) and PD1+CTLA4-treated (Group 1) conditions.
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
]

# ── Parameters to infer with prior bounds ───────────────────────────────────
PARAM_DEFS = [
    # name,                    lo,    hi,    description
    ("cd8_base_kill",          0.02,  0.40),  # CD8 killing efficiency
    ("pd_l1_penalty",          0.1,   0.9),   # PD-L1 suppression of killing
    ("cd8_exhaustion_rate",    0.005, 0.15),   # exhaustion accumulation rate
    ("cd8_activation_gain",    0.02,  0.30),   # activation boost per kill
    ("tumor_proliferation_rate", 0.005, 0.08), # tumor division probability/tick
    ("macrophage_suppr_base",  0.05,  0.50),   # M2 macrophage suppression
    ("suppressive_background", 0.01,  0.15),   # baseline immunosuppression
    ("immune_base_death_rate", 0.001, 0.015),  # immune cell turnover rate
    ("recruitment_rate",       0.001, 0.015),  # immune cell recruitment rate
    ("treg_suppression",       0.08,  0.45),   # Treg suppression strength
]
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
                         ticks: int, seed: int) -> np.ndarray:
    """Run one ABM simulation and return summary statistics vector."""
    preset = make_preset(params_vec)
    tracker = MetricsTracker(distance_interval=999, interaction_interval=999,
                             grid_interval=999, capture_grids=False)
    model = LUADModel(preset=preset, interventions=interventions,
                      seed=seed, metrics_tracker=tracker)

    # Apply inferred parameters to model.params
    p = dict(zip(PARAM_NAMES, params_vec))
    for key in ["cd8_base_kill", "pd_l1_penalty", "cd8_exhaustion_rate",
                "cd8_activation_gain", "tumor_proliferation_rate",
                "macrophage_suppr_base", "suppressive_background",
                "immune_base_death_rate", "recruitment_rate"]:
        model.params[key] = float(p[key])

    for _ in range(ticks):
        model.step()

    stats = extract_summary_stats(model)
    return np.array([stats.get(k, 0.0) for k in STAT_KEYS], dtype=np.float32)


def simulator(params_vec: np.ndarray, ticks: int = 120, seed: int | None = None) -> np.ndarray:
    """Run ABM for both conditions, return concatenated summary statistics.

    Returns a vector of length 2 * len(STAT_KEYS):
      [control_stats..., treated_stats...]
    """
    if seed is None:
        seed = np.random.randint(0, 2**31)

    control_stats = run_single_condition(params_vec, [], ticks, seed)
    treated_stats = run_single_condition(params_vec, ["PD1", "CTLA4"], ticks, seed)
    return np.concatenate([control_stats, treated_stats])


def _worker(args):
    """Worker function for parallel simulation."""
    idx, params_vec, ticks = args
    t0 = time.time()
    try:
        x = simulator(params_vec, ticks=ticks)
        dt = time.time() - t0
        return idx, x, dt
    except Exception as e:
        print(f"  Sim {idx} failed: {e}")
        return idx, None, 0.0


def generate_training_data(n_sims: int, ticks: int, n_workers: int,
                           save_path: Path, resume: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (theta, x) training pairs for SNPE, with checkpointing."""
    # Check for existing checkpoint
    theta_all = []
    x_all = []
    start_idx = 0

    if resume and save_path.exists():
        checkpoint = np.load(save_path)
        theta_all = list(checkpoint["theta"])
        x_all = list(checkpoint["x"])
        start_idx = len(theta_all)
        print(f"Resumed from checkpoint: {start_idx}/{n_sims} simulations done")
        if start_idx >= n_sims:
            return (torch.tensor(np.array(theta_all), dtype=torch.float32),
                    torch.tensor(np.array(x_all), dtype=torch.float32))

    # Sample parameters from prior
    remaining = n_sims - start_idx
    rng = np.random.default_rng(42 + start_idx)
    theta_new = rng.uniform(
        PARAM_LO.numpy(), PARAM_HI.numpy(),
        size=(remaining, len(PARAM_NAMES))
    )

    # Run simulations in batches
    batch_size = max(n_workers * 2, 20)
    completed = start_idx
    t_start = time.time()

    for batch_start in range(0, remaining, batch_size):
        batch_end = min(batch_start + batch_size, remaining)
        batch_theta = theta_new[batch_start:batch_end]
        tasks = [(start_idx + batch_start + i, batch_theta[i], ticks)
                 for i in range(len(batch_theta))]

        if n_workers > 1:
            with Pool(n_workers) as pool:
                results = pool.map(_worker, tasks)
        else:
            results = [_worker(t) for t in tasks]

        for idx, x, dt in results:
            if x is not None:
                theta_all.append(theta_new[idx - start_idx])
                x_all.append(x)

        completed = len(theta_all)
        elapsed = time.time() - t_start
        rate = (completed - start_idx) / elapsed if elapsed > 0 else 0
        eta = (n_sims - completed) / rate if rate > 0 else float("inf")
        print(f"  {completed}/{n_sims} simulations | "
              f"{rate:.1f} sims/min | ETA {eta/60:.1f} min")

        # Checkpoint
        np.savez(save_path,
                 theta=np.array(theta_all),
                 x=np.array(x_all))

    theta_tensor = torch.tensor(np.array(theta_all), dtype=torch.float32)
    x_tensor = torch.tensor(np.array(x_all), dtype=torch.float32)
    return theta_tensor, x_tensor


def run_inference(theta: torch.Tensor, x: torch.Tensor,
                  x_observed: torch.Tensor, n_posterior_samples: int = 10000):
    """Train SNPE and sample from the posterior."""
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform

    prior = BoxUniform(low=PARAM_LO, high=PARAM_HI)

    inference = SNPE(prior=prior)
    inference.append_simulations(theta, x)

    print("Training neural density estimator...")
    density_estimator = inference.train(show_train_summary=True)
    posterior = inference.build_posterior(density_estimator)

    print(f"Sampling {n_posterior_samples} posterior samples...")
    samples = posterior.sample((n_posterior_samples,), x=x_observed)
    return posterior, samples


def plot_posterior(samples: torch.Tensor, observed_stats: dict, out_dir: Path) -> None:
    """Plot marginal posterior distributions and pairwise correlations."""
    import matplotlib.pyplot as plt

    samples_np = samples.numpy()
    n_params = len(PARAM_NAMES)

    # Marginal posteriors
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
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
    pred_stats = simulator(mean_params, ticks=120)
    x_obs_np = np.concatenate([observed_stats["5wk_control"]["mean"],
                                observed_stats["5wk_treated"]["mean"]])

    labels = [f"ctrl_{k}" for k in STAT_KEYS] + [f"trt_{k}" for k in STAT_KEYS]
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    x_pos = np.arange(len(labels))
    ax2.bar(x_pos - 0.15, x_obs_np, 0.3, label="Observed (Gaglia)", color="#e86464", alpha=0.8)
    ax2.bar(x_pos + 0.15, pred_stats, 0.3, label="Predicted (posterior mean)", color="#3399e6", alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Value")
    ax2.set_title("Posterior predictive check: Observed vs Predicted", fontweight="bold")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "posterior_predictive.png", dpi=150, bbox_inches="tight")
    print(f"Saved posterior_predictive.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian inference for Gaglia ABM")
    parser.add_argument("--n-sims", type=int, default=2000, help="Number of training simulations")
    parser.add_argument("--ticks", type=int, default=120, help="Ticks per simulation")
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

    # Load observed data
    observed = load_observed_stats(args.data)
    x_observed = torch.tensor(
        np.concatenate([observed["5wk_control"]["mean"],
                        observed["5wk_treated"]["mean"]]),
        dtype=torch.float32
    )
    print(f"Observed summary stats (control + treated): {x_observed.shape[0]} dimensions")
    print(f"Parameters to infer: {len(PARAM_NAMES)}")

    checkpoint_path = out_dir / "training_data.npz"

    if not args.infer_only:
        print(f"\n=== Phase 1: Generating {args.n_sims} training simulations ===")
        print(f"  {args.ticks} ticks/sim, {args.workers} workers")
        print(f"  Each sim runs 2 conditions (control + treated)")
        t0 = time.time()
        theta, x = generate_training_data(
            args.n_sims, args.ticks, args.workers, checkpoint_path
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
