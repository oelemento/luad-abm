"""Combine chunk .npz files and run SNPE inference.

Usage:
    python sbi_combine.py \
        --chunks-dir outputs/bayesian_inference_v3/chunks \
        --out outputs/bayesian_inference_v3 \
        --data data/gaglia_2023/gaglia_summary_stats.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runs.bayesian_inference import (
    PARAM_DEFS, PARAM_NAMES, PARAM_LO, PARAM_HI,
    STAT_KEYS, load_observed_stats, run_inference, plot_posterior
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--n-posterior", type=int, default=10000)
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combine all chunks
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
    print(f"Found {len(chunk_files)} chunk files")

    theta_all, x_all = [], []
    for cf in chunk_files:
        d = np.load(cf)
        theta_all.append(d["theta"])
        x_all.append(d["x"])
        print(f"  {cf.name}: {d['theta'].shape[0]} sims")

    theta = np.concatenate(theta_all)
    x = np.concatenate(x_all)
    print(f"Total: {theta.shape[0]} simulations, {theta.shape[1]} params, {x.shape[1]} stats")

    # Save combined training data
    np.savez(out_dir / "training_data_v4.npz", theta=theta, x=x)

    # Filter NaN/inf
    theta_t = torch.tensor(theta, dtype=torch.float32)
    x_t = torch.tensor(x, dtype=torch.float32)
    valid = torch.isfinite(x_t).all(dim=1)
    theta_t = theta_t[valid]
    x_t = x_t[valid]
    print(f"{len(theta_t)} valid simulations after filtering")

    # Load observed data
    observed = load_observed_stats(args.data)
    cond_order = ["5wk_control", "5wk_treated", "8wk_control", "8wk_treated"]
    x_observed = torch.tensor(
        np.concatenate([observed[c]["mean"] for c in cond_order]),
        dtype=torch.float32
    )

    # Run SNPE
    print(f"\n=== Neural posterior estimation (SNPE) ===")
    posterior, samples = run_inference(theta_t, x_t, x_observed, args.n_posterior)
    np.save(out_dir / "posterior_samples.npy", samples.numpy())

    # Visualize
    print(f"\n=== Visualization ===")
    plot_posterior(samples, observed, out_dir)

    # Summary
    print("\n=== Posterior summary ===")
    s = samples.numpy()
    for i, name in enumerate(PARAM_NAMES):
        mean = s[:, i].mean()
        ci_lo = np.percentile(s[:, i], 2.5)
        ci_hi = np.percentile(s[:, i], 97.5)
        print(f"  {name:30s}  {mean:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")


if __name__ == "__main__":
    main()
