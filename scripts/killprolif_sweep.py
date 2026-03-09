"""Sweep cd8_kill_prolif_prob values to find the right balance.

Tests multiple kill-prolif probabilities across all patients and arms.
Saves per-value checkpoints for robustness.

Usage:
    python3.11 scripts/killprolif_sweep.py \
        --posterior outputs/bayesian_inference_v6/posterior_samples.npy \
        --seeds 5 --workers 15 --out outputs/killprolif_sweep
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GAGLIA_ROOT = REPO_ROOT / "gaglia_abm"
if str(GAGLIA_ROOT) not in sys.path:
    sys.path.insert(0, str(GAGLIA_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runs.bayesian_inference import PARAM_NAMES, FIXED_PARAMS, TICKS_PER_WEEK
from scripts.human_luad_sweep import (
    load_patient_compositions, patient_to_initial_agents,
    run_all, ARMS, GRID_TOTAL,
)

KILL_PROLIF_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=str, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--data-dir", type=str,
                        default="data/gaglia_2023/Dataset05_Human_Lung_Adenocarcinoma")
    parser.add_argument("--workers", type=int,
                        default=max(1, os.cpu_count() - 1))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load posterior
    samples = np.load(args.posterior)
    params_mean = samples.mean(axis=0).astype(np.float32)
    print(f"Posterior mean ({len(params_mean)} params)")

    # Load patients
    patients = load_patient_compositions(Path(args.data_dir))
    print(f"Loaded {len(patients)} patients")

    all_sweep_results = {}

    for kp_val in KILL_PROLIF_VALUES:
        print(f"\n{'='*80}")
        print(f"  kill_prolif = {kp_val}")
        print(f"{'='*80}")

        extra_params = {"cd8_kill_prolif_prob": kp_val} if kp_val > 0 else None

        results = run_all(params_mean, patients, args.seeds,
                          out_dir=None,  # don't save per-config checkpoints
                          n_workers=args.workers,
                          extra_params=extra_params)

        # Prefix keys with kp value
        for key, val in results.items():
            all_sweep_results[f"kp{kp_val}__{key}"] = val

        # Save checkpoint after each kp value
        ckpt_path = out_dir / "killprolif_sweep_checkpoint.npz"
        np.savez(ckpt_path,
                 **{k: [r for r in v] for k, v in all_sweep_results.items()})
        print(f"\n  [checkpoint saved: {len(all_sweep_results)} conditions after kp={kp_val}]",
              flush=True)

    # Save final results
    np.savez(out_dir / "killprolif_sweep_results.npz",
             **{k: [r for r in v] for k, v in all_sweep_results.items()})

    # Summary table
    configs = ["KP_mouse"] + [pt["name"] for pt in patients]
    print(f"\n{'='*120}")
    print("Kill-Prolif Sweep Summary (PD1_CTLA4 arm, ΔTumor% vs untreated)")
    print(f"{'='*120}")
    header = f"{'Config':<12}"
    for kp_val in KILL_PROLIF_VALUES:
        header += f" {'kp='+str(kp_val):>12}"
    print(header)
    print("-" * 120)

    for config in configs:
        row = f"{config:<12}"
        for kp_val in KILL_PROLIF_VALUES:
            ut_key = f"kp{kp_val}__{config}__untreated"
            tx_key = f"kp{kp_val}__{config}__PD1_CTLA4"
            if ut_key in all_sweep_results and tx_key in all_sweep_results:
                baseline = np.mean([r["n_tumor"] for r in all_sweep_results[ut_key]])
                treated = np.mean([r["n_tumor"] for r in all_sweep_results[tx_key]])
                if baseline > 0:
                    delta = (treated - baseline) / baseline * 100
                    row += f" {delta:>+11.1f}%"
                else:
                    row += f" {'cleared':>12}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
