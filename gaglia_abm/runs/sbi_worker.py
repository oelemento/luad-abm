"""Worker script for distributed SBI simulation.

Each worker generates a chunk of (theta, x) training pairs and saves to a
separate .npz file. A final combine step merges all chunks.

Usage:
    python sbi_worker.py --chunk-id 0 --n-chunks 10 --n-total 2000 \
        --workers 6 --out-dir outputs/bayesian_inference_v3/chunks
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runs.bayesian_inference import (
    PARAM_NAMES, PARAM_LO, PARAM_HI, simulator
)


def _worker(args):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--n-chunks", type=int, default=10)
    parser.add_argument("--n-total", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = out_dir / f"chunk_{args.chunk_id:03d}.npz"

    # Determine this chunk's range
    per_chunk = args.n_total // args.n_chunks
    start = args.chunk_id * per_chunk
    end = start + per_chunk if args.chunk_id < args.n_chunks - 1 else args.n_total
    n_sims = end - start

    print(f"Chunk {args.chunk_id}/{args.n_chunks}: sims {start}-{end-1} "
          f"({n_sims} sims, {args.workers} workers)", flush=True)

    # Sample parameters (deterministic per chunk)
    rng = np.random.default_rng(42 + start)
    theta = rng.uniform(
        PARAM_LO.numpy(), PARAM_HI.numpy(),
        size=(n_sims, len(PARAM_NAMES))
    )

    theta_all = []
    x_all = []
    failed = 0
    batch_size = args.workers * 2
    t_start = time.time()

    for batch_start in range(0, n_sims, batch_size):
        batch_end = min(batch_start + batch_size, n_sims)
        batch_theta = theta[batch_start:batch_end]
        tasks = [(batch_start + i, batch_theta[i]) for i in range(len(batch_theta))]

        with Pool(args.workers, maxtasksperchild=1) as pool:
            results = pool.map(_worker, tasks, chunksize=1)

        for idx, x, dt in results:
            if x is not None:
                theta_all.append(batch_theta[idx - batch_start])
                x_all.append(x)
            else:
                failed += 1

        elapsed = time.time() - t_start
        done = len(theta_all)
        rate = done / (elapsed / 60) if elapsed > 0 else 0
        print(f"  {done}/{n_sims} | {rate:.1f} sims/min | {failed} failed", flush=True)

        # Checkpoint after each batch
        np.savez(chunk_path,
                 theta=np.array(theta_all),
                 x=np.array(x_all))

    elapsed = time.time() - t_start
    print(f"Chunk {args.chunk_id} done: {len(theta_all)} sims in {elapsed/60:.1f} min "
          f"({failed} failed)", flush=True)


if __name__ == "__main__":
    main()
