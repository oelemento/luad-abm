#!/bin/bash
#SBATCH --job-name=trt_sweep
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/trt_sweep_%j.out
#SBATCH --error=logs/trt_sweep_%j.err

set -eo pipefail
source /etc/profile
module load anaconda3/2023.09-3
source activate luad_abm
pip install --quiet mesa 2>/dev/null || true

cd ~/LungCancerSim2/

echo "=== Treatment timing sweep: $(date) ==="

python scripts/treatment_timing_sweep.py \
    --posterior outputs/bayesian_inference_v6/posterior_samples.npy \
    --seeds 20 \
    --out outputs/treatment_timing_sweep

echo "=== Done: $(date) ==="
