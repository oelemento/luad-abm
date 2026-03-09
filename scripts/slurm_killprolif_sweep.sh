#!/bin/bash
#SBATCH --job-name=kp_sweep
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/kp_sweep_%j.out
#SBATCH --error=logs/kp_sweep_%j.err

set -eo pipefail
HISTCONTROL=ignoredups
export HISTCONTROL
source /etc/profile
module load anaconda3/2023.09-3
source activate luad_abm
pip install --quiet mesa h5py 2>/dev/null || true

export PYTHONUNBUFFERED=1

cd ~/LungCancerSim2/

echo "=== Kill-prolif sweep: $(date) ==="

python scripts/killprolif_sweep.py \
    --posterior outputs/bayesian_inference_v6/posterior_samples.npy \
    --seeds 5 \
    --workers 15 \
    --out outputs/killprolif_sweep

echo "=== Done: $(date) ==="
