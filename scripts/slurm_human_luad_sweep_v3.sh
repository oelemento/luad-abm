#!/bin/bash
#SBATCH --job-name=luad_v3
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/luad_v3_%j.out
#SBATCH --error=logs/luad_v3_%j.err

set -eo pipefail
HISTCONTROL=ignoredups
export HISTCONTROL
source /etc/profile
module load anaconda3/2023.09-3
source activate luad_abm
pip install --quiet mesa h5py 2>/dev/null || true

export PYTHONUNBUFFERED=1

cd ~/LungCancerSim2/

echo "=== Human LUAD sweep v3 (SBI v7 posterior, 18 params incl kill-prolif): $(date) ==="

python scripts/human_luad_sweep.py \
    --posterior outputs/bayesian_inference_v7/posterior_samples.npy \
    --seeds 20 \
    --workers 15 \
    --out outputs/human_luad_sweep_v3

echo "=== Done: $(date) ==="
