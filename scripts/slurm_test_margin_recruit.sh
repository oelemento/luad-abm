#!/bin/bash
#SBATCH --job-name=margin_test
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/margin_test_%j.out
#SBATCH --error=logs/margin_test_%j.err

set -eo pipefail
HISTCONTROL=ignoredups
export HISTCONTROL
source /etc/profile
module load anaconda3/2023.09-3
source activate luad_abm
pip install --quiet mesa h5py 2>/dev/null || true

export PYTHONUNBUFFERED=1

cd ~/LungCancerSim2/

echo "=== Margin + recruit boost test: $(date) ==="

python scripts/test_margin_recruit.py \
    --posterior outputs/bayesian_inference_v8/posterior_samples.npy \
    --seeds 10 \
    --workers 15 \
    --out outputs/test_margin_recruit

echo "=== Done: $(date) ==="
