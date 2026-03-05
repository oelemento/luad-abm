#!/bin/bash
#SBATCH --job-name=seq_dose
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/seq_dose_%j.out
#SBATCH --error=logs/seq_dose_%j.err

set -eo pipefail
source /etc/profile
module load anaconda3/2023.09-3
source activate luad_abm
pip install --quiet mesa 2>/dev/null || true

cd ~/LungCancerSim2/

echo "=== Sequential dosing sweep: $(date) ==="

python scripts/sequential_dosing_sweep.py \
    --posterior outputs/bayesian_inference_v6/posterior_samples.npy \
    --seeds 20 \
    --out outputs/sequential_dosing

echo "=== Done: $(date) ==="
