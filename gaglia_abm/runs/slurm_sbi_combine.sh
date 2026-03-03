#!/bin/bash
#SBATCH --job-name=sbi_v5_snpe
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/sbi_v5_combine_%j.out
#SBATCH --error=logs/sbi_v5_combine_%j.err

set -eo pipefail
source /etc/profile
module load anaconda3/2023.09-3
source activate luad_abm
pip install --quiet sbi mesa torch 2>/dev/null || true

cd ~/LungCancerSim2/

echo "=== SBI v5 combine + SNPE: $(date) ==="

python gaglia_abm/runs/sbi_combine.py \
    --chunks-dir outputs/bayesian_inference_v5/chunks \
    --out outputs/bayesian_inference_v5 \
    --data data/gaglia_2023/gaglia_summary_stats.csv \
    --n-posterior 10000

echo "=== Done: $(date) ==="
