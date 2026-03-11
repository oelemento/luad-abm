#!/bin/bash
#SBATCH --job-name=sbi_v8
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/sbi_v8_%j.out
#SBATCH --error=logs/sbi_v8_%j.err

set -eo pipefail
HISTCONTROL=ignoredups
export HISTCONTROL
source /etc/profile
module load anaconda3/2023.09-3
source activate luad_abm
pip install --quiet mesa h5py sbi torch 2>/dev/null || true

export PYTHONUNBUFFERED=1

cd ~/LungCancerSim2/gaglia_abm

echo "=== SBI v8 (19 params, +pd1_recruit_boost): $(date) ==="

python runs/bayesian_inference.py \
    --n-sims 2000 \
    --workers 15 \
    --out ../outputs/bayesian_inference_v8 \
    --data ../data/gaglia_2023/gaglia_summary_stats.csv \
    --n-posterior 10000

echo "=== Done: $(date) ==="
