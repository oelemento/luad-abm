#!/bin/bash
#SBATCH --job-name=gaglia_sbi
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gaglia_sbi_%j.out
#SBATCH --error=logs/gaglia_sbi_%j.err

set -eo pipefail

HISTCONTROL=ignoredups
export HISTCONTROL
source /etc/profile

module load anaconda3/2023.09-3
source activate luad_abm

# Install missing dependencies (idempotent)
pip install --quiet sbi mesa torch 2>/dev/null || true

cd ~/LungCancerSim2/

echo "=== Starting Bayesian inference: $(date) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"

python gaglia_abm/runs/bayesian_inference.py \
    --n-sims 2000 \
    --ticks 120 \
    --workers 14 \
    --out outputs/bayesian_inference \
    --data data/gaglia_2023/gaglia_summary_stats.csv \
    --n-posterior 10000

echo "=== Completed: $(date) ==="
