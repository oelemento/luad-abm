#!/bin/bash
#SBATCH --job-name=sbi_v6_sim
#SBATCH --partition=scu-cpu
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/sbi_v6_sim_%A_%a.out
#SBATCH --error=logs/sbi_v6_sim_%A_%a.err

set -eo pipefail
source /etc/profile
module load anaconda3/2023.09-3
source activate luad_abm
pip install --quiet sbi mesa torch 2>/dev/null || true

cd ~/LungCancerSim2/

echo "=== SBI v5 chunk ${SLURM_ARRAY_TASK_ID}/10: $(date) ==="

python gaglia_abm/runs/sbi_worker.py \
    --chunk-id ${SLURM_ARRAY_TASK_ID} \
    --n-chunks 10 \
    --n-total 2000 \
    --workers 6 \
    --out-dir outputs/bayesian_inference_v6/chunks

echo "=== Chunk ${SLURM_ARRAY_TASK_ID} done: $(date) ==="
