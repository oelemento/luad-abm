#!/bin/bash
#SBATCH --job-name=abm_calibration
#SBATCH --partition=scu-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/calibration_%j.out
#SBATCH --error=logs/calibration_%j.err

set -eo pipefail

# Fix HISTCONTROL issue with /etc/profile
HISTCONTROL=ignoredups
export HISTCONTROL
source /etc/profile

module load anaconda3/2023.09-3
source activate luad_abm

cd ~/LungCancerSim2

python scripts/calibration_search.py \
    --obs data/gaglia_2023/gaglia_summary_stats.csv \
    --preset luad_abm/config/G3_inflammatory.json \
    --n-samples 50 --seeds 1 --workers 8 \
    --out data/calibration_results_baseline.csv

echo "Calibration complete"
