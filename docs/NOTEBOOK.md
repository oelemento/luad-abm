# Lab Notebook — LungCancerSim2

## 2026-02-28: Bayesian calibration pipeline (Sanofi proposal)

### Goal
Build proof-of-concept calibration pipeline: infer ABM parameters from Gaglia et al. mouse CyCIF data. Produce 4-panel figure for Sanofi iDEA-TECH full proposal.

### What was done

**1. Extracted Gaglia summary statistics**
```bash
python3.11 scripts/extract_gaglia_stats.py \
    --dataset data/gaglia_2023/Dataset03_KP_LucOS_anti_PD1_CTLA4 \
    --out data/gaglia_2023/gaglia_summary_stats.csv
```
- 24 mice, 4 groups of 6 (Ctrl, anti-PD1, anti-CTLA4, Combo)
- 59 columns: cell fractions, infiltration profiles, CD8 distance histograms, lymphonet stats
- Key debugging: .mat files use hierarchical integer codes for cell types (1111=Treg etc.), DistResults.Tumor has 3 columns (need col 2 for signed distance)

**2. Created ABM calibration module**
- `luad_abm/luad/calibration.py`: `extract_summary_stats()` and `compute_distance()`
- Extracts cell fractions, infiltration profiles (core/cuff/periphery), CD8-tumor distances
- Variance-weighted distance metric: d = Σ (obs_i - sim_i)² / var_i

**3. Created parameter search script**
- `scripts/calibration_search.py`: Latin Hypercube Sampling over 6 parameters
- Parameters: cd8_base_kill, cd8_exhaustion_rate, pd_l1_penalty, macrophage_suppr_base, suppressive_background, tumor_proliferation_rate
- Uses ProcessPoolExecutor for parallel simulation runs

**4. Ran calibration on Cayuga cluster**
```bash
# On Cayuga (scu-cpu, 8 CPUs, 32G RAM)
python scripts/calibration_search.py \
    --obs data/gaglia_2023/gaglia_summary_stats.csv \
    --preset luad_abm/config/G3_inflammatory.json \
    --n-samples 200 --ticks 200 --seeds 2 --workers 8 \
    --out data/calibration_results_baseline.csv
```
- 200 LHS samples × 2 seeds = 400 simulations
- Completed in ~45 min on Cayuga node c0001

**5. Generated proposal figure**
```bash
python3.11 scripts/plot_calibration_results.py \
    --obs data/gaglia_2023/gaglia_summary_stats.csv \
    --results data/calibration_results_baseline.csv \
    --out luad_abm/summary_plots/calibration_figure.png
```

### Critical fix: immune-only fraction normalization

First run (500 ticks) used all-cell denominators → massive mismatch (observed ~1% vs simulated ~20% for CD8+ T). The Gaglia tissue is ~90%+ tumor cells, so immune fractions computed over all cells are tiny.

**Fix:** Changed both `extract_gaglia_stats.py` and `calibration.py` to compute fractions among immune/non-tumor cells only. Distance score dropped from 856 → 92 (10x improvement).

### Results

Best fit parameters (distance = 91.86):
- cd8_base_kill: 0.297
- cd8_exhaustion_rate: 0.022
- pd_l1_penalty: 0.541
- macrophage_suppr_base: 0.190
- suppressive_background: 0.075
- tumor_proliferation_rate: 0.015

Remaining systematic offset: ABM still overestimates immune fractions ~2-6x because it lacks some stromal cell types (epithelial, endothelial) present in real tissue that dilute the immune-only denominator.

### Output files
- `data/gaglia_2023/gaglia_summary_stats.csv` — 24 mice × 59 columns
- `data/calibration_results_baseline.csv` — 400 simulations with distance scores
- `luad_abm/summary_plots/calibration_figure.png` — 4-panel proposal figure
