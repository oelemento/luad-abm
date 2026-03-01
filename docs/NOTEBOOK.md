# Lab Notebook — LungCancerSim2

## 2026-02-28: Clean Gaglia ABM rebuild + Bayesian inference

### Goal
Rebuild ABM from scratch to match only what Gaglia et al. 2023 (Cancer Cell 41, 871-886) actually measures. Then use simulation-based Bayesian inference (SNPE) to extract posterior distributions over mechanistic parameters from CyCIF data.

### What was done

**1. Built clean `gaglia_abm/` codebase**
- 5 agent types only: Tumor, CD8, CD4, Treg, Macrophage
- Removed all components not in Gaglia data: CAFs, TLS, ECM, CXCL13, TGF-beta, EMT, VISTA/CSF1R/TIM3/TLS_BOOST/antiTGFB interventions
- 2 fields: cxcl9_10 (diffusive chemokine), ifng_signal (fast-decaying)
- 2 interventions: PD1, CTLA4

**2. Fixed flat immune cell trajectories**
- Root cause: no death or recruitment mechanics for immune cells
- Added `remove_immune_agent()` in `rules.py`
- Death mechanics per cell type:
  - CD8: base rate + bonus when exhaustion > threshold (0.85)
  - CD4/Treg: base rate (0.005/tick)
  - Macrophage: 70% of base rate (longer-lived)
- Recruitment: Poisson-driven from periphery, boosted by avg chemokine level
- Periphery-biased placement (acceptance ∝ distance^1.5 from center)

**3. Treatment comparison runs**
```bash
# Baseline (no treatment)
python3.11 gaglia_abm/runs/run.py --preset gaglia_abm/config/gaglia_baseline.json --ticks 169 --out outputs/gaglia_turnover_test --movie-scale 6
# Anti-PD1
python3.11 gaglia_abm/runs/run.py --preset gaglia_abm/config/gaglia_baseline.json --ticks 169 --out outputs/gaglia_pd1_test --movie-scale 6 --add PD1
# Anti-PD1 + Anti-CTLA4
python3.11 gaglia_abm/runs/run.py --preset gaglia_abm/config/gaglia_baseline.json --ticks 169 --out outputs/gaglia_pd1_ctla4_test --movie-scale 6 --add PD1 --add CTLA4
```
- Results: Tumor at tick 169: Baseline ~1280, PD1 ~1051, PD1+CTLA4 ~1070
- CD8 decline with exhaustion, CD4 increases via recruitment, treatment preserves CD8 better

**4. Built SBI Bayesian inference pipeline**
- `gaglia_abm/runs/bayesian_inference.py` (~370 lines)
- 10 mechanistic parameters with uniform priors (cd8_base_kill, pd_l1_penalty, cd8_exhaustion_rate, cd8_activation_gain, tumor_proliferation_rate, macrophage_suppr_base, suppressive_background, immune_base_death_rate, recruitment_rate, treg_suppression)
- 20-dimensional summary statistics: 10 stats × 2 conditions (control + PD1+CTLA4 treated)
- Joint inference: same biological params must explain both conditions
- SNPE (Sequential Neural Posterior Estimation) via `sbi` package
- Parallel simulation with checkpointing to .npz
- Tested locally with 20 sims — pipeline works end-to-end

**5. Submitted full inference on Cayuga HPC**
```bash
# SLURM job 2682125 on scu-cpu, 16 CPUs, 32G RAM, 6h limit
python gaglia_abm/runs/bayesian_inference.py \
    --n-sims 2000 --ticks 120 --workers 14 \
    --out outputs/bayesian_inference \
    --data data/gaglia_2023/gaglia_summary_stats.csv \
    --n-posterior 10000
```
- Running on node c0009 at ~0.2 sims/min
- As of last check: 672/2000 simulations complete, ETA ~2 hours
- Outputs will be: posterior_marginals.png, posterior_predictive.png, posterior_summary.csv, posterior_samples.npy

**6. Code review findings (important issues to fix for next run)**
- `treg_suppression` flows through PresetConfig path (not `model.params`) — fragile but correct
- Missing CD4 helper infiltration stats in summary statistics
- Posterior predictive check should use 50+ samples, not just posterior mean
- Should add z-score normalization of summary stats before SNPE training
- Same seed for control/treated may bias posterior (real mice are independent)
- Pool recreated per batch — should create once

### Key files
- `gaglia_abm/luad/agents.py` — 5 agent types with death mechanics
- `gaglia_abm/luad/model.py` — Mesa model with turnover/recruitment
- `gaglia_abm/luad/rules.py` — Movement, killing, immune removal
- `gaglia_abm/luad/calibration.py` — Summary statistic extraction
- `gaglia_abm/runs/bayesian_inference.py` — Full SNPE pipeline
- `gaglia_abm/runs/slurm_sbi.sh` — SLURM submission script
- `gaglia_abm/config/gaglia_baseline.json` — Baseline preset (5 cell types)
- `scripts/compare_baseline_pd1.py` — 3-way treatment comparison plot
- `data/gaglia_2023/gaglia_summary_stats.csv` — Observed CyCIF data (24 mice, 4 groups)

### Next steps
- Fetch and analyze posterior results from Cayuga
- Fix code review issues (add CD4 infiltration stats, normalize summary stats, improve PPC)
- Validate on held-out groups (8-week timepoints: Groups 3 & 4)
- Run posterior mean parameters for longer simulations to check temporal predictions

---

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
