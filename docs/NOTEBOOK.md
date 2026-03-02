# Lab Notebook — LungCancerSim2

## 2026-03-02: Functional markers + prior widening → v3 SBI

### Goal
Add CyCIF functional marker statistics (PD-1, TIM-3, GrzB, Ki67, B2m) to both the Gaglia extraction and ABM calibration pipelines, then widen priors for 6 ceiling-hitting parameters identified in v2.

### Functional marker extraction from Gaglia CyCIF data
Discovered that Dataset03 has 32 channels (not 28 as listed in Markers.csv) — an unlisted 8th cycle contains PD-L1 (ch29), B2m (ch30), pStat1 (ch31). Extracted from `Results_Norm_20210526.mat`:
- `MedianNucNorm` / `MedianCytNorm`: (9.7M cells, 32 channels) int16
- Positivity threshold: normalized intensity > 0

Added 5 per-mouse stats to `scripts/extract_gaglia_stats.py`:
- cd8_frac_pd1_pos, cd8_frac_exhausted (PD-1+ & TIM-3+), cd8_frac_grzb_pos, cd8_frac_ki67_pos, tumor_frac_b2m_pos
- Skipped PD-L1 (very low positivity: 0.3-1.5%, near noise floor)

Re-extracted CSV: `python3.11 scripts/extract_gaglia_stats.py --dataset data/gaglia_2023/Dataset03_KP_LucOS_anti_PD1_CTLA4 --out data/gaglia_2023/gaglia_summary_stats.csv`
→ 72 columns (was 67), 24 mice

### ABM functional marker mapping (calibration.py)
Mapped ABM agent continuous state → binary CyCIF positivity:
| CyCIF marker | ABM variable | Threshold | Rationale |
|---|---|---|---|
| PD-1+ | cd8.exhaustion > 0.4 | moderate exhaustion | PD-1 upregulates early |
| Exhausted (PD-1+TIM-3+) | cd8.exhaustion > 0.8 | terminal exhaustion | TIM-3 = late marker |
| GrzB+ | cd8.activation > 0.3 | active cytotoxicity | GrzB in effector CD8 |
| Ki67+ | cd8.activation > 0.1 | recently activated | Ki67 = broad proliferation |
| B2m+ | tumor.mhc_i > 0.8 | high MHC-I | B2m tracks surface MHC-I |

### IFNg → MHC-I pathway (agents.py)
Added to TumorAgent:
- `interactions_step`: IFNg induces MHC-I via `mhc_i_induction_rate` (new SBI param)
- `state_updates_step`: MHC-I decays via `mhc_i_decay_rate` (new SBI param, immune evasion)
- MHC-I floor: 0.1 (tumors always retain some antigen presentation)

### v2 posterior analysis — ceiling-hitting parameters
6 of 15 parameters had p95 > 90% of prior range:
| Parameter | v2 mean | v2 p95 | v2 ceiling | v3 ceiling |
|---|---|---|---|---|
| cd8_base_kill | 0.310 | 0.387 | 0.40 | **0.60** |
| cd8_activation_gain | 0.187 | 0.282 | 0.30 | **0.50** |
| tumor_proliferation_rate | 0.084 | 0.098 | 0.10 | **0.20** |
| recruitment_rate | 0.019 | 0.024 | 0.025 | **0.05** |
| treg_prolif_rate | 0.066 | 0.079 | 0.08 | **0.15** |
| mac_recruit_suppression | 3.620 | 4.846 | 5.0 | **8.0** |

CD8/CD4 ratio checked: v2 predicts 0.40 vs observed 0.42 — well-matched, no fix needed.

### v3 SBI configuration
- **17 parameters** (v2's 15 + mhc_i_induction_rate + mhc_i_decay_rate)
- **60-dim output** (15 stats × 4 conditions)
- 6 priors widened (see table above)
- SLURM job **2682447** submitted on Cayuga (16 CPUs, 64G, 48h, 6 workers, 2000 sims)
- Checkpoint: `outputs/bayesian_inference_v3/training_data_v3.npz`

### Files modified
- `gaglia_abm/runs/bayesian_inference.py` — widened 6 priors, added 2 MHC params, v3 checkpoint
- `gaglia_abm/runs/slurm_sbi.sh` — v3 job name and output dir
- `gaglia_abm/luad/agents.py` — IFNg → MHC-I induction + MHC-I decay on TumorAgent
- `gaglia_abm/luad/calibration.py` — 5 functional marker stats from ABM agent state
- `scripts/extract_gaglia_stats.py` — 5 functional marker stats from CyCIF NormResults
- `data/gaglia_2023/gaglia_summary_stats.csv` — re-extracted with 72 columns

---

## 2026-03-01: Temporal dynamics fix + 8-week holdout validation + v2 inference

### Goal
The v1 SBI posterior (10 params, 2 conditions) couldn't predict the 8-week held-out data because the model reached equilibrium too fast. Add mechanisms for progressive immune remodeling, validate against 8-week timepoints, and re-run inference with all 4 conditions jointly.

### v1 posterior results (completed ~00:12 EST)
SLURM job 2682125: 2000 sims, 120 ticks, 14 workers, ~3.3 hours on c0009.
SNPE converged after 113 epochs. Key posteriors:
- `cd8_base_kill`: 0.258 [0.138, 0.370]
- `tumor_proliferation_rate`: 0.062 [0.036, 0.079] — near prior ceiling (0.08)
- `immune_base_death_rate`: 0.014 [0.012, 0.015] — pushed to prior ceiling
- `recruitment_rate`: 0.012 [0.009, 0.015] — pushed to prior ceiling
- `pd_l1_penalty`: 0.452 [0.134, 0.822] — poorly constrained (wide CI)
- `cd8_exhaustion_rate`: 0.024 [0.007, 0.045] — well-constrained at low end

### Holdout validation failure (v1 model)
Ran posterior mean params at 192 ticks (8/5 × 120) vs observed 8-week data:

| Statistic | 8wk Obs | 8wk Sim (v1) | Error |
|-----------|---------|-------------|-------|
| frac_t_cytotox | 0.207 | 0.244 | 18% |
| frac_t_helper | 0.646 | 0.593 | 8% |
| frac_t_reg | 0.126 | 0.087 | 31% |
| frac_macrophage | 0.021 | 0.076 | 262% |
| ratio_cd8_cd4 | 0.323 | 0.412 | 28% |
| ratio_cd8_treg | 2.042 | 2.847 | 39% |

Root cause: no temporal dynamics — recruitment ≈ death at equilibrium by tick 120, so extending to 192 barely changes anything.

### Three mechanisms added

**1. CD8 exhaustion-scaled death + recruit priming** (`agents.py`, `model.py`)
- Death rate now continuous: `base + bonus × exhaustion` (not threshold-based)
- New recruits arrive with exhaustion primed by avg CD8 exhaustion in population
- `_avg_cd8_exhaustion` tracked each step; priming = `0.2 + priming_factor × avg_exh`
- Creates progressive CD8 decline: as population exhausts, recruits start weaker

**2. Treg tumor-proximity proliferation** (`agents.py`)
- Tregs divide with probability `treg_prolif_rate × local_tumor_density`
- Daughters inherit base suppression strength
- Tregs also more resistant to death (80% of base rate)
- Creates accumulation: more tumor → more Treg division → more suppression

**3. Macrophage tumor-driven attrition** (`agents.py`, `model.py`)
- Local death: `mac_tumor_death_rate × (tumor_density + M2_factor × 0.3)`
- Recruitment suppression: rate × `max(0.1, 1 - mac_recruit_suppression × tumor_expansion)`
- `tumor_expansion = (current_tumor - initial_tumor) / initial_tumor`
- Creates decline: expanding tumor reduces macrophage viability and recruitment

### Holdout validation (v2 model, old posterior params)

| Statistic | 8wk Obs | 8wk Sim (v2) | Error | Improvement |
|-----------|---------|-------------|-------|-------------|
| frac_t_cytotox | 0.207 | 0.212 | **2.5%** | 18% → 2.5% |
| frac_t_helper | 0.646 | 0.638 | **1.3%** | 8% → 1.3% |
| frac_t_reg | 0.126 | 0.115 | **9.0%** | 31% → 9.0% |
| frac_macrophage | 0.021 | 0.036 | **71%** | 262% → 71% |
| ratio_cd8_cd4 | 0.323 | 0.333 | **3.2%** | 28% → 3.2% |
| ratio_cd8_treg | 2.042 | 1.873 | **8.3%** | 39% → 8.3% |

All 6 statistics now move in the correct direction from 5wk→8wk. Five of six within 10% error even without re-inference. Macrophage still hardest (71%) — needs proper parameter fitting.

### v2 SBI inference (submitted)
SLURM job **2682157** on c0009 (16 CPUs, 32G, 12h limit). Changes:
- **15 parameters** (added `cd8_exhaustion_death_bonus`, `treg_prolif_rate`, `mac_tumor_death_rate`, `mac_recruit_suppression`, `recruit_exhaustion_priming`)
- Widened priors: `immune_base_death_rate` (→0.025), `recruitment_rate` (→0.025), `tumor_proliferation_rate` (→0.10)
- **4 conditions** jointly: 5wk control, 5wk treated, 8wk control, 8wk treated
- **40-dimensional** summary statistics (10 stats × 4 conditions)
- Different seeds for control vs treated (independent mice)
- 2000 sims, ~6-8 hours expected
- Checkpoint: `outputs/bayesian_inference_v2/training_data_v2.npz`

### New parameters and priors (v2)

| Parameter | Prior | Mechanism |
|-----------|-------|-----------|
| cd8_exhaustion_death_bonus | [0.005, 0.05] | CD8 death scales with exhaustion |
| treg_prolif_rate | [0.005, 0.08] | Treg division near tumor |
| mac_tumor_death_rate | [0.01, 0.15] | Macrophage death from tumor proximity |
| mac_recruit_suppression | [0.5, 5.0] | Tumor burden suppresses mac recruitment |
| recruit_exhaustion_priming | [0.05, 0.6] | CD8 recruits arrive pre-exhausted |

### Code review findings (from v1, partially addressed)
- ✅ Different seeds for control/treated (fixed in v2)
- ✅ Widened priors for ceiling-hitting parameters
- ⬜ Missing CD4 helper infiltration stats — should add in v3
- ⬜ No z-score normalization of summary stats before SNPE
- ⬜ PPC should use 50+ posterior samples, not just posterior mean
- ⬜ Pool recreated per batch — should create once

### Key questions / hypotheses

**Q1: Will the macrophage decline be captured by the 4-condition joint inference?**
With `mac_recruit_suppression` and `mac_tumor_death_rate` as free parameters, the inference should find values that reproduce the dramatic 5wk→8wk macrophage decline (0.069→0.021). Current default params give 71% error; proper fitting should improve this.

**Q2: Are 2000 simulations sufficient for 15 parameters × 40 dimensions?**
The dimensionality increased from 10→15 params and 20→40 stats. SNPE handles high dimensions better than ABC, but we may need 3000-5000 sims for good convergence. Monitor the training loss and posterior quality.

**Q3: Does the model correctly predict treatment × time interactions?**
The key prediction: PD1+CTLA4 treatment should slow both the CD8 decline and the Treg accumulation from 5wk→8wk. The observed data shows this: 8wk treated has higher CD8/Treg ratio (1.49) than 8wk control (2.04). The model should capture this via the PD1/CTLA4 intervention mechanics.

**Q4: Is the Treg proliferation mechanism identifiable from the data?**
Treg accumulation could also be explained by differential death rates or recruitment. The posterior on `treg_prolif_rate` will reveal whether the data specifically supports tumor-proximity proliferation vs alternative mechanisms.

**Q5: Should we add a time-varying suppressive background?**
The current `suppressive_background` is constant. In reality, the TME becomes more immunosuppressive over time. If the posterior pushes `suppressive_background` to its ceiling, consider making it tumor-burden-dependent.

### Output files
- `outputs/bayesian_inference/posterior_*.{png,csv,npy}` — v1 results (10 params, 2 conditions)
- `outputs/bayesian_inference/holdout_validation.png` — v2 model validation figure
- `outputs/bayesian_inference_v2/` — v2 results (pending, job 2682157)
- `scripts/validate_8wk_holdout.py` — holdout validation script

### v2 SBI parallelization saga (jobs 2682157 → 2682164 → 2682177 → 2682185)

Three failed attempts before finding stable parallelization for Mesa ABM on HPC:

1. **Job 2682157** — `multiprocessing.Pool` created/destroyed per batch → **deadlock** at 28/2000 sims. Pool teardown on HPC causes zombie processes.
2. **Job 2682164** — Switched to `ProcessPoolExecutor` (single persistent executor). Hit **OOM at 404/2000** (32GB, 14 workers). Mesa models leak memory in long-lived workers.
3. **Job 2682177** — Added `max_tasks_per_child=5`, 64GB, 10 workers. **Hung at 444/2000** — `as_completed` timeout was effectively infinite (1200s × batch_size).
4. **Job 2682185** — Nuclear option: `Pool(maxtasksperchild=1)` + fresh Pool per small batch + `gc.collect()` in worker. **6 workers, 64GB, 48h**. Ran 2000/2000 with zero failures at 2.2 sims/min (~12h).

**Lesson:** Mesa ABM agents accumulate memory that Python's allocator never returns to the OS. The only reliable strategy is `maxtasksperchild=1` — kill and restart the worker process after every single simulation.

### v2 SBI results (completed 2026-03-02 03:13 EST)

SLURM job **2682185**: 2000 sims (0 failed), SNPE converged after **114 epochs** (best validation: -28.01), 10,000 posterior samples.

**Well-constrained parameters** (posterior ≪ prior):

| Parameter | Posterior Mean | 95% CI | Prior | Interpretation |
|-----------|---------------|--------|-------|----------------|
| tumor_proliferation_rate | 0.084 ± 0.011 | [0.058, 0.099] | [0.005, 0.10] | Near upper bound — aggressive growth |
| recruitment_rate | 0.019 ± 0.003 | [0.014, 0.024] | [0.001, 0.025] | Near upper bound — strong immune influx |
| immune_base_death_rate | 0.010 ± 0.002 | [0.007, 0.014] | [0.001, 0.025] | Tightly constrained |
| treg_prolif_rate | 0.066 ± 0.010 | [0.041, 0.079] | [0.005, 0.08] | Near ceiling — aggressive Treg accumulation |
| cd8_exhaustion_death_bonus | 0.010 ± 0.003 | [0.005, 0.016] | [0.005, 0.05] | Near floor — exhaustion kills slowly |
| cd8_exhaustion_rate | 0.025 ± 0.013 | [0.006, 0.055] | [0.005, 0.15] | Low accumulation rate |
| cd8_base_kill | 0.310 ± 0.052 | [0.199, 0.394] | [0.02, 0.40] | High killing efficiency |
| cd8_activation_gain | 0.187 ± 0.062 | [0.058, 0.291] | [0.02, 0.30] | Moderate activation |
| suppressive_background | 0.086 ± 0.030 | [0.026, 0.142] | [0.01, 0.15] | Moderate background suppression |

**Poorly constrained** (posterior ≈ prior):

| Parameter | Posterior Mean | 95% CI | Prior |
|-----------|---------------|--------|-------|
| pd_l1_penalty | 0.489 ± 0.192 | [0.140, 0.852] | [0.1, 0.9] |
| recruit_exhaustion_priming | 0.287 ± 0.133 | [0.066, 0.552] | [0.05, 0.6] |

**Posterior predictive check** (posterior mean params):
- Cell fractions: generally well-matched across all 4 conditions
- CD8/CD4 and CD8/Treg ratios: systematic underestimate (observed ~3.5, predicted ~2.3 for 5wk control) — biggest mismatch
- Infiltration profiles: reasonable but imperfect
- 8wk predictions (held-out in v1): similar quality to 5wk — temporal mechanisms are working

**Answers to key questions:**
- **Q1 (macrophage decline):** `mac_recruit_suppression` = 3.62 [1.59, 4.92] — well-constrained, model captures macrophage decline via tumor-burden suppression of recruitment
- **Q2 (2000 sims sufficient?):** Yes for most parameters. 9/15 well-constrained. PD-L1 penalty and recruit exhaustion priming remain unidentifiable — would need functional marker data
- **Q4 (Treg proliferation identifiable?):** Yes — `treg_prolif_rate` tightly constrained at 0.066 near ceiling. Data specifically supports tumor-proximity proliferation
- **Q5 (time-varying suppression?):** `suppressive_background` at 0.086 [0.026, 0.142] — not hitting ceiling, so constant background is sufficient for now

### Output files (v2)
- `outputs/bayesian_inference_v2/posterior_marginals.png` — 15-panel posterior distributions
- `outputs/bayesian_inference_v2/posterior_predictive.png` — observed vs predicted (4 conditions, green = 8wk)
- `outputs/bayesian_inference_v2/posterior_summary.csv` — parameter estimates with 95% CIs
- `outputs/bayesian_inference_v2/posterior_samples.npy` — 10,000 × 15 posterior samples
- `outputs/bayesian_inference_v2/training_data_v2.npz` — 2000 × 15 (theta) + 2000 × 40 (x) training data

### Next steps
- Widen priors for ceiling-hitting params (`tumor_proliferation_rate`, `treg_prolif_rate`, `recruitment_rate`) and re-run
- Investigate CD8/CD4 ratio mismatch — may need separate recruitment rates for CD8 vs CD4
- Add z-score normalization of summary stats before SNPE (code review item)
- Improve PPC: run 50+ posterior samples instead of just posterior mean
- Consider adding CD4 helper infiltration stats to summary statistics

---

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
