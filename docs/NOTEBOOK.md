# Lab Notebook — LungCancerSim2

## Hypotheses

| ID | Hypothesis | Status | Figure | Script |
|----|-----------|--------|--------|--------|
| H1 | Earlier treatment timing improves tumor control in KP mice | Supported (marginal) | `outputs/treatment_timing_sweep/treatment_timing_sweep.png` | `scripts/treatment_timing_sweep.py` |
| H2 | CTLA4→PD1 sequential dosing outperforms PD1→CTLA4 (reducing suppression before boosting killing is more effective) | **Not supported** — CTLA4→PD1 (−1.6%) worse than PD1→CTLA4 (−4.0%) | `outputs/sequential_dosing/sequential_dosing.png` | `scripts/sequential_dosing_sweep.py` |
| H3 | CTLA4 with ADCC-mediated Treg depletion outperforms suppression-only CTLA4 | **Not supported** — ADCC depletes Tregs (255 vs 428) but tumor kill negligible (+0.3% vs −0.5%) | same as H2 | same as H2 |
| H4 | Anti-PD1/IL-15 fusion (SAR445877) outperforms PD1+CTLA4 combo by expanding CD8 pool in situ | **Supported** — PD1/IL-15 (−13.6%) >> PD1+CTLA4 combo (−4.1%), CD8 count 1248 vs 1133 | same as H2 | same as H2 |
| H5 | Triple combination (PD1/IL-15 + CTLA4-ADCC) produces synergistic tumor control | **Not supported** (additive, not synergistic) — −14.6% vs −13.6% for PD1/IL-15 alone | same as H2 | same as H2 |
| H6 | Human LUAD tumors initialized from patient CyCIF data respond better to ICB than KP mice, due to more favorable CD8:Treg ratio (4.49 vs 2.65) and higher immune activation potential | **Partially supported** — immune-hot patients (CD8:Treg > 5) respond dramatically better; immune-cold patients (CD8:Treg < 2) respond worse. CD8:Treg ratio predicts response. | `outputs/human_luad_sweep_v2/human_luad_sweep.png` | `scripts/human_luad_sweep.py` |
| H7 | Antigen-driven CD8 clonal expansion (kill-triggered proliferation) fixes the ABM's underestimation of response in high-ratio/low-CD8 patients | **Not supported** — SBI v7 converged (kp=0.085) but 2D biomarker discrepancy persists: high-ratio/low-CD8 still 0% vs 88% clinical | `outputs/human_luad_sweep_v3/v2_v3_sorin_2d_comparison.png` | `scripts/compare_v2_v3_2d_biomarker.py` |
| H8 | PD1-driven suppression-modulated CD8 recruitment boost fixes high-ratio/low-CD8 discrepancy | Pending — SBI v8 complete (pd1_recruit_boost=1.82), v4 sweep running | `outputs/human_luad_sweep_v4/` | `scripts/human_luad_sweep.py` (with v8 posterior) |

---

## 2026-03-12: SBI v8 — Joint inference with pd1_recruit_boost (19 params)

### Motivation

H7 showed that kill-triggered proliferation alone cannot fix the high-ratio/low-CD8 discrepancy. Both Claude and Codex (consulted independently) agreed: the fundamental problem is that the ABM's PD1 blockade only enhances per-contact killing. Clinical responses in low-CD8 patients require **new CD8 influx** from extratumoral reservoirs, which is kill-independent and therapy-triggered (Yost 2019, Siddiqui 2019, Tumeh 2014).

Key design: the recruitment boost is **damped by local Treg/M2 suppression**. This preserves quadrant structure:
- High ratio + low CD8 → low suppression → full boost → should respond
- Low ratio + low CD8 → high suppression → dampened boost → shouldn't respond as well

### Results

SBI v8 completed (2000 sims, 26.5h runtime, job 2702064). Posterior summary:

| Parameter | v8 Mean | v8 95% CI | v7 Mean | Prior |
|-----------|---------|-----------|---------|-------|
| cd8_base_kill | 0.573 | [0.315, 0.826] | 0.653 | [0.02, 0.90] |
| cd8_exhaustion_rate | 0.031 | [0.004, 0.063] | 0.035 | [0.001, 0.15] |
| cd8_activation_gain | 0.387 | [0.070, 0.721] | 0.361 | [0.02, 0.80] |
| tumor_proliferation_rate | 0.250 | [0.143, 0.340] | 0.198 | [0.005, 0.35] |
| macrophage_suppr_base | 0.263 | [0.070, 0.469] | 0.282 | [0.05, 0.50] |
| suppressive_background | 0.078 | [0.017, 0.141] | 0.077 | [0.01, 0.15] |
| immune_base_death_rate | 0.002 | [0.001, 0.005] | 0.004 | [0.0005, 0.025] |
| recruitment_rate | 0.021 | [0.008, 0.035] | 0.029 | [0.001, 0.05] |
| treg_suppression | 0.267 | [0.101, 0.431] | 0.272 | [0.08, 0.45] |
| cd8_exhaustion_death_bonus | 0.007 | [0.002, 0.011] | 0.010 | [0.001, 0.05] |
| treg_prolif_rate | 0.093 | [0.027, 0.145] | 0.083 | [0.005, 0.15] |
| treg_death_rate | 0.004 | [0.001, 0.007] | 0.004 | [0.001, 0.03] |
| mac_tumor_death_rate | 0.078 | [0.017, 0.142] | 0.080 | [0.01, 0.15] |
| mac_recruit_suppression | 3.999 | [0.816, 7.519] | 4.771 | [0.5, 8.0] |
| recruit_exhaustion_priming | 0.234 | [0.088, 0.383] | 0.276 | [0.05, 0.60] |
| mhc_i_induction_rate | 0.054 | [0.011, 0.095] | 0.056 | [0.005, 0.10] |
| mhc_i_decay_rate | 0.004 | [0.001, 0.007] | 0.004 | [0.0005, 0.01] |
| cd8_kill_prolif_prob | 0.076 | [0.011, 0.139] | 0.085 | [0.0, 0.15] |
| **pd1_recruit_boost** | **1.821** | **[0.193, 3.884]** | — | [0.0, 10.0] |

### Key findings

1. **`pd1_recruit_boost = 1.82 [0.19, 3.88]`** — well-identified, non-zero. During PD1 blockade in a low-suppression environment, CD8 recruitment rate is ~2.8× baseline (1 + 1.82 × (1 - suppression)). In high-suppression environments, the boost is dampened.

2. **`recruitment_rate` dropped from 0.029 → 0.021** — the model compensated for the new PD1 boost by lowering baseline recruitment, maintaining calibration to untreated controls.

3. **`tumor_proliferation_rate` increased from 0.198 → 0.250** — with stronger immune response during treatment, the model needs faster tumor growth to match untreated tumor counts.

4. **`cd8_base_kill` dropped from 0.653 → 0.573** — with more CD8s arriving during treatment (via recruitment boost), each individual kill doesn't need to be as efficient.

5. **`cd8_kill_prolif_prob` stable at 0.076** (was 0.085 in v7) — both mechanisms coexist.

Files: `outputs/bayesian_inference_v8/`, `scripts/slurm_sbi_v8.sh`

### Next step

LUAD v4 sweep running (job 2702376) with v8 posterior. Will test whether the suppression-modulated recruitment boost fixes the 2D biomarker discrepancy.

---

## 2026-03-11: H7 result — Kill-prolif alone does NOT fix 2D biomarker discrepancy

### Human LUAD sweep v3 (SBI v7 posterior, 18 params incl cd8_kill_prolif_prob=0.085)

v3 sweep completed (job 2701853, 11h38m). Key results vs v2:

| Config | v2 ΔTumor% (PD1+CTLA4) | v3 ΔTumor% (PD1+CTLA4) | Change |
|--------|----------------------|----------------------|--------|
| KP_mouse | −4.1% | −7.2% | improved |
| CASE7 (ratio 5.97) | −45.0% | −24.6% | weaker (lower baseline tumor) |
| CASE9 (ratio 4.66) | −22.2% | −17.8% | similar |
| CASE12 (ratio 3.11) | −2.3% | −2.6% | unchanged |
| CASE18 (ratio 2.24) | −1.8% | −2.1% | unchanged |
| CASE14 (ratio 2.04) | — (cleared) | −61.4% | new responder |

### 2D biomarker comparison (v2 vs v3 vs Sorin clinical)

| Quadrant | v2 ABM | v3 ABM | Sorin clinical |
|----------|--------|--------|---------------|
| High ratio + High CD8 | 3/4 (75%) | 3/4 (75%) | 15/21 (71%) |
| **High ratio + Low CD8** | **0/2 (0%)** | **0/2 (0%)** | **7/8 (88%)** |
| Low ratio + High CD8 | 0/0 (N/A) | 1/1 (100%) | 2/8 (25%) |
| Low ratio + Low CD8 | 0/5 (0%) | 0/5 (0%) | 12/21 (57%) |

**Conclusion**: Kill-triggered proliferation (p=0.085) does not fix the high-ratio/low-CD8 discrepancy. The mechanism has the same bootstrap failure as the original model — patients with few CD8s rarely achieve enough initial kills for proliferation to compound. The low-ratio/low-CD8 quadrant (0% vs 57%) also remains discrepant.

### Diagnosis (consensus from Claude + Codex)

The fundamental problem: the ABM's PD1 blockade only enhances per-contact killing. But clinical responses in low-CD8 patients require **new CD8 influx** from extratumoral reservoirs (lymph nodes, TLS), which is kill-independent and therapy-triggered (Yost 2019, Siddiqui 2019, Tumeh 2014). Both kill-prolif and kill-driven recruitment fail because they require the first kills to happen.

**Proposed fix (H8)**: PD1-driven suppression-modulated CD8 recruitment boost:
- During PD1 blockade, multiply `recruitment_rate` by a therapy-induced factor
- Crucially, dampen the boost by local Treg/M2 suppression — this preserves quadrant structure:
  - High ratio + low CD8 → low suppression → full recruitment boost → responds
  - Low ratio + low CD8 → high suppression → dampened boost → doesn't respond as well
- SBI-calibrate the boost magnitude and decay as new parameters

Figure: `outputs/human_luad_sweep_v3/v2_v3_sorin_2d_comparison.png`

---

## 2026-03-07: H7 — Antigen-driven CD8 clonal expansion (kill-triggered proliferation)

### Motivation

2D biomarker analysis revealed a key discrepancy between ABM predictions and clinical data (Sorin et al. 2025): the ABM predicts that high-ratio/low-CD8 patients don't respond (0/1), but clinical data shows they respond best (88%, 7/8). The ABM overweights CD8 quantity because proliferation is density-dependent (requires empty neighbor), not antigen-driven. In reality, even a few unsuppressed CD8s can expand rapidly upon TCR engagement after PD1 blockade.

### Model change

Added `cd8_kill_prolif_prob` parameter to `rules.kill_target()`: when a CD8 successfully kills a tumor cell, it has probability `cd8_kill_prolif_prob` of dividing into the cleared grid square. The daughter inherits the parent's activation level and 50% of its exhaustion (asymmetric division favoring effector phenotype).

### Literature basis for p=0.5

Upon TCR engagement, a single naive CD8 T cell produces ~10,000-100,000 effector cells over 7-10 days (Blattman et al., J Exp Med 2002; Badovinac et al., Nat Immunol 2002), corresponding to ~13-17 doublings in ~170-240h, or one division per ~12-18h. At 24 ticks/week (~7h/tick), this gives p≈0.3-0.5 per tick. We use p=0.5 as a literature-informed upper estimate for effector CD8s in an active anti-tumor response. This is NOT SBI-fitted — it is a fixed parameter based on published clonal expansion kinetics.

### Experiment

Same design as H6 sweep: 14 human LUAD patients + KP baseline × 3 arms (untreated, PD1+CTLA4, PD1/IL-15) × 20 seeds = 900 sims. Only difference: `--kill-prolif 0.5`. Running on Cayuga (16 CPUs, 15 workers).

Files modified: `luad_abm/luad/rules.py` (kill_target), `luad_abm/luad/model.py` (default_params), `scripts/human_luad_sweep.py` (--kill-prolif arg), `scripts/slurm_human_luad_killprolif.sh` (new SLURM script)

---

## 2026-03-07: Validation of CD8:Treg ratio as ICB biomarker (Carvajal-Hausdorf et al. 2025)

### Data source

Carvajal-Hausdorf et al., Science Advances 2025 — mIF data from Dryad (doi:10.5061/dryad.b8gtht7p4). 1.45M cells, 33 biomarkers, 132 patients (22 with IO + mIF data: 7 responders, 15 non-responders). Cell phenotypes pre-classified: Tc (CD8), Treg, Th, Monocyte, Macrophage, Tumor, etc.

### Results

| Metric | Responders (median) | Non-responders (median) | p-value (Mann-Whitney) |
|--------|-------------------|------------------------|----------------------|
| CD8:Treg ratio | 3.59 | 2.06 | 0.490 |
| CD8 count | 1453 | 799 | 0.448 |
| Treg count | 649 | 489 | 0.680 |
| **CD8 fraction** | **0.06** | **0.03** | **0.004** |
| Treg fraction | 0.03 | 0.01 | 0.142 |

### Key finding

**CD8:Treg ratio is NOT significantly associated with IO response** (p=0.49) in this cohort, but **CD8 fraction is** (p=0.004). This supports our ABM observation that it's a 2D problem — both absolute CD8 density and ratio matter, with density being the stronger predictor. High-ratio patients with low CD8 fraction don't respond (e.g., P_51: ratio 7.88, CD8 frac 2%, NR). Low-ratio patients with high CD8 fraction can respond (e.g., P_45: ratio 0.78, CD8 frac 6.3%, R).

### Caveats

- Small sample (n=22 with IO + mIF data, only 7 responders) — very low statistical power for ratio
- mIF from TMA cores (small tissue area) may not represent full tumor composition
- CD8 fraction significance (p=0.004) should be validated in larger cohort

---

## 2026-03-07: Validation with Sorin et al. 2025 DSP-GeoMx data

### Data source

Sorin et al. 2025 — DSP-GeoMx spatial transcriptomics of NSCLC tumors treated with checkpoint inhibitors. Data from authors' GitHub repo (tznaung/NSCLC_SpatialOmics). Cell type proportions pre-computed via deconvolution in two compartments: Stroma (CD45+) and Tumor (CK+). 58 unique patients in stroma (36 R, 22 NR), 56 in tumor (34 R, 22 NR). Cell types include CD8_cells, Treg, Exhausted CD8, Cytotoxic CD8, CD4, M1/M2 macrophages, B cells, DCs.

### Results

**Stroma compartment (n=58):**

| Metric | Responders (median) | Non-responders (median) | p-value (Mann-Whitney) |
|--------|-------------------|------------------------|----------------------|
| **CD8:Treg ratio** | **3.29** | **1.89** | **0.012** |
| CD8 fraction | 0.040 | 0.043 | 0.659 |
| **Treg fraction** | **0.011** | **0.025** | **0.001** |

**Tumor (CK) compartment (n=56):**

| Metric | Responders (median) | Non-responders (median) | p-value (Mann-Whitney) |
|--------|-------------------|------------------------|----------------------|
| CD8:Treg ratio | 3.12 | 1.89 | 0.596 |
| CD8 fraction | 0.006 | 0.006 | 0.620 |
| **Treg fraction** | **0.001** | **0.005** | **0.012** |

### Key finding

**Stromal CD8:Treg ratio IS significantly associated with IO response** (p=0.012), validating the ABM prediction. However, the effect is driven primarily by **Treg depletion** (p=0.001), not CD8 enrichment (p=0.66). In the tumor compartment, only Treg fraction is significant — consistent with the stroma result but weaker signal due to sparse immune infiltration in tumor nests.

### Synthesis across validation cohorts

| Cohort | N (IO+data) | CD8:Treg ratio | CD8 fraction | Treg fraction |
|--------|------------|---------------|-------------|--------------|
| Carvajal-Hausdorf 2025 (mIF) | 22 | p=0.49 (NS) | **p=0.004** | p=0.14 (NS) |
| Sorin 2025 (DSP stroma) | 58 | **p=0.012** | p=0.66 (NS) | **p=0.001** |
| Sorin 2025 (DSP tumor) | 56 | p=0.60 (NS) | p=0.62 (NS) | **p=0.012** |

Different cohorts highlight different aspects of the same underlying biology: the CD8-Treg immunosuppressive balance determines ICB response, but the dominant signal depends on the measurement platform (mIF whole-cell vs DSP compartment-level) and tumor architecture. Treg fraction is the most consistently significant marker across datasets.

Figure: `outputs/human_luad_sweep_v2/sorin_validation.png`

### 2D biomarker analysis: ABM vs Sorin clinical data

The ABM predicts response is a 2D function of CD8:Treg ratio AND absolute CD8 fraction. We tested this by splitting patients into quadrants (above/below median for each axis):

| Quadrant | ABM (PD1+CTLA4) | Sorin (stroma) |
|----------|-----------------|----------------|
| High ratio + High CD8 | 5/5 respond (100%) | 15/21 (71%) |
| **High ratio + Low CD8** | **0/1 (0%)** | **7/8 (88%)** |
| Low ratio + High CD8 | 0/1 (0%) | 2/8 (25%) |
| Low ratio + Low CD8 | 0/5 (0%) | 12/21 (57%) |

**Key discrepancy**: The ABM predicts that high ratio + low CD8 = non-responder (you need enough CD8 killers), but clinical data shows this is the **best** quadrant (88%). Clinically, even a small number of unsuppressed CD8s can mount effective anti-tumor responses through antigen-driven clonal expansion — a mechanism the ABM underrepresents.

**What the ABM gets right**: Low ratio + high CD8 = worst outcome (25% in Sorin, 0% in ABM). Lots of CD8 cells that are heavily suppressed by Tregs cannot respond to ICB.

**What the ABM gets wrong**: It overweights CD8 quantity vs CD8 functionality. In the current model, CD8 proliferation is density-dependent (requires empty neighbor), not antigen-driven. A few functional CD8s in low-Treg environments can expand rapidly in vivo upon PD1 blockade, but the ABM doesn't capture this positive feedback.

**Potential model fix**: Add antigen-driven CD8 proliferation — CD8s that successfully kill a tumor cell get a proliferation bonus (binary fission into the cleared grid square). This would make small numbers of unsuppressed CD8s more effective, consistent with the clinical observation. The current model has CD8 proliferation controlled by `cd8_prolif_rate` (SBI-fitted ~0.04/tick) gated on empty neighbors. An antigen-driven term could add a second proliferation pathway: `if killed_tumor: spawn_daughter_in_cleared_spot()`. This is biologically motivated by clonal expansion upon TCR engagement.

Figure: `outputs/human_luad_sweep_v2/sorin_2d_biomarker.png`

---

## 2026-03-10: SBI v7 — Joint inference with cd8_kill_prolif_prob (18 params)

### Motivation

The kill-prolif sweep (kp=0.0 to 0.3) showed that bolting on any non-zero `cd8_kill_prolif_prob` without recalibrating all other parameters causes universal tumor clearance. Even kp=0.05 dropped untreated KP_mouse tumor from 1682→525. The parameter interacts strongly with `cd8_base_kill`, `tumor_proliferation_rate`, and other dynamics — it cannot be tuned independently. Therefore we added it as the 18th SBI-inferred parameter with prior [0, 0.15].

### Results

SBI v7 completed (2000 sims, 26h runtime, job 2701060). SNPE inference converged. Posterior summary:

| Parameter | Mean | 95% CI | Prior |
|-----------|------|--------|-------|
| cd8_base_kill | 0.653 | [0.356, 0.877] | [0.02, 0.90] |
| cd8_exhaustion_rate | 0.035 | [0.004, 0.072] | [0.001, 0.15] |
| cd8_activation_gain | 0.361 | [0.049, 0.722] | [0.02, 0.80] |
| tumor_proliferation_rate | 0.198 | [0.125, 0.279] | [0.005, 0.35] |
| macrophage_suppr_base | 0.282 | [0.083, 0.475] | [0.05, 0.50] |
| suppressive_background | 0.077 | [0.017, 0.140] | [0.01, 0.15] |
| immune_base_death_rate | 0.004 | [0.002, 0.007] | [0.0005, 0.025] |
| recruitment_rate | 0.029 | [0.016, 0.043] | [0.001, 0.05] |
| treg_suppression | 0.272 | [0.105, 0.429] | [0.08, 0.45] |
| cd8_exhaustion_death_bonus | 0.010 | [0.007, 0.013] | [0.001, 0.05] |
| treg_prolif_rate | 0.083 | [0.018, 0.143] | [0.005, 0.15] |
| treg_death_rate | 0.004 | [0.001, 0.009] | [0.001, 0.03] |
| mac_tumor_death_rate | 0.080 | [0.017, 0.143] | [0.01, 0.15] |
| mac_recruit_suppression | 4.771 | [1.345, 7.700] | [0.5, 8.0] |
| recruit_exhaustion_priming | 0.276 | [0.178, 0.364] | [0.05, 0.60] |
| mhc_i_induction_rate | 0.056 | [0.013, 0.095] | [0.005, 0.10] |
| mhc_i_decay_rate | 0.004 | [0.001, 0.007] | [0.0005, 0.01] |
| **cd8_kill_prolif_prob** | **0.085** | **[0.020, 0.143]** | [0.0, 0.15] |

### Key finding

**`cd8_kill_prolif_prob` converged to 0.085 [0.020, 0.143]** — the Gaglia data supports ~8.5% probability of CD8 clonal expansion per kill. This is well within the prior and far from edges, suggesting a well-identified parameter. The other 17 parameters remained stable compared to v6, indicating the new parameter was absorbed without destabilizing the existing calibration.

Files: `outputs/bayesian_inference_v7/`, `scripts/slurm_sbi_v7.sh`

### Next step

Re-run human LUAD sweep with v7 posterior (all 18 params jointly calibrated) to test whether the 2D biomarker prediction now matches Sorin clinical data. Running as `outputs/human_luad_sweep_v3/`.

---

## 2026-03-07: Applied for SU2C-MARK NSCLC data (dbGaP phs002822)

Applied for controlled access to the SU2C-MARK NSCLC cohort (Jung et al., Nature Genetics 2023) via dbGaP (project #43102). This dataset contains 393 advanced NSCLC patients treated with checkpoint inhibitors, with 152 having RNA-seq + RECIST response data. Plan: run CIBERSORTx immune deconvolution to estimate CD8 and Treg fractions, then validate the CD8:Treg ratio as a predictor of ICB response — testing the quantitative thresholds predicted by our ABM (H6 results).

---

## 2026-03-06: Human LUAD immunotherapy sweep (H6)

### Motivation

KP mouse tumors show modest ICB response (−4% PD1+CTLA4, −14% PD1/IL-15). Human LUAD tumors have different immune compositions — potentially more favorable CD8:Treg ratios due to higher neoantigen burden. Using CyCIF spatial phenotyping data from Gaglia 2023 Dataset05 (14 human LUAD patients, 7.8M cells), we initialized the ABM with real patient cell compositions while keeping KP-calibrated kinetic parameters (v6 posterior mean). This tests whether tumor microenvironment composition alone predicts ICB response.

### Method

- **Data source**: Gaglia 2023 Dataset05, CyCIF quantification (Results_Aggr_20210619.mat)
- **Cell phenotyping**: Tumor (Keratin+/TTF1+), CD8 (CD3D+CD8a+), Treg (CD3D+CD4+FOXP3+), Macrophage (CD68+/CD163+), using 75th percentile intensity threshold
- **Grid mapping**: Patient fractions scaled to fill ~80% of 100×100 grid, preserving relative composition
- **Arms**: Untreated, PD1+CTLA4, PD1/IL-15 (1-week pulse at week 7, measured week 9)
- **Configs**: KP_mouse baseline + 14 human patients × 3 arms × 20 seeds = 900 sims
- **Compute**: Cayuga HPC, 16 CPUs (15 parallel workers), 11h 21m runtime (job 2700333)

### Results (20 seeds per condition)

| Config | Arm | Tumor (mean±sd) | CD8 | Treg | CD8:Treg | ΔTumor% |
|--------|-----|-----------------|-----|------|----------|---------|
| KP_mouse | untreated | 1516 ± 675 | 1135 | 428 | 2.66 | +0.0% |
| KP_mouse | PD1_CTLA4 | 1453 ± 665 | 1133 | 441 | 2.58 | −4.1% |
| KP_mouse | PD1_IL15 | 1309 ± 606 | 1248 | 427 | 2.93 | −13.6% |
| CASE1 | untreated | 4408 ± 80 | 1387 | 787 | 1.77 | +0.0% |
| CASE1 | PD1_CTLA4 | 4321 ± 78 | 1392 | 801 | 1.74 | −2.0% |
| CASE1 | PD1_IL15 | 4254 ± 84 | 1494 | 783 | 1.91 | −3.5% |
| CASE7 | untreated | 41 ± 56 | 3486 | 506 | 6.91 | +0.0% |
| CASE7 | PD1_CTLA4 | 22 ± 37 | 3477 | 502 | 6.94 | −45.0% |
| CASE7 | PD1_IL15 | 13 ± 24 | 3648 | 497 | 7.35 | −68.6% |
| CASE8 | untreated | 4440 ± 85 | 1504 | 1233 | 1.22 | +0.0% |
| CASE8 | PD1_CTLA4 | 4343 ± 89 | 1533 | 1252 | 1.23 | −2.2% |
| CASE8 | PD1_IL15 | 4316 ± 96 | 1628 | 1224 | 1.33 | −2.8% |
| CASE9 | untreated | 212 ± 132 | 3711 | 706 | 5.26 | +0.0% |
| CASE9 | PD1_CTLA4 | 165 ± 112 | 3734 | 700 | 5.34 | −22.2% |
| CASE9 | PD1_IL15 | 126 ± 90 | 3906 | 683 | 5.72 | −40.5% |
| CASE10 | untreated | 3466 ± 211 | 1742 | 986 | 1.77 | +0.0% |
| CASE10 | PD1_CTLA4 | 3381 ± 226 | 1757 | 998 | 1.76 | −2.5% |
| CASE10 | PD1_IL15 | 3325 ± 227 | 1864 | 985 | 1.89 | −4.1% |
| CASE11 | untreated | 5590 ± 85 | 966 | 448 | 2.16 | +0.0% |
| CASE11 | PD1_CTLA4 | 5509 ± 92 | 970 | 466 | 2.09 | −1.5% |
| CASE11 | PD1_IL15 | 5441 ± 92 | 1071 | 450 | 2.39 | −2.7% |
| CASE12 | untreated | 4390 ± 173 | 1458 | 462 | 3.17 | +0.0% |
| CASE12 | PD1_CTLA4 | 4289 ± 172 | 1469 | 473 | 3.12 | −2.3% |
| CASE12 | PD1_IL15 | 4127 ± 153 | 1602 | 462 | 3.48 | −6.0% |
| CASE13 | untreated | 2246 ± 230 | 2054 | 531 | 3.88 | +0.0% |
| CASE13 | PD1_CTLA4 | 2144 ± 240 | 2080 | 536 | 3.89 | −4.6% |
| CASE13 | PD1_IL15 | 2007 ± 228 | 2227 | 529 | 4.22 | −10.7% |
| CASE14 | untreated | 0 ± 0 | 2985 | 1280 | 2.33 | +0.0% |
| CASE14 | PD1_CTLA4 | 0 ± 0 | 2985 | 1280 | 2.33 | — |
| CASE14 | PD1_IL15 | 0 ± 0 | 3124 | 1251 | 2.50 | — |
| CASE15 | untreated | 1071 ± 192 | 3036 | 871 | 3.49 | +0.0% |
| CASE15 | PD1_CTLA4 | 972 ± 184 | 3065 | 884 | 3.47 | −9.3% |
| CASE15 | PD1_IL15 | 893 ± 195 | 3238 | 859 | 3.77 | −16.6% |
| CASE16 | untreated | 3009 ± 147 | 1575 | 1068 | 1.48 | +0.0% |
| CASE16 | PD1_CTLA4 | 2946 ± 151 | 1571 | 1075 | 1.46 | −2.1% |
| CASE16 | PD1_IL15 | 2898 ± 154 | 1673 | 1055 | 1.59 | −3.7% |
| CASE18 | untreated | 6660 ± 190 | 1087 | 479 | 2.27 | +0.0% |
| CASE18 | PD1_CTLA4 | 6539 ± 188 | 1125 | 491 | 2.29 | −1.8% |
| CASE18 | PD1_IL15 | 6376 ± 200 | 1254 | 480 | 2.62 | −4.3% |
| CASE19 | untreated | 0 ± 0 | 3531 | 672 | 5.26 | +0.0% |
| CASE19 | PD1_CTLA4 | 0 ± 0 | 3531 | 672 | 5.26 | — |
| CASE19 | PD1_IL15 | 0 ± 0 | 3670 | 662 | 5.56 | — |
| CASE21 | untreated | 0 ± 0 | 3449 | 859 | 4.03 | +0.0% |
| CASE21 | PD1_CTLA4 | 0 ± 0 | 3449 | 859 | 4.03 | — |
| CASE21 | PD1_IL15 | 0 ± 0 | 3570 | 847 | 4.22 | — |

### Key findings

1. **CD8:Treg ratio is the dominant predictor of ICB response** (panels C, D). Patients with ratio > 5 (CASE7, CASE9) show dramatic treatment effects (−22% to −69%). Patients with ratio < 2 (CASE1, CASE8, CASE10, CASE16) show minimal response (−2% to −4%).

2. **Three patients show spontaneous tumor clearance** (CASE14, CASE19, CASE21) — tumor=0 across all 20 seeds even without treatment. These patients have compositions where KP-calibrated immune dynamics are sufficient to overwhelm tumor growth.

3. **PD1/IL-15 consistently outperforms PD1+CTLA4** across all patients, confirming H4 generalizes beyond KP mice. The advantage is largest in immune-hot tumors (CASE7: −69% vs −45%; CASE9: −41% vs −22%).

4. **Human tumor heterogeneity spans a wide response spectrum**: from near-zero response (CASE8: −2.2% PD1+CTLA4) to near-complete elimination (CASE7: −45% PD1+CTLA4). This 20-fold range far exceeds the variation seen in the KP model.

### Interpretation

The hypothesis is **partially supported**. Human tumors don't uniformly respond better than KP — rather, the CD8:Treg composition ratio determines response magnitude. The KP model (CD8:Treg = 2.66) sits in the middle of the human distribution. Key insight: **initial immune composition, not species-specific biology, is the primary driver of ICB efficacy in this model**. This is consistent with clinical observations that CD8 T-cell infiltration and Treg abundance predict immunotherapy outcomes (e.g., Phase II nivolumab trial using PD-1+ CD8/Treg ratio for patient selection; Cancer Cell 2024 showing Treg IL-2 sequestration drives CD8 exhaustion).

### What is learned vs hardcoded

The CD8:Treg ratio predicting ICB response is **largely a consequence of the hardcoded model architecture**, not an emergent discovery:

- **Hardcoded rules** (model architecture): CD8 cells kill tumors; Tregs suppress CD8 killing in Moore neighborhood; PD1 boosts kill rate; CTLA4 reduces suppression. Given these rules, more CD8 + fewer Tregs → better response is a near-tautology.
- **Learned from data** (SBI v6 posterior, 17 parameters): CD8 kill rate, proliferation/death rates for all cell types, Treg suppression strength, macrophage suppression, suppressive background. These calibrated *rates* determine the quantitative thresholds (e.g., ratio > 5 → near-elimination) and nonlinear dynamics (spontaneous clearance regimes).

The honest framing: the model **recapitulates** the known clinical association between CD8:Treg ratio and ICB response, and provides **quantitative predictions about response thresholds** using parameters calibrated to KP mouse data. It does not independently *discover* CD8:Treg as a biomarker — that is baked into the mechanistic assumptions. What is genuinely predictive is *where the tipping points are* given KP-fitted kinetics.

### Caveats

- KP kinetic parameters (proliferation, death, kill rates) may not accurately represent human biology — only the initial composition is patient-specific
- CyCIF phenotyping uses intensity thresholds (75th percentile) which may misclassify cell types
- The ABM lacks human-specific features (MHC-I diversity, neoantigen load, spatial organization from CyCIF data is not used)
- Spontaneous clearance cases (CASE14, CASE19, CASE21) likely reflect model overshoot — real tumors at these compositions persist, suggesting missing biology (e.g., tumor immune evasion mechanisms, immune exhaustion not fully captured)

---

## 2026-03-05: Sequential dosing + PD1/IL-15 fusion experiment

### Motivation

The timing sweep showed ICB modifies killing efficiency but not immune cell numbers. An anti-PD1/IL-15 fusion protein (SAR445877) could break this pattern by expanding CD8s in situ via IL-15-driven proliferation. We compare monotherapy, combo, sequential dosing, and the fusion protein.

### Model additions

- **CTLA4 ADCC mode**: `CTLA4_ADCC` intervention adds `ctla4_treg_death_bonus = 0.03/tick` (~3x baseline Treg death rate), modeling Fc-mediated Treg depletion (Arce Vargas et al. 2018)
- **PD1/IL-15 fusion**: `PD1_IL15` intervention combines PD1 blockade with IL-15 effects:
  - `il15_cd8_prolif_rate = 0.04` — activated CD8s (activation > 0.3) divide proportional to activation level
  - `il15_cd8_survival_factor = 0.5` — halved CD8 death rate during treatment
  - CD8 daughters inherit 50% parent activation, 30% parent exhaustion
- **CD8 division**: New `_attempt_division()` on CD8TCell class (Tregs already had this)

### Experimental design

8 arms × 2 CTLA4 modes (13 unique conditions), 20 seeds each, all starting week 7, measured week 9:

| Arm | Week 7 | Week 8 |
|-----|--------|--------|
| Untreated | — | — |
| PD1 only | PD1 | — |
| CTLA4 only | CTLA4 | — |
| PD1+CTLA4 combo | PD1+CTLA4 | — |
| PD1→CTLA4 | PD1 | CTLA4 |
| CTLA4→PD1 | CTLA4 | PD1 |
| PD1/IL-15 | PD1_IL15 | — |
| PD1/IL-15 + CTLA4 | PD1_IL15+CTLA4 | — |

### Local test (1 seed)

| Condition | Tumor | CD8 | Treg | CD8:Treg | ΔTumor% |
|-----------|-------|-----|------|----------|---------|
| Untreated | 2937 | 1084 | 442 | 2.45 | — |
| PD1 only | 2896 | 1133 | 396 | 2.86 | -1.4% |
| CTLA4 only | 2780 | 1126 | 461 | 2.44 | -5.3% |
| PD1+CTLA4 combo | 2713 | 1079 | 440 | 2.45 | -7.6% |
| PD1→CTLA4 | 2856 | 1097 | 413 | 2.66 | -2.8% |
| CTLA4→PD1 | 2735 | 1121 | 451 | 2.49 | -6.9% |
| **PD1/IL-15** | **2532** | **1290** | 438 | 2.95 | **-13.8%** |
| **PD1/IL-15 + CTLA4** | **2435** | **1296** | 415 | 3.12 | **-17.1%** |
| **PD1/IL-15 + CTLA4-ADCC** | **2292** | **1310** | 246 | 5.33 | **-22.0%** |

### Key observations (preliminary, 1 seed)

1. **PD1/IL-15 is the best monotherapy** — doubles the effect of PD1+CTLA4 combo (-13.8% vs -7.6%), and it's the first intervention that actually increases CD8 counts (1290 vs 1084 untreated)
2. **CTLA4→PD1 > PD1→CTLA4** in suppression mode (-6.9% vs -2.8%) — clearing suppression first helps
3. **Triple combo (PD1/IL-15 + CTLA4-ADCC) is best overall** at -22.0%, with CD8:Treg ratio of 5.33 (vs 2.45 untreated)
4. **ADCC alone disappoints** — CTLA4-ADCC without IL-15 gives only -1.1% to -4.0% despite depleting Tregs to ~250

### Full results (20 seeds, Cayuga job 2699807, 5h46m)

#### Suppression-only CTLA4 mode

| Condition | Tumor (mean±sd) | CD8 | Treg | CD8:Treg | ΔTumor% |
|-----------|----------------|-----|------|----------|---------|
| Untreated | 1516 ± 675 | 1135 | 428 | 2.66 | — |
| PD1 only | 1470 ± 679 | 1136 | 425 | 2.68 | −3.0% |
| CTLA4 only | 1508 ± 663 | 1133 | 437 | 2.60 | −0.5% |
| PD1+CTLA4 combo | 1453 ± 665 | 1133 | 441 | 2.58 | −4.1% |
| PD1→CTLA4 | 1456 ± 664 | 1137 | 427 | 2.67 | −4.0% |
| CTLA4→PD1 | 1492 ± 655 | 1138 | 439 | 2.60 | −1.6% |
| PD1/IL-15 | 1309 ± 606 | 1248 | 427 | 2.93 | −13.6% |
| PD1/IL-15 + CTLA4 | 1318 ± 593 | 1258 | 432 | 2.92 | −13.1% |

#### ADCC CTLA4 mode (suppression + Treg depletion)

| Condition | Tumor (mean±sd) | CD8 | Treg | CD8:Treg | ΔTumor% |
|-----------|----------------|-----|------|----------|---------|
| CTLA4-ADCC only | 1520 ± 692 | 1134 | 255 | 4.46 | +0.3% |
| PD1+CTLA4-ADCC combo | 1502 ± 708 | 1144 | 250 | 4.59 | −0.9% |
| PD1→CTLA4-ADCC | 1470 ± 683 | 1146 | 224 | 5.14 | −3.0% |
| CTLA4-ADCC→PD1 | 1497 ± 681 | 1139 | 257 | 4.45 | −1.3% |
| PD1/IL-15 + CTLA4-ADCC | 1294 ± 596 | 1262 | 247 | 5.14 | −14.6% |

### Key findings (20 seeds)

1. **PD1/IL-15 is the dominant intervention** — −13.6% as monotherapy, 3× better than PD1+CTLA4 combo (−4.1%). It is the only intervention that meaningfully increases CD8 count (1248 vs 1135 untreated), confirming that expanding the effector pool matters more than modulating existing immune cells.

2. **H2 rejected: PD1→CTLA4 ≈ combo > CTLA4→PD1** — Sequential PD1-first (−4.0%) matches simultaneous combo (−4.1%) and beats CTLA4-first (−1.6%). Boosting CD8 killing first, then reducing Treg suppression works as well as combo; clearing suppression first does not help.

3. **H3 rejected: ADCC depletes Tregs but doesn't kill tumors** — CTLA4-ADCC halves Treg count (255 vs 428) and doubles CD8:Treg ratio (4.46 vs 2.66) but produces essentially zero tumor reduction (+0.3%). The model's Treg suppression mechanism acts locally and reducing Treg numbers doesn't proportionally reduce their suppressive effect on nearby CD8s.

4. **H5 not synergistic: PD1/IL-15 + CTLA4-ADCC is additive** — −14.6% vs −13.6% for PD1/IL-15 alone. Adding ADCC-mediated Treg depletion to the fusion protein contributes only ~1% additional tumor reduction despite dramatic Treg depletion (247 vs 427).

5. **CTLA4 monotherapy is nearly inert** — −0.5% (suppression) and +0.3% (ADCC). Reducing Treg suppressive capacity alone has minimal impact on tumor burden.

### Interpretation

The model strongly predicts that **CD8 effector expansion** (via IL-15) is the rate-limiting step for tumor control, not immune checkpoint relief or Treg depletion. This suggests SAR445877-like agents that combine PD1 blockade with CD8-expanding cytokines should outperform conventional ICB combinations. The failure of CTLA4-ADCC to improve outcomes despite halving Treg counts implies that in this tumor microenvironment, Treg-mediated suppression is not the dominant resistance mechanism.

### Caveats

- CTLA4 ADCC death bonus (0.03/tick) is a rough estimate; true depletion kinetics may differ
- IL-15 proliferation parameters (rate=0.04, survival factor=0.5) are not calibrated to data
- Model lacks NK cells, which are also IL-15-responsive
- 9-week endpoint; longer simulations may show different dynamics

---

## 2026-03-04: Treatment timing sweep — two protocols compared

### Motivation

With v6 posterior (correct Gaglia treatment timing, no ceiling-hitting), we ask: **what if treatment started earlier?** The Gaglia protocol treats for 1 week at the end (week 7→8). We simulate treatment starting at weeks 1–7 and compare tumor burden at the 8-week endpoint.

### Protocol correction

Initial sweep applied treatment **continuously from start week through endpoint** — confounding timing with duration (week 1 = 7 weeks of treatment, week 7 = 1 week). Re-ran with a **1-week pulse at each start time** to isolate the timing effect. Added `remove_interventions()` to `LUADModel` for treatment withdrawal.

### Experiment 1: Continuous treatment (treatment ON from start week to week 8)

- Script: `scripts/treatment_timing_sweep.py` (before pulse fix)
- Cluster: Cayuga job 2685954, 2h38m on 16 CPUs
- Seeds: 20 per condition, 160 total simulations

| Treatment Start | Tumor (mean±sd) | CD8 | Treg | CD8:Treg | Tumor Δ% |
|---|---|---|---|---|---|
| No treatment | 1298 ± 578 | 1087 | 398 | 2.73 | — |
| **Week 1** | **658 ± 437** | 1087 | 385 | 2.84 | **−49.3%** |
| Week 2 | 830 ± 607 | 1091 | 388 | 2.82 | −36.1% |
| Week 3 | 1016 ± 551 | 1096 | 403 | 2.73 | −21.7% |
| Week 4 | 1087 ± 552 | 1093 | 393 | 2.79 | −16.3% |
| Week 5 | 1141 ± 559 | 1089 | 401 | 2.73 | −12.1% |
| Week 6 | 1210 ± 528 | 1098 | 406 | 2.71 | −6.8% |
| **Week 7** (Gaglia) | **1257 ± 564** | 1088 | 408 | 2.67 | **−3.2%** |

Clear monotonic trend: earlier (=longer) treatment → more tumor reduction. Week 1 start halves the tumor. But this confounds timing with duration.

### Experiment 2: 1-week pulse (matching Gaglia protocol duration)

- Script: `scripts/treatment_timing_sweep.py` (with `intervention_duration_ticks=TICKS_PER_WEEK`)
- Cluster: Cayuga job 2686022, 2h50m on 16 CPUs
- Seeds: 20 per condition, 160 total simulations

| Treatment Start | Tumor (mean±sd) | CD8 | Treg | CD8:Treg | Tumor Δ% |
|---|---|---|---|---|---|
| No treatment | 1298 ± 578 | 1087 | 398 | 2.73 | — |
| Week 1 | 1208 ± 614 | 1092 | 392 | 2.80 | −6.9% |
| **Week 2** | **1114 ± 624** | 1091 | 390 | 2.81 | **−14.2%** |
| Week 3 | 1321 ± 592 | 1090 | 400 | 2.73 | +1.8% |
| Week 4 | 1250 ± 588 | 1086 | 390 | 2.80 | −3.7% |
| Week 5 | 1208 ± 554 | 1088 | 400 | 2.73 | −7.0% |
| Week 6 | 1249 ± 543 | 1095 | 405 | 2.71 | −3.8% |
| Week 7 (Gaglia) | 1257 ± 564 | 1088 | 408 | 2.67 | −3.2% |

### Comparison of the two protocols

1. **Continuous treatment is dramatically more effective** — week 1 continuous gives −49% vs week 1 pulse gives only −7%. Most of the benefit comes from treatment *duration*, not timing alone.
2. **For a fixed 1-week pulse, week 2 is optimal** — 14.2% reduction, the only clearly beneficial window. Week 3 is paradoxically neutral (+1.8%), suggesting a timing where treatment may disrupt an ongoing immune response.
3. **Late pulse treatment is marginal regardless of timing** — weeks 5–7 all give ~3–7% reduction, consistent with Gaglia's observed modest treatment effects.
4. **Immune cell counts invariant across all conditions** — CD8 (~1090) and Treg (~400) are nearly identical. Treatment affects CD8 killing efficiency, not immune cell numbers.
5. **High stochastic variance** (sd ~550–624) dominates — timing effects are small relative to seed-to-seed variability, consistent with the heterogeneous clinical response to checkpoint blockade.

### Clinical interpretation

The continuous protocol is closer to clinical checkpoint blockade (patients receive treatment every 2–4 weeks for months). The 1-week pulse matches the Gaglia mouse protocol. The comparison suggests that **treatment duration matters more than timing** — a sustained course of immunotherapy provides cumulative benefit that a single pulse cannot achieve, regardless of when it's given.

### Limitation: monotherapy data unavailable

Gaglia Dataset03 only has combo PD1+CTLA4 vs control (4 groups: 5wk/8wk × treated/control, 6 mice each). No PD1-alone or CTLA4-alone arms. The individual intervention parameters (`pd1_kill_bonus=0.02`, `treg_mod_factor=0.5`) are fixed assumptions, not inferred. Monotherapy or sequencing predictions would require external calibration data.

### Output files

- `outputs/treatment_timing_sweep/sweep_results.npz` — 1-week pulse results (Experiment 2)
- `outputs/treatment_timing_sweep/treatment_timing_sweep.png` — 4-panel summary figure (Experiment 2)
- Experiment 1 results were overwritten by Experiment 2

### SLURM note

Cayuga SLURM upgraded to v25.05.0 but login nodes still have old v22.05.2 in default PATH. Must use full path: `/opt/ohpc/pub/software/slurm/25.05.0/bin/sbatch`.

---

## 2026-03-03: v5 SBI — correct treatment timing + early intervention exploration

### Critical finding: v1–v4 had wrong treatment timing

All previous inference runs applied treatment (PD1+CTLA4) from tick 0 (simulation start). But the Gaglia et al. 2023 paper's actual protocol (STAR Methods) was:

- Tumors initiated by intratracheal lentiviral Cre at week 0
- Tumors grow **untreated** for 5 or 8 weeks
- Treatment: 3 doses over 7 days (200μg each antibody i.p. on days 0, 3, 6)
- Sacrifice ~1 week after treatment start

So treatment is only a **1-week pulse at the end**, not continuous from the start:
- **5wk group**: tumor grows 4 weeks untreated → 1 week treatment → sacrifice at week 5
- **8wk group**: tumor grows 7 weeks untreated → 1 week treatment → sacrifice at week 8

(Note: "5 weeks" and "8 weeks" in the metadata refer to total time post-initiation. The paper's supplemental figures compare 6wk vs 9wk total timepoints, suggesting sacrifice may be slightly later. Using 5wk/8wk as labeled in the shared data.)

### Consequences of wrong timing in v1–v4

1. **Treatment effect sizes underestimated**: The model spread a small observed effect over 120-192 ticks instead of ~24 ticks. The inferred pd1_kill_bonus and treg_mod_factor are likely too weak.
2. **Immune dynamics distorted**: In reality, the TME evolves for weeks without interference, developing exhaustion and immunosuppression. Then treatment briefly disrupts this. Our model instead had treatment shaping the TME from the beginning.
3. **Parameter compensation**: Other parameters (exhaustion rates, recruitment, suppression) may have compensated for the wrong timing, making the posterior biologically misleading despite fitting the summary statistics.

### v5 plan

1. Add `intervention_start_tick` parameter to `run_single_condition()` — treatment turns on partway through simulation
2. Correct timing:
   - 5wk control: 120 ticks, no treatment
   - 5wk treated: 120 ticks, treatment starts at tick 96 (week 4 → last 24 ticks = 1 week)
   - 8wk control: 192 ticks, no treatment
   - 8wk treated: 192 ticks, treatment starts at tick 168 (week 7 → last 24 ticks = 1 week)
3. Re-run 2000 sims across 10 nodes with same v4 priors/params
4. After fitting: use corrected posterior for early intervention counterfactuals

### v5 results (completed 2026-03-03, ~4.5h)

All 10 chunks completed. SNPE converged after 148 epochs (best validation: -39.07). Treatment timing fix significantly shifted posteriors:

**Key parameter shifts (v4 → v5):**
- tumor_proliferation_rate: 0.111 → **0.168** (+51%) — tumors grow faster when unchecked for 4-7 weeks
- cd8_base_kill: 0.307 → **0.459** (+50%) — CD8s need to kill harder in just 1 week
- cd8_activation_gain: 0.265 → **0.370** (+40%) — stronger activation needed
- treg_death_rate: 0.007 → **0.004** (−43%) — Tregs live longer during untreated growth

**New ceiling-hitting issues (3 parameters):**
- tumor_proliferation_rate: 0.168, p95=0.197 vs ceiling 0.20
- cd8_base_kill: 0.459, p95=0.591 vs ceiling 0.60
- cd8_activation_gain: 0.370, p95=0.489 vs ceiling 0.50

PPC: CD8:Treg ratios still underestimated (~2.5 predicted vs ~3.4 observed).

### v6: widen 3 ceilings

| Parameter | v5 ceiling | v6 ceiling |
|---|---|---|
| cd8_base_kill | 0.60 | **0.90** |
| cd8_activation_gain | 0.50 | **0.80** |
| tumor_proliferation_rate | 0.20 | **0.35** |

SLURM array **2685871**, combine **2685872**.

### v6 results (completed 2026-03-03, ~5.5h)

All 10 chunks completed (298-327 min each). SNPE converged after 122 epochs (best validation: -36.35). **Zero wall-hitting — best posterior yet.**

**All 3 ceiling issues resolved:**
- cd8_base_kill: 0.459 (at wall) → **0.610** [0.376, 0.844] — found natural home
- tumor_proliferation_rate: 0.168 (at wall) → **0.204** [0.129, 0.276] — centered
- cd8_activation_gain: 0.370 (at wall) → **0.383** [0.068, 0.718] — barely moved, ceiling wasn't constraining

**Full v4 → v6 shift (treatment timing + prior widening):**

| Parameter | v4 (wrong timing) | v6 (correct) | Change | Interpretation |
|---|---|---|---|---|
| cd8_base_kill | 0.307 | 0.610 | +99% | CD8s must kill hard in 1 week |
| tumor_proliferation_rate | 0.111 | 0.204 | +84% | Tumors grow unchecked for weeks |
| cd8_exhaustion_rate | 0.027 | 0.016 | −41% | Exhaustion was proxying for timing |
| cd8_exhaustion_death_bonus | 0.009 | 0.006 | −35% | Less exhaustion-driven death |
| recruit_exhaustion_priming | 0.328 | 0.221 | −33% | Recruits less pre-exhausted |
| treg_prolif_rate | 0.120 | 0.085 | −29% | Less aggressive Treg accumulation |

**Key biological rates (24 ticks/week):**
- Tumor cells divide almost daily (0.204/tick → 4.9/week)
- 61% CD8 kill probability per encounter (before modifiers)
- CD8s reach substantial exhaustion in ~2-3 weeks
- Tregs near tumor double every ~3.5 days (prolif 2.0/wk vs death 0.11/wk)

**PPC:** 5wk conditions well-matched. 8wk control good. 8wk treated CD8:Treg ratio overpredicted (model underestimates adaptive resistance at late timepoints).

**Output files:** `outputs/bayesian_inference_v6/posterior_{marginals,predictive}.png`, `posterior_summary.csv`, `posterior_samples.npy`

### Exploration: what the model can tell us that the paper can't

With correctly fitted mechanistic parameters, we can simulate counterfactual treatment schedules:

| Scenario | Description | Biological question |
|----------|-------------|-------------------|
| **Early intervention** | Treat at week 1-2 instead of 5/8 | Is the TME more responsive before immunosuppression establishes? |
| **Prolonged treatment** | Continuous blockade for 3-4 weeks | Does sustained treatment prevent adaptive resistance? |
| **Optimal timing sweep** | Treat at weeks 1, 2, 3, ..., 7 | Is there a critical window for maximum efficacy? |
| **Sequential therapy** | PD1 first (1wk) then CTLA4 (1wk) | Does order matter? |
| **Dose duration** | 1wk vs 2wk vs 3wk treatment | Minimum effective duration? |

These would take months of mouse experiments but can be explored in silico in hours with the fitted model.

---

## 2026-03-02: v4 SBI — z-score normalization, floor widening, Treg death rate

### Goal
Address v3 posterior issues: 3 floor-hitting parameters, non-identifiable pd_l1_penalty, underestimated CD8:Treg ratio, and unequal stat scale weighting.

### v3 results summary
- 17 params, 60-dim stats, 2000 sims across 10 nodes (~4.5h total)
- Prior widening succeeded: tumor_proliferation_rate 0.084→0.131, treg_prolif_rate 0.066→0.083
- New MHC-I params well-constrained: induction=0.055 (fast), decay=0.002 (slow)
- 12/17 params reasonably constrained
- Issues: 3 floor-hitting (cd8_exhaustion_rate, immune_base_death_rate, cd8_exhaustion_death_bonus), pd_l1_penalty still flat, CD8:Treg ratio underestimated

### Changes from v3 → v4

1. **Z-score normalization** of summary stats before SNPE training. Ratios (~3.5) and fractions (~0.05) now contribute equally to the loss.

2. **Widened floors** for 3 floor-hitting parameters:
   - cd8_exhaustion_rate: 0.005 → **0.001**
   - immune_base_death_rate: 0.001 → **0.0005**
   - cd8_exhaustion_death_bonus: 0.005 → **0.001**

3. **Dropped pd_l1_penalty** from inference (non-identifiable in v2+v3). Fixed at 0.5.

4. **Added treg_death_rate** [0.001, 0.03] — dedicated Treg turnover to decouple CD8:Treg ratio from shared immune_base_death_rate.

### v4 configuration
- **17 parameters** (dropped pd_l1_penalty, added treg_death_rate) + 1 fixed
- **60-dim output**, z-score normalized
- SLURM array **2682586** (10 nodes), combine **2682587**

### v4 results (completed 2026-03-03, ~4h total)

All 10 chunks completed (2h56m–4h06m each), combine in 4m58s. SNPE converged after 131 epochs (best validation: -23.94). 1921/2000 valid sims after NaN filtering.

**Major improvement: zero wall-hitting.** All 17 parameters have posterior mass away from prior boundaries. v3's 3 floor-hitting issues are fully resolved.

| Parameter | Posterior Mean | 95% CI | Prior | Notes |
|-----------|---------------|--------|-------|-------|
| cd8_base_kill | 0.307 ± 0.079 | [0.158, 0.468] | [0.02, 0.60] | Well-constrained |
| cd8_exhaustion_rate | 0.027 ± 0.016 | [0.003, 0.062] | [0.001, 0.15] | Floor fix worked |
| cd8_activation_gain | 0.265 ± 0.113 | [0.051, 0.472] | [0.02, 0.50] | Wide but centered |
| tumor_proliferation_rate | 0.111 ± 0.015 | [0.087, 0.146] | [0.005, 0.20] | **Tightest** — well away from ceiling |
| macrophage_suppr_base | 0.270 ± 0.105 | [0.077, 0.469] | [0.05, 0.50] | Moderate |
| suppressive_background | 0.078 ± 0.034 | [0.017, 0.142] | [0.01, 0.15] | Consistent with v2/v3 |
| immune_base_death_rate | 0.004 ± 0.001 | [0.002, 0.007] | [0.0005, 0.025] | Floor fix worked |
| recruitment_rate | 0.018 ± 0.005 | [0.008, 0.028] | [0.001, 0.05] | Consistent |
| treg_suppression | 0.278 ± 0.086 | [0.110, 0.432] | [0.08, 0.45] | Moderate–strong |
| cd8_exhaustion_death_bonus | 0.009 ± 0.001 | [0.007, 0.011] | [0.001, 0.05] | **Very tight** — floor fix worked |
| treg_prolif_rate | 0.120 ± 0.020 | [0.076, 0.148] | [0.005, 0.15] | Near ceiling but not hitting |
| treg_death_rate | **0.007 ± 0.001** | [0.004, 0.009] | [0.001, 0.03] | **New param well-constrained** |
| mac_tumor_death_rate | 0.072 ± 0.032 | [0.016, 0.136] | [0.01, 0.15] | Moderate |
| mac_recruit_suppression | 3.77 ± 1.73 | [0.80, 7.30] | [0.5, 8.0] | Wide but centered |
| recruit_exhaustion_priming | 0.328 ± 0.051 | [0.227, 0.426] | [0.05, 0.6] | **Now constrained** (was flat in v2/v3) |
| mhc_i_induction_rate | 0.057 ± 0.022 | [0.014, 0.096] | [0.005, 0.10] | Fast induction |
| mhc_i_decay_rate | 0.002 ± 0.001 | [0.001, 0.003] | [0.0005, 0.01] | Slow decay (immune evasion) |

**Key improvements over v3:**
- recruit_exhaustion_priming now well-constrained at 0.33 (was flat) — z-score normalization fixed the scale imbalance
- cd8_exhaustion_death_bonus extremely tight: 0.009 [0.007, 0.011] — data strongly constrains this
- treg_death_rate well-identified at 0.007, much lower than immune_base_death_rate (0.004) — Tregs die ~1.6× faster than other immune cells, balancing their proliferative advantage

**Remaining issues:**
- 8wk CD8:Treg ratio now slightly overpredicted (was underpredicted in v2/v3)
- mac_recruit_suppression still wide — macrophage dynamics remain hardest to pin down

### Treatment mechanism analysis (posterior mean params)

Ran posterior mean through all 4 conditions (5wk control, 5wk treated, 8wk control, 8wk treated) with 169 ticks. Key findings:

**1. Treatment slows but does not eliminate tumor growth (~20% reduction)**

| Condition | Tumor count | Immune total | CD8 | Treg |
|-----------|------------|-------------|-----|------|
| 5wk control | 1414 | 222 | 59 | 26 |
| 5wk treated | 1262 | 264 | 82 | 29 |
| 8wk control | 1752 | 182 | 42 | 42 |
| 8wk treated | 1633 | 195 | 51 | 36 |

**2. Primary mechanism: CD8 activation and infiltration, NOT Treg depletion**

| Functional marker | 5wk ctrl | 5wk tx | Δ | Interpretation |
|---|---|---|---|---|
| CD8 GrzB+ (cytotoxic) | 0.35 | 0.54 | +54% | More active killers |
| CD8 PD-1+ (exhaustion marker) | 0.60 | 0.47 | −22% | Less exhaustion |
| CD8 Ki67+ (proliferating) | 0.76 | 0.82 | +8% | Maintained cycling |
| Tumor B2m+ (MHC-I) | 0.72 | 0.82 | +14% | More antigen presentation |

Anti-PD1 primarily works by:
- Boosting CD8 cytotoxic capacity (GrzB+ up 54%)
- Reducing exhaustion (PD-1+ down 22%)
- Enabling deeper CD8 infiltration into tumor core

Anti-CTLA4 contributes by reducing Treg suppression strength (not depleting Tregs — Treg counts actually increase due to continued tumor-proximity proliferation).

**3. 8-week adaptive resistance**

By 8 weeks, tumors recover partially:
- Treg accumulation outpaces CD8 maintenance (Treg:CD8 ratio worsens)
- CD8 exhaustion increases despite treatment
- Tumor expands, creating more suppressive microenvironment
- Macrophages decline further due to tumor-driven attrition

This matches the clinical observation that checkpoint immunotherapy shows initial response followed by acquired resistance in many patients.

### Tick-to-time mapping

The model uses **24 ticks per week** (1 tick ≈ 7 hours):
- 5-week timepoint: tick 120
- 8-week timepoint: tick 192
- Treatment window (2 weeks pre-sacrifice): starts at tick 72 (5wk) or tick 144 (8wk)

**Parameter rates in real time:**

| Parameter | Per-tick rate | Per-day rate | Per-week rate | Interpretation |
|---|---|---|---|---|
| tumor_proliferation_rate | 0.111 | 0.37 | 2.6 | Each tumor cell divides ~2-3× per week |
| immune_base_death_rate | 0.004 | 0.014 | 0.10 | ~10% immune cell turnover/week |
| treg_death_rate | 0.007 | 0.023 | 0.16 | ~16% Treg turnover/week |
| cd8_exhaustion_rate | 0.027 | 0.091 | 0.64 | CD8 substantially exhausted in ~10 days |
| treg_prolif_rate | 0.120 | 0.40 | 2.8 | Tregs near tumor divide 2-3×/week |

**Important caveat:** The 24 ticks/week mapping is a convention chosen to match the Gaglia experimental timepoints (5 weeks and 8 weeks post-tumor establishment). The model starts from an already-established tumor mass, not from tumor initiation. The tick-to-time relationship is not mechanistically derived — it simply ensures the simulation duration spans the experimental observation window.

### Output files (v4)
- `outputs/bayesian_inference_v4/posterior_marginals.png` — 17-panel posterior distributions
- `outputs/bayesian_inference_v4/posterior_predictive.png` — observed vs predicted (4 conditions)
- `outputs/bayesian_inference_v4/posterior_summary.csv` — parameter estimates with 95% CIs
- `outputs/bayesian_inference_v4/posterior_samples.npy` — 10,000 × 17 posterior samples
- `outputs/bayesian_inference_v4/training_data_v4.npz` — 2000 × 17 (theta) + 2000 × 60 (x)

### Distributed SBI infrastructure

Built for v3, reused for v4. Reduces wall-clock from ~31h (single node) to ~4h (10 nodes):

- `gaglia_abm/runs/sbi_worker.py` — generates chunk of (theta, x) pairs with deterministic RNG
- `gaglia_abm/runs/sbi_combine.py` — merges chunks, filters NaN/Inf, runs SNPE
- `gaglia_abm/runs/slurm_sbi_array.sh` — 10-node SLURM array (16 CPUs, 32G, 8h each)
- `gaglia_abm/runs/slurm_sbi_combine.sh` — dependent combine job (8 CPUs, 32G, 2h)

Submission pattern:
```bash
JOB=$(sbatch --parsable slurm_sbi_array.sh)
sbatch --dependency=afterok:$JOB slurm_sbi_combine.sh
```

### Next steps
- Fix 8wk CD8:Treg overprediction — may need condition-specific Treg dynamics
- Improve PPC: use 50+ posterior samples instead of posterior mean
- Consider adding CD4 helper infiltration stats
- Explore sensitivity analysis around posterior mean
- Run longer simulations (12+ weeks) to predict treatment durability

---

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
