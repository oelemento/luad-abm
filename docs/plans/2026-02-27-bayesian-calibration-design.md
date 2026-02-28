# Bayesian Calibration Pipeline — Proposal Demo

**Date:** 2026-02-27
**Goal:** Proof-of-concept showing ABM parameters can be inferred from Gaglia et al. mouse spatial data, producing a figure for the Sanofi iDEA-TECH full proposal (deadline: March 20, 2026).

## Approach

Direct extraction of summary statistics from Gaglia CyCIF data + Latin Hypercube parameter search to find ABM parameters that recapitulate observed spatial patterns. Compare untreated vs ICB-treated mice to validate treatment effect predictions.

## Data Source

Gaglia et al., Cancer Cell 2023 (doi:10.1016/j.ccell.2023.03.015)
- Synapse: syn30715952
- Downloaded to: `data/gaglia_2023/`
- Key dataset: **Dataset03_KP_LucOS_anti_PD1_CTLA4** (9.7M cells, 24 mice, 4 treatment groups)
- Also: **Dataset05_Human_Lung_Adenocarcinoma** (7.8M cells, 14 patients) for cross-species validation

## Design

### Step 1: Extract Gaglia Summary Statistics

**Script:** `scripts/extract_gaglia_stats.py`

Parse `.mat` files from Dataset03 and produce per-mouse summary stats:

- **Cell type fractions:** Tc, Th, Treg, B, MAC, TAM counts normalized per mouse
- **Infiltration profiles:** Bin cells by distance-to-tumor-boundary into core (<50µm), cuff (50-130µm), periphery (>130µm). Compute fraction of each cell type in each region.
- **CD8 density gradient:** Tc count as function of distance from tumor boundary (binned histogram)
- **Lymphonet stats:** Mean network size, fraction of lymphocytes in networks
- **Treatment groups:** Mouse groups 1-4 from Settings/options (Ctrl, anti-PD1, anti-CTLA4, combo)

**Output:** `data/gaglia_2023/gaglia_summary_stats.csv` — one row per mouse, columns for all stats.

**Data files used per dataset:**
- `Results_Morp_*.mat` → X, Y coordinates
- `Results_CellType_*.mat` → cell type assignments (hierarchical: 4 levels)
- `Results_RoiDist_*.mat` → distance to tumor boundary and blood vessels
- `Results_Nets_*_dist50.mat` → lymphonet assignments (NetworkID, Size)
- `Results_Settings_*.mat` → MouseGroup, MouseNum

### Step 2: ABM Summary Statistic Extraction

**Module:** `luad_abm/luad/calibration.py`

Functions to extract matching summary stats from a completed ABM simulation:
- `extract_infiltration_profile(model)` → core/cuff/periphery fractions per cell type
- `extract_cell_fractions(model)` → normalized cell type counts
- `extract_cd8_distance_profile(model)` → CD8 distance-to-tumor histogram
- `compute_distance(obs_stats, sim_stats, obs_variance)` → weighted sum of squared differences

Distance metric: `d = Σ w_i × (obs_i - sim_i)² / var_i` where variance comes from across-mouse variation in Gaglia data.

### Step 3: Parameter Search

**Script:** `scripts/calibration_search.py`

**Parameters to search (6):**

| Parameter | Range | Default |
|---|---|---|
| `cd8_base_kill` | [0.05, 0.30] | 0.15 |
| `cd8_exhaustion_rate` | [0.02, 0.12] | 0.05 |
| `pd_l1_penalty` | [0.3, 0.9] | 0.6 |
| `macrophage_suppr_base` | [0.1, 0.5] | 0.25 |
| `suppressive_background` | [0.02, 0.15] | 0.05 |
| `tumor_proliferation_rate` | [0.01, 0.06] | 0.03 |

**Method:** Latin Hypercube Sampling, ~500 combinations.

**Simulation config:**
- Base preset: G3_inflammatory.json (closest to KP-LucOS immune-active phenotype)
- Ticks: 500 (sufficient for spatial pattern stabilization)
- Seeds: 3 replicates per parameter set (account for stochasticity)
- Total: 1500 simulations × ~30s = ~12.5 hours locally, or submit as batch on Cayuga

**Two-phase calibration:**
1. **Baseline:** Match to Gaglia Ctrl group (MouseGroup 1). Find parameter sets where d(obs, sim) is minimal.
2. **Treatment:** Using top-10 baseline parameter sets, enable PD1 intervention, compare to Gaglia ICB group (MouseGroup 3-4). Score treatment prediction accuracy.

**Output:** `data/calibration_results.csv` — parameter sets with distance scores for both phases.

### Step 4: Proposal Figure

**Script:** `scripts/plot_calibration_results.py`

Multi-panel figure:
- **A:** Gaglia observed infiltration profiles (Ctrl vs ICB) — grouped bar chart
- **B:** Inferred parameter distributions (top 10% of fits shown as violin/box plots)
- **C:** Side-by-side spatial patterns — Gaglia observed vs ABM simulated (grid snapshots)
- **D:** Treatment validation — ABM baseline→PD1 prediction vs Gaglia Ctrl→ICB observed shift

## File Structure

```
scripts/
  extract_gaglia_stats.py
  calibration_search.py
  plot_calibration_results.py

luad_abm/luad/
  calibration.py

data/gaglia_2023/
  gaglia_summary_stats.csv          (generated)
data/
  calibration_results.csv           (generated)
```

## Timeline

- Day 1: Steps 1-2 (data extraction + ABM stat matching)
- Day 2-3: Step 3 (parameter search, possibly on Cayuga)
- Day 4: Step 4 (figure generation + iterate)
