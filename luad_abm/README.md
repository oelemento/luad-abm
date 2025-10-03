# LUAD Microenvironment Agent-Based Model

## 1. Overview
- **Goal:** Reproduce key spatial/immune features of lung adenocarcinoma (LUAD) microenvironments and explore interventions.
- **Framework:** Mesa (Python), with custom scheduler and visualization utilities.
- **Focus preset:** Fibrotic/immune-excluded LUAD (clinical "G4") phenotype drawn from the project's instruction brief.

## 2. Repository layout
```
luad_abm/
  config/
    G3_inflammatory.json   # alternative preset
    G4_fibrotic.json       # main preset
  luad/
    __init__.py
    agents.py              # agent classes
    fields.py              # ECM / chemokine / TGFβ grids
    model.py               # LUADModel core
    rules.py               # movement/interaction helpers
    scheduler.py           # staged scheduler (Mesa replacement)
    metrics.py             # distance & adjacency metrics
    viz.py                 # plotting + movie writer
    io.py                  # CSV/NPZ persistence
  outputs/                 # default output root
  runs/
    run.py                 # CLI entry point
scripts/
  remake_movie.py          # rebuild movie.gif at new scale/legend
  regenerate_plots.py      # recolor/re-style population plots
```

## 3. Environment setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mesa numpy pandas matplotlib scipy imageio pillow tqdm
```
Use `../.venv/bin/python` when launching from inside `luad_abm/`.

## 4. Running a simulation
General command:
```bash
../.venv/bin/python runs/run.py \
    --preset config/G4_fibrotic.json \
    --ticks 2000 \
    --seed 7 \
    --out outputs/G4_baseline_v2 \
    --movie-scale 8 \
    --grid-interval 10
```

Common flags:
- `--add <INTERVENTION>`: `PD1`, `antiTGFb`, `VISTA`, `TLS_boost`, `TIM3`, `Treg`.
- `--no-movie`: skip GIF generation.
- `--movie-scale N`: upsample frames (default 6).
- `--no-legend`: omit movie legend sidebar.
- `--grid-interval K`: record every Kth tick (also controls movie frame density).
- `--distance-interval`, `--interaction-interval`: sampling cadence for metrics.
- `--no-progress`: hide tqdm progress bar.

Outputs under `--out/`:
```
timeseries.csv
agent_counts.csv
cd8_tumor_distances.csv
cd8_tumor_distance_hist.npz
interactions.csv
grid_snapshots.npz
summary_plots/
  trajectories.png
  state_metrics.png
  distance_cdf.png
  interaction_heatmap.png
  tls_quality.png
movie.gif (optional)
```

## 5. Agents and state
| Agent | Color | State variables | Key behaviors |
|-------|-------|-----------------|---------------|
| Tumor/epithelium | pink | EMT ∈ [0,1], PD-L1, MHC-I, proliferative flag | EMT ↑ with TGFβ/ECM; PD-L1 induced by IFNγ; divides into empty neighbor with prob `tumor_proliferation_rate × (1 − EMT)` |
| CAF/αSMA fibroblast | gold | activation | Deposits ECM & TGFβ proportional to activation |
| CD8 T cell | blue | activation, exhaustion, recent kill count | Chemotaxis up CXCL9/10; attempts kill with probability `base × modifiers × (1 − suppression)` |
| CD4 T cell | teal | activation | Mild CXCL9/10 support when active |
| Treg | purple | suppression strength | Chemotaxis up TGFβ; contributes to local suppression |
| Macrophage | gray | polarization p ∈ [−1,+1] | Drift toward TGFβ; suppressive when M2-like; secretes CXCL9 when M1 |
| TLS node | green disk | quality q ∈ [0,1] | Emits CXCL13/CXCL9/10; quality drifts unless boosted |

Movement order each tick: 1) fields update, 2) movement, 3) interactions, 4) state updates. One tick ≈ 1 minute.

## 6. Global fields (`fields.py`)
Maintains ECM, CXCL9/10, CXCL13, TGFβ grids. Users deposit contributions into per-field buffers; each tick applies diffusion (3×3 kernel) and exponential decay. ECM values cap at 1.0; default decay 0.0015.

## 7. Selected parameter table
| Parameter | Value (baseline) | Description |
|-----------|------------------|-------------|
| Grid size | 100×100 | ≈1 mm² ROI (10 µm per site) |
| Tick duration | 1 minute | 2 000 ticks ≈ 33 h |
| `tumor_proliferation_rate` | 0.03 | Base division probability per tick |
| `cd8_base_kill` | 0.15 | Baseline CD8 kill probability |
| `pd_l1_penalty` | 0.6 | Kill penalty for PD-L1 high targets |
| `ecm_penalty` (G4) | 0.52 | Movement penalty in dense ECM |
| `caf_ecm_deposition` | 0.02 | ECM added per tick per CAF |
| `caf_tgfb_secretion` | 0.03 × preset gain | CAF TGFβ secretion |
| `tls_cxcl13_emission` | 0.6 × preset gain | TLS chemokine emission |
| `tls_quality_drift` | −0.003 | Quality decay per tick |
| `treg` suppression | 0.30 | Treg contribution to local suppression |
| `macrophage_suppr_base` | 0.26 | Base TAM suppression strength |

Full parameter list lives in `model.py` and preset JSON files.

## 8. Interventions (`--add`)
| Flag | Effect |
|------|--------|
| `PD1` | Enables PD-1 blockade (halve exhaustion accrual, add kill bonus) |
| `antiTGFb` | Reduces CAF deposition/ECM penalty and EMT gain |
| `VISTA`/`CSF1R` | Reprograms TAM polarization towards M1, reduces suppression |
| `TLS_boost` | Raises TLS quality floor (default 0.75) |
| `TIM3` | Slows TLS quality drift (+0.004 per tick) |
| `Treg` | Halves Treg suppression strength |

## 9. Regenerating movies (no re-run)
If `grid_snapshots.npz` exists:
```bash
../.venv/bin/python scripts/remake_movie.py \
    outputs/G4_baseline_v2 \
    --movie-scale 8 \
    --no-legend            # optional
```
The script overwrites `movie.gif`; rename beforehand to keep old versions.

To shorten a movie without re-running:
```bash
../.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from luad import viz

run_dir = Path("outputs/G4_antiTGFb_PD1")
keep = 4  # keep every 4th frame
snapshots = list(zip(*np.load(run_dir / "grid_snapshots.npz", allow_pickle=True).values()))
snapshots = snapshots[::keep]
viz.write_movie(snapshots, run_dir, scale=8, legend=True)
PY
```

## 10. Regenerating plots with aligned styling
```bash
../.venv/bin/python scripts/regenerate_plots.py
# or limit to specific runs:
../.venv/bin/python scripts/regenerate_plots.py outputs/G4_baseline_v2
```
This updates `summary_plots/trajectories.png` with movie-matched colors, thicker lines (2 pt), larger fonts (labels 12 pt, title 14 pt), and an external legend.

## 11. Metrics produced
- `timeseries.csv`: counts per agent class, EMT-high fraction, ECM fraction, TLS quality.
- `cd8_tumor_distances.csv` / `.npz`: CD8→tumor CDF summaries and raw histograms.
- `interactions.csv`: adjacency enrichment vs shuffled baseline.
- `grid_snapshots.npz`: saved lattices for post hoc movies.
- `summary_plots/`: trajectories, state metrics, distance CDF, interaction heatmap, TLS quality.
- `movie.gif`: optional animation (legend + scale configurable).

## 12. Explain-it slide deck (quick reference)
- **Slide 1**: “Agent-Based Modeling in One Minute” (concept, why ABM, building blocks, outputs).
- **Slide 2**: “LUAD G4 ABM snapshot” (agents/colors, rules, preset counts, interventions, outputs). See project notes for sample text.

## 13. Parameter provenance (condensed)
- Lattice scale mirrors multiplex imaging fields used to quantify TLS per mm².
- Initial counts approximate fibrotic LUAD (CAF rich, immune poor, single TLS node); refine with absolute counts from pathomics data as available.
- Movement speeds and chemotaxis heuristics draw from published immune-tissue ABMs (T cell motility ~µm/min).
- Kill and suppression formulas mirror PD-1 checkpoint ABMs; tuned so untreated G4 remains persistent after recent adjustments.
- Field kinetics (ECM/TGFβ) follow instruction defaults; swap in lung-specific diffusion/decay constants when measured values are collected.
- Intervention effects reflect qualitative outcomes reported for anti-TGFβ/PD-1 combos in fibrotic LUAD cases (CAF ring thinning, CD8 re-entry).

## 14. Troubleshooting
- **Slow runs**: increase `--grid-interval` / `--distance-interval`; reduce `--ticks` for tests.
- **Movie too long**: resample frames or set `--grid-interval` higher; adjust FPS if desired.
- **Tumor shrinking unintentionally**: lower `cd8_base_kill`, increase `pd_l1_penalty`, or raise `tumor_proliferation_rate`.
- **Need different color schemes**: modify `plot_trajectory_palette()` in `viz.py` and re-run `scripts/regenerate_plots.py`.

## 15. Scripts
### `scripts/remake_movie.py`
```
usage: remake_movie.py [-h] [--movie-scale MOVIE_SCALE] [--no-legend] run_dir
```
Rebuilds `movie.gif` from `grid_snapshots.npz` with optional scaling and legend toggle.

### `scripts/regenerate_plots.py`
```
usage: regenerate_plots.py [run_dirs ...]
```
Regenerates `summary_plots/trajectories.png` using movie-aligned colors and styling. Provide specific output directories or leave blank to update all folders containing `timeseries.csv`.

---

## 16. Patient-specific simulations
Use real patient compositions to seed the ABM. Provide the cell-density and metadata CSVs (see `data/`):
```bash
../.venv/bin/python scripts/run_patient_sims.py \
    --cells data/patient_celltype_broad_density_group.csv \
    --meta data/patient_celltype_broad_density_group.obs.csv \
    --preset luad_abm/config/G4_fibrotic.json \
    --config-dir luad_abm/config/patients \
    --outputs-dir luad_abm/outputs/patients \
    --group Group4 \
    --run-args '--movie-scale 8 --no-movie' \
    --run
```
- Densities (per mm²) are scaled by each ROI\_area to obtain agent counts.
- Counts are automatically rescaled if they exceed the 100×100 lattice capacity.
- Generated configs live in `luad_abm/config/patients/patient_<ID>.json`; metadata is embedded in the JSON.
- Omit `--run` to only generate configs; add `--patient` or another `--group` to filter cohorts.

`run_patient_sims.py` can be extended with additional mappings once more agent classes are introduced.
