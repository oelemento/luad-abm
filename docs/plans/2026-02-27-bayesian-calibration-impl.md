# Bayesian Calibration Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a proof-of-concept calibration pipeline that infers ABM parameters from Gaglia et al. mouse spatial CyCIF data and validates treatment predictions, producing a figure for the Sanofi iDEA-TECH proposal.

**Architecture:** Extract per-mouse summary statistics from Gaglia .mat files, define matching summary statistics from ABM simulations, run Latin Hypercube parameter search to find best-fit parameters for untreated mice, then validate by predicting ICB treatment response.

**Tech Stack:** Python 3.11, scipy.io / h5py (MATLAB file reading), numpy, pandas, matplotlib, pyDOE2 (Latin Hypercube), existing Mesa ABM framework.

---

### Task 1: Extract Gaglia Summary Statistics

**Files:**
- Create: `scripts/extract_gaglia_stats.py`

**Step 1: Write the extraction script**

This script reads Dataset03 .mat files and outputs per-mouse summary statistics.

```python
"""Extract per-mouse summary statistics from Gaglia et al. CyCIF data.

Usage:
    python scripts/extract_gaglia_stats.py \
        --dataset data/gaglia_2023/Dataset03_KP_LucOS_anti_PD1_CTLA4 \
        --out data/gaglia_2023/gaglia_summary_stats.csv
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio


CELL_TYPE_LEVEL = 3  # Level in hierarchy: 0=Immune/Epi/Other, 1=Lymphoid/Myeloid, 2=T/B/NK..., 3=Treg/Th/Tc
CELL_TYPES_OF_INTEREST = ["T cytotox", "T helper", "T reg", "B", "TAM", "Alveolar MAC", "DC", "Neutrophil", "NK_L", "NK_M"]
IMMUNE_TYPES = {"T cytotox", "T helper", "T reg", "B", "TAM", "Alveolar MAC", "DC", "Neutrophil", "NK_L", "NK_M"}
LYMPHOCYTE_TYPES = {"T cytotox", "T helper", "T reg", "B"}

# Distance bins in microns for infiltration profile
# Core: inside tumor (negative distance = inside), Cuff: 0-50µm outside, Periphery: >50µm outside
REGION_BINS = {"inside": (-np.inf, 0), "cuff": (0, 50), "periphery": (50, np.inf)}


def load_mat_or_hdf5(path: str, key: str):
    """Load a .mat file, falling back to HDF5 for v7.3 files."""
    try:
        return sio.loadmat(path)[key]
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            return f[key]


def extract_dataset(dataset_dir: Path) -> pd.DataFrame:
    quant = dataset_dir / "Quantification"

    # --- Load settings: mouse groups and IDs ---
    settings = sio.loadmat(str(quant / next(quant.glob("Results_Settings_*.mat"))))
    opts = settings["options"][0, 0]
    mouse_groups = opts["MouseGroup"].flatten()  # shape (n_mice,)
    mouse_nums = opts["MouseNum"].flatten()

    # --- Load morphology: X, Y per cell + sample index ---
    morp = sio.loadmat(str(quant / next(quant.glob("Results_Morp_*.mat"))))
    m = morp["MorpResults"][0, 0]
    x = m["X"].flatten().astype(np.float32)
    y = m["Y"].flatten().astype(np.float32)
    sample_idx = m["Indexes"].flatten().astype(np.int32)  # 1-based mouse index

    # --- Load cell types ---
    ct = sio.loadmat(str(quant / next(quant.glob("Results_CellType_*.mat"))))
    c = ct["CellType"][0, 0]
    type_names = [str(n[0]) for n in c["names"].flatten()]
    type_matrix = c["Matrix"]  # (n_cells, n_levels) — integer codes per level
    # Get level-3 (finest) type names
    level_codes = type_matrix[:, CELL_TYPE_LEVEL] if type_matrix.shape[1] > CELL_TYPE_LEVEL else type_matrix[:, -1]

    # Build code -> name mapping for finest level
    # codes are 1-based indices into the names list for that level
    # names list corresponds to all unique types at that level
    code_to_name = {}
    for i, name in enumerate(type_names):
        code_to_name[i + 1] = name
    cell_type_labels = np.array([code_to_name.get(int(c), "Unknown") for c in level_codes])

    # --- Load distance to tumor boundary ---
    dist_file = list(quant.glob("Results_RoiDist_*.mat"))
    has_distances = len(dist_file) > 0
    if has_distances:
        dist_mat = sio.loadmat(str(dist_file[0]))
        d = dist_mat["DistResults"][0, 0]
        # Tumor distance: negative = inside tumor, positive = outside
        tumor_dist = d["Tumor"].flatten().astype(np.float32)
    else:
        tumor_dist = np.zeros_like(x)

    # --- Load lymphonet data ---
    nets_file = list(quant.glob("Results_Nets_*_dist50.mat"))
    has_nets = len(nets_file) > 0
    if has_nets:
        nets = sio.loadmat(str(nets_file[0]))
        ln = nets["LymphoNets"][0, 0]
        net_ids = ln["NetworkID"].flatten().astype(np.int32)
        net_sizes = ln["Size"].flatten().astype(np.int32)
    else:
        net_ids = np.zeros(len(x), dtype=np.int32)
        net_sizes = np.zeros(len(x), dtype=np.int32)

    # --- Compute per-mouse stats ---
    n_cells = len(x)
    unique_mice = np.unique(sample_idx)
    rows = []

    for mouse_idx in unique_mice:
        mask = sample_idx == mouse_idx
        mouse_num = mouse_nums[mouse_idx - 1] if mouse_idx <= len(mouse_nums) else mouse_idx
        mouse_group = mouse_groups[mouse_idx - 1] if mouse_idx <= len(mouse_groups) else 0

        n = mask.sum()
        types_m = cell_type_labels[mask]
        dist_m = tumor_dist[mask] if has_distances else np.zeros(n)
        net_ids_m = net_ids[mask] if has_nets else np.zeros(n, dtype=np.int32)
        net_sizes_m = net_sizes[mask] if has_nets else np.zeros(n, dtype=np.int32)

        row = {"mouse_num": int(mouse_num), "mouse_group": int(mouse_group), "n_cells": int(n)}

        # Cell type fractions (among all cells)
        for ct_name in CELL_TYPES_OF_INTEREST:
            ct_count = (types_m == ct_name).sum()
            row[f"frac_{ct_name.replace(' ', '_').lower()}"] = ct_count / n if n > 0 else 0.0

        # Immune fraction
        immune_mask = np.isin(types_m, list(IMMUNE_TYPES))
        row["frac_immune"] = immune_mask.sum() / n if n > 0 else 0.0

        # Infiltration profile per immune cell type
        if has_distances:
            for ct_name in LYMPHOCYTE_TYPES:
                ct_mask = types_m == ct_name
                ct_dist = dist_m[ct_mask]
                ct_total = ct_mask.sum()
                if ct_total == 0:
                    for region in REGION_BINS:
                        row[f"infilt_{ct_name.replace(' ', '_').lower()}_{region}"] = 0.0
                else:
                    for region, (lo, hi) in REGION_BINS.items():
                        in_region = ((ct_dist >= lo) & (ct_dist < hi)).sum()
                        row[f"infilt_{ct_name.replace(' ', '_').lower()}_{region}"] = in_region / ct_total

        # CD8 distance histogram (bins: -100 to 200 µm, 10µm steps)
        cd8_mask = types_m == "T cytotox"
        cd8_dist = dist_m[cd8_mask]
        bins = np.arange(-100, 210, 10)
        if len(cd8_dist) > 0:
            hist, _ = np.histogram(cd8_dist, bins=bins, density=True)
            for i, h in enumerate(hist):
                row[f"cd8_dist_bin_{bins[i]:.0f}"] = h

        # Lymphonet stats
        if has_nets:
            lymph_mask = np.isin(types_m, list(LYMPHOCYTE_TYPES))
            lymph_net_ids = net_ids_m[lymph_mask]
            in_network = (lymph_net_ids > 0).sum()
            row["frac_lymph_in_network"] = in_network / lymph_mask.sum() if lymph_mask.sum() > 0 else 0.0
            unique_nets = np.unique(net_ids_m[net_ids_m > 0])
            if len(unique_nets) > 0:
                sizes = np.array([net_sizes_m[net_ids_m == nid][0] for nid in unique_nets if (net_ids_m == nid).any()])
                row["mean_network_size"] = float(sizes.mean()) if len(sizes) > 0 else 0.0
                row["n_networks"] = len(unique_nets)
            else:
                row["mean_network_size"] = 0.0
                row["n_networks"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    df = extract_dataset(Path(args.dataset))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Extracted {len(df)} mice -> {args.out}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMouse groups: {df['mouse_group'].value_counts().to_dict()}")
    print(f"\nSample stats for Tc fraction:")
    for g in sorted(df["mouse_group"].unique()):
        sub = df[df["mouse_group"] == g]
        col = "frac_t_cytotox"
        if col in sub.columns:
            print(f"  Group {g}: mean={sub[col].mean():.4f}, std={sub[col].std():.4f}")


if __name__ == "__main__":
    main()
```

**Step 2: Run the extraction**

Run:
```bash
cd /Users/ole2001/PROGRAMS/LungCancerSim2
python3.11 scripts/extract_gaglia_stats.py \
    --dataset data/gaglia_2023/Dataset03_KP_LucOS_anti_PD1_CTLA4 \
    --out data/gaglia_2023/gaglia_summary_stats.csv
```

Expected: CSV with 24 rows (one per mouse), columns for cell fractions, infiltration profiles, distance histograms, lymphonet stats. Print summary of group means for Tc fraction.

**Step 3: Inspect output, verify group structure**

Visually confirm: Group 1 = Ctrl has lower Tc infiltration into tumor than Group 3-4 = ICB (consistent with paper).

**Step 4: Commit**

```bash
git add scripts/extract_gaglia_stats.py
git commit -m "feat: add Gaglia CyCIF summary stat extraction for calibration"
```

---

### Task 2: ABM Calibration Module

**Files:**
- Create: `luad_abm/luad/calibration.py`

**Step 1: Write the calibration module**

```python
"""Summary statistics and distance metrics for ABM calibration."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .agents import AgentType


REGION_THRESHOLDS = {"core": 0.9, "cuff": 1.3}  # fractions of tumor_radius
CELL_TYPE_MAP = {
    AgentType.CD8: "t_cytotox",
    AgentType.CD4: "t_helper",
    AgentType.TREG: "t_reg",
    AgentType.MACROPHAGE: "macrophage",
    AgentType.TLS: "tls",
    AgentType.CAF: "caf",
    AgentType.TUMOR: "tumor",
}


def extract_summary_stats(model) -> Dict[str, float]:
    """Extract summary statistics from a completed ABM simulation.

    Returns dict with keys matching Gaglia extraction where possible.
    """
    stats = {}
    width, height = model.grid.width, model.grid.height
    center = np.array([width / 2.0, height / 2.0])

    # --- Cell counts and fractions ---
    total = 0
    counts = {}
    positions = {}
    for at in AgentType:
        agents = list(model.iter_agents(at))
        key = CELL_TYPE_MAP.get(at, at.name.lower())
        counts[key] = len(agents)
        positions[key] = np.array([a.pos for a in agents], dtype=np.float32) if agents else np.empty((0, 2))
        total += len(agents)

    non_tumor = total - counts.get("tumor", 0)
    for key, count in counts.items():
        stats[f"frac_{key}"] = count / total if total > 0 else 0.0

    # --- Infiltration profile (core/cuff/periphery) ---
    tumor_pos = positions.get("tumor", np.empty((0, 2)))
    if len(tumor_pos) > 0:
        tumor_center = tumor_pos.mean(axis=0)
        tumor_dists = np.sqrt(((tumor_pos - tumor_center) ** 2).sum(axis=1))
        tumor_radius = np.percentile(tumor_dists, 90) if len(tumor_dists) > 5 else 15.0
    else:
        tumor_center = center
        tumor_radius = 15.0

    core_r = tumor_radius * REGION_THRESHOLDS["core"]
    cuff_r = tumor_radius * REGION_THRESHOLDS["cuff"]

    for at_key in ["t_cytotox", "t_helper", "t_reg"]:
        pos = positions.get(at_key, np.empty((0, 2)))
        n = len(pos)
        if n == 0:
            for region in ["inside", "cuff", "periphery"]:
                stats[f"infilt_{at_key}_{region}"] = 0.0
            continue
        dists = np.sqrt(((pos - tumor_center) ** 2).sum(axis=1))
        n_inside = (dists <= core_r).sum()
        n_cuff = ((dists > core_r) & (dists <= cuff_r)).sum()
        n_periph = (dists > cuff_r).sum()
        stats[f"infilt_{at_key}_inside"] = n_inside / n
        stats[f"infilt_{at_key}_cuff"] = n_cuff / n
        stats[f"infilt_{at_key}_periphery"] = n_periph / n

    # --- CD8-tumor distance distribution ---
    cd8_pos = positions.get("t_cytotox", np.empty((0, 2)))
    if len(cd8_pos) > 0 and len(tumor_pos) > 0:
        from scipy.spatial.distance import cdist
        dists = cdist(cd8_pos, tumor_pos).min(axis=1)
        stats["cd8_mean_dist_to_tumor"] = float(dists.mean())
        stats["cd8_median_dist_to_tumor"] = float(np.median(dists))
        stats["cd8_frac_within_5"] = float((dists <= 5).mean())
    else:
        stats["cd8_mean_dist_to_tumor"] = 50.0
        stats["cd8_median_dist_to_tumor"] = 50.0
        stats["cd8_frac_within_5"] = 0.0

    # --- Tumor count trajectory endpoint ---
    stats["final_tumor_count"] = counts.get("tumor", 0)
    stats["final_cd8_count"] = counts.get("t_cytotox", 0)

    return stats


def compute_distance(obs: Dict[str, float], sim: Dict[str, float],
                     obs_var: Dict[str, float], keys: list[str] | None = None) -> float:
    """Weighted sum of squared differences between observed and simulated stats.

    d = Σ (obs_i - sim_i)² / var_i for each shared key.
    """
    if keys is None:
        keys = [k for k in obs if k in sim and k in obs_var]
    total = 0.0
    n = 0
    for k in keys:
        if k not in obs or k not in sim or k not in obs_var:
            continue
        var = obs_var[k]
        if var < 1e-10:
            continue
        total += (obs[k] - sim[k]) ** 2 / var
        n += 1
    return total / n if n > 0 else float("inf")
```

**Step 2: Commit**

```bash
git add luad_abm/luad/calibration.py
git commit -m "feat: add calibration module with summary stats and distance metric"
```

---

### Task 3: Parameter Search Script

**Files:**
- Create: `scripts/calibration_search.py`

**Step 1: Write the search script**

```python
"""Latin Hypercube parameter search for ABM calibration against Gaglia data.

Usage:
    python scripts/calibration_search.py \
        --obs data/gaglia_2023/gaglia_summary_stats.csv \
        --preset luad_abm/config/G3_inflammatory.json \
        --n-samples 200 \
        --ticks 500 \
        --seeds 2 \
        --out data/calibration_results.csv \
        --workers 4
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pyDOE2 import lhs

ROOT = Path(__file__).resolve().parents[0] / ".." / "luad_abm"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luad.model import LUADModel, load_preset, PresetConfig
from luad.calibration import extract_summary_stats, compute_distance
from luad.metrics import MetricsTracker

# Parameter search space
PARAM_SPACE = {
    "cd8_base_kill":          (0.05, 0.30),
    "cd8_exhaustion_rate":    (0.02, 0.12),
    "pd_l1_penalty":          (0.30, 0.90),
    "macrophage_suppr_base":  (0.10, 0.50),
    "suppressive_background": (0.02, 0.15),
    "tumor_proliferation_rate": (0.01, 0.06),
}

# Keys to use for distance computation
CALIBRATION_KEYS = [
    "frac_t_cytotox", "frac_t_helper", "frac_t_reg", "frac_macrophage",
    "infilt_t_cytotox_inside", "infilt_t_cytotox_cuff", "infilt_t_cytotox_periphery",
    "infilt_t_helper_inside", "infilt_t_helper_cuff", "infilt_t_helper_periphery",
    "infilt_t_reg_inside", "infilt_t_reg_cuff", "infilt_t_reg_periphery",
    "cd8_mean_dist_to_tumor", "cd8_frac_within_5",
    "final_tumor_count",
]


def generate_lhs_samples(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Generate Latin Hypercube samples scaled to parameter ranges."""
    names = list(PARAM_SPACE.keys())
    ranges = list(PARAM_SPACE.values())
    rng = np.random.default_rng(seed)
    raw = lhs(len(names), samples=n_samples, random_state=rng.integers(0, 2**31))
    scaled = np.zeros_like(raw)
    for i, (lo, hi) in enumerate(ranges):
        scaled[:, i] = lo + raw[:, i] * (hi - lo)
    return pd.DataFrame(scaled, columns=names)


def run_single(preset_path: str, param_overrides: dict, ticks: int, seed: int,
               interventions: list | None = None) -> dict:
    """Run one ABM simulation and return summary stats."""
    preset = load_preset(Path(preset_path))
    tracker = MetricsTracker(capture_grids=False, distance_interval=ticks + 1,
                             interaction_interval=ticks + 1)
    model = LUADModel(preset=preset, interventions=interventions or [],
                      seed=seed, metrics_tracker=tracker)
    # Override params
    for k, v in param_overrides.items():
        model.params[k] = v
    # Also update preset-derived params that depend on overrides
    if "macrophage_suppr_base" in param_overrides:
        for agent in model.iter_agents(model.scheduler.agents.__class__):
            pass  # Macrophage suppression updates dynamically via rules
    for _ in range(ticks):
        model.step()
    return extract_summary_stats(model)


def compute_obs_targets(obs_df: pd.DataFrame, group: int) -> tuple[dict, dict]:
    """Compute mean and variance of observed stats for a mouse group."""
    sub = obs_df[obs_df["mouse_group"] == group]
    means = {}
    variances = {}
    for col in sub.columns:
        if col in ("mouse_num", "mouse_group", "n_cells"):
            continue
        vals = sub[col].dropna().values
        if len(vals) > 0:
            means[col] = float(vals.mean())
            variances[col] = float(vals.var()) if len(vals) > 1 else float(vals.mean() ** 2 * 0.1 + 1e-6)
    return means, variances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs", required=True, help="Gaglia summary stats CSV")
    parser.add_argument("--preset", required=True, help="Base preset JSON")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=2, help="Replicates per param set")
    parser.add_argument("--out", required=True, help="Output CSV")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ctrl-group", type=int, default=1, help="Mouse group for control")
    args = parser.parse_args()

    obs_df = pd.read_csv(args.obs)
    obs_mean, obs_var = compute_obs_targets(obs_df, args.ctrl_group)

    samples = generate_lhs_samples(args.n_samples)
    print(f"Running {args.n_samples} parameter sets × {args.seeds} seeds = {args.n_samples * args.seeds} simulations")

    results = []
    total = args.n_samples * args.seeds

    def _run_one(idx, seed, overrides):
        stats = run_single(args.preset, overrides, args.ticks, seed)
        dist = compute_distance(obs_mean, stats, obs_var, CALIBRATION_KEYS)
        return {**overrides, "seed": seed, "sample_idx": idx, "distance": dist, **{f"sim_{k}": v for k, v in stats.items()}}

    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for idx, row in samples.iterrows():
            overrides = row.to_dict()
            for s in range(args.seeds):
                seed = 1000 * idx + s
                futures.append(pool.submit(_run_one, idx, seed, overrides))

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                if completed % 20 == 0:
                    print(f"  {completed}/{total} done")
            except Exception as e:
                print(f"  Error: {e}")
                completed += 1

    df = pd.DataFrame(results)
    df.sort_values("distance", inplace=True)
    df.to_csv(args.out, index=False)
    print(f"\nResults saved to {args.out}")
    print(f"Best distance: {df['distance'].iloc[0]:.4f}")
    print(f"Best params:")
    for k in PARAM_SPACE:
        print(f"  {k}: {df[k].iloc[0]:.4f}")


if __name__ == "__main__":
    main()
```

**Step 2: Install pyDOE2 if needed**

Run: `pip install pyDOE2`

**Step 3: Run a small test (10 samples, 1 seed) to verify pipeline**

```bash
cd /Users/ole2001/PROGRAMS/LungCancerSim2
python3.11 scripts/calibration_search.py \
    --obs data/gaglia_2023/gaglia_summary_stats.csv \
    --preset luad_abm/config/G3_inflammatory.json \
    --n-samples 10 --ticks 200 --seeds 1 --workers 2 \
    --out data/calibration_test.csv
```

Expected: CSV with 10 rows, each with parameter values, distance score, and simulated stats. Should complete in ~5 minutes.

**Step 4: Commit**

```bash
git add scripts/calibration_search.py luad_abm/luad/calibration.py
git commit -m "feat: add LHS parameter search for ABM calibration"
```

---

### Task 4: Run Full Calibration

**Step 1: Run baseline calibration (200 samples × 2 seeds)**

```bash
python3.11 scripts/calibration_search.py \
    --obs data/gaglia_2023/gaglia_summary_stats.csv \
    --preset luad_abm/config/G3_inflammatory.json \
    --n-samples 200 --ticks 500 --seeds 2 --workers 4 \
    --out data/calibration_results_baseline.csv
```

Expected runtime: ~2.5 hours with 4 workers. If too slow, reduce to 100 samples or submit to Cayuga.

**Step 2: Inspect results**

```bash
python3.11 -c "
import pandas as pd
df = pd.read_csv('data/calibration_results_baseline.csv')
print('Distance distribution:')
print(df['distance'].describe())
print('\nTop 10 parameter sets:')
top = df.nsmallest(10, 'distance')
for col in ['cd8_base_kill','cd8_exhaustion_rate','pd_l1_penalty','macrophage_suppr_base','suppressive_background','tumor_proliferation_rate']:
    print(f'  {col}: {top[col].mean():.3f} ± {top[col].std():.3f}')
"
```

**Step 3: Run treatment validation using best baseline params + PD1**

This will be a separate script invocation using the top-10 baseline parameter sets with PD1 intervention enabled, comparing against Gaglia ICB groups.

---

### Task 5: Proposal Figure

**Files:**
- Create: `scripts/plot_calibration_results.py`

**Step 1: Write the plotting script**

```python
"""Generate multi-panel calibration figure for Sanofi proposal.

Usage:
    python scripts/plot_calibration_results.py \
        --obs data/gaglia_2023/gaglia_summary_stats.csv \
        --results data/calibration_results_baseline.csv \
        --out luad_abm/summary_plots/calibration_figure.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARAM_NAMES = {
    "cd8_base_kill": "CD8 kill rate",
    "cd8_exhaustion_rate": "Exhaustion rate",
    "pd_l1_penalty": "PD-L1 penalty",
    "macrophage_suppr_base": "Macrophage suppression",
    "suppressive_background": "Background suppression",
    "tumor_proliferation_rate": "Tumor proliferation",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-frac", type=float, default=0.1, help="Fraction of best fits to show")
    args = parser.parse_args()

    obs = pd.read_csv(args.obs)
    res = pd.read_csv(args.results)

    top_n = max(5, int(len(res) * args.top_frac))
    top = res.nsmallest(top_n, "distance")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Panel A: Observed infiltration profiles by group ---
    ax = axes[0, 0]
    groups = sorted(obs["mouse_group"].unique())
    group_labels = {1: "Ctrl", 2: "anti-PD1", 3: "anti-CTLA4", 4: "Combo"}
    colors = {1: "#4477AA", 2: "#EE6677", 3: "#228833", 4: "#CCBB44"}
    cell_types = ["t_cytotox", "t_helper", "t_reg"]
    regions = ["inside", "cuff", "periphery"]
    x_pos = np.arange(len(cell_types) * len(regions))
    bar_width = 0.8 / len(groups)
    for gi, g in enumerate(groups):
        sub = obs[obs["mouse_group"] == g]
        vals = []
        for ct in cell_types:
            for reg in regions:
                col = f"infilt_{ct}_{reg}"
                vals.append(sub[col].mean() if col in sub.columns else 0)
        ax.bar(x_pos + gi * bar_width, vals, bar_width, label=group_labels.get(g, f"G{g}"),
               color=colors.get(g, "gray"), alpha=0.8)
    labels = [f"{ct.split('_')[-1]}\n{reg[:3]}" for ct in cell_types for reg in regions]
    ax.set_xticks(x_pos + bar_width * (len(groups) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Fraction of cell type")
    ax.set_title("A. Observed infiltration (Gaglia)")
    ax.legend(fontsize=8)

    # --- Panel B: Inferred parameter distributions ---
    ax = axes[0, 1]
    param_keys = list(PARAM_NAMES.keys())
    positions = np.arange(len(param_keys))
    # Normalize to [0,1] range for display
    for i, pk in enumerate(param_keys):
        vals = top[pk].values
        parts = ax.violinplot([vals], positions=[i], showmeans=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4477AA")
            pc.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([PARAM_NAMES[k] for k in param_keys], rotation=45, ha="right", fontsize=8)
    ax.set_title("B. Inferred parameters (top 10%)")

    # --- Panel C: Observed vs simulated cell fractions ---
    ax = axes[1, 0]
    ctrl_obs = obs[obs["mouse_group"] == 1]
    frac_cols = [c for c in obs.columns if c.startswith("frac_") and c in top.columns.str.replace("sim_", "")]
    sim_frac_cols = [f"sim_{c}" for c in frac_cols if f"sim_{c}" in top.columns]
    if frac_cols and sim_frac_cols:
        obs_means = [ctrl_obs[c].mean() for c in frac_cols[:6]]
        sim_means = [top[f"sim_{c}"].mean() for c in frac_cols[:6]]
        x = np.arange(len(frac_cols[:6]))
        ax.bar(x - 0.2, obs_means, 0.35, label="Observed (Gaglia)", color="#4477AA")
        ax.bar(x + 0.2, sim_means, 0.35, label="Simulated (ABM)", color="#EE6677")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("frac_", "") for c in frac_cols[:6]], rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
    ax.set_ylabel("Fraction")
    ax.set_title("C. Observed vs simulated composition")

    # --- Panel D: Distance score distribution ---
    ax = axes[1, 1]
    ax.hist(res["distance"].values, bins=30, color="#4477AA", alpha=0.7, edgecolor="white")
    ax.axvline(top["distance"].max(), color="#EE6677", linestyle="--", label=f"Top {args.top_frac*100:.0f}% threshold")
    ax.set_xlabel("Distance score")
    ax.set_ylabel("Count")
    ax.set_title("D. Parameter search scores")
    ax.legend(fontsize=8)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
```

**Step 2: Generate figure**

```bash
python3.11 scripts/plot_calibration_results.py \
    --obs data/gaglia_2023/gaglia_summary_stats.csv \
    --results data/calibration_results_baseline.csv \
    --out luad_abm/summary_plots/calibration_figure.png
open -a Preview luad_abm/summary_plots/calibration_figure.png
```

**Step 3: Iterate on figure aesthetics as needed**

**Step 4: Commit**

```bash
git add scripts/plot_calibration_results.py luad_abm/summary_plots/calibration_figure.png
git commit -m "feat: add calibration figure for Sanofi proposal"
```
