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
    settings = sio.loadmat(str(next(quant.glob("Results_Settings_*.mat"))))
    opts = settings["options"][0, 0]
    mouse_groups = opts["MouseGroup"].flatten()  # shape (n_mice,)
    mouse_nums = opts["MouseNum"].flatten()

    # --- Load morphology: X, Y per cell + sample index ---
    morp = sio.loadmat(str(next(quant.glob("Results_Morp_*.mat"))))
    m = morp["MorpResults"][0, 0]
    x = m["X"].flatten().astype(np.float32)
    y = m["Y"].flatten().astype(np.float32)
    sample_idx = m["Indexes"].flatten().astype(np.int32)  # 1-based mouse index

    # --- Load cell types ---
    ct = sio.loadmat(str(next(quant.glob("Results_CellType_*.mat"))))
    c = ct["CellType"][0, 0]
    type_matrix = c["Matrix"]  # (n_cells, n_layers) — hierarchical integer codes

    # Build code -> name mapping from the codes and names arrays
    all_codes = c["codes"].flatten()
    all_names = [str(n[0]) for n in c["names"].flatten()]
    code_to_name = {int(code): name for code, name in zip(all_codes, all_names)}
    code_to_name[0] = "Unknown"

    # Assign finest type: use deepest non-zero layer per cell
    # Matrix cols are layers 1..4; use last non-zero code
    n_layers = type_matrix.shape[1]
    finest_codes = np.zeros(type_matrix.shape[0], dtype=np.int32)
    for col in range(n_layers):
        nonzero = type_matrix[:, col] != 0
        finest_codes[nonzero] = type_matrix[nonzero, col]

    cell_type_labels = np.array([code_to_name.get(int(c), "Unknown") for c in finest_codes])

    # --- Load distance to tumor boundary ---
    dist_file = list(quant.glob("Results_RoiDist_*.mat"))
    has_distances = len(dist_file) > 0
    if has_distances:
        dist_mat = sio.loadmat(str(dist_file[0]))
        d = dist_mat["DistResults"][0, 0]
        # Tumor distance: col 0 = indicator, col 1 = unsigned dist, col 2 = signed dist
        # Negative = inside tumor, positive = outside
        tumor_dist = d["Tumor"][:, 2].astype(np.float32)
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
