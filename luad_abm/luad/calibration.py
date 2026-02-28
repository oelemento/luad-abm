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

    # Shared-type fractions: among {CD8, CD4, Treg, Macrophage} only
    # This ensures identical denominators between ABM and Gaglia
    shared_keys = ["t_cytotox", "t_helper", "t_reg", "macrophage"]
    shared_total = sum(counts.get(k, 0) for k in shared_keys)
    for key in shared_keys:
        stats[f"frac_{key}"] = counts[key] / shared_total if shared_total > 0 else 0.0

    # Ratios (denominator-independent)
    cd8 = counts.get("t_cytotox", 0)
    cd4 = counts.get("t_helper", 0)
    treg = counts.get("t_reg", 0)
    mac = counts.get("macrophage", 0)
    stats["ratio_cd8_cd4"] = cd8 / cd4 if cd4 > 0 else 0.0
    stats["ratio_cd8_treg"] = cd8 / treg if treg > 0 else 0.0
    stats["ratio_cd4_treg"] = cd4 / treg if treg > 0 else 0.0
    stats["ratio_cd8_mac"] = cd8 / mac if mac > 0 else 0.0

    stats["frac_tumor"] = counts.get("tumor", 0) / total if total > 0 else 0.0

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

    d = sum (obs_i - sim_i)^2 / var_i for each shared key.
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
