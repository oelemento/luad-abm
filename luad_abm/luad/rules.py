"""Local interaction rules for the LUAD Mesa simulator."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def chemotactic_move(
    model,
    agent,
    field_name: str,
    step_probability: float,
    noise_scale: float = 0.01,
) -> None:
    """Move the agent one step toward higher chemokine while avoiding dense ECM."""
    if model.random.random() > step_probability:
        return

    neighbors = model.grid.get_neighborhood(agent.pos, moore=True, include_center=True)
    best_score = -np.inf
    best_pos = agent.pos

    for candidate in neighbors:
        if candidate == agent.pos:
            continue
        # Respect the "1 agent per site" design
        if not model.grid.is_cell_empty(candidate):
            continue
        chemokine = model.field_engine.field_value(field_name, candidate)
        ecm_penalty = model.field_engine.field_value("ecm", candidate) * model.params["ecm_penalty"]
        # Encourage proximity to tumor nests slightly for cytotoxic cells
        tumor_bias = 0.0
        if getattr(agent, "tumor_attraction", 0.0) > 0.0:
            tumor_bias = agent.tumor_attraction * local_tumor_density(model, candidate)
        score = chemokine - ecm_penalty + tumor_bias + model.random.uniform(-noise_scale, noise_scale)
        if score > best_score:
            best_score = score
            best_pos = candidate

    if best_pos != agent.pos:
        model.grid.move_agent(agent, best_pos)


def local_tumor_density(model, pos: Tuple[int, int]) -> float:
    """Return the fraction of neighbors occupied by tumor/epithelial cells."""
    neighbors = model.grid.get_neighbors(pos, moore=True, include_center=False)
    if not neighbors:
        return 0.0
    tumor_like = sum(1 for agent in neighbors if getattr(agent, "is_tumor", False))
    return tumor_like / len(neighbors)


def compute_cd8_kill_probability(model, killer, target, local_suppression: float) -> float:
    """Probability that a CD8 T cell kills a tumor cell in contact."""
    params = model.params
    base = params["cd8_base_kill"]
    mhc_factor = 0.5 + 0.5 * getattr(target, "mhc_i", 0.7)
    pdl1_penalty = 1.0 - params["pd_l1_penalty"] * getattr(target, "pd_l1", 0.2)
    exhaustion_penalty = 1.0 - killer.exhaustion
    effector_boost = 1.0 + killer.activation * params["cd8_activation_bonus"]

    prob = base * mhc_factor * pdl1_penalty * exhaustion_penalty * effector_boost
    prob *= (1.0 - local_suppression)

    if model.active_interventions.get("PD1", False):
        prob += params["pd1_kill_bonus"]

    return float(np.clip(prob, 0.0, 1.0))


def kill_target(model, killer, target) -> bool:
    """Remove the target from the grid and scheduler."""
    model.grid.remove_agent(target)
    target.remove()
    model.scheduler.remove(target)
    model.removed_agents.append(target)
    killer.recent_kills += 1
    return True


def compute_local_suppression(model, pos: Tuple[int, int]) -> float:
    """Aggregate Treg, macrophage, and background suppression in a Moore hood."""
    neighbors = model.grid.get_neighbors(pos, moore=True, include_center=True)
    treg_effect = 0.0
    macro_effect = 0.0
    for neighbor in neighbors:
        if getattr(neighbor, "is_treg", False):
            treg_effect += neighbor.suppression_strength
        if getattr(neighbor, "is_macrophage", False):
            macro_effect += neighbor.suppression_strength
    background = model.params["suppressive_background"]
    total = treg_effect + macro_effect + background
    return float(np.clip(total, 0.0, 0.95))


def update_exhaustion(killer, suppression: float, params: dict) -> None:
    killer.exhaustion = float(np.clip(
        killer.exhaustion + params["cd8_exhaustion_rate"] * (1.0 + suppression) - killer.activation * 0.01,
        0.0,
        1.0,
    ))
    if params.get("pd1_blockade", False):
        killer.exhaustion *= 0.5


def reinforce_activation(killer, params: dict) -> None:
    killer.activation = float(np.clip(killer.activation + params["cd8_activation_gain"], 0.0, 1.0))


def decay_activation(killer, params: dict) -> None:
    killer.activation = float(np.clip(killer.activation - params["cd8_activation_decay"], 0.0, 1.0))


def macrophage_polarization_step(macrophage, params: dict) -> None:
    bias = params.get("macrophage_bias", 0.0)
    drift = params.get("macrophage_drift", 0.02)
    macrophage.polarization = float(np.clip(
        macrophage.polarization + bias * 0.01 + macrophage.model.random.uniform(-drift, drift),
        -1.0,
        1.0,
    ))
    macrophage.suppression_strength = macrophage.model.params["macrophage_suppr_base"] * (1.0 - macrophage.polarization) * 0.5


def tls_quality_drift_step(tls_node, params: dict) -> None:
    drift = params.get("tls_quality_drift", -0.002)
    tls_node.quality = float(np.clip(tls_node.quality + drift, 0.0, 1.0))
    if params.get("tls_quality_floor") is not None:
        tls_node.quality = max(tls_node.quality, params["tls_quality_floor"])


def balanced_choice(sequence: Sequence[Tuple[int, int]], rng: np.random.Generator) -> Optional[Tuple[int, int]]:
    if not sequence:
        return None
    weights = np.linspace(1.0, 2.0, num=len(sequence))
    weights /= weights.sum()
    idx = rng.choice(len(sequence), p=weights)
    return sequence[int(idx)]
