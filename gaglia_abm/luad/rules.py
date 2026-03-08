"""Local interaction rules for the Gaglia-grounded LUAD model.

Simplified: no ECM penalty, no TLS drift, macrophage polarization
driven by tumor vs T-cell proximity instead of TGF-beta.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def chemotactic_move(
    model,
    agent,
    step_probability: float,
    noise_scale: float = 0.01,
) -> None:
    """Move agent one step toward higher CXCL9/10."""
    if model.random.random() > step_probability:
        return

    neighbors = model.grid.get_neighborhood(agent.pos, moore=True, include_center=True)
    best_score = -np.inf
    best_pos = agent.pos

    for candidate in neighbors:
        if candidate == agent.pos:
            continue
        if not model.grid.is_cell_empty(candidate):
            continue
        chemokine = model.field_engine.field_value(candidate)
        tumor_bias = 0.0
        if getattr(agent, "tumor_attraction", 0.0) > 0.0:
            tumor_bias = agent.tumor_attraction * local_tumor_density(model, candidate)
        score = chemokine + tumor_bias + model.random.uniform(-noise_scale, noise_scale)
        if score > best_score:
            best_score = score
            best_pos = candidate

    if best_pos != agent.pos:
        model.grid.move_agent(agent, best_pos)


def local_tumor_density(model, pos: Tuple[int, int]) -> float:
    neighbors = model.grid.get_neighbors(pos, moore=True, include_center=False)
    if not neighbors:
        return 0.0
    tumor_like = sum(1 for agent in neighbors if getattr(agent, "is_tumor", False))
    return tumor_like / len(neighbors)


def local_tcell_density(model, pos: Tuple[int, int]) -> float:
    """Fraction of neighbors that are CD8 or CD4 T cells."""
    from .agents import AgentType
    neighbors = model.grid.get_neighbors(pos, moore=True, include_center=False)
    if not neighbors:
        return 0.0
    t_cells = sum(1 for a in neighbors if getattr(a, "agent_type", None) in (AgentType.CD8, AgentType.CD4))
    return t_cells / len(neighbors)


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
    """Remove target; optionally trigger CD8 clonal expansion into cleared spot."""
    killed_pos = target.pos
    model.grid.remove_agent(target)
    target.remove()
    model.scheduler.remove(target)
    model.removed_agents.append(target)
    killer.recent_kills += 1

    # Antigen-driven CD8 proliferation: divide into the cleared grid square
    kill_prolif = model.params.get("cd8_kill_prolif_prob", 0.0)
    if kill_prolif > 0.0 and model.grid.is_cell_empty(killed_pos):
        if model.random.random() < kill_prolif:
            from gaglia_abm.luad.agents import CD8TCell
            daughter = CD8TCell(model, activation=killer.activation,
                                exhaustion=killer.exhaustion * 0.5)
            model.grid.place_agent(daughter, killed_pos)
            model.scheduler.add(daughter)

    return True


def compute_local_suppression(model, pos: Tuple[int, int]) -> float:
    """Aggregate Treg + M2 macrophage + background suppression in Moore hood."""
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
    increment = params["cd8_exhaustion_rate"] * (1.0 + suppression)
    if params.get("pd1_blockade", False):
        increment *= 0.7
    killer.exhaustion = float(np.clip(
        killer.exhaustion + increment - killer.activation * 0.01,
        0.0, 1.0,
    ))


def reinforce_activation(killer, params: dict) -> None:
    killer.activation = float(np.clip(killer.activation + params["cd8_activation_gain"], 0.0, 1.0))


def decay_activation(killer, params: dict) -> None:
    killer.activation = float(np.clip(killer.activation - params["cd8_activation_decay"], 0.0, 1.0))


def remove_immune_agent(model, agent) -> None:
    """Remove an immune cell from the simulation (death / emigration)."""
    model.grid.remove_agent(agent)
    agent.remove()
    model.scheduler.remove(agent)


def macrophage_polarization_step(macrophage, params: dict) -> None:
    """Polarize based on local tumor vs T-cell density (no TGF-beta)."""
    model = macrophage.model
    tumor_dens = local_tumor_density(model, macrophage.pos)
    tcell_dens = local_tcell_density(model, macrophage.pos)

    # More tumor → M2 (negative), more T cells → M1 (positive)
    bias = params.get("macrophage_bias", 0.0)
    drift = params.get("macrophage_drift", 0.02)
    signal = (tcell_dens - tumor_dens) * 0.1 + bias * 0.01
    noise = model.random.uniform(-drift, drift)

    macrophage.polarization = float(np.clip(macrophage.polarization + signal + noise, -1.0, 1.0))
    macrophage.suppression_strength = params["macrophage_suppr_base"] * (1.0 - macrophage.polarization) * 0.5
