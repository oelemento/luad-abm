"""Mesa model for the Gaglia-grounded LUAD simulator.

5 agent types, 1 chemokine field, 2 interventions (PD1, CTLA4).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import mesa
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

from .scheduler import StagedScheduler
import numpy as np

from . import agents
from .agents import AgentType
from .fields import FieldEngine, FieldParameters, default_field_parameters
from .metrics import MetricsTracker, count_agents

REGION_SEQUENCE = ("core", "cuff", "periphery")

BASE_INFILTRATION = {
    "cd8": {"core": 0.16, "cuff": 0.46, "periphery": 0.38},
    "cd4": {"core": 0.16, "cuff": 0.44, "periphery": 0.40},
    "treg": {"core": 0.32, "cuff": 0.44, "periphery": 0.24},
    "macrophage": {"core": 0.22, "cuff": 0.44, "periphery": 0.34},
    "__fallback__": {"core": 0.15, "cuff": 0.45, "periphery": 0.40},
}


def default_params() -> Dict[str, float]:
    return {
        "cd8_base_kill": 0.15,
        "pd_l1_penalty": 0.6,
        "cd8_activation_bonus": 0.25,
        "cd8_exhaustion_rate": 0.05,
        "cd8_activation_gain": 0.12,
        "cd8_activation_decay": 0.015,
        "pd1_kill_bonus": 0.02,
        "pd1_blockade": False,
        "tumor_proliferation_rate": 0.03,
        "pd_l1_induction_rate": 0.01,
        "tumor_cxcl9_emission": 0.05,
        "macrophage_suppr_base": 0.25,
        "macrophage_bias": -0.1,
        "macrophage_drift": 0.05,
        "treg_mod_factor": 1.0,
        "suppressive_background": 0.05,
        "immune_base_death_rate": 0.005,
        "cd8_exhaustion_death_bonus": 0.015,
        "recruitment_rate": 0.005,
        "recruitment_chemokine_boost": 0.5,
        "treg_prolif_rate": 0.02,
        "mac_tumor_death_rate": 0.06,
        "mac_recruit_suppression": 2.0,
        "recruit_exhaustion_priming": 0.3,
    }


@dataclass
class PresetConfig:
    name: str
    grid_width: int
    grid_height: int
    initial_agents: Dict[str, int]
    cxcl9_10_mean: float
    macrophage_bias: float
    macrophage_m2_fraction: float
    suppression: Dict[str, float]
    suppressive_background: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_preset(path: Path) -> PresetConfig:
    payload = json.loads(path.read_text())
    grid = payload.get("grid", {})
    macrophage = payload.get("macrophage", {})
    suppression = payload.get("suppression", {})

    return PresetConfig(
        name=payload.get("name", path.stem),
        grid_width=grid.get("width", 100),
        grid_height=grid.get("height", 100),
        initial_agents=payload.get("initial_agents", {}),
        cxcl9_10_mean=payload.get("cxcl9_10_mean", 0.1),
        macrophage_bias=macrophage.get("polarization_bias", -0.1),
        macrophage_m2_fraction=macrophage.get("initial_m2_fraction", 0.5),
        suppression={
            "treg": suppression.get("treg", 0.25),
            "macrophage": suppression.get("macrophage", 0.2),
        },
        suppressive_background=payload.get("suppressive_background", 0.05),
        metadata=payload.get("metadata", {}),
    )


class LUADModel(mesa.Model):
    def __init__(
        self,
        preset: PresetConfig,
        interventions: Sequence[str] | None = None,
        seed: int | None = None,
        field_params: FieldParameters | None = None,
        metrics_tracker: MetricsTracker | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.preset = preset
        self.step_count = 0
        self.params = default_params()
        self.params["macrophage_suppr_base"] = preset.suppression["macrophage"]
        self.params["suppressive_background"] = preset.suppressive_background
        self.params["macrophage_bias"] = preset.macrophage_bias

        self.active_interventions: Dict[str, bool] = {}
        self.apply_interventions(interventions or [])

        self.grid = MultiGrid(preset.grid_width, preset.grid_height, torus=False)
        self.scheduler = StagedScheduler(
            self,
            stage_list=("movement", "interactions", "state_updates"),
            shuffle=True,
        )
        self.removed_agents: List[mesa.Agent] = []

        self.np_rng = np.random.default_rng(seed)
        self.metrics_rng = np.random.default_rng(None if seed is None else seed + 1234)

        self.field_engine = FieldEngine(
            preset.grid_width,
            preset.grid_height,
            field_params or default_field_parameters(),
            self.np_rng,
        )

        self.ifng_signal = np.zeros((preset.grid_height, preset.grid_width), dtype=np.float32)

        self.metrics_tracker = metrics_tracker or MetricsTracker()

        self.datacollector = DataCollector(
            model_reporters={
                "tumor_count": lambda m: count_agents(m, AgentType.TUMOR),
                "cd8_count": lambda m: count_agents(m, AgentType.CD8),
                "cd4_count": lambda m: count_agents(m, AgentType.CD4),
                "treg_count": lambda m: count_agents(m, AgentType.TREG),
                "macrophage_count": lambda m: count_agents(m, AgentType.MACROPHAGE),
            }
        )

        self._place_initial_agents()
        self.datacollector.collect(self)

    def iter_agents(self, agent_type: AgentType) -> Iterator[agents.LUADBaseAgent]:
        for agent in self.scheduler.agents:
            if getattr(agent, "agent_type", None) == agent_type:
                yield agent

    def iter_all_agents(self) -> Iterator[agents.LUADBaseAgent]:
        yield from list(self.scheduler.agents)

    def _place_initial_agents(self) -> None:
        preset = self.preset
        counts = preset.initial_agents
        width, height = self.grid.width, self.grid.height
        center = np.array([width / 2.0, height / 2.0])

        yy, xx = np.mgrid[0:height, 0:width]
        dist_map = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
        available_mask = np.ones((height, width), dtype=bool)

        tumor_total = counts.get("tumor", 0)
        tumor_radius = self._estimate_cluster_radius(tumor_total)

        def pick_positions(mask: np.ndarray, count: int, weights: np.ndarray | None = None) -> List[tuple[int, int]]:
            if count <= 0:
                return []
            candidate_mask = mask & available_mask
            coords = np.argwhere(candidate_mask)
            if coords.size == 0:
                return []
            choose = min(count, len(coords))
            probs = None
            if weights is not None:
                sampled_weights = weights[candidate_mask]
                sampled_weights = np.clip(sampled_weights, 0.0, None)
                if sampled_weights.sum() > 0:
                    probs = sampled_weights / sampled_weights.sum()
            if probs is not None:
                idx = self.np_rng.choice(len(coords), size=choose, replace=False, p=probs)
            else:
                idx = self.np_rng.choice(len(coords), size=choose, replace=False)
            selected: List[tuple[int, int]] = []
            for i in idx:
                y, x = coords[i]
                available_mask[y, x] = False
                selected.append((int(x), int(y)))
            return selected

        def place_agents(agent_type: AgentType, positions: List[tuple[int, int]]) -> List[tuple[int, int]]:
            placed: List[tuple[int, int]] = []
            for pos in positions:
                if not self.grid.is_cell_empty(pos):
                    continue
                agent = self._build_agent(agent_type, pos)
                self.grid.place_agent(agent, pos)
                self.scheduler.add(agent)
                placed.append(pos)
            return placed

        # Place tumor in a central cluster
        sigma = max(4.0, tumor_radius * 0.45)
        tumor_weights = np.exp(-(dist_map ** 2) / (2 * sigma ** 2))
        tumor_weights += self.np_rng.normal(0.0, 0.02, size=tumor_weights.shape)
        tumor_weights = np.clip(tumor_weights, 0.0, None)
        primary_mask = dist_map <= tumor_radius * 1.05
        tumor_positions = pick_positions(primary_mask, tumor_total, tumor_weights)
        if len(tumor_positions) < tumor_total:
            expanded_mask = dist_map <= tumor_radius * 1.25
            remaining = tumor_total - len(tumor_positions)
            additional = pick_positions(expanded_mask, remaining, tumor_weights)
            tumor_positions.extend(additional)
        if len(tumor_positions) < tumor_total:
            fallback = pick_positions(np.ones_like(tumor_weights, dtype=bool), tumor_total - len(tumor_positions), tumor_weights)
            tumor_positions.extend(fallback)
        tumor_coords = place_agents(AgentType.TUMOR, tumor_positions)

        # Define infiltration regions
        core_radius = tumor_radius * 0.9
        cuff_outer = tumor_radius * 1.3
        region_masks = {
            "core": dist_map <= core_radius,
            "cuff": (dist_map > core_radius) & (dist_map <= cuff_outer),
            "periphery": dist_map > cuff_outer,
        }

        def place_with_profile(agent_key: str, agent_type: AgentType, total: int) -> None:
            if total <= 0:
                return
            profile = BASE_INFILTRATION.get(agent_key, BASE_INFILTRATION["__fallback__"]).copy()
            fractions = np.array([profile.get(region, 0.0) for region in REGION_SEQUENCE], dtype=float)
            if fractions.sum() <= 0:
                fractions = np.array([0.0, 0.3, 0.7], dtype=float)
            fractions /= fractions.sum()
            ideal = fractions * total
            counts_per_region = np.floor(ideal).astype(int)
            remainder = total - counts_per_region.sum()
            if remainder > 0:
                order = np.argsort(ideal - counts_per_region)[::-1]
                for idx in order[:remainder]:
                    counts_per_region[idx] += 1

            positions: List[tuple[int, int]] = []
            spill = 0
            for idx, region in enumerate(REGION_SEQUENCE):
                target = counts_per_region[idx] + spill
                if target <= 0:
                    spill = 0
                    continue
                chosen = pick_positions(region_masks[region], target)
                positions.extend(chosen)
                spill = target - len(chosen)
            if spill > 0:
                overflow = pick_positions(np.ones_like(available_mask, dtype=bool), spill)
                positions.extend(overflow)
            place_agents(agent_type, positions)

        place_with_profile("cd8", AgentType.CD8, counts.get("cd8", 0))
        place_with_profile("cd4", AgentType.CD4, counts.get("cd4", 0))
        place_with_profile("treg", AgentType.TREG, counts.get("treg", 0))
        place_with_profile("macrophage", AgentType.MACROPHAGE, counts.get("macrophage", 0))

        # Initialize chemokine field
        self.field_engine.initialize_field(cxcl9_mean=preset.cxcl9_10_mean)

    def _estimate_cluster_radius(self, total_agents: int) -> float:
        min_dim = min(self.grid.width, self.grid.height)
        if total_agents <= 0:
            return min_dim * 0.12
        radius = np.sqrt(total_agents / np.pi)
        return float(np.clip(radius, min_dim * 0.12, min_dim * 0.45))

    def _build_agent(self, agent_type: AgentType, pos: tuple[int, int]) -> agents.LUADBaseAgent:
        if agent_type == AgentType.TUMOR:
            mhc_i = float(np.clip(self.np_rng.normal(0.75, 0.08), 0.2, 1.0))
            pd_l1 = float(np.clip(self.np_rng.normal(0.25, 0.05), 0.0, 1.0))
            return agents.TumorAgent(self, mhc_i=mhc_i, pd_l1=pd_l1)
        if agent_type == AgentType.CD8:
            activation = float(np.clip(self.np_rng.normal(0.4, 0.1), 0.05, 1.0))
            # Recruits arrive with exhaustion primed by microenvironment
            base_exh = 0.2
            if hasattr(self, '_avg_cd8_exhaustion'):
                base_exh = 0.2 + self.params.get("recruit_exhaustion_priming", 0.3) * self._avg_cd8_exhaustion
            exhaustion = float(np.clip(self.np_rng.normal(base_exh, 0.05), 0.0, 0.8))
            return agents.CD8TCell(self, activation=activation, exhaustion=exhaustion)
        if agent_type == AgentType.CD4:
            activation = float(np.clip(self.np_rng.normal(0.35, 0.1), 0.05, 1.0))
            return agents.CD4TCell(self, activation=activation)
        if agent_type == AgentType.TREG:
            base = self.preset.suppression["treg"]
            return agents.TregCell(self, suppression_strength=base)
        if agent_type == AgentType.MACROPHAGE:
            frac_m2 = self.preset.macrophage_m2_fraction
            pol = float(self.np_rng.normal(-0.5, 0.2) if self.np_rng.random() < frac_m2 else self.np_rng.normal(0.4, 0.2))
            return agents.Macrophage(self, polarization=np.clip(pol, -1.0, 1.0))
        raise ValueError(f"Unknown agent type {agent_type}")

    def apply_interventions(self, interventions: Sequence[str]) -> None:
        for name in interventions:
            key = name.strip()
            if not key:
                continue
            upper = key.upper()
            if upper == "PD1":
                self.active_interventions["PD1"] = True
                self.params["pd1_blockade"] = True
                self.params["pd1_kill_bonus"] = 0.02
            elif upper in {"CTLA4", "TREG", "TREG_MOD"}:
                self.active_interventions["CTLA4"] = True
                self.params["treg_mod_factor"] = 0.5

    def step(self) -> None:
        self.field_engine.begin_step()
        self.ifng_signal *= 0.6
        # Track avg CD8 exhaustion for recruit priming
        cd8_cells = list(self.iter_agents(AgentType.CD8))
        if cd8_cells:
            self._avg_cd8_exhaustion = sum(c.exhaustion for c in cd8_cells) / len(cd8_cells)
        else:
            self._avg_cd8_exhaustion = 0.0
        self.scheduler.step()
        self._recruit_step()
        self.field_engine.diffuse_and_decay()
        self.datacollector.collect(self)
        self.metrics_tracker.record(self)
        self.step_count += 1

    def _recruit_step(self) -> None:
        """Recruit new immune cells from periphery, driven by chemokine."""
        avg_cxcl = float(self.field_engine.cxcl9_10.mean())
        boost = 1.0 + self.params["recruitment_chemokine_boost"] * avg_cxcl
        rate = self.params["recruitment_rate"]

        # Tumor burden suppresses macrophage recruitment over time
        n_tumor = count_agents(self, AgentType.TUMOR)
        initial_tumor = self.preset.initial_agents.get("tumor", 1200)
        tumor_expansion = max(0.0, (n_tumor - initial_tumor) / initial_tumor)

        for cell_key, initial_n in self.preset.initial_agents.items():
            if cell_key == "tumor":
                continue
            cell_rate = rate
            # Macrophage recruitment declines as tumor burden grows
            if cell_key == "macrophage":
                cell_rate *= max(0.1, 1.0 - self.params["mac_recruit_suppression"] * tumor_expansion)
            expected = cell_rate * initial_n * boost
            n_new = int(self.np_rng.poisson(expected))
            if n_new <= 0:
                continue
            agent_type = AgentType(cell_key)
            self._place_recruits(agent_type, n_new)

    def _place_recruits(self, agent_type: AgentType, count: int) -> None:
        """Place recruited immune cells at peripheral grid positions."""
        width, height = self.grid.width, self.grid.height
        center_x, center_y = width / 2.0, height / 2.0
        max_dist = min(width, height) / 2.0
        placed = 0
        attempts = 0
        max_attempts = count * 10

        while placed < count and attempts < max_attempts:
            # Bias toward periphery: sample random position, reject if too central
            x = int(self.np_rng.integers(0, width))
            y = int(self.np_rng.integers(0, height))
            dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            # Accept with probability proportional to distance from center
            accept_prob = min(1.0, (dist / max_dist) ** 1.5)
            if self.np_rng.random() > accept_prob:
                attempts += 1
                continue
            pos = (x, y)
            if self.grid.is_cell_empty(pos):
                agent = self._build_agent(agent_type, pos)
                self.grid.place_agent(agent, pos)
                self.scheduler.add(agent)
                placed += 1
            attempts += 1


__all__ = ["LUADModel", "load_preset", "PresetConfig"]
