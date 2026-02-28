"""Agent definitions for the LUAD Mesa simulator."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import mesa

from . import rules


class AgentType(str, Enum):
    TUMOR = "tumor"
    CAF = "caf"
    CD8 = "cd8"
    CD4 = "cd4"
    TREG = "treg"
    MACROPHAGE = "macrophage"
    TLS = "tls"


@dataclass
class MovementProfile:
    step_probability: float
    chemokine_field: str
    noise_scale: float = 0.01


class LUADBaseAgent(mesa.Agent):
    agent_type: AgentType

    def movement_step(self) -> None:  # pragma: no cover - default no-op
        return

    def interactions_step(self) -> None:  # pragma: no cover - default no-op
        return

    def state_updates_step(self) -> None:  # pragma: no cover - default no-op
        return


class EpithelialTumorAgent(LUADBaseAgent):
    agent_type = AgentType.TUMOR

    def __init__(
        self,
        model,
        cell_state: str = "tumor",
        mhc_i: float = 0.8,
        pd_l1: float = 0.2,
        emt: float = 0.2,
    ):
        super().__init__(model)
        self.cell_state = cell_state
        self.is_tumor = True
        self.mhc_i = mhc_i
        self.pd_l1 = pd_l1
        self.emt = emt
        self.proliferative = True
        self.alive = True

    @property
    def is_group4(self) -> bool:
        return self.model.group_flags.get("is_group4", False)

    def killed(self) -> None:
        self.alive = False

    def interactions_step(self) -> None:
        # Tumor cells can increase PD-L1 when exposed to IFNγ-like signal from recent kills
        if self.model.params["pd_l1_induction_rate"] > 0 and self.model.ifng_signal[self.pos[1], self.pos[0]] > 0:
            self.pd_l1 = min(1.0, self.pd_l1 + self.model.params["pd_l1_induction_rate"]) 

    def state_updates_step(self) -> None:
        fields = self.model.field_engine
        tgfb = fields.field_value("tgfb", self.pos)
        ecm = fields.field_value("ecm", self.pos)
        cxcl_emit = self.model.params.get("tumor_cxcl9_emission", 0.0)
        if cxcl_emit > 0:
            self.model.field_engine.deposit("cxcl9_10", self.pos, cxcl_emit * (1.0 - self.emt))
        tgfb_component = self.model.params.get("tgfb_gain", 1.0) * tgfb
        delta = self.model.params["emt_gain"] * (tgfb_component + ecm)
        if self.model.active_interventions.get("antiTGFb", False):
            delta *= self.model.params["anti_tgfb_factor"]
        self.emt = float(max(0.0, min(1.0, self.emt + delta - 0.01)))

        if self.is_group4 and self.emt >= 0.6:
            self._group4_motility_step()

        if self.proliferative and self.model.random.random() < self.model.params["tumor_proliferation_rate"] * (1.0 - self.emt):
            self._attempt_division()

    def _attempt_division(self) -> None:
        empty_neighbors = [
            pos for pos in self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            if self.model.grid.is_cell_empty(pos)
        ]
        if not empty_neighbors:
            return
        new_pos = self.model.random.choice(empty_neighbors)
        daughter = EpithelialTumorAgent(
            self.model,
            cell_state=self.cell_state,
            mhc_i=max(0.0, min(1.0, self.mhc_i + self.model.random.uniform(-0.05, 0.05))),
            pd_l1=max(0.0, min(1.0, self.pd_l1 + self.model.random.uniform(-0.05, 0.05))),
            emt=float(max(0.0, min(1.0, self.emt + self.model.random.uniform(-0.05, 0.05)))),
        )
        self.model.grid.place_agent(daughter, new_pos)
        self.model.scheduler.add(daughter)

    def _group4_motility_step(self) -> None:
        """Allow EMT-high Group4 tumor cells to locally remodel ECM and migrate."""
        model = self.model
        rng = model.random
        move_prob = 0.35
        if rng.random() > move_prob:
            return
        neighborhood = model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empty_sites = [pos for pos in neighborhood if model.grid.is_cell_empty(pos)]
        if not empty_sites:
            return
        best_pos = None
        best_score = -1e9
        for pos in empty_sites:
            ecm = model.field_engine.field_value("ecm", pos)
            tgfb = model.field_engine.field_value("tgfb", pos)
            cxcl = model.field_engine.field_value("cxcl9_10", pos)
            tumor_neighbors = rules.local_tumor_density(model, pos)
            score = (
                tgfb * 0.8
                - ecm * model.params["ecm_penalty"] * 0.5
                - cxcl * 0.2
                + tumor_neighbors * 0.3
                + rng.uniform(-0.05, 0.05)
            )
            if score > best_score:
                best_score = score
                best_pos = pos
        if best_pos is None:
            return
        model.grid.move_agent(self, best_pos)
        model.field_engine.deposit("ecm", best_pos, -0.15)
        model.field_engine.deposit("ecm", self.pos, -0.1)


class CAF(LUADBaseAgent):
    agent_type = AgentType.CAF

    def __init__(self, model, activation: float = 0.6):
        super().__init__(model)
        self.activation = activation
        self.is_caf = True
        self.movement = MovementProfile(step_probability=0.05, chemokine_field="tgfb", noise_scale=0.02)

    def movement_step(self) -> None:
        if self.model.random.random() < 0.1:
            rules.chemotactic_move(
                self.model,
                self,
                self.movement.chemokine_field,
                self.movement.step_probability,
                self.movement.noise_scale,
            )

    def state_updates_step(self) -> None:
        deposition = self.model.params["caf_ecm_deposition"] * self.activation
        if self.model.active_interventions.get("antiTGFb", False):
            deposition *= self.model.params["anti_tgfb_caf_penalty"]
        self.model.field_engine.deposit("ecm", self.pos, deposition)
        self.model.field_engine.deposit("tgfb", self.pos, self.model.params["caf_tgfb_secretion"] * self.activation)


class CD8TCell(LUADBaseAgent):
    agent_type = AgentType.CD8

    def __init__(self, model, activation: float = 0.4, exhaustion: float = 0.2):
        super().__init__(model)
        self.activation = activation
        self.exhaustion = exhaustion
        self.tumor_attraction = 0.1
        self.recent_kills = 0
        self.movement = MovementProfile(step_probability=1.0, chemokine_field="cxcl9_10", noise_scale=0.02)

    def movement_step(self) -> None:
        rules.chemotactic_move(
            self.model,
            self,
            self.movement.chemokine_field,
            self.movement.step_probability,
            self.movement.noise_scale,
        )

    def interactions_step(self) -> None:
        neighbors = [
            agent for agent in self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
            if getattr(agent, "is_tumor", False)
        ]
        if not neighbors:
            rules.decay_activation(self, self.model.params)
            return

        target = self.model.random.choice(neighbors)
        suppression = rules.compute_local_suppression(self.model, target.pos)
        kill_prob = rules.compute_cd8_kill_probability(self.model, self, target, suppression)
        if self.model.random.random() < kill_prob:
            rules.kill_target(self.model, self, target)
            rules.reinforce_activation(self, self.model.params)
            self.model.ifng_signal[self.pos[1], self.pos[0]] += 1.0
        else:
            rules.update_exhaustion(self, suppression, self.model.params)

    def state_updates_step(self) -> None:
        self.activation = float(max(0.0, min(1.0, self.activation * 0.99)))
        self.recent_kills = max(0, self.recent_kills - 1)


class CD4TCell(LUADBaseAgent):
    agent_type = AgentType.CD4

    def __init__(self, model, activation: float = 0.3):
        super().__init__(model)
        self.activation = activation
        self.tumor_attraction = 0.05
        self.movement = MovementProfile(step_probability=0.9, chemokine_field="cxcl9_10", noise_scale=0.02)

    def movement_step(self) -> None:
        rules.chemotactic_move(
            self.model,
            self,
            self.movement.chemokine_field,
            self.movement.step_probability,
            self.movement.noise_scale,
        )

    def state_updates_step(self) -> None:
        # Provide mild CXCL9/10 support when activated
        self.model.field_engine.deposit("cxcl9_10", self.pos, self.activation * 0.02)
        self.activation = float(max(0.0, min(1.0, self.activation * 0.995)))


class TregCell(LUADBaseAgent):
    agent_type = AgentType.TREG

    def __init__(self, model, suppression_strength: float = 0.25):
        super().__init__(model)
        self.is_treg = True
        self.base_suppression = suppression_strength
        self.suppression_strength = suppression_strength
        self.movement = MovementProfile(step_probability=0.85, chemokine_field="tgfb", noise_scale=0.025)

    def movement_step(self) -> None:
        rules.chemotactic_move(
            self.model,
            self,
            self.movement.chemokine_field,
            self.movement.step_probability,
            self.movement.noise_scale,
        )

    def state_updates_step(self) -> None:
        factor = self.model.params.get("treg_mod_factor", 1.0) if self.model.active_interventions.get("Treg_mod", False) else 1.0
        self.suppression_strength = float(max(0.05, min(0.6, self.base_suppression * factor)))


class Macrophage(LUADBaseAgent):
    agent_type = AgentType.MACROPHAGE

    def __init__(self, model, polarization: float = 0.0):
        super().__init__(model)
        self.is_macrophage = True
        self.polarization = polarization  # -1 (M2) to +1 (M1)
        self.suppression_strength = max(0.05, (1.0 - polarization) * model.params["macrophage_suppr_base"])
        self.movement = MovementProfile(step_probability=0.5, chemokine_field="tgfb", noise_scale=0.02)

    def movement_step(self) -> None:
        if self.model.random.random() < 0.6:
            rules.chemotactic_move(
                self.model,
                self,
                self.movement.chemokine_field,
                self.movement.step_probability,
                self.movement.noise_scale,
            )

    def state_updates_step(self) -> None:
        rules.macrophage_polarization_step(self, self.model.params)
        if self.polarization > 0:
            self.model.field_engine.deposit("cxcl9_10", self.pos, 0.05 * self.polarization)
        else:
            self.model.field_engine.deposit("tgfb", self.pos, 0.04 * abs(self.polarization))


class TLSNode(LUADBaseAgent):
    agent_type = AgentType.TLS

    def __init__(self, model, quality: float = 0.6):
        super().__init__(model)
        self.quality = quality
        self.static = True

    def interactions_step(self) -> None:
        self.model.field_engine.deposit("cxcl13", self.pos, self.model.params["tls_cxcl13_emission"] * self.quality)
        self.model.field_engine.deposit("cxcl9_10", self.pos, self.model.params["tls_cxcl9_emission"] * self.quality)

    def state_updates_step(self) -> None:
        rules.tls_quality_drift_step(self, self.model.params)
        if self.model.active_interventions.get("TLS_boost", False):
            self.quality = max(self.quality, self.model.params["tls_boost_floor"])
        if self.model.active_interventions.get("TIM3", False):
            self.quality = min(1.0, self.quality + self.model.params["tim3_quality_gain"])


AGENT_TYPE_TO_CLASS = {
    AgentType.TUMOR: EpithelialTumorAgent,
    AgentType.CAF: CAF,
    AgentType.CD8: CD8TCell,
    AgentType.CD4: CD4TCell,
    AgentType.TREG: TregCell,
    AgentType.MACROPHAGE: Macrophage,
    AgentType.TLS: TLSNode,
}


def create_agent(agent_type: AgentType, model, **kwargs) -> LUADBaseAgent:
    agent_cls = AGENT_TYPE_TO_CLASS[agent_type]
    return agent_cls(model, **kwargs)
