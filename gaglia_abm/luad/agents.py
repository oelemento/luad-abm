"""Agent definitions for the Gaglia-grounded LUAD model.

5 agent types only: Tumor, CD8, CD4, Treg, Macrophage.
No CAF, TLS, or EMT mechanics.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import mesa

from . import rules


class AgentType(str, Enum):
    TUMOR = "tumor"
    CD8 = "cd8"
    CD4 = "cd4"
    TREG = "treg"
    MACROPHAGE = "macrophage"


class LUADBaseAgent(mesa.Agent):
    agent_type: AgentType

    def movement_step(self) -> None:
        return

    def interactions_step(self) -> None:
        return

    def state_updates_step(self) -> None:
        return


class TumorAgent(LUADBaseAgent):
    agent_type = AgentType.TUMOR

    def __init__(self, model, mhc_i: float = 0.8, pd_l1: float = 0.2):
        super().__init__(model)
        self.is_tumor = True
        self.mhc_i = mhc_i
        self.pd_l1 = pd_l1
        self.proliferative = True
        self.alive = True

    def killed(self) -> None:
        self.alive = False

    def interactions_step(self) -> None:
        ifng = self.model.ifng_signal[self.pos[1], self.pos[0]]
        if ifng > 0:
            # IFNg induces both PD-L1 and MHC-I (B2m) on tumor cells
            if self.model.params["pd_l1_induction_rate"] > 0:
                self.pd_l1 = min(1.0, self.pd_l1 + self.model.params["pd_l1_induction_rate"])
            mhc_induction = self.model.params.get("mhc_i_induction_rate", 0.02)
            self.mhc_i = min(1.0, self.mhc_i + mhc_induction)

    def state_updates_step(self) -> None:
        # MHC-I decays slowly (tumor immune evasion)
        mhc_decay = self.model.params.get("mhc_i_decay_rate", 0.005)
        self.mhc_i = max(0.1, self.mhc_i - mhc_decay)

        cxcl_emit = self.model.params.get("tumor_cxcl9_emission", 0.05)
        if cxcl_emit > 0:
            self.model.field_engine.deposit(self.pos, cxcl_emit)

        if self.proliferative and self.model.random.random() < self.model.params["tumor_proliferation_rate"]:
            self._attempt_division()

    def _attempt_division(self) -> None:
        empty_neighbors = [
            pos for pos in self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            if self.model.grid.is_cell_empty(pos)
        ]
        if not empty_neighbors:
            return
        new_pos = self.model.random.choice(empty_neighbors)
        daughter = TumorAgent(
            self.model,
            mhc_i=max(0.0, min(1.0, self.mhc_i + self.model.random.uniform(-0.05, 0.05))),
            pd_l1=max(0.0, min(1.0, self.pd_l1 + self.model.random.uniform(-0.05, 0.05))),
        )
        self.model.grid.place_agent(daughter, new_pos)
        self.model.scheduler.add(daughter)


class CD8TCell(LUADBaseAgent):
    agent_type = AgentType.CD8

    def __init__(self, model, activation: float = 0.4, exhaustion: float = 0.2):
        super().__init__(model)
        self.activation = activation
        self.exhaustion = exhaustion
        self.tumor_attraction = 0.1
        self.recent_kills = 0
        self.step_probability = 1.0
        self.noise_scale = 0.02

    def movement_step(self) -> None:
        rules.chemotactic_move(self.model, self, self.step_probability, self.noise_scale)

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
        # Death: base rate + scaled by exhaustion level (continuous, not threshold)
        death_rate = self.model.params["immune_base_death_rate"]
        death_rate += self.model.params["cd8_exhaustion_death_bonus"] * self.exhaustion
        if self.model.random.random() < death_rate:
            rules.remove_immune_agent(self.model, self)


class CD4TCell(LUADBaseAgent):
    agent_type = AgentType.CD4

    def __init__(self, model, activation: float = 0.3):
        super().__init__(model)
        self.activation = activation
        self.tumor_attraction = 0.05
        self.step_probability = 0.9
        self.noise_scale = 0.02

    def movement_step(self) -> None:
        rules.chemotactic_move(self.model, self, self.step_probability, self.noise_scale)

    def state_updates_step(self) -> None:
        self.model.field_engine.deposit(self.pos, self.activation * 0.02)
        self.activation = float(max(0.0, min(1.0, self.activation * 0.995)))
        if self.model.random.random() < self.model.params["immune_base_death_rate"]:
            rules.remove_immune_agent(self.model, self)


class TregCell(LUADBaseAgent):
    agent_type = AgentType.TREG

    def __init__(self, model, suppression_strength: float = 0.25):
        super().__init__(model)
        self.is_treg = True
        self.base_suppression = suppression_strength
        self.suppression_strength = suppression_strength
        self.step_probability = 0.85
        self.noise_scale = 0.025
        self.tumor_attraction = 0.0

    def movement_step(self) -> None:
        rules.chemotactic_move(self.model, self, self.step_probability, self.noise_scale)

    def state_updates_step(self) -> None:
        factor = self.model.params.get("treg_mod_factor", 1.0) if self.model.active_interventions.get("CTLA4", False) else 1.0
        self.suppression_strength = float(max(0.05, min(0.6, self.base_suppression * factor)))
        # Tregs proliferate in tumor-proximal / suppressive regions
        tumor_dens = rules.local_tumor_density(self.model, self.pos)
        prolif_rate = self.model.params["treg_prolif_rate"] * tumor_dens
        if prolif_rate > 0 and self.model.random.random() < prolif_rate:
            self._attempt_division()
        # Tregs are more resistant to death in TME (80% of base rate)
        if self.model.random.random() < self.model.params["immune_base_death_rate"] * 0.8:
            rules.remove_immune_agent(self.model, self)

    def _attempt_division(self) -> None:
        empty = [
            pos for pos in self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            if self.model.grid.is_cell_empty(pos)
        ]
        if not empty:
            return
        new_pos = self.model.random.choice(empty)
        daughter = TregCell(self.model, suppression_strength=self.base_suppression)
        self.model.grid.place_agent(daughter, new_pos)
        self.model.scheduler.add(daughter)


class Macrophage(LUADBaseAgent):
    agent_type = AgentType.MACROPHAGE

    def __init__(self, model, polarization: float = 0.0):
        super().__init__(model)
        self.is_macrophage = True
        self.polarization = polarization  # -1 (M2) to +1 (M1)
        self.suppression_strength = max(0.05, (1.0 - polarization) * model.params["macrophage_suppr_base"])
        self.step_probability = 0.5
        self.noise_scale = 0.02
        self.tumor_attraction = 0.0

    def movement_step(self) -> None:
        if self.model.random.random() < 0.6:
            rules.chemotactic_move(self.model, self, self.step_probability, self.noise_scale)

    def state_updates_step(self) -> None:
        rules.macrophage_polarization_step(self, self.model.params)
        if self.polarization > 0:
            self.model.field_engine.deposit(self.pos, 0.05 * self.polarization)
        # Macrophage death: base rate + tumor-proximity death (hostile TME)
        # M2-polarized macrophages near tumor are cleared faster
        tumor_dens = rules.local_tumor_density(self.model, self.pos)
        m2_factor = max(0.0, (1.0 - self.polarization) * 0.5)  # higher for M2
        death_rate = self.model.params["immune_base_death_rate"] * 0.7
        death_rate += self.model.params["mac_tumor_death_rate"] * (tumor_dens + m2_factor * 0.3)
        if self.model.random.random() < death_rate:
            rules.remove_immune_agent(self.model, self)


AGENT_TYPE_TO_CLASS = {
    AgentType.TUMOR: TumorAgent,
    AgentType.CD8: CD8TCell,
    AgentType.CD4: CD4TCell,
    AgentType.TREG: TregCell,
    AgentType.MACROPHAGE: Macrophage,
}


def create_agent(agent_type: AgentType, model, **kwargs) -> LUADBaseAgent:
    agent_cls = AGENT_TYPE_TO_CLASS[agent_type]
    return agent_cls(model, **kwargs)
