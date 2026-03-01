"""Lightweight staged scheduler compatible with Mesa models."""
from __future__ import annotations

from typing import Iterable, List


class StagedScheduler:
    def __init__(self, model, stage_list: Iterable[str], shuffle: bool = True) -> None:
        self.model = model
        self.stage_list = list(stage_list)
        self.shuffle = shuffle
        self.agents: List[object] = []
        self.steps = 0
        self.time = 0

    def add(self, agent) -> None:
        self.agents.append(agent)

    def remove(self, agent) -> None:
        if agent in self.agents:
            self.agents.remove(agent)

    def step(self) -> None:
        for stage in self.stage_list:
            agent_list = list(self.agents)
            if self.shuffle:
                self.model.random.shuffle(agent_list)
            for agent in agent_list:
                if agent not in self.agents:
                    continue
                handler = getattr(agent, f"{stage}_step", None)
                if handler is not None:
                    handler()
        self.steps += 1
        self.time += 1
