"""Metrics and analytics for LUAD simulations."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .agents import AgentType


@dataclass
class InteractionRecord:
    tick: int
    observed: Dict[Tuple[str, str], float]
    expected: Dict[Tuple[str, str], float]


@dataclass
class DistanceRecord:
    tick: int
    distances: np.ndarray
    hist_bins: np.ndarray
    hist_cdf: np.ndarray


@dataclass
class MetricsTracker:
    distance_interval: int = 20
    interaction_interval: int = 20
    shuffle_count: int = 10
    capture_grids: bool = True
    grid_interval: int = 5

    distance_records: List[DistanceRecord] = field(default_factory=list)
    interaction_records: List[InteractionRecord] = field(default_factory=list)
    grid_snapshots: List[Tuple[int, np.ndarray]] = field(default_factory=list)

    def record(self, model) -> None:
        tick = model.step_count
        if self.capture_grids and (tick % self.grid_interval == 0):
            self.grid_snapshots.append((tick, grid_encoding(model)))

        if tick % self.distance_interval == 0:
            distances = compute_cd8_to_tumor_distances(model)
            if distances.size:
                hist, bins = np.histogram(distances, bins=np.arange(0, 31, 1), density=True)
                cdf = np.cumsum(hist)
                self.distance_records.append(
                    DistanceRecord(tick=tick, distances=distances, hist_bins=bins[:-1], hist_cdf=cdf)
                )

        if tick % self.interaction_interval == 0:
            observed, expected = interaction_matrix(model, self.shuffle_count, model.metrics_rng)
            self.interaction_records.append(InteractionRecord(tick=tick, observed=observed, expected=expected))


# ---------------------------------------------------------------------------
# Helper metrics
# ---------------------------------------------------------------------------

def compute_cd8_to_tumor_distances(model) -> np.ndarray:
    cd8_positions = np.array([agent.pos for agent in model.iter_agents(AgentType.CD8)], dtype=np.float32)
    tumor_positions = np.array([agent.pos for agent in model.iter_agents(AgentType.TUMOR)], dtype=np.float32)
    if cd8_positions.size == 0 or tumor_positions.size == 0:
        return np.array([], dtype=np.float32)
    distances = cdist(cd8_positions, tumor_positions, metric="euclidean")
    return distances.min(axis=1)


def interaction_matrix(model, shuffle_count: int = 10, rng: np.random.Generator | None = None) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    categories = {}
    for agent in model.iter_all_agents():
        categories[agent.pos] = category_for_agent(agent)

    observed = adjacency_counts(model, categories)
    expected_accumulator: Dict[Tuple[str, str], float] = defaultdict(float)

    positions = list(categories.keys())
    labels = list(categories.values())

    if positions:
        labels_arr = np.array(labels)
        rng = rng or np.random.default_rng()
        for _ in range(shuffle_count):
            shuffled = rng.permutation(labels_arr)
            shuffled_mapping = {pos: label for pos, label in zip(positions, shuffled)}
            counts = adjacency_counts(model, shuffled_mapping)
            for key, value in counts.items():
                expected_accumulator[key] += value
        for key in expected_accumulator:
            expected_accumulator[key] /= shuffle_count

    return observed, expected_accumulator


def adjacency_counts(model, categories: Dict[Tuple[int, int], str]) -> Dict[Tuple[str, str], float]:
    counts: Dict[Tuple[str, str], float] = defaultdict(float)
    width, height = model.grid.width, model.grid.height

    def record_pair(cat_a: str, cat_b: str) -> None:
        key = tuple(sorted((cat_a, cat_b)))
        counts[key] += 1.0

    for x in range(width):
        for y in range(height):
            cat = categories.get((x, y))
            if not cat:
                continue
            for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    cat_b = categories.get((nx, ny))
                    if not cat_b:
                        continue
                    record_pair(cat, cat_b)
    return counts


def category_for_agent(agent) -> str:
    if getattr(agent, "is_tumor", False):
        return "epithelial"
    if getattr(agent, "is_caf", False):
        return "alpha_sma"
    if getattr(agent, "is_treg", False):
        return "treg"
    if getattr(agent, "agent_type", None) == AgentType.CD8:
        return "cd8"
    if getattr(agent, "agent_type", None) == AgentType.CD4:
        return "cd4"
    if getattr(agent, "agent_type", None) == AgentType.MACROPHAGE:
        return "macrophage"
    if getattr(agent, "agent_type", None) == AgentType.TLS:
        return "tls"
    return "other"


def grid_encoding(model) -> np.ndarray:
    width, height = model.grid.width, model.grid.height
    canvas = np.zeros((height, width), dtype=np.int16)
    type_to_code = {
        "epithelial": 1,
        "alpha_sma": 2,
        "cd8": 3,
        "cd4": 4,
        "treg": 5,
        "macrophage": 6,
        "tls": 7,
        "other": 8,
    }
    for agent in model.iter_all_agents():
        code = type_to_code.get(category_for_agent(agent), 0)
        x, y = agent.pos
        canvas[y, x] = code
    return canvas


# ---------------------------------------------------------------------------
# Convenience accessors for DataCollector
# ---------------------------------------------------------------------------

def count_agents(model, agent_type: AgentType) -> int:
    return sum(1 for _ in model.iter_agents(agent_type))


def emt_high_fraction(model, threshold: float = 0.6) -> float:
    tumor_cells = [agent for agent in model.iter_agents(AgentType.TUMOR)]
    if not tumor_cells:
        return 0.0
    high = sum(1 for agent in tumor_cells if agent.emt >= threshold)
    return high / len(tumor_cells)


def ecm_fraction(model, cutoff: float = 0.5) -> float:
    return float((model.field_engine.ecm >= cutoff).mean())


def tls_quality_mean(model) -> float:
    tls_nodes = [agent for agent in model.iter_agents(AgentType.TLS)]
    if not tls_nodes:
        return 0.0
    return float(np.mean([node.quality for node in tls_nodes]))
