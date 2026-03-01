"""Field management for the Gaglia-grounded LUAD model.

Single diffusive field: CXCL9/10 chemokine gradient.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.signal import convolve2d


@dataclass
class FieldParameters:
    diffusion_kernel: np.ndarray
    cxcl9_decay: float


class FieldEngine:
    """Holds the CXCL9/10 diffusive field and manages deposition/decay each tick."""

    def __init__(self, width: int, height: int, params: FieldParameters, rng: np.random.Generator) -> None:
        self.width = width
        self.height = height
        self.shape = (height, width)
        self.params = params
        self.rng = rng

        self.cxcl9_10 = np.zeros(self.shape, dtype=np.float32)
        self._pending = np.zeros(self.shape, dtype=np.float32)

    def initialize_field(self, cxcl9_mean: float) -> None:
        self.cxcl9_10 = np.clip(
            self.rng.normal(cxcl9_mean, 0.02, size=self.shape).astype(np.float32),
            0.0, None,
        )

    def begin_step(self) -> None:
        self._pending.fill(0.0)

    def deposit(self, pos: Tuple[int, int], amount: float) -> None:
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height:
            self._pending[y, x] += amount

    def diffuse_and_decay(self) -> None:
        self.cxcl9_10 = np.clip(self.cxcl9_10 + self._pending, 0.0, None)
        convolved = convolve2d(self.cxcl9_10, self.params.diffusion_kernel,
                               mode="same", boundary="fill", fillvalue=0.0)
        self.cxcl9_10 = np.clip(convolved * (1.0 - self.params.cxcl9_decay), 0.0, None)

    def field_value(self, pos: Tuple[int, int]) -> float:
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.cxcl9_10[y, x])
        return 0.0


DEFAULT_KERNEL = np.array(
    [[0.0, 1.0, 0.0],
     [1.0, 4.0, 1.0],
     [0.0, 1.0, 0.0]], dtype=np.float32
) / 8.0


def default_field_parameters() -> FieldParameters:
    return FieldParameters(
        diffusion_kernel=DEFAULT_KERNEL,
        cxcl9_decay=0.02,
    )
