"""Field management utilities for the LUAD Mesa model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.signal import convolve2d


@dataclass
class FieldParameters:
    """Numeric knobs controlling field dynamics."""

    diffusion_kernel: np.ndarray
    cxcl9_decay: float
    cxcl13_decay: float
    tgfb_decay: float
    ecm_decay: float
    ecm_cap: float


class FieldEngine:
    """Holds diffusive fields and manages deposition/decay each tick."""

    def __init__(self, width: int, height: int, params: FieldParameters, rng: np.random.Generator) -> None:
        self.width = width
        self.height = height
        self.shape = (height, width)
        self.params = params
        self.rng = rng

        # Persistent fields
        self.ecm = np.zeros(self.shape, dtype=np.float32)
        self.cxcl9_10 = np.zeros(self.shape, dtype=np.float32)
        self.cxcl13 = np.zeros(self.shape, dtype=np.float32)
        self.tgfb = np.zeros(self.shape, dtype=np.float32)

        # Accumulators for sources during the step
        self._pending: Dict[str, np.ndarray] = {
            "ecm": np.zeros(self.shape, dtype=np.float32),
            "cxcl9_10": np.zeros(self.shape, dtype=np.float32),
            "cxcl13": np.zeros(self.shape, dtype=np.float32),
            "tgfb": np.zeros(self.shape, dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def initialize_fields(
        self,
        ecm_mean: float,
        cxcl9_mean: float,
        cxcl13_mean: float,
        tgfb_mean: float,
        tumor_coords: Iterable[Tuple[int, int]] | None = None,
        ecm_ring_strength: float = 0.0,
    ) -> None:
        """Seed fields with simple patterns derived from the preset configuration."""

        self.ecm.fill(ecm_mean)
        self.cxcl9_10 = self._clip_nonnegative(
            ecm_mean * 0.1 + self.rng.normal(cxcl9_mean, 0.02, size=self.shape)
        )
        self.cxcl13 = self._clip_nonnegative(self.rng.normal(cxcl13_mean, 0.02, size=self.shape))
        self.tgfb = self._clip_nonnegative(
            self.rng.normal(tgfb_mean, 0.015, size=self.shape)
        )

        if tumor_coords and ecm_ring_strength > 0:
            ring_map = np.zeros(self.shape, dtype=np.float32)
            for (x, y) in tumor_coords:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            ring_map[ny, nx] += 1.0
            if ring_map.sum() > 0:
                ring_map /= ring_map.max()
                self.ecm = self._clip_nonnegative(self.ecm + ecm_ring_strength * ring_map)

    # ------------------------------------------------------------------
    def begin_step(self) -> None:
        for buffer in self._pending.values():
            buffer.fill(0.0)

    def deposit(self, field: str, pos: Tuple[int, int], amount: float) -> None:
        """Queue a local deposition for a field (applied at finalize step)."""
        if field not in self._pending:
            raise KeyError(f"Unknown field '{field}'")
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height:
            self._pending[field][y, x] += amount

    def diffuse_and_decay(self) -> None:
        """Apply pending sources, diffusion, and decay in a single pass."""
        # First apply pending sources
        self.ecm = self._clip_nonnegative(self.ecm + self._pending["ecm"])
        self.cxcl9_10 = self._clip_nonnegative(self.cxcl9_10 + self._pending["cxcl9_10"])
        self.cxcl13 = self._clip_nonnegative(self.cxcl13 + self._pending["cxcl13"])
        self.tgfb = self._clip_nonnegative(self.tgfb + self._pending["tgfb"])

        kernel = self.params.diffusion_kernel

        self.cxcl9_10 = self._diffuse_field(self.cxcl9_10, kernel, self.params.cxcl9_decay)
        self.cxcl13 = self._diffuse_field(self.cxcl13, kernel, self.params.cxcl13_decay)
        self.tgfb = self._diffuse_field(self.tgfb, kernel, self.params.tgfb_decay)

        # ECM decays without diffusion
        self.ecm *= (1.0 - self.params.ecm_decay)
        np.clip(self.ecm, 0.0, self.params.ecm_cap, out=self.ecm)

    # ------------------------------------------------------------------
    def field_value(self, field: str, pos: Tuple[int, int]) -> float:
        x, y = pos
        arr = {
            "ecm": self.ecm,
            "cxcl9_10": self.cxcl9_10,
            "cxcl13": self.cxcl13,
            "tgfb": self.tgfb,
        }[field]
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(arr[y, x])
        return 0.0

    # ------------------------------------------------------------------
    def _diffuse_field(self, field: np.ndarray, kernel: np.ndarray, decay: float) -> np.ndarray:
        convolved = convolve2d(field, kernel, mode="same", boundary="fill", fillvalue=0.0)
        convolved *= (1.0 - decay)
        return self._clip_nonnegative(convolved)

    @staticmethod
    def _clip_nonnegative(field: np.ndarray) -> np.ndarray:
        np.clip(field, 0.0, None, out=field)
        return field


DEFAULT_KERNEL = np.array([[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32) / 8.0


def default_field_parameters() -> FieldParameters:
    return FieldParameters(
        diffusion_kernel=DEFAULT_KERNEL,
        cxcl9_decay=0.02,
        cxcl13_decay=0.01,
        tgfb_decay=0.005,
        ecm_decay=0.0015,
        ecm_cap=1.0,
    )
