from __future__ import annotations

import numpy as np

_PHASE = 0.0


def set_phase(phase: float) -> None:
    """Set the current Cranley-Patterson phase (wrapped into [0, 1))."""
    global _PHASE
    _PHASE = float(phase) % 1.0


def get_phase() -> float:
    return _PHASE


def reset() -> None:
    """Restore the deterministic run (phase = 0)."""
    global _PHASE
    _PHASE = 0.0


def rotate(quantile: float) -> float:
    """Apply the rotation to a unit-interval quantile. Identity when phase = 0."""
    return (quantile + _PHASE) % 1.0


def phases_for_ensemble(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phases = rng.random(count)
    if count > 0:
        phases[0] = 0.0
    return phases


__all__ = ["set_phase", "get_phase", "reset", "rotate", "phases_for_ensemble"]