from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from config.simulation_config import SimulationConfig, default_config


_LOG2 = np.log(2.0)


class CapacityModel:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()

    def instantaneous_capacity(self, channel_matrix: np.ndarray, snr_linear: float) -> float:
        channel = np.asarray(channel_matrix, dtype=complex).T
        transmit_count = channel.shape[1]
        if transmit_count == 0:
            return 0.0
        gram = channel @ channel.conj().T
        matrix = np.eye(channel.shape[0]) + (snr_linear / transmit_count) * gram
        sign, log_abs_det = np.linalg.slogdet(matrix)
        if sign.real <= 0.0:
            return 0.0
        return float(log_abs_det / _LOG2)

    def ergodic_capacity(self, channel_matrices: Sequence[np.ndarray], snr_linear: float) -> float:
        if not channel_matrices:
            return 0.0
        values = [self.instantaneous_capacity(matrix, snr_linear) for matrix in channel_matrices]
        return float(np.mean(values))

    def capacity_vs_snr(
        self,
        channel_matrices: Sequence[np.ndarray],
        snr_db_values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        snr_db = self.snr_grid() if snr_db_values is None else np.asarray(snr_db_values, dtype=float)
        capacities = np.array(
            [self.ergodic_capacity(channel_matrices, 10.0 ** (value / 10.0)) for value in snr_db]
        )
        return snr_db, capacities

    def snr_grid(self) -> np.ndarray:
        link_budget = self._config.link_budget
        return np.linspace(link_budget.snr_min_db, link_budget.snr_max_db, link_budget.snr_point_count)


__all__ = ["CapacityModel"]