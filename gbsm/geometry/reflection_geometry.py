from __future__ import annotations
import math
from dataclasses import dataclass
from statistics import NormalDist
import numpy as np
from config.paper_parameters import CLUSTER_GEOMETRY
from config.simulation_config import SimulationConfig, default_config
from gbsm import qmc


_STANDARD_NORMAL = NormalDist()
_GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0
_MIN_SINE = 1.0e-6


@dataclass(frozen=True)
class ReflectionGeometry:
    reflection_point_enu_m: np.ndarray
    azimuth_rad: float
    elevation_rad: float
    tx_distance_m: float
    rx_distance_m: float
    tx_distance_vectors_m: tuple[np.ndarray, ...]
    rx_distance_vectors_m: tuple[np.ndarray, ...]


class ReflectionModel:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()
        self._geometry = CLUSTER_GEOMETRY

    def compute(self, tx_position_enu_m: np.ndarray, rx_position_enu_m: np.ndarray) -> ReflectionGeometry:
        tx = np.asarray(tx_position_enu_m, dtype=float)
        rx = np.asarray(rx_position_enu_m, dtype=float)
        line_of_sight = rx - tx
        azimuth = math.atan2(line_of_sight[1], line_of_sight[0])
        height_sum = tx[2] + rx[2]
        line_of_sight_distance = float(np.linalg.norm(line_of_sight))
        elevation = -math.atan2(height_sum, line_of_sight_distance)
        sine = math.sin(elevation)
        sine = sine if abs(sine) >= _MIN_SINE else -_MIN_SINE
        tx_distance = abs(tx[2] / sine)
        rx_distance = abs(rx[2] / sine)
        tx_direction = np.array(
            [math.cos(azimuth) * math.cos(elevation), math.sin(azimuth) * math.cos(elevation), math.sin(elevation)]
        )
        rx_direction = np.array(
            [-math.cos(azimuth) * math.cos(elevation), -math.sin(azimuth) * math.cos(elevation), math.sin(elevation)]
        )
        tx_vectors, rx_vectors = self._ray_vectors(
            tx_distance * tx_direction, rx_distance * rx_direction
        )
        return ReflectionGeometry(
            reflection_point_enu_m=self._reflection_point(tx, rx),
            azimuth_rad=azimuth,
            elevation_rad=elevation,
            tx_distance_m=tx_distance,
            rx_distance_m=rx_distance,
            tx_distance_vectors_m=tx_vectors,
            rx_distance_vectors_m=rx_vectors,
        )

    def _ray_vectors(
        self, tx_center: np.ndarray, rx_center: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        tx_vectors = []
        rx_vectors = []
        for ray_index in range(self._geometry.rays_per_cluster):
            offset = self._ray_offset(ray_index)
            tx_vectors.append(tx_center + offset)
            rx_vectors.append(rx_center + offset)
        return tuple(tx_vectors), tuple(rx_vectors)

    def _ray_offset(self, seed: int) -> np.ndarray:
        std = self._geometry.intra_cluster_std_m
        quantiles = (
            qmc.rotate(((seed + 0.5) * _GOLDEN_RATIO) % 1.0),
            qmc.rotate(((seed + 0.5) * _GOLDEN_RATIO**2) % 1.0),
            qmc.rotate(((seed + 0.5) * _GOLDEN_RATIO**3) % 1.0),
        )
        return np.array([std * _STANDARD_NORMAL.inv_cdf(min(max(q, 1.0e-6), 1.0 - 1.0e-6)) for q in quantiles])

    @staticmethod
    def _reflection_point(tx: np.ndarray, rx: np.ndarray) -> np.ndarray:
        height_sum = tx[2] + rx[2]
        fraction = tx[2] / height_sum if height_sum > 0.0 else 0.5
        east = tx[0] + fraction * (rx[0] - tx[0])
        north = tx[1] + fraction * (rx[1] - tx[1])
        return np.array([east, north, 0.0])


__all__ = ["ReflectionGeometry", "ReflectionModel"]