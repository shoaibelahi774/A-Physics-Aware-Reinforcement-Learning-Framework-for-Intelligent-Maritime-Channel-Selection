from __future__ import annotations
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable
import numpy as np
from config.paper_parameters import CLUSTER_EVOLUTION, CLUSTER_GEOMETRY
from config.simulation_config import SimulationConfig, default_config
from gbsm.geometry.cluster_angles import ClusterAngleModel
from gbsm.geometry.scenarios import ClusterKind
from gbsm import qmc


_STANDARD_NORMAL = NormalDist()
_GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0

VelocityProvider = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ClusterRay:
    position_enu_m: np.ndarray
    azimuth_rad: float
    elevation_rad: float


@dataclass
class Cluster:
    identifier: int
    kind: ClusterKind
    center_enu_m: np.ndarray
    departure_azimuth_rad: float
    departure_elevation_rad: float
    distance_m: float
    velocity_enu_m_s: np.ndarray
    rays: tuple[ClusterRay, ...]
    survival_quantile: float
    accumulated_decay: float = 0.0


class ClusterField:
    def __init__(
        self,
        kind: ClusterKind,
        antenna_height_m: float,
        los_azimuth_rad: float,
        on_land: bool,
        config: SimulationConfig | None = None,
        angle_model: ClusterAngleModel | None = None,
    ) -> None:
        self._kind = kind
        self._height = antenna_height_m
        self._los_azimuth = los_azimuth_rad
        self._on_land = on_land
        self._config = config or default_config()
        self._angles = angle_model or ClusterAngleModel(self._config)
        self._geometry = CLUSTER_GEOMETRY
        self._evolution = CLUSTER_EVOLUTION
        self._angle_pool = self._angles.cluster_angles(
            kind, self._geometry.angle_pool_size, antenna_height_m, los_azimuth_rad, on_land
        )
        self._clusters: list[Cluster] = []
        self._birth_counter = 0
        self._new_cluster_carry = 0.0

    @property
    def kind(self) -> ClusterKind:
        return self._kind

    @property
    def clusters(self) -> list[Cluster]:
        return list(self._clusters)

    def initialize(self, tx_origin_enu_m: np.ndarray, velocity_provider: VelocityProvider | None = None) -> list[Cluster]:
        self._clusters = []
        self._birth_counter = 0
        self._new_cluster_carry = 0.0
        for _ in range(self._geometry.initial_cluster_count):
            self._clusters.append(self._spawn(tx_origin_enu_m, velocity_provider))
        return self.clusters

    def evolve(
        self,
        tx_origin_enu_m: np.ndarray,
        terminal_speed_m_s: float,
        array_span_m: float,
        delta_t_s: float,
        velocity_provider: VelocityProvider | None = None,
    ) -> list[Cluster]:
        decay = self._evolution.recombination_rate * (
            terminal_speed_m_s * delta_t_s / self._evolution.time_correlated_distance_m
            + array_span_m / self._evolution.array_correlated_distance_m
        )
        survivors = []
        for cluster in self._clusters:
            cluster.accumulated_decay += decay
            if math.exp(-cluster.accumulated_decay) >= cluster.survival_quantile:
                survivors.append(cluster)
        self._clusters = survivors
        survival_probability = math.exp(-decay)
        expected_new = (
            self._evolution.generation_rate / self._evolution.recombination_rate
        ) * (1.0 - survival_probability) + self._new_cluster_carry
        new_count = int(expected_new)
        self._new_cluster_carry = expected_new - new_count
        for _ in range(new_count):
            if len(self._clusters) >= self._geometry.max_cluster_count:
                break
            self._clusters.append(self._spawn(tx_origin_enu_m, velocity_provider))
        return self.clusters

    def _spawn(self, tx_origin_enu_m: np.ndarray, velocity_provider: VelocityProvider | None) -> Cluster:
        index = self._birth_counter
        self._birth_counter += 1
        pool_index = int((qmc.rotate(((index + 0.5) * _GOLDEN_RATIO) % 1.0)) * len(self._angle_pool))
        azimuth, elevation = self._angle_pool[pool_index]
        distance = self._distance(index, elevation)
        direction = np.array(
            [
                math.cos(azimuth) * math.cos(elevation),
                math.sin(azimuth) * math.cos(elevation),
                math.sin(elevation),
            ]
        )
        center = np.asarray(tx_origin_enu_m, dtype=float) + distance * direction
        rays = self._rays(index, center, azimuth, elevation)
        velocity = (
            np.zeros(3) if velocity_provider is None else np.asarray(velocity_provider(center), dtype=float)
        )
        return Cluster(
            identifier=index,
            kind=self._kind,
            center_enu_m=center,
            departure_azimuth_rad=azimuth,
            departure_elevation_rad=elevation,
            distance_m=distance,
            velocity_enu_m_s=velocity,
            rays=rays,
            survival_quantile=self._survival_quantile(index),
        )

    def _distance(self, index: int, elevation: float) -> float:
        if self._kind is ClusterKind.SEA_SURFACE and math.sin(elevation) < -1.0e-3:
            return self._height / (-math.sin(elevation))
        quantile = qmc.rotate(((index + 0.5) * _GOLDEN_RATIO) % 1.0)
        return self._geometry.min_cluster_distance_m + quantile * (
            self._geometry.max_cluster_distance_m - self._geometry.min_cluster_distance_m
        )

    def _rays(self, index: int, center: np.ndarray, azimuth: float, elevation: float) -> tuple[ClusterRay, ...]:
        rays = []
        for ray_index in range(self._geometry.rays_per_cluster):
            offset = self._ray_offset(index * self._geometry.rays_per_cluster + ray_index)
            rays.append(ClusterRay(center + offset, azimuth, elevation))
        return tuple(rays)

    def _ray_offset(self, seed: int) -> np.ndarray:
        std = self._geometry.intra_cluster_std_m
        quantiles = (
            qmc.rotate(((seed + 0.5) * _GOLDEN_RATIO) % 1.0),
            qmc.rotate(((seed + 0.5) * _GOLDEN_RATIO**2) % 1.0),
            qmc.rotate(((seed + 0.5) * _GOLDEN_RATIO**3) % 1.0),
        )
        return np.array([std * _STANDARD_NORMAL.inv_cdf(min(max(q, 1.0e-6), 1.0 - 1.0e-6)) for q in quantiles])

    @staticmethod
    def _survival_quantile(index: int) -> float:
        return min(max(qmc.rotate(((index + 0.5) * _GOLDEN_RATIO) % 1.0), 1.0e-6), 1.0 - 1.0e-6)


__all__ = ["ClusterRay", "Cluster", "ClusterField", "VelocityProvider"]