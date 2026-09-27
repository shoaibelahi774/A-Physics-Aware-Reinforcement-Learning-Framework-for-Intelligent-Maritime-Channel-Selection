from __future__ import annotations
import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from config.paper_parameters import DELAY_POWER, PHYSICAL_CONSTANTS
from config.simulation_config import SimulationConfig, default_config
from gbsm.geometry.clusters import Cluster
from gbsm.geometry.reflection_geometry import ReflectionGeometry
from gbsm import qmc


_STANDARD_NORMAL = NormalDist()
_GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class Ray:
    delay_s: float
    power: float
    departure_azimuth_rad: float
    departure_elevation_rad: float
    arrival_azimuth_rad: float
    arrival_elevation_rad: float
    doppler_hz: float


@dataclass(frozen=True)
class PropagationSnapshot:
    los: Ray
    reflection: tuple[Ray, ...]
    nlos1: tuple[Ray, ...]
    nlos2: tuple[Ray, ...]


class PropagationModel:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()
        self._light_speed = PHYSICAL_CONSTANTS.speed_of_light_m_s
        self._carrier = self._config.carrier_frequency_hz
        self._delay_scalar = DELAY_POWER.delay_scalar
        self._delay_spread = DELAY_POWER.delay_spread_s
        self._shadow_std_db = DELAY_POWER.per_cluster_shadowing_std_db

    def compute(
        self,
        tx_origin_enu_m: np.ndarray,
        tx_velocity_enu_m_s: np.ndarray,
        rx_origin_enu_m: np.ndarray,
        rx_velocity_enu_m_s: np.ndarray,
        nlos1_clusters: list[Cluster],
        nlos2_clusters: list[Cluster],
        reflection: ReflectionGeometry,
    ) -> PropagationSnapshot:
        tx = np.asarray(tx_origin_enu_m, dtype=float)
        rx = np.asarray(rx_origin_enu_m, dtype=float)
        v_tx = np.asarray(tx_velocity_enu_m_s, dtype=float)
        v_rx = np.asarray(rx_velocity_enu_m_s, dtype=float)
        los = self._line_of_sight_ray(tx, v_tx, rx, v_rx)
        reflection_rays = self._reflection_rays(tx, v_tx, rx, v_rx, reflection)
        nlos1_rays = self._cluster_rays(tx, v_tx, rx, v_rx, nlos1_clusters)
        nlos2_rays = self._cluster_rays(tx, v_tx, rx, v_rx, nlos2_clusters)
        return PropagationSnapshot(los, reflection_rays, nlos1_rays, nlos2_rays)

    def _line_of_sight_ray(self, tx, v_tx, rx, v_rx) -> Ray:
        separation = tx - rx
        distance = float(np.linalg.norm(separation))
        departure_azimuth, departure_elevation = self._angles(rx - tx)
        arrival_azimuth, arrival_elevation = self._angles(tx - rx)
        unit = separation / distance if distance > 0.0 else np.zeros(3)
        range_rate = float(np.dot(v_tx - v_rx, unit))
        return Ray(
            delay_s=distance / self._light_speed,
            power=1.0,
            departure_azimuth_rad=departure_azimuth,
            departure_elevation_rad=departure_elevation,
            arrival_azimuth_rad=arrival_azimuth,
            arrival_elevation_rad=arrival_elevation,
            doppler_hz=-(self._carrier / self._light_speed) * range_rate,
        )

    def _cluster_rays(self, tx, v_tx, rx, v_rx, clusters: list[Cluster]) -> tuple[Ray, ...]:
        records = []
        for cluster in clusters:
            shadow = self._shadow_gain(cluster.identifier)
            for ray in cluster.rays:
                records.append(
                    self._scatter_ray(tx, v_tx, rx, v_rx, ray.position_enu_m, cluster.velocity_enu_m_s, shadow)
                )
        return self._normalize(records)

    def _reflection_rays(self, tx, v_tx, rx, v_rx, reflection: ReflectionGeometry) -> tuple[Ray, ...]:
        shadow = self._shadow_gain(0)
        zero_velocity = np.zeros(3)
        records = []
        for tx_vector, rx_vector in zip(reflection.tx_distance_vectors_m, reflection.rx_distance_vectors_m):
            scatterer = tx + tx_vector
            records.append(
                self._scatter_ray(tx, v_tx, rx, v_rx, scatterer, zero_velocity, shadow, rx_vector)
            )
        return self._normalize(records)

    def _scatter_ray(self, tx, v_tx, rx, v_rx, scatterer, scatterer_velocity, shadow, rx_vector=None):
        departure = scatterer - tx
        arrival = scatterer - rx if rx_vector is None else rx_vector
        departure_distance = float(np.linalg.norm(departure))
        arrival_distance = float(np.linalg.norm(arrival))
        delay = (departure_distance + arrival_distance) / self._light_speed
        departure_azimuth, departure_elevation = self._angles(departure)
        arrival_azimuth, arrival_elevation = self._angles(arrival)
        unit_departure = departure / departure_distance if departure_distance > 0.0 else np.zeros(3)
        unit_arrival = arrival / arrival_distance if arrival_distance > 0.0 else np.zeros(3)
        range_rate = float(
            np.dot(scatterer_velocity - v_tx, unit_departure) + np.dot(scatterer_velocity - v_rx, unit_arrival)
        )
        doppler = -(self._carrier / self._light_speed) * range_rate
        return _RayRecord(delay, shadow, departure_azimuth, departure_elevation, arrival_azimuth, arrival_elevation, doppler)

    def _normalize(self, records: list["_RayRecord"]) -> tuple[Ray, ...]:
        if not records:
            return ()
        minimum_delay = min(record.delay_s for record in records)
        slope = (self._delay_scalar - 1.0) / (self._delay_scalar * self._delay_spread)
        powers = []
        for record in records:
            excess = record.delay_s - minimum_delay
            powers.append(math.exp(-excess * slope) * record.shadow_gain)
        total = sum(powers)
        if total <= 0.0:
            powers = [1.0 / len(records)] * len(records)
            total = 1.0
        return tuple(
            Ray(
                delay_s=record.delay_s,
                power=power / total,
                departure_azimuth_rad=record.departure_azimuth_rad,
                departure_elevation_rad=record.departure_elevation_rad,
                arrival_azimuth_rad=record.arrival_azimuth_rad,
                arrival_elevation_rad=record.arrival_elevation_rad,
                doppler_hz=record.doppler_hz,
            )
            for record, power in zip(records, powers)
        )

    def _shadow_gain(self, identifier: int) -> float:
        quantile = min(max(qmc.rotate(((identifier + 0.5) * _GOLDEN_RATIO) % 1.0), 1.0e-6), 1.0 - 1.0e-6)
        shadow_db = self._shadow_std_db * _STANDARD_NORMAL.inv_cdf(quantile)
        return 10.0 ** (-shadow_db / 10.0)

    @staticmethod
    def _angles(vector: np.ndarray) -> tuple[float, float]:
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            return 0.0, 0.0
        azimuth = math.atan2(vector[1], vector[0])
        elevation = math.asin(max(-1.0, min(1.0, vector[2] / norm)))
        return azimuth, elevation


@dataclass(frozen=True)
class _RayRecord:
    delay_s: float
    shadow_gain: float
    departure_azimuth_rad: float
    departure_elevation_rad: float
    arrival_azimuth_rad: float
    arrival_elevation_rad: float
    doppler_hz: float


__all__ = ["Ray", "PropagationSnapshot", "PropagationModel"]