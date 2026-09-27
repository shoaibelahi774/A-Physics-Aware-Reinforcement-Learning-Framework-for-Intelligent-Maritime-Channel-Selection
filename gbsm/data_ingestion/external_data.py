from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

import math

import numpy as np

from config.simulation_config import SimulationConfig, default_config
from gbsm.data_ingestion.sea_state_loader import SeaStateSeries

if TYPE_CHECKING:
    from gbsm.data_ingestion.building_ingestion import BuildingInventory
    from gbsm.geometry.satellite_dynamics import SatelliteEphemeris


_WGS84_SEMI_MAJOR_AXIS_M = 6_378_137.0
_WGS84_FLATTENING = 1.0 / 298.257223563
_WGS84_ECCENTRICITY_SQ = _WGS84_FLATTENING * (2.0 - _WGS84_FLATTENING)


@dataclass(frozen=True)
class GeodeticPoint:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0


class ENUReferenceFrame:
    def __init__(self, origin: GeodeticPoint) -> None:
        self._origin = origin
        latitude = math.radians(origin.latitude_deg)
        longitude = math.radians(origin.longitude_deg)
        self._origin_ecef = self._geodetic_to_ecef(
            origin.latitude_deg, origin.longitude_deg, origin.altitude_m
        )
        self._rotation = np.array(
            [
                [-math.sin(longitude), math.cos(longitude), 0.0],
                [
                    -math.sin(latitude) * math.cos(longitude),
                    -math.sin(latitude) * math.sin(longitude),
                    math.cos(latitude),
                ],
                [
                    math.cos(latitude) * math.cos(longitude),
                    math.cos(latitude) * math.sin(longitude),
                    math.sin(latitude),
                ],
            ]
        )

    @property
    def origin(self) -> GeodeticPoint:
        return self._origin

    def geodetic_to_enu(
        self, latitude_deg: float, longitude_deg: float, altitude_m: float = 0.0
    ) -> np.ndarray:
        ecef = self._geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m)
        return self._rotation @ (ecef - self._origin_ecef)

    def enu_to_geodetic(self, enu: np.ndarray) -> GeodeticPoint:
        ecef = self._origin_ecef + self._rotation.T @ np.asarray(enu, dtype=float)
        return self._ecef_to_geodetic(ecef)

    def ecef_to_enu(self, ecef: np.ndarray) -> np.ndarray:
        return self._rotation @ (np.asarray(ecef, dtype=float) - self._origin_ecef)

    def rotate_ecef_to_enu(self, vector: np.ndarray) -> np.ndarray:
        return self._rotation @ np.asarray(vector, dtype=float)

    @staticmethod
    def _geodetic_to_ecef(latitude_deg: float, longitude_deg: float, altitude_m: float) -> np.ndarray:
        latitude = math.radians(latitude_deg)
        longitude = math.radians(longitude_deg)
        prime_vertical = _WGS84_SEMI_MAJOR_AXIS_M / math.sqrt(
            1.0 - _WGS84_ECCENTRICITY_SQ * math.sin(latitude) ** 2
        )
        return np.array(
            [
                (prime_vertical + altitude_m) * math.cos(latitude) * math.cos(longitude),
                (prime_vertical + altitude_m) * math.cos(latitude) * math.sin(longitude),
                (prime_vertical * (1.0 - _WGS84_ECCENTRICITY_SQ) + altitude_m) * math.sin(latitude),
            ]
        )

    @staticmethod
    def _ecef_to_geodetic(ecef: np.ndarray) -> GeodeticPoint:
        x, y, z = float(ecef[0]), float(ecef[1]), float(ecef[2])
        longitude = math.atan2(y, x)
        planar = math.hypot(x, y)
        latitude = math.atan2(z, planar * (1.0 - _WGS84_ECCENTRICITY_SQ))
        for _ in range(6):
            prime_vertical = _WGS84_SEMI_MAJOR_AXIS_M / math.sqrt(
                1.0 - _WGS84_ECCENTRICITY_SQ * math.sin(latitude) ** 2
            )
            altitude = planar / math.cos(latitude) - prime_vertical
            latitude = math.atan2(
                z, planar * (1.0 - _WGS84_ECCENTRICITY_SQ * prime_vertical / (prime_vertical + altitude))
            )
        prime_vertical = _WGS84_SEMI_MAJOR_AXIS_M / math.sqrt(
            1.0 - _WGS84_ECCENTRICITY_SQ * math.sin(latitude) ** 2
        )
        altitude = planar / math.cos(latitude) - prime_vertical
        return GeodeticPoint(math.degrees(latitude), math.degrees(longitude), altitude)


@dataclass(frozen=True)
class KinematicSample:
    timestamp: datetime
    position_enu_m: np.ndarray
    velocity_enu_m_s: np.ndarray

    @property
    def speed_m_s(self) -> float:
        return float(np.linalg.norm(self.velocity_enu_m_s))


class Trajectory:
    def __init__(self, identifier: str, samples: Sequence[KinematicSample]) -> None:
        if not samples:
            raise ValueError(f"Trajectory '{identifier}' has no kinematic samples.")
        ordered = sorted(samples, key=lambda sample: sample.timestamp)
        self._identifier = identifier
        self._samples = tuple(ordered)
        self._epochs = np.array(
            [sample.timestamp.timestamp() for sample in ordered], dtype=float
        )

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def samples(self) -> tuple[KinematicSample, ...]:
        return self._samples

    @property
    def start_time(self) -> datetime:
        return self._samples[0].timestamp

    @property
    def end_time(self) -> datetime:
        return self._samples[-1].timestamp

    def sample_at(self, moment: datetime) -> KinematicSample:
        target = moment.timestamp()
        if target <= self._epochs[0]:
            return self._samples[0]
        if target >= self._epochs[-1]:
            return self._samples[-1]
        upper = int(np.searchsorted(self._epochs, target))
        lower = upper - 1
        span = self._epochs[upper] - self._epochs[lower]
        weight = 0.0 if span == 0.0 else (target - self._epochs[lower]) / span
        start = self._samples[lower]
        end = self._samples[upper]
        position = start.position_enu_m + weight * (end.position_enu_m - start.position_enu_m)
        velocity = start.velocity_enu_m_s + weight * (end.velocity_enu_m_s - start.velocity_enu_m_s)
        return KinematicSample(moment, position, velocity)


@dataclass(frozen=True)
class TelemetryObservation:
    timestamp: datetime
    source: str
    fields: Mapping[str, float]


@dataclass(frozen=True)
class ExternalDataBundle:
    sea_state: SeaStateSeries
    reference_frame: ENUReferenceFrame
    trajectories: Mapping[str, Trajectory] = field(default_factory=dict)
    telemetry: Sequence[TelemetryObservation] = field(default_factory=tuple)
    building_inventory: Optional["BuildingInventory"] = None
    satellite_ephemeris: Optional["SatelliteEphemeris"] = None
    provenance: Mapping[str, str] = field(default_factory=dict)


def build_reference_frame(config: SimulationConfig | None = None) -> ENUReferenceFrame:
    resolved = config or default_config()
    origin = GeodeticPoint(
        latitude_deg=resolved.reference_latitude_deg,
        longitude_deg=resolved.reference_longitude_deg,
        altitude_m=0.0,
    )
    return ENUReferenceFrame(origin)


__all__ = [
    "GeodeticPoint",
    "ENUReferenceFrame",
    "KinematicSample",
    "Trajectory",
    "TelemetryObservation",
    "ExternalDataBundle",
    "build_reference_frame",
]