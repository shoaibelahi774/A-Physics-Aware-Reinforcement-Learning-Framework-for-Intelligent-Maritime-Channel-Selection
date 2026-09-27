from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from config.paper_parameters import UAV_ROTATION
from gbsm.data_ingestion.external_data import KinematicSample
from gbsm.environment.sea_surface import SeaSurface


@dataclass(frozen=True)
class AntennaArrayState:
    positions_enu_m: np.ndarray
    rotation: np.ndarray
    origin_enu_m: np.ndarray


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    rotation_z = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(pitch), -math.sin(pitch)], [0.0, math.sin(pitch), math.cos(pitch)]]
    )
    rotation_y = np.array(
        [[math.cos(roll), 0.0, math.sin(roll)], [0.0, 1.0, 0.0], [-math.sin(roll), 0.0, math.cos(roll)]]
    )
    return rotation_z @ rotation_x @ rotation_y


def element_offsets(count: int, spacing_m: float, azimuth_rad: float, elevation_rad: float) -> np.ndarray:
    direction = np.array(
        [
            math.cos(azimuth_rad) * math.cos(elevation_rad),
            math.sin(azimuth_rad) * math.cos(elevation_rad),
            math.sin(elevation_rad),
        ]
    )
    return np.array([spacing_m * index * direction for index in range(count)])


@dataclass(frozen=True)
class UAVWobble:
    pitch_amplitude_rad: float = UAV_ROTATION.pitch_amplitude_rad
    roll_amplitude_rad: float = UAV_ROTATION.roll_amplitude_rad
    yaw_amplitude_rad: float = UAV_ROTATION.yaw_amplitude_rad
    pitch_frequency_hz: float = UAV_ROTATION.pitch_frequency_hz
    roll_frequency_hz: float = UAV_ROTATION.roll_frequency_hz
    yaw_frequency_hz: float = UAV_ROTATION.yaw_frequency_hz
    pitch_phase_rad: float = UAV_ROTATION.pitch_phase_rad
    roll_phase_rad: float = UAV_ROTATION.roll_phase_rad
    yaw_phase_rad: float = UAV_ROTATION.yaw_phase_rad

    def angles(self, t: float) -> tuple[float, float, float]:
        pitch = self.pitch_amplitude_rad * math.cos(2.0 * math.pi * self.pitch_frequency_hz * t + self.pitch_phase_rad)
        roll = self.roll_amplitude_rad * math.cos(2.0 * math.pi * self.roll_frequency_hz * t + self.roll_phase_rad)
        yaw = self.yaw_amplitude_rad * math.cos(2.0 * math.pi * self.yaw_frequency_hz * t + self.yaw_phase_rad)
        return yaw, pitch, roll


class LandArray:
    def __init__(self, origin_enu_m: np.ndarray, offsets: np.ndarray) -> None:
        self._origin = np.asarray(origin_enu_m, dtype=float)
        self._offsets = np.asarray(offsets, dtype=float)

    def state(self) -> AntennaArrayState:
        return AntennaArrayState(
            positions_enu_m=self._origin + self._offsets,
            rotation=np.eye(3),
            origin_enu_m=self._origin,
        )

    def state_at(self, sample: KinematicSample | None = None, t: float = 0.0) -> AntennaArrayState:
        return self.state()


class ShipArray:
    def __init__(
        self,
        sea_surface: SeaSurface,
        offsets: np.ndarray,
        length_m: float,
        width_m: float,
        antenna_height_m: float,
    ) -> None:
        self._sea = sea_surface
        self._offsets = np.asarray(offsets, dtype=float)
        self._length = length_m
        self._width = width_m
        self._height = antenna_height_m

    def state_at(self, sample: KinematicSample, t: float) -> AntennaArrayState:
        east, north = float(sample.position_enu_m[0]), float(sample.position_enu_m[1])
        yaw = self._heading(sample.velocity_enu_m_s)
        elevation_a = self._sea.elevation(east, north, t)
        origin = np.array([east, north, elevation_a])
        pitch = self._pitch(east, north, yaw, elevation_a, t)
        roll = self._roll(east, north, yaw, elevation_a, t)
        rotation = rotation_matrix(yaw, pitch, roll)
        local = np.array([0.0, self._length, self._height]) + self._offsets
        positions = (rotation @ local.T).T + origin
        return AntennaArrayState(positions_enu_m=positions, rotation=rotation, origin_enu_m=origin)

    def _pitch(self, east: float, north: float, yaw: float, elevation_a: float, t: float) -> float:
        bow_east = east - self._length * math.sin(yaw)
        bow_north = north + self._length * math.cos(yaw)
        elevation_b = self._sea.elevation(bow_east, bow_north, t)
        return math.asin(_clip((elevation_b - elevation_a) / self._length))

    def _roll(self, east: float, north: float, yaw: float, elevation_a: float, t: float) -> float:
        corner_east = east - 0.5 * self._width * math.cos(yaw)
        corner_north = north - 0.5 * self._width * math.sin(yaw)
        elevation_c = self._sea.elevation(corner_east, corner_north, t)
        return math.asin(_clip(2.0 * (elevation_c - elevation_a) / self._width))

    @staticmethod
    def _heading(velocity_enu: np.ndarray) -> float:
        east, north = float(velocity_enu[0]), float(velocity_enu[1])
        if east == 0.0 and north == 0.0:
            return 0.0
        return math.atan2(north, east) - math.pi / 2.0


class UAVArray:
    def __init__(self, offsets: np.ndarray, wobble: UAVWobble | None = None) -> None:
        self._offsets = np.asarray(offsets, dtype=float)
        self._wobble = wobble or UAVWobble()

    def state_at(self, sample: KinematicSample, t: float) -> AntennaArrayState:
        yaw, pitch, roll = self._wobble.angles(t)
        rotation = rotation_matrix(yaw, pitch, roll)
        origin = np.asarray(sample.position_enu_m, dtype=float)
        positions = (rotation @ self._offsets.T).T + origin
        return AntennaArrayState(positions_enu_m=positions, rotation=rotation, origin_enu_m=origin)


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


__all__ = [
    "AntennaArrayState",
    "rotation_matrix",
    "element_offsets",
    "UAVWobble",
    "LandArray",
    "ShipArray",
    "UAVArray",
]