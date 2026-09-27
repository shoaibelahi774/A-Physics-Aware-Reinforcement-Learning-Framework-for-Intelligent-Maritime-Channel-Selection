from __future__ import annotations
import math
from enum import Enum
import numpy as np
from config.simulation_config import SimulationConfig, default_config
from gbsm.channel.channel_impulse_response import ChannelImpulseResponse


class AngularDomain(Enum):
    AZIMUTH_DEPARTURE = "azimuth_departure"
    ELEVATION_DEPARTURE = "elevation_departure"
    AZIMUTH_ARRIVAL = "azimuth_arrival"
    ELEVATION_ARRIVAL = "elevation_arrival"


class AngularPSDModel:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()

    def ray_angles(self, impulse_response: ChannelImpulseResponse, domain: AngularDomain) -> tuple[np.ndarray, np.ndarray]:
        angles = []
        powers = []
        for tap in impulse_response.taps:
            direction = tap.departure_direction if "departure" in domain.value else tap.arrival_direction
            angles.append(self._extract_angle(direction, domain))
            powers.append(abs(tap.amplitude) ** 2)
        return np.array(angles), np.array(powers)

    def angular_spectrum(
        self,
        impulse_response: ChannelImpulseResponse,
        domain: AngularDomain,
        angle_grid_rad: np.ndarray,
        bandwidth_rad: float | None = None,
    ) -> np.ndarray:
        angles, powers = self.ray_angles(impulse_response, domain)
        grid = np.asarray(angle_grid_rad, dtype=float)
        if angles.size == 0:
            return np.zeros_like(grid)
        if bandwidth_rad is None:
            bandwidth_rad = self._default_bandwidth(grid)
        circular = self._is_azimuth(domain)
        density = np.zeros_like(grid)
        for angle, power in zip(angles, powers):
            difference = self._angular_difference(grid, angle, circular)
            density += power * np.exp(-0.5 * (difference / bandwidth_rad) ** 2)
        total = density.sum()
        return density / total if total > 0.0 else density

    def dominant_angle(self, impulse_response: ChannelImpulseResponse, domain: AngularDomain) -> float:
        angles, powers = self.ray_angles(impulse_response, domain)
        if powers.sum() <= 0.0:
            return 0.0
        if self._is_azimuth(domain):
            sine = float(np.sum(powers * np.sin(angles)))
            cosine = float(np.sum(powers * np.cos(angles)))
            return math.atan2(sine, cosine)
        return float(np.sum(powers * angles) / powers.sum())

    def angular_spread(self, impulse_response: ChannelImpulseResponse, domain: AngularDomain) -> float:
        angles, powers = self.ray_angles(impulse_response, domain)
        total = powers.sum()
        if total <= 0.0:
            return 0.0
        mean = self.dominant_angle(impulse_response, domain)
        difference = self._angular_difference(angles, mean, self._is_azimuth(domain))
        return float(math.sqrt(np.sum(powers * difference**2) / total))

    @staticmethod
    def _extract_angle(direction: np.ndarray, domain: AngularDomain) -> float:
        if domain in (AngularDomain.AZIMUTH_DEPARTURE, AngularDomain.AZIMUTH_ARRIVAL):
            return math.atan2(direction[1], direction[0])
        return math.asin(max(-1.0, min(1.0, direction[2])))

    @staticmethod
    def _is_azimuth(domain: AngularDomain) -> bool:
        return domain in (AngularDomain.AZIMUTH_DEPARTURE, AngularDomain.AZIMUTH_ARRIVAL)

    @staticmethod
    def _angular_difference(grid: np.ndarray, angle: float, circular: bool) -> np.ndarray:
        difference = grid - angle
        if circular:
            difference = (difference + math.pi) % (2.0 * math.pi) - math.pi
        return difference

    @staticmethod
    def _default_bandwidth(grid: np.ndarray) -> float:
        if grid.size < 2:
            return 0.05
        return float(abs(grid[1] - grid[0]) * 2.0)


__all__ = ["AngularDomain", "AngularPSDModel"]