from __future__ import annotations
import math
from dataclasses import dataclass
from statistics import NormalDist
from config.paper_parameters import EVAPORATION_DUCT, PHYSICAL_CONSTANTS
from config.simulation_config import SimulationConfig, default_config
from gbsm.geometry.scenarios import ClusterKind
from gbsm import qmc


_STANDARD_NORMAL = NormalDist()
_GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class ClusterAngleLimits:
    elevation_lower_rad: float
    elevation_upper_rad: float
    azimuth_lower_rad: float
    azimuth_upper_rad: float


class ClusterAngleModel:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()
        self._duct_height = EVAPORATION_DUCT.duct_height_m
        self._gradient = EVAPORATION_DUCT.refractivity_gradient_per_m
        self._surface_refractivity = PHYSICAL_CONSTANTS.sea_level_refractivity
        self._earth_radius = PHYSICAL_CONSTANTS.earth_radius_m

    def duct_flag(self, antenna_height_m: float) -> int:
        return 1 if antenna_height_m < self._duct_height else 0

    def duct_angle_limit(self, antenna_height_m: float) -> float:
        curvature = self._gradient / self._surface_refractivity + 1.0 / self._earth_radius
        argument = 2.0 * curvature * (antenna_height_m - self._duct_height)
        if argument <= 0.0:
            return 0.0
        return math.sqrt(argument)

    def angle_limits(
        self,
        kind: ClusterKind,
        antenna_height_m: float,
        los_azimuth_rad: float,
        on_land: bool,
    ) -> ClusterAngleLimits:
        if kind is ClusterKind.EVAPORATION_DUCT:
            limit = self.duct_angle_limit(antenna_height_m)
            return ClusterAngleLimits(-limit, limit, los_azimuth_rad - limit, los_azimuth_rad + limit)
        if kind is ClusterKind.SEA_SURFACE:
            upper = -self.duct_angle_limit(antenna_height_m)
            return ClusterAngleLimits(-math.pi / 2.0, upper, -math.pi, math.pi)
        if on_land:
            return ClusterAngleLimits(-math.pi / 2.0, math.pi / 2.0, -math.pi, math.pi)
        return ClusterAngleLimits(
            -math.pi / 2.0, math.pi / 2.0, los_azimuth_rad - math.pi / 2.0, los_azimuth_rad + math.pi / 2.0
        )

    def cluster_angles(
        self,
        kind: ClusterKind,
        count: int,
        antenna_height_m: float,
        los_azimuth_rad: float,
        on_land: bool,
        spread_factor: float = 0.25,
    ) -> list[tuple[float, float]]:
        limits = self.angle_limits(kind, antenna_height_m, los_azimuth_rad, on_land)
        return self.draw_cluster_angles(limits, count, spread_factor)

    def draw_cluster_angles(
        self, limits: ClusterAngleLimits, count: int, spread_factor: float = 0.25
    ) -> list[tuple[float, float]]:
        azimuth_span = limits.azimuth_upper_rad - limits.azimuth_lower_rad
        elevation_span = limits.elevation_upper_rad - limits.elevation_lower_rad
        azimuth = self._truncated_gaussian(
            count,
            0.5 * (limits.azimuth_lower_rad + limits.azimuth_upper_rad),
            spread_factor * azimuth_span,
            limits.azimuth_lower_rad,
            limits.azimuth_upper_rad,
            _GOLDEN_RATIO,
        )
        elevation = self._truncated_gaussian(
            count,
            0.5 * (limits.elevation_lower_rad + limits.elevation_upper_rad),
            spread_factor * elevation_span,
            limits.elevation_lower_rad,
            limits.elevation_upper_rad,
            0.0,
        )
        return list(zip(azimuth, elevation))

    @staticmethod
    def _truncated_gaussian(
        count: int, mean: float, std: float, lower: float, upper: float, quantile_offset: float
    ) -> list[float]:
        if count <= 0:
            return []
        if upper <= lower or std <= 0.0:
            return [0.5 * (lower + upper)] * count
        lower_cdf = _STANDARD_NORMAL.cdf((lower - mean) / std)
        upper_cdf = _STANDARD_NORMAL.cdf((upper - mean) / std)
        angles: list[float] = []
        for index in range(count):
            quantile = qmc.rotate(((index + 0.5) / count + quantile_offset) % 1.0)
            probability = min(max(lower_cdf + quantile * (upper_cdf - lower_cdf), 1.0e-9), 1.0 - 1.0e-9)
            angles.append(mean + std * _STANDARD_NORMAL.inv_cdf(probability))
        return angles


__all__ = ["ClusterAngleLimits", "ClusterAngleModel"]