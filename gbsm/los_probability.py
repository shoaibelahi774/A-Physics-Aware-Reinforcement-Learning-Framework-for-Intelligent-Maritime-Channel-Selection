from __future__ import annotations
import math
import numpy as np
from config.paper_parameters import BUILDINGS, PHYSICAL_CONSTANTS, SEA_SPECTRUM, BuildingParameters
from config.simulation_config import SimulationConfig, default_config


_ERF = np.vectorize(math.erf)


class LoSProbabilityModel:
    def __init__(
        self,
        config: SimulationConfig | None = None,
        building_parameters: BuildingParameters | None = None,
    ) -> None:
        self._config = config or default_config()
        self._buildings = building_parameters or BUILDINGS
        self._pm_alpha = SEA_SPECTRUM.pierson_moskowitz_alpha
        self._pm_beta = SEA_SPECTRUM.pierson_moskowitz_beta
        self._gravity = PHYSICAL_CONSTANTS.gravitational_acceleration_m_s2

    def wave_height_std(self, wind_speed_reference_m_s: float) -> float:
        numerator = self._pm_alpha * wind_speed_reference_m_s**4
        denominator = 4.0 * self._pm_beta * self._gravity**2
        return math.sqrt(numerator / denominator) if denominator > 0.0 else 0.0

    def los_probability(
        self,
        tx_position_enu_m: np.ndarray,
        rx_position_enu_m: np.ndarray,
        coastline_distance_m: float,
        wind_speed_reference_m_s: float,
    ) -> float:
        tx = np.asarray(tx_position_enu_m, dtype=float)
        rx = np.asarray(rx_position_enu_m, dtype=float)
        horizontal_distance = math.hypot(tx[0] - rx[0], tx[1] - rx[1])
        if horizontal_distance <= 0.0:
            return 1.0
        coastline = min(max(coastline_distance_m, 0.0), horizontal_distance)
        building = self._building_probability(tx[2], rx[2], horizontal_distance, coastline)
        sea = self._sea_probability(tx[2], rx[2], horizontal_distance, coastline, wind_speed_reference_m_s)
        return building * sea

    def _building_probability(self, tx_height, rx_height, horizontal_distance, coastline) -> float:
        if coastline <= 0.0:
            return 1.0
        area_ratio = self._buildings.occupied_area_ratio
        density = self._buildings.building_density_per_km2
        scale = self._buildings.rayleigh_height_scale_m
        building_count = int(math.floor(coastline * math.sqrt(area_ratio * density) / 1000.0))
        if building_count < 1 or scale <= 0.0:
            return 1.0
        width = 1000.0 * math.sqrt(area_ratio / density)
        indices = np.arange(1, building_count + 1)
        positions = (indices - 0.5) * coastline / building_count - width / 2.0
        path_height = self._path_height(tx_height, rx_height, horizontal_distance, positions)
        path_height = np.maximum(path_height, 0.0)
        probability = 1.0 - np.exp(-(path_height**2) / (2.0 * scale**2))
        return self._stable_product(probability)

    def _sea_probability(self, tx_height, rx_height, horizontal_distance, coastline, wind_speed) -> float:
        over_sea_distance = horizontal_distance - coastline
        if over_sea_distance <= 0.0:
            return 1.0
        deviation = self.wave_height_std(wind_speed)
        if deviation <= 0.0:
            return 1.0
        steps = np.arange(1, int(math.floor(over_sea_distance)) + 1)
        positions = coastline + steps
        path_height = self._path_height(tx_height, rx_height, horizontal_distance, positions)
        probability = 0.5 * (1.0 + _ERF(path_height / (deviation * math.sqrt(2.0))))
        return self._stable_product(probability)

    @staticmethod
    def _path_height(tx_height, rx_height, horizontal_distance, positions) -> np.ndarray:
        return rx_height + (positions / horizontal_distance) * (tx_height - rx_height)

    @staticmethod
    def _stable_product(probabilities: np.ndarray) -> float:
        clipped = np.clip(probabilities, 1.0e-12, 1.0)
        return float(np.exp(np.sum(np.log(clipped))))


__all__ = ["LoSProbabilityModel"]