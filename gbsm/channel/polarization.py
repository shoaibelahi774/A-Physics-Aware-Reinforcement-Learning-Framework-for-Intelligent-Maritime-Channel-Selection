from __future__ import annotations
import math
from statistics import NormalDist

import numpy as np

from config.paper_parameters import POLARIZATION
from config.simulation_config import SimulationConfig, default_config
from gbsm import qmc


_STANDARD_NORMAL = NormalDist()
_GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0
_LOCAL_POLARIZATION = np.array([0.0, 0.0, 1.0])


class PolarizationModel:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()
        self._xpr_mean_db = POLARIZATION.cross_polarization_ratio_mean_db
        self._xpr_std_db = POLARIZATION.cross_polarization_ratio_std_db
        self._imbalance = POLARIZATION.copolar_imbalance

    def antenna_field(self, azimuth_rad: float, elevation_rad: float, rotation: np.ndarray) -> np.ndarray:
        polarization = np.asarray(rotation, dtype=float) @ _LOCAL_POLARIZATION
        elevation_basis, azimuth_basis = self._spherical_basis(azimuth_rad, elevation_rad)
        return np.array([float(polarization @ elevation_basis), float(polarization @ azimuth_basis)])

    def polarization_matrix(self, ray_index: int, is_line_of_sight: bool) -> np.ndarray:
        vv, hh, vh, hv = self._phases(ray_index)
        if is_line_of_sight:
            return np.array([[np.exp(1j * vv), 0.0], [0.0, -np.exp(1j * hh)]], dtype=complex)
        inverse_root = 1.0 / math.sqrt(self._cross_polarization_ratio(ray_index))
        imbalance_root = math.sqrt(self._imbalance)
        return np.array(
            [
                [np.exp(1j * vv), imbalance_root * inverse_root * np.exp(1j * vh)],
                [inverse_root * np.exp(1j * hv), imbalance_root * np.exp(1j * hh)],
            ],
            dtype=complex,
        )

    def response(
        self,
        departure_azimuth_rad: float,
        departure_elevation_rad: float,
        tx_rotation: np.ndarray,
        arrival_azimuth_rad: float,
        arrival_elevation_rad: float,
        rx_rotation: np.ndarray,
        ray_index: int,
        is_line_of_sight: bool,
    ) -> complex:
        transmit_field = self.antenna_field(departure_azimuth_rad, departure_elevation_rad, tx_rotation)
        receive_field = self.antenna_field(arrival_azimuth_rad, arrival_elevation_rad, rx_rotation)
        matrix = self.polarization_matrix(ray_index, is_line_of_sight)
        return complex(receive_field @ matrix @ transmit_field)

    def _cross_polarization_ratio(self, ray_index: int) -> float:
        quantile = min(max(qmc.rotate(((ray_index + 1) * _GOLDEN_RATIO) % 1.0), 1.0e-6), 1.0 - 1.0e-6)
        exponent = self._xpr_mean_db + self._xpr_std_db * _STANDARD_NORMAL.inv_cdf(quantile)
        return 10.0 ** (exponent / 10.0)

    @staticmethod
    def _phases(ray_index: int) -> tuple[float, float, float, float]:
        base = ray_index + 1
        return (
            2.0 * math.pi * (qmc.rotate((base * _GOLDEN_RATIO) % 1.0)),
            2.0 * math.pi * (qmc.rotate((base * _GOLDEN_RATIO**2) % 1.0)),
            2.0 * math.pi * (qmc.rotate((base * _GOLDEN_RATIO**3) % 1.0)),
            2.0 * math.pi * (qmc.rotate((base * _GOLDEN_RATIO**4) % 1.0)),
        )

    @staticmethod
    def _spherical_basis(azimuth_rad: float, elevation_rad: float) -> tuple[np.ndarray, np.ndarray]:
        elevation_basis = np.array(
            [
                -math.sin(elevation_rad) * math.cos(azimuth_rad),
                -math.sin(elevation_rad) * math.sin(azimuth_rad),
                math.cos(elevation_rad),
            ]
        )
        azimuth_basis = np.array([-math.sin(azimuth_rad), math.cos(azimuth_rad), 0.0])
        return elevation_basis, azimuth_basis


__all__ = ["PolarizationModel"]