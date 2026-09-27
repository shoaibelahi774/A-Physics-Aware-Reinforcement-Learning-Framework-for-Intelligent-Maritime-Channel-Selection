from __future__ import annotations
import numpy as np
from config.simulation_config import SimulationConfig
from gbsm.environment.sea_spectrum import SeaSpectrumModel, WaveSpectrum


class SeaSurface:
    def __init__(self, spectrum: WaveSpectrum) -> None:
        components = spectrum.components
        self._spectrum = spectrum
        self._amplitude = np.array([c.amplitude_m for c in components])
        self._angular_frequency = np.array([c.angular_frequency_rad_s for c in components])
        self._wavenumber = np.array([c.wavenumber_rad_m for c in components])
        self._cos_direction = np.array([np.cos(c.direction_rad) for c in components])
        self._sin_direction = np.array([np.sin(c.direction_rad) for c in components])
        self._phase = np.array([c.phase_rad for c in components])

    @classmethod
    def from_sea_state(cls, record, config: SimulationConfig | None = None) -> "SeaSurface":
        return cls(SeaSpectrumModel(config).from_sea_state(record))

    @property
    def spectrum(self) -> WaveSpectrum:
        return self._spectrum

    @property
    def significant_wave_height_m(self) -> float:
        return self._spectrum.significant_wave_height_m

    def elevation(self, x: float, y: float, t: float) -> float:
        return float(np.sum(self._amplitude * np.cos(self._phase_angle(x, y, t))))

    def elevations(self, points_xy: np.ndarray, t: float) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points_xy, dtype=float))
        x = points[:, 0][:, None]
        y = points[:, 1][:, None]
        angle = (
            self._angular_frequency[None, :] * t
            - self._wavenumber[None, :] * (x * self._cos_direction[None, :] + y * self._sin_direction[None, :])
            + self._phase[None, :]
        )
        return (self._amplitude[None, :] * np.cos(angle)).sum(axis=1)

    def slope(self, x: float, y: float, t: float) -> np.ndarray:
        sine = np.sin(self._phase_angle(x, y, t))
        gain = self._amplitude * self._wavenumber * sine
        return np.array(
            [float(np.sum(gain * self._cos_direction)), float(np.sum(gain * self._sin_direction))]
        )

    def surface_normal(self, x: float, y: float, t: float) -> np.ndarray:
        gradient = self.slope(x, y, t)
        normal = np.array([-gradient[0], -gradient[1], 1.0])
        return normal / np.linalg.norm(normal)

    def orbital_velocity(self, x: float, y: float, t: float) -> np.ndarray:
        angle = self._phase_angle(x, y, t)
        horizontal = self._amplitude * self._angular_frequency * np.cos(angle)
        vertical = -self._amplitude * self._angular_frequency * np.sin(angle)
        return np.array(
            [
                float(np.sum(horizontal * self._cos_direction)),
                float(np.sum(horizontal * self._sin_direction)),
                float(np.sum(vertical)),
            ]
        )

    def vertical_velocity(self, x: float, y: float, t: float) -> float:
        return -float(np.sum(self._amplitude * self._angular_frequency * np.sin(self._phase_angle(x, y, t))))

    def _phase_angle(self, x: float, y: float, t: float) -> np.ndarray:
        return (
            self._angular_frequency * t
            - self._wavenumber * (x * self._cos_direction + y * self._sin_direction)
            + self._phase
        )


def build_sea_surface(record, config: SimulationConfig | None = None) -> SeaSurface:
    return SeaSurface.from_sea_state(record, config)


__all__ = ["SeaSurface", "build_sea_surface"]