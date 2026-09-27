from __future__ import annotations
import math
from dataclasses import dataclass
from config.paper_parameters import PHYSICAL_CONSTANTS, SEA_SPECTRUM, WIND
from config.simulation_config import SimulationConfig, default_config
from gbsm import qmc


_GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class WaveComponent:
    angular_frequency_rad_s: float
    wavenumber_rad_m: float
    amplitude_m: float
    direction_rad: float
    phase_rad: float


@dataclass(frozen=True)
class WaveSpectrum:
    components: tuple[WaveComponent, ...]
    wind_speed_reference_m_s: float
    mean_direction_rad: float
    significant_wave_height_m: float


class SeaSpectrumModel:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()
        self._gravity = PHYSICAL_CONSTANTS.gravitational_acceleration_m_s2
        self._alpha = SEA_SPECTRUM.pierson_moskowitz_alpha
        self._beta = SEA_SPECTRUM.pierson_moskowitz_beta
        self._isotropic = SEA_SPECTRUM.directional_isotropic_term
        self._second_base = SEA_SPECTRUM.directional_second_harmonic_base
        self._second_gain = SEA_SPECTRUM.directional_second_harmonic_gain
        self._fourth_gain = SEA_SPECTRUM.directional_fourth_harmonic_gain
        self._frequency_bins = SEA_SPECTRUM.frequency_bin_count
        self._direction_bins = SEA_SPECTRUM.direction_bin_count
        self._min_frequency = SEA_SPECTRUM.min_angular_frequency_rad_s
        self._max_frequency = SEA_SPECTRUM.max_angular_frequency_rad_s
        self._half_width = SEA_SPECTRUM.spreading_half_width_rad
        self._min_wind = SEA_SPECTRUM.min_wind_speed_m_s
        self._reference_height = WIND.reference_height_m
        self._measurement_height = WIND.measurement_height_m
        self._roughness = WIND.sea_surface_roughness_m

    def frequency_spectrum(self, angular_frequency: float, wind_speed: float) -> float:
        wind = max(wind_speed, self._min_wind)
        ratio = self._gravity / (wind * angular_frequency)
        return (
            self._alpha
            * self._gravity**2
            / angular_frequency**5
            * math.exp(-self._beta * ratio**4)
        )

    def directional_spread(
        self, angular_frequency: float, offset: float, wind_speed: float
    ) -> float:
        if abs(offset) >= math.pi / 2.0:
            return 0.0
        wind = max(wind_speed, self._min_wind)
        peakedness = math.exp(-((angular_frequency * wind / self._gravity) ** 4) / 2.0)
        second = self._second_base + self._second_gain * peakedness
        fourth = self._fourth_gain * peakedness
        return (self._isotropic + second * math.cos(2.0 * offset) + fourth * math.cos(4.0 * offset)) / math.pi

    def wind_to_reference_height(self, wind_speed_measurement: float) -> float:
        if wind_speed_measurement <= 0.0:
            return 0.0
        return wind_speed_measurement * (
            math.log(self._reference_height / self._roughness)
            / math.log(self._measurement_height / self._roughness)
        )

    def build(
        self,
        wind_speed_reference: float,
        mean_direction_rad: float,
        target_height_m: float | None = None,
    ) -> WaveSpectrum:
        wind = max(wind_speed_reference, self._min_wind)
        frequencies, delta_frequency = self._frequency_grid()
        offsets, delta_offset = self._direction_grid()
        components: list[WaveComponent] = []
        index = 0
        for angular_frequency in frequencies:
            wavenumber = angular_frequency**2 / self._gravity
            spectral_density = self.frequency_spectrum(angular_frequency, wind)
            for offset in offsets:
                directional = self.directional_spread(angular_frequency, offset, wind)
                energy = 2.0 * spectral_density * directional * delta_frequency * delta_offset
                amplitude = math.sqrt(energy) if energy > 0.0 else 0.0
                components.append(
                    WaveComponent(
                        angular_frequency_rad_s=angular_frequency,
                        wavenumber_rad_m=wavenumber,
                        amplitude_m=amplitude,
                        direction_rad=mean_direction_rad + offset,
                        phase_rad=self._deterministic_phase(index),
                    )
                )
                index += 1
        return self._finalize(components, wind, mean_direction_rad, target_height_m)

    def from_sea_state(self, record) -> WaveSpectrum:
        wind = self.wind_to_reference_height(record.wind_speed_10m_m_s)
        mean_direction = self._mean_direction(record)
        return self.build(wind, mean_direction, record.significant_wave_height_m)

    def _finalize(
        self,
        components: list[WaveComponent],
        wind: float,
        mean_direction_rad: float,
        target_height_m: float | None,
    ) -> WaveSpectrum:
        modelled_height = 4.0 * math.sqrt(
            sum(component.amplitude_m**2 for component in components) / 2.0
        )
        if target_height_m is not None and target_height_m > 0.0 and modelled_height > 1.0e-6:
            scale = target_height_m / modelled_height
            components = [
                WaveComponent(
                    component.angular_frequency_rad_s,
                    component.wavenumber_rad_m,
                    component.amplitude_m * scale,
                    component.direction_rad,
                    component.phase_rad,
                )
                for component in components
            ]
            height = target_height_m
        else:
            height = modelled_height
        return WaveSpectrum(tuple(components), wind, mean_direction_rad, height)

    def _frequency_grid(self) -> tuple[list[float], float]:
        step = (self._max_frequency - self._min_frequency) / self._frequency_bins
        centers = [self._min_frequency + (i + 0.5) * step for i in range(self._frequency_bins)]
        return centers, step

    def _direction_grid(self) -> tuple[list[float], float]:
        step = 2.0 * self._half_width / self._direction_bins
        centers = [-self._half_width + (j + 0.5) * step for j in range(self._direction_bins)]
        return centers, step

    def _deterministic_phase(self, index: int) -> float:
        return 2.0 * math.pi * (qmc.rotate(((index + 1) * _GOLDEN_RATIO) % 1.0))

    @staticmethod
    def _mean_direction(record) -> float:
        compass = math.radians(record.mean_wave_from_direction_deg)
        return math.atan2(-math.cos(compass), -math.sin(compass))


__all__ = ["WaveComponent", "WaveSpectrum", "SeaSpectrumModel"]