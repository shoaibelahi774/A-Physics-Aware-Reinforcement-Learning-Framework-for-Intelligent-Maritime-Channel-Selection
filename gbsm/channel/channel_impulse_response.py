from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np

from config.simulation_config import SimulationConfig, default_config
from gbsm.channel.polarization import PolarizationModel
from gbsm.channel.power_ratios import ComponentWeights
from gbsm.channel.propagation import PropagationSnapshot, Ray


@dataclass(frozen=True)
class ChannelTap:
    delay_s: float
    amplitude: complex
    doppler_hz: float
    departure_direction: np.ndarray
    arrival_direction: np.ndarray


@dataclass(frozen=True)
class ChannelImpulseResponse:
    taps: tuple[ChannelTap, ...]

    def narrowband_gain(self, t: float = 0.0) -> complex:
        return sum(
            (tap.amplitude * cmath.exp(2j * math.pi * tap.doppler_hz * t) for tap in self.taps),
            0j,
        )

    def transfer_function(self, frequency_hz: float, t: float = 0.0) -> complex:
        return sum(
            (
                tap.amplitude
                * cmath.exp(2j * math.pi * tap.doppler_hz * t)
                * cmath.exp(-2j * math.pi * frequency_hz * tap.delay_s)
                for tap in self.taps
            ),
            0j,
        )

    def power(self) -> float:
        return float(sum(abs(tap.amplitude) ** 2 for tap in self.taps))

    @property
    def delays_s(self) -> np.ndarray:
        return np.array([tap.delay_s for tap in self.taps])

    @property
    def powers(self) -> np.ndarray:
        return np.array([abs(tap.amplitude) ** 2 for tap in self.taps])


class ChannelImpulseResponseModel:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()
        self._carrier = self._config.carrier_frequency_hz
        self._polarization = PolarizationModel(self._config)

    def assemble(
        self,
        snapshot: PropagationSnapshot,
        weights: ComponentWeights,
        tx_rotation: np.ndarray,
        rx_rotation: np.ndarray,
        los_present: bool,
    ) -> ChannelImpulseResponse:
        taps: list[ChannelTap] = []
        counter = _Counter()
        if los_present and weights.line_of_sight > 0.0:
            taps.append(
                self._tap(snapshot.los, weights.line_of_sight, 1.0, tx_rotation, rx_rotation, counter, True)
            )
        self._append_component(taps, snapshot.reflection, weights.reflection, tx_rotation, rx_rotation, counter)
        self._append_component(taps, snapshot.nlos1, weights.nlos1, tx_rotation, rx_rotation, counter)
        self._append_component(taps, snapshot.nlos2, weights.nlos2, tx_rotation, rx_rotation, counter)
        return ChannelImpulseResponse(tuple(taps))

    def _append_component(
        self,
        taps: list[ChannelTap],
        rays: tuple[Ray, ...],
        weight: float,
        tx_rotation: np.ndarray,
        rx_rotation: np.ndarray,
        counter: "_Counter",
    ) -> None:
        if weight <= 0.0:
            for _ in rays:
                counter.next()
            return
        for ray in rays:
            taps.append(self._tap(ray, weight, math.sqrt(ray.power), tx_rotation, rx_rotation, counter, False))

    def _tap(
        self,
        ray: Ray,
        weight: float,
        amplitude_gain: float,
        tx_rotation: np.ndarray,
        rx_rotation: np.ndarray,
        counter: "_Counter",
        is_line_of_sight: bool,
    ) -> ChannelTap:
        ray_index = counter.next()
        response = self._polarization.response(
            ray.departure_azimuth_rad,
            ray.departure_elevation_rad,
            tx_rotation,
            ray.arrival_azimuth_rad,
            ray.arrival_elevation_rad,
            rx_rotation,
            ray_index,
            is_line_of_sight,
        )
        carrier_phase = cmath.exp(2j * math.pi * self._carrier * ray.delay_s)
        amplitude = weight * amplitude_gain * response * carrier_phase
        return ChannelTap(
            delay_s=ray.delay_s,
            amplitude=amplitude,
            doppler_hz=ray.doppler_hz,
            departure_direction=_unit_direction(ray.departure_azimuth_rad, ray.departure_elevation_rad),
            arrival_direction=_unit_direction(ray.arrival_azimuth_rad, ray.arrival_elevation_rad),
        )


def _unit_direction(azimuth_rad: float, elevation_rad: float) -> np.ndarray:
    cosine_elevation = math.cos(elevation_rad)
    return np.array(
        [cosine_elevation * math.cos(azimuth_rad), cosine_elevation * math.sin(azimuth_rad), math.sin(elevation_rad)]
    )


class _Counter:
    def __init__(self) -> None:
        self._value = 0

    def next(self) -> int:
        current = self._value
        self._value += 1
        return current


__all__ = ["ChannelTap", "ChannelImpulseResponse", "ChannelImpulseResponseModel"]