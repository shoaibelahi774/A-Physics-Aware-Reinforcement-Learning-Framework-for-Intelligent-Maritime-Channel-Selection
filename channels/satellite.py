from __future__ import annotations
from math import isfinite
from channels.channel import Channel

class GEOSatellite(Channel):
    def __init__(self) -> None:
        super().__init__(
            name="GEO Satellite",
            latency=600.0,
            bandwidth=20.0,
            cost=5.0,
            stability=0.95,
        )

class LEOSatellite(Channel):
    SPEED_OF_LIGHT_MPS = 299_792_458.0

    def __init__(self) -> None:
        super().__init__(
            name="LEO Satellite",
            latency=80.0,
            bandwidth=50.0,
            cost=3.0,
            stability=0.85,
        )
        self.leo_doppler_hz = 0.0
        self.leo_doppler_penalty = 0.0

    @staticmethod
    def _finite(name: str, value: float) -> float:
        result = float(value)
        if not isfinite(result):
            raise ValueError(f"{name} must be finite.")
        return result

    @classmethod
    def calculate_doppler_shift_hz(
        cls,
        carrier_frequency_hz: float,
        radial_relative_velocity_mps: float,
    ) -> float:
        frequency = cls._finite("carrier_frequency_hz", carrier_frequency_hz)
        velocity = cls._finite(
            "radial_relative_velocity_mps", radial_relative_velocity_mps
        )
        if frequency <= 0.0:
            raise ValueError("carrier_frequency_hz must be positive.")
        return -frequency * velocity / cls.SPEED_OF_LIGHT_MPS

    @classmethod
    def calculate_doppler_penalty(
        cls,
        doppler_shift_hz: float,
        reference_doppler_hz: float,
    ) -> float:
        shift = cls._finite("doppler_shift_hz", doppler_shift_hz)
        reference = cls._finite("reference_doppler_hz", reference_doppler_hz)
        if reference <= 0.0:
            raise ValueError("reference_doppler_hz must be positive.")
        return min(1.0, abs(shift) / reference)

    def update_doppler(
        self,
        carrier_frequency_hz: float,
        radial_relative_velocity_mps: float,
        reference_doppler_hz: float,
    ) -> tuple[float, float]:
        shift = self.calculate_doppler_shift_hz(
            carrier_frequency_hz, radial_relative_velocity_mps
        )
        penalty = self.calculate_doppler_penalty(shift, reference_doppler_hz)
        self.leo_doppler_hz = shift
        self.leo_doppler_penalty = penalty
        return shift, penalty
