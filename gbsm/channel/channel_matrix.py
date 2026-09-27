from __future__ import annotations
import numpy as np

from config.simulation_config import SimulationConfig, default_config
from gbsm.channel.channel_impulse_response import ChannelImpulseResponse


class ChannelMatrixModel:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()
        self._wavenumber = 2.0 * np.pi / self._config.wavelength_m

    def build(
        self,
        impulse_response: ChannelImpulseResponse,
        tx_offsets_m: np.ndarray,
        rx_offsets_m: np.ndarray,
        t: float = 0.0,
        frequency_hz: float = 0.0,
    ) -> np.ndarray:
        """Return the N_t x N_r channel matrix, normalised so that
        ||H||_F^2 = N_t * N_r (i.e. E[|h_ij|^2] = 1) for every realisation.

        H therefore represents only the *spatial structure* of the channel;
        the large-scale gain (path loss / operating point) is carried
        separately by the SNR used in the capacity computation. This makes the
        SNR parameter's meaning exact (average per-branch SNR) and the absolute
        capacity reproducible, instead of depending on the incidental scale of
        the polarisation/steering products.
        """
        taps = impulse_response.taps
        tx_offsets = np.asarray(tx_offsets_m, dtype=float)
        rx_offsets = np.asarray(rx_offsets_m, dtype=float)
        if not taps:
            return np.zeros((tx_offsets.shape[0], rx_offsets.shape[0]), dtype=complex)
        amplitudes = np.array([tap.amplitude for tap in taps], dtype=complex)
        delays = np.array([tap.delay_s for tap in taps])
        dopplers = np.array([tap.doppler_hz for tap in taps])
        departure = np.array([tap.departure_direction for tap in taps])
        arrival = np.array([tap.arrival_direction for tap in taps])
        base = amplitudes * np.exp(2j * np.pi * dopplers * t) * np.exp(-2j * np.pi * frequency_hz * delays)
        transmit_steering = np.exp(1j * self._wavenumber * (tx_offsets @ departure.T))
        receive_steering = np.exp(1j * self._wavenumber * (rx_offsets @ arrival.T))
        matrix = (transmit_steering * base) @ receive_steering.T
        return self._normalize(matrix)

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        power = float(np.sum(np.abs(matrix) ** 2))
        if power <= 0.0:
            return matrix
        target = float(matrix.shape[0] * matrix.shape[1])  # N_t * N_r
        return matrix * np.sqrt(target / power)

    def frequency_response(
        self,
        impulse_response: ChannelImpulseResponse,
        tx_offsets_m: np.ndarray,
        rx_offsets_m: np.ndarray,
        frequencies_hz: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        return np.stack(
            [self.build(impulse_response, tx_offsets_m, rx_offsets_m, t, float(frequency)) for frequency in frequencies_hz]
        )


__all__ = ["ChannelMatrixModel"]