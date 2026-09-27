from __future__ import annotations
import numpy as np
from config.paper_parameters import CLUSTER_EVOLUTION
from config.simulation_config import SimulationConfig, default_config
from gbsm.channel.channel_impulse_response import ChannelImpulseResponse


class STFCorrelationModel:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()
        self._wavenumber = 2.0 * np.pi / self._config.wavelength_m
        self._evolution = CLUSTER_EVOLUTION

    def stf_cf(
        self,
        impulse_response: ChannelImpulseResponse,
        delta_t: float = 0.0,
        delta_xi_tx: np.ndarray | None = None,
        delta_xi_rx: np.ndarray | None = None,
        delta_f: float = 0.0,
        terminal_speed_m_s: float = 0.0,
        los_tap_count: int = 0,
    ) -> complex:
        taps = impulse_response.taps
        if not taps:
            return 0j
        delta_xi_tx = np.zeros(3) if delta_xi_tx is None else np.asarray(delta_xi_tx, dtype=float)
        delta_xi_rx = np.zeros(3) if delta_xi_rx is None else np.asarray(delta_xi_rx, dtype=float)
        powers = np.array([abs(tap.amplitude) ** 2 for tap in taps])
        dopplers = np.array([tap.doppler_hz for tap in taps])
        delays = np.array([tap.delay_s for tap in taps])
        departure = np.array([tap.departure_direction for tap in taps])
        arrival = np.array([tap.arrival_direction for tap in taps])
        kernel = (
            np.exp(-2j * np.pi * dopplers * delta_t)
            * np.exp(2j * np.pi * delays * delta_f)
            * np.exp(1j * self._wavenumber * (departure @ delta_xi_tx))
            * np.exp(1j * self._wavenumber * (arrival @ delta_xi_rx))
        )
        survival = self._survival_mask(
            len(taps), los_tap_count, delta_t, delta_xi_tx, delta_xi_rx, terminal_speed_m_s
        )
        total = powers.sum()
        if total <= 0.0:
            return 0j
        return complex(np.sum(powers * kernel * survival) / total)

    def temporal_acf(
        self,
        impulse_response: ChannelImpulseResponse,
        time_lags_s: np.ndarray,
        terminal_speed_m_s: float = 0.0,
        los_tap_count: int = 0,
    ) -> np.ndarray:
        return np.array(
            [
                self.stf_cf(
                    impulse_response,
                    delta_t=float(lag),
                    terminal_speed_m_s=terminal_speed_m_s,
                    los_tap_count=los_tap_count,
                )
                for lag in time_lags_s
            ]
        )

    def spatial_ccf(
        self,
        impulse_response: ChannelImpulseResponse,
        spacings_m: np.ndarray,
        axis_unit: np.ndarray,
        side: str = "tx",
        los_tap_count: int = 0,
    ) -> np.ndarray:
        axis = np.asarray(axis_unit, dtype=float)
        results = []
        for spacing in spacings_m:
            displacement = float(spacing) * axis
            if side == "tx":
                results.append(self.stf_cf(impulse_response, delta_xi_tx=displacement, los_tap_count=los_tap_count))
            else:
                results.append(self.stf_cf(impulse_response, delta_xi_rx=displacement, los_tap_count=los_tap_count))
        return np.array(results)

    def frequency_correlation(
        self, impulse_response: ChannelImpulseResponse, frequency_lags_hz: np.ndarray
    ) -> np.ndarray:
        return np.array([self.stf_cf(impulse_response, delta_f=float(lag)) for lag in frequency_lags_hz])

    def _survival_mask(
        self,
        tap_count: int,
        los_tap_count: int,
        delta_t: float,
        delta_xi_tx: np.ndarray,
        delta_xi_rx: np.ndarray,
        terminal_speed_m_s: float,
    ) -> np.ndarray:
        mask = np.ones(tap_count)
        if los_tap_count <= 0 or (
            delta_t == 0.0
            and terminal_speed_m_s == 0.0
            and not np.any(delta_xi_tx)
            and not np.any(delta_xi_rx)
        ):
            return mask
        time_term = terminal_speed_m_s * abs(delta_t) / self._evolution.time_correlated_distance_m
        transmit = float(
            np.exp(
                -self._evolution.recombination_rate
                * (time_term + np.linalg.norm(delta_xi_tx) / self._evolution.array_correlated_distance_m)
            )
        )
        receive = float(
            np.exp(
                -self._evolution.recombination_rate
                * (time_term + np.linalg.norm(delta_xi_rx) / self._evolution.array_correlated_distance_m)
            )
        )
        mask[los_tap_count:] = transmit * receive
        return mask


__all__ = ["STFCorrelationModel"]