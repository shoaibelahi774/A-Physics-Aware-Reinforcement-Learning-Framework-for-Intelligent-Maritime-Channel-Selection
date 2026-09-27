from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from config.paper_parameters import STATISTICS
from config.simulation_config import SimulationConfig, default_config
from gbsm.channel.channel_impulse_response import ChannelImpulseResponse
from gbsm.statistics.delay_spread import DelaySpreadModel


class StationaryIntervalModel:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()
        self._threshold = STATISTICS.stationary_correlation_threshold
        self._delay_spread = DelaySpreadModel(self._config)

    def correlation(self, reference_profile: np.ndarray, other_profile: np.ndarray) -> float:
        cross = float(np.sum(reference_profile * other_profile))
        energy = max(float(np.sum(reference_profile**2)), float(np.sum(other_profile**2)))
        return cross / energy if energy > 0.0 else 0.0

    def stationary_interval(self, profiles: Sequence[np.ndarray], time_step_s: float) -> float:
        if len(profiles) < 2:
            return 0.0
        reference = profiles[0]
        previous_correlation = 1.0
        for index in range(1, len(profiles)):
            current = self.correlation(reference, profiles[index])
            if current <= self._threshold:
                if previous_correlation > self._threshold and previous_correlation != current:
                    fraction = (previous_correlation - self._threshold) / (previous_correlation - current)
                    return (index - 1 + fraction) * time_step_s
                return (index - 1) * time_step_s
            previous_correlation = current
        return (len(profiles) - 1) * time_step_s

    def stationary_distance(self, interval_s: float, terminal_speed_m_s: float) -> float:
        return interval_s * terminal_speed_m_s

    def from_impulse_responses(
        self,
        impulse_responses: Sequence[ChannelImpulseResponse],
        delay_grid_s: np.ndarray,
        time_step_s: float,
        bandwidth_s: float | None = None,
    ) -> float:
        profiles = [
            self._delay_spread.delay_psd(impulse_response, delay_grid_s, bandwidth_s)
            for impulse_response in impulse_responses
        ]
        return self.stationary_interval(profiles, time_step_s)


__all__ = ["StationaryIntervalModel"]