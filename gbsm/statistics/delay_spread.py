from __future__ import annotations
import numpy as np
from config.simulation_config import SimulationConfig, default_config
from gbsm.channel.channel_impulse_response import ChannelImpulseResponse


class DelaySpreadModel:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()

    def delay_profile(self, impulse_response: ChannelImpulseResponse) -> tuple[np.ndarray, np.ndarray]:
        return impulse_response.delays_s, impulse_response.powers

    def mean_delay(self, impulse_response: ChannelImpulseResponse) -> float:
        return self.mean_delay_from_profile(impulse_response.delays_s, impulse_response.powers)

    def rms_delay_spread(self, impulse_response: ChannelImpulseResponse) -> float:
        return self.rms_delay_spread_from_profile(impulse_response.delays_s, impulse_response.powers)

    def delay_psd(
        self,
        impulse_response: ChannelImpulseResponse,
        delay_grid_s: np.ndarray,
        bandwidth_s: float | None = None,
    ) -> np.ndarray:
        delays, powers = self.delay_profile(impulse_response)
        grid = np.asarray(delay_grid_s, dtype=float)
        if delays.size == 0:
            return np.zeros_like(grid)
        if bandwidth_s is None:
            bandwidth_s = self._default_bandwidth(grid)
        density = np.zeros_like(grid)
        for delay, power in zip(delays, powers):
            density += power * np.exp(-0.5 * ((grid - delay) / bandwidth_s) ** 2)
        total = density.sum()
        return density / total if total > 0.0 else density

    @staticmethod
    def mean_delay_from_profile(delays_s: np.ndarray, powers: np.ndarray) -> float:
        total = float(powers.sum())
        if total <= 0.0:
            return 0.0
        return float(np.sum(powers * delays_s) / total)

    @staticmethod
    def rms_delay_spread_from_profile(delays_s: np.ndarray, powers: np.ndarray) -> float:
        total = float(powers.sum())
        if total <= 0.0:
            return 0.0
        mean = float(np.sum(powers * delays_s) / total)
        variance = float(np.sum(powers * (delays_s - mean) ** 2) / total)
        return float(np.sqrt(max(variance, 0.0)))

    @staticmethod
    def _default_bandwidth(grid: np.ndarray) -> float:
        if grid.size < 2:
            return 1.0e-8
        return float(abs(grid[1] - grid[0]) * 2.0)


__all__ = ["DelaySpreadModel"]