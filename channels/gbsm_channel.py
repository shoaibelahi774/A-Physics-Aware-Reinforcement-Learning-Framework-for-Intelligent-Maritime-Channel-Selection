from __future__ import annotations
import math
from config.simulation_config import SimulationConfig, default_config
from gbsm.channel_generator import ChannelSnapshot


_DELAY_SPREAD_LATENCY_MS_PER_NS = 0.02
_SATELLITE_PROCESSING_MS = 20.0
from config.paper_parameters import SATELLITE_DOPPLER_REFERENCE_HZ as _DOPPLER_REFERENCE_HZ


class GBSMChannelBridge:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()
        self._system_bandwidth_hz = self._config.link_budget.bandwidth_hz

    def apply_maritime(self, channel, snapshot: ChannelSnapshot) -> None:
        base_latency = getattr(channel, "configured_latency_ms", channel.latency)
        delay_penalty = snapshot.rms_delay_spread_s * 1.0e9 * _DELAY_SPREAD_LATENCY_MS_PER_NS
        outage_penalty = 0.0 if snapshot.los_present else base_latency
        capacity_mbps = snapshot.capacity_bit_s_hz * self._system_bandwidth_hz / 1.0e6

        metrics = {
            "physical_available": True,
            "latency_ms": max(1.0, base_latency + delay_penalty + outage_penalty),
            "cost_index": getattr(channel, "configured_cost_index", channel.cost),
            "stability_score": min(1.0, max(0.0, snapshot.los_probability)),
            # capacity is kept DISTINCT from the channel's nominal bandwidth
            "capacity_mbps": capacity_mbps,
            "capacity_qos": min(1.0, snapshot.capacity_bit_s_hz / 10.0),
            "los_score": snapshot.los_probability,
            "los_present": snapshot.los_present,
            "rms_delay_spread_s": snapshot.rms_delay_spread_s,
            "maritime_doppler_hz": snapshot.max_doppler_hz,
            "rms_doppler_hz": snapshot.rms_doppler_hz,
            "k_factor": snapshot.k_factor,
            "angular_spread_rad": snapshot.departure_azimuth_spread_rad,
            "path_loss_db": snapshot.path_loss_db,
            "snr_db": snapshot.snr_db,
            "link_distance_m": snapshot.link_distance_m,
            "significant_wave_height_m": snapshot.significant_wave_height_m,
            "scenario": snapshot.scenario,
            "model_note": "gbsm_maritime",
        }
        channel.attach_metrics(metrics)

    def apply_satellite(self, channel, snapshot: ChannelSnapshot) -> None:
        if not snapshot.satellite_visible:
            channel.available = False
            channel.metrics = {"physical_available": False, "availability_reason": "no_satellite_in_view"}
            return
        round_trip_ms = 2.0 * snapshot.satellite_delay_s * 1000.0
        elevation_factor = math.sin(max(0.0, snapshot.satellite_elevation_rad))
        doppler_penalty = min(1.0, abs(snapshot.satellite_doppler_hz) / _DOPPLER_REFERENCE_HZ)
        stability = (0.5 + 0.5 * elevation_factor) * (1.0 - 0.5 * doppler_penalty)

        metrics = {
            "physical_available": True,
            "latency_ms": max(1.0, _SATELLITE_PROCESSING_MS + round_trip_ms),
            "cost_index": getattr(channel, "configured_cost_index", channel.cost),
            "stability_score": min(1.0, max(0.0, stability)),
            "doppler_hz": snapshot.satellite_doppler_hz,
            "doppler_penalty": doppler_penalty,
            "snr_db": snapshot.snr_db,
            "los_score": elevation_factor,
            "model_note": "gbsm_leo",
        }
        channel.attach_metrics(metrics)
        if hasattr(channel, "leo_doppler_hz"):
            channel.leo_doppler_hz = snapshot.satellite_doppler_hz
        if hasattr(channel, "leo_doppler_penalty"):
            channel.leo_doppler_penalty = doppler_penalty

    def apply(self, channels_by_name: dict, snapshot: ChannelSnapshot,
              maritime_channel_name: str, satellite_channel_name: str = "LEO Satellite") -> None:
        if maritime_channel_name in channels_by_name:
            self.apply_maritime(channels_by_name[maritime_channel_name], snapshot)
        if satellite_channel_name in channels_by_name:
            self.apply_satellite(channels_by_name[satellite_channel_name], snapshot)


__all__ = ["GBSMChannelBridge"]