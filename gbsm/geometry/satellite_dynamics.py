from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
import numpy as np
from config.paper_parameters import PHYSICAL_CONSTANTS
from config.simulation_config import SimulationConfig, default_config
from gbsm.data_ingestion.tle_ephemeris import SatelliteEphemeris, SatelliteState


@dataclass(frozen=True)
class SatelliteLinkState:
    timestamp: datetime
    name: str
    slant_range_m: float
    range_rate_m_s: float
    elevation_rad: float
    azimuth_rad: float
    doppler_hz: float
    propagation_delay_s: float
    visible: bool


class SatelliteDynamics:
    def __init__(
        self,
        ephemeris: SatelliteEphemeris,
        config: SimulationConfig | None = None,
        min_elevation_deg: float = 0.0,
        carrier_frequency_hz: float | None = None,
    ) -> None:
        self._ephemeris = ephemeris
        self._config = config or default_config()
        self._carrier = carrier_frequency_hz or self._config.carrier_frequency_hz
        self._light_speed = PHYSICAL_CONSTANTS.speed_of_light_m_s
        self._min_elevation = math.radians(min_elevation_deg)

    def link_states(self, moment: datetime) -> dict[str, SatelliteLinkState]:
        return {name: self._link_state(state) for name, state in self._ephemeris.state_at(moment).items()}

    def visible_links(self, moment: datetime) -> list[SatelliteLinkState]:
        links = [link for link in self.link_states(moment).values() if link.visible]
        return sorted(links, key=lambda link: link.elevation_rad, reverse=True)

    def best_link(self, moment: datetime) -> SatelliteLinkState | None:
        visible = self.visible_links(moment)
        return visible[0] if visible else None

    def doppler_track(self, name: str, times: Iterable[datetime]) -> list[SatelliteLinkState]:
        track = []
        for moment in times:
            states = self._ephemeris.state_at(moment)
            if name in states:
                track.append(self._link_state(states[name]))
        return track

    def _link_state(self, state: SatelliteState) -> SatelliteLinkState:
        position = np.asarray(state.position_enu_m, dtype=float)
        velocity = np.asarray(state.velocity_enu_m_s, dtype=float)
        slant_range = float(np.linalg.norm(position))
        if slant_range <= 0.0:
            return SatelliteLinkState(state.timestamp, state.name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
        range_rate = float(np.dot(position, velocity) / slant_range)
        elevation = math.asin(max(-1.0, min(1.0, position[2] / slant_range)))
        azimuth = math.atan2(position[0], position[1])
        doppler = -(self._carrier / self._light_speed) * range_rate
        delay = slant_range / self._light_speed
        return SatelliteLinkState(
            timestamp=state.timestamp,
            name=state.name,
            slant_range_m=slant_range,
            range_rate_m_s=range_rate,
            elevation_rad=elevation,
            azimuth_rad=azimuth,
            doppler_hz=doppler,
            propagation_delay_s=delay,
            visible=elevation >= self._min_elevation,
        )


__all__ = ["SatelliteLinkState", "SatelliteDynamics"]