from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from config.simulation_config import SimulationConfig, default_config
from gbsm.data_ingestion.external_data import (
    ExternalDataBundle,
    KinematicSample,
    Trajectory,
)
from gbsm.environment.antenna_motion import (
    LandArray,
    ShipArray,
    UAVArray,
    UAVWobble,
    element_offsets,
)
from gbsm.environment.sea_surface import SeaSurface


class TerminalKind(Enum):
    SHIP = "ship"
    UAV = "uav"
    LAND = "land"


class ClusterKind(Enum):
    SEA_SURFACE = "sea_surface"
    BUILDINGS = "buildings"
    EVAPORATION_DUCT = "evaporation_duct"


@dataclass(frozen=True)
class PowerRatios:
    k_factor: float
    nlos1: float
    nlos2: float
    reflection: float

    def normalized(self) -> "PowerRatios":
        total = self.nlos1 + self.nlos2 + self.reflection
        if total <= 0.0:
            return PowerRatios(self.k_factor, 1.0, 0.0, 0.0)
        return PowerRatios(
            self.k_factor, self.nlos1 / total, self.nlos2 / total, self.reflection / total
        )


@dataclass(frozen=True)
class TerminalSpec:
    kind: TerminalKind
    height_m: float
    vessel_slot: int | None = None
    ship_length_m: float = 100.0
    ship_width_m: float = 20.0
    speed_m_s: float = 30.0
    heading_rad: float = 0.0


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    transmitter: TerminalSpec
    receiver: TerminalSpec
    nlos1_kind: ClusterKind
    nlos2_kind: ClusterKind
    power_ratios: PowerRatios
    communication_distance_m: float = 1000.0
    coastline_distance_m: float = 200.0
    bearing_rad: float = 0.0


SCENARIOS: dict[str, ScenarioDefinition] = {
    "S2S": ScenarioDefinition(
        name="S2S",
        transmitter=TerminalSpec(TerminalKind.SHIP, height_m=8.0, vessel_slot=0),
        receiver=TerminalSpec(TerminalKind.SHIP, height_m=8.0, vessel_slot=1),
        nlos1_kind=ClusterKind.SEA_SURFACE,
        nlos2_kind=ClusterKind.EVAPORATION_DUCT,
        power_ratios=PowerRatios(5.0, 0.5, 0.3, 0.2),
    ),
    "S2L": ScenarioDefinition(
        name="S2L",
        transmitter=TerminalSpec(TerminalKind.SHIP, height_m=8.0, vessel_slot=0),
        receiver=TerminalSpec(TerminalKind.LAND, height_m=10.0),
        nlos1_kind=ClusterKind.BUILDINGS,
        nlos2_kind=ClusterKind.EVAPORATION_DUCT,
        power_ratios=PowerRatios(10.0, 0.5, 0.2, 0.3),
    ),
    "U2L": ScenarioDefinition(
        name="U2L",
        transmitter=TerminalSpec(TerminalKind.UAV, height_m=35.0, speed_m_s=30.0),
        receiver=TerminalSpec(TerminalKind.LAND, height_m=10.0),
        nlos1_kind=ClusterKind.BUILDINGS,
        nlos2_kind=ClusterKind.EVAPORATION_DUCT,
        power_ratios=PowerRatios(8.0, 0.6, 0.1, 0.3),
    ),
    "U2S": ScenarioDefinition(
        name="U2S",
        transmitter=TerminalSpec(TerminalKind.UAV, height_m=35.0, speed_m_s=30.0),
        receiver=TerminalSpec(TerminalKind.SHIP, height_m=8.0, vessel_slot=0),
        nlos1_kind=ClusterKind.SEA_SURFACE,
        nlos2_kind=ClusterKind.EVAPORATION_DUCT,
        power_ratios=PowerRatios(7.0, 0.5, 0.2, 0.3),
    ),
}


class ResolvedTerminal:
    def __init__(
        self,
        spec: TerminalSpec,
        base_position_enu_m: np.ndarray,
        trajectory: Trajectory | None,
        simulation_start: datetime,
    ) -> None:
        self._spec = spec
        self._base = np.asarray(base_position_enu_m, dtype=float)
        self._trajectory = trajectory
        self._simulation_start = simulation_start
        self._wobble = UAVWobble()

    @property
    def kind(self) -> TerminalKind:
        return self._spec.kind

    @property
    def height_m(self) -> float:
        return self._spec.height_m

    @property
    def base_position_enu_m(self) -> np.ndarray:
        return self._base

    def kinematic_at(self, moment: datetime) -> KinematicSample:
        return KinematicSample(moment, self._base, self._velocity_at(moment))

    def build_array(self, sea_surface: SeaSurface, offsets: np.ndarray):
        if self._spec.kind is TerminalKind.LAND:
            return LandArray(self._base, offsets)
        if self._spec.kind is TerminalKind.SHIP:
            return ShipArray(
                sea_surface, offsets, self._spec.ship_length_m, self._spec.ship_width_m, self._spec.height_m
            )
        return UAVArray(offsets, self._wobble)

    def _velocity_at(self, moment: datetime) -> np.ndarray:
        if self._spec.kind is TerminalKind.LAND:
            return np.zeros(3)
        if self._spec.kind is TerminalKind.UAV or self._trajectory is None:
            return self._spec.speed_m_s * np.array(
                [math.cos(self._spec.heading_rad), math.sin(self._spec.heading_rad), 0.0]
            )
        return self._trajectory.sample_at(self._wrapped_moment(moment)).velocity_enu_m_s

    def _wrapped_moment(self, moment: datetime) -> datetime:
        start = self._trajectory.start_time
        span = (self._trajectory.end_time - start).total_seconds()
        if span <= 0.0:
            return start
        elapsed = (moment - self._simulation_start).total_seconds() % span
        return start + timedelta(seconds=elapsed)


@dataclass(frozen=True)
class ResolvedScenario:
    definition: ScenarioDefinition
    transmitter: ResolvedTerminal
    receiver: ResolvedTerminal
    tx_offsets: np.ndarray
    rx_offsets: np.ndarray


class ScenarioBuilder:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()

    def resolve(self, definition: ScenarioDefinition, bundle: ExternalDataBundle) -> ResolvedScenario:
        vessels = list(bundle.trajectories.values())
        start = bundle.sea_state.start_time
        tx_base, rx_base = self._base_positions(definition)
        transmitter = ResolvedTerminal(
            definition.transmitter, tx_base, self._vessel(definition.transmitter, vessels, tx_base), start
        )
        receiver = ResolvedTerminal(
            definition.receiver, rx_base, self._vessel(definition.receiver, vessels, rx_base), start
        )
        return ResolvedScenario(
            definition=definition,
            transmitter=transmitter,
            receiver=receiver,
            tx_offsets=self._tx_offsets(),
            rx_offsets=self._rx_offsets(),
        )

    def _base_positions(self, definition: ScenarioDefinition) -> tuple[np.ndarray, np.ndarray]:
        distance = definition.communication_distance_m
        bearing = definition.bearing_rad
        tx_height = definition.transmitter.height_m if definition.transmitter.kind is not TerminalKind.SHIP else 0.0
        rx_height = definition.receiver.height_m if definition.receiver.kind is not TerminalKind.SHIP else 0.0
        tx_base = np.array([0.0, 0.0, tx_height])
        rx_base = np.array([distance * math.cos(bearing), distance * math.sin(bearing), rx_height])
        return tx_base, rx_base

    def _vessel(
        self, spec: TerminalSpec, vessels: list[Trajectory], base: np.ndarray
    ) -> Trajectory | None:
        if spec.kind is not TerminalKind.SHIP or spec.vessel_slot is None or not vessels:
            return None
        return self._rebase(vessels[spec.vessel_slot % len(vessels)], base)

    @staticmethod
    def _rebase(trajectory: Trajectory, base: np.ndarray) -> Trajectory:
        offset = base - trajectory.samples[0].position_enu_m
        rebased = [
            KinematicSample(sample.timestamp, sample.position_enu_m + offset, sample.velocity_enu_m_s)
            for sample in trajectory.samples
        ]
        return Trajectory(trajectory.identifier, rebased)

    def _tx_offsets(self) -> np.ndarray:
        array = self._config.array
        return element_offsets(
            array.tx_element_count, self._config.element_spacing_m, array.tx_azimuth_rad, array.tx_elevation_rad
        )

    def _rx_offsets(self) -> np.ndarray:
        array = self._config.array
        return element_offsets(
            array.rx_element_count, self._config.element_spacing_m, array.rx_azimuth_rad, array.rx_elevation_rad
        )


def build_scenario(
    name: str, bundle: ExternalDataBundle, config: SimulationConfig | None = None
) -> ResolvedScenario:
    return ScenarioBuilder(config).resolve(SCENARIOS[name], bundle)


__all__ = [
    "TerminalKind",
    "ClusterKind",
    "PowerRatios",
    "TerminalSpec",
    "ScenarioDefinition",
    "SCENARIOS",
    "ResolvedTerminal",
    "ResolvedScenario",
    "ScenarioBuilder",
    "build_scenario",
]