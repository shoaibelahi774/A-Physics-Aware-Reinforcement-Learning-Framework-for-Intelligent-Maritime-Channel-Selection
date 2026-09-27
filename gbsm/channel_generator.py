from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from config.simulation_config import SimulationConfig, default_config
from gbsm.channel.channel_impulse_response import ChannelImpulseResponse, ChannelImpulseResponseModel
from gbsm.channel.channel_matrix import ChannelMatrixModel
from gbsm.channel.power_ratios import PowerRatioResolver
from gbsm.channel.propagation import PropagationModel
from gbsm.data_ingestion.building_ingestion import load_building_inventory
from gbsm.data_ingestion.data_audit import DataAuditor
from gbsm.data_ingestion.external_data import ExternalDataBundle, build_reference_frame
from gbsm.data_ingestion.sea_state_loader import load_sea_state
from gbsm.data_ingestion.trajectory_ingestion import AISTrajectoryLoader
from gbsm.environment.sea_spectrum import SeaSpectrumModel
from gbsm.environment.sea_surface import SeaSurface
from gbsm.geometry.clusters import ClusterField
from gbsm.geometry.reflection_geometry import ReflectionModel
from gbsm.geometry.scenarios import ResolvedScenario, ScenarioBuilder, SCENARIOS, TerminalKind
from gbsm.statistics.angular_psd import AngularDomain, AngularPSDModel
from gbsm.statistics.capacity import CapacityModel
from gbsm.statistics.delay_spread import DelaySpreadModel
from gbsm.statistics.stationary_interval import StationaryIntervalModel
from gbsm.los_probability import LoSProbabilityModel
from gbsm import qmc


_GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class ChannelSnapshot:
    index: int
    timestamp: datetime
    scenario: str
    significant_wave_height_m: float
    wind_speed_reference_m_s: float
    los_probability: float
    los_present: bool
    capacity_bit_s_hz: float
    rms_delay_spread_s: float
    rms_doppler_hz: float
    max_doppler_hz: float
    departure_azimuth_spread_rad: float
    satellite_visible: bool
    satellite_name: str | None
    satellite_doppler_hz: float
    satellite_delay_s: float
    satellite_elevation_rad: float
    k_factor: float
    link_distance_m: float
    path_loss_db: float
    snr_db: float


@dataclass(frozen=True)
class FastTimeSeries:
    """Fast-axis (ms-scale) channel evolution within a single large-scale
    (hourly) snapshot. This is where GBSM non-stationarity is physically
    demonstrable: the terminal moves << the cluster correlation distance
    between samples, so clusters persist and the narrowband gain decorrelates
    through Doppler phase evolution and cluster drift, not full turnover.
    """

    scenario: str
    record_index: int
    timestamp: datetime
    significant_wave_height_m: float
    sample_interval_s: float
    times_s: np.ndarray
    narrowband_gain: np.ndarray          # complex h(t) series
    acf_lags_s: np.ndarray               # lag axis for the temporal ACF
    temporal_acf: np.ndarray             # |normalised complex ACF| vs lag
    stationary_interval_s: float         # PDP-correlation stationarity interval
    coherence_time_s: float              # first lag where |ACF| < 0.5
    mean_cluster_survival: float         # fraction of clusters persisting end/start


def _complex_temporal_acf(series: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Normalised channel temporal correlation |R(tau)| with
    R(tau) = mean_t h*(t) h(t+tau), R(0)=mean|h|^2. This is the standard GBSM
    correlation function (no mean subtraction), so the coherence time it yields
    is window-robust and reflects the true Doppler-driven decorrelation rather
    than a finite-window covariance estimate.
    """
    values = np.asarray(series, dtype=complex)
    n = values.size
    r0 = float(np.mean(np.abs(values) ** 2))
    if r0 <= 0.0:
        return np.array([1.0])
    lags = n - 1 if max_lag is None else min(max_lag, n - 1)
    acf = np.array(
        [float(np.abs(np.mean(np.conj(values[: n - k]) * values[k:]))) / r0 for k in range(lags + 1)]
    )
    return acf


class ChannelGenerator:
    def __init__(self, config: SimulationConfig | None = None, operating_snr_db: float = 20.0) -> None:
        self._config = config or default_config()
        self._operating_snr = 10.0 ** (operating_snr_db / 10.0)
        self._reference_link_m = 1000.0
        self._max_terrestrial_link_m = 80_000.0
        self._spectrum = SeaSpectrumModel(self._config)
        self._propagation = PropagationModel(self._config)
        self._power = PowerRatioResolver(self._config)
        self._impulse_response = ChannelImpulseResponseModel(self._config)
        self._matrix = ChannelMatrixModel(self._config)
        self._delay_spread = DelaySpreadModel(self._config)
        self._capacity = CapacityModel(self._config)
        self._angular = AngularPSDModel(self._config)
        self._reflection = ReflectionModel(self._config)
        self._stationary = StationaryIntervalModel(self._config)

    def build_bundle(self, write_audit: bool = True) -> ExternalDataBundle:
        frame = build_reference_frame(self._config)
        sea_state = load_sea_state(self._config)
        trajectories = AISTrajectoryLoader(frame, self._config).load()
        provenance = {"wave": "CMEMS", "wind": "ERA5", "ais": "DMA"}
        building_inventory = None
        try:
            building_inventory = load_building_inventory(frame, self._config)
            provenance["buildings"] = "OpenStreetMap"
        except FileNotFoundError:
            pass
        satellite_ephemeris = self._load_satellites(frame, provenance)
        bundle = ExternalDataBundle(
            sea_state=sea_state,
            reference_frame=frame,
            trajectories=trajectories,
            building_inventory=building_inventory,
            satellite_ephemeris=satellite_ephemeris,
            provenance=provenance,
        )
        if write_audit:
            DataAuditor(self._config).run(bundle)
        return bundle

    def generate(
        self,
        scenario_name: str,
        bundle: ExternalDataBundle | None = None,
        max_snapshots: int | None = None,
        distances_m: list[float] | None = None,
    ) -> list[ChannelSnapshot]:
        bundle = bundle or self.build_bundle()
        scenario = ScenarioBuilder(self._config).resolve(SCENARIOS[scenario_name], bundle)
        los_model = LoSProbabilityModel(
            self._config,
            building_parameters=bundle.building_inventory.to_building_parameters()
            if bundle.building_inventory is not None
            else None,
        )
        dynamics = self._satellite_dynamics(bundle)
        context = _ScenarioContext(scenario, bundle, los_model, dynamics, self._config)
        records = list(bundle.sea_state)
        if max_snapshots is not None:
            records = records[:max_snapshots]
        # The optional ship-to-shore distance schedule couples the GBSM link
        # geometry to the vessel's voyage so delay/path-loss/LoS/capacity track
        # the real distance (clamped to a realistic maritime terrestrial reach).
        snapshots = []
        for position, record in enumerate(records):
            distance_m = distances_m[position] if distances_m is not None and position < len(distances_m) else None
            snapshots.append(self._generate_snapshot(record, context, distance_m))
        return snapshots

    def generate_fast_series(
        self,
        scenario_name: str,
        bundle: ExternalDataBundle | None = None,
        record_index: int = 0,
        sample_count: int | None = None,
    ) -> FastTimeSeries:
        """Evolve the channel on the FAST (ms) axis inside one large-scale
        snapshot, using config.time_grid.fast_sample_{count,interval_s}. The
        clusters evolve continuously at the fast dt (near-unity survival), and
        the wave-driven antenna motion + per-ray Doppler make the narrowband
        gain decorrelate over milliseconds. This demonstrates the GBSM's
        non-stationarity at the scale where it is physically meaningful.
        """
        bundle = bundle or self.build_bundle()
        scenario = ScenarioBuilder(self._config).resolve(SCENARIOS[scenario_name], bundle)
        los_model = LoSProbabilityModel(
            self._config,
            building_parameters=bundle.building_inventory.to_building_parameters()
            if bundle.building_inventory is not None
            else None,
        )
        dynamics = self._satellite_dynamics(bundle)
        context = _ScenarioContext(scenario, bundle, los_model, dynamics, self._config)

        records = list(bundle.sea_state)
        record = records[min(max(record_index, 0), len(records) - 1)]
        dt = float(self._config.time_grid.fast_sample_interval_s)
        n = int(sample_count or self._config.time_grid.fast_sample_count)

        surface = SeaSurface(self._spectrum.from_sea_state(record))
        transmit = scenario.transmitter
        receive = scenario.receiver
        moment = record.timestamp
        kin_tx = transmit.kinematic_at(moment)
        kin_rx = receive.kinematic_at(moment)
        tx_velocity = kin_tx.velocity_enu_m_s
        rx_velocity = kin_rx.velocity_enu_m_s
        speed = float(np.linalg.norm(tx_velocity))
        tx_array = transmit.build_array(surface, scenario.tx_offsets)
        rx_array = receive.build_array(surface, scenario.rx_offsets)
        weights = self._power.resolve(
            scenario.definition.power_ratios,
            scenario.definition.nlos1_kind,
            scenario.definition.nlos2_kind,
            transmit.height_m,
            receive.height_m,
        )
        provider = lambda position: surface.orbital_velocity(position[0], position[1], 0.0)

        initial_ids = {c.identifier for c in context.nlos1.clusters} | {c.identifier for c in context.nlos2.clusters}
        times = np.empty(n, dtype=float)
        gains = np.empty(n, dtype=complex)
        impulse_responses: list[ChannelImpulseResponse] = []
        for k in range(n):
            t = k * dt
            tx_state = tx_array.state_at(kin_tx, t)
            rx_state = rx_array.state_at(kin_rx, t)
            tx_origin = np.asarray(tx_state.origin_enu_m, dtype=float) + tx_velocity * t
            rx_origin = np.asarray(rx_state.origin_enu_m, dtype=float) + rx_velocity * t
            # Continuous cluster evolution at the FAST dt (this is the C2 point).
            context.nlos1.evolve(tx_origin, speed, 0.0, dt, provider)
            context.nlos2.evolve(tx_origin, speed, 0.0, dt)
            reflection = self._reflection.compute(tx_origin, rx_origin)
            propagation = self._propagation.compute(
                tx_origin, tx_velocity, rx_origin, rx_velocity,
                context.nlos1.clusters, context.nlos2.clusters, reflection,
            )
            # LoS state is held fixed within a stationarity interval.
            impulse_response = self._impulse_response.assemble(
                propagation, weights, tx_state.rotation, rx_state.rotation, True
            )
            times[k] = t
            gains[k] = impulse_response.narrowband_gain(t)
            impulse_responses.append(impulse_response)

        final_ids = {c.identifier for c in context.nlos1.clusters} | {c.identifier for c in context.nlos2.clusters}
        survival = len(initial_ids & final_ids) / len(initial_ids) if initial_ids else 0.0

        # Estimate the ACF only over the reliable lag region (<= n/2); at larger
        # lags too few sample pairs remain and the estimate becomes noisy (>1).
        acf = _complex_temporal_acf(gains, max_lag=max(1, n // 2))
        acf_lags = np.arange(acf.size) * dt
        threshold = 0.5
        below = np.where(acf < threshold)[0]
        # If R never crosses 0.5 in the window, the coherence time is >= the
        # reliable span; report that span (a lower bound) rather than a fake value.
        coherence_time = float(below[0] * dt) if below.size else float(acf_lags[-1])

        max_delay = max((float(ir.delays_s.max()) if ir.taps else 0.0) for ir in impulse_responses)
        delay_grid = np.linspace(0.0, max(max_delay, 1.0e-7), 256)
        stationary_interval = self._stationary.from_impulse_responses(impulse_responses, delay_grid, dt)

        return FastTimeSeries(
            scenario=scenario_name,
            record_index=record_index,
            timestamp=moment,
            significant_wave_height_m=record.significant_wave_height_m,
            sample_interval_s=dt,
            times_s=times,
            narrowband_gain=gains,
            acf_lags_s=acf_lags,
            temporal_acf=acf,
            stationary_interval_s=float(stationary_interval),
            coherence_time_s=coherence_time,
            mean_cluster_survival=float(survival),
        )

    def _generate_snapshot(self, record, context: "_ScenarioContext", distance_m: float | None = None) -> ChannelSnapshot:
        surface = SeaSurface(self._spectrum.from_sea_state(record))
        wind_reference = self._spectrum.wind_to_reference_height(record.wind_speed_10m_m_s)
        moment = record.timestamp
        transmit = context.scenario.transmitter
        receive = context.scenario.receiver
        transmit_state = transmit.build_array(surface, context.scenario.tx_offsets).state_at(transmit.kinematic_at(moment), 0.0)
        receive_state = receive.build_array(surface, context.scenario.rx_offsets).state_at(receive.kinematic_at(moment), 0.0)
        tx_velocity = transmit.kinematic_at(moment).velocity_enu_m_s
        rx_velocity = receive.kinematic_at(moment).velocity_enu_m_s
        speed = float(np.linalg.norm(tx_velocity))

        tx_origin = np.asarray(transmit_state.origin_enu_m, dtype=float)
        rx_origin = np.asarray(receive_state.origin_enu_m, dtype=float)
        if distance_m is not None:
            # Place the shore/receiver terminal at the requested ship-to-shore
            # distance along the original line-of-sight azimuth (clamped to a
            # realistic maritime terrestrial reach), keeping its height.
            clamped = float(min(max(distance_m, 1.0), self._max_terrestrial_link_m))
            horizontal = rx_origin[:2] - tx_origin[:2]
            norm = float(np.linalg.norm(horizontal))
            unit = horizontal / norm if norm > 1e-6 else np.array([1.0, 0.0])
            rx_origin = np.array([tx_origin[0] + unit[0] * clamped,
                                  tx_origin[1] + unit[1] * clamped,
                                  rx_origin[2]])

        context.evolve_clusters(tx_origin, speed, surface)
        reflection = self._reflection.compute(tx_origin, rx_origin)
        snapshot_propagation = self._propagation.compute(
            tx_origin, tx_velocity, rx_origin, rx_velocity,
            context.nlos1.clusters, context.nlos2.clusters, reflection,
        )
        los_probability = context.los_probability_at(tx_origin, rx_origin, wind_reference)
        los_present = los_probability > self._los_realization(record.index)
        weights = self._power.resolve(
            context.scenario.definition.power_ratios,
            context.scenario.definition.nlos1_kind,
            context.scenario.definition.nlos2_kind,
            transmit.height_m,
            receive.height_m,
        )
        impulse_response = self._impulse_response.assemble(
            snapshot_propagation, weights, transmit_state.rotation, receive_state.rotation, los_present
        )
        matrix = self._matrix.build(impulse_response, context.scenario.tx_offsets, context.scenario.rx_offsets)
        rms_doppler, max_doppler = self._doppler_statistics(impulse_response)
        satellite = context.satellite_link(moment)
        link_distance_m = float(np.linalg.norm(rx_origin - tx_origin))
        path_loss_db = self._free_space_path_loss_db(link_distance_m)

        # Distance-aware effective SNR: the operating SNR is defined at a 1 km
        # reference link; extra propagation loss reduces it, so the ergodic
        # capacity now falls with distance (physically consistent with M1).
        reference_loss = self._free_space_path_loss_db(self._reference_link_m)
        effective_snr = self._operating_snr * 10.0 ** (-(path_loss_db - reference_loss) / 10.0)
        effective_snr = max(effective_snr, 1.0e-4)
        mean_channel_gain = float(np.mean(np.abs(matrix) ** 2)) if matrix.size else 0.0
        snr_db = 10.0 * math.log10(max(effective_snr * mean_channel_gain, 1e-12))
        capacity_bit_s_hz = self._capacity.instantaneous_capacity(matrix, effective_snr)
        return ChannelSnapshot(
            index=record.index,
            timestamp=moment,
            scenario=context.scenario.definition.name,
            significant_wave_height_m=record.significant_wave_height_m,
            wind_speed_reference_m_s=wind_reference,
            los_probability=los_probability,
            los_present=los_present,
            capacity_bit_s_hz=capacity_bit_s_hz,
            rms_delay_spread_s=self._delay_spread.rms_delay_spread(impulse_response),
            rms_doppler_hz=rms_doppler,
            max_doppler_hz=max_doppler,
            departure_azimuth_spread_rad=self._angular.angular_spread(impulse_response, AngularDomain.AZIMUTH_DEPARTURE),
            satellite_visible=satellite is not None,
            satellite_name=satellite.name if satellite else None,
            satellite_doppler_hz=satellite.doppler_hz if satellite else 0.0,
            satellite_delay_s=satellite.propagation_delay_s if satellite else 0.0,
            satellite_elevation_rad=satellite.elevation_rad if satellite else 0.0,
            k_factor=float(weights.k_factor),
            link_distance_m=link_distance_m,
            path_loss_db=path_loss_db,
            snr_db=snr_db,
        )

    def _free_space_path_loss_db(self, distance_m: float) -> float:
        if distance_m <= 0.0:
            return 0.0
        carrier = self._config.carrier_frequency_hz
        speed_of_light = 299_792_458.0
        return 20.0 * math.log10(4.0 * math.pi * distance_m * carrier / speed_of_light)

    def _doppler_statistics(self, impulse_response: ChannelImpulseResponse) -> tuple[float, float]:
        taps = impulse_response.taps
        if not taps:
            return 0.0, 0.0
        powers = np.array([abs(tap.amplitude) ** 2 for tap in taps])
        dopplers = np.array([tap.doppler_hz for tap in taps])
        total = powers.sum()
        rms = math.sqrt(float(np.sum(powers * dopplers**2) / total)) if total > 0.0 else 0.0
        return rms, float(np.max(np.abs(dopplers)))

    @staticmethod
    def _los_realization(index: int) -> float:
        return qmc.rotate(((index + 0.5) * _GOLDEN_RATIO) % 1.0)

    def _load_satellites(self, frame, provenance):
        try:
            from gbsm.data_ingestion.tle_ephemeris import load_satellite_ephemeris

            ephemeris = load_satellite_ephemeris(frame, self._config)
            provenance["satellite"] = "Space-Track"
            return ephemeris
        except (FileNotFoundError, ImportError):
            return None

    def _satellite_dynamics(self, bundle: ExternalDataBundle):
        if bundle.satellite_ephemeris is None:
            return None
        from gbsm.geometry.satellite_dynamics import SatelliteDynamics

        return SatelliteDynamics(bundle.satellite_ephemeris, self._config, min_elevation_deg=5.0)


class _ScenarioContext:
    def __init__(self, scenario: ResolvedScenario, bundle, los_model, dynamics, config) -> None:
        self.scenario = scenario
        self._bundle = bundle
        self._los_model = los_model
        self._dynamics = dynamics
        self._config = config
        definition = scenario.definition
        on_land = scenario.transmitter.kind is TerminalKind.LAND or scenario.receiver.kind is TerminalKind.LAND
        los_azimuth = self._los_azimuth(scenario)
        self.nlos1 = ClusterField(definition.nlos1_kind, scenario.transmitter.height_m, los_azimuth, on_land, config)
        self.nlos2 = ClusterField(definition.nlos2_kind, scenario.transmitter.height_m, los_azimuth, on_land, config)
        self.nlos1.initialize(scenario.transmitter.base_position_enu_m)
        self.nlos2.initialize(scenario.transmitter.base_position_enu_m)
        # Cluster birth-death must evolve with the SAME time step the snapshots
        # advance by, otherwise the non-stationary decay lambda*(v*dt/D_t + ...)
        # is applied over the wrong interval. Read the real cadence from config
        # (config.time_grid.snapshot_interval_s) rather than assuming 3 h.
        self._snapshot_interval = float(self._config.time_grid.snapshot_interval_s)

    def evolve_clusters(self, tx_origin, speed, surface: SeaSurface) -> None:
        provider = lambda position: surface.orbital_velocity(position[0], position[1], 0.0)
        self.nlos1.evolve(tx_origin, speed, 0.0, self._snapshot_interval, provider)
        self.nlos2.evolve(tx_origin, speed, 0.0, self._snapshot_interval)

    def los_probability_at(self, tx_origin, rx_origin, wind_reference) -> float:
        tx_position = np.array([tx_origin[0], tx_origin[1], self.scenario.transmitter.height_m])
        rx_position = np.array([rx_origin[0], rx_origin[1], self.scenario.receiver.height_m])
        coastline = self.scenario.definition.coastline_distance_m if self.scenario.receiver.kind is TerminalKind.LAND else 0.0
        return self._los_model.los_probability(tx_position, rx_position, coastline, wind_reference)

    def satellite_link(self, moment):
        return self._dynamics.best_link(moment) if self._dynamics is not None else None

    @staticmethod
    def _los_azimuth(scenario: ResolvedScenario) -> float:
        separation = scenario.receiver.base_position_enu_m - scenario.transmitter.base_position_enu_m
        return math.atan2(separation[1], separation[0])


__all__ = ["ChannelSnapshot", "FastTimeSeries", "ChannelGenerator"]