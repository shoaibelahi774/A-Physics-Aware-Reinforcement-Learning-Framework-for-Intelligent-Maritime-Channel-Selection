from __future__ import annotations
import math
from dataclasses import dataclass
from statistics import NormalDist

# ----- named model constants (no inline magic numbers) -----
_MIN_USABLE_SNR_DB = 2.0             # below this a terrestrial link is out of service
_NLOS_PENALTY_DB = 6.0               # extra loss when the GBSM reports NLoS
_SEA_FADE_DB_PER_M = 1.5             # sea-state fading per metre of significant wave height
_DELAY_TO_LATENCY_MS_PER_NS = 0.02   # excess latency from the GBSM RMS delay spread
from config.paper_parameters import SATELLITE_DOPPLER_REFERENCE_HZ as _DOPPLER_REFERENCE_HZ
_LEO_FALLBACK_ELEVATION_DEG = 45.0   # constellation model when no visible TLE pass
_LEO_FALLBACK_DOPPLER_HZ = 38_000.0
_GEO_ELEVATION_DEG = 30.0
_SATELLITE_PREFERENCE_PENALTY = 0.22  # prefer terrestrial while it is in coverage
_GBSM_CARRIER_HZ = 5.8e9             # carrier at which the GBSM physics is generated
_GBSM_REFERENCE_SNR_DB = 8.0         # nominal GBSM SNR (fading deviations measured from here)
_GBSM_FADE_WEIGHT = 0.3             # weight of the GBSM SNR realisation on the link SNR
_BAND_PL_WEIGHT = 0.15             # weight of the per-band GBSM path-loss correction
_K_MARGIN_DB_MAX = 3.0             # max SNR margin from a strongly Rician (high-K) channel
_GBSM_REFERENCE_SE = 6.0           # reference ergodic spectral efficiency (bit/s/Hz)
_GBSM_QUALITY_FLOOR = 0.60         # min multiplicative effect of the GBSM channel quality
# --- physical link-budget constants (M3) ---
_THERMAL_NOISE_DBM_HZ = -173.977   # kT0 at 290 K, in dBm/Hz (thermal noise floor)
_BOLTZMANN_DBW = -228.6            # 10*log10(k), k=1.38e-23 W/K/Hz (satellite C/N form)
_LIGHT_SPEED_M_S = 299_792_458.0
_EARTH_RADIUS_KM = 6371.0
# Radio horizon over a 4/3-Earth-radius (standard atmospheric refraction):
#   d_horizon[km] = 4.12 * (sqrt(h_tx[m]) + sqrt(h_rx[m]))
_RADIO_HORIZON_K = 4.12             # km per sqrt(metre), 4/3 Earth-radius refraction
_LEO_ALTITUDE_KM = 600.0           # nominal LEO shell (slant range when no SGP4 delay)
_GEO_ALTITUDE_KM = 35_786.0        # geostationary altitude
_GOLDEN = 0.6180339887498949
_NORMAL = NormalDist()

STATUS_AVAILABLE = "Available"
STATUS_AVAILABLE_SGP4 = "Avail (SGP4)"
STATUS_AVAILABLE_MODEL = "Avail (model)"
STATUS_OUT_OF_COVERAGE = "Out of Coverage"
STATUS_NOT_VISIBLE = "Satellite Not Visible"
STATUS_UNAVAILABLE = "Unavailable"


@dataclass(frozen=True)
class TechProfile:
    name: str
    max_range_km: float             # terrestrial coverage limit (inf for satellites)
    rf_bandwidth_mhz: float
    carrier_hz: float               # per-technology carrier band (M5)
    base_latency_ms: float
    cost_index: float
    shadow_std_db: float
    d_min_km: float = 0.05
    is_satellite: bool = False
    # --- antenna heights for the radio-horizon reach (M4) ---
    tx_height_m: float = 0.0              # shore/base-station antenna height (m)
    rx_height_m: float = 0.0              # ship antenna height (m)
    # --- physical link-budget parameters (M3) ---
    # Terrestrial budget: SNR = EIRP + G_rx - FSPL(d,f) - N,  N = -174 dBm/Hz + 10log10(B) + NF
    eirp_dbm: float = 0.0                 # effective isotropic radiated power (dBm)
    rx_gain_dbi: float = 0.0              # receiver antenna gain (dBi)
    noise_figure_db: float = 7.0          # receiver noise figure (dB)
    implementation_margin_db: float = 3.0 # cables/impl/misc losses (dB)
    # Satellite budget: C/N = EIRP + G/T - FSPL(slant,f) - 10log10(k) - 10log10(B) - margin
    sat_eirp_dbw: float = 0.0             # satellite EIRP (dBW)
    sat_gt_dbk: float = 0.0               # earth-station figure of merit G/T (dB/K)
    sat_margin_db: float = 0.0            # rain + pointing + atmospheric + PFD backoff (dB)
    standard: str = ""                    # governing standard (M3 provenance)


# Standards-referenced maritime technology envelopes with PHYSICAL link budgets.
# Terrestrial values are typical figures from the cited standards (EIRP, Rx gain,
# noise figure); satellite values are typical Ku-band EIRP / earth-station G/T.
PROFILES = {
    "WiFi":              TechProfile("WiFi",           math.inf, 40.0, 2.4e9,   8.0, 0.20, 3.0,
                                     d_min_km=0.03, tx_height_m=20.0, rx_height_m=10.0,
                                     eirp_dbm=30.0, rx_gain_dbi=6.0, noise_figure_db=8.0,
                                     implementation_margin_db=3.0,
                                     standard="IEEE 802.11"),
    "4G Cellular":       TechProfile("4G Cellular",    math.inf, 20.0, 3.5e9,  30.0, 1.00, 4.0,
                                     d_min_km=0.10, tx_height_m=35.0, rx_height_m=12.0,
                                     eirp_dbm=53.0, rx_gain_dbi=2.0, noise_figure_db=7.0,
                                     implementation_margin_db=3.0,
                                     standard="3GPP TR 38.901"),
    "Short-Range Radio": TechProfile("Short-Range Radio", math.inf, 6.0, 0.90e9, 12.0, 0.30, 4.0,
                                     d_min_km=0.50, tx_height_m=80.0, rx_height_m=15.0,
                                     eirp_dbm=43.0, rx_gain_dbi=3.0, noise_figure_db=6.0,
                                     implementation_margin_db=4.0,
                                     standard="ITU-R M.1371"),
    "LEO Satellite":     TechProfile("LEO Satellite",  math.inf, 25.0, 12.0e9, 35.0, 3.00, 3.0,
                                     is_satellite=True,
                                     sat_eirp_dbw=22.0, sat_gt_dbk=12.0, sat_margin_db=3.0,
                                     standard="3GPP TR 38.811 / DVB-S2X"),
    "GEO Satellite":     TechProfile("GEO Satellite",  math.inf, 12.0, 12.0e9, 550.0, 5.00, 2.0,
                                     is_satellite=True,
                                     sat_eirp_dbw=44.0, sat_gt_dbk=18.0, sat_margin_db=4.0,
                                     standard="DVB-S2X"),
}


class CoveragePhysicsModel:
    def __init__(self, config=None) -> None:
        self._config = config
        self._gbsm_carrier_hz = getattr(config, "carrier_frequency_hz", _GBSM_CARRIER_HZ) if config else _GBSM_CARRIER_HZ

    def coverage_range_km(self, name: str) -> float:
        profile = PROFILES.get(name)
        if profile is None or profile.is_satellite:
            return math.inf
        return self._radio_horizon_km(profile)

    @staticmethod
    def _radio_horizon_km(profile: TechProfile) -> float:
        cap = getattr(profile, "max_range_km", math.inf)
        horizon = _RADIO_HORIZON_K * (math.sqrt(profile.tx_height_m) + math.sqrt(profile.rx_height_m))
        return min(horizon, cap)

    def apply(self, channels_by_name: dict, distance_km: float, snapshot) -> None:
        for name, channel in channels_by_name.items():
            profile = PROFILES.get(name)
            if profile is None:
                continue
            metrics = (self._satellite_metrics(profile, distance_km, snapshot)
                       if profile.is_satellite
                       else self._terrestrial_metrics(profile, distance_km, snapshot))
            channel.attach_metrics(metrics)
            # NOTE (Mo4): channel.bandwidth stays the configured value in Mbps.
            # The RF bandwidth in MHz lives in metrics['bandwidth_mhz'] (and the
            # gbsm_physical_bandwidth_mhz alias) so the two units never collide
            # on one field. The status table reads bandwidth_mhz for its column.
            channel.status = metrics["status"]

    # ---------------- terrestrial technologies ----------------
    def _terrestrial_metrics(self, profile: TechProfile, distance_km: float, snapshot) -> dict:
        in_range = distance_km <= self._radio_horizon_km(profile)

        # 1) PHYSICAL link budget (M3): SNR = EIRP + G_rx - FSPL(d,f) - N,
        #    N = -174 dBm/Hz + 10log10(B) + NF. Replaces the earlier
        #    max/span reach interpolation with a real, standards-parameterised
        #    budget; the technology reach now emerges from the budget itself.
        link_snr_db = self._link_budget_snr_db(profile, distance_km)

        # 2) GBSM environmental modulation (consumes the GBSM physics: M4/M8)
        shadow_db = self._shadowing(profile.shadow_std_db, distance_km, profile.name)
        nlos_db = 0.0 if snapshot.los_present else _NLOS_PENALTY_DB
        sea_fade_db = _SEA_FADE_DB_PER_M * snapshot.significant_wave_height_m
        gbsm_fade_db = _GBSM_FADE_WEIGHT * max(-6.0, min(6.0, getattr(snapshot, "snr_db", _GBSM_REFERENCE_SNR_DB) - _GBSM_REFERENCE_SNR_DB))
        k_factor = max(0.0, getattr(snapshot, "k_factor", 0.0))
        k_margin_db = _K_MARGIN_DB_MAX * (k_factor / (k_factor + 1.0))          # Rician LoS margin
        band_pl_db = self._band_path_loss_db(profile, snapshot)                 # per-band (M5/M8)

        snr_db = (link_snr_db + k_margin_db + gbsm_fade_db
                  - shadow_db - nlos_db - sea_fade_db - _BAND_PL_WEIGHT * band_pl_db)

        available = in_range and snr_db > _MIN_USABLE_SNR_DB
        status = STATUS_AVAILABLE if available else STATUS_OUT_OF_COVERAGE

        # 3) capacity driven by the GBSM ergodic capacity (C1) x per-tech Shannon
        gbsm_quality = self._gbsm_quality(snapshot)
        capacity = (self._shannon_capacity(profile.rf_bandwidth_mhz, snr_db) * gbsm_quality
                    if available else 0.0)
        packet_loss = self._packet_loss(snr_db) if available else 1.0
        latency = profile.base_latency_ms + snapshot.rms_delay_spread_s * 1.0e9 * _DELAY_TO_LATENCY_MS_PER_NS

        # 4) stability from LoS probability, Rician K and angular diversity (M8)
        angular_spread = getattr(snapshot, "departure_azimuth_spread_rad", 0.0)
        diversity = min(0.15, 0.15 * angular_spread / (math.pi / 2.0))
        rician = k_factor / (k_factor + 1.0)
        stability = min(1.0, max(0.0, (0.45 + 0.35 * self._snr_quality(snr_db) + 0.10 * rician + diversity)
                                 * (0.6 + 0.4 * snapshot.los_probability)))

        # 5) per-band ship-motion Doppler (M5): scales with the technology's carrier
        band_doppler = snapshot.max_doppler_hz * (profile.carrier_hz / self._gbsm_carrier_hz)
        qos = self._qos(capacity, latency, packet_loss, stability, profile.cost_index, 0.0, is_satellite=False)
        return self._pack(profile, available, status, latency, stability, capacity, snr_db,
                          packet_loss, qos, snapshot, band_doppler, shadow_db, distance_km,
                          los_score=snapshot.los_probability, los_present=snapshot.los_present,
                          band_path_loss_db=band_pl_db, leo_source="terrestrial")

    # ---------------- satellite technologies ----------------
    def _satellite_metrics(self, profile: TechProfile, distance_km: float, snapshot) -> dict:
        is_leo = profile.name.startswith("LEO")
        if is_leo and snapshot.satellite_visible:
            # real SGP4 pass resolved from the TLE (M6: labelled transparently)
            elevation = snapshot.satellite_elevation_rad
            doppler = snapshot.satellite_doppler_hz
            status, source = STATUS_AVAILABLE_SGP4, "sgp4"
        elif is_leo:
            # constellation model between resolved passes (aggregate multi-satellite availability)
            elevation = math.radians(_LEO_FALLBACK_ELEVATION_DEG)
            doppler = _LEO_FALLBACK_DOPPLER_HZ * math.cos(distance_km * 0.01)
            status, source = STATUS_AVAILABLE_MODEL, "constellation_model"
        else:  # GEO geostationary: always visible, negligible Doppler
            elevation, doppler = math.radians(_GEO_ELEVATION_DEG), 0.0
            status, source = STATUS_AVAILABLE, "geostationary"

        doppler_penalty = min(abs(doppler) / _DOPPLER_REFERENCE_HZ, 1.0)
        elevation_quality = max(0.05, math.sin(elevation))
        shadow_db = self._shadowing(profile.shadow_std_db, distance_km, profile.name)
        slant_range_km = self._slant_range_km(snapshot, is_leo, elevation)
        noise_bw_dbhz = 10.0 * math.log10(profile.rf_bandwidth_mhz * 1.0e6)
        cn_db = (profile.sat_eirp_dbw + profile.sat_gt_dbk
                 - self._fspl_db(slant_range_km * 1000.0, profile.carrier_hz)
                 - _BOLTZMANN_DBW - noise_bw_dbhz - profile.sat_margin_db)
        snr_db = cn_db - shadow_db - 6.0 * doppler_penalty
        capacity = self._shannon_capacity(profile.rf_bandwidth_mhz, snr_db)
        packet_loss = self._packet_loss(snr_db)
        latency = profile.base_latency_ms
        stability = min(1.0, max(0.0, (0.4 + 0.6 * elevation_quality) * (1.0 - 0.5 * doppler_penalty)))
        qos = self._qos(capacity, latency, packet_loss, stability, profile.cost_index, doppler_penalty,
                        is_satellite=True)
        metrics = self._pack(profile, True, status, latency, stability, capacity, snr_db,
                             packet_loss, qos, snapshot, doppler, shadow_db, distance_km,
                             los_score=min(1.0, elevation_quality + 0.2), los_present=True,
                             band_path_loss_db=0.0, leo_source=source)
        metrics["doppler_hz"] = doppler
        metrics["doppler_penalty"] = doppler_penalty
        metrics["elevation_rad"] = elevation
        return metrics

    # ---------------- shared helpers ----------------
    def _link_budget_snr_db(self, profile: TechProfile, distance_km: float) -> float:
        distance_m = max(distance_km, profile.d_min_km) * 1000.0
        pl_db = self._two_ray_path_loss_db(distance_m, profile.carrier_hz,
                                           profile.tx_height_m, profile.rx_height_m)
        noise_dbm = self._thermal_noise_dbm(profile.rf_bandwidth_mhz, profile.noise_figure_db)
        return (profile.eirp_dbm + profile.rx_gain_dbi - pl_db
                - noise_dbm - profile.implementation_margin_db)

    @staticmethod
    def _fspl_db(distance_m: float, carrier_hz: float) -> float:
        if distance_m <= 0.0:
            return 0.0
        return 20.0 * math.log10(4.0 * math.pi * distance_m * carrier_hz / _LIGHT_SPEED_M_S)

    def _two_ray_path_loss_db(self, distance_m: float, carrier_hz: float,
                              tx_height_m: float, rx_height_m: float) -> float:
        fspl = self._fspl_db(distance_m, carrier_hz)
        if tx_height_m <= 0.0 or rx_height_m <= 0.0 or distance_m <= 0.0:
            return fspl
        wavelength_m = _LIGHT_SPEED_M_S / carrier_hz
        break_distance_m = 4.0 * tx_height_m * rx_height_m / wavelength_m
        if distance_m <= break_distance_m:
            return fspl
        return fspl + 20.0 * math.log10(distance_m / break_distance_m)

    @staticmethod
    def _thermal_noise_dbm(bandwidth_mhz: float, noise_figure_db: float) -> float:
        return _THERMAL_NOISE_DBM_HZ + 10.0 * math.log10(bandwidth_mhz * 1.0e6) + noise_figure_db

    def _slant_range_km(self, snapshot, is_leo: bool, elevation_rad: float) -> float:
        delay_s = getattr(snapshot, "satellite_delay_s", 0.0) or 0.0
        if delay_s > 0.0:
            return delay_s * _LIGHT_SPEED_M_S / 1000.0
        altitude = _LEO_ALTITUDE_KM if is_leo else _GEO_ALTITUDE_KM
        r_sin = _EARTH_RADIUS_KM * math.sin(elevation_rad)
        return float(-r_sin + math.sqrt(r_sin * r_sin + altitude * altitude + 2.0 * _EARTH_RADIUS_KM * altitude))

    def _band_path_loss_db(self, profile: TechProfile, snapshot) -> float:
        gbsm_pl = getattr(snapshot, "path_loss_db", 0.0)
        band_pl = gbsm_pl + 20.0 * math.log10(profile.carrier_hz / self._gbsm_carrier_hz)
        return float(band_pl - gbsm_pl)   # deviation of this band from the GBSM band

    @staticmethod
    def _gbsm_quality(snapshot) -> float:
        se = getattr(snapshot, "capacity_bit_s_hz", _GBSM_REFERENCE_SE)
        quality = _GBSM_QUALITY_FLOOR + (1.0 - _GBSM_QUALITY_FLOOR) * (se / _GBSM_REFERENCE_SE)
        return float(min(max(quality, _GBSM_QUALITY_FLOOR), 1.0))

    @staticmethod
    def _pack(profile, available, status, latency, stability, capacity, snr_db, packet_loss,
              qos, snapshot, doppler_hz, shadow_db, distance_km, los_score, los_present,
              band_path_loss_db, leo_source) -> dict:
        return {
            "physical_available": available,
            "status": status,
            "latency_ms": max(1.0, latency),
            "cost_index": profile.cost_index,
            "stability_score": stability,
            "capacity_mbps": capacity,
            "capacity_qos": min(capacity / 200.0, 1.0),
            "bandwidth_mhz": profile.rf_bandwidth_mhz,
            "carrier_hz": profile.carrier_hz,
            "snr_db": snr_db,
            "packet_loss": packet_loss,
            "qos_score": qos,
            "los_score": los_score,
            "los_present": los_present,
            "maritime_doppler_hz": doppler_hz,
            "rms_delay_spread_s": snapshot.rms_delay_spread_s,
            "shadowing_db": shadow_db,
            "k_factor": getattr(snapshot, "k_factor", 0.0),
            "angular_spread_rad": getattr(snapshot, "departure_azimuth_spread_rad", 0.0),
            "gbsm_path_loss_db": getattr(snapshot, "path_loss_db", 0.0),
            "band_path_loss_db": band_path_loss_db,
            "gbsm_snr_db": getattr(snapshot, "snr_db", 0.0),
            "gbsm_capacity_bit_s_hz": getattr(snapshot, "capacity_bit_s_hz", 0.0),
            "gbsm_quality_factor": CoveragePhysicsModel._gbsm_quality(snapshot),
            "leo_source": leo_source,
            "standard": profile.standard,
            "significant_wave_height_m": snapshot.significant_wave_height_m,
            "wind_speed_m_s": snapshot.wind_speed_reference_m_s,
            "distance_km": distance_km,
        }

    @staticmethod
    def _shannon_capacity(bandwidth_mhz: float, snr_db: float) -> float:
        snr_linear = 10.0 ** (max(snr_db, -30.0) / 10.0)
        return float(bandwidth_mhz * math.log2(1.0 + snr_linear))

    @staticmethod
    def _snr_quality(snr_db: float) -> float:
        return min(1.0, max(0.0, (snr_db + 5.0) / 35.0))

    @staticmethod
    def _packet_loss(snr_db: float) -> float:
        return float(1.0 / (1.0 + math.exp((snr_db - 8.0) / 2.5)))

    @staticmethod
    def _qos(capacity_mbps, latency_ms, packet_loss, stability, cost_index, doppler_penalty,
             is_satellite=False) -> float:
        cap_term = min(capacity_mbps / 200.0, 1.0)
        lat_term = max(0.0, 1.0 - latency_ms / 600.0)
        loss_term = 1.0 - packet_loss
        cost_term = min(cost_index / 5.0, 1.0)
        qos = (0.42 * cap_term + 0.18 * lat_term + 0.14 * loss_term + 0.14 * stability
               - 0.08 * cost_term - 0.06 * doppler_penalty)
        if is_satellite:
            qos -= _SATELLITE_PREFERENCE_PENALTY
        return float(qos)

    def _shadowing(self, std_db: float, distance_km: float, name: str) -> float:
        seed = (distance_km * _GOLDEN + len(name) * 0.123) % 1.0
        return float(std_db * _NORMAL.inv_cdf(min(max(seed, 1e-4), 1.0 - 1e-4)))


__all__ = ["CoveragePhysicsModel", "PROFILES", "TechProfile",
           "STATUS_AVAILABLE", "STATUS_AVAILABLE_SGP4", "STATUS_AVAILABLE_MODEL",
           "STATUS_OUT_OF_COVERAGE", "STATUS_NOT_VISIBLE", "STATUS_UNAVAILABLE"]