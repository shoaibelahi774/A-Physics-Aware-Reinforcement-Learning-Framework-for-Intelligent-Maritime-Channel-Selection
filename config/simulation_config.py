from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
from config.paper_parameters import PHYSICAL_CONSTANTS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class ArrayConfig:
    tx_element_count: int = 8
    rx_element_count: int = 8
    element_spacing_wavelengths: float = 0.5
    tx_azimuth_rad: float = 5.0 * 3.141592653589793 / 4.0
    tx_elevation_rad: float = 3.141592653589793 / 3.0
    rx_azimuth_rad: float = 3.141592653589793 / 4.0
    rx_elevation_rad: float = 3.141592653589793 / 3.0


@dataclass(frozen=True)
class TimeGridConfig:
    # The Baltic wave product (cmems_mod_bal_wav_anfc_PT1H-i) is HOURLY. The AIS
    # 56-day window yields 56 * 24 = 1344 records at a 1-hour cadence.
    snapshot_count: int = 504
    snapshot_interval_s: float = 1.0 * 3600.0
    fast_sample_count: int = 512
    fast_sample_interval_s: float = 1.0e-3


@dataclass(frozen=True)
class LinkBudgetConfig:
    snr_min_db: float = -30.0
    snr_max_db: float = 30.0
    snr_point_count: int = 61
    bandwidth_hz: float = 20.0e6
    noise_figure_db: float = 7.0


@dataclass(frozen=True)
class DatasetPaths:
    """Aarhus / Kattegat, Denmark — all sources co-located with the AIS vessels.

    wave      : CMEMS BALTICSEA_ANALYSISFORECAST_WAV_003_010 (hourly)
    wind      : ERA5 reanalysis, 10 m u/v components (3-hourly)
    ais       : Danish Maritime Authority (unchanged)
    buildings : OpenStreetMap, Aarhus coast (41,385 footprints)
    tle       : Space-Track ORBCOMM (unchanged)
    """
    wave_file: str = "Denmark_Wave_56_Days.nc"
    wind_march_file: str = "Denmark_Wind_March_2026.nc"
    wind_april_file: str = "Denmark_Wind_April_2026.nc"
    ais_file: str = "ais/ships_multiday.csv"
    building_file: str = "buildings/aarhus_coast_buildings.geojson"
    tle_file: str = "orbcomm_constellation_2026-03.3le.txt"

    def resolve(self, name: str) -> Path:
        return DATA_DIR / getattr(self, name)


@dataclass(frozen=True)
class SimulationConfig:
    carrier_frequency_hz: float = 5.8e9
    default_wind_speed_m_s: float = 5.0
    # Aarhus, Denmark (Kattegat) — matches the AIS vessel centroid.
    reference_latitude_deg: float = 56.15
    reference_longitude_deg: float = 10.21
    random_seed: int = 20260306
    array: ArrayConfig = field(default_factory=ArrayConfig)
    time_grid: TimeGridConfig = field(default_factory=TimeGridConfig)
    link_budget: LinkBudgetConfig = field(default_factory=LinkBudgetConfig)
    datasets: DatasetPaths = field(default_factory=DatasetPaths)

    @property
    def wavelength_m(self) -> float:
        return PHYSICAL_CONSTANTS.speed_of_light_m_s / self.carrier_frequency_hz

    @property
    def element_spacing_m(self) -> float:
        return self.array.element_spacing_wavelengths * self.wavelength_m

    def with_overrides(self, **overrides) -> "SimulationConfig":
        return replace(self, **overrides)


def default_config() -> SimulationConfig:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return SimulationConfig()


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
    "ArrayConfig",
    "TimeGridConfig",
    "LinkBudgetConfig",
    "DatasetPaths",
    "SimulationConfig",
    "default_config",
]