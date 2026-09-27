from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class PhysicalConstants:
    speed_of_light_m_s: float = 299_792_458.0
    gravitational_acceleration_m_s2: float = 9.80665
    earth_radius_m: float = 6_371_000.0
    sea_level_refractivity: float = 1.00035


@dataclass(frozen=True)
class WindParameters:
    reference_height_m: float = 19.5
    measurement_height_m: float = 10.0
    sea_surface_roughness_m: float = 2.0e-4


@dataclass(frozen=True)
class SeaSpectrumParameters:
    pierson_moskowitz_alpha: float = 8.1e-3
    pierson_moskowitz_beta: float = 0.74
    directional_isotropic_term: float = 1.0
    directional_second_harmonic_base: float = 0.5
    directional_second_harmonic_gain: float = 0.82
    directional_fourth_harmonic_gain: float = 0.32
    frequency_bin_count: int = 24
    direction_bin_count: int = 12
    min_angular_frequency_rad_s: float = 0.2
    max_angular_frequency_rad_s: float = 3.0
    spreading_half_width_rad: float = math.pi / 2.0
    min_wind_speed_m_s: float = 0.1


@dataclass(frozen=True)
class BuildingParameters:
    rayleigh_height_scale_m: float = 8.0
    occupied_area_ratio: float = 0.1
    building_density_per_km2: float = 750.0
    sea_wave_search_step_m: float = 1.0


@dataclass(frozen=True)
class UAVRotationParameters:
    pitch_amplitude_rad: float = math.radians(2.0)
    roll_amplitude_rad: float = math.radians(2.0)
    yaw_amplitude_rad: float = math.radians(1.0)
    pitch_frequency_hz: float = 1.0
    roll_frequency_hz: float = 1.2
    yaw_frequency_hz: float = 0.8
    pitch_phase_rad: float = 0.0
    roll_phase_rad: float = 1.5707963267948966
    yaw_phase_rad: float = 3.141592653589793


@dataclass(frozen=True)
class EvaporationDuctParameters:
    duct_height_m: float = 40.0
    refractivity_gradient_per_m: float = -3.9e-7


@dataclass(frozen=True)
class ClusterEvolutionParameters:
    generation_rate: float = 30.0
    recombination_rate: float = 1.0
    time_correlated_distance_m: float = 120.0
    array_correlated_distance_m: float = 5.0


@dataclass(frozen=True)
class ClusterGeometryParameters:
    initial_cluster_count: int = 20
    rays_per_cluster: int = 10
    intra_cluster_std_m: float = 2.0
    min_cluster_distance_m: float = 50.0
    max_cluster_distance_m: float = 800.0
    angle_pool_size: int = 128
    max_cluster_count: int = 64


@dataclass(frozen=True)
class DelayPowerParameters:
    delay_scalar: float = 2.3
    delay_spread_s: float = 2.0e-7
    per_cluster_shadowing_std_db: float = 3.0
    virtual_link_delay_mean_s: float = 0.0


@dataclass(frozen=True)
class PolarizationParameters:
    cross_polarization_ratio_mean_db: float = 8.0
    cross_polarization_ratio_std_db: float = 3.0
    copolar_imbalance: float = 1.0


@dataclass(frozen=True)
class StatisticalAnalysisParameters:
    stationary_correlation_threshold: float = 0.8


PHYSICAL_CONSTANTS = PhysicalConstants()
WIND = WindParameters()
SEA_SPECTRUM = SeaSpectrumParameters()
BUILDINGS = BuildingParameters()
UAV_ROTATION = UAVRotationParameters()
EVAPORATION_DUCT = EvaporationDuctParameters()
CLUSTER_EVOLUTION = ClusterEvolutionParameters()
CLUSTER_GEOMETRY = ClusterGeometryParameters()
DELAY_POWER = DelayPowerParameters()
POLARIZATION = PolarizationParameters()
STATISTICS = StatisticalAnalysisParameters()


# --- Doppler normalisation references (Mi4: single home for what were scattered
# magic numbers). These are two DIFFERENT physical scales and must stay separate:
CHANNEL_DOPPLER_REFERENCE_HZ = 40_000.0      # GBSM small-scale channel Doppler (reward)
SATELLITE_DOPPLER_REFERENCE_HZ = 150_000.0   # LEO satellite Doppler (coverage / penalty)


__all__ = [
    "PhysicalConstants",
    "WindParameters",
    "SeaSpectrumParameters",
    "BuildingParameters",
    "UAVRotationParameters",
    "EvaporationDuctParameters",
    "ClusterEvolutionParameters",
    "ClusterGeometryParameters",
    "DelayPowerParameters",
    "PolarizationParameters",
    "StatisticalAnalysisParameters",
    "PHYSICAL_CONSTANTS",
    "WIND",
    "SEA_SPECTRUM",
    "BUILDINGS",
    "UAV_ROTATION",
    "EVAPORATION_DUCT",
    "CLUSTER_EVOLUTION",
    "CLUSTER_GEOMETRY",
    "DELAY_POWER",
    "POLARIZATION",
    "STATISTICS",
    "CHANNEL_DOPPLER_REFERENCE_HZ",
    "SATELLITE_DOPPLER_REFERENCE_HZ",
]