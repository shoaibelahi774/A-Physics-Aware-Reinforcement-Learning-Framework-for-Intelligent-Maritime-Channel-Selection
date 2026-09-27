from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import math

import numpy as np
import xarray as xr

from config.simulation_config import SimulationConfig, default_config


_WAVE_HEIGHT = "VHM0"
_WAVE_PEAK_PERIOD = "VTPK"
_WAVE_MEAN_PERIOD = "VTM02"
_WAVE_MEAN_DIRECTION = "VMDR"
_WAVE_PEAK_DIRECTION = "VPED"
_STOKES_EAST = "VSDX"
_STOKES_NORTH = "VSDY"
_WIND_EAST = "u10"
_WIND_NORTH = "v10"


@dataclass(frozen=True)
class SeaStateRecord:
    index: int
    timestamp: datetime
    significant_wave_height_m: float
    peak_wave_period_s: float
    mean_wave_period_s: float
    mean_wave_from_direction_deg: float
    peak_wave_from_direction_deg: float
    stokes_drift_east_m_s: float
    stokes_drift_north_m_s: float
    wind_speed_10m_m_s: float
    wind_direction_rad: float
    wind_east_10m_m_s: float
    wind_north_10m_m_s: float


class SeaStateSeries(Sequence[SeaStateRecord]):
    def __init__(self, records: Sequence[SeaStateRecord]) -> None:
        if not records:
            raise ValueError("Sea-state series is empty; no records were loaded.")
        self._records = tuple(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index):
        return self._records[index]

    @property
    def records(self) -> tuple[SeaStateRecord, ...]:
        return self._records

    @property
    def start_time(self) -> datetime:
        return self._records[0].timestamp

    @property
    def end_time(self) -> datetime:
        return self._records[-1].timestamp

    def nearest(self, moment: datetime) -> SeaStateRecord:
        return min(
            self._records,
            key=lambda record: abs((record.timestamp - moment).total_seconds()),
        )


class SeaStateLoader:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()

    def load(self) -> SeaStateSeries:
        wave = self._open_wave()
        try:
            wind = self._open_wind(wave["time"].values)
            records = self._build_records(wave, wind)
        finally:
            wave.close()
        return SeaStateSeries(records)

    def _open_wave(self) -> xr.Dataset:
        path = self._require(self._config.datasets.resolve("wave_file"))
        dataset = xr.open_dataset(path)
        return dataset.sortby("latitude").sortby("longitude")

    def _open_wind(self, target_time: np.ndarray) -> xr.Dataset:
        march = xr.open_dataset(self._require(self._config.datasets.resolve("wind_march_file")))
        april = xr.open_dataset(self._require(self._config.datasets.resolve("wind_april_file")))
        combined = xr.concat([march, april], dim="valid_time").squeeze(drop=True)
        march.close()
        april.close()
        for scalar in ("number", "expver"):
            if scalar in combined.coords:
                combined = combined.reset_coords(scalar, drop=True)
        combined = combined.rename({"valid_time": "time"})
        combined = combined.sortby("time").sortby("latitude").sortby("longitude")
        return combined.reindex(time=target_time, method="nearest")

    def _build_records(self, wave: xr.Dataset, wind: xr.Dataset) -> list[SeaStateRecord]:
        latitude = self._config.reference_latitude_deg
        longitude = self._config.reference_longitude_deg
        times = wave["time"].values

        height = self._point_series(wave, _WAVE_HEIGHT, latitude, longitude)
        peak_period = self._point_series(wave, _WAVE_PEAK_PERIOD, latitude, longitude)
        mean_period = self._point_series(wave, _WAVE_MEAN_PERIOD, latitude, longitude)
        mean_direction = self._point_series(wave, _WAVE_MEAN_DIRECTION, latitude, longitude)
        peak_direction = self._point_series(wave, _WAVE_PEAK_DIRECTION, latitude, longitude)
        stokes_east = self._point_series(wave, _STOKES_EAST, latitude, longitude)
        stokes_north = self._point_series(wave, _STOKES_NORTH, latitude, longitude)
        wind_east = self._point_series(wind, _WIND_EAST, latitude, longitude)
        wind_north = self._point_series(wind, _WIND_NORTH, latitude, longitude)

        records: list[SeaStateRecord] = []
        for i, moment in enumerate(times):
            east = float(wind_east[i])
            north = float(wind_north[i])
            records.append(
                SeaStateRecord(
                    index=i,
                    timestamp=self._to_datetime(moment),
                    significant_wave_height_m=float(height[i]),
                    peak_wave_period_s=float(peak_period[i]),
                    mean_wave_period_s=float(mean_period[i]),
                    mean_wave_from_direction_deg=float(mean_direction[i]),
                    peak_wave_from_direction_deg=float(peak_direction[i]),
                    stokes_drift_east_m_s=float(stokes_east[i]),
                    stokes_drift_north_m_s=float(stokes_north[i]),
                    wind_speed_10m_m_s=math.hypot(east, north),
                    wind_direction_rad=math.atan2(north, east),
                    wind_east_10m_m_s=east,
                    wind_north_10m_m_s=north,
                )
            )
        return records

    def _point_series(
        self, dataset: xr.Dataset, name: str, latitude: float, longitude: float
    ) -> np.ndarray:
        if name not in dataset.variables:
            return np.zeros(dataset.sizes["time"], dtype=float)
        field = dataset[name]
        sampled = field.interp(latitude=latitude, longitude=longitude).values.astype(float)
        fallback = field.mean(dim=("latitude", "longitude"), skipna=True).values.astype(float)
        return np.where(np.isnan(sampled), fallback, sampled)

    @staticmethod
    def _to_datetime(value: np.datetime64) -> datetime:
        seconds = np.datetime64(value, "s").astype("int64")
        return datetime.fromtimestamp(int(seconds), tz=timezone.utc)

    @staticmethod
    def _require(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Required sea-state dataset not found: {path}")
        return path


def load_sea_state(config: SimulationConfig | None = None) -> SeaStateSeries:
    return SeaStateLoader(config).load()


__all__ = [
    "SeaStateRecord",
    "SeaStateSeries",
    "SeaStateLoader",
    "load_sea_state",
]