from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.simulation_config import SimulationConfig, default_config
from gbsm.data_ingestion.external_data import (
    ENUReferenceFrame,
    TelemetryObservation,
    build_reference_frame,
)


_TIMESTAMP_CANDIDATES = ("timestamp", "time", "datetime", "date_time", "valid_time")
_SOURCE_CANDIDATES = ("source", "sensor", "node", "station", "terminal")
_LATITUDE_CANDIDATES = ("latitude", "lat")
_LONGITUDE_CANDIDATES = ("longitude", "lon", "lng")
_TIMESTAMP_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d",
)


class TelemetryLoader:
    def __init__(
        self,
        reference_frame: ENUReferenceFrame | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        self._config = config or default_config()
        self._frame = reference_frame or build_reference_frame(self._config)

    def load(self, path: str | Path | None = None) -> tuple[TelemetryObservation, ...]:
        resolved = self._resolve_path(path)
        if resolved is None or not resolved.exists():
            return ()
        frame = pd.read_csv(resolved)
        frame.columns = [column.strip() for column in frame.columns]
        lowered = {column.lower(): column for column in frame.columns}
        time_column = self._match(lowered, _TIMESTAMP_CANDIDATES)
        if time_column is None:
            raise ValueError("Telemetry file has no recognizable timestamp column.")
        source_column = self._match(lowered, _SOURCE_CANDIDATES)
        latitude_column = self._match(lowered, _LATITUDE_CANDIDATES)
        longitude_column = self._match(lowered, _LONGITUDE_CANDIDATES)
        timestamps = self._parse_timestamps(frame[time_column])
        reserved = {time_column, source_column, latitude_column, longitude_column}
        numeric_columns = [
            column
            for column in frame.columns
            if column not in reserved and self._is_numeric(frame[column])
        ]
        observations: list[TelemetryObservation] = []
        for position, row in enumerate(frame.itertuples(index=False)):
            record = dict(zip(frame.columns, row))
            moment = timestamps[position]
            if moment is None:
                continue
            fields = self._numeric_fields(record, numeric_columns)
            self._attach_position(fields, record, latitude_column, longitude_column)
            source = str(record[source_column]) if source_column else "telemetry"
            observations.append(
                TelemetryObservation(timestamp=moment, source=source, fields=fields)
            )
        return tuple(observations)

    def _resolve_path(self, explicit: str | Path | None) -> Path | None:
        if explicit is not None:
            return Path(explicit)
        configured = getattr(self._config.datasets, "telemetry_file", None)
        if configured is None:
            return None
        return self._config.datasets.resolve("telemetry_file")

    def _attach_position(self, fields, record, latitude_column, longitude_column) -> None:
        if not latitude_column or not longitude_column:
            return
        latitude = self._to_float(record.get(latitude_column))
        longitude = self._to_float(record.get(longitude_column))
        if latitude is None or longitude is None:
            return
        east, north, up = self._frame.geodetic_to_enu(latitude, longitude, 0.0)
        fields["position_east_m"] = float(east)
        fields["position_north_m"] = float(north)
        fields["position_up_m"] = float(up)

    @staticmethod
    def _numeric_fields(record, numeric_columns) -> dict[str, float]:
        fields: dict[str, float] = {}
        for column in numeric_columns:
            value = TelemetryLoader._to_float(record.get(column))
            if value is not None:
                fields[column] = value
        return fields

    @staticmethod
    def _parse_timestamps(series: pd.Series) -> list[datetime | None]:
        parsed = None
        for fmt in _TIMESTAMP_FORMATS:
            candidate = pd.to_datetime(series, format=fmt, errors="coerce")
            if candidate.notna().mean() >= 0.5:
                parsed = candidate
                break
        if parsed is None:
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
        result: list[datetime | None] = []
        for value in parsed:
            if pd.isna(value):
                result.append(None)
            else:
                seconds = np.datetime64(value, "s").astype("int64")
                result.append(datetime.fromtimestamp(int(seconds), tz=timezone.utc))
        return result

    @staticmethod
    def _match(lowered, candidates) -> str | None:
        for candidate in candidates:
            if candidate in lowered:
                return lowered[candidate]
        return None

    @staticmethod
    def _is_numeric(series: pd.Series) -> bool:
        return pd.to_numeric(series, errors="coerce").notna().any()

    @staticmethod
    def _to_float(value) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(result) else result


def load_telemetry(
    reference_frame: ENUReferenceFrame | None = None,
    config: SimulationConfig | None = None,
    path: str | Path | None = None,
) -> tuple[TelemetryObservation, ...]:
    return TelemetryLoader(reference_frame, config).load(path)


__all__ = ["TelemetryLoader", "load_telemetry"]