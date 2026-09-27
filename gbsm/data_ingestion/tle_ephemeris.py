from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from sgp4.api import Satrec, jday
from config.simulation_config import SimulationConfig, default_config
from gbsm.data_ingestion.external_data import ENUReferenceFrame, build_reference_frame


_EARTH_ROTATION_RATE_RAD_S = 7.2921159e-5
_KM_TO_M = 1000.0
_J2000_JD = 2451545.0
_J2000_EPOCH = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
_DEFAULT_MAX_TLE_AGE_DAYS = 14.0


@dataclass(frozen=True)
class SatelliteState:
    timestamp: datetime
    name: str
    catalog_number: int
    position_enu_m: np.ndarray
    velocity_enu_m_s: np.ndarray

    @property
    def slant_range_m(self) -> float:
        return float(np.linalg.norm(self.position_enu_m))


@dataclass(frozen=True)
class _TleRecord:
    name: str
    catalog_number: int
    epoch: datetime
    satrec: Satrec


class SatelliteEphemeris:
    def __init__(
        self,
        records: list[_TleRecord],
        reference_frame: ENUReferenceFrame,
        max_tle_age_days: float = _DEFAULT_MAX_TLE_AGE_DAYS,
    ) -> None:
        if not records:
            raise ValueError("No TLE records were parsed from the ephemeris file.")
        self._frame = reference_frame
        self._max_age_s = max_tle_age_days * 86400.0
        self._by_satellite: dict[int, list[_TleRecord]] = {}
        for record in records:
            self._by_satellite.setdefault(record.catalog_number, []).append(record)
        for series in self._by_satellite.values():
            series.sort(key=lambda item: item.epoch)
        self._names = {
            number: series[0].name for number, series in self._by_satellite.items()
        }

    @property
    def catalog_numbers(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_satellite))

    def __len__(self) -> int:
        return len(self._by_satellite)

    def state_at(self, moment: datetime) -> dict[str, SatelliteState]:
        states: dict[str, SatelliteState] = {}
        for number, series in self._by_satellite.items():
            record = self._nearest_record(series, moment)
            if record is None:
                continue
            state = self._propagate(record, moment)
            if state is not None:
                states[self._names[number]] = state
        return states

    def _nearest_record(self, series: list[_TleRecord], moment: datetime) -> _TleRecord | None:
        best: _TleRecord | None = None
        best_gap = math.inf
        for record in series:
            gap = abs((record.epoch - moment).total_seconds())
            if gap < best_gap:
                best, best_gap = record, gap
        if best is None or best_gap > self._max_age_s:
            return None
        return best

    def _propagate(self, record: _TleRecord, moment: datetime) -> SatelliteState | None:
        julian_day, fraction = jday(
            moment.year,
            moment.month,
            moment.day,
            moment.hour,
            moment.minute,
            moment.second + moment.microsecond / 1.0e6,
        )
        error, position_teme, velocity_teme = record.satrec.sgp4(julian_day, fraction)
        if error != 0:
            return None
        sidereal = self._greenwich_sidereal_time(julian_day + fraction)
        position_ecef, velocity_ecef = self._teme_to_ecef(
            np.array(position_teme), np.array(velocity_teme), sidereal
        )
        return SatelliteState(
            timestamp=moment,
            name=record.name,
            catalog_number=record.catalog_number,
            position_enu_m=self._frame.ecef_to_enu(position_ecef * _KM_TO_M),
            velocity_enu_m_s=self._frame.rotate_ecef_to_enu(velocity_ecef * _KM_TO_M),
        )

    @staticmethod
    def _teme_to_ecef(
        position: np.ndarray, velocity: np.ndarray, sidereal: float
    ) -> tuple[np.ndarray, np.ndarray]:
        cosine = math.cos(sidereal)
        sine = math.sin(sidereal)
        rotation = np.array(
            [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        position_ecef = rotation @ position
        angular_velocity = np.array([0.0, 0.0, _EARTH_ROTATION_RATE_RAD_S])
        velocity_ecef = rotation @ velocity - np.cross(angular_velocity, position_ecef)
        return position_ecef, velocity_ecef

    @staticmethod
    def _greenwich_sidereal_time(julian_day: float) -> float:
        centuries = (julian_day - _J2000_JD) / 36525.0
        degrees = (
            280.46061837
            + 360.98564736629 * (julian_day - _J2000_JD)
            + 0.000387933 * centuries * centuries
            - centuries**3 / 38_710_000.0
        )
        return math.radians(degrees % 360.0)


class TleEphemerisLoader:
    def __init__(
        self,
        reference_frame: ENUReferenceFrame | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        self._config = config or default_config()
        self._frame = reference_frame or build_reference_frame(self._config)

    def load(self) -> SatelliteEphemeris:
        path = self._require(self._config.datasets.resolve("tle_file"))
        records = self._parse(path)
        return SatelliteEphemeris(records, self._frame)

    def _parse(self, path: Path) -> list[_TleRecord]:
        # Keep only non-empty lines. Some downloads (e.g. Space-Track saved on
        # Windows) use double carriage returns ("\r\r\n"), which str.splitlines()
        # expands into blank lines between every TLE line. Dropping blanks here
        # lets the "1 "/"2 " pairing work regardless of line-ending quirks.
        lines = [
            stripped
            for line in path.read_text().splitlines()
            if (stripped := line.strip())
        ]
        records: list[_TleRecord] = []
        pending_name: str | None = None
        index = 0
        while index < len(lines):
            stripped = lines[index]
            if stripped.startswith("1 ") and index + 1 < len(lines):
                follow = lines[index + 1]
                if follow.startswith("2 "):
                    record = self._build_record(stripped, follow, pending_name)
                    if record is not None:
                        records.append(record)
                    pending_name = None
                    index += 2
                    continue
            elif not stripped.startswith("2 "):
                pending_name = stripped[2:].strip() if stripped.startswith("0 ") else stripped
            index += 1
        return records

    def _build_record(
        self, line_one: str, line_two: str, name: str | None
    ) -> _TleRecord | None:
        try:
            satrec = Satrec.twoline2rv(line_one, line_two)
        except (ValueError, RuntimeError):
            return None
        catalog_number = int(satrec.satnum)
        epoch = _J2000_EPOCH + timedelta(
            days=(satrec.jdsatepoch + satrec.jdsatepochF) - _J2000_JD
        )
        return _TleRecord(
            name=name or f"SAT-{catalog_number}",
            catalog_number=catalog_number,
            epoch=epoch,
            satrec=satrec,
        )

    @staticmethod
    def _require(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"TLE ephemeris dataset not found: {path}")
        return path


def load_satellite_ephemeris(
    reference_frame: ENUReferenceFrame | None = None,
    config: SimulationConfig | None = None,
) -> SatelliteEphemeris:
    return TleEphemerisLoader(reference_frame, config).load()


__all__ = [
    "SatelliteState",
    "SatelliteEphemeris",
    "TleEphemerisLoader",
    "load_satellite_ephemeris",
]