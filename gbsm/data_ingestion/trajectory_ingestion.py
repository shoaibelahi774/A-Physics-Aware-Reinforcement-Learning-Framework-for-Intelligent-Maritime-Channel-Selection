from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from config.simulation_config import SimulationConfig, default_config
from gbsm.data_ingestion.external_data import (
    ENUReferenceFrame,
    KinematicSample,
    Trajectory,
    build_reference_frame,
)


_KNOTS_TO_M_S = 0.514444
_REQUIRED_COLUMNS = ("Timestamp", "MMSI", "Latitude", "Longitude", "SOG", "COG")
_TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"


class AISTrajectoryLoader:
    def __init__(
        self,
        reference_frame: ENUReferenceFrame | None = None,
        config: SimulationConfig | None = None,
        resample_interval_s: float = 30.0,
    ) -> None:
        self._config = config or default_config()
        self._frame = reference_frame or build_reference_frame(self._config)
        self._resample_interval_s = max(1.0, float(resample_interval_s))

    def load(self) -> dict[str, Trajectory]:
        frame = self._read_frame()
        trajectories: dict[str, Trajectory] = {}
        for mmsi, group in frame.groupby("MMSI", sort=False):
            trajectory = self._build_trajectory(str(int(mmsi)), group)
            if trajectory is not None:
                trajectories[trajectory.identifier] = trajectory
        return dict(
            sorted(trajectories.items(), key=lambda item: len(item[1].samples), reverse=True)
        )

    def load_longest(self, count: int) -> dict[str, Trajectory]:
        return dict(list(self.load().items())[: max(0, count)])

    def _read_frame(self) -> pd.DataFrame:
        path = self._require(self._config.datasets.resolve("ais_file"))
        frame = pd.read_csv(path)
        frame.columns = [column.strip() for column in frame.columns]
        missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"AIS file is missing required columns: {missing}")
        frame["Timestamp"] = pd.to_datetime(
            frame["Timestamp"], format=_TIMESTAMP_FORMAT, errors="coerce"
        )
        for column in ("Latitude", "Longitude", "SOG", "COG"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=list(_REQUIRED_COLUMNS))
        return frame[
            frame["Latitude"].between(-90.0, 90.0)
            & frame["Longitude"].between(-180.0, 180.0)
            & frame["SOG"].between(0.0, 80.0)
        ]

    def _build_trajectory(self, identifier: str, group: pd.DataFrame) -> Trajectory | None:
        ordered = group.drop_duplicates("Timestamp").sort_values("Timestamp")
        bucket = ordered["Timestamp"].dt.floor(f"{int(self._resample_interval_s)}s")
        ordered = ordered.loc[~bucket.duplicated()]
        if len(ordered) < 2:
            return None
        epochs = ordered["Timestamp"].values.astype("datetime64[s]").astype("int64")
        latitudes = ordered["Latitude"].to_numpy(dtype=float)
        longitudes = ordered["Longitude"].to_numpy(dtype=float)
        speeds = ordered["SOG"].to_numpy(dtype=float) * _KNOTS_TO_M_S
        courses = np.radians(ordered["COG"].to_numpy(dtype=float))
        east = speeds * np.sin(courses)
        north = speeds * np.cos(courses)
        samples = [
            KinematicSample(
                timestamp=datetime.fromtimestamp(int(epochs[i]), tz=timezone.utc),
                position_enu_m=self._frame.geodetic_to_enu(latitudes[i], longitudes[i], 0.0),
                velocity_enu_m_s=np.array([east[i], north[i], 0.0]),
            )
            for i in range(len(ordered))
        ]
        return Trajectory(identifier, samples)

    @staticmethod
    def _require(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"AIS trajectory dataset not found: {path}")
        return path


def load_ais_trajectories(
    reference_frame: ENUReferenceFrame | None = None,
    config: SimulationConfig | None = None,
) -> dict[str, Trajectory]:
    return AISTrajectoryLoader(reference_frame, config).load()