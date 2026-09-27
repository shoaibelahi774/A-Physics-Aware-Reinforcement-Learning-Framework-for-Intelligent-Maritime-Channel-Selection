from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.simulation_config import RESULTS_DIR, SimulationConfig, default_config
from gbsm.data_ingestion.external_data import ExternalDataBundle


_AUDIT_FILENAME = "external_data_audit.json"


class DataAuditor:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_config()

    def audit(self, bundle: ExternalDataBundle) -> dict[str, Any]:
        sources = {
            "sea_state": self._audit_sea_state(bundle),
            "ship_trajectories": self._audit_trajectories(bundle),
            "telemetry": self._audit_telemetry(bundle),
            "coastal_buildings": self._audit_buildings(bundle),
            "satellite_ephemeris": self._audit_satellites(bundle),
        }
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reference_point": {
                "latitude_deg": self._config.reference_latitude_deg,
                "longitude_deg": self._config.reference_longitude_deg,
            },
            "carrier_frequency_hz": self._config.carrier_frequency_hz,
            "sources": sources,
            "summary": self._summarize(sources),
            "provenance": dict(bundle.provenance),
        }

    def write(self, report: dict[str, Any], path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else RESULTS_DIR / _AUDIT_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2))
        return target

    def run(self, bundle: ExternalDataBundle, path: str | Path | None = None) -> dict[str, Any]:
        report = self.audit(bundle)
        self.write(report, path)
        return report

    def _audit_sea_state(self, bundle: ExternalDataBundle) -> dict[str, Any]:
        series = bundle.sea_state
        if series is None or len(series) == 0:
            return {"status": "absent"}
        heights = [getattr(record, "significant_wave_height_m", float("nan")) for record in series]
        winds = [getattr(record, "wind_speed_10m_m_s", float("nan")) for record in series]
        return {
            "status": "real",
            "record_count": len(series),
            "start": series[0].timestamp.isoformat(),
            "end": series[-1].timestamp.isoformat(),
            "significant_wave_height_m": self._range(heights),
            "wind_speed_10m_m_s": self._range(winds),
            "path": self._path("wave_file"),
        }

    def _audit_trajectories(self, bundle: ExternalDataBundle) -> dict[str, Any]:
        tracks = dict(bundle.trajectories or {})
        if not tracks:
            return {"status": "absent", "vessel_count": 0}
        vessels = []
        for identifier, trajectory in tracks.items():
            speeds = [sample.speed_m_s for sample in trajectory.samples]
            vessels.append(
                {
                    "id": identifier,
                    "sample_count": len(trajectory.samples),
                    "start": trajectory.start_time.isoformat(),
                    "end": trajectory.end_time.isoformat(),
                    "mean_speed_m_s": round(sum(speeds) / len(speeds), 2) if speeds else None,
                }
            )
        return {
            "status": "real",
            "vessel_count": len(vessels),
            "vessels": vessels,
            "path": self._path("ais_file"),
        }

    def _audit_telemetry(self, bundle: ExternalDataBundle) -> dict[str, Any]:
        observations = tuple(bundle.telemetry or ())
        if not observations:
            return {"status": "absent", "observation_count": 0}
        return {
            "status": "real",
            "observation_count": len(observations),
            "sources": sorted({observation.source for observation in observations}),
        }

    def _audit_buildings(self, bundle: ExternalDataBundle) -> dict[str, Any]:
        inventory = bundle.building_inventory
        if inventory is None:
            return {"status": "absent"}
        measured = getattr(inventory, "measured_height_count", 0)
        return {
            "status": "real",
            "building_count": inventory.building_count,
            "region_area_km2": round(inventory.region_area_km2, 3),
            "occupied_area_ratio": round(inventory.occupied_area_ratio, 4),
            "building_density_per_km2": round(inventory.building_density_per_km2, 1),
            "rayleigh_height_scale_m": round(inventory.rayleigh_height_scale_m, 2),
            "mean_building_height_m": round(inventory.mean_building_height_m, 2),
            "measured_height_count": measured,
            "height_source": "measured" if measured > 0 else "paper_rayleigh_fallback",
            "path": self._path("building_file"),
        }

    def _audit_satellites(self, bundle: ExternalDataBundle) -> dict[str, Any]:
        ephemeris = bundle.satellite_ephemeris
        if ephemeris is None:
            return {"status": "absent"}
        return {
            "status": "real",
            "satellite_count": len(ephemeris),
            "catalog_numbers": list(ephemeris.catalog_numbers),
            "path": self._path("tle_file"),
        }

    @staticmethod
    def _summarize(sources: dict[str, Any]) -> dict[str, Any]:
        return {
            "real_sources": [name for name, value in sources.items() if value.get("status") == "real"],
            "absent_sources": [name for name, value in sources.items() if value.get("status") == "absent"],
            "fallbacks_applied": [
                name
                for name, value in sources.items()
                if str(value.get("height_source", "")).endswith("fallback")
            ],
        }

    @staticmethod
    def _range(values) -> dict[str, float] | None:
        clean = [float(value) for value in values if value == value]
        if not clean:
            return None
        return {
            "min": round(min(clean), 3),
            "mean": round(sum(clean) / len(clean), 3),
            "max": round(max(clean), 3),
        }

    def _path(self, name: str) -> str | None:
        try:
            return str(self._config.datasets.resolve(name))
        except Exception:
            return None


def run_data_audit(
    bundle: ExternalDataBundle,
    config: SimulationConfig | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    return DataAuditor(config).run(bundle, path)


__all__ = ["DataAuditor", "run_data_audit"]