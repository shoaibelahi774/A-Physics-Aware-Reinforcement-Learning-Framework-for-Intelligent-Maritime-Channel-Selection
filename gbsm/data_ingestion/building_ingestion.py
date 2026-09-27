from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config.paper_parameters import BUILDINGS, BuildingParameters
from config.simulation_config import SimulationConfig, default_config
from gbsm.data_ingestion.external_data import ENUReferenceFrame, build_reference_frame


_STOREY_HEIGHT_M = 3.0
_BUILDING_KEY = "building"


@dataclass(frozen=True)
class BuildingInventory:
    building_count: int
    region_area_km2: float
    occupied_area_ratio: float
    building_density_per_km2: float
    rayleigh_height_scale_m: float
    mean_building_height_m: float
    average_building_width_m: float
    measured_height_count: int
    centroids_enu_m: np.ndarray
    footprint_areas_m2: np.ndarray
    heights_m: np.ndarray

    def to_building_parameters(self) -> BuildingParameters:
        return BuildingParameters(
            rayleigh_height_scale_m=self.rayleigh_height_scale_m,
            occupied_area_ratio=self.occupied_area_ratio,
            building_density_per_km2=self.building_density_per_km2,
            sea_wave_search_step_m=BUILDINGS.sea_wave_search_step_m,
        )


class BuildingFootprintLoader:
    def __init__(
        self,
        reference_frame: ENUReferenceFrame | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        self._config = config or default_config()
        self._frame = reference_frame or build_reference_frame(self._config)

    def load(self) -> BuildingInventory:
        features = self._read_features()
        centroids: list[np.ndarray] = []
        areas: list[float] = []
        heights: list[float] = []
        for feature in features:
            parsed = self._parse_feature(feature)
            if parsed is None:
                continue
            centroid, area, height = parsed
            centroids.append(centroid)
            areas.append(area)
            heights.append(height)
        if not areas:
            raise ValueError("No building polygons were found in the footprint dataset.")
        return self._assemble(np.array(centroids), np.array(areas), np.array(heights))

    def _read_features(self) -> list[dict]:
        path = self._require(self._config.datasets.resolve("building_file"))
        with path.open(encoding="utf-8") as handle:
            collection = json.load(handle)
        return [
            feature
            for feature in collection.get("features", [])
            if feature.get("properties", {}).get(_BUILDING_KEY)
        ]

    def _parse_feature(self, feature: dict):
        rings = self._exterior_rings(feature.get("geometry", {}))
        if not rings:
            return None
        total_area = 0.0
        weighted_centroid = np.zeros(2)
        for ring in rings:
            projected = self._project_ring(ring)
            if projected is None:
                continue
            area, centroid = projected
            total_area += area
            weighted_centroid += area * centroid
        if total_area <= 0.0:
            return None
        centroid = weighted_centroid / total_area
        height = self._parse_height(feature.get("properties", {}))
        return centroid, total_area, height

    def _project_ring(self, ring: list):
        points = np.array(
            [self._frame.geodetic_to_enu(vertex[1], vertex[0], 0.0)[:2] for vertex in ring]
        )
        if len(points) < 3:
            return None
        x = points[:, 0]
        y = points[:, 1]
        area = abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0
        return area, points.mean(axis=0)

    def _assemble(
        self, centroids: np.ndarray, areas: np.ndarray, heights: np.ndarray
    ) -> BuildingInventory:
        east = centroids[:, 0]
        north = centroids[:, 1]
        region_area_m2 = (east.max() - east.min()) * (north.max() - north.min())
        region_area_km2 = max(region_area_m2 / 1.0e6, 1.0e-9)
        measured = heights[~np.isnan(heights)]
        if measured.size:
            rayleigh_scale = math.sqrt(float(np.mean(measured**2)) / 2.0)
            mean_height = float(measured.mean())
        else:
            rayleigh_scale = BUILDINGS.rayleigh_height_scale_m
            mean_height = BUILDINGS.rayleigh_height_scale_m * math.sqrt(math.pi / 2.0)
        return BuildingInventory(
            building_count=int(areas.size),
            region_area_km2=region_area_km2,
            occupied_area_ratio=min(float(areas.sum() / region_area_m2), 1.0),
            building_density_per_km2=float(areas.size / region_area_km2),
            rayleigh_height_scale_m=rayleigh_scale,
            mean_building_height_m=mean_height,
            average_building_width_m=float(np.sqrt(areas).mean()),
            measured_height_count=int(measured.size),
            centroids_enu_m=centroids,
            footprint_areas_m2=areas,
            heights_m=heights,
        )

    @staticmethod
    def _exterior_rings(geometry: dict) -> list:
        kind = geometry.get("type")
        if kind == "Polygon":
            coordinates = geometry.get("coordinates", [])
            return [coordinates[0]] if coordinates else []
        if kind == "MultiPolygon":
            return [polygon[0] for polygon in geometry.get("coordinates", []) if polygon]
        return []

    @staticmethod
    def _parse_height(properties: dict) -> float:
        explicit = properties.get("height")
        if explicit is not None:
            value = BuildingFootprintLoader._leading_float(explicit)
            if value is not None:
                return value
        levels = properties.get("building:levels")
        if levels is not None:
            value = BuildingFootprintLoader._leading_float(levels)
            if value is not None:
                return value * _STOREY_HEIGHT_M
        return math.nan

    @staticmethod
    def _leading_float(raw) -> float | None:
        try:
            return float(str(raw).split()[0].replace(",", "."))
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _require(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Building footprint dataset not found: {path}")
        return path


def load_building_inventory(
    reference_frame: ENUReferenceFrame | None = None,
    config: SimulationConfig | None = None,
) -> BuildingInventory:
    return BuildingFootprintLoader(reference_frame, config).load()


__all__ = [
    "BuildingInventory",
    "BuildingFootprintLoader",
    "load_building_inventory",
]