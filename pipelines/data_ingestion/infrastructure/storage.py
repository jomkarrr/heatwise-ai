"""Filesystem repository and GeoJSON boundary reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipelines.data_ingestion.domain.errors import BoundaryError
from pipelines.data_ingestion.domain.models import BoundingBox, CityBoundary


class LocalOutputRepository:
    """Creates the structured output folders used by all provider adapters."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def prepare_city_run(self, city: str) -> Path:
        run_root = self.base_dir / city.lower()
        for dataset in ("boundary", "landsat", "sentinel", "era5", "osm", "logs", "metadata"):
            (run_root / dataset).mkdir(parents=True, exist_ok=True)
        return run_root

    def dataset_dir(self, run_root: Path, dataset_name: str) -> Path:
        path = run_root / dataset_name
        path.mkdir(parents=True, exist_ok=True)
        return path


class GeoJsonBoundaryReader:
    """Reads a city boundary GeoJSON and computes its bounding box."""

    def read(self, path: Path) -> CityBoundary:
        if not path.exists():
            raise BoundaryError(f"Boundary file does not exist: {path}")

        try:
            geojson = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BoundaryError(f"Invalid GeoJSON in {path}: {exc}") from exc

        coordinates = list(_iter_positions(geojson))
        if not coordinates:
            raise BoundaryError(f"No coordinates found in boundary GeoJSON: {path}")

        longitudes = [position[0] for position in coordinates]
        latitudes = [position[1] for position in coordinates]
        bbox = BoundingBox(west=min(longitudes), south=min(latitudes), east=max(longitudes), north=max(latitudes))

        return CityBoundary(path=path, bbox=bbox, geojson=geojson)


def _iter_positions(node: Any):
    if isinstance(node, dict):
        if "coordinates" in node:
            yield from _iter_positions(node["coordinates"])
        else:
            for value in node.values():
                yield from _iter_positions(value)
    elif isinstance(node, list):
        if len(node) >= 2 and all(isinstance(value, (int, float)) for value in node[:2]):
            yield (float(node[0]), float(node[1]))
        else:
            for value in node:
                yield from _iter_positions(value)
