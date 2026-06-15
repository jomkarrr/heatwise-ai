"""OpenStreetMap downloader for roads and building footprints."""

from __future__ import annotations

from pathlib import Path

from pipelines.data_ingestion.domain.errors import DownloadError
from pipelines.data_ingestion.domain.models import CityBoundary, DatasetResult, DateRange


class OsmDownloader:
    """Downloads OSM roads and buildings for the city bounding box."""

    source_name = "osm"

    def __init__(self, network_type: str = "drive", overpass_timeout: int = 180) -> None:
        self.network_type = network_type
        self.overpass_timeout = overpass_timeout

    def download(self, boundary: CityBoundary, date_range: DateRange, output_dir: Path) -> DatasetResult:
        del date_range
        try:
            import osmnx as ox
        except ImportError as exc:
            raise DownloadError("OSM downloads require osmnx and geopandas.") from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        road_path = output_dir / "roads.geojson"
        building_path = output_dir / "buildings.geojson"
        bbox = _bbox_from_boundary(boundary)

        skipped: list[str] = []
        metadata: dict = {"bbox": list(bbox), "network_type": self.network_type}

        try:
            if hasattr(ox, "settings"):
                ox.settings.timeout = self.overpass_timeout

            if road_path.exists():
                skipped.append(road_path.name)
            else:
                roads_graph = ox.graph_from_bbox(bbox, network_type=self.network_type)
                roads = ox.graph_to_gdfs(roads_graph, nodes=False, edges=True)
                _write_geojson(roads, road_path)
                metadata["roads"] = len(roads)

            if building_path.exists():
                skipped.append(building_path.name)
            else:
                buildings = ox.features_from_bbox(bbox, tags={"building": True})
                _write_geojson(buildings, building_path)
                metadata["buildings"] = len(buildings)
        except Exception as exc:
            raise DownloadError(f"OSM download failed: {exc}") from exc

        if skipped:
            metadata["skipped"] = skipped

        return DatasetResult(
            source=self.source_name,
            files=[road_path, building_path],
            metadata=metadata,
        )


def _bbox_from_boundary(boundary: CityBoundary) -> tuple[float, float, float, float]:
    """Return OSMnx v2 bbox order: west, south, east, north."""

    bbox = boundary.bbox
    return (bbox.west, bbox.south, bbox.east, bbox.north)


def _write_geojson(frame, path: Path) -> None:
    if frame.empty:
        path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        return
    frame.to_file(path, driver="GeoJSON")
