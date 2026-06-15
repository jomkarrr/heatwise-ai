from datetime import date
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from pipelines.data_ingestion.domain.errors import DownloadError
from pipelines.data_ingestion.domain.models import BoundingBox, CityBoundary, DateRange
from pipelines.data_ingestion.infrastructure.clients import osm


class FakeGeoDataFrame:
    def __init__(self, rows: int, empty: bool = False) -> None:
        self.rows = rows
        self.empty = empty

    def __len__(self) -> int:
        return self.rows

    def to_file(self, path: Path, driver: str) -> None:
        assert driver == "GeoJSON"
        path.write_text('{"type":"FeatureCollection","features":[{}]}', encoding="utf-8")


@pytest.fixture
def boundary(tmp_path: Path) -> CityBoundary:
    return CityBoundary(
        path=tmp_path / "boundary.geojson",
        bbox=BoundingBox(west=72.77, south=18.89, east=72.99, north=19.27),
        geojson={},
    )


@pytest.fixture
def date_range() -> DateRange:
    return DateRange(start=date(2024, 1, 1), end=date(2024, 1, 31))


def test_osm_bbox_conversion_uses_city_boundary_bbox(boundary: CityBoundary):
    assert osm._bbox_from_boundary(boundary) == (72.77, 18.89, 72.99, 19.27)


def test_osm_downloader_downloads_roads_and_buildings_from_bbox(monkeypatch, tmp_path: Path, boundary, date_range):
    calls = {}
    fake_ox = SimpleNamespace(settings=SimpleNamespace(timeout=None))

    def fake_graph_from_bbox(bbox, network_type):
        calls["road_bbox"] = bbox
        calls["network_type"] = network_type
        return "road-graph"

    def fake_graph_to_gdfs(graph, nodes, edges):
        calls["graph"] = graph
        calls["nodes"] = nodes
        calls["edges"] = edges
        return FakeGeoDataFrame(rows=3)

    def fake_features_from_bbox(bbox, tags):
        calls["building_bbox"] = bbox
        calls["tags"] = tags
        return FakeGeoDataFrame(rows=5)

    fake_ox.graph_from_bbox = fake_graph_from_bbox
    fake_ox.graph_to_gdfs = fake_graph_to_gdfs
    fake_ox.features_from_bbox = fake_features_from_bbox
    monkeypatch.setitem(sys.modules, "osmnx", fake_ox)

    result = osm.OsmDownloader(overpass_timeout=90).download(boundary, date_range, tmp_path)

    assert calls["road_bbox"] == (72.77, 18.89, 72.99, 19.27)
    assert calls["building_bbox"] == (72.77, 18.89, 72.99, 19.27)
    assert calls["network_type"] == "drive"
    assert calls["graph"] == "road-graph"
    assert calls["nodes"] is False
    assert calls["edges"] is True
    assert calls["tags"] == {"building": True}
    assert fake_ox.settings.timeout == 90
    assert result.files == [tmp_path / "roads.geojson", tmp_path / "buildings.geojson"]
    assert result.metadata == {
        "bbox": [72.77, 18.89, 72.99, 19.27],
        "network_type": "drive",
        "roads": 3,
        "buildings": 5,
    }
    assert (tmp_path / "roads.geojson").exists()
    assert (tmp_path / "buildings.geojson").exists()


def test_osm_downloader_skips_existing_files(monkeypatch, tmp_path: Path, boundary, date_range):
    (tmp_path / "roads.geojson").write_text("existing roads", encoding="utf-8")
    (tmp_path / "buildings.geojson").write_text("existing buildings", encoding="utf-8")
    fake_ox = SimpleNamespace(settings=SimpleNamespace(timeout=None))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("OSMnx should not be called when outputs already exist.")

    fake_ox.graph_from_bbox = fail_if_called
    fake_ox.graph_to_gdfs = fail_if_called
    fake_ox.features_from_bbox = fail_if_called
    monkeypatch.setitem(sys.modules, "osmnx", fake_ox)

    result = osm.OsmDownloader().download(boundary, date_range, tmp_path)

    assert result.files == [tmp_path / "roads.geojson", tmp_path / "buildings.geojson"]
    assert result.metadata == {
        "bbox": [72.77, 18.89, 72.99, 19.27],
        "network_type": "drive",
        "skipped": ["roads.geojson", "buildings.geojson"],
    }


def test_osm_downloader_wraps_osmnx_errors(monkeypatch, tmp_path: Path, boundary, date_range):
    fake_ox = SimpleNamespace(settings=SimpleNamespace(timeout=None))

    def fake_graph_from_bbox(*args, **kwargs):
        raise RuntimeError("Overpass timed out")

    fake_ox.graph_from_bbox = fake_graph_from_bbox
    monkeypatch.setitem(sys.modules, "osmnx", fake_ox)

    with pytest.raises(DownloadError, match="OSM download failed: Overpass timed out"):
        osm.OsmDownloader().download(boundary, date_range, tmp_path)


def test_osm_downloader_writes_empty_feature_collection(monkeypatch, tmp_path: Path, boundary, date_range):
    fake_ox = SimpleNamespace(settings=SimpleNamespace(timeout=None))
    fake_ox.graph_from_bbox = lambda bbox, network_type: "road-graph"
    fake_ox.graph_to_gdfs = lambda graph, nodes, edges: FakeGeoDataFrame(rows=0, empty=True)
    fake_ox.features_from_bbox = lambda bbox, tags: FakeGeoDataFrame(rows=0, empty=True)
    monkeypatch.setitem(sys.modules, "osmnx", fake_ox)

    osm.OsmDownloader().download(boundary, date_range, tmp_path)

    assert (tmp_path / "roads.geojson").read_text(encoding="utf-8") == '{"type":"FeatureCollection","features":[]}'
    assert (tmp_path / "buildings.geojson").read_text(encoding="utf-8") == '{"type":"FeatureCollection","features":[]}'
