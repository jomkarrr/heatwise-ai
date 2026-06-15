from datetime import date
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from pipelines.data_ingestion.domain.models import BoundingBox, CityBoundary, DateRange
from pipelines.data_ingestion.infrastructure.clients import era5


def test_era5_expands_small_mumbai_suburban_bbox_to_minimum_retrieval_area():
    bbox = BoundingBox(west=72.775, south=19.033, east=72.99, north=19.272)

    expanded = era5._expand_bbox(bbox)

    assert expanded.west == pytest.approx(72.6325)
    assert expanded.east == pytest.approx(73.1325)
    assert expanded.south == pytest.approx(18.9025)
    assert expanded.north == pytest.approx(19.4025)
    assert expanded.as_cds_area() == pytest.approx([19.4025, 72.6325, 18.9025, 73.1325])
    assert bbox.as_stac_bbox() == [72.775, 19.033, 72.99, 19.272]


def test_era5_downloader_uses_expanded_area_and_data_format(monkeypatch, tmp_path: Path):
    requests = []

    class FakeClient:
        def retrieve(self, dataset, request, target):
            requests.append({"dataset": dataset, "request": request, "target": target})

    monkeypatch.setitem(sys.modules, "cdsapi", SimpleNamespace(Client=lambda: FakeClient()))
    boundary = CityBoundary(
        path=tmp_path / "mumbai-suburban.geojson",
        bbox=BoundingBox(west=72.775, south=19.033, east=72.99, north=19.272),
        geojson={"properties": {"name": "Mumbai Suburban"}},
    )
    date_range = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 31))

    result = era5.Era5Downloader(variables=["2m_temperature"]).download(boundary, date_range, tmp_path)

    request = requests[0]["request"]
    assert requests[0]["dataset"] == "reanalysis-era5-single-levels"
    assert requests[0]["target"] == str(tmp_path / "era5_single_levels.nc")
    assert request["area"] == pytest.approx([19.4025, 72.6325, 18.9025, 73.1325])
    assert request["data_format"] == "netcdf"
    assert "format" not in request
    assert boundary.bbox.as_stac_bbox() == [72.775, 19.033, 72.99, 19.272]
    assert result.files == [tmp_path / "era5_single_levels.nc"]
    assert result.metadata["dataset"] == "reanalysis-era5-single-levels"
    assert result.metadata["variables"] == ["2m_temperature"]
    assert result.metadata["boundary_bbox"] == [72.775, 19.033, 72.99, 19.272]
    assert result.metadata["retrieval_area"] == pytest.approx([19.4025, 72.6325, 18.9025, 73.1325])


def test_era5_keeps_large_bbox_unchanged_for_retrieval():
    bbox = BoundingBox(west=72.5, south=18.5, east=73.4, north=19.4)

    expanded = era5._expand_bbox(bbox)

    assert expanded == bbox
