from datetime import date
from pathlib import Path

from pipelines.data_ingestion.domain.models import BoundingBox, CityBoundary, DateRange
from pipelines.data_ingestion.infrastructure.clients import landsat


def test_landsat_downloader_requests_only_required_assets(monkeypatch, tmp_path: Path):
    searched = {}
    downloaded = {}
    expected_files = [tmp_path / "red.tif"]

    def fake_search_planetary_computer_items(**kwargs):
        searched.update(kwargs)
        return ["item-1"]

    def fake_download_assets(items, asset_keys, output_dir, prefix):
        downloaded["items"] = items
        downloaded["asset_keys"] = asset_keys
        downloaded["output_dir"] = output_dir
        downloaded["prefix"] = prefix
        return expected_files

    monkeypatch.setattr(landsat, "search_planetary_computer_items", fake_search_planetary_computer_items)
    monkeypatch.setattr(landsat, "download_assets", fake_download_assets)

    boundary = CityBoundary(
        path=tmp_path / "boundary.geojson",
        bbox=BoundingBox(west=72.77, south=18.89, east=72.99, north=19.27),
        geojson={},
    )
    date_range = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 31))

    result = landsat.Landsat8Downloader(max_items=2, cloud_cover=25).download(boundary, date_range, tmp_path)

    assert downloaded["asset_keys"] == ["red", "nir08", "swir16", "lwir11"]
    assert downloaded["items"] == ["item-1"]
    assert downloaded["output_dir"] == tmp_path
    assert downloaded["prefix"] == "landsat8"
    assert searched["collection"] == "landsat-c2-l2"
    assert searched["bbox"] == [72.77, 18.89, 72.99, 19.27]
    assert searched["datetime_range"] == "2024-01-01/2024-01-31"
    assert searched["query"] == {
        "platform": {"eq": "landsat-8"},
        "eo:cloud_cover": {"lt": 25},
    }
    assert searched["max_items"] == 2
    assert result.files == expected_files
    assert result.metadata == {"collection": "landsat-c2-l2", "items": 1, "cloud_cover_lt": 25}
