from datetime import date
from pathlib import Path

from pipelines.data_ingestion.domain.models import BoundingBox, CityBoundary, DateRange
from pipelines.data_ingestion.infrastructure.clients import sentinel


def test_sentinel_downloader_skips_stac_when_required_assets_exist(monkeypatch, tmp_path: Path):
    boundary = _boundary(tmp_path)
    date_range = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 31))
    expected_files = [
        tmp_path / f"sentinel2_01_item-1_{asset_key}.tif"
        for asset_key in ["B02", "B03", "B04", "B08", "B11", "SCL"]
    ]
    for path in expected_files:
        path.write_text("existing asset", encoding="utf-8")

    def fail_search(*args, **kwargs):
        raise AssertionError("STAC search should not run when all required assets already exist.")

    def fail_download(*args, **kwargs):
        raise AssertionError("Asset download should not run when all required assets already exist.")

    monkeypatch.setattr(sentinel, "search_planetary_computer_items", fail_search)
    monkeypatch.setattr(sentinel, "download_assets", fail_download)

    result = sentinel.Sentinel2Downloader(cloud_cover=15).download(boundary, date_range, tmp_path)

    assert result.files == expected_files
    assert result.metadata == {
        "collection": "sentinel-2-l2a",
        "items": 0,
        "cloud_cover_lt": 15,
        "skipped": True,
        "reason": "required assets already exist",
    }


def test_sentinel_downloader_queries_stac_when_any_required_asset_is_missing(monkeypatch, tmp_path: Path):
    boundary = _boundary(tmp_path)
    date_range = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 31))
    for asset_key in ["B02", "B03", "B04", "B08", "B11"]:
        (tmp_path / f"sentinel2_01_item-1_{asset_key}.tif").write_text("existing asset", encoding="utf-8")
    expected_files = [tmp_path / "sentinel2_01_item-1_SCL.tif"]
    searched = {}
    downloaded = {}

    def fake_search_planetary_computer_items(**kwargs):
        searched.update(kwargs)
        return ["item-1"]

    def fake_download_assets(items, asset_keys, output_dir, prefix):
        downloaded["items"] = items
        downloaded["asset_keys"] = asset_keys
        downloaded["output_dir"] = output_dir
        downloaded["prefix"] = prefix
        return expected_files

    monkeypatch.setattr(sentinel, "search_planetary_computer_items", fake_search_planetary_computer_items)
    monkeypatch.setattr(sentinel, "download_assets", fake_download_assets)

    result = sentinel.Sentinel2Downloader(max_items=2, cloud_cover=15).download(boundary, date_range, tmp_path)

    assert searched["collection"] == "sentinel-2-l2a"
    assert searched["bbox"] == [72.77, 18.89, 72.99, 19.27]
    assert searched["datetime_range"] == "2024-01-01/2024-01-31"
    assert searched["query"] == {"eo:cloud_cover": {"lt": 15}}
    assert searched["max_items"] == 2
    assert downloaded["items"] == ["item-1"]
    assert downloaded["asset_keys"] == ["B02", "B03", "B04", "B08", "B11", "SCL"]
    assert downloaded["output_dir"] == tmp_path
    assert downloaded["prefix"] == "sentinel2"
    assert result.files == expected_files
    assert result.metadata == {"collection": "sentinel-2-l2a", "items": 1, "cloud_cover_lt": 15}


def test_sentinel_existing_asset_detection_ignores_partial_files(tmp_path: Path):
    for asset_key in ["B02", "B03", "B04", "B08", "B11"]:
        (tmp_path / f"sentinel2_01_item-1_{asset_key}.tif").write_text("existing asset", encoding="utf-8")
    (tmp_path / "sentinel2_01_item-1_SCL.tif.part").write_text("partial asset", encoding="utf-8")

    assert sentinel._existing_required_assets(tmp_path, ["B02", "B03", "B04", "B08", "B11", "SCL"]) is None


def _boundary(tmp_path: Path) -> CityBoundary:
    return CityBoundary(
        path=tmp_path / "boundary.geojson",
        bbox=BoundingBox(west=72.77, south=18.89, east=72.99, north=19.27),
        geojson={},
    )
