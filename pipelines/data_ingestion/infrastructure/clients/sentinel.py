"""Sentinel-2 Level-2A downloader."""

from __future__ import annotations

from pathlib import Path

from pipelines.data_ingestion.domain.models import CityBoundary, DatasetResult, DateRange
from pipelines.data_ingestion.infrastructure.clients.stac_utils import download_assets, search_planetary_computer_items


class Sentinel2Downloader:
    """Downloads Sentinel-2 bands commonly used for NDVI and built-up indices."""

    source_name = "sentinel"

    def __init__(self, max_items: int = 5, cloud_cover: int = 20) -> None:
        self.max_items = max_items
        self.cloud_cover = cloud_cover
        self.asset_keys = ["B02", "B03", "B04", "B08", "B11", "SCL"]

    def download(self, boundary: CityBoundary, date_range: DateRange, output_dir: Path) -> DatasetResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        existing_files = _existing_required_assets(output_dir, self.asset_keys)
        if existing_files is not None:
            return DatasetResult(
                source=self.source_name,
                files=existing_files,
                metadata={
                    "collection": "sentinel-2-l2a",
                    "items": 0,
                    "cloud_cover_lt": self.cloud_cover,
                    "skipped": True,
                    "reason": "required assets already exist",
                },
            )

        items = search_planetary_computer_items(
            collection="sentinel-2-l2a",
            bbox=boundary.bbox.as_stac_bbox(),
            datetime_range=date_range.as_stac_datetime(),
            query={"eo:cloud_cover": {"lt": self.cloud_cover}},
            max_items=self.max_items,
        )
        files = download_assets(items, self.asset_keys, output_dir, "sentinel2")
        return DatasetResult(
            source=self.source_name,
            files=files,
            metadata={"collection": "sentinel-2-l2a", "items": len(items), "cloud_cover_lt": self.cloud_cover},
        )


def _existing_required_assets(output_dir: Path, asset_keys: list[str]) -> list[Path] | None:
    files: list[Path] = []
    for asset_key in asset_keys:
        matches = sorted(
            path
            for path in output_dir.glob(f"sentinel2_*_{asset_key}.*")
            if path.is_file() and path.suffix != ".part"
        )
        if not matches:
            return None
        files.append(matches[0])
    return files
