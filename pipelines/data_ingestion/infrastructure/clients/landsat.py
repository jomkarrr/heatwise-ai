"""Landsat 8 Collection 2 Level-2 downloader."""

from __future__ import annotations

from pathlib import Path

from pipelines.data_ingestion.domain.models import CityBoundary, DatasetResult, DateRange
from pipelines.data_ingestion.infrastructure.clients.stac_utils import download_assets, search_planetary_computer_items


class Landsat8Downloader:
    """Downloads Landsat 8 surface reflectance and thermal assets."""

    source_name = "landsat"
    asset_keys = ("red", "nir08", "swir16", "lwir11")

    def __init__(self, max_items: int = 5, cloud_cover: int = 30) -> None:
        self.max_items = max_items
        self.cloud_cover = cloud_cover

    def download(self, boundary: CityBoundary, date_range: DateRange, output_dir: Path) -> DatasetResult:
        items = search_planetary_computer_items(
            collection="landsat-c2-l2",
            bbox=boundary.bbox.as_stac_bbox(),
            datetime_range=date_range.as_stac_datetime(),
            query={
                "platform": {"eq": "landsat-8"},
                "eo:cloud_cover": {"lt": self.cloud_cover},
            },
            max_items=self.max_items,
        )
        files = download_assets(items, list(self.asset_keys), output_dir, "landsat8")
        return DatasetResult(
            source=self.source_name,
            files=files,
            metadata={"collection": "landsat-c2-l2", "items": len(items), "cloud_cover_lt": self.cloud_cover},
        )
