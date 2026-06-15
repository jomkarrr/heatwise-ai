"""Application service that orchestrates all data ingestion providers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pipelines.data_ingestion.config import IngestionConfig
from pipelines.data_ingestion.domain.errors import IngestionError
from pipelines.data_ingestion.domain.models import DatasetResult, IngestionResult
from pipelines.data_ingestion.infrastructure.storage import GeoJsonBoundaryReader, LocalOutputRepository
from pipelines.data_ingestion.ports import BoundaryReader, DatasetDownloader, OutputRepository

LOGGER = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Coordinates boundary parsing, downloads, folder organization, and metadata."""

    def __init__(
        self,
        config: IngestionConfig,
        downloaders: list[DatasetDownloader],
        boundary_reader: BoundaryReader | None = None,
        output_repository: OutputRepository | None = None,
    ) -> None:
        self.config = config
        self.downloaders = downloaders
        self.boundary_reader = boundary_reader or GeoJsonBoundaryReader()
        self.output_repository = output_repository or LocalOutputRepository(config.output_dir)

    def run(self) -> IngestionResult:
        LOGGER.info("Starting ingestion for city=%s boundary=%s", self.config.city, self.config.boundary_path)
        run_root = self.output_repository.prepare_city_run(self.config.city)
        boundary = self.boundary_reader.read(self.config.boundary_path)
        boundary_copy = run_root / "boundary" / self.config.boundary_path.name
        boundary_copy.write_text(json.dumps(boundary.geojson, indent=2), encoding="utf-8")

        datasets: list[DatasetResult] = []
        for downloader in self.downloaders:
            dataset_dir = self.output_repository.dataset_dir(run_root, downloader.source_name)
            try:
                LOGGER.info("Downloading %s into %s", downloader.source_name, dataset_dir)
                result = downloader.download(boundary, self.config.date_range, dataset_dir)
                datasets.append(result)
                LOGGER.info("Finished %s with %s files", downloader.source_name, len(result.files))
            except Exception as exc:
                LOGGER.exception("Failed downloading %s", downloader.source_name)
                raise IngestionError(f"{downloader.source_name} ingestion failed: {exc}") from exc

        ingestion_result = IngestionResult(city=self.config.city, output_root=run_root, datasets=datasets)
        self._write_manifest(ingestion_result)
        LOGGER.info("Completed ingestion for city=%s", self.config.city)
        return ingestion_result

    def _write_manifest(self, result: IngestionResult) -> Path:
        manifest = {
            "city": result.city,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_root": str(result.output_root),
            "datasets": [
                {
                    "source": dataset.source,
                    "files": [str(path) for path in dataset.files],
                    "metadata": dataset.metadata,
                }
                for dataset in result.datasets
            ],
        }
        path = result.output_root / "metadata" / "ingestion_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path


def build_default_pipeline(config: IngestionConfig) -> DataIngestionPipeline:
    from pipelines.data_ingestion.infrastructure.clients.era5 import Era5Downloader
    from pipelines.data_ingestion.infrastructure.clients.landsat import Landsat8Downloader
    from pipelines.data_ingestion.infrastructure.clients.osm import OsmDownloader
    from pipelines.data_ingestion.infrastructure.clients.sentinel import Sentinel2Downloader

    downloaders = [
        Landsat8Downloader(max_items=config.max_items, cloud_cover=config.landsat_cloud_cover),
        Sentinel2Downloader(max_items=config.max_items, cloud_cover=config.sentinel_cloud_cover),
        Era5Downloader(),
        OsmDownloader(),
    ]
    return DataIngestionPipeline(config=config, downloaders=downloaders)
