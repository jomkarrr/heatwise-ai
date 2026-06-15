"""Ports used by the application layer."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pipelines.data_ingestion.domain.models import CityBoundary, DatasetResult, DateRange


class BoundaryReader(Protocol):
    def read(self, path: Path) -> CityBoundary:
        """Read and validate a city boundary file."""


class DatasetDownloader(Protocol):
    source_name: str

    def download(self, boundary: CityBoundary, date_range: DateRange, output_dir: Path) -> DatasetResult:
        """Download a dataset for the boundary and date range."""


class OutputRepository(Protocol):
    def prepare_city_run(self, city: str) -> Path:
        """Create and return the output root for a city run."""

    def dataset_dir(self, run_root: Path, dataset_name: str) -> Path:
        """Create and return a dataset-specific output directory."""
