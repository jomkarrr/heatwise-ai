"""Core domain models for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BoundingBox:
    """Geographic bounding box in WGS84 coordinates."""

    west: float
    south: float
    east: float
    north: float

    def as_stac_bbox(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    def as_cds_area(self) -> list[float]:
        return [self.north, self.west, self.south, self.east]


@dataclass(frozen=True)
class CityBoundary:
    """Parsed city boundary with derived metadata."""

    path: Path
    bbox: BoundingBox
    geojson: dict


@dataclass(frozen=True)
class DateRange:
    """Inclusive date range used by satellite and weather clients."""

    start: date
    end: date

    def as_stac_datetime(self) -> str:
        return f"{self.start.isoformat()}/{self.end.isoformat()}"

    def years(self) -> list[str]:
        return [str(year) for year in range(self.start.year, self.end.year + 1)]

    def months(self) -> list[str]:
        return _unique_padded_values(self.start.month, self.end.month, self.start.year, self.end.year, "month")

    def days(self) -> list[str]:
        return [f"{day:02d}" for day in range(1, 32)]


@dataclass(frozen=True)
class DatasetResult:
    """Result returned by a provider adapter."""

    source: str
    files: list[Path]
    metadata: dict


@dataclass(frozen=True)
class IngestionResult:
    """Aggregate result for a completed ingestion run."""

    city: str
    output_root: Path
    datasets: list[DatasetResult]

    @property
    def files(self) -> list[Path]:
        return [path for dataset in self.datasets for path in dataset.files]


def _unique_padded_values(start_month: int, end_month: int, start_year: int, end_year: int, _: str) -> list[str]:
    values: Iterable[int]
    if start_year == end_year:
        values = range(start_month, end_month + 1)
    else:
        values = range(1, 13)
    return [f"{value:02d}" for value in values]
