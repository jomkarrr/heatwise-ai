"""Configuration primitives for the data ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from pipelines.data_ingestion.domain.models import DateRange


@dataclass(frozen=True)
class IngestionConfig:
    """Runtime settings for one ingestion run."""

    city: str
    boundary_path: Path
    output_dir: Path = Path("data/raw")
    date_range: DateRange = field(
        default_factory=lambda: DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31))
    )
    max_items: int = 5
    landsat_cloud_cover: int = 30
    sentinel_cloud_cover: int = 20

    @classmethod
    def from_strings(
        cls,
        city: str,
        boundary_path: str,
        output_dir: str,
        start_date: str,
        end_date: str,
        max_items: int,
    ) -> "IngestionConfig":
        return cls(
            city=city,
            boundary_path=Path(boundary_path),
            output_dir=Path(output_dir),
            date_range=DateRange(start=date.fromisoformat(start_date), end=date.fromisoformat(end_date)),
            max_items=max_items,
        )
