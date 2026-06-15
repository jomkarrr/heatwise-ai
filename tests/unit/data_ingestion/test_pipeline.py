import json
from datetime import date
from pathlib import Path

import pytest

from pipelines.data_ingestion.application.pipeline import DataIngestionPipeline
from pipelines.data_ingestion.config import IngestionConfig
from pipelines.data_ingestion.domain.errors import IngestionError
from pipelines.data_ingestion.domain.models import CityBoundary, DatasetResult, DateRange


class FakeDownloader:
    source_name = "fake_source"

    def download(self, boundary: CityBoundary, date_range: DateRange, output_dir: Path) -> DatasetResult:
        assert boundary.bbox.west == 72.77
        assert date_range.start == date(2024, 1, 1)
        output_file = output_dir / "sample.txt"
        output_file.write_text("downloaded", encoding="utf-8")
        return DatasetResult(source=self.source_name, files=[output_file], metadata={"ok": True})


class FailingDownloader:
    source_name = "failing_source"

    def download(self, boundary: CityBoundary, date_range: DateRange, output_dir: Path) -> DatasetResult:
        raise RuntimeError("provider unavailable")


def test_pipeline_writes_structured_outputs_and_manifest(tmp_path: Path):
    boundary_path = _write_boundary(tmp_path)
    config = IngestionConfig(
        city="mumbai",
        boundary_path=boundary_path,
        output_dir=tmp_path / "raw",
        date_range=DateRange(start=date(2024, 1, 1), end=date(2024, 1, 31)),
    )

    result = DataIngestionPipeline(config=config, downloaders=[FakeDownloader()]).run()

    run_root = tmp_path / "raw" / "mumbai"
    assert result.output_root == run_root
    assert (run_root / "boundary" / "boundary.geojson").exists()
    assert (run_root / "fake_source" / "sample.txt").exists()

    manifest = json.loads((run_root / "metadata" / "ingestion_manifest.json").read_text(encoding="utf-8"))
    assert manifest["city"] == "mumbai"
    assert manifest["datasets"][0]["source"] == "fake_source"
    assert manifest["datasets"][0]["metadata"] == {"ok": True}


def test_pipeline_wraps_provider_errors(tmp_path: Path):
    boundary_path = _write_boundary(tmp_path)
    config = IngestionConfig(city="mumbai", boundary_path=boundary_path, output_dir=tmp_path / "raw")

    with pytest.raises(IngestionError, match="failing_source ingestion failed"):
        DataIngestionPipeline(config=config, downloaders=[FailingDownloader()]).run()


def _write_boundary(tmp_path: Path) -> Path:
    boundary_path = tmp_path / "boundary.geojson"
    boundary_path.write_text(
        json.dumps(
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [72.77, 18.89],
                        [72.99, 18.89],
                        [72.99, 19.27],
                        [72.77, 19.27],
                        [72.77, 18.89],
                    ]
                ],
            }
        ),
        encoding="utf-8",
    )
    return boundary_path
