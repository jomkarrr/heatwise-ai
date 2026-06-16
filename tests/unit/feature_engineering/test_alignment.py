from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError
from pipelines.feature_engineering.infrastructure.alignment import RasterAligner


def test_raster_aligner_matches_reference_crs(tmp_path: Path):
    source_path = tmp_path / "source.tif"
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "aligned.tif"

    _write_raster(source_path, np.array([[1]], dtype=np.float32), nodata=-9999, crs="EPSG:4326")
    _write_raster(reference_path, np.array([[1]], dtype=np.float32), nodata=-9999, crs="EPSG:32643")

    RasterAligner().align(source_path, reference_path, output_path)

    with rasterio.open(reference_path) as reference_src, rasterio.open(output_path) as output_src:
        assert output_src.crs == reference_src.crs


def test_raster_aligner_matches_reference_transform(tmp_path: Path):
    source_path = tmp_path / "source.tif"
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "aligned.tif"
    source_transform = from_origin(0.0, 2.0, 1.0, 1.0)
    reference_transform = from_origin(72.0, 19.0, 30.0, 30.0)

    _write_raster(source_path, np.array([[1]], dtype=np.float32), nodata=-9999, transform=source_transform)
    _write_raster(reference_path, np.array([[1]], dtype=np.float32), nodata=-9999, transform=reference_transform)

    RasterAligner().align(source_path, reference_path, output_path)

    with rasterio.open(reference_path) as reference_src, rasterio.open(output_path) as output_src:
        assert output_src.transform == reference_src.transform


def test_raster_aligner_matches_reference_dimensions(tmp_path: Path):
    source_path = tmp_path / "source.tif"
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "aligned.tif"

    _write_raster(source_path, np.array([[1, 2], [3, 4]], dtype=np.float32), nodata=-9999)
    _write_raster(reference_path, np.zeros((3, 4), dtype=np.float32), nodata=-9999)

    RasterAligner().align(source_path, reference_path, output_path)

    with rasterio.open(reference_path) as reference_src, rasterio.open(output_path) as output_src:
        assert output_src.width == reference_src.width
        assert output_src.height == reference_src.height
        assert output_src.count == 1
        assert output_src.dtypes == ("float32",)


def test_raster_aligner_preserves_nodata(tmp_path: Path):
    source_path = tmp_path / "source.tif"
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "aligned.tif"
    nodata = -9999
    transform = from_origin(0.0, 2.0, 1.0, 1.0)

    _write_raster(
        source_path,
        np.array([[nodata, 2], [3, 4]], dtype=np.float32),
        nodata=nodata,
        transform=transform,
    )
    _write_raster(
        reference_path,
        np.ones((2, 2), dtype=np.float32),
        nodata=-32768,
        transform=transform,
    )

    RasterAligner().align(source_path, reference_path, output_path)

    with rasterio.open(output_path) as output_src:
        aligned = output_src.read(1)

        assert output_src.nodata == nodata
        assert aligned[0, 0] == nodata


def test_raster_aligner_creates_output_file_and_parent_directories(tmp_path: Path):
    source_path = tmp_path / "source.tif"
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "nested" / "features" / "aligned.tif"

    _write_raster(source_path, np.array([[1]], dtype=np.float32), nodata=-9999)
    _write_raster(reference_path, np.array([[1]], dtype=np.float32), nodata=-9999)

    result = RasterAligner().align(source_path, reference_path, output_path)

    assert result == output_path
    assert output_path.exists()


def test_raster_aligner_produces_valid_output_for_already_aligned_rasters(tmp_path: Path):
    source_path = tmp_path / "source.tif"
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "aligned.tif"
    transform = from_origin(72.0, 19.0, 30.0, 30.0)
    source = np.array([[1, 2], [3, 4]], dtype=np.float32)

    _write_raster(source_path, source, nodata=-9999, crs="EPSG:32643", transform=transform)
    _write_raster(
        reference_path,
        np.ones((2, 2), dtype=np.float32),
        nodata=-9999,
        crs="EPSG:32643",
        transform=transform,
    )

    RasterAligner().align(source_path, reference_path, output_path)

    with rasterio.open(output_path) as output_src:
        aligned = output_src.read(1)

    np.testing.assert_allclose(aligned, source, rtol=1e-6)


def test_raster_aligner_wraps_failures_in_feature_engineering_error(tmp_path: Path):
    with pytest.raises(FeatureEngineeringError, match="Failed to align raster"):
        RasterAligner().align(
            tmp_path / "missing-source.tif",
            tmp_path / "missing-reference.tif",
            tmp_path / "aligned.tif",
        )


def _write_raster(
    path: Path,
    data: np.ndarray,
    nodata: float | None,
    crs: str = "EPSG:4326",
    transform=None,
) -> None:
    if transform is None:
        transform = from_origin(0.0, float(data.shape[0]), 1.0, 1.0)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
