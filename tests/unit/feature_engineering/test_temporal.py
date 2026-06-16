from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError
from pipelines.feature_engineering.infrastructure.temporal import TemporalAggregator


def test_temporal_aggregator_calculates_mean_values(tmp_path: Path):
    first_path = tmp_path / "thermal-1.tif"
    second_path = tmp_path / "thermal-2.tif"
    output_path = tmp_path / "features" / "temporal-mean.tif"

    _write_raster(first_path, np.array([[1, 2], [3, 4]], dtype=np.float32), nodata=-9999)
    _write_raster(second_path, np.array([[5, 6], [7, 8]], dtype=np.float32), nodata=-9999)

    result = TemporalAggregator().aggregate([first_path, second_path], output_path, "mean")

    with rasterio.open(result) as src:
        aggregate = src.read(1)

    expected = np.array([[3, 4], [5, 6]], dtype=np.float32)
    np.testing.assert_allclose(aggregate, expected, rtol=1e-6)


def test_temporal_aggregator_calculates_max_values(tmp_path: Path):
    first_path = tmp_path / "thermal-1.tif"
    second_path = tmp_path / "thermal-2.tif"
    output_path = tmp_path / "temporal-max.tif"

    _write_raster(first_path, np.array([[1, 8], [3, 4]], dtype=np.float32), nodata=-9999)
    _write_raster(second_path, np.array([[5, 6], [7, 2]], dtype=np.float32), nodata=-9999)

    TemporalAggregator().aggregate([first_path, second_path], output_path, "max")

    with rasterio.open(output_path) as src:
        aggregate = src.read(1)

    expected = np.array([[5, 8], [7, 4]], dtype=np.float32)
    np.testing.assert_allclose(aggregate, expected, rtol=1e-6)


def test_temporal_aggregator_ignores_nodata_and_preserves_empty_pixels(tmp_path: Path):
    first_path = tmp_path / "thermal-1.tif"
    second_path = tmp_path / "thermal-2.tif"
    output_path = tmp_path / "temporal-mean.tif"
    nodata = -9999

    _write_raster(
        first_path,
        np.array([[nodata, 2], [nodata, 4]], dtype=np.float32),
        nodata=nodata,
    )
    _write_raster(
        second_path,
        np.array([[nodata, 6], [7, nodata]], dtype=np.float32),
        nodata=nodata,
    )

    TemporalAggregator().aggregate([first_path, second_path], output_path, "mean")

    with rasterio.open(output_path) as src:
        aggregate = src.read(1)

    np.testing.assert_array_equal(
        aggregate,
        np.array([[nodata, 4], [7, 4]], dtype=np.float32),
    )


def test_temporal_aggregator_preserves_reference_metadata(tmp_path: Path):
    first_path = tmp_path / "thermal-1.tif"
    second_path = tmp_path / "thermal-2.tif"
    output_path = tmp_path / "temporal-mean.tif"
    transform = from_origin(72.0, 19.0, 30.0, 30.0)
    nodata = -9999

    _write_raster(
        first_path,
        np.array([[1, 2], [3, 4]], dtype=np.float32),
        nodata=nodata,
        crs="EPSG:32643",
        transform=transform,
    )
    _write_raster(
        second_path,
        np.array([[5, 6], [7, 8]], dtype=np.float32),
        nodata=nodata,
        crs="EPSG:32643",
        transform=transform,
    )

    TemporalAggregator().aggregate([first_path, second_path], output_path, "mean")

    with rasterio.open(first_path) as first_src, rasterio.open(output_path) as output_src:
        assert output_src.crs == first_src.crs
        assert output_src.transform == first_src.transform
        assert output_src.width == first_src.width
        assert output_src.height == first_src.height
        assert output_src.nodata == nodata
        assert output_src.count == 1
        assert output_src.dtypes == ("float32",)


def test_temporal_aggregator_creates_output_file_and_parent_directories(tmp_path: Path):
    input_path = tmp_path / "thermal.tif"
    output_path = tmp_path / "nested" / "features" / "temporal-mean.tif"

    _write_raster(input_path, np.array([[1]], dtype=np.float32), nodata=-9999)

    result = TemporalAggregator().aggregate([input_path], output_path, "mean")

    assert result == output_path
    assert output_path.exists()


def test_temporal_aggregator_rejects_incompatible_rasters(tmp_path: Path):
    first_path = tmp_path / "thermal-1.tif"
    second_path = tmp_path / "thermal-2.tif"
    output_path = tmp_path / "temporal-mean.tif"

    _write_raster(first_path, np.array([[1, 2]], dtype=np.float32), nodata=-9999)
    _write_raster(second_path, np.array([[1], [2]], dtype=np.float32), nodata=-9999)

    with pytest.raises(FeatureEngineeringError, match="different dimensions"):
        TemporalAggregator().aggregate([first_path, second_path], output_path, "mean")


def test_temporal_aggregator_rejects_unsupported_statistic(tmp_path: Path):
    input_path = tmp_path / "thermal.tif"
    output_path = tmp_path / "temporal-median.tif"

    _write_raster(input_path, np.array([[1]], dtype=np.float32), nodata=-9999)

    with pytest.raises(FeatureEngineeringError, match="Unsupported temporal aggregation statistic"):
        TemporalAggregator().aggregate([input_path], output_path, "median")


def test_temporal_aggregator_rejects_empty_input_list(tmp_path: Path):
    with pytest.raises(FeatureEngineeringError, match="requires at least one input raster"):
        TemporalAggregator().aggregate([], tmp_path / "temporal-mean.tif", "mean")


def _write_raster(
    path: Path,
    data: np.ndarray,
    nodata: float | None,
    crs: str = "EPSG:4326",
    transform=None,
) -> None:
    if transform is None:
        transform = from_origin(0.0, 2.0, 1.0, 1.0)

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
