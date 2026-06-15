from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError
from pipelines.feature_engineering.infrastructure.thermal import ThermalGenerator


def test_thermal_generator_calculates_expected_values(tmp_path: Path):
    thermal_path = tmp_path / "lwir11.tif"
    output_path = tmp_path / "features" / "thermal.tif"

    _write_raster(
        thermal_path,
        np.array([[290, 300], [310, 330]], dtype=np.float32),
        nodata=-9999,
    )

    result = ThermalGenerator().generate(thermal_path, output_path)

    with rasterio.open(result) as src:
        thermal = src.read(1)

    expected = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(thermal, expected, rtol=1e-6)


def test_thermal_generator_keeps_valid_values_within_normalized_range(tmp_path: Path):
    thermal_path = tmp_path / "lwir11.tif"
    output_path = tmp_path / "thermal.tif"
    nodata = -9999

    _write_raster(
        thermal_path,
        np.array([[nodata, 240], [300, 360]], dtype=np.float32),
        nodata=nodata,
    )

    ThermalGenerator().generate(thermal_path, output_path)

    with rasterio.open(output_path) as src:
        thermal = src.read(1)

    valid_values = thermal[thermal != nodata]
    assert valid_values.min() >= 0
    assert valid_values.max() <= 1


def test_thermal_generator_preserves_metadata_and_nodata(tmp_path: Path):
    thermal_path = tmp_path / "lwir11.tif"
    output_path = tmp_path / "thermal.tif"
    transform = from_origin(72.0, 19.0, 30.0, 30.0)
    nodata = -9999

    _write_raster(
        thermal_path,
        np.array([[nodata, 290], [300, 310]], dtype=np.float32),
        nodata=nodata,
        crs="EPSG:32643",
        transform=transform,
    )

    ThermalGenerator().generate(thermal_path, output_path)

    with rasterio.open(thermal_path) as thermal_src, rasterio.open(output_path) as output_src:
        thermal = output_src.read(1)

        assert output_src.crs == thermal_src.crs
        assert output_src.transform == thermal_src.transform
        assert output_src.width == thermal_src.width
        assert output_src.height == thermal_src.height
        assert output_src.nodata == nodata
        assert output_src.count == 1
        assert output_src.dtypes == ("float32",)
        assert thermal[0, 0] == nodata


def test_thermal_generator_propagates_nodata_without_using_it_for_normalization(tmp_path: Path):
    thermal_path = tmp_path / "lwir11.tif"
    output_path = tmp_path / "thermal.tif"
    nodata = -9999

    _write_raster(
        thermal_path,
        np.array([[nodata, 10], [20, 30]], dtype=np.float32),
        nodata=nodata,
    )

    ThermalGenerator().generate(thermal_path, output_path)

    with rasterio.open(output_path) as src:
        thermal = src.read(1)

    np.testing.assert_allclose(
        thermal,
        np.array([[nodata, 0.0], [0.5, 1.0]], dtype=np.float32),
        rtol=1e-6,
    )


def test_thermal_generator_assigns_zero_when_valid_pixels_are_constant(tmp_path: Path):
    thermal_path = tmp_path / "lwir11.tif"
    output_path = tmp_path / "thermal.tif"
    nodata = -9999

    _write_raster(
        thermal_path,
        np.array([[nodata, 300], [300, 300]], dtype=np.float32),
        nodata=nodata,
    )

    ThermalGenerator().generate(thermal_path, output_path)

    with rasterio.open(output_path) as src:
        thermal = src.read(1)

    np.testing.assert_array_equal(
        thermal,
        np.array([[nodata, 0.0], [0.0, 0.0]], dtype=np.float32),
    )


def test_thermal_generator_creates_output_file_and_parent_directories(tmp_path: Path):
    thermal_path = tmp_path / "lwir11.tif"
    output_path = tmp_path / "nested" / "features" / "thermal.tif"

    _write_raster(thermal_path, np.array([[300]], dtype=np.float32), nodata=-9999)

    result = ThermalGenerator().generate(thermal_path, output_path)

    assert result == output_path
    assert output_path.exists()


def test_thermal_generator_wraps_failures_in_feature_engineering_error(tmp_path: Path):
    with pytest.raises(FeatureEngineeringError, match="Failed to generate thermal intensity raster"):
        ThermalGenerator().generate(
            tmp_path / "missing-lwir11.tif",
            tmp_path / "thermal.tif",
        )


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
