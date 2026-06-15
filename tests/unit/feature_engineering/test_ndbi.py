from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError
from pipelines.feature_engineering.infrastructure.ndbi import NdbiGenerator


def test_ndbi_generator_calculates_expected_values(tmp_path: Path):
    swir_path = tmp_path / "swir.tif"
    nir_path = tmp_path / "nir08.tif"
    output_path = tmp_path / "features" / "ndbi.tif"

    _write_raster(
        swir_path,
        np.array([[3, 2], [9, 12]], dtype=np.float32),
        nodata=-9999,
    )
    _write_raster(
        nir_path,
        np.array([[1, 2], [3, 4]], dtype=np.float32),
        nodata=-9999,
    )

    result = NdbiGenerator().generate(swir_path, nir_path, output_path)

    with rasterio.open(result) as src:
        ndbi = src.read(1)

    expected = np.array([[0.5, 0.0], [0.5, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(ndbi, expected, rtol=1e-6)


def test_ndbi_generator_clips_values_to_valid_range(tmp_path: Path):
    swir_path = tmp_path / "swir.tif"
    nir_path = tmp_path / "nir08.tif"
    output_path = tmp_path / "ndbi.tif"

    _write_raster(swir_path, np.array([[3, -1]], dtype=np.float32), nodata=-9999)
    _write_raster(nir_path, np.array([[-1, 3]], dtype=np.float32), nodata=-9999)

    NdbiGenerator().generate(swir_path, nir_path, output_path)

    with rasterio.open(output_path) as src:
        ndbi = src.read(1)

    assert ndbi.min() >= -1
    assert ndbi.max() <= 1


def test_ndbi_generator_handles_divide_by_zero(tmp_path: Path):
    swir_path = tmp_path / "swir.tif"
    nir_path = tmp_path / "nir08.tif"
    output_path = tmp_path / "ndbi.tif"

    _write_raster(swir_path, np.array([[0, 1]], dtype=np.float32), nodata=-9999)
    _write_raster(nir_path, np.array([[0, -1]], dtype=np.float32), nodata=-9999)

    NdbiGenerator().generate(swir_path, nir_path, output_path)

    with rasterio.open(output_path) as src:
        ndbi = src.read(1)

    assert np.isfinite(ndbi).all()
    np.testing.assert_array_equal(
        ndbi,
        np.array([[-9999, -9999]], dtype=np.float32),
    )


def test_ndbi_generator_preserves_swir_band_metadata_and_nodata(tmp_path: Path):
    swir_path = tmp_path / "swir.tif"
    nir_path = tmp_path / "nir08.tif"
    output_path = tmp_path / "ndbi.tif"
    transform = from_origin(72.0, 19.0, 30.0, 30.0)
    nodata = -9999

    _write_raster(
        swir_path,
        np.array([[nodata, 4], [6, 8]], dtype=np.float32),
        nodata=nodata,
        crs="EPSG:32643",
        transform=transform,
    )
    _write_raster(
        nir_path,
        np.array([[10, 2], [nodata, 4]], dtype=np.float32),
        nodata=nodata,
        crs="EPSG:32643",
        transform=transform,
    )

    NdbiGenerator().generate(swir_path, nir_path, output_path)

    with rasterio.open(swir_path) as swir_src, rasterio.open(output_path) as ndbi_src:
        ndbi = ndbi_src.read(1)

        assert ndbi_src.crs == swir_src.crs
        assert ndbi_src.transform == swir_src.transform
        assert ndbi_src.width == swir_src.width
        assert ndbi_src.height == swir_src.height
        assert ndbi_src.nodata == nodata
        assert ndbi_src.count == 1
        assert ndbi_src.dtypes == ("float32",)
        assert ndbi[0, 0] == nodata
        assert ndbi[1, 0] == nodata


def test_ndbi_generator_uses_nir_nodata_when_swir_nodata_is_missing(tmp_path: Path):
    swir_path = tmp_path / "swir.tif"
    nir_path = tmp_path / "nir08.tif"
    output_path = tmp_path / "ndbi.tif"
    nir_nodata = -32768

    _write_raster(
        swir_path,
        np.array([[2, 0]], dtype=np.float32),
        nodata=None,
    )
    _write_raster(
        nir_path,
        np.array([[nir_nodata, 0]], dtype=np.float32),
        nodata=nir_nodata,
    )

    NdbiGenerator().generate(swir_path, nir_path, output_path)

    with rasterio.open(output_path) as src:
        ndbi = src.read(1)

        assert src.nodata == nir_nodata
        np.testing.assert_array_equal(
            ndbi,
            np.array([[nir_nodata, nir_nodata]], dtype=np.float32),
        )


def test_ndbi_generator_creates_output_file_and_parent_directories(tmp_path: Path):
    swir_path = tmp_path / "swir.tif"
    nir_path = tmp_path / "nir08.tif"
    output_path = tmp_path / "nested" / "features" / "ndbi.tif"

    _write_raster(swir_path, np.array([[3]], dtype=np.float32), nodata=-9999)
    _write_raster(nir_path, np.array([[1]], dtype=np.float32), nodata=-9999)

    result = NdbiGenerator().generate(swir_path, nir_path, output_path)

    assert result == output_path
    assert output_path.exists()


def test_ndbi_generator_wraps_failures_in_feature_engineering_error(tmp_path: Path):
    with pytest.raises(FeatureEngineeringError, match="Failed to generate NDBI raster"):
        NdbiGenerator().generate(
            tmp_path / "missing-swir.tif",
            tmp_path / "missing-nir.tif",
            tmp_path / "ndbi.tif",
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
