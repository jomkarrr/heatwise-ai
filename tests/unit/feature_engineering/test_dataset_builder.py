from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError
from pipelines.feature_engineering.infrastructure.dataset_builder import DatasetBuilder


def test_dataset_builder_reads_rasters_and_creates_dataframe(tmp_path: Path):
    paths = _write_feature_rasters(
        tmp_path,
        {
            "ndvi": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            "ndbi": np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
            "thermal": np.array([[0.9, 1.0], [0.8, 0.7]], dtype=np.float32),
            "building_density": np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32),
            "road_density": np.array([[0.4, 0.3], [0.2, 0.1]], dtype=np.float32),
        },
    )

    dataframe = DatasetBuilder().build(
        ndvi_path=paths["ndvi"],
        ndbi_path=paths["ndbi"],
        thermal_path=paths["thermal"],
        building_density_path=paths["building_density"],
        road_density_path=paths["road_density"],
    )

    assert isinstance(dataframe, pd.DataFrame)
    assert list(dataframe.columns) == [
        "ndvi",
        "ndbi",
        "thermal",
        "building_density",
        "road_density",
    ]
    assert len(dataframe) == 4
    np.testing.assert_allclose(dataframe["ndvi"].to_numpy(), np.array([0.1, 0.2, 0.3, 0.4]), rtol=1e-6)
    np.testing.assert_allclose(dataframe["ndbi"].to_numpy(), np.array([0.5, 0.6, 0.7, 0.8]), rtol=1e-6)
    np.testing.assert_allclose(dataframe["thermal"].to_numpy(), np.array([0.9, 1.0, 0.8, 0.7]), rtol=1e-6)
    np.testing.assert_allclose(
        dataframe["building_density"].to_numpy(),
        np.array([0.0, 0.1, 0.2, 0.3]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(dataframe["road_density"].to_numpy(), np.array([0.4, 0.3, 0.2, 0.1]), rtol=1e-6)


def test_dataset_builder_rejects_rasters_with_different_crs(tmp_path: Path):
    paths = _write_default_feature_rasters(tmp_path)
    _write_raster(paths["road_density"], np.ones((2, 2), dtype=np.float32), crs="EPSG:32643")

    with pytest.raises(FeatureEngineeringError, match="different CRS"):
        _build_dataset(paths)


def test_dataset_builder_rejects_rasters_with_different_transform(tmp_path: Path):
    paths = _write_default_feature_rasters(tmp_path)
    _write_raster(
        paths["road_density"],
        np.ones((2, 2), dtype=np.float32),
        transform=from_origin(72.0, 19.0, 30.0, 30.0),
    )

    with pytest.raises(FeatureEngineeringError, match="different transforms"):
        _build_dataset(paths)


def test_dataset_builder_rejects_rasters_with_different_dimensions(tmp_path: Path):
    paths = _write_default_feature_rasters(tmp_path)
    _write_raster(
    paths["road_density"],
    np.ones((3, 2), dtype=np.float32),
    transform=from_origin(0.0, 2.0, 1.0, 1.0),
)

    with pytest.raises(FeatureEngineeringError, match="different dimensions"):
        _build_dataset(paths)


def test_dataset_builder_wraps_unexpected_errors_in_feature_engineering_error(tmp_path: Path):
    paths = _write_default_feature_rasters(tmp_path)
    paths["road_density"] = tmp_path / "missing-road-density.tif"

    with pytest.raises(FeatureEngineeringError, match="Failed to build feature dataset"):
        _build_dataset(paths)


def _build_dataset(
    paths: dict[str, Path],
) -> pd.DataFrame:
    return DatasetBuilder().build(
        ndvi_path=paths["ndvi"],
        ndbi_path=paths["ndbi"],
        thermal_path=paths["thermal"],
        building_density_path=paths["building_density"],
        road_density_path=paths["road_density"],
    )


def _write_default_feature_rasters(
    tmp_path: Path,
) -> dict[str, Path]:
    return _write_feature_rasters(
        tmp_path,
        {
            "ndvi": np.ones((2, 2), dtype=np.float32),
            "ndbi": np.ones((2, 2), dtype=np.float32),
            "thermal": np.ones((2, 2), dtype=np.float32),
            "building_density": np.ones((2, 2), dtype=np.float32),
            "road_density": np.ones((2, 2), dtype=np.float32),
        },
    )


def _write_feature_rasters(
    tmp_path: Path,
    feature_data: dict[str, np.ndarray],
) -> dict[str, Path]:
    paths = {
        feature_name: tmp_path / f"{feature_name}.tif"
        for feature_name in feature_data
    }
    for feature_name, data in feature_data.items():
        _write_raster(paths[feature_name], data)
    return paths


def _write_raster(
    path: Path,
    data: np.ndarray,
    nodata: float | None = -9999,
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
