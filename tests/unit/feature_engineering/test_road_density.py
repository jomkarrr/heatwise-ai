from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError
from pipelines.feature_engineering.infrastructure.road_density import RoadDensityGenerator
from pipelines.feature_engineering.infrastructure.vector_rasterizer import VectorRasterizer


def test_road_density_generator_calculates_normalized_density(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    roads_path = tmp_path / "roads.geojson"
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "features" / "road-density.tif"
    road_mask = np.zeros((5, 5), dtype=np.uint8)
    road_mask[2, 2] = 1

    _write_raster(reference_path, np.zeros((5, 5), dtype=np.float32), nodata=-9999)
    _stub_rasterizer(monkeypatch, road_mask)
    monkeypatch.setattr(RoadDensityGenerator, "WINDOW_SIZE", 3)

    result = RoadDensityGenerator().generate(roads_path, reference_path, output_path)

    with rasterio.open(result) as src:
        density = src.read(1)

    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(density, expected, rtol=1e-6)


def test_road_density_generator_uses_vector_rasterizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    roads_path = tmp_path / "roads.geojson"
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "road-density.tif"
    calls = []

    def fake_rasterize(self, vector_path: Path, reference_raster_path: Path) -> np.ndarray:
        calls.append((vector_path, reference_raster_path))
        return np.zeros((2, 2), dtype=np.uint8)

    _write_raster(reference_path, np.zeros((2, 2), dtype=np.float32), nodata=-9999)
    monkeypatch.setattr(VectorRasterizer, "rasterize", fake_rasterize)

    RoadDensityGenerator().generate(roads_path, reference_path, output_path)

    assert calls == [(roads_path, reference_path)]


def test_road_density_generator_preserves_reference_metadata_and_nodata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    roads_path = tmp_path / "roads.geojson"
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "road-density.tif"
    transform = from_origin(72.0, 19.0, 30.0, 30.0)
    nodata = -9999

    _write_raster(
        reference_path,
        np.array([[nodata, 1], [1, 1]], dtype=np.float32),
        nodata=nodata,
        crs="EPSG:32643",
        transform=transform,
    )
    _stub_rasterizer(monkeypatch, np.ones((2, 2), dtype=np.uint8))

    RoadDensityGenerator().generate(roads_path, reference_path, output_path)

    with rasterio.open(reference_path) as reference_src, rasterio.open(output_path) as output_src:
        density = output_src.read(1)

        assert output_src.crs == reference_src.crs
        assert output_src.transform == reference_src.transform
        assert output_src.width == reference_src.width
        assert output_src.height == reference_src.height
        assert output_src.nodata == nodata
        assert output_src.count == 1
        assert output_src.dtypes == ("float32",)
        assert density[0, 0] == nodata


def test_road_density_generator_creates_output_file_and_parent_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    roads_path = tmp_path / "roads.geojson"
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "nested" / "features" / "road-density.tif"

    _write_raster(reference_path, np.zeros((1, 1), dtype=np.float32), nodata=-9999)
    _stub_rasterizer(monkeypatch, np.ones((1, 1), dtype=np.uint8))

    result = RoadDensityGenerator().generate(roads_path, reference_path, output_path)

    assert result == output_path
    assert output_path.exists()


def test_road_density_generator_keeps_zero_density_when_no_roads_are_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    roads_path = tmp_path / "roads.geojson"
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "road-density.tif"

    _write_raster(reference_path, np.zeros((3, 3), dtype=np.float32), nodata=-9999)
    _stub_rasterizer(monkeypatch, np.zeros((3, 3), dtype=np.uint8))

    RoadDensityGenerator().generate(roads_path, reference_path, output_path)

    with rasterio.open(output_path) as src:
        density = src.read(1)

    np.testing.assert_array_equal(density, np.zeros((3, 3), dtype=np.float32))


def test_road_density_generator_wraps_failures_in_feature_engineering_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    roads_path = tmp_path / "roads.geojson"
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "road-density.tif"

    def fail_rasterize(self, vector_path: Path, reference_raster_path: Path) -> np.ndarray:
        raise RuntimeError("rasterizer failed")

    monkeypatch.setattr(VectorRasterizer, "rasterize", fail_rasterize)

    with pytest.raises(FeatureEngineeringError, match="Failed to generate road density raster"):
        RoadDensityGenerator().generate(roads_path, reference_path, output_path)


def _stub_rasterizer(
    monkeypatch: pytest.MonkeyPatch,
    road_mask: np.ndarray,
) -> None:
    def fake_rasterize(self, vector_path: Path, reference_raster_path: Path) -> np.ndarray:
        return road_mask

    monkeypatch.setattr(VectorRasterizer, "rasterize", fake_rasterize)


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


def test_road_density_generator_handles_reference_without_nodata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    roads_path = tmp_path / "roads.geojson"
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "road-density.tif"

    _write_raster(
        reference_path,
        np.zeros((2, 2), dtype=np.float32),
        nodata=None,
    )

    _stub_rasterizer(
        monkeypatch,
        np.ones((2, 2), dtype=np.uint8),
    )

    RoadDensityGenerator().generate(
        roads_path,
        reference_path,
        output_path,
    )

    with rasterio.open(output_path) as src:
        density = src.read(1)

    assert np.isfinite(density).all()
