from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError
from pipelines.feature_engineering.infrastructure.vector_rasterizer import VectorRasterizer


def test_vector_rasterizer_rasterizes_polygon_features(tmp_path: Path):
    vector_path = tmp_path / "features.geojson"
    reference_path = tmp_path / "reference.tif"

    _write_raster(reference_path, np.zeros((2, 2), dtype=np.float32))
    _write_vector(vector_path, [box(0, 1, 1, 2)])

    mask = VectorRasterizer().rasterize(vector_path, reference_path)

    np.testing.assert_array_equal(
        mask,
        np.array([[1, 0], [0, 0]], dtype=np.uint8),
    )


def test_vector_rasterizer_keeps_only_polygon_geometries(tmp_path: Path):
    vector_path = tmp_path / "features.geojson"
    reference_path = tmp_path / "reference.tif"

    _write_raster(reference_path, np.zeros((2, 2), dtype=np.float32))
    _write_vector(
        vector_path,
        [
            box(0, 1, 1, 2),
            Point(1.5, 0.5),
        ],
    )

    mask = VectorRasterizer().rasterize(vector_path, reference_path)

    np.testing.assert_array_equal(
        mask,
        np.array([[1, 0], [0, 0]], dtype=np.uint8),
    )


def test_vector_rasterizer_supports_multipolygon_features(tmp_path: Path):
    vector_path = tmp_path / "features.geojson"
    reference_path = tmp_path / "reference.tif"
    multipolygon = MultiPolygon(
        [
            box(0, 1, 1, 2),
            box(1, 0, 2, 1),
        ]
    )

    _write_raster(reference_path, np.zeros((2, 2), dtype=np.float32))
    _write_vector(vector_path, [multipolygon])

    mask = VectorRasterizer().rasterize(vector_path, reference_path)

    np.testing.assert_array_equal(
        mask,
        np.array([[1, 0], [0, 1]], dtype=np.uint8),
    )


def test_vector_rasterizer_ignores_point_geometries(tmp_path: Path):
    vector_path = tmp_path / "features.geojson"
    reference_path = tmp_path / "reference.tif"

    _write_raster(reference_path, np.zeros((2, 2), dtype=np.float32))
    _write_vector(vector_path, [Point(0.5, 1.5), Point(1.5, 0.5)])

    mask = VectorRasterizer().rasterize(vector_path, reference_path)

    np.testing.assert_array_equal(mask, np.zeros((2, 2), dtype=np.uint8))


def test_vector_rasterizer_reprojects_vectors_to_reference_crs(tmp_path: Path):
    vector_path = tmp_path / "features.geojson"
    reference_path = tmp_path / "reference.tif"
    one_meter_in_degrees = 0.000008983152841195214

    _write_raster(
        reference_path,
        np.zeros((1, 1), dtype=np.float32),
        crs="EPSG:3857",
        transform=from_origin(0.0, 1.0, 1.0, 1.0),
    )
    _write_vector(
        vector_path,
        [
            Polygon(
                [
                    (0.0, 0.0),
                    (one_meter_in_degrees, 0.0),
                    (one_meter_in_degrees, one_meter_in_degrees),
                    (0.0, one_meter_in_degrees),
                    (0.0, 0.0),
                ]
            )
        ],
        crs="EPSG:4326",
    )

    mask = VectorRasterizer().rasterize(vector_path, reference_path)

    np.testing.assert_array_equal(mask, np.array([[1]], dtype=np.uint8))


def test_vector_rasterizer_output_shape_matches_reference_raster(tmp_path: Path):
    vector_path = tmp_path / "features.geojson"
    reference_path = tmp_path / "reference.tif"

    _write_raster(reference_path, np.zeros((3, 4), dtype=np.float32))
    _write_vector(vector_path, [box(0, 2, 1, 3)])

    mask = VectorRasterizer().rasterize(vector_path, reference_path)

    assert mask.shape == (3, 4)


def test_vector_rasterizer_returns_uint8_array(tmp_path: Path):
    vector_path = tmp_path / "features.geojson"
    reference_path = tmp_path / "reference.tif"

    _write_raster(reference_path, np.zeros((2, 2), dtype=np.float32))
    _write_vector(vector_path, [box(0, 1, 1, 2)])

    mask = VectorRasterizer().rasterize(vector_path, reference_path)

    assert mask.dtype == np.uint8


def test_vector_rasterizer_wraps_failures_in_feature_engineering_error(tmp_path: Path):
    with pytest.raises(FeatureEngineeringError, match="Failed to rasterize vector features"):
        VectorRasterizer().rasterize(
            tmp_path / "missing.geojson",
            tmp_path / "missing-reference.tif",
        )


def _write_vector(
    path: Path,
    geometries: list,
    crs: str = "EPSG:4326",
) -> None:
    vectors = gpd.GeoDataFrame({"id": range(len(geometries))}, geometry=geometries, crs=crs)
    vectors.to_file(path, driver="GeoJSON")


def _write_raster(
    path: Path,
    data: np.ndarray,
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
        nodata=-9999,
    ) as dst:
        dst.write(data, 1)



from shapely.geometry import LineString

def test_vector_rasterizer_rasterizes_linestring_features(tmp_path: Path):
    vector_path = tmp_path / "roads.geojson"
    reference_path = tmp_path / "reference.tif"

    _write_raster(reference_path, np.zeros((5, 5), dtype=np.float32))

    _write_vector(
        vector_path,
        [
            LineString(
                [
                    (0, 5),
                    (5, 0),
                ]
            )
        ],
    )

    mask = VectorRasterizer().rasterize(vector_path, reference_path)

    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) > 0