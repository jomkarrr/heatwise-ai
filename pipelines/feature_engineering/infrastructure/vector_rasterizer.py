"""Vector-to-raster feature generation infrastructure.

This module contains the production adapter that burns polygon vector features
onto the grid of a reference raster. It reads GeoJSON data with geopandas,
reprojects geometries to the raster CRS, and returns a uint8 NumPy mask without
writing any output files.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError


class VectorRasterizer:
    """Rasterize polygon vector features onto a reference raster grid.

    The rasterized output uses the reference raster's CRS, transform, width,
    and height. Polygon, MultiPolygon, and LineString geometries are burned into a uint8
    array with ``1`` for feature presence and ``0`` for background. Non-polygon
    geometries, null geometries, and empty geometries are ignored.
    """

    def rasterize(
        self,
        vector_path: Path,
        reference_raster_path: Path,
    ) -> np.ndarray:
        """Return a uint8 mask of vector polygons aligned to a reference raster.

        Args:
            vector_path: Path to a GeoJSON vector dataset.
            reference_raster_path: Path to the raster whose grid should define
                the returned array shape, transform, and CRS.

        Returns:
            A uint8 NumPy array where ``1`` indicates a polygon feature and
            ``0`` indicates background.

        Raises:
            FeatureEngineeringError: If the vector or reference raster cannot be
                read, the vector data cannot be reprojected, or rasterization
                fails.
        """

        try:
            vectors = gpd.read_file(vector_path)

            with rasterio.open(reference_raster_path) as reference:
                vectors = vectors.to_crs(reference.crs)
                geometries = self._supported_geometries(vectors)

                if not geometries:
                    return np.zeros((reference.height, reference.width), dtype=np.uint8)

                return rasterize(
                    shapes=((geometry, 1) for geometry in geometries),
                    out_shape=(reference.height, reference.width),
                    transform=reference.transform,
                    fill=0,
                    dtype="uint8",
                )
        except FeatureEngineeringError:
            raise
        except Exception as exc:
            raise FeatureEngineeringError(f"Failed to rasterize vector features: {exc}") from exc

    def _supported_geometries(
       self,
       vectors: gpd.GeoDataFrame,
    ) -> list:
       """Return geometries supported for rasterization."""

       return [
        geometry
        for geometry in vectors.geometry
        if geometry is not None
        and not geometry.is_empty
        and geometry.geom_type in {
            "Polygon",
            "MultiPolygon",
            "LineString",
        }
    ]