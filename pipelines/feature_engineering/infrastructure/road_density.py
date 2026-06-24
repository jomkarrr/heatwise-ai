"""Road density raster generation infrastructure.

This module contains the production adapter that derives a normalized road
density feature from road network vectors. It delegates vector rasterization to
``VectorRasterizer``, computes a moving-window neighborhood mean with
numpy/scipy tooling, and writes a GeoTIFF that keeps the geospatial metadata of
the reference raster.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError
from pipelines.feature_engineering.infrastructure.vector_rasterizer import VectorRasterizer


class RoadDensityGenerator:
    """Generate normalized road density GeoTIFF rasters from road vectors.

    Road features are rasterized to the reference raster grid with ``1`` for
    road pixels and ``0`` for background. Density is computed as a moving-window
    neighborhood mean over that binary mask, then normalized to the range ``0``
    to ``1`` when at least one road pixel is present. Pixels marked as nodata in
    the reference raster are carried through to the output nodata value.
    """

    WINDOW_SIZE = 11

    def generate(
        self,
        roads_path: Path,
        reference_raster_path: Path,
        output_path: Path,
    ) -> Path:
        """Create a normalized road density GeoTIFF.

        Args:
            roads_path: Path to a roads GeoJSON file.
            reference_raster_path: Path to the raster whose grid and metadata
                should define the output GeoTIFF.
            output_path: Destination path for the generated road density
                GeoTIFF.

        Returns:
            The path to the generated road density GeoTIFF.

        Raises:
            FeatureEngineeringError: If the roads cannot be rasterized, the
                reference raster cannot be read, or the output file cannot be
                written.
        """

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            road_mask = VectorRasterizer().rasterize(roads_path, reference_raster_path)

            with rasterio.open(reference_raster_path) as reference:
                nodata = reference.nodata
                reference_data = reference.read(1).astype(np.float32)
                valid_mask = self._build_valid_mask(reference_data, nodata)

                density = uniform_filter(
                    road_mask.astype(np.float32),
                    size=self.WINDOW_SIZE,
                    mode="constant",
                )

                max_density = density.max()
                if max_density > 0:
                    density /= max_density

                density = density.astype(np.float32)

                if nodata is not None:
                    density[~valid_mask] = np.float32(nodata)
                else:
                    density[~valid_mask] = np.nan

                profile = reference.profile.copy()
                profile.update(
                    driver="GTiff",
                    dtype="float32",
                    count=1,
                    nodata=nodata,
                )

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(density.astype(np.float32), 1)

            return output_path
        except FeatureEngineeringError:
            raise
        except Exception as exc:
            raise FeatureEngineeringError(f"Failed to generate road density raster: {exc}") from exc

    def _build_valid_mask(
        self,
        reference: np.ndarray,
        reference_nodata: float | int | None,
    ) -> np.ndarray:
        """Return reference pixels where road density should be retained."""

        valid_mask = np.ones(reference.shape, dtype=bool)
        if reference_nodata is not None:
            valid_mask &= reference != np.float32(reference_nodata)
        return valid_mask
