"""Raster alignment infrastructure.

This module contains the production adapter that aligns one raster to the grid
of another raster. It reads source and reference rasters with rasterio,
reprojects and resamples the source with rasterio's warp utilities, and writes
a GeoTIFF that matches the reference raster's spatial grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError


class RasterAligner:
    """Align a source raster to a reference raster's geospatial grid.

    The source raster's first band is read as float32 and reprojected using
    bilinear resampling so the output matches the reference raster's CRS,
    transform, width, and height. Source nodata values are passed through the
    reprojection operation and written as the output nodata value.
    """

    def align(
        self,
        source_path: Path,
        reference_path: Path,
        output_path: Path,
    ) -> Path:
        """Create an aligned GeoTIFF from a source and reference raster.

        Args:
            source_path: Path to the raster that should be reprojected and
                resampled.
            reference_path: Path to the raster whose CRS, transform, width, and
                height should define the output grid.
            output_path: Destination path for the aligned GeoTIFF.

        Returns:
            The path to the generated aligned GeoTIFF.

        Raises:
            FeatureEngineeringError: If either raster cannot be read, the source
                cannot be reprojected to the reference grid, or the output file
                cannot be written.
        """

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(source_path) as source_src, rasterio.open(reference_path) as reference_src:
                source = source_src.read(1).astype(np.float32)
                output_nodata = source_src.nodata
                destination = self._create_destination(reference_src.height, reference_src.width, output_nodata)

                reproject(
                    source=source,
                    destination=destination,
                    src_transform=source_src.transform,
                    src_crs=source_src.crs,
                    src_nodata=output_nodata,
                    dst_transform=reference_src.transform,
                    dst_crs=reference_src.crs,
                    dst_nodata=output_nodata,
                    resampling=Resampling.bilinear,
                )

                profile = reference_src.profile.copy()
                profile.update(
                    driver="GTiff",
                    dtype="float32",
                    count=1,
                    nodata=output_nodata,
                )

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(destination, 1)

            return output_path
        except FeatureEngineeringError:
            raise
        except Exception as exc:
            raise FeatureEngineeringError(f"Failed to align raster: {exc}") from exc

    def _create_destination(
        self,
        height: int,
        width: int,
        output_nodata: float | int | None,
    ) -> np.ndarray:
        """Return a float32 destination array initialized for reprojection."""

        destination = np.zeros((height, width), dtype=np.float32)
        if output_nodata is not None:
            destination.fill(np.float32(output_nodata))
        return destination
