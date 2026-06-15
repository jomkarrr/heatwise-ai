"""NDVI raster generation infrastructure.

This module contains the production adapter that derives the Normalized
Difference Vegetation Index (NDVI) from Landsat red and near-infrared bands.
It reads source rasters with rasterio, performs numeric work with numpy, and
writes a GeoTIFF that keeps the geospatial metadata of the red band.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError


class NdviGenerator:
    """Generate NDVI GeoTIFF rasters from Landsat red and NIR bands.

    NDVI is computed as ``(NIR - RED) / (NIR + RED)`` using float32 arrays.
    Pixels marked as nodata in either input band are carried through to the
    output nodata value. The output nodata value is taken from the red band
    when available, otherwise from the NIR band. Pixels with a zero denominator
    are also treated as nodata so divide-by-zero never produces NaN or infinite
    values.
    """

    def generate(
        self,
        red_path: Path,
        nir_path: Path,
        output_path: Path,
    ) -> Path:
        """Create an NDVI GeoTIFF from Landsat red and NIR08 band files.

        Args:
            red_path: Path to the Landsat red band raster.
            nir_path: Path to the Landsat NIR08 band raster.
            output_path: Destination path for the generated NDVI GeoTIFF.

        Returns:
            The path to the generated NDVI GeoTIFF.

        Raises:
            FeatureEngineeringError: If the rasters cannot be read, are not
                spatially compatible, or the output file cannot be written.
        """

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:
                self._validate_compatible_rasters(red_src, nir_src, red_path, nir_path)

                red = red_src.read(1).astype(np.float32)
                nir = nir_src.read(1).astype(np.float32)
                output_nodata = red_src.nodata if red_src.nodata is not None else nir_src.nodata
                valid_mask = self._build_valid_mask(red, nir, red_src.nodata, nir_src.nodata)

                denominator = nir + red
                ndvi = np.zeros(red.shape, dtype=np.float32)
                compute_mask = valid_mask & (denominator != 0)
                nodata_mask = ~compute_mask

                np.divide(
                    nir - red,
                    denominator,
                    out=ndvi,
                    where=compute_mask,
                )

                np.clip(ndvi, -1.0, 1.0, out=ndvi)
                if output_nodata is not None:
                    ndvi[nodata_mask] = np.float32(output_nodata)
                else:
                    ndvi[nodata_mask] = np.nan

                profile = red_src.profile.copy()
                profile.update(
                    driver="GTiff",
                    dtype="float32",
                    count=1,
                    nodata=output_nodata,
                )

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(ndvi, 1)

            return output_path
        except FeatureEngineeringError:
            raise
        except Exception as exc:
            raise FeatureEngineeringError(f"Failed to generate NDVI raster: {exc}") from exc

    def _validate_compatible_rasters(
        self,
        red_src: rasterio.io.DatasetReader,
        nir_src: rasterio.io.DatasetReader,
        red_path: Path,
        nir_path: Path,
    ) -> None:
        """Ensure input rasters can be combined pixel-by-pixel."""

        if red_src.width != nir_src.width or red_src.height != nir_src.height:
            raise FeatureEngineeringError(
                f"Red and NIR rasters have different dimensions: {red_path} and {nir_path}"
            )
        if red_src.transform != nir_src.transform:
            raise FeatureEngineeringError(
                f"Red and NIR rasters have different transforms: {red_path} and {nir_path}"
            )
        if red_src.crs != nir_src.crs:
            raise FeatureEngineeringError(f"Red and NIR rasters have different CRS: {red_path} and {nir_path}")

    def _build_valid_mask(
        self,
        red: np.ndarray,
        nir: np.ndarray,
        red_nodata: float | int | None,
        nir_nodata: float | int | None,
    ) -> np.ndarray:
        """Return pixels that are valid in both source rasters."""

        valid_mask = np.ones(red.shape, dtype=bool)
        if red_nodata is not None:
            valid_mask &= red != np.float32(red_nodata)
        if nir_nodata is not None:
            valid_mask &= nir != np.float32(nir_nodata)
        return valid_mask
