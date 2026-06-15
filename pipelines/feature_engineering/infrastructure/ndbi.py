"""NDBI raster generation infrastructure.

This module contains the production adapter that derives the Normalized
Difference Built-up Index (NDBI) from Landsat shortwave-infrared and
near-infrared bands. It reads source rasters with rasterio, performs numeric
work with numpy, and writes a GeoTIFF that keeps the geospatial metadata of the
SWIR band.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError


class NdbiGenerator:
    """Generate NDBI GeoTIFF rasters from Landsat SWIR and NIR bands.

    NDBI is computed as ``(SWIR - NIR) / (SWIR + NIR)`` using float32 arrays.
    Pixels marked as nodata in either input band are carried through to the
    output nodata value. The output nodata value is taken from the SWIR band
    when available, otherwise from the NIR band. Pixels with a zero denominator
    are also treated as nodata so divide-by-zero never produces NaN or infinite
    values.
    """

    def generate(
        self,
        swir_path: Path,
        nir_path: Path,
        output_path: Path,
    ) -> Path:
        """Create an NDBI GeoTIFF from Landsat SWIR and NIR band files.

        Args:
            swir_path: Path to the Landsat SWIR band raster.
            nir_path: Path to the Landsat NIR band raster.
            output_path: Destination path for the generated NDBI GeoTIFF.

        Returns:
            The path to the generated NDBI GeoTIFF.

        Raises:
            FeatureEngineeringError: If the rasters cannot be read, are not
                spatially compatible, or the output file cannot be written.
        """

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(swir_path) as swir_src, rasterio.open(nir_path) as nir_src:
                self._validate_compatible_rasters(swir_src, nir_src, swir_path, nir_path)

                swir = swir_src.read(1).astype(np.float32)
                nir = nir_src.read(1).astype(np.float32)
                output_nodata = swir_src.nodata if swir_src.nodata is not None else nir_src.nodata
                valid_mask = self._build_valid_mask(swir, nir, swir_src.nodata, nir_src.nodata)

                denominator = swir + nir
                ndbi = np.zeros(swir.shape, dtype=np.float32)
                compute_mask = valid_mask & (denominator != 0)
                nodata_mask = ~compute_mask

                np.divide(
                    swir - nir,
                    denominator,
                    out=ndbi,
                    where=compute_mask,
                )

                np.clip(ndbi, -1.0, 1.0, out=ndbi)
                if output_nodata is not None:
                    ndbi[nodata_mask] = np.float32(output_nodata)
                else:
                    ndbi[nodata_mask] = np.nan

                profile = swir_src.profile.copy()
                profile.update(
                    driver="GTiff",
                    dtype="float32",
                    count=1,
                    nodata=output_nodata,
                )

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(ndbi, 1)

            return output_path
        except FeatureEngineeringError:
            raise
        except Exception as exc:
            raise FeatureEngineeringError(f"Failed to generate NDBI raster: {exc}") from exc

    def _validate_compatible_rasters(
        self,
        swir_src: rasterio.io.DatasetReader,
        nir_src: rasterio.io.DatasetReader,
        swir_path: Path,
        nir_path: Path,
    ) -> None:
        """Ensure input rasters can be combined pixel-by-pixel."""

        if swir_src.width != nir_src.width or swir_src.height != nir_src.height:
            raise FeatureEngineeringError(
                f"SWIR and NIR rasters have different dimensions: {swir_path} and {nir_path}"
            )
        if swir_src.transform != nir_src.transform:
            raise FeatureEngineeringError(
                f"SWIR and NIR rasters have different transforms: {swir_path} and {nir_path}"
            )
        if swir_src.crs != nir_src.crs:
            raise FeatureEngineeringError(f"SWIR and NIR rasters have different CRS: {swir_path} and {nir_path}")

    def _build_valid_mask(
        self,
        swir: np.ndarray,
        nir: np.ndarray,
        swir_nodata: float | int | None,
        nir_nodata: float | int | None,
    ) -> np.ndarray:
        """Return pixels that are valid in both source rasters."""

        valid_mask = np.ones(swir.shape, dtype=bool)
        if swir_nodata is not None:
            valid_mask &= swir != np.float32(swir_nodata)
        if nir_nodata is not None:
            valid_mask &= nir != np.float32(nir_nodata)
        return valid_mask
