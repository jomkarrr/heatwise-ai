"""Thermal intensity raster generation infrastructure.

This module contains the production adapter that derives a normalized thermal
intensity feature from a Landsat LWIR11 band. It reads source rasters with
rasterio, performs numeric work with numpy, and writes a GeoTIFF that keeps the
geospatial metadata of the LWIR11 band.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError


class ThermalGenerator:
    """Generate normalized thermal intensity GeoTIFF rasters from LWIR11 data.

    Thermal intensity is computed with min-max normalization over valid LWIR11
    pixels only: ``(value - min_valid) / (max_valid - min_valid)``. Pixels
    marked as nodata in the input raster are carried through to the output
    nodata value and are ignored while calculating the valid minimum and
    maximum. When all valid pixels have the same value, every valid output pixel
    is assigned ``0`` so the result stays finite and deterministic.
    """

    def generate(
        self,
        thermal_path: Path,
        output_path: Path,
    ) -> Path:
        """Create a normalized thermal intensity GeoTIFF from an LWIR11 raster.

        Args:
            thermal_path: Path to the Landsat LWIR11 band raster.
            output_path: Destination path for the generated thermal intensity
                GeoTIFF.

        Returns:
            The path to the generated thermal intensity GeoTIFF.

        Raises:
            FeatureEngineeringError: If the raster cannot be read, valid pixels
                cannot be normalized, or the output file cannot be written.
        """

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(thermal_path) as thermal_src:
                thermal = thermal_src.read(1).astype(np.float32)
                output_nodata = thermal_src.nodata
                valid_mask = self._build_valid_mask(thermal, output_nodata)

                thermal_intensity = np.zeros(thermal.shape, dtype=np.float32)
                if valid_mask.any():
                    valid_values = thermal[valid_mask]
                    min_valid = np.min(valid_values)
                    max_valid = np.max(valid_values)

                    if max_valid != min_valid:
                        np.divide(
                            thermal - min_valid,
                            max_valid - min_valid,
                            out=thermal_intensity,
                            where=valid_mask,
                        )
                        np.clip(thermal_intensity, 0.0, 1.0, out=thermal_intensity)

                if output_nodata is not None:
                    thermal_intensity[~valid_mask] = np.float32(output_nodata)
                else:
                    thermal_intensity[~valid_mask] = np.nan

                profile = thermal_src.profile.copy()
                profile.update(
                    driver="GTiff",
                    dtype="float32",
                    count=1,
                    nodata=output_nodata,
                )

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(thermal_intensity, 1)

            return output_path
        except FeatureEngineeringError:
            raise
        except Exception as exc:
            raise FeatureEngineeringError(f"Failed to generate thermal intensity raster: {exc}") from exc

    def _build_valid_mask(
        self,
        thermal: np.ndarray,
        thermal_nodata: float | int | None,
    ) -> np.ndarray:
        """Return pixels that should participate in thermal normalization."""

        valid_mask = np.ones(thermal.shape, dtype=bool)
        if thermal_nodata is not None:
            valid_mask &= thermal != np.float32(thermal_nodata)
        return valid_mask
