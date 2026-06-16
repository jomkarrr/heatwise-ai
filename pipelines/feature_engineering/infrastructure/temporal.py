"""Temporal raster aggregation infrastructure.

This module contains the production adapter that aggregates multiple feature
rasters into a single temporal summary. It reads source rasters with rasterio,
performs numeric work with numpy, and writes a GeoTIFF that keeps the
geospatial metadata of the first input raster.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError


class TemporalAggregator:
    """Aggregate compatible GeoTIFF rasters into temporal summary features.

    Supported statistics are pixel-wise ``mean`` and ``max``. Source rasters are
    read as float32 arrays, and each raster's nodata value is ignored for that
    raster while computing the requested statistic. Output pixels with no valid
    input values are written as the output nodata value when one is available,
    otherwise as NaN.
    """

    SUPPORTED_STATISTICS = {"mean", "max"}

    def aggregate(
        self,
        input_paths: list[Path],
        output_path: Path,
        statistic: str,
    ) -> Path:
        """Create a temporal aggregate GeoTIFF from compatible input rasters.

        Args:
            input_paths: Paths to input rasters that share the same grid.
            output_path: Destination path for the generated aggregate GeoTIFF.
            statistic: Aggregation statistic to compute. Supported values are
                ``"mean"`` and ``"max"``.

        Returns:
            The path to the generated temporal aggregate GeoTIFF.

        Raises:
            FeatureEngineeringError: If no input rasters are provided, the
                statistic is unsupported, rasters are spatially incompatible, or
                the output file cannot be written.
        """

        try:
            if not input_paths:
                raise FeatureEngineeringError("Temporal aggregation requires at least one input raster")
            if statistic not in self.SUPPORTED_STATISTICS:
                raise FeatureEngineeringError(f"Unsupported temporal aggregation statistic: {statistic}")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with ExitStack() as stack:
                sources = [stack.enter_context(rasterio.open(path)) for path in input_paths]
                reference_src = sources[0]

                for source, path in zip(sources[1:], input_paths[1:]):
                    self._validate_compatible_rasters(reference_src, source, input_paths[0], path)

                arrays = [source.read(1).astype(np.float32) for source in sources]
                valid_masks = [
                    self._build_valid_mask(array, source.nodata) for array, source in zip(arrays, sources)
                ]
                output_nodata = self._resolve_output_nodata(sources)

                aggregate = self._aggregate_arrays(arrays, valid_masks, statistic)
                valid_count = np.sum(valid_masks, axis=0)
                no_valid_data_mask = valid_count == 0

                if output_nodata is not None:
                    aggregate[no_valid_data_mask] = np.float32(output_nodata)
                else:
                    aggregate[no_valid_data_mask] = np.nan

                profile = reference_src.profile.copy()
                profile.update(
                    driver="GTiff",
                    dtype="float32",
                    count=1,
                    nodata=output_nodata,
                )

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(aggregate, 1)

            return output_path
        except FeatureEngineeringError:
            raise
        except Exception as exc:
            raise FeatureEngineeringError(f"Failed to aggregate temporal rasters: {exc}") from exc

    def _validate_compatible_rasters(
        self,
        reference_src: rasterio.io.DatasetReader,
        candidate_src: rasterio.io.DatasetReader,
        reference_path: Path,
        candidate_path: Path,
    ) -> None:
        """Ensure input rasters can be aggregated pixel-by-pixel."""

        if reference_src.width != candidate_src.width or reference_src.height != candidate_src.height:
            raise FeatureEngineeringError(
                f"Temporal rasters have different dimensions: {reference_path} and {candidate_path}"
            )
        if reference_src.transform != candidate_src.transform:
            raise FeatureEngineeringError(
                f"Temporal rasters have different transforms: {reference_path} and {candidate_path}"
            )
        if reference_src.crs != candidate_src.crs:
            raise FeatureEngineeringError(
                f"Temporal rasters have different CRS: {reference_path} and {candidate_path}"
            )

    def _build_valid_mask(
        self,
        raster: np.ndarray,
        raster_nodata: float | int | None,
    ) -> np.ndarray:
        """Return pixels that should participate in temporal aggregation."""

        valid_mask = np.ones(raster.shape, dtype=bool)
        if raster_nodata is not None:
            valid_mask &= raster != np.float32(raster_nodata)
        return valid_mask

    def _resolve_output_nodata(
        self,
        sources: list[rasterio.io.DatasetReader],
    ) -> float | int | None:
        """Return the first declared source nodata value for the output raster."""

        for source in sources:
            if source.nodata is not None:
                return source.nodata
        return None

    def _aggregate_arrays(
        self,
        arrays: list[np.ndarray],
        valid_masks: list[np.ndarray],
        statistic: str,
    ) -> np.ndarray:
        """Compute the requested statistic from arrays and per-raster masks."""

        stacked_arrays = np.stack(arrays)
        stacked_masks = np.stack(valid_masks)

        if statistic == "mean":
            valid_count = np.sum(stacked_masks, axis=0)
            valid_sum = np.sum(np.where(stacked_masks, stacked_arrays, 0.0), axis=0)
            mean = np.zeros(valid_count.shape, dtype=np.float32)
            np.divide(
                valid_sum,
                valid_count,
                out=mean,
                where=valid_count != 0,
            )
            return mean.astype(np.float32)

        valid_values = np.where(stacked_masks, stacked_arrays, -np.inf)
        maximum = np.max(valid_values, axis=0)
        return maximum.astype(np.float32)
