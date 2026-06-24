"""Feature dataset construction infrastructure.

This module contains the production adapter that combines aligned feature
rasters into a tabular pandas dataset. It reads raster bands with rasterio,
validates that every input shares the same spatial grid, flattens each raster,
and returns a DataFrame ready for downstream modeling workflows.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import rasterio

from pipelines.feature_engineering.domain.errors import FeatureEngineeringError


class DatasetBuilder:
    """Build a tabular feature dataset from aligned feature rasters.

    All input rasters must have identical CRS, transform, width, and height so
    each flattened row represents the same pixel location across every feature.
    The returned DataFrame always uses the feature column order expected by the
    HeatWise AI feature engineering pipeline.
    """

    def build(
        self,
        ndvi_path: Path,
        ndbi_path: Path,
        thermal_path: Path,
        building_density_path: Path,
        road_density_path: Path,
    ) -> pd.DataFrame:
        """Create a pandas DataFrame from aligned raster feature files.

        Args:
            ndvi_path: Path to the NDVI feature raster.
            ndbi_path: Path to the NDBI feature raster.
            thermal_path: Path to the thermal intensity feature raster.
            building_density_path: Path to the building density feature raster.
            road_density_path: Path to the road density feature raster.

        Returns:
            A DataFrame with columns ``ndvi``, ``ndbi``, ``thermal``,
            ``building_density``, and ``road_density``.

        Raises:
            FeatureEngineeringError: If rasters are not spatially aligned or an
                unexpected read/build failure occurs.
        """

        feature_paths = {
            "ndvi": ndvi_path,
            "ndbi": ndbi_path,
            "thermal": thermal_path,
            "building_density": building_density_path,
            "road_density": road_density_path,
        }

        try:
            with ExitStack() as stack:
                sources = {
                    feature_name: stack.enter_context(rasterio.open(path))
                    for feature_name, path in feature_paths.items()
                }
                reference_name = "ndvi"
                reference_src = sources[reference_name]

                for feature_name, source in sources.items():
                    if feature_name == reference_name:
                        continue
                    self._validate_aligned(reference_src, source)

                return pd.DataFrame(
                    {
                        feature_name: source.read(1).flatten()
                        for feature_name, source in sources.items()
                    }
                )
        except FeatureEngineeringError:
            raise
        except Exception as exc:
            raise FeatureEngineeringError(f"Failed to build feature dataset: {exc}") from exc

    def _validate_aligned(
        self,
        reference_src: rasterio.io.DatasetReader,
        candidate_src: rasterio.io.DatasetReader,
    ) -> None:
        """Ensure candidate raster metadata matches the reference raster."""

        if reference_src.crs != candidate_src.crs:
            raise FeatureEngineeringError("different CRS")
        if reference_src.transform != candidate_src.transform:
            raise FeatureEngineeringError("different transforms")
        if reference_src.width != candidate_src.width or reference_src.height != candidate_src.height:
            raise FeatureEngineeringError("different dimensions")
