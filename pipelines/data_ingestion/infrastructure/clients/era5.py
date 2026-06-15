"""ERA5 weather data downloader using the Copernicus CDS API."""

from __future__ import annotations

from pathlib import Path

from pipelines.data_ingestion.domain.errors import DownloadError
from pipelines.data_ingestion.domain.models import BoundingBox, CityBoundary, DatasetResult, DateRange


MIN_ERA5_BBOX_SIZE_DEGREES = 0.5


class Era5Downloader:
    """Downloads ERA5 single-level weather variables clipped to a boundary bbox."""

    source_name = "era5"

    def __init__(self, variables: list[str] | None = None) -> None:
        self.variables = variables or [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
        ]

    def download(self, boundary: CityBoundary, date_range: DateRange, output_dir: Path) -> DatasetResult:
        try:
            import cdsapi
        except ImportError as exc:
            raise DownloadError("ERA5 downloads require cdsapi and configured CDS credentials.") from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "era5_single_levels.nc"
        retrieval_bbox = _expand_bbox(boundary.bbox, minimum_size=MIN_ERA5_BBOX_SIZE_DEGREES)
        client = cdsapi.Client()
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": self.variables,
                "year": date_range.years(),
                "month": date_range.months(),
                "day": date_range.days(),
                "time": [f"{hour:02d}:00" for hour in range(24)],
                "area": retrieval_bbox.as_cds_area(),
                "data_format": "netcdf",
            },
            str(target),
        )
        return DatasetResult(
            source=self.source_name,
            files=[target],
            metadata={
                "dataset": "reanalysis-era5-single-levels",
                "variables": self.variables,
                "boundary_bbox": boundary.bbox.as_stac_bbox(),
                "retrieval_area": retrieval_bbox.as_cds_area(),
            },
        )


def _expand_bbox(bbox: BoundingBox, minimum_size: float = MIN_ERA5_BBOX_SIZE_DEGREES) -> BoundingBox:
    west, east = _expand_interval(bbox.west, bbox.east, minimum_size, lower_limit=-180.0, upper_limit=180.0)
    south, north = _expand_interval(bbox.south, bbox.north, minimum_size, lower_limit=-90.0, upper_limit=90.0)
    return BoundingBox(west=west, south=south, east=east, north=north)


def _expand_interval(
    lower: float,
    upper: float,
    minimum_size: float,
    *,
    lower_limit: float,
    upper_limit: float,
) -> tuple[float, float]:
    current_size = upper - lower
    if current_size >= minimum_size:
        return lower, upper

    midpoint = (lower + upper) / 2
    half_size = minimum_size / 2
    expanded_lower = midpoint - half_size
    expanded_upper = midpoint + half_size

    if expanded_lower < lower_limit:
        expanded_upper += lower_limit - expanded_lower
        expanded_lower = lower_limit
    if expanded_upper > upper_limit:
        expanded_lower -= expanded_upper - upper_limit
        expanded_upper = upper_limit

    return expanded_lower, expanded_upper
