"""Command-line entrypoint for the data ingestion pipeline."""

from __future__ import annotations

import argparse
import logging

from pipelines.data_ingestion.application.pipeline import build_default_pipeline
from pipelines.data_ingestion.config import IngestionConfig
from pipelines.data_ingestion.infrastructure.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download urban heat mitigation source datasets.")
    parser.add_argument("--city", default="mumbai", help="City identifier used for output folders.")
    parser.add_argument("--boundary", required=True, help="Path to city boundary GeoJSON.")
    parser.add_argument("--output-dir", default="data/raw", help="Base output directory.")
    parser.add_argument("--start-date", required=True, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="End date, YYYY-MM-DD.")
    parser.add_argument("--max-items", type=int, default=5, help="Maximum STAC items per satellite source.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = IngestionConfig.from_strings(
        city=args.city,
        boundary_path=args.boundary,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        max_items=args.max_items,
    )
    result = build_default_pipeline(config).run()
    logging.getLogger(__name__).info("Wrote %s files to %s", len(result.files), result.output_root)


if __name__ == "__main__":
    main()
