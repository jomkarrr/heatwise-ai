# Data Ingestion

The reusable ingestion implementation lives in `pipelines/data_ingestion/`.

It takes a city boundary GeoJSON and writes Landsat 8, Sentinel-2, ERA5, and OpenStreetMap outputs into `data/raw/<city>/` with a manifest at `metadata/ingestion_manifest.json`.

See `pipelines/data_ingestion/README.md` for setup, credentials, command-line usage, and the generated folder structure.
