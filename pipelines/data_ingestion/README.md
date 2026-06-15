# Data Ingestion Pipeline

Reusable ingestion package for urban heat mitigation studies. The default Mumbai run accepts a city boundary GeoJSON and downloads:

- Landsat 8 Collection 2 Level-2 imagery from Microsoft Planetary Computer STAC
- Sentinel-2 Level-2A imagery from Microsoft Planetary Computer STAC
- ERA5 single-level weather data from Copernicus CDS
- OpenStreetMap roads and building footprints through OSMnx

## Folder Structure

```text
pipelines/data_ingestion/
  application/
    pipeline.py              # orchestration service
  domain/
    errors.py                # ingestion exceptions
    models.py                # boundary, date range, result models
  infrastructure/
    clients/
      era5.py                # CDS API adapter
      landsat.py             # Landsat 8 STAC adapter
      osm.py                 # OSMnx adapter
      sentinel.py            # Sentinel-2 STAC adapter
      stac_utils.py          # shared satellite helpers
    logging_config.py
    storage.py               # local folders + GeoJSON reader
  cli.py                     # python -m entrypoint
  requirements-data-ingestion.txt
```

Outputs are organized under:

```text
data/raw/<city>/
  boundary/
  landsat/
  sentinel/
  era5/
  osm/
  logs/
  metadata/ingestion_manifest.json
```

## Setup

Install the additions listed in `pipelines/data_ingestion/requirements-data-ingestion.txt`.

ERA5 requires Copernicus CDS credentials configured in `%USERPROFILE%\.cdsapirc` or the equivalent environment for your runtime. Planetary Computer public STAC downloads do not require a project-specific token for typical public assets.

## Run

```powershell
python -m pipelines.data_ingestion.cli `
  --city mumbai `
  --boundary data/external/mumbai_boundary.geojson `
  --output-dir data/raw `
  --start-date 2024-01-01 `
  --end-date 2024-12-31 `
  --max-items 5
```

## Architecture

The package follows clean architecture boundaries:

- `domain`: pure models and exceptions.
- `ports`: protocols for boundary readers, downloaders, and output repositories.
- `application`: orchestration logic with no provider-specific SDK imports.
- `infrastructure`: concrete filesystem, STAC, CDS, and OSM adapters.

Provider failures are logged and re-raised as `IngestionError` so API workers or schedulers can retry the whole task consistently.

## Notes For Mumbai

Use a Mumbai municipal or study-area boundary GeoJSON in WGS84 coordinates. The boundary reader derives a bbox for STAC, CDS, and OSM queries while preserving the original GeoJSON in the output folder for lineage.
