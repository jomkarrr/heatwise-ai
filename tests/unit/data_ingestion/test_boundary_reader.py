import json
from pathlib import Path

import pytest

from pipelines.data_ingestion.domain.errors import BoundaryError
from pipelines.data_ingestion.infrastructure.storage import GeoJsonBoundaryReader


def test_boundary_reader_extracts_bbox_from_feature_collection(tmp_path: Path):
    boundary_path = tmp_path / "mumbai.geojson"
    boundary_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "Mumbai sample"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [72.77, 18.89],
                                    [72.99, 18.89],
                                    [72.99, 19.27],
                                    [72.77, 19.27],
                                    [72.77, 18.89],
                                ]
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    boundary = GeoJsonBoundaryReader().read(boundary_path)

    assert boundary.bbox.as_stac_bbox() == [72.77, 18.89, 72.99, 19.27]
    assert boundary.bbox.as_cds_area() == [19.27, 72.77, 18.89, 72.99]


def test_boundary_reader_rejects_missing_file(tmp_path: Path):
    with pytest.raises(BoundaryError):
        GeoJsonBoundaryReader().read(tmp_path / "missing.geojson")
