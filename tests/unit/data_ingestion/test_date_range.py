from datetime import date

from pipelines.data_ingestion.domain.models import DateRange


def test_date_range_formats_stac_datetime():
    date_range = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 31))

    assert date_range.as_stac_datetime() == "2024-01-01/2024-01-31"
    assert date_range.years() == ["2024"]
    assert date_range.months() == ["01"]
