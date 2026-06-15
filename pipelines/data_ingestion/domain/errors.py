"""Domain-specific exceptions for ingestion workflows."""


class IngestionError(Exception):
    """Base exception for data ingestion failures."""


class BoundaryError(IngestionError):
    """Raised when a city boundary cannot be parsed or is invalid."""


class DownloadError(IngestionError):
    """Raised when a data provider cannot complete a download."""
