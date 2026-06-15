"""Shared STAC download helpers for satellite providers."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from pipelines.data_ingestion.domain.errors import DownloadError

LOGGER = logging.getLogger(__name__)
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_CHUNK_SIZE = 1024 * 1024


def search_planetary_computer_items(
    collection: str,
    bbox: list[float],
    datetime_range: str,
    query: dict,
    max_items: int,
) -> list:
    try:
        import planetary_computer
        from pystac_client import Client
    except ImportError as exc:
        raise DownloadError(
            "Satellite downloads require pystac-client and planetary-computer. "
            "Install pipelines/data_ingestion/requirements-data-ingestion.txt."
        ) from exc

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(collections=[collection], bbox=bbox, datetime=datetime_range, query=query, max_items=max_items)
    return [planetary_computer.sign(item) for item in search.items()]


def download_assets(
    items: list,
    asset_keys: list[str],
    output_dir: Path,
    prefix: str,
    *,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    request_get: Callable | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    if request_get is None:
        try:
            import requests
        except ImportError as exc:
            raise DownloadError("Satellite asset downloads require requests.") from exc
        request_get = requests.get

    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for item_index, item in enumerate(items, start=1):
        for asset_key in asset_keys:
            if asset_key not in item.assets:
                LOGGER.warning("Asset %s missing from item %s", asset_key, item.id)
                continue
            href = item.assets[asset_key].href
            suffix = Path(urlparse(href).path).suffix or ".tif"
            target = output_dir / f"{prefix}_{item_index:02d}_{item.id}_{asset_key}{suffix}"
            if target.exists():
                LOGGER.info("Skipping existing asset %s", target)
                files.append(target)
                continue
            _download_with_resume(
                href=href,
                target=target,
                request_get=request_get,
                retries=retries,
                backoff_seconds=backoff_seconds,
                chunk_size=chunk_size,
                sleep=sleep,
            )
            files.append(target)
    if not files:
        raise DownloadError(f"No assets downloaded for {prefix}; check date range, cloud cover, and boundary.")
    return files


def _download_with_resume(
    *,
    href: str,
    target: Path,
    request_get: Callable,
    retries: int,
    backoff_seconds: float,
    chunk_size: int,
    sleep: Callable[[float], None],
) -> None:
    partial_target = target.with_suffix(f"{target.suffix}.part")
    attempt = 0
    while True:
        resume_from = partial_target.stat().st_size if partial_target.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        mode = "ab" if resume_from else "wb"
        try:
            if resume_from:
                LOGGER.info("Resuming download for %s from byte %s", target, resume_from)
            else:
                LOGGER.info("Starting download for %s", target)

            response = request_get(href, headers=headers, stream=True, timeout=120)
            response.raise_for_status()
            status_code = getattr(response, "status_code", None)
            if resume_from and status_code == 200:
                LOGGER.info("Server ignored resume for %s; restarting download", target)
                resume_from = 0
                mode = "wb"

            total_bytes = _expected_total_bytes(response, resume_from)
            downloaded = resume_from
            with partial_target.open(mode) as file_obj:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    file_obj.write(chunk)
                    downloaded += len(chunk)
                    _log_download_progress(target, downloaded, total_bytes)

            partial_target.replace(target)
            LOGGER.info("Completed download for %s (%s bytes)", target, downloaded)
            return
        except Exception as exc:
            attempt += 1
            if attempt > retries:
                raise DownloadError(f"Failed downloading {target.name} after {retries + 1} attempts: {exc}") from exc
            delay = backoff_seconds * (2 ** (attempt - 1))
            LOGGER.warning(
                "Download failed for %s on attempt %s/%s; retrying in %.1f seconds",
                target,
                attempt,
                retries + 1,
                delay,
            )
            sleep(delay)


def _expected_total_bytes(response: object, resume_from: int) -> int | None:
    headers = getattr(response, "headers", {}) or {}
    content_length = headers.get("content-length") or headers.get("Content-Length")
    if content_length is None:
        return None
    try:
        remaining_bytes = int(content_length)
    except ValueError:
        return None
    return resume_from + remaining_bytes


def _log_download_progress(target: Path, downloaded: int, total: int | None) -> None:
    if total:
        percent = min(100.0, (downloaded / total) * 100)
        LOGGER.info("Download progress for %s: %.1f%% (%s/%s bytes)", target, percent, downloaded, total)
    else:
        LOGGER.info("Download progress for %s: %s bytes", target, downloaded)
