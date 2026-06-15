import logging
from pathlib import Path

import pytest

from pipelines.data_ingestion.domain.errors import DownloadError
from pipelines.data_ingestion.infrastructure.clients.stac_utils import download_assets


class FakeAsset:
    def __init__(self, href: str) -> None:
        self.href = href


class FakeItem:
    def __init__(self, item_id: str, assets: dict[str, FakeAsset]) -> None:
        self.id = item_id
        self.assets = assets


class FakeResponse:
    def __init__(self, chunks: list[bytes], *, status_code: int = 200, headers: dict[str, str] | None = None) -> None:
        self.chunks = chunks
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        yield from self.chunks


def test_download_assets_skips_existing_files(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    target = tmp_path / "landsat8_01_scene_red.TIF"
    target.write_bytes(b"already here")
    item = FakeItem("scene", {"red": FakeAsset("https://example.test/scene_red.TIF")})

    def fail_request(*args, **kwargs):
        raise AssertionError("existing assets should not be requested")

    caplog.set_level(logging.INFO)

    files = download_assets([item], ["red"], tmp_path, "landsat8", request_get=fail_request)

    assert files == [target]
    assert target.read_bytes() == b"already here"
    assert "Skipping existing asset" in caplog.text


def test_download_assets_resumes_partial_download(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    target = tmp_path / "landsat8_01_scene_nir08.TIF"
    target.with_suffix(".TIF.part").write_bytes(b"abc")
    item = FakeItem("scene", {"nir08": FakeAsset("https://example.test/scene_nir08.TIF")})
    calls = []

    def fake_request(href, **kwargs):
        calls.append({"href": href, **kwargs})
        return FakeResponse([b"def"], status_code=206, headers={"content-length": "3"})

    caplog.set_level(logging.INFO)

    files = download_assets([item], ["nir08"], tmp_path, "landsat8", request_get=fake_request, chunk_size=2)

    assert files == [target]
    assert target.read_bytes() == b"abcdef"
    assert not target.with_suffix(".TIF.part").exists()
    assert calls[0]["headers"] == {"Range": "bytes=3-"}
    assert calls[0]["stream"] is True
    assert "Resuming download" in caplog.text
    assert "Download progress" in caplog.text


def test_download_assets_retries_with_exponential_backoff(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    item = FakeItem("scene", {"swir16": FakeAsset("https://example.test/scene_swir16.TIF")})
    sleeps = []
    attempts = {"count": 0}

    def flaky_request(href, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary outage")
        return FakeResponse([b"ok"], headers={"content-length": "2"})

    caplog.set_level(logging.INFO)

    files = download_assets(
        [item],
        ["swir16"],
        tmp_path,
        "landsat8",
        retries=3,
        backoff_seconds=0.5,
        request_get=flaky_request,
        sleep=sleeps.append,
    )

    assert files == [tmp_path / "landsat8_01_scene_swir16.TIF"]
    assert files[0].read_bytes() == b"ok"
    assert sleeps == [0.5, 1.0]
    assert attempts["count"] == 3
    assert "retrying in 0.5 seconds" in caplog.text
    assert "retrying in 1.0 seconds" in caplog.text


def test_download_assets_raises_after_retries_are_exhausted(tmp_path: Path):
    item = FakeItem("scene", {"lwir11": FakeAsset("https://example.test/scene_lwir11.TIF")})

    with pytest.raises(DownloadError, match="after 2 attempts"):
        download_assets(
            [item],
            ["lwir11"],
            tmp_path,
            "landsat8",
            retries=1,
            backoff_seconds=0,
            request_get=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
            sleep=lambda _: None,
        )
