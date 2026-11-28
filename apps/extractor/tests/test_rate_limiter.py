import asyncio
import importlib
import time

import pytest

from src.services.vlm_service import VLMService, _retry_delay
from src.utils.rate_limiter import RequestRateLimiter

# The package re-exports the `vlm_service` instance, so fetch the module explicitly.
vlm_module = importlib.import_module("src.services.vlm_service")


@pytest.mark.asyncio
async def test_disabled_limiter_never_waits():
    limiter = RequestRateLimiter(0)
    assert not limiter.enabled
    start = time.monotonic()
    for _ in range(20):
        assert await limiter.acquire() == 0.0
    assert time.monotonic() - start < 0.05


@pytest.mark.asyncio
async def test_limiter_spaces_requests():
    limiter = RequestRateLimiter(600)  # one slot every 0.1s
    start = time.monotonic()
    await asyncio.gather(*(limiter.acquire() for _ in range(4)))
    elapsed = time.monotonic() - start
    # 4 requests -> slots at 0, 0.1, 0.2, 0.3
    assert 0.28 <= elapsed < 0.5


def test_set_rpm_toggles_limiting():
    limiter = RequestRateLimiter()
    assert not limiter.enabled
    limiter.set_rpm(15)
    assert limiter.enabled
    assert limiter.rpm == 15


def test_retry_delay_honours_server_hint():
    err = "429 RESOURCE_EXHAUSTED. {'details': [{'@type': '...RetryInfo', 'retryDelay': '42s'}]}"
    delay = _retry_delay(err, attempt=0)
    assert 42 <= delay <= 42 * 1.25


def test_retry_delay_exponential_without_hint():
    assert 1.5 <= _retry_delay("429", 0) <= 1.5 * 1.25
    assert 6.0 <= _retry_delay("429", 2) <= 6.0 * 1.25


@pytest.mark.asyncio
async def test_non_retryable_error_fails_fast_and_is_counted(tmp_path, monkeypatch):
    from PIL import Image

    img = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(img)

    calls = 0

    class FakeModels:
        def generate_content(self, **_):
            nonlocal calls
            calls += 1
            raise RuntimeError("404 NOT_FOUND: model is no longer available")

    class FakeClient:
        models = FakeModels()

    svc = VLMService(api_key="k", max_retries=3, delay_seconds=0)
    monkeypatch.setattr(svc, "_get_client", lambda: FakeClient())
    assert await svc.process_image(img) == ""
    assert calls == 1, "non-retryable errors must not be retried"
    assert svc.failed_calls == 1


@pytest.mark.asyncio
async def test_transient_error_is_retried(tmp_path, monkeypatch):
    from PIL import Image

    img = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(img)

    calls = 0

    class Resp:
        text = "# ok"

    class FakeModels:
        def generate_content(self, **_):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("503 UNAVAILABLE: model overloaded")
            return Resp()

    class FakeClient:
        models = FakeModels()

    svc = VLMService(api_key="k", max_retries=2, delay_seconds=0)
    monkeypatch.setattr(svc, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(vlm_module, "_retry_delay", lambda *_: 0.0)
    assert await svc.process_image(img) == "# ok"
    assert calls == 2
    assert svc.failed_calls == 0


@pytest.mark.asyncio
async def test_exhausted_daily_quota_fails_fast(tmp_path, monkeypatch):
    from PIL import Image

    img = tmp_path / "page.png"
    Image.new("RGB", (10, 10), "white").save(img)

    calls = 0

    class FakeModels:
        def generate_content(self, **_):
            nonlocal calls
            calls += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED {'retryDelay': '8046s'}")

    class FakeClient:
        models = FakeModels()

    svc = VLMService(api_key="k", max_retries=3, delay_seconds=0)
    monkeypatch.setattr(svc, "_get_client", lambda: FakeClient())
    start = time.monotonic()
    assert await svc.process_image(img) == ""
    assert time.monotonic() - start < 1.0, "must not sleep out a daily quota"
    assert calls == 1
    assert svc.failed_calls == 1
