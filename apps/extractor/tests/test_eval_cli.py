import json
import sys
from types import SimpleNamespace

import pymupdf
import pytest

from src import eval_cli
from src.parsing.evaluation_trace import TraceRecorder
from src.services.vlm_service import vlm_service

VLM_MARKDOWN = "# Scanned Heading\n\nText read by the fake VLM."


def _digital_pdf(path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A plain digital page with enough text to be routed natively. " * 3)
    doc.save(path)
    doc.close()


def _scanned_pdf(path):
    """One image-only page (no text layer), which every VLM tier routes to the VLM."""
    src = pymupdf.open()
    src.new_page().insert_text((72, 120), "Scanned words", fontsize=28)
    pix = src[0].get_pixmap(dpi=100)
    src.close()
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    doc.close()


class _FakeModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, model, contents):
        self.calls += 1
        return SimpleNamespace(
            text=VLM_MARKDOWN,
            usage_metadata=SimpleNamespace(prompt_token_count=1200, candidates_token_count=40),
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        )


@pytest.fixture
def fake_vlm(monkeypatch):
    models = _FakeModels()
    monkeypatch.setattr(vlm_service, "api_key", "test-key")
    monkeypatch.setattr(vlm_service, "delay_seconds", 0)
    monkeypatch.setattr(vlm_service, "_get_client", lambda: SimpleNamespace(models=models))
    return models


def _read(out):
    return (
        json.loads((out / "result.json").read_text()),
        json.loads((out / "trace.json").read_text()),
    )


def test_digital_pdf_writes_all_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_service, "api_key", None)
    pdf = tmp_path / "digital.pdf"
    _digital_pdf(pdf)
    out = tmp_path / "out"

    assert eval_cli.main(["--input", str(pdf), "--tier", "max", "--output", str(out)]) == 0

    result, trace = _read(out)
    assert result["status"] == "success"
    assert result["total_pages"] == 1
    assert result["vlm_page_count"] == 0
    assert len(result["input"]["sha256"]) == 64
    assert trace["pages"][0]["route"] == "native"
    assert trace["pages"][0]["complexity"]["needs_ocr"] is False
    page_md = (out / "pages" / "001.md").read_text()
    assert "plain digital page" in page_md
    assert (out / "document.md").read_text().strip() == page_md.strip()


def test_scanned_page_records_vlm_route_and_call(tmp_path, fake_vlm):
    pdf = tmp_path / "scan.pdf"
    _scanned_pdf(pdf)
    out = tmp_path / "out"

    # "smart" and "balanced" both resolve to whichever name this checkout uses.
    assert eval_cli.main(["--input", str(pdf), "--tier", "smart", "--output", str(out)]) == 0

    result, trace = _read(out)
    page = trace["pages"][0]
    assert page["route"] == "vlm"
    assert page["parser"] == "gemini"
    assert page["vlm_failed"] is False
    assert (out / "pages" / "001.md").read_text() == VLM_MARKDOWN

    [call] = trace["vlm_calls"]
    assert call["page_number"] == 1
    assert call["kind"] == "full_page"
    assert call["attempts"] == 1 and call["retries"] == 0
    assert (call["input_tokens"], call["output_tokens"]) == (1200, 40)
    assert call["finish_reason"] == "STOP"
    assert result["vlm_page_count"] == 1
    assert result["input_tokens"] == 1200


def test_unknown_tier_is_rejected(tmp_path):
    pdf = tmp_path / "digital.pdf"
    _digital_pdf(pdf)
    assert eval_cli.main(["--input", str(pdf), "--tier", "nope", "--output", str(tmp_path)]) == 2


def test_recorder_restores_patched_functions():
    strategies = [m for n, m in sys.modules.items() if n.startswith("src.parsing.strategies.")]
    before = {
        (m.__name__, a): getattr(m, a)
        for m in strategies
        for a in ("inject_links_inline",)
        if hasattr(m, a)
    }
    process_image = vlm_service.process_image

    with TraceRecorder():
        assert vlm_service.process_image != process_image

    after = {
        (m.__name__, a): getattr(m, a)
        for m in strategies
        for a in ("inject_links_inline",)
        if hasattr(m, a)
    }
    assert before == after
    assert "process_image" not in vlm_service.__dict__
