import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pymupdf
import pytest

from src.parsing.engine import ParsingEngine
from src.parsing.strategies.balanced import BalancedParsingStrategy
from src.schemas.parsing import ParsedDocument, ParseTier


def _create_sample_pdf(pdf_path: Path, pages: list[tuple[str, str]]) -> None:
    doc = pymupdf.open()
    try:
        for heading, body in pages:
            page = doc.new_page(width=600, height=800)
            page.insert_text(pymupdf.Point(50, 50), heading, fontsize=18)
            page.insert_text(pymupdf.Point(50, 90), body)
        doc.save(str(pdf_path))
    finally:
        doc.close()


@pytest.mark.asyncio
async def test_balanced_strategy_extracts_digital_pdf():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_path = temp_path / "sample.pdf"
        _create_sample_pdf(
            pdf_path,
            [
                (
                    "# Whimsync Document Header",
                    "This is section one of the document with clean digital text.",
                ),
                (
                    "## Second Section",
                    "This is section two with additional digital text content.",
                ),
            ],
        )

        strategy = BalancedParsingStrategy()
        with (
            patch("src.parsing.strategies.balanced.vlm_service") as mock_vlm,
            pymupdf.open(str(pdf_path)) as open_doc,
        ):
            mock_vlm.is_available.return_value = False
            result = await strategy.parse(open_doc, pdf_path, temp_path)

        assert result.tier_used == ParseTier.BALANCED
        assert result.total_pages == 2
        assert result.metadata["strategy"] == "hybrid_balanced"
        assert result.metadata["vlm_page_count"] == 0
        assert "Whimsync" in result.markdown
        assert "Second Section" in result.markdown


@pytest.mark.asyncio
async def test_parsing_engine_dispatches_and_falls_back_to_balanced():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_path = temp_path / "dispatch.pdf"
        _create_sample_pdf(
            pdf_path,
            [("Dispatch Test", "Sample text used to verify tier routing in ParsingEngine.")],
        )

        engine = ParsingEngine()
        balanced_mock = MagicMock()
        balanced_mock.parse = AsyncMock(
            return_value=ParsedDocument(
                markdown="balanced output",
                total_pages=1,
                tier_used=ParseTier.BALANCED,
            )
        )
        max_mock = MagicMock()
        max_mock.parse = AsyncMock(
            return_value=ParsedDocument(
                markdown="max output",
                total_pages=1,
                tier_used=ParseTier.MAX,
            )
        )
        engine._strategies = {
            ParseTier.BALANCED: balanced_mock,
            ParseTier.MAX: max_mock,
        }

        balanced_res = await engine.parse(pdf_path, temp_path, tier="balanced")
        max_res = await engine.parse(pdf_path, temp_path, tier="max")
        fallback_res = await engine.parse(pdf_path, temp_path, tier="unknown-tier")

        assert balanced_res.tier_used == ParseTier.BALANCED
        assert max_res.tier_used == ParseTier.MAX
        assert fallback_res.tier_used == ParseTier.BALANCED
        assert balanced_mock.parse.await_count == 2
        assert max_mock.parse.await_count == 1
