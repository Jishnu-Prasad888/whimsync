from pathlib import Path

import pymupdf

from ..schemas.parsing import ParsedDocument, ParseTier
from ..utils.logger import get_logger
from .strategies.balanced import BalancedParsingStrategy
from .strategies.base import BaseParsingStrategy
from .strategies.max import MaxParsingStrategy

logger = get_logger("parsing_engine")


class ParsingEngine:
    """
    Central orchestration engine for PDF document layout parsing.
    Dispatches to Balanced or Max tier strategies based on configuration.
    """

    def __init__(self):
        self._strategies: dict[ParseTier, BaseParsingStrategy] = {
            ParseTier.BALANCED: BalancedParsingStrategy(),
            ParseTier.MAX: MaxParsingStrategy(),
        }

    async def parse(
        self,
        pdf_path: Path,
        work_dir: Path,
        tier: ParseTier | str = ParseTier.BALANCED,
    ) -> ParsedDocument:
        if not pdf_path.exists():
            raise FileNotFoundError(f"Target PDF does not exist: {pdf_path}")

        # Normalize tier
        if isinstance(tier, str):
            try:
                tier = ParseTier(tier.lower())
            except ValueError:
                logger.warning(f"Unknown tier '{tier}'. Defaulting to BALANCED.")
                tier = ParseTier.BALANCED

        strategy = self._strategies.get(tier, self._strategies[ParseTier.BALANCED])
        logger.info(f"Dispatching {pdf_path.name} to {tier.value.upper()} parsing strategy")

        doc = pymupdf.open(str(pdf_path))
        try:
            return await strategy.parse(doc, pdf_path, work_dir)
        finally:
            doc.close()


parsing_engine = ParsingEngine()
