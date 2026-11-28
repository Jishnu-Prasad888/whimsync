"""
Evaluation CLI: parse one local PDF with one tier and write evaluation artifacts.

    python -m src.eval_cli --input /input/source.pdf --tier balanced --output /output

It runs `ParsingEngine.parse` exactly as the parser consumer does, but without
Redis, BullMQ, MinIO or Postgres, then exits. Outputs in `--output`:

    document.md     the stitched Markdown production would store and chunk
    pages/NNN.md    each page's final Markdown (after VLM replacement + link injection)
    trace.json      per-page complexity, route and timing, plus every VLM call
    result.json     run configuration and summary (status, model, counts, timing)

Gemini settings come from the environment (`GEMINI_API_KEY`, `GEMINI_MODEL`,
`VLM_*`). The evaluator passes `EXTRACTOR_COMMIT`, `IMAGE_TAG` and `IMAGE_DIGEST` so
results can be traced back to the exact build. Exit code is 0 on success, 1 on a
parse failure (artifacts are still written), 2 on bad arguments.
"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from importlib import metadata
from pathlib import Path

from .config import settings
from .parsing.engine import parsing_engine
from .parsing.evaluation_trace import TraceRecorder, prompt_versions, render_scales
from .schemas.parsing import ParseTier
from .services.vlm_service import vlm_service

# The tier formerly called Smart is now Balanced; accept either name on any checkout.
_TIER_ALIASES = {"smart": "balanced", "balanced": "smart"}


def resolve_tier(name: str) -> ParseTier:
    values = {t.value for t in ParseTier}
    name = name.strip().lower()
    if name in values:
        return ParseTier(name)
    alias = _TIER_ALIASES.get(name)
    if alias in values:
        return ParseTier(alias)
    raise ValueError(f"Unknown tier '{name}'. Available: {', '.join(sorted(values))}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _settings_snapshot() -> dict:
    return {
        name.lower(): getattr(settings, name)
        for name in ("VLM_CONCURRENCY", "VLM_MAX_RETRIES", "VLM_DELAY_SECONDS", "VLM_RPM")
        if hasattr(settings, name)
    }


def run(input_pdf: Path, tier: ParseTier, output: Path, model: str | None) -> int:
    if model:
        vlm_service.model = model

    output.mkdir(parents=True, exist_ok=True)
    pages_dir = output / "pages"
    if pages_dir.exists():
        shutil.rmtree(pages_dir)
    pages_dir.mkdir()

    status, error, parsed = "success", None, None
    recorder = TraceRecorder()
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="eval-work-") as work, recorder:
        try:
            parsed = asyncio.run(parsing_engine.parse(input_pdf, Path(work), tier))
        except Exception as e:  # recorded in result.json; the evaluator decides what it means
            status, error = "failed", f"{type(e).__name__}: {e}"
            traceback.print_exc()
    duration = time.perf_counter() - start

    total_pages = parsed.total_pages if parsed else len(recorder.complexities)
    trace = recorder.trace(total_pages)

    if parsed:
        (output / "document.md").write_text(parsed.markdown, encoding="utf-8")
    for index in range(total_pages):
        page_md = recorder.page_markdown.get(index, "")
        (pages_dir / f"{index + 1:03d}.md").write_text(page_md, encoding="utf-8")
    _write_json(output / "trace.json", trace)

    calls = trace["vlm_calls"]
    tokens_in = [c["input_tokens"] for c in calls if c["input_tokens"] is not None]
    tokens_out = [c["output_tokens"] for c in calls if c["output_tokens"] is not None]
    result = {
        "extractor_commit": os.environ.get("EXTRACTOR_COMMIT", "unknown"),
        "image_tag": os.environ.get("IMAGE_TAG"),
        "image_digest": os.environ.get("IMAGE_DIGEST"),
        "input": {"name": input_pdf.name, "sha256": _sha256(input_pdf)},
        "tier": tier.value,
        "tier_used": parsed.tier_used.value if parsed else None,
        "model": vlm_service.model,
        "vlm_available": vlm_service.is_available(),
        "prompt_versions": prompt_versions(),
        "render_scale": render_scales(),
        "settings": _settings_snapshot(),
        "versions": {
            "pymupdf": _version("pymupdf"),
            "pymupdf4llm": _version("pymupdf4llm"),
            "google-genai": _version("google-genai"),
            "python": sys.version.split()[0],
        },
        "total_pages": total_pages,
        "vlm_page_count": sum(p["route"] == "vlm" for p in trace["pages"]),
        "vlm_calls": len(calls),
        "vlm_failures": sum(not c["ok"] for c in calls),
        "retries": sum(c["retries"] for c in calls),
        "input_tokens": sum(tokens_in) if tokens_in else None,
        "output_tokens": sum(tokens_out) if tokens_out else None,
        "output_chars": len(parsed.markdown) if parsed else 0,
        "duration_seconds": round(duration, 3),
        "status": status,
        "error": error,
        "strategy_metadata": parsed.metadata if parsed else None,
    }
    _write_json(output / "result.json", result)
    print(
        f"[eval] {input_pdf.name} tier={tier.value} status={status} pages={total_pages} "
        f"vlm_pages={result['vlm_page_count']} {duration:.1f}s",
        file=sys.stderr,
    )
    return 0 if status == "success" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whimsync-extractor-eval",
        description="Parse one PDF with one tier and write evaluation artifacts.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Local PDF path")
    parser.add_argument("--tier", required=True, help="Parsing tier, e.g. balanced or max")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--model", help="Override GEMINI_MODEL for this run")
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"Input PDF not found: {args.input}", file=sys.stderr)
        return 2
    try:
        tier = resolve_tier(args.tier)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    return run(args.input, tier, args.output, args.model)


if __name__ == "__main__":
    sys.exit(main())
