"""
Evaluation-only instrumentation for the parsing strategies.

`TraceRecorder` observes one `ParsingEngine.parse` run without changing production
code. Inside its `with` block it wraps the functions the strategies already call:

- `analyze_page_complexity`: the classifier facts that drive routing;
- `inject_links_inline`: always the last per-page step, so its result is the page's
  final Markdown (after VLM replacement and link injection);
- `vlm_service.process_image` and the Gemini client's `generate_content`: one record
  per VLM request with its page, attempts, tokens and timing.

Everything is restored on exit. Only `src.eval_cli` uses this module; the production
parser consumer never imports it.
"""

import contextvars
import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Any

import pymupdf4llm

from ..services.vlm_service import vlm_service

# The module that defines the prompts (not the `vlm_service` instance of the same name).
vlm_module = sys.modules[type(vlm_service).__module__]

STRATEGIES_PACKAGE = "src.parsing.strategies"

# Full-page renders are saved as `page_<n>.png` (1-based) by the strategies.
_PAGE_SCAN_RE = re.compile(r"page_(\d+)\.png$")
# PyMuPDF4LLM names embedded images `<file>-<page:04d>-<index>.<ext>` (1-based page).
_EMBEDDED_IMAGE_RE = re.compile(r"-(\d{4})-\d+\.\w+$")

_PROMPTS = {"full_page_scan": "FULL_PAGE_SCAN_PROMPT", "image": "DEFAULT_IMAGE_PROMPT"}

_current_call: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "eval_vlm_call", default=None
)


def prompt_versions() -> dict[str, str]:
    """Short content hash of each VLM prompt, so prompt edits are traceable."""
    out = {}
    for kind, attr in _PROMPTS.items():
        text = getattr(vlm_module, attr, None)
        if isinstance(text, str):
            out[kind] = hashlib.sha256(text.encode()).hexdigest()[:12]
    return out


def render_scales() -> dict[str, float]:
    """Full-page render scale per strategy, read from the strategy source."""
    scales = {}
    for name, module in _strategy_modules().items():
        path = getattr(module, "__file__", None)
        if not path:
            continue
        m = re.search(r"Matrix\(\s*([\d.]+)", Path(path).read_text())
        if m:
            scales[name.rsplit(".", 1)[-1]] = float(m.group(1))
    return scales


def _strategy_modules() -> dict[str, Any]:
    return {
        name: module
        for name, module in list(sys.modules.items())
        if name.startswith(STRATEGIES_PACKAGE + ".") and module is not None
    }


def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    return getattr(reason, "name", None) or (str(reason) if reason is not None else None)


def _usage(response: Any) -> tuple[int | None, int | None]:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None, None
    return getattr(meta, "prompt_token_count", None), getattr(meta, "candidates_token_count", None)


class TraceRecorder:
    """Context manager recording page complexity, routing, final page Markdown and VLM calls."""

    def __init__(self) -> None:
        self.complexities: dict[int, dict] = {}
        self.page_markdown: dict[int, str] = {}
        self._inject_input: dict[int, str] = {}
        self._full_page_results: dict[int, str] = {}
        self._unassigned_calls: list[dict] = []
        self.vlm_calls: list[dict] = []
        self.native_extraction_ms = 0.0
        self._restore: list[tuple[Any, str, Any]] = []

    # ---- patching -------------------------------------------------------------

    def _patch(self, owner: Any, attr: str, value: Any) -> None:
        self._restore.append((owner, attr, owner.__dict__.get(attr, _MISSING)))
        setattr(owner, attr, value)

    def __enter__(self) -> "TraceRecorder":
        for module in _strategy_modules().values():
            if hasattr(module, "analyze_page_complexity"):
                self._patch(
                    module,
                    "analyze_page_complexity",
                    self._wrap_complexity(module.analyze_page_complexity),
                )
            if hasattr(module, "inject_links_inline"):
                self._patch(
                    module, "inject_links_inline", self._wrap_inject(module.inject_links_inline)
                )
        self._patch(pymupdf4llm, "to_markdown", self._wrap_native(pymupdf4llm.to_markdown))
        self._patch(
            vlm_service, "process_image", self._wrap_process_image(vlm_service.process_image)
        )
        self._patch(vlm_service, "_get_client", self._wrap_get_client(vlm_service._get_client))
        return self

    def __exit__(self, *exc: object) -> None:
        for owner, attr, original in reversed(self._restore):
            if original is _MISSING:
                try:
                    delattr(owner, attr)
                except AttributeError:
                    pass
            else:
                setattr(owner, attr, original)
        self._restore.clear()

    def _wrap_complexity(self, original):
        def wrapped(page, *args, **kwargs):
            result = original(page, *args, **kwargs)
            self.complexities[result.page_number] = result.model_dump()
            return result

        return wrapped

    def _wrap_inject(self, original):
        def wrapped(page, md_text: str, *args, **kwargs):
            index = page.number
            result = original(page, md_text, *args, **kwargs)
            self._inject_input[index] = md_text
            self.page_markdown[index] = result
            # Embedded-image calls whose file name didn't reveal a page ran for this page:
            # strategies enrich a page's images right before injecting its links.
            for call in self._unassigned_calls:
                call["page_number"] = index + 1
            self._unassigned_calls.clear()
            return result

        return wrapped

    def _wrap_native(self, original):
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.native_extraction_ms += (time.perf_counter() - start) * 1000

        return wrapped

    def _wrap_process_image(self, original):
        async def wrapped(image_path, prompt_type: str = "full_page_scan", *args, **kwargs):
            name = Path(image_path).name
            full_page = prompt_type == "full_page_scan"
            match = (_PAGE_SCAN_RE if full_page else _EMBEDDED_IMAGE_RE).search(name)
            call: dict[str, Any] = {
                "page_number": int(match.group(1)) if match else None,
                "kind": "full_page" if full_page else "embedded_image",
                "image": name,
                "model": vlm_service.model,
                "prompt_version": prompt_versions().get("full_page_scan" if full_page else "image"),
                "attempts": 0,
                "retries": 0,
                "input_tokens": None,
                "output_tokens": None,
                "api_ms": 0.0,
                "duration_ms": None,
                "ok": False,
                "chars": 0,
                "finish_reason": None,
                "errors": [],
            }
            self.vlm_calls.append(call)
            if call["page_number"] is None:
                self._unassigned_calls.append(call)
            token = _current_call.set(call)
            start = time.perf_counter()
            try:
                result = await original(image_path, prompt_type, *args, **kwargs)
            finally:
                _current_call.reset(token)
                call["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
                call["api_ms"] = round(call["api_ms"], 1)
                call["retries"] = max(0, call["attempts"] - 1)
            call["ok"] = bool(result and result.strip())
            call["chars"] = len(result or "")
            if full_page and call["page_number"] is not None:
                self._full_page_results[call["page_number"] - 1] = (result or "").strip()
            return result

        return wrapped

    def _wrap_get_client(self, original):
        def wrapped(*args, **kwargs):
            client = original(*args, **kwargs)
            models = client.models
            if not getattr(models, "_eval_traced", False):
                generate = models.generate_content

                def traced_generate(*g_args, **g_kwargs):
                    call = _current_call.get()
                    start = time.perf_counter()
                    try:
                        response = generate(*g_args, **g_kwargs)
                    except Exception as e:
                        if call is not None:
                            call["attempts"] += 1
                            call["api_ms"] += (time.perf_counter() - start) * 1000
                            call["errors"].append(str(e)[:300])
                        raise
                    if call is not None:
                        call["attempts"] += 1
                        call["api_ms"] += (time.perf_counter() - start) * 1000
                        tokens_in, tokens_out = _usage(response)
                        call["input_tokens"] = tokens_in
                        call["output_tokens"] = tokens_out
                        call["finish_reason"] = _finish_reason(response)
                    return response

                models.generate_content = traced_generate
                models._eval_traced = True
            return client

        return wrapped

    # ---- results --------------------------------------------------------------

    def route(self, index: int) -> str:
        if index not in self.page_markdown:
            return "empty"
        if self.complexities.get(index, {}).get("is_empty"):
            return "empty"
        vlm_md = self._full_page_results.get(index)
        if vlm_md and self._inject_input.get(index, "").strip() == vlm_md:
            return "vlm"
        return "native"

    def pages(self, total_pages: int) -> list[dict]:
        pages = []
        for index in range(total_pages):
            route = self.route(index)
            calls = [c for c in self.vlm_calls if c["page_number"] == index + 1]
            complexity = self.complexities.get(index)
            pages.append(
                {
                    "page_number": index + 1,
                    "complexity": (
                        {
                            "needs_ocr": complexity["needs_ocr"],
                            "reasons": complexity["reasons"],
                            "is_empty": complexity["is_empty"],
                            "text_coverage": complexity["text_coverage"],
                            "image_coverage": complexity["image_coverage"],
                            "vg_coverage": complexity["vg_coverage"],
                            "text_length": complexity["text_length"],
                        }
                        if complexity
                        else None
                    ),
                    "route": route,
                    "parser": {"vlm": "gemini", "native": "pymupdf4llm"}.get(route),
                    "vlm_failed": any(c["kind"] == "full_page" and not c["ok"] for c in calls),
                    "embedded_images_enriched": sum(c["kind"] == "embedded_image" for c in calls),
                    # VLM time for this page; native pages are extracted in one
                    # document-wide pass (see `native_extraction_ms`).
                    "duration_ms": round(sum(c["api_ms"] for c in calls), 1) if calls else None,
                    "markdown_chars": len(self.page_markdown.get(index, "")),
                }
            )
        return pages

    def trace(self, total_pages: int) -> dict:
        return {
            "pages": self.pages(total_pages),
            "vlm_calls": self.vlm_calls,
            "native_extraction_ms": round(self.native_extraction_ms, 1),
        }


_MISSING = object()
