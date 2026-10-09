"""Chunking orchestrator: the single entry point for chunking markdown.

Flow per document:

    read markdown  ->  classify (rule-based)  ->  route to chunker
                   ->  fallback to agentic on failure  ->  validate  ->  result

Usage (Python):

    from chunker.orchestrator import ChunkingOrchestrator

    orch = ChunkingOrchestrator()                       # LLM_PROVIDER from .env
    result = orch.chunk_file("files/resume_1.md")
    result.chunks          # list[dict], same schema for every chunker
    result.strategy_used   # "adaptive" | "semantic" | "agentic"

    # async callers (e.g. FastAPI) — runs in a worker thread
    result = await orch.achunk_text(markdown, source="upload.md")

Usage (CLI):

    python -m chunker.orchestrator files/                 # chunk every .md
    python -m chunker.orchestrator files/resume_1.md --classify-only
    python -m chunker.orchestrator files/ --out chunks.json

Scope: chunking only. Embedding and the Qdrant upsert are not done here;
``result.chunks`` is what you hand to that stage.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chunker._schema import missing_keys
from chunker.document_classifier import (
    ChunkingStrategy,
    ClassificationResult,
    DocumentRouter,
    classify_document,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".md", ".markdown")

_LINK_FIELDS: tuple[str, ...] = ("parent_chunk_id", "prev_chunk_id", "next_chunk_id")


# ──────────────────────────────────────────────
#  Result type
# ──────────────────────────────────────────────

@dataclass
class ChunkingResult:
    doc_id: str
    source: str
    success: bool
    document_type: str | None = None
    confidence: float | None = None
    strategy_selected: str | None = None     # what the classifier picked
    strategy_used: str | None = None         # what actually produced chunks
    fell_back: bool = False                  # selected != used
    scores: dict[str, float] = field(default_factory=dict)
    chunks: list[dict] = field(default_factory=list)
    chunk_count: int = 0
    total_tokens: int = 0
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> str:
        if not self.success:
            return f"{self.source}: FAILED ({self.error})"
        fb = f" (fallback from {self.strategy_selected})" if self.fell_back else ""
        return (
            f"{self.source}: type={self.document_type} conf={self.confidence} "
            f"strategy={self.strategy_used}{fb} chunks={self.chunk_count} "
            f"tokens={self.total_tokens} time={self.elapsed_s:.1f}s"
        )

    def to_dict(self, include_chunks: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_chunks:
            data.pop("chunks")
        return data


# ──────────────────────────────────────────────
#  Orchestrator
# ──────────────────────────────────────────────

class ChunkingOrchestrator:
    """Classify, route, chunk and validate markdown documents.

    One instance can be shared process-wide; chunkers are built lazily and
    cached by ``DocumentRouter``. Chunking calls are serialised with a lock
    because the router's lazy chunker construction and the shared HF
    tokenizers are not designed for concurrent first use.
    """

    def __init__(
        self,
        provider: str | None = None,
        max_chunk_tokens: int | None = None,
        min_chunk_tokens: int | None = None,
        router: DocumentRouter | None = None,
        agentic_provider: str | None = None,
    ) -> None:
        self.router = router or DocumentRouter(
            provider=provider,
            max_chunk_tokens=max_chunk_tokens,
            min_chunk_tokens=min_chunk_tokens,
            agentic_provider=agentic_provider,
        )
        self._lock = threading.Lock()

    # ── Classification only (cheap, no models loaded) ─────────────────

    @staticmethod
    def classify_text(markdown: str) -> ClassificationResult:
        return classify_document(markdown)

    def classify_file(self, path: str | Path) -> ClassificationResult:
        return classify_document(self._read(Path(path)))

    # ── Chunking ──────────────────────────────────────────────────────

    def chunk_text(
        self,
        markdown: str,
        source: str = "document.md",
        doc_id: str | None = None,
    ) -> ChunkingResult:
        """Chunk one markdown string. Never raises; check ``result.success``."""
        doc_id = doc_id or self.make_doc_id(source, markdown or "")
        result = ChunkingResult(doc_id=doc_id, source=source, success=False)
        start = time.perf_counter()

        if not markdown or not markdown.strip():
            result.error = "empty document"
            return result

        try:
            with self._lock:
                chunks, classification = self.router.chunk(doc_id, source, markdown)
        except Exception as exc:  # noqa: BLE001 - surface any failure in the result
            logger.exception("chunking failed for %s", source)
            result.error = f"{type(exc).__name__}: {exc}"
            result.elapsed_s = time.perf_counter() - start
            return result

        result.elapsed_s = time.perf_counter() - start
        result.document_type = classification.document_type.value
        result.confidence = classification.confidence
        result.scores = classification.scores
        result.strategy_selected = classification.strategy.value
        result.strategy_used = (
            chunks[0].get("chunking_strategy") if chunks else ChunkingStrategy.AGENTIC.value
        )
        result.fell_back = result.strategy_used != result.strategy_selected
        result.chunks = chunks
        result.chunk_count = len(chunks)
        result.total_tokens = sum(int(c.get("token_count") or 0) for c in chunks)

        errors, warnings = self.validate(chunks)
        result.warnings = warnings
        if errors:
            result.error = "; ".join(errors)
        result.success = not errors

        logger.info(result.summary())
        return result

    def chunk_file(self, path: str | Path, doc_id: str | None = None) -> ChunkingResult:
        path = Path(path)
        try:
            markdown = self._read(path)
        except (OSError, ValueError) as exc:
            return ChunkingResult(
                doc_id=doc_id or "", source=str(path), success=False, error=str(exc)
            )
        return self.chunk_text(markdown, source=str(path), doc_id=doc_id)

    def chunk_directory(
        self, directory: str | Path, recursive: bool = False
    ) -> list[ChunkingResult]:
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        pattern = "**/*" if recursive else "*"
        files = sorted(
            p for p in directory.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        return [self.chunk_file(p) for p in files]

    async def achunk_text(
        self, markdown: str, source: str = "document.md", doc_id: str | None = None
    ) -> ChunkingResult:
        """Async wrapper — chunkers are blocking (LLM / embedding calls)."""
        return await asyncio.to_thread(self.chunk_text, markdown, source, doc_id)

    async def achunk_file(self, path: str | Path, doc_id: str | None = None) -> ChunkingResult:
        return await asyncio.to_thread(self.chunk_file, path, doc_id)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def make_doc_id(source: str, markdown: str) -> str:
        """Deterministic id: same file name + same content -> same doc_id,
        so re-ingesting an unchanged document is idempotent."""
        digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        return uuid.uuid5(uuid.NAMESPACE_URL, f"{Path(source).name}:{digest}").hex

    @staticmethod
    def validate(chunks: list[dict]) -> tuple[list[str], list[str]]:
        """Return ``(errors, warnings)`` for a chunk list.

        Errors: no chunks, missing payload keys, duplicate ids, links to
        chunk ids that don't exist. Warnings: empty-content chunks.
        """
        errors: list[str] = []
        warnings: list[str] = []
        if not chunks:
            return ["no chunks produced"], warnings

        missing = missing_keys(chunks)
        if missing:
            errors.append(f"missing payload keys (first 3): {missing[:3]}")

        ids = [c.get("chunk_id") for c in chunks]
        if len(set(ids)) != len(ids):
            errors.append("duplicate chunk_id values")

        id_set = set(ids)
        dangling = [
            (i, f) for i, c in enumerate(chunks) for f in _LINK_FIELDS
            if c.get(f) and c[f] not in id_set
        ]
        dangling += [
            (i, "sibling_chunk_ids") for i, c in enumerate(chunks)
            for sid in c.get("sibling_chunk_ids") or [] if sid not in id_set
        ]
        if dangling:
            errors.append(f"{len(dangling)} links point to unknown chunk ids")

        empty = sum(1 for c in chunks if not (c.get("content") or "").strip())
        if empty:
            warnings.append(f"{empty} chunk(s) with empty content")
        return errors, warnings

    @staticmethod
    def _read(path: Path) -> str:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"unsupported file type '{path.suffix}' (expected {SUPPORTED_EXTENSIONS}); "
                "convert to markdown first"
            )
        return path.read_text(encoding="utf-8")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify and chunk markdown documents.")
    parser.add_argument("path", help="markdown file or directory")
    parser.add_argument("--recursive", action="store_true", help="recurse into subdirectories")
    parser.add_argument("--classify-only", action="store_true", help="skip chunking")
    parser.add_argument("--provider", default=None, help="ollama | azure | aws (default: LLM_PROVIDER)")
    parser.add_argument("--out", default=None, help="write results (with chunks) to this JSON file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    target = Path(args.path)
    if target.is_dir():
        files = sorted(
            p for p in target.glob("**/*" if args.recursive else "*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        files = [target]
    if not files:
        print(f"no markdown files found in {target}", file=sys.stderr)
        return 1

    if args.classify_only:
        for f in files:
            try:
                r = ChunkingOrchestrator.classify_text(ChunkingOrchestrator._read(f))
                print(f"{f}: type={r.document_type.value} conf={r.confidence} "
                      f"strategy={r.strategy.value} scores={r.scores}")
            except (OSError, ValueError) as exc:
                print(f"{f}: FAILED ({exc})")
        return 0

    orch = ChunkingOrchestrator(provider=args.provider)
    results = [orch.chunk_file(f) for f in files]
    for r in results:
        print(r.summary())
        for w in r.warnings:
            print(f"    warning: {w}")

    if args.out:
        Path(args.out).write_text(
            json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")

    return 0 if all(r.success for r in results) else 2


if __name__ == "__main__":
    sys.exit(_main())
