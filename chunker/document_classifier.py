"""Upfront markdown document classification + chunker routing.

``classify_document`` is a cheap, rule-based classifier (no LLM, no
embeddings). It computes structural features from the raw markdown and
scores five document types. ``DocumentRouter`` then sends the document to
the matching chunker; the agentic chunker is the fallback.

Routing table
-------------
    TECHNICAL_CODE   -> adaptive   (code fences kept whole, heading tree)
    TABULAR          -> adaptive   (tables split by rows, header repeated)
    STRUCTURED       -> adaptive   (healthy heading hierarchy)
    SHORT            -> adaptive   (fits in ~one chunk, no need for an LLM)
    NARRATIVE_PROSE  -> semantic   (few headings, long paragraphs)
    FLAT_NOISY       -> agentic    (headings used as plain lines, fragments,
                                    e.g. PDF-converted resumes / slides)
    UNKNOWN          -> agentic    (low classifier confidence)

Fallback: if the selected chunker raises, returns no chunks, or returns
chunks missing required payload keys, the document is re-chunked with the
agentic chunker. Agentic itself falls back per window to the rule-based
splitter inside ``agentic_chunking.FileIngestor``.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chunker._schema import missing_keys

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Types
# ──────────────────────────────────────────────

class DocumentType(str, Enum):
    TECHNICAL_CODE = "technical_code"
    TABULAR = "tabular"
    STRUCTURED = "structured"
    NARRATIVE_PROSE = "narrative_prose"
    FLAT_NOISY = "flat_noisy"
    SHORT = "short"
    UNKNOWN = "unknown"


class ChunkingStrategy(str, Enum):
    ADAPTIVE = "adaptive"
    SEMANTIC = "semantic"
    AGENTIC = "agentic"


ROUTING: dict[DocumentType, ChunkingStrategy] = {
    DocumentType.TECHNICAL_CODE: ChunkingStrategy.ADAPTIVE,
    DocumentType.TABULAR: ChunkingStrategy.ADAPTIVE,
    DocumentType.STRUCTURED: ChunkingStrategy.ADAPTIVE,
    DocumentType.SHORT: ChunkingStrategy.ADAPTIVE,
    DocumentType.NARRATIVE_PROSE: ChunkingStrategy.SEMANTIC,
    DocumentType.FLAT_NOISY: ChunkingStrategy.AGENTIC,
    DocumentType.UNKNOWN: ChunkingStrategy.AGENTIC,
}


@dataclass
class MarkdownFeatures:
    total_words: int = 0
    nonblank_lines: int = 0
    heading_count: int = 0
    heading_levels: int = 0
    code_line_ratio: float = 0.0      # lines inside ``` fences / nonblank lines
    table_line_ratio: float = 0.0     # table rows / nonblank lines
    list_line_ratio: float = 0.0      # bullet / numbered lines / body lines
    short_line_ratio: float = 0.0     # body lines with <= 3 words / body lines
    empty_section_ratio: float = 0.0  # headings with < 8 body words / headings
    median_section_words: float = 0.0
    words_per_heading: float = 0.0
    avg_paragraph_words: float = 0.0  # prose paragraphs only


@dataclass
class ClassificationResult:
    document_type: DocumentType
    strategy: ChunkingStrategy
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    features: MarkdownFeatures = field(default_factory=MarkdownFeatures)


# ──────────────────────────────────────────────
#  Tunables
# ──────────────────────────────────────────────

SHORT_DOC_WORDS = 150          # below this the whole doc is ~one chunk
MIN_CONFIDENCE = 0.5           # below this -> UNKNOWN -> agentic
CODE_RATIO_FULL = 0.25         # code_line_ratio that scores 1.0
TABLE_RATIO_FULL = 0.30
EMPTY_SECTION_FULL = 0.5
SHORT_LINE_FULL = 0.3
SECTION_WORDS_FULL = 60        # median words per section for a "real" section
PARAGRAPH_WORDS_FULL = 60      # avg words per prose paragraph
WORDS_PER_HEADING_FULL = 300   # prose docs have few headings per word

_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# Table separator row, e.g. "|---|:--:|" or "--- | ---". Must contain a pipe so
# that markdown horizontal rules ("---") are not counted as table lines.
_TABLE_SEP_RE = re.compile(r"^(?=.*\|)\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_WORD_RE = re.compile(r"\w+")


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _words(text: str) -> int:
    return len(_WORD_RE.findall(text))


# ──────────────────────────────────────────────
#  Feature extraction
# ──────────────────────────────────────────────

def extract_features(markdown: str) -> MarkdownFeatures:
    f = MarkdownFeatures()
    lines = markdown.splitlines()

    in_code = False
    code_lines = table_lines = list_lines = short_lines = body_lines = 0
    levels: set[int] = set()
    section_words: list[int] = []      # body words under each heading
    current_section: int | None = None
    paragraphs: list[int] = []         # word counts of prose paragraphs
    para_words = 0

    def end_paragraph() -> None:
        nonlocal para_words
        if para_words:
            paragraphs.append(para_words)
        para_words = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            end_paragraph()
            continue
        f.nonblank_lines += 1

        if line.startswith("```"):
            end_paragraph()
            code_lines += 1
            in_code = not in_code
            continue
        if in_code:
            code_lines += 1
            words = _words(line)
            f.total_words += words
            if current_section is not None:
                section_words[current_section] += words
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            end_paragraph()
            f.heading_count += 1
            levels.add(len(heading.group(1)))
            section_words.append(0)
            current_section = len(section_words) - 1
            continue

        words = _words(line)
        f.total_words += words
        if current_section is not None:
            section_words[current_section] += words

        if line.startswith("|") or _TABLE_SEP_RE.match(line):
            end_paragraph()
            table_lines += 1
            continue

        body_lines += 1
        if _LIST_RE.match(line):
            end_paragraph()
            list_lines += 1
            continue
        if words <= 3:
            short_lines += 1
        para_words += words

    end_paragraph()

    n = max(f.nonblank_lines, 1)
    b = max(body_lines, 1)
    f.code_line_ratio = code_lines / n
    f.table_line_ratio = table_lines / n
    f.list_line_ratio = list_lines / b
    f.short_line_ratio = short_lines / b if body_lines else 0.0
    f.heading_levels = len(levels)
    if section_words:
        f.empty_section_ratio = sum(1 for w in section_words if w < 8) / len(section_words)
        f.median_section_words = float(statistics.median(section_words))
    f.words_per_heading = f.total_words / max(f.heading_count, 1)
    f.avg_paragraph_words = (sum(paragraphs) / len(paragraphs)) if paragraphs else 0.0
    return f


# ──────────────────────────────────────────────
#  Classification
# ──────────────────────────────────────────────

def _score(f: MarkdownFeatures) -> dict[DocumentType, float]:
    code = _clamp(f.code_line_ratio / CODE_RATIO_FULL)
    table = _clamp(f.table_line_ratio / TABLE_RATIO_FULL)

    flat = 0.0
    if f.heading_count >= 3:
        flat = max(
            _clamp(f.empty_section_ratio / EMPTY_SECTION_FULL),
            _clamp(f.short_line_ratio / SHORT_LINE_FULL),
        )

    structured = 0.0
    if f.heading_count >= 2:
        structured = (
            (1.0 - f.empty_section_ratio)
            * _clamp(f.median_section_words / SECTION_WORDS_FULL)
            * (1.0 - f.short_line_ratio)
        )

    prose = (
        _clamp(f.avg_paragraph_words / PARAGRAPH_WORDS_FULL)
        * _clamp(f.words_per_heading / WORDS_PER_HEADING_FULL)
        * (1.0 - f.list_line_ratio)
        * (1.0 - code)
        * (1.0 - table)
    )

    # Insertion order = tie-break priority.
    return {
        DocumentType.TECHNICAL_CODE: code,
        DocumentType.TABULAR: table,
        DocumentType.FLAT_NOISY: flat,
        DocumentType.STRUCTURED: structured,
        DocumentType.NARRATIVE_PROSE: prose,
    }


def classify_document(markdown: str) -> ClassificationResult:
    """Classify a markdown document and pick its chunking strategy."""
    f = extract_features(markdown)

    if f.total_words < SHORT_DOC_WORDS:
        doc_type = DocumentType.SHORT
        return ClassificationResult(doc_type, ROUTING[doc_type], 1.0, {}, f)

    scores = _score(f)
    best_type = max(scores, key=lambda t: scores[t])  # first max wins ties
    confidence = scores[best_type]
    doc_type = best_type if confidence >= MIN_CONFIDENCE else DocumentType.UNKNOWN

    return ClassificationResult(
        document_type=doc_type,
        strategy=ROUTING[doc_type],
        confidence=round(confidence, 3),
        scores={t.value: round(s, 3) for t, s in scores.items()},
        features=f,
    )


# ──────────────────────────────────────────────
#  Router
# ──────────────────────────────────────────────

class DocumentRouter:
    """Classify a document and chunk it with the matching chunker.

    Chunkers are built lazily and cached, so a deployment that never sees
    prose docs never loads the Ollama embedding client, etc. Token budgets
    default to ``agentic_chunking.config.build_config(provider)`` so all
    strategies produce comparably sized chunks.
    """

    def __init__(
        self,
        provider: str | None = None,
        max_chunk_tokens: int | None = None,
        min_chunk_tokens: int | None = None,
        chunker_kwargs: dict[ChunkingStrategy, dict[str, Any]] | None = None,
        agentic_provider: str | None = None,
    ) -> None:
        from agentic_chunking.config import LLM_PROVIDER, build_config

        # ``provider`` drives the rule-based chunkers' tokenizer + budgets.
        # The agentic chunker has its own provider: ``agentic_provider`` or
        # CHUNKER_PROVIDER from .env (resolved inside AgenticChunker).
        self.provider = (provider or LLM_PROVIDER).strip().lower()
        self.agentic_provider = agentic_provider
        cfg = build_config(self.provider)
        self.max_chunk_tokens = max_chunk_tokens or cfg.max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens or cfg.min_chunk_tokens
        self.chunker_kwargs = chunker_kwargs or {}
        self._chunkers: dict[ChunkingStrategy, Any] = {}

    def _get_chunker(self, strategy: ChunkingStrategy) -> Any:
        if strategy in self._chunkers:
            return self._chunkers[strategy]

        extra = self.chunker_kwargs.get(strategy, {})
        if strategy is ChunkingStrategy.ADAPTIVE:
            from chunker.adaptive_chunker.chunk import AdaptiveChunker

            chunker = AdaptiveChunker(
                provider=self.provider,
                max_chunk_tokens=self.max_chunk_tokens,
                min_chunk_tokens=self.min_chunk_tokens,
                **extra,
            )
        elif strategy is ChunkingStrategy.SEMANTIC:
            from chunker.semantic_chunker.chunk import SemanticChunker

            chunker = SemanticChunker(
                provider=self.provider,
                max_chunk_tokens=self.max_chunk_tokens,
                min_chunk_tokens=self.min_chunk_tokens,
                **extra,
            )
        else:
            from chunker.agentic_chunker.chunk import AgenticChunker

            chunker = AgenticChunker(provider=self.agentic_provider, **extra)

        self._chunkers[strategy] = chunker
        return chunker

    def chunk(
        self, doc_id: str, source: str, markdown: str
    ) -> tuple[list[dict], ClassificationResult]:
        """Return ``(chunks, classification)``.

        Every chunk gets ``document_type`` and ``chunking_strategy`` keys
        (the strategy actually used, i.e. "agentic" after a fallback).
        """
        if not markdown or not markdown.strip():
            raise ValueError("empty document")

        result = classify_document(markdown)
        strategy = result.strategy
        logger.info(
            "classified doc_id=%s type=%s confidence=%.3f strategy=%s",
            doc_id, result.document_type.value, result.confidence, strategy.value,
        )

        chunks: list[dict] | None = None
        if strategy is not ChunkingStrategy.AGENTIC:
            try:
                chunks = self._get_chunker(strategy).chunk(doc_id, source, markdown)
                problem = (
                    "no chunks" if not chunks
                    else f"missing keys {missing_keys(chunks)[:3]}" if missing_keys(chunks)
                    else None
                )
            except Exception as exc:  # noqa: BLE001 - any failure -> fallback
                problem = f"{type(exc).__name__}: {exc}"
            if problem:
                logger.warning(
                    "%s chunker failed for doc_id=%s (%s); falling back to agentic",
                    strategy.value, doc_id, problem,
                )
                chunks, strategy = None, ChunkingStrategy.AGENTIC

        if chunks is None:
            chunks = self._get_chunker(ChunkingStrategy.AGENTIC).chunk(doc_id, source, markdown)

        for chunk in chunks:
            chunk["document_type"] = result.document_type.value
            chunk["chunking_strategy"] = strategy.value

        return chunks, result
