"""Data models for the agentic chunking pipeline.

Mirrors the Data Models section of the design document exactly:
- ``Window`` is the unit handed to the agent (one LLM call per window).
- ``RawChunk`` is the agent's tool-call response after slicing.
- ``ChunkMetadata`` is the inline metadata the agent emits per chunk.
- ``ValidationResult`` is what ``ChunkValidator.validate`` returns.
- ``EMPTY_METADATA`` is the canonical empty-defaults metadata dict
  (used by the validator's auto-repair and by ``FallbackSplitter``).
- ``AgenticChunkerError`` is raised when the agent exhausts retries on
  a window so the caller can route to the fallback splitter.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

@dataclass
class Window:
    """An ordered slice of the source markdown handed to the agent.

    Field shapes match the design's Data Models section. ``oversized`` is set
    by ``ParagraphPreSegmenter`` when a single fenced code block or table
    exceeds ``agent_context_budget_tokens`` on its own; downstream components
    (``AgenticChunker``, ``ChunkValidator``) relax token bounds for such
    windows and emit a single window-spanning chunk.
    """

    text: str
    start_offset: int           # char offset in original markdown
    end_offset: int             # char offset in original markdown
    window_index: int           # 0-based, used in chunk_id derivation
    token_count: int
    leading_heading_path: list[str]   # H1 -> Hn chain at window start
    oversized: bool = False


# ---------------------------------------------------------------------------
# ChunkMetadata
# ---------------------------------------------------------------------------

@dataclass
class ChunkMetadata:
    """Per-chunk metadata emitted inline by the agent tool call.

    Same shape as the existing ``MetadataGenerator`` output so the Qdrant
    payload schema does not change. All eight keys are required; the
    validator fills missing keys from ``EMPTY_METADATA`` during auto-repair.
    """

    summary: str
    keywords: list[str]
    topics: list[str]
    entities: list[str]
    references: list[str]
    questions: list[str]
    document_type: str
    action_items: list[str]


# ---------------------------------------------------------------------------
# RawChunk
# ---------------------------------------------------------------------------

@dataclass
class RawChunk:
    """Agent tool response after offset-slicing, before assembly.

    ``content`` is set by ``AgenticChunker`` from ``window.text[start:end]``;
    any ``content`` value returned by the model is ignored. ``split_group_id``
    is set to ``window.window_index`` and is diagnostic only — it does not
    drive sibling grouping in ``ChunkAssembler``.
    """

    start: int                  # char offset within window
    end: int                    # char offset within window
    title: str | None
    summary: str                # 1-2 sentence summary emitted by the agent;
                                # empty string for fallback / oversized chunks
    level: int | None           # 1..6 for headings, None otherwise
    chunk_type: str             # "text" | "code" | "table"
    primary_type: str
    has_text: bool
    has_code: bool
    has_table: bool
    content: str
    metadata: ChunkMetadata
    split_group_id: str         # = str(window.window_index), diagnostic only


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of ``ChunkValidator.validate``.

    On failure ``repaired_chunks`` is ``None``; on success it is a non-null
    list satisfying every validation rule (with auto-repairs applied).
    """

    is_valid: bool
    error_reasons: list[str]
    repaired_chunks: list[dict] | None


# ---------------------------------------------------------------------------
# Empty metadata constant
# ---------------------------------------------------------------------------

# Canonical empty-defaults metadata dict used for two purposes:
#   1. ChunkValidator auto-repair fills missing metadata keys from this.
#   2. FallbackSplitter seeds each fallback chunk's metadata with a copy of
#      this so the IngestionOrchestrator's MetadataGenerator pass can populate
#      it later (Req 9.3 / 9.4).
#
# Consumers MUST copy this dict (e.g. ``copy.deepcopy(EMPTY_METADATA)`` or
# manual key-by-key copy) before mutating per-chunk to avoid accidental
# cross-chunk aliasing of the inner list defaults.
EMPTY_METADATA: dict = {
    "summary": "",
    "keywords": [],
    "topics": [],
    "entities": [],
    "references": [],
    "questions": [],
    "document_type": "",
    "action_items": [],
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AgenticChunkerError(Exception):
    """Raised by ``AgenticChunker.chunk_window`` when retries are exhausted.

    The caller (``FileIngestor``) catches this and routes the offending
    window to ``FallbackSplitter`` so one bad window does not abort the
    rest of the file.
    """
