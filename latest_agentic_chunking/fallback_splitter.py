"""FallbackSplitter — adapter around the existing MarkdownSplitter.

Invoked per-window by ``FileIngestor`` when ``AgenticChunker`` exhausts
retries on that window (Req 9.1). The adapter calls
``MarkdownSplitter.split`` on the window's text only and reshapes its
output so it slots in alongside agent-produced chunks from other windows
before ``ChunkAssembler`` runs.

Reshaping rules (Req 9.2 / 9.3 / 9.5, 14.4, 16.1, 16.2):

* Strip every linking field the rule-based splitter populated
  (``chunk_id``, ``parent_chunk_id``, ``prev_chunk_id``, ``next_chunk_id``,
  ``sibling_chunk_ids``, ``parent_titles``, ``embedding_text``,
  ``token_count``) so ``ChunkAssembler`` rebuilds them deterministically
  over the whole file.
* Replace ``metadata`` with a fresh copy of ``EMPTY_METADATA`` so the
  ``IngestionOrchestrator``'s ``MetadataGenerator`` pass can populate it
  later — agent-produced chunks already have inline metadata and skip
  that pass (Req 14.5).
* Tag every chunk ``split_group_id = window.window_index`` for
  diagnostics (Req 9.5, 16.1).
* Tag every chunk ``producer = "fallback"`` so ``FileIngestor`` can route
  fallback chunks through ``MetadataGenerator`` and agent chunks past it
  (Req 14.4, 16.2).

The adapter does not modify ``MarkdownSplitter`` itself (Req 14.4) — it
calls the existing ``split`` method unchanged.
"""

from __future__ import annotations

import copy

from agentic_chunking.data_models import EMPTY_METADATA, Window
from markdown_splitter import MarkdownSplitter


# Fields the underlying ``MarkdownSplitter`` populates that
# ``ChunkAssembler`` must rebuild over the whole file. Stripped here so
# stale per-window values cannot leak into the assembler.
_LINKING_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "parent_chunk_id",
    "prev_chunk_id",
    "next_chunk_id",
    "sibling_chunk_ids",
    "parent_titles",
    "embedding_text",
    "token_count",
)


class FallbackSplitter:
    """Per-window fallback wrapping the existing rule-based splitter.

    Returns raw chunks compatible with ``ChunkValidator`` /
    ``ChunkAssembler`` input — i.e. no linking fields and an
    empty-defaults ``metadata`` placeholder ready for the
    ``MetadataGenerator`` pass on fallback chunks only.
    """

    def __init__(self, splitter: MarkdownSplitter) -> None:
        self.splitter = splitter

    def split_window(
        self,
        window: Window,
        doc_id: str,
        source: str,
    ) -> list[dict]:
        """Run the rule-based splitter on this window's text only.

        ``MarkdownSplitter.split`` is called with ``window.text`` rather
        than the full document; ``ChunkAssembler`` is responsible for
        stitching this window's chunks into the file's overall chunk
        graph using ``window.window_index`` and the corresponding
        window's ``leading_heading_path``.
        """
        chunks = self.splitter.split(doc_id, source, window.text)

        for chunk in chunks:
            # Drop linking fields so ``ChunkAssembler`` rebuilds them.
            for field in _LINKING_FIELDS:
                chunk.pop(field, None)

            # Per-chunk deepcopy so the inner list defaults
            # (keywords / topics / entities / references / questions /
            # action_items) are independent across chunks — mutating
            # one chunk's metadata must not bleed into another's.
            chunk["metadata"] = copy.deepcopy(EMPTY_METADATA)

            # Normalise the type flags ``MarkdownSplitter`` does not
            # set. ``ChunkValidator`` does this for agent chunks; we
            # mirror it here so the orchestrator's payload-key check
            # passes for fallback chunks too. Schema parity with the
            # retrieval stack (Req 10.3).
            chunk_type = chunk.get("chunk_type", "text")
            chunk["chunk_type"] = chunk_type
            chunk["primary_type"] = chunk_type
            chunk["has_text"] = chunk_type == "text"
            chunk["has_code"] = chunk_type == "code"
            chunk["has_table"] = chunk_type == "table"

            # Ensure ``level`` is present (None for non-heading chunks).
            # ``MarkdownSplitter`` sets this on every chunk, but be
            # defensive so the orchestrator's payload-key check never
            # trips on a fallback chunk.
            if "level" not in chunk:
                chunk["level"] = None
            if "title" not in chunk:
                chunk["title"] = None

            # Fallback chunks have no LLM-generated summary. Setting
            # to empty string preserves schema parity with agent
            # chunks; ``ChunkAssembler`` skips the summary line in
            # ``embedding_text`` when this is empty (graceful
            # degradation, no retrieval lift on fallback chunks).
            chunk["summary"] = ""

            # Diagnostic tag tying the chunk back to its source window.
            chunk["split_group_id"] = window.window_index

            # Producer tag so ``FileIngestor`` / ``IngestionOrchestrator``
            # can route fallback chunks through ``MetadataGenerator``
            # while skipping agent-produced chunks.
            chunk["producer"] = "fallback"

        return chunks
