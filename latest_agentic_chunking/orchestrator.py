"""IngestionOrchestrator — top-level single-file ingestion entry point.

Replaces the body of ``embed.start_embed`` while preserving its
``(success: bool, message: str)`` return shape so callers in ``api.py``
do not need to change (Req 1.2). One call per markdown document — there
is no file-level fan-out at this layer; callers loop over files
themselves.

Pipeline (Algorithm 1 in design.md):

1. Validate inputs (``user_id``, ``doc_id``, ``markdown``) and short-
   circuit on empty values without invoking the agent or touching
   Qdrant (Req 1.4 / 1.5).
2. Delegate windowing + agent chunking + assembly to ``FileIngestor``
   which returns ``(linked_chunks, fallback_chunks)``. ``fallback_chunks``
   is kept in the return shape for diagnostics but is no longer used
   for metadata generation — see step 3.
3. Set ``kb_ids = None`` on every chunk (Req 10.4). Per-chunk metadata
   generation has been removed: the retrieval stack does not consume
   the ``metadata`` payload field, so spending an LLM call per fallback
   chunk to populate it was pure cost. ``EMPTY_METADATA`` is still
   attached upstream by the validator / fallback splitter so the
   Qdrant payload schema stays stable.
4. Verify every Qdrant payload key is present (Req 10.3) before any
   provider call so a schema regression surfaces as a single clear
   error rather than a downstream upsert failure.
5. Run dense embedding, then sparse embedding (Req 14.1 / 14.2). On
   exception, return ``(False, "Error while embedding: <msg>")``
   matching the existing ``start_embed`` error string (Req 1.6).
6. Run ``Qdrant.upsert`` (Req 10.1 / 14.3). ``Qdrant.upsert`` returns
   ``(error: bool, msg: str)`` where ``error=False`` means success;
   we invert that to the orchestrator's ``(success, msg)`` convention
   (Req 10.5 / 10.6 / 1.6 / 1.7).

The orchestrator never mutates ``Qdrant.upsert``'s payload assembly —
that wrapper already builds the per-point payload dict from chunk keys
(see ``embed.Qdrant.upsert``). Our job here is to guarantee every key
``Qdrant.upsert`` reads is present on every chunk before the call.
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_chunking.file_ingestor import FileIngestor


logger = logging.getLogger(__name__)


# Keys that ``embed.Qdrant.upsert`` reads off each chunk dict when
# building the Qdrant point payload. Every chunk MUST carry every key
# before the upsert call (Req 10.3). ``chunk_id`` is the point id and
# ``embedding`` / ``sparse_embedding`` are vector data, not payload —
# they are validated implicitly by the embed stages raising on
# absence, so we don't include them in this payload-only list.
_REQUIRED_PAYLOAD_KEYS: tuple[str, ...] = (
    "doc_id",
    "source",
    "title",
    "level",
    "chunk_type",
    "primary_type",
    "has_text",
    "has_code",
    "has_table",
    "split_group_id",
    "content",
    "token_count",
    "parent_chunk_id",
    "parent_titles",
    "prev_chunk_id",
    "next_chunk_id",
    "sibling_chunk_ids",
    "kb_ids",
)


class IngestionOrchestrator:
    """Wire components together and drive end-to-end ingest for one file.

    Stateless across calls — instance fields are wiring only. Safe to
    construct once at module import time (matching ``embed.py``'s
    existing pattern for ``embedder``, ``sparse_embedder``, ``qdrant``).
    """

    def __init__(
        self,
        ingestor: FileIngestor,
        embedder: Any,
        sparse_embedder: Any,
        qdrant: Any,
        vector_size: int,
    ) -> None:
        if vector_size <= 0:
            raise ValueError(f"vector_size must be positive, got {vector_size}")

        self._ingestor = ingestor
        self._embedder = embedder
        self._sparse_embedder = sparse_embedder
        self._qdrant = qdrant
        self._vector_size = int(vector_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        user_id: str,
        doc_id: str,
        source: str,
        markdown: str,
    ) -> tuple[bool, str]:
        """Ingest one markdown document end-to-end.

        Returns ``(success, message)`` matching ``start_embed``'s shape
        (Req 1.1 / 1.2). On any failure, the message contains the
        underlying error string so callers can surface it.
        """

        # ── Step 1: Input validation (Req 1.4 / 1.5). ─────────────────
        # No agent call, no Qdrant write on bad inputs.
        print("Ingesting data")
        if not user_id:
            return False, "user_id required"
        if not doc_id:
            return False, "doc_id required"
        if markdown is None or markdown.strip() == "":
            return False, "empty document"

        # ── Step 2: Windowing + agent chunking + assembly. ────────────
        # FileIngestor handles single-window vs paragraph-segmented
        # path internally (Req 2.1 / 2.2). The second tuple element
        # (fallback_chunks) is no longer consumed for metadata
        # generation; we discard it.
        try:
            linked_chunks, _ = self._ingestor.ingest(
                doc_id=doc_id,
                source=source,
                markdown=markdown,
            )
        except Exception as exc:
            return False, f"Error while chunking: {exc}"

        if not linked_chunks:
            # Defensive — FileIngestor should always produce at least
            # one chunk for non-empty markdown (Req 2.3). If it doesn't
            # there's nothing to upsert; treat as success with a
            # diagnostic message rather than crashing the embed stage
            # on an empty list.
            return True, "no chunks"

        # ── Step 3: Set kb_ids on every chunk (Req 10.4). ─────────────
        # The ``/set-payload`` endpoint sets real values later; at
        # ingest time the key must exist with value ``None`` so the
        # Qdrant payload index over ``kb_ids`` stays consistent.
        for chunk in linked_chunks:
            chunk["kb_ids"] = None

        # ── Step 4: Verify payload schema (Req 10.3). ─────────────────
        # Catch any missing key before the embed stage so a schema
        # regression surfaces here rather than as a confusing Qdrant
        # error or KeyError deep inside the upsert.
        missing = self._first_missing_key(linked_chunks)
        if missing is not None:
            chunk_id, key = missing
            return (
                False,
                f"chunk {chunk_id} missing required payload key '{key}'",
            )

        # ── Step 5: Dense + sparse embedding (Req 14.1 / 14.2 / 1.6).
        # Match ``start_embed``'s exact error string so callers see
        # consistent messages across the old and new pipelines.
        try:
            self._embedder.embed(linked_chunks)
        except Exception as exc:
            return False, f"Error while embedding: {exc}"

        try:
            self._sparse_embedder.embed(linked_chunks)
        except Exception as exc:
            return False, f"Error while embedding: {exc}"

        # ── Step 6: Qdrant upsert (Req 10.1 / 10.5 / 10.6). ───────────
        # ``Qdrant.upsert`` returns ``(error, msg)`` with
        # ``error=False`` indicating success — invert to the
        # orchestrator's ``(success, msg)`` convention.
        try:
            error, msg = self._qdrant.upsert(
                user_id, linked_chunks, vector_size=self._vector_size
            )
        except Exception as exc:
            return False, f"Error while upserting: {exc}"

        if error:
            return False, msg
        return True, msg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_missing_key(chunks: list[dict]) -> tuple[Any, str] | None:
        """Return ``(chunk_id, missing_key)`` for the first incomplete chunk.

        Used to validate Req 10.3 before the embed stage. Returns
        ``None`` when every chunk carries every required payload key.
        """
        for chunk in chunks:
            for key in _REQUIRED_PAYLOAD_KEYS:
                if key not in chunk:
                    return chunk.get("chunk_id"), key
        return None
