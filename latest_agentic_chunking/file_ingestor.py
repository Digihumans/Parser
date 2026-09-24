"""FileIngestor — top-level per-file driver for the agentic chunking pipeline.

Implements Algorithm 1 from ``design.md``. Selects between the
single-window path (small files) and the paragraph-segmented path
(large files), runs the agent across windows in parallel via a
``ThreadPoolExecutor``, routes failed windows to the rule-based
``FallbackSplitter``, and hands off ``(raw_chunks_per_window, windows)``
to ``ChunkAssembler`` for cross-window stitching.

Design highlights
-----------------

* **Submission-order alignment.** ``futures`` are iterated in submission
  order (``[future.result() for future in futures]``) so
  ``raw_chunks_per_window[i]`` always corresponds to ``windows[i]``
  regardless of which thread completed first (Req 8.2). This is what
  makes parallelism a pure execution detail — the chunk graph is
  identical to a sequential run (Req 7.11).

* **Single-window path still routes through the executor.** Per the
  design, even when ``len(windows) == 1`` the call goes through the
  ``ThreadPoolExecutor``; the executor collapses to a sequential call
  but the code path stays uniform (Req 8.4).

* **Per-window fallback.** When ``AgenticChunker.chunk_window`` raises
  ``AgenticChunkerError`` for a single window, only that window is
  re-run through ``FallbackSplitter`` — the other windows are left
  untouched. We never abort the whole file because of one bad window
  (Req 9.1).

* **Fallback subset extraction.** The second tuple element returned by
  ``ingest`` is the subset of *linked* chunks (post-assembly) whose
  ``producer == "fallback"``. The assembler's deepcopy preserves the
  ``producer`` field set by ``AgenticChunker`` and ``FallbackSplitter``,
  so we read it directly from the linked list — no chunk_id tracking
  needed.

* **Producer tagging is upstream.** ``AgenticChunker`` already tags its
  chunks ``producer="agent"`` and ``FallbackSplitter`` already tags its
  chunks ``producer="fallback"`` (Req 16.2). ``FileIngestor`` does not
  re-tag — it only consumes the field when filtering the fallback
  subset for the orchestrator's ``MetadataGenerator`` pass (Req 9.4 /
  14.5).

* **Fallback-transition logging.** Per Req 16.4, when a window falls
  back the ingestor emits a log entry with ``(window_index,
  agent_attempts_before_fallback, AgenticChunkerError message)``.
  ``agent_attempts_before_fallback`` is read from
  ``chunker.max_retries`` (the cap reached when the chunker raised) —
  ``AgenticChunker`` always exhausts the full retry budget before
  raising, so this is the actual attempt count.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agentic_chunking.agentic_chunker import AgenticChunker
from agentic_chunking.chunk_assembler import ChunkAssembler
from agentic_chunking.data_models import AgenticChunkerError, Window
from agentic_chunking.fallback_splitter import FallbackSplitter
from agentic_chunking.paragraph_pre_segmenter import ParagraphPreSegmenter
from agentic_chunking.tokenizer_factory import count_tokens


logger = logging.getLogger(__name__)


class FileIngestor:
    """Drives chunking for one markdown file end-to-end (excluding embedding).

    Stateless across calls — instance fields are wiring only. Safe to
    share one ``FileIngestor`` across the whole process; concurrency
    happens inside ``ingest`` via the per-call ``ThreadPoolExecutor``.
    """

    def __init__(
        self,
        chunker: AgenticChunker,
        pre_segmenter: ParagraphPreSegmenter,
        fallback: FallbackSplitter,
        assembler: ChunkAssembler,
        tokenizer: Any,
        paragraph_threshold_tokens: int,
        agent_context_budget_tokens: int,
        max_window_workers: int = 4,
    ) -> None:
        if paragraph_threshold_tokens <= 0:
            raise ValueError(
                "paragraph_threshold_tokens must be positive, "
                f"got {paragraph_threshold_tokens}"
            )
        if agent_context_budget_tokens <= 0:
            raise ValueError(
                "agent_context_budget_tokens must be positive, "
                f"got {agent_context_budget_tokens}"
            )
        if max_window_workers <= 0:
            raise ValueError(
                f"max_window_workers must be positive, got {max_window_workers}"
            )

        self._chunker = chunker
        self._pre_segmenter = pre_segmenter
        self._fallback = fallback
        self._assembler = assembler
        self._tokenizer = tokenizer
        self._paragraph_threshold_tokens = int(paragraph_threshold_tokens)
        self._agent_context_budget_tokens = int(agent_context_budget_tokens)
        self._max_window_workers = int(max_window_workers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        doc_id: str,
        source: str,
        markdown: str,
    ) -> tuple[list[dict], list[dict]]:
        """Run windowing + agent chunking + assembly for one document.

        Args:
            doc_id: Document identifier propagated to every chunk.
            source: Source path / filename propagated to every chunk.
            markdown: The full markdown source. Empty / whitespace-only
                input is the orchestrator's responsibility to handle
                (Req 1.4 / 15.3); ``ingest`` itself accepts any input
                the pre-segmenter accepts.

        Returns:
            ``(linked_chunks, fallback_chunks_needing_metadata)`` where:

            * ``linked_chunks`` is the full assembled chunk list ready
              for embedding + Qdrant upsert (chunks tagged
              ``producer="agent"`` or ``producer="fallback"``).
            * ``fallback_chunks_needing_metadata`` is the subset of
              ``linked_chunks`` with ``producer == "fallback"``;
              ``IngestionOrchestrator`` runs ``MetadataGenerator`` over
              this list before embedding (agent-produced chunks already
              carry inline metadata — Req 14.5).

            Both lists share chunk-dict references — mutating a chunk
            in ``fallback_chunks_needing_metadata`` mutates it in
            ``linked_chunks`` too. This is intentional so the
            orchestrator's metadata pass updates the linked list in
            place.
        """
        windows = self._build_windows(markdown)

        # Always route through the executor — even for one window — so
        # the code path is uniform between the single-window and
        # paragraph-segmented cases (Req 8.4). The thread pool collapses
        # to a sequential call when ``len(windows) == 1``.
        raw_chunks_per_window: list[list[dict]] = []
        with ThreadPoolExecutor(max_workers=self._max_window_workers) as executor:
            futures = [
                executor.submit(
                    self._chunk_window_with_fallback, window, doc_id, source
                )
                for window in windows
            ]
            # Iterate in submission order so window[i] aligns with
            # raw_chunks_per_window[i] regardless of completion order
            # (Req 8.2). ``future.result()`` blocks until that specific
            # future is done, which gives us deterministic ordering
            # without sacrificing parallelism — workers still run
            # concurrently.
            for future in futures:
                raw_chunks_per_window.append(future.result())

        linked_chunks = self._assembler.assemble(raw_chunks_per_window, windows)

        # Phase A.0 of ChunkAssembler deepcopies each input chunk, so
        # the ``producer`` field set upstream by AgenticChunker /
        # FallbackSplitter is preserved on the linked list. Filter for
        # the orchestrator's MetadataGenerator pass.
        fallback_chunks_needing_metadata = [
            ch for ch in linked_chunks if ch.get("producer") == "fallback"
        ]

        return linked_chunks, fallback_chunks_needing_metadata

    # ------------------------------------------------------------------
    # Windowing strategy (Req 2.1 / 2.2 / 2.4)
    # ------------------------------------------------------------------

    def _build_windows(self, markdown: str) -> list[Window]:
        """Choose single-window or paragraph-segmented path.

        Small-file path: a single Window covering the whole document
        with ``window_index=0`` and an empty ``leading_heading_path``
        (Req 2.4). Large-file path: delegate to
        ``ParagraphPreSegmenter.segment``.
        """
        total_tokens = count_tokens(self._tokenizer, markdown)
        if total_tokens <= self._paragraph_threshold_tokens:
            return [
                Window(
                    text=markdown,
                    start_offset=0,
                    end_offset=len(markdown),
                    window_index=0,
                    token_count=total_tokens,
                    leading_heading_path=[],
                    oversized=False,
                )
            ]
        return self._pre_segmenter.segment(markdown)

    # ------------------------------------------------------------------
    # Per-window worker (runs inside the ThreadPoolExecutor)
    # ------------------------------------------------------------------

    def _chunk_window_with_fallback(
        self,
        window: Window,
        doc_id: str,
        source: str,
    ) -> list[dict]:
        """Run the agent on one window, falling back on retry exhaustion.

        Catches ``AgenticChunkerError`` (Req 9.1) and routes only that
        window to ``FallbackSplitter``. Other exceptions propagate so a
        truly broken setup (auth failure, missing client) surfaces
        rather than being papered over by the rule-based fallback.
        """
        try:
            return self._chunker.chunk_window(window, doc_id, source)
        except AgenticChunkerError as exc:
            self._log_fallback_transition(window, exc)
            return self._fallback.split_window(window, doc_id, source)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_fallback_transition(
        self,
        window: Window,
        error: AgenticChunkerError,
    ) -> None:
        """Emit the fallback log entry mandated by Req 16.4.

        ``agent_attempts_before_fallback`` is read from
        ``chunker.max_retries``: ``AgenticChunker.chunk_window`` only
        raises ``AgenticChunkerError`` after exhausting the full retry
        budget, so this is the actual attempt count for this window.
        """
        agent_attempts = getattr(self._chunker, "max_retries", None)
        logger.warning(
            "FileIngestor fallback: window_index=%d "
            "agent_attempts_before_fallback=%s error=%s",
            window.window_index,
            agent_attempts,
            str(error),
        )
