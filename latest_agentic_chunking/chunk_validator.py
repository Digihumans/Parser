"""Chunk validator and auto-repair for the agentic chunking pipeline.

Implements Algorithm 4 from design.md. Pure validation — no I/O. Used by
``AgenticChunker`` after parsing the agent's tool-call response.

Validation rules (Requirements 6.1–6.11, 12.3, 15.2):

  1. Schema completeness — every chunk has ``start``, ``end``, ``title``,
     ``chunk_type``, and ``metadata`` (Req 6.1).
  2. ``chunk_type`` enum — must be one of {"text", "code", "table"} (Req 6.6).
  3. Range checks — ``0 <= start < end <= len(window.text)`` (Req 6.2).
  4. Sorted, no overlap — after sorting by ``start`` no adjacent pair
     overlaps (Req 6.3).
  5. Coverage — every non-whitespace index in
     ``[first_non_ws_idx, last_non_ws_idx + 1)`` of ``window.text`` is
     covered by some chunk's ``[start, end)`` range. Whitespace-only gaps
     are allowed (Req 6.4, 15.2). Failures emit error reasons identifying
     the uncovered index ranges.
  6. Token bound — each chunk's ``content`` token count is
     ``<= int(max_chunk_tokens * 1.1)`` (Req 6.5). Relaxed when
     ``window.oversized=True`` (Req 12.3).

Auto-repair (applied to a copy of ``raw_chunks`` before validation):

  * Trim leading/trailing whitespace from each chunk's ``[start, end)``
    range (Req 6.9).
  * Fill any missing metadata keys from ``EMPTY_METADATA`` (Req 6.7).
  * Set ``primary_type``, ``has_text``, ``has_code``, ``has_table``
    consistent with ``chunk_type`` (Req 6.8).

Postconditions:
  * ``is_valid=True`` ⇒ ``repaired_chunks`` is a non-null list satisfying
    every rule (Req 6.11).
  * ``is_valid=False`` ⇒ ``repaired_chunks is None`` and
    ``error_reasons`` is a non-empty list (Req 6.10).
  * The input ``raw_chunks`` list is never mutated; auto-repair operates
    on a deep copy. ``window.text`` is never modified (Req 15.2).
"""

from __future__ import annotations

import copy
from typing import Any

from agentic_chunking.data_models import (
    EMPTY_METADATA,
    ValidationResult,
    Window,
)
from agentic_chunking.tokenizer_factory import count_tokens

# Required top-level chunk fields. Mirrors the tool schema's required
# items list — the agent must produce these on every chunk. The
# validator's auto-repair attaches an empty ``metadata`` dict so the
# Qdrant payload schema stays stable, but the agent itself is not
# asked to generate metadata anymore.
_REQUIRED_CHUNK_KEYS: tuple[str, ...] = (
    "start",
    "end",
    "title",
    "chunk_type",
)

# Allowed chunk_type values (Req 6.6).
_VALID_CHUNK_TYPES: frozenset[str] = frozenset({"text", "code", "table"})

# 10% slack on the per-chunk token ceiling (design.md Algorithm 4).
_TOKEN_BOUND_SLACK: float = 1.1

# Max gap (chars) between adjacent chunks that we silently close by
# extending the previous chunk's end. LLMs (notably gpt-oss on Bedrock)
# drift by 1-2 chars when computing character offsets over long
# windows, dropping heading markers, list bullets, or a single
# punctuation into the gap. We keep the threshold small so real
# coverage drops still fail.
_GAP_SNAP_THRESHOLD: int = 3


class ChunkValidator:
    """Validate and auto-repair raw chunks emitted by the agent.

    Stateless — the same instance can be shared across windows and threads.
    Configuration is captured at construction so each ``validate`` call only
    needs the chunk list and the originating window.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_chunk_tokens: int,
        min_chunk_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_chunk_tokens = int(max_chunk_tokens)
        self.min_chunk_tokens = int(min_chunk_tokens)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate(
        self,
        raw_chunks: list[dict],
        window: Window,
    ) -> ValidationResult:
        """Validate ``raw_chunks`` against ``window`` and return a result.

        Does not mutate ``raw_chunks`` or ``window``. On success returns a
        ``ValidationResult`` whose ``repaired_chunks`` is a fresh list with
        auto-repairs applied. On failure ``repaired_chunks`` is ``None``.
        """

        errors: list[str] = []

        # ------------------------------------------------------------------
        # Step 1 — Schema completeness (Req 6.1). Early return on failure
        # because every later step assumes the keys are present.
        # ------------------------------------------------------------------
        for i, chunk in enumerate(raw_chunks):
            if not isinstance(chunk, dict):
                errors.append(f"chunk {i} is not an object")
                continue
            for key in _REQUIRED_CHUNK_KEYS:
                if key not in chunk:
                    errors.append(f"chunk {i} missing required field {key!r}")

        if errors:
            return ValidationResult(
                is_valid=False, error_reasons=errors, repaired_chunks=None
            )

        # ------------------------------------------------------------------
        # Step 2 — Work on a deep copy from here on so the input is
        # untouched and the chunk metadata dicts can be safely repaired.
        # ------------------------------------------------------------------
        repaired: list[dict] = copy.deepcopy(raw_chunks)
        window_text: str = window.text
        window_len: int = len(window_text)

        # ------------------------------------------------------------------
        # Step 3 — chunk_type enum check (Req 6.6).
        # ------------------------------------------------------------------
        for i, chunk in enumerate(repaired):
            if chunk.get("chunk_type") not in _VALID_CHUNK_TYPES:
                errors.append(
                    f"chunk {i} has invalid chunk_type "
                    f"{chunk.get('chunk_type')!r} (expected one of "
                    f"{sorted(_VALID_CHUNK_TYPES)})"
                )

        # ------------------------------------------------------------------
        # Step 4 — Auto-repair: trim leading/trailing whitespace from each
        # chunk's [start, end) range (Req 6.9). Done before range/coverage
        # checks so a model that produced boundaries straddling whitespace
        # can still pass. If the entire range is whitespace, leave it as-is
        # — the range/coverage checks will reject it.
        # ------------------------------------------------------------------
        for i, chunk in enumerate(repaired):
            start = chunk.get("start")
            end = chunk.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"chunk {i} start/end must be integers")
                continue
            if start < 0 or end > window_len or start >= end:
                # Defer the diagnostic to the range step below.
                continue

            slice_text = window_text[start:end]
            stripped = slice_text.strip()
            if not stripped:
                # All-whitespace range — flag in range step (start >= end
                # after trim would be the natural failure).
                continue

            # Compute trimmed offsets relative to the window.
            lead_ws = len(slice_text) - len(slice_text.lstrip())
            trail_ws = len(slice_text) - len(slice_text.rstrip())
            new_start = start + lead_ws
            new_end = end - trail_ws
            if new_start != start or new_end != end:
                chunk["start"] = new_start
                chunk["end"] = new_end
                # Keep ``content`` in sync if the agent (or earlier stage)
                # already populated it from the pre-trim slice.
                if "content" in chunk:
                    chunk["content"] = window_text[new_start:new_end]

        # ------------------------------------------------------------------
        # Step 5 — Range checks (Req 6.2).
        # ------------------------------------------------------------------
        for i, chunk in enumerate(repaired):
            start = chunk.get("start")
            end = chunk.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                # Already reported in step 4.
                continue
            if start < 0:
                errors.append(f"chunk {i} has start={start} < 0")
            if end > window_len:
                errors.append(
                    f"chunk {i} has end={end} > len(window.text)={window_len}"
                )
            if start >= end:
                errors.append(
                    f"chunk {i} has start={start} >= end={end}"
                )

        # If we already have range/type errors, the later sort+coverage
        # logic would either crash or produce confusing diagnostics — bail.
        if errors:
            return ValidationResult(
                is_valid=False, error_reasons=errors, repaired_chunks=None
            )

        # ------------------------------------------------------------------
        # Step 6 — Sort by start and check pairwise overlap (Req 6.3).
        # Sort indexes are diagnostic — refer to the sorted position, which
        # is what the model will see in feedback.
        # ------------------------------------------------------------------
        repaired.sort(key=lambda c: (c["start"], c["end"]))

        for i in range(1, len(repaired)):
            prev_end = repaired[i - 1]["end"]
            curr_start = repaired[i]["start"]
            if curr_start < prev_end:
                errors.append(
                    f"overlap: chunk {i} starts at {curr_start} but "
                    f"chunk {i - 1} ends at {prev_end}"
                )

        # ------------------------------------------------------------------
        # Step 6.5 — Auto-repair: close small non-whitespace gaps between
        # adjacent chunks. LLMs drift by 1-2 chars when computing
        # character offsets over long windows, dropping heading markers,
        # list bullets, or a single punctuation into the gap. When the
        # gap is ``<= _GAP_SNAP_THRESHOLD`` chars we extend chunk[i-1]'s
        # end to chunk[i]'s start so the coverage check passes. Larger
        # gaps fall through and still fail — those are real drops, not
        # drift. Content is re-sliced from ``window.text`` at Step 9,
        # so this repair is complete once the boundaries are updated.
        # Skipped on windows where an overlap was already recorded so we
        # don't paper over a real ordering bug.
        # ------------------------------------------------------------------
        if not errors:
            for i in range(1, len(repaired)):
                prev_end = repaired[i - 1]["end"]
                curr_start = repaired[i]["start"]
                gap = curr_start - prev_end
                if 0 < gap <= _GAP_SNAP_THRESHOLD:
                    repaired[i - 1]["end"] = curr_start

        # ------------------------------------------------------------------
        # Step 7 — Coverage (Req 6.4, 15.2). Every non-whitespace index in
        # [first_non_ws_idx, last_non_ws_idx + 1) of window.text must lie
        # inside some chunk's [start, end) range. Whitespace-only gaps are
        # allowed. Failures emit the uncovered ranges so the agent can be
        # told exactly which spans it dropped.
        # ------------------------------------------------------------------
        coverage_errors = _check_coverage(window_text, repaired)
        errors.extend(coverage_errors)

        # ------------------------------------------------------------------
        # Step 8 — Token bound (Req 6.5). Skipped when the window itself is
        # oversized (Req 12.3) — a single fenced code block / table that
        # already exceeds the budget cannot be split further so the cap is
        # relaxed entirely for that window.
        # ------------------------------------------------------------------
        if not window.oversized:
            ceiling = int(self.max_chunk_tokens * _TOKEN_BOUND_SLACK)
            for i, chunk in enumerate(repaired):
                # Always re-slice from window.text so we measure the real
                # content even if the model echoed back something stale.
                content = window_text[chunk["start"]:chunk["end"]]
                tk = count_tokens(self.tokenizer, content)
                if tk > ceiling:
                    errors.append(
                        f"chunk {i} content token count {tk} exceeds "
                        f"max_chunk_tokens*1.1={ceiling}"
                    )

        # ------------------------------------------------------------------
        # Step 9 — Auto-repair: metadata keys (Req 6.7) and consistent
        # boolean / primary_type flags (Req 6.8). Always run, even when
        # we're about to fail validation, so the repair is observable in
        # tests; the repaired list is only returned on success though.
        # ------------------------------------------------------------------
        for chunk in repaired:
            metadata = chunk.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                chunk["metadata"] = metadata
            for key, default in EMPTY_METADATA.items():
                if key not in metadata:
                    # Use a fresh copy so list defaults don't alias across
                    # chunks — empty list mutations on one chunk must not
                    # bleed into another.
                    metadata[key] = copy.deepcopy(default)

            chunk_type = chunk["chunk_type"]
            chunk["primary_type"] = chunk_type
            chunk["has_text"] = chunk_type == "text"
            chunk["has_code"] = chunk_type == "code"
            chunk["has_table"] = chunk_type == "table"

            # ``level`` is optional in the tool schema (1..6 for headings,
            # absent otherwise). Normalise to ``None`` when absent so the
            # orchestrator's payload-key check passes — Qdrant's payload
            # tolerates None level on non-heading chunks.
            if "level" not in chunk:
                chunk["level"] = None

            # ``summary`` is required by the tool schema but the
            # validator stays fail-soft: missing / non-string values
            # become an empty string so legacy responses or partial
            # tool-call recoveries still flow through. Empty summary
            # means the chunk's ``embedding_text`` will skip the
            # summary line — graceful degradation rather than a hard
            # rejection.
            summary = chunk.get("summary")
            if not isinstance(summary, str):
                summary = ""
            chunk["summary"] = summary.strip()

            # Keep ``content`` aligned with the (possibly trimmed) range so
            # downstream stages don't have to re-slice. The agent's value
            # is ignored per Req 4.3.
            chunk["content"] = window_text[chunk["start"]:chunk["end"]]

        # ------------------------------------------------------------------
        # Final verdict.
        # ------------------------------------------------------------------
        if errors:
            return ValidationResult(
                is_valid=False, error_reasons=errors, repaired_chunks=None
            )

        return ValidationResult(
            is_valid=True, error_reasons=[], repaired_chunks=repaired
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_coverage(text: str, sorted_chunks: list[dict]) -> list[str]:
    """Return error strings for any uncovered non-whitespace ranges.

    Coverage is checked over ``[first_non_ws_idx, last_non_ws_idx + 1)`` of
    ``text``. A whitespace-only gap between adjacent chunks is allowed; a
    gap that contains any non-whitespace character is reported with its
    `[start, end)` index range so the agent (or test reader) can see
    exactly what was dropped.

    The returned messages also cover content before the first chunk's
    start and after the last chunk's end inside the
    `[first_non_ws_idx, last_non_ws_idx + 1)` window.
    """

    # Find first/last non-whitespace indexes. If the window is empty or
    # all-whitespace coverage is trivially satisfied — and the orchestrator
    # short-circuits this case anyway (Req 15.3).
    first_non_ws = -1
    last_non_ws = -1
    for i, ch in enumerate(text):
        if not ch.isspace():
            if first_non_ws == -1:
                first_non_ws = i
            last_non_ws = i
    if first_non_ws == -1:
        return []

    coverage_end = last_non_ws + 1

    # If there are no chunks at all, the entire non-whitespace span is
    # uncovered. Empty chunk lists shouldn't happen on the agent path but
    # the validator must report rather than crash.
    if not sorted_chunks:
        return [
            f"uncovered non-whitespace range "
            f"[{first_non_ws}, {coverage_end}) — no chunks supplied"
        ]

    errors: list[str] = []

    # Region before the first chunk's start.
    first_start = sorted_chunks[0]["start"]
    leading_lo = first_non_ws
    leading_hi = min(first_start, coverage_end)
    if leading_lo < leading_hi:
        uncovered = _non_whitespace_subrange(text, leading_lo, leading_hi)
        if uncovered is not None:
            lo, hi = uncovered
            errors.append(
                f"uncovered non-whitespace range [{lo}, {hi}) before chunk 0"
            )

    # Gaps between adjacent chunks.
    for i in range(1, len(sorted_chunks)):
        gap_lo = sorted_chunks[i - 1]["end"]
        gap_hi = sorted_chunks[i]["start"]
        if gap_lo >= gap_hi:
            continue  # no gap (or overlap, already flagged elsewhere)
        # Clamp the inspection window to the non-whitespace span — gaps
        # beyond it cannot be uncovered by definition.
        clamped_lo = max(gap_lo, first_non_ws)
        clamped_hi = min(gap_hi, coverage_end)
        if clamped_lo >= clamped_hi:
            continue
        uncovered = _non_whitespace_subrange(text, clamped_lo, clamped_hi)
        if uncovered is not None:
            lo, hi = uncovered
            errors.append(
                f"uncovered non-whitespace range [{lo}, {hi}) "
                f"between chunk {i - 1} and chunk {i}"
            )

    # Region after the last chunk's end.
    last_end = sorted_chunks[-1]["end"]
    trailing_lo = max(last_end, first_non_ws)
    trailing_hi = coverage_end
    if trailing_lo < trailing_hi:
        uncovered = _non_whitespace_subrange(text, trailing_lo, trailing_hi)
        if uncovered is not None:
            lo, hi = uncovered
            errors.append(
                f"uncovered non-whitespace range [{lo}, {hi}) "
                f"after the last chunk"
            )

    return errors


def _non_whitespace_subrange(
    text: str, lo: int, hi: int
) -> tuple[int, int] | None:
    """Return the smallest ``[start, end)`` covering all non-ws chars in
    ``text[lo:hi]``, or ``None`` if the slice is entirely whitespace.
    """
    sub = text[lo:hi]
    stripped_left = len(sub) - len(sub.lstrip())
    stripped_right = len(sub) - len(sub.rstrip())
    if stripped_left + stripped_right >= len(sub):
        # All whitespace.
        return None
    return lo + stripped_left, hi - stripped_right
