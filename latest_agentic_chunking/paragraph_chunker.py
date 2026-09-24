"""ParagraphChunker — paragraph-ID variant of ``AgenticChunker``.

Motivation
----------
``AgenticChunker`` asks the LLM to emit **character** ``start``/``end``
offsets. That works reliably for gpt-5-mini (Azure) but fails on
weaker AWS Bedrock models (DeepSeek, GLM, Mistral, gpt-oss) — those
models cannot count characters accurately over multi-thousand-char
windows and their tool calls end up with wrong offsets, triggering
the fallback splitter on almost every window.

``ParagraphChunker`` sidesteps character arithmetic entirely:

1. The window text is pre-split into paragraphs, each with a stable
   integer ID and its exact ``[start_offset, end_offset)`` inside the
   window.
2. The LLM sees the window as ``[P0] ... / [P1] ... / [P2] ...`` and
   emits *paragraph* ranges, not character ranges. Grouping labeled
   items is a task every capable LLM handles well.
3. Post-processing converts each ``(start_paragraph, end_paragraph)``
   tuple back into character offsets by looking up the paragraph
   positions we already know. The existing ``ChunkValidator`` then
   runs unchanged — the resulting boundaries are guaranteed to sit on
   paragraph edges so coverage / overlap checks pass trivially.

The public interface (``__init__`` signature, ``chunk_window``) is
byte-for-byte identical to ``AgenticChunker`` so ``FileIngestor``
treats the two chunkers interchangeably. Selection lives in
``embed.py`` via the ``CHUNKER_STRATEGY`` env variable.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from agentic_chunking.chunk_validator import ChunkValidator
from agentic_chunking.data_models import (
    EMPTY_METADATA,
    AgenticChunkerError,
    Window,
)
# Reuse helpers that are shape-agnostic — same retry heuristic and
# oversized-window handling apply regardless of tool schema.
from agentic_chunking.agentic_chunker import (
    _is_retryable_provider_error,
    _AZURE_ENDPOINT,
    _AZURE_API_KEY,
    _AZURE_API_VERSION,
    _AZURE_FAST_DEPLOYMENT_DEFAULT,
    _OLLAMA_CHAT_MODEL_DEFAULT,
    _OLLAMA_NUM_CTX,
    AgenticChunker,
)
from aws_helpers import (
    AWS_MODEL_ID as _AWS_MODEL_ID_DEFAULT,
    openai_tools_to_aws_config,
    extract_aws_tool_use,
    default_additional_fields,
)


load_dotenv()

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Paragraph unit
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class Paragraph:
    """A pre-computed paragraph within a window.

    ``start_offset`` / ``end_offset`` are character offsets into
    ``window.text``. ``text`` is exactly ``window.text[start:end]`` so
    callers can render or hash the paragraph without re-slicing.
    """

    paragraph_id: int
    text: str
    start_offset: int
    end_offset: int


# Character-per-token approximation used by the merger. cl100k averages
# ~4 chars/token on English prose, closer to 3 on markdown-heavy text.
# We use 4 to stay conservative — better to merge slightly less than
# to accidentally exceed the chunker's token cap.
_CHARS_PER_TOKEN_ESTIMATE: int = 4

# How many multiples of ``min_chunk_tokens`` a merged paragraph should
# reach before we stop growing it. Higher values → fewer, chunkier
# labelled units for the LLM to reason about. On website content 2x
# is where DeepSeek's chunk count starts matching gpt-5-mini's
# char-offset baseline — enough merging to eliminate fragment noise,
# not so much that we destroy semantic boundaries.
_MIN_PARAGRAPH_MULTIPLIER: int = 2


def _is_atomic_paragraph(text: str) -> bool:
    """Return True for paragraphs that must NOT be merged with
    neighbours: fenced code blocks and markdown tables. Merging these
    into adjacent prose would destroy the structural signal downstream
    consumers rely on (chunk_type inference, has_code / has_table
    flags, table reconstruction).
    """
    stripped = text.lstrip()
    if stripped.startswith("```"):
        return True
    non_empty = [l.strip() for l in text.splitlines() if l.strip()]
    if not non_empty:
        return False
    pipe_lines = sum(1 for l in non_empty if l.startswith("|"))
    # If at least half the non-empty lines start with |, treat as
    # table-like and keep atomic.
    if pipe_lines * 2 >= len(non_empty):
        return True
    return False


def _split_paragraphs(
    text: str,
    min_paragraph_chars: int = 800,
    max_merged_chars: int = 4800,
) -> list[Paragraph]:
    """Split ``text`` into paragraphs preserving fenced code blocks
    atomic AND merging small consecutive paragraphs so the LLM sees
    semantically meaningful units instead of hundreds of website
    fragments.

    Blank lines are the primary structural boundary. Inside a fenced
    code block (triple backticks) blank lines are treated as content
    rather than boundaries.

    Parameters
    ----------
    min_paragraph_chars:
        Merge threshold. A paragraph shorter than this is absorbed
        into its right-hand neighbour (or previous, if it's last)
        until the group clears the bar. Larger values = fewer,
        chunkier labelled units for the LLM to reason about. Sized
        by ``ParagraphChunker`` from the chunker's own
        ``min_chunk_tokens`` × chars/token ratio.
    max_merged_chars:
        Cap on any merged group's total size. Prevents runaway
        merging that would create a single blob bigger than the
        chunker's downstream token cap. Sized by ``ParagraphChunker``
        from ``max_chunk_tokens``.

    Atomic blocks — fenced code and tables — are never merged with
    prose neighbours so ``chunk_type`` inference downstream stays
    correct.

    Returns paragraphs with strict, byte-accurate offsets so downstream
    boundary reconstruction is exact.
    """
    paragraphs: list[Paragraph] = []
    lines = text.splitlines(keepends=True)
    i = 0
    n = len(lines)
    offset = 0

    while i < n:
        # Skip inter-paragraph whitespace-only lines.
        while i < n and not lines[i].strip():
            offset += len(lines[i])
            i += 1
        if i >= n:
            break

        para_start = offset

        # A paragraph that OPENS with a code fence stays atomic until
        # the matching closing fence — blank lines inside a fenced
        # block are content, not boundaries.
        opens_with_fence = lines[i].lstrip().startswith("```")
        in_fence = opens_with_fence

        # Consume the first line of the paragraph unconditionally.
        offset += len(lines[i])
        i += 1

        while i < n:
            line = lines[i]
            stripped = line.lstrip()

            # Blank line ends a NORMAL paragraph. Inside a fence blank
            # lines are treated as content.
            if not line.strip() and not in_fence:
                break

            if stripped.startswith("```"):
                if in_fence:
                    # Closing fence — include it and end the paragraph.
                    offset += len(line)
                    i += 1
                    in_fence = False
                    break
                # A fence marker in the middle of a prose paragraph
                # indicates a new fenced block is starting — end the
                # current paragraph here so the code block becomes its
                # own paragraph on the next iteration.
                break

            offset += len(line)
            i += 1

        paragraphs.append(Paragraph(
            paragraph_id=len(paragraphs),
            text=text[para_start:offset],
            start_offset=para_start,
            end_offset=offset,
        ))

    # Sub-split any raw paragraph that's already bigger than the
    # chunker's token cap. Without this, the LLM would make an
    # oversized chunk from a single paragraph and the validator
    # would reject on the token-bound check — no retry can save it
    # because the paragraph itself is atomic to the LLM.
    paragraphs = _split_oversized_paragraphs(
        paragraphs, text, max_merged_chars,
    )

    return _merge_small_paragraphs(
        paragraphs, text, min_paragraph_chars, max_merged_chars,
    )


def _split_oversized_paragraphs(
    paragraphs: list[Paragraph],
    source_text: str,
    max_chars: int,
) -> list[Paragraph]:
    """Split any paragraph exceeding ``max_chars`` into line-aligned
    sub-paragraphs. Preserves byte-accurate offsets. Atomic blocks
    (fenced code, tables) are left intact — splitting a code block
    would garble its semantics and downstream ``chunk_type``
    inference; better to accept the overflow than to destroy the
    structural signal.

    Uses line boundaries rather than sentence boundaries because
    markdown mixes prose, lists, headings, and images — line-based
    splitting cuts cleanly at those transitions and never breaks
    inline markup like ``[text](url)`` or ``**bold**``.
    """
    result: list[Paragraph] = []
    for p in paragraphs:
        if len(p.text) <= max_chars or _is_atomic_paragraph(p.text):
            result.append(Paragraph(
                paragraph_id=len(result),
                text=p.text,
                start_offset=p.start_offset,
                end_offset=p.end_offset,
            ))
            continue

        # Line-based sub-split. Accumulate lines into a buffer, flush
        # when adding the next line would exceed the cap.
        lines = p.text.splitlines(keepends=True)
        sub_start = p.start_offset
        acc: list[str] = []
        acc_len = 0

        def flush(sub_start_local: int, chunk_lines: list[str]) -> int:
            if not chunk_lines:
                return sub_start_local
            body = "".join(chunk_lines)
            sub_end_local = sub_start_local + len(body)
            result.append(Paragraph(
                paragraph_id=len(result),
                text=body,
                start_offset=sub_start_local,
                end_offset=sub_end_local,
            ))
            return sub_end_local

        for line in lines:
            # Empty accumulator — always take at least one line so
            # we make forward progress even for lines longer than
            # the cap (rare, but possible for long paste-in URLs).
            if acc and acc_len + len(line) > max_chars:
                sub_start = flush(sub_start, acc)
                acc = [line]
                acc_len = len(line)
            else:
                acc.append(line)
                acc_len += len(line)

        flush(sub_start, acc)

    return result


def _merge_small_paragraphs(
    paragraphs: list[Paragraph],
    source_text: str,
    min_paragraph_chars: int,
    max_merged_chars: int,
) -> list[Paragraph]:
    """Merge adjacent tiny paragraphs so the LLM sees fewer, more
    meaningful units. Runs on the raw split output.

    Rules:
      * A paragraph shorter than ``min_paragraph_chars`` is absorbed
        into its right-hand neighbour (or left-hand if it's the last).
      * Fenced code blocks and tables are ATOMIC — never merged, and
        merging never crosses one of them.
      * Merging stops once the accumulated group meets the minimum,
        so we don't end up with a single giant blob.
      * Runaway is guarded by ``max_merged_chars``: if adding the
        next paragraph would exceed the cap, start a new group even
        if the current one hasn't hit the minimum.

    Merged paragraphs cover a contiguous span of ``source_text`` from
    the first sub-paragraph's ``start_offset`` to the last's
    ``end_offset``. The ``text`` field is re-sliced from
    ``source_text`` so inter-paragraph blank lines get included —
    that's what the LLM sees rendered, and it matches what the
    validator's coverage check operates on.
    """
    if not paragraphs:
        return paragraphs

    groups: list[list[Paragraph]] = []
    for p in paragraphs:
        atomic = _is_atomic_paragraph(p.text)

        if not groups:
            groups.append([p])
            continue

        current = groups[-1]
        current_atomic = _is_atomic_paragraph(current[-1].text) or any(
            _is_atomic_paragraph(sub.text) for sub in current
        )

        # Never merge across an atomic boundary (code fence or table
        # on either side). Keeps chunk_type inference honest.
        if atomic or current_atomic:
            groups.append([p])
            continue

        current_span = current[-1].end_offset - current[0].start_offset
        # Already big enough — start a new group.
        if current_span >= min_paragraph_chars:
            groups.append([p])
            continue

        # Would exceed the cap — start a new group defensively.
        prospective_span = p.end_offset - current[0].start_offset
        if prospective_span > max_merged_chars:
            groups.append([p])
            continue

        current.append(p)

    # Second pass: absorb a trailing tiny group into the previous one
    # if we have more than one group (avoids a lone tiny last paragraph
    # being its own labelled unit). Skip if either group is atomic.
    if len(groups) >= 2:
        last = groups[-1]
        prev = groups[-2]
        last_span = last[-1].end_offset - last[0].start_offset
        last_atomic = any(_is_atomic_paragraph(p.text) for p in last)
        prev_atomic = any(_is_atomic_paragraph(p.text) for p in prev)
        if (
            last_span < min_paragraph_chars
            and not last_atomic
            and not prev_atomic
            and (last[-1].end_offset - prev[0].start_offset)
                <= max_merged_chars
        ):
            prev.extend(last)
            groups.pop()

    merged: list[Paragraph] = []
    for group in groups:
        first = group[0]
        last = group[-1]
        # Slice from the source so blank lines between sub-paragraphs
        # are preserved verbatim — the LLM's rendered view should
        # reflect the original whitespace structure.
        span_text = source_text[first.start_offset:last.end_offset]
        merged.append(Paragraph(
            paragraph_id=len(merged),
            text=span_text,
            start_offset=first.start_offset,
            end_offset=last.end_offset,
        ))
    return merged


# ──────────────────────────────────────────────
#  Tool schema (paragraph-ID variant)
# ──────────────────────────────────────────────

PARAGRAPH_CHUNK_TOOL_SCHEMA: dict = {
    "name": "extract_chunks_by_paragraph",
    "description": (
        "Group the labeled paragraphs shown in the user message into "
        "semantically coherent chunks. Each chunk covers a contiguous "
        "range of paragraph IDs [start_paragraph..end_paragraph]. "
        "Every paragraph in the window MUST be covered exactly once, "
        "with no overlaps and in ascending order."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chunks": {
                "type": "array",
                "description": (
                    "Ordered list of chunks. The first chunk starts at "
                    "paragraph 0; the last chunk ends at the "
                    "highest-numbered paragraph."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "start_paragraph": {
                            "type": "integer",
                            "description": (
                                "First paragraph ID in this chunk "
                                "(inclusive). Must be ≥ 0 and ≤ "
                                "end_paragraph."
                            ),
                        },
                        "end_paragraph": {
                            "type": "integer",
                            "description": (
                                "Last paragraph ID in this chunk "
                                "(inclusive). Must be ≥ start_paragraph "
                                "and ≤ (total_paragraphs - 1)."
                            ),
                        },
                        "title": {
                            "type": "string",
                            "description": (
                                "Specific, content-reflective title for "
                                "the chunk. Name the entities, topics, "
                                "sections, dates, metrics, or events "
                                "present. Do NOT use generic "
                                "document-level labels like 'Full "
                                "document', 'Full resume', 'Content', "
                                "'Section', or 'Chunk'. 5-12 words."
                            ),
                        },
                        "summary": {
                            "type": "string",
                            "description": (
                                "1-2 sentence summary of the chunk's "
                                "specific content. MUST mention concrete "
                                "entities, names, numbers, dates, or "
                                "domain terms found in the chunk. Do "
                                "NOT use generic phrasing like 'this "
                                "chunk discusses' or 'an overview of'. "
                                "Aim for 15-40 words."
                            ),
                        },
                        "chunk_type": {
                            "type": "string",
                            "enum": ["text", "code", "table"],
                            "description": (
                                "Dominant content type of the chunk: "
                                "prose, fenced code block, or markdown "
                                "table."
                            ),
                        },
                    },
                    "required": [
                        "start_paragraph", "end_paragraph",
                        "title", "summary", "chunk_type",
                    ],
                },
            }
        },
        "required": ["chunks"],
    },
}


PARAGRAPH_SYSTEM_PROMPT: str = (
    "You MUST call the tool extract_chunks_by_paragraph.\n"
    "Do NOT respond with text.\n"
    "Do NOT return JSON in the message body.\n"
    "Only call the tool, exactly once.\n"
    "\n"
    "You will receive a window of markdown pre-split into paragraphs. "
    "Each paragraph is labeled with [P<N>] at its start where N is a "
    "unique integer ID starting from 0. Multi-line paragraphs (like "
    "tables or code blocks) have the label on the first line only.\n"
    "\n"
    "Your job: group the paragraphs into semantically coherent chunks. "
    "Return a list of chunks; each chunk is a contiguous range of "
    "paragraph IDs specified by (start_paragraph, end_paragraph) "
    "INCLUSIVE.\n"
    "\n"
    "Grouping rules:\n"
    "- Every paragraph MUST be covered by exactly one chunk.\n"
    "- Chunks MUST cover paragraphs in ascending order.\n"
    "- Chunks MUST NOT overlap (chunk k's end_paragraph must be less "
    "than chunk k+1's start_paragraph).\n"
    "- The first chunk MUST start at paragraph 0 and the last chunk "
    "MUST end at the highest-numbered paragraph. No paragraph may be "
    "left out.\n"
    "- A chunk can be a single paragraph (start == end) or many "
    "paragraphs.\n"
    "- Prefer keeping semantically related paragraphs together — "
    "same heading section, same topic, same table.\n"
    "- Break at natural boundaries: heading changes, topic shifts, "
    "new sections, transitions between prose and tables/code.\n"
    "- Never split a table across chunks — keep all rows together "
    "as one chunk when they share a table.\n"
    "- Never split a fenced code block across chunks.\n"
    "\n"
    "Chunk size: each chunk should target between {min_tokens} and "
    "{max_tokens} tokens of content. If a semantic section is longer "
    "than max_tokens, split it into multiple chunks at paragraph "
    "boundaries (never mid-paragraph).\n"
    "\n"
    "Title rules (the `title` field for each chunk):\n"
    "- Title MUST reflect the chunk's actual content — name the "
    "specific entities, topics, sections, dates, metrics, or events "
    "present.\n"
    "- NEVER use generic document-level titles like 'Full document', "
    "'Full resume', 'Full report', 'Section', 'Chunk'.\n"
    "- If the chunk covers a whole short document, the title MUST "
    "still be specific. Bad: 'Full resume'. Good: 'Mahesh Kumar "
    "resume — software engineer at Acme'.\n"
    "- Aim for 5-12 words.\n"
    "\n"
    "Summary rules (the `summary` field for each chunk):\n"
    "- Summary is a 1-2 sentence description of WHAT IS IN THE CHUNK. "
    "It is used as the retrieval embedding anchor.\n"
    "- Mention concrete entities, names, numbers, dates, products, "
    "or domain terms found in the chunk. Lead with the entity or "
    "topic, not with framing words.\n"
    "- Aim for 15-40 words."
)


# AWS Converse toolConfig built once at import time.
_AWS_TOOL_CONFIG_PARAGRAPH: dict = openai_tools_to_aws_config(
    [PARAGRAPH_CHUNK_TOOL_SCHEMA],
    force_tool_name=PARAGRAPH_CHUNK_TOOL_SCHEMA["name"],
)

_TOOL_SPEC_PARAGRAPH: dict = {
    "type": "function", "function": PARAGRAPH_CHUNK_TOOL_SCHEMA,
}
_FORCED_TOOL_CHOICE_PARAGRAPH: dict = {
    "type": "function",
    "function": {"name": PARAGRAPH_CHUNK_TOOL_SCHEMA["name"]},
}


# ──────────────────────────────────────────────
#  ParagraphChunker
# ──────────────────────────────────────────────

class ParagraphChunker:
    """Drop-in replacement for ``AgenticChunker`` using paragraph IDs.

    Same public interface: ``__init__`` accepts the same arguments and
    ``chunk_window(window, doc_id, source)`` returns raw chunks
    compatible with ``ChunkAssembler``. ``FileIngestor`` never has to
    know which chunker is in use.
    """

    def __init__(
        self,
        provider: str,
        model: str | None,
        max_chunk_tokens: int,
        min_chunk_tokens: int,
        max_retries: int = 3,
        *,
        rate_limit_semaphore: threading.Semaphore,
        validator: ChunkValidator,
    ) -> None:
        normalized = (provider or "").strip().lower()
        if normalized not in ("azure", "ollama", "aws"):
            raise ValueError(
                f"Unknown provider: {provider!r}. "
                f"Expected 'azure', 'ollama', or 'aws'."
            )
        self.provider: str = normalized
        self.max_chunk_tokens: int = int(max_chunk_tokens)
        self.min_chunk_tokens: int = int(min_chunk_tokens)
        self.max_retries: int = int(max_retries)
        self.rate_limit_semaphore: threading.Semaphore = rate_limit_semaphore
        self.validator: ChunkValidator = validator

        if self.max_retries <= 0:
            raise ValueError(
                f"max_retries must be positive, got {self.max_retries}"
            )

        # Paragraph merging thresholds derived from the chunker's own
        # config so labelled units approach chunk-sized targets. The
        # LLM's job then becomes "map paragraph groups to chunks",
        # not "find useful semantic boundaries in a sea of website
        # fragments". Tune-able via env variables for pathological
        # docs — the derived defaults work for typical prose and
        # website markdown.
        #
        # Default derivation:
        #   min = min_chunk_tokens × chars_per_tok × 2  (aggressive
        #     enough that website fragment noise gets collapsed —
        #     validated against digihumans_ai.md where this brings
        #     DeepSeek's chunk count from ~100 to ~12).
        #   max = max_chunk_tokens × chars_per_tok        (chunker
        #     token ceiling, never exceeded by pre-merger.)
        _default_min = (
            self.min_chunk_tokens
            * _CHARS_PER_TOKEN_ESTIMATE
            * _MIN_PARAGRAPH_MULTIPLIER
        )
        _default_max = self.max_chunk_tokens * _CHARS_PER_TOKEN_ESTIMATE
        self.min_paragraph_chars: int = int(
            os.getenv("PARAGRAPH_MIN_CHARS", str(_default_min))
        )
        self.max_merged_paragraph_chars: int = int(
            os.getenv("PARAGRAPH_MAX_MERGED_CHARS", str(_default_max))
        )

        # Resolve default model per provider — mirrors AgenticChunker.
        if model is None:
            if self.provider == "azure":
                self.model: str = _AZURE_FAST_DEPLOYMENT_DEFAULT
            elif self.provider == "aws":
                self.model = _AWS_MODEL_ID_DEFAULT
            else:
                self.model = _OLLAMA_CHAT_MODEL_DEFAULT
        else:
            self.model = model

        # Lazy-init client — same lazy-import pattern as AgenticChunker
        # so a deployment that uses only one provider doesn't pay the
        # import cost of the others.
        if self.provider == "azure":
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=_AZURE_ENDPOINT,
                api_key=_AZURE_API_KEY,
                api_version=_AZURE_API_VERSION,
            )
        elif self.provider == "aws":
            import boto3
            # No region_name — resolves from AWS env / profile chain.
            self.client = boto3.client("bedrock-runtime", region_name="ap-south-1")
        else:
            import ollama
            self.client = ollama.Client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_window(
        self,
        window: Window,
        doc_id: str,
        source: str,
    ) -> list[dict]:
        """Run the paragraph-ID agent on ``window`` and return validated
        raw chunks. Signature and semantics match
        ``AgenticChunker.chunk_window``.
        """
        # Oversized window short-circuit — delegate to AgenticChunker's
        # existing single-chunk builder so behaviour matches.
        if window.oversized:
            return [AgenticChunker._build_oversized_chunk(
                window, doc_id, source,
            )]

        paragraphs = _split_paragraphs(
            window.text,
            min_paragraph_chars=self.min_paragraph_chars,
            max_merged_chars=self.max_merged_paragraph_chars,
        )
        if not paragraphs:
            # Empty / whitespace-only window — orchestrator upstream
            # already short-circuits this case, but be defensive.
            return []

        feedback: str = ""

        for attempt in range(self.max_retries):
            messages = self._build_messages(paragraphs, feedback)

            try:
                response = self._call_provider(messages)
            except Exception as exc:
                error_str = str(exc)
                if _is_retryable_provider_error(error_str):
                    reasons = [f"provider HTTP error: {error_str[:200]}"]
                    self._log_retry(window, attempt, reasons)
                    feedback = (
                        "The previous request failed with a provider "
                        "error (rate limit / 5xx). Try again."
                    )
                    self._sleep_for_attempt(attempt)
                    continue
                raise

            tool_args_obj = self._extract_tool_arguments(response)
            if tool_args_obj is None:
                reasons = [f"model did not call {PARAGRAPH_CHUNK_TOOL_SCHEMA['name']}"]
                self._log_retry(window, attempt, reasons)
                feedback = (
                    "You did not call the tool. You MUST call "
                    f"{PARAGRAPH_CHUNK_TOOL_SCHEMA['name']}."
                )
                self._sleep_for_attempt(attempt)
                continue

            parsed = self._parse_tool_arguments(tool_args_obj)
            if parsed is None or not isinstance(parsed, dict):
                reasons = ["tool arguments were not valid JSON"]
                self._log_retry(window, attempt, reasons)
                feedback = (
                    "Your tool arguments were not valid JSON. Return a "
                    "single tool call whose arguments are a JSON object "
                    "with a 'chunks' array."
                )
                self._sleep_for_attempt(attempt)
                continue

            raw_groupings = parsed.get("chunks")
            if not isinstance(raw_groupings, list) or not raw_groupings:
                reasons = ["tool arguments missing 'chunks' array"]
                self._log_retry(window, attempt, reasons)
                feedback = (
                    "Your tool arguments must include a non-empty "
                    "'chunks' array."
                )
                self._sleep_for_attempt(attempt)
                continue

            prepared, prep_error = self._prepare_chunks(
                raw_groupings, paragraphs, window, doc_id, source,
            )
            if prep_error is not None:
                self._log_retry(window, attempt, [prep_error])
                feedback = (
                    f"Your chunk boundaries were invalid: {prep_error}. "
                    "Every paragraph must be covered exactly once, in "
                    "ascending order, with no overlaps."
                )
                self._sleep_for_attempt(attempt)
                continue

            validation = self.validator.validate(prepared, window)
            if validation.is_valid:
                final = validation.repaired_chunks or prepared
                for c in final:
                    c["producer"] = "agent"
                return final

            reasons = list(validation.error_reasons)
            self._log_retry(window, attempt, reasons)
            feedback = (
                "Your chunk boundaries were invalid: " + "; ".join(reasons)
            )
            self._sleep_for_attempt(attempt)

        raise AgenticChunkerError(
            f"max retries reached for window {window.window_index} "
            f"(paragraph strategy)"
        )

    # ------------------------------------------------------------------
    # Provider call
    # ------------------------------------------------------------------

    def _call_provider(self, messages: list[dict]) -> Any:
        """Wrap the provider call with the rate-limit semaphore."""
        self.rate_limit_semaphore.acquire()
        try:
            if self.provider == "azure":
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=[_TOOL_SPEC_PARAGRAPH],
                    tool_choice=_FORCED_TOOL_CHOICE_PARAGRAPH,
                )
            if self.provider == "aws":
                system_text = messages[0]["content"]
                user_text = messages[1]["content"]
                return self.client.converse(
                    modelId=self.model,
                    system=[{"text": system_text}],
                    messages=[{
                        "role": "user",
                        "content": [{"text": user_text}],
                    }],
                    inferenceConfig={
                        "temperature": 0.2,
                        "maxTokens": 16000,
                    },
                    toolConfig=_AWS_TOOL_CONFIG_PARAGRAPH,
                    additionalModelRequestFields=default_additional_fields(),
                )
            # Ollama
            return self.client.chat(
                model=self.model,
                messages=messages,
                tools=[_TOOL_SPEC_PARAGRAPH],
                think=False,
                options={"num_ctx": _OLLAMA_NUM_CTX, "temperature": 0.4},
            )
        finally:
            self.rate_limit_semaphore.release()

    # ------------------------------------------------------------------
    # Tool-call extraction
    # ------------------------------------------------------------------

    def _extract_tool_arguments(self, response: Any) -> Any:
        try:
            if self.provider == "azure":
                choice = response.choices[0]
                tool_calls = getattr(choice.message, "tool_calls", None)
                if not tool_calls:
                    return None
                return tool_calls[0].function.arguments
            if self.provider == "aws":
                content_blocks = (
                    response.get("output", {})
                    .get("message", {})
                    .get("content", [])
                )
                tool_use = extract_aws_tool_use(content_blocks)
                if tool_use is None:
                    return None
                return tool_use.get("input")
            # Ollama
            tool_calls = response.message.tool_calls
            if not tool_calls:
                return None
            return tool_calls[0].function.arguments
        except (IndexError, AttributeError, KeyError):
            return None

    @staticmethod
    def _parse_tool_arguments(args_obj: Any) -> Any:
        if isinstance(args_obj, dict):
            return args_obj
        if isinstance(args_obj, str):
            try:
                return json.loads(args_obj)
            except (json.JSONDecodeError, ValueError):
                return None
        if hasattr(args_obj, "model_dump"):
            try:
                value = args_obj.model_dump()
                return value if isinstance(value, dict) else None
            except Exception:  # noqa: BLE001
                return None
        return None

    # ------------------------------------------------------------------
    # Paragraph-ID → char offset conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_chunks(
        raw_groupings: list[Any],
        paragraphs: list[Paragraph],
        window: Window,
        doc_id: str,
        source: str,
    ) -> tuple[list[dict], str | None]:
        """Convert paragraph-ID groupings into char-offset chunks.

        Returns ``(prepared, error)``. On any structural problem
        (wrong type, out-of-range IDs, overlap, gap, missing coverage)
        returns ``([], reason)`` so the caller retries with feedback.

        The validator downstream re-checks all boundaries, but doing a
        structural pass here lets us give the model precise feedback
        in the retry message ("chunk 2 skips paragraph 5") rather
        than the validator's char-offset-oriented diagnostics.
        """
        if not raw_groupings:
            return [], "chunks list was empty"

        total = len(paragraphs)
        prepared: list[dict] = []
        prev_end = -1

        for i, group in enumerate(raw_groupings):
            if not isinstance(group, dict):
                return [], f"chunk {i} is not an object"

            start_p = group.get("start_paragraph")
            end_p = group.get("end_paragraph")

            if not isinstance(start_p, int) or not isinstance(end_p, int):
                return [], (
                    f"chunk {i}: start_paragraph and end_paragraph must "
                    "be integers"
                )
            if start_p < 0 or end_p >= total:
                return [], (
                    f"chunk {i}: paragraph range [{start_p}..{end_p}] "
                    f"outside [0..{total - 1}]"
                )
            if start_p > end_p:
                return [], (
                    f"chunk {i}: start_paragraph {start_p} > "
                    f"end_paragraph {end_p}"
                )
            if start_p != prev_end + 1:
                return [], (
                    f"chunk {i}: start_paragraph {start_p} does not "
                    f"follow the previous chunk (expected "
                    f"{prev_end + 1}). Cover paragraphs in ascending "
                    "contiguous order with no gaps or overlaps."
                )
            prev_end = end_p

            first = paragraphs[start_p]
            last = paragraphs[end_p]
            start_char = first.start_offset
            end_char = last.end_offset

            chunk_type = group.get("chunk_type", "text")
            if chunk_type not in ("text", "code", "table"):
                chunk_type = "text"

            chunk: dict = {
                "start": start_char,
                "end": end_char,
                "title": group.get("title", "") or "",
                "summary": group.get("summary", "") or "",
                "level": None,
                "chunk_type": chunk_type,
                "primary_type": chunk_type,
                "has_text": chunk_type == "text",
                "has_code": chunk_type == "code",
                "has_table": chunk_type == "table",
                "content": window.text[start_char:end_char],
                # Fresh deep copy so mutable list defaults don't alias
                # across chunks — same precaution AgenticChunker takes.
                "metadata": copy.deepcopy(EMPTY_METADATA),
                "split_group_id": window.window_index,
                "doc_id": doc_id,
                "source": source,
            }
            prepared.append(chunk)

        # Completeness — every paragraph must be covered.
        if prev_end != total - 1:
            return [], (
                f"chunks stop at paragraph {prev_end} but the window "
                f"has {total} paragraphs. The last chunk must end at "
                f"paragraph {total - 1}."
            )

        return prepared, None

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        paragraphs: list[Paragraph],
        feedback: str,
    ) -> list[dict]:
        sys_parts: list[str] = [
            PARAGRAPH_SYSTEM_PROMPT.format(
                min_tokens=self.min_chunk_tokens,
                max_tokens=self.max_chunk_tokens,
            )
        ]
        if feedback:
            sys_parts.append(feedback)

        rendered = self._render_paragraphs(paragraphs)
        total = len(paragraphs)

        user_content = (
            f"Window contains {total} paragraphs numbered [P0]..[P{total - 1}].\n"
            f"Return start_paragraph / end_paragraph (INCLUSIVE) integer "
            f"IDs. Every paragraph MUST be covered by exactly one chunk.\n"
            f"--- BEGIN WINDOW PARAGRAPHS ---\n"
            f"{rendered}\n"
            f"--- END WINDOW PARAGRAPHS ---"
        )

        return [
            {"role": "system", "content": "\n\n".join(sys_parts)},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _render_paragraphs(paragraphs: list[Paragraph]) -> str:
        """Render paragraphs as ``[P<N>] <text>`` blocks separated by
        blank lines. Multi-line paragraphs preserve their internal
        newlines so tables and code blocks render structurally.
        """
        parts: list[str] = []
        for p in paragraphs:
            # Preserve internal newlines but strip any trailing
            # whitespace so the block boundaries stay clean.
            body = p.text.rstrip()
            parts.append(f"[P{p.paragraph_id}] {body}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Retry helpers
    # ------------------------------------------------------------------

    def _log_retry(
        self,
        window: Window,
        attempt: int,
        error_reasons: list[str],
    ) -> None:
        # Mirror AgenticChunker: warn via logger AND print for
        # stdout visibility even when the root logger isn't
        # configured. Same message shape so log parsers can consume
        # both chunker strategies uniformly.
        msg = (
            f"ParagraphChunker retry: window_index={window.window_index} "
            f"attempt={attempt + 1} reasons={error_reasons}"
        )
        logger.warning(msg)
        print(msg)

    @staticmethod
    def _sleep_for_attempt(attempt: int) -> None:
        time.sleep(2 ** attempt)
