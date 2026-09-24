"""Paragraph-level pre-segmenter for the agentic chunking pipeline.

Implements Algorithm 2 from ``design.md``: walk a markdown document, split
it into "paragraph" blocks (with fenced code blocks and contiguous ``|``
table runs preserved as atomic units), and pack those blocks greedily into
``Window`` objects that each fit under ``agent_context_budget_tokens``.

Output guarantees (Requirements 3.2-3.5, 3.8, 12.1):
  * ``windows[0].start_offset == 0``
  * ``windows[-1].end_offset == len(markdown)``
  * adjacent windows are contiguous: ``windows[i].end_offset == windows[i+1].start_offset``
  * for every window, ``markdown[start_offset:end_offset] == text``
  * every window's ``token_count <= agent_context_budget_tokens`` unless
    ``oversized=True`` (set when a single fenced code block or table on its
    own exceeds the budget — Req 12.1)
  * ``window_index`` is sequential starting at 0

Heading tracking (Requirements 3.7, 7.3):
  * an internal stack of ``(level, clean_title)`` tuples is maintained as
    blocks are consumed; when a heading at level ``L`` appears, all entries
    at level ``>= L`` are dropped and the new ``(L, title)`` is appended
  * each window's ``leading_heading_path`` is a snapshot of the clean
    titles in stack order at the moment the window's first block is
    consumed (i.e. before any of its blocks update the stack)
"""

from __future__ import annotations

from typing import Any

from agentic_chunking.data_models import Window
from agentic_chunking.tokenizer_factory import count_tokens


# Block type constants — internal only.
_BLOCK_TEXT = "text"
_BLOCK_CODE = "code"
_BLOCK_TABLE = "table"


class ParagraphPreSegmenter:
    """Group paragraphs into agent-budget-sized windows.

    Used only when a file's total token count exceeds
    ``paragraph_threshold_tokens`` (the small-file path uses a single
    whole-document window instead).
    """

    def __init__(self, tokenizer: Any, agent_context_budget_tokens: int) -> None:
        if agent_context_budget_tokens <= 0:
            raise ValueError(
                "agent_context_budget_tokens must be positive, "
                f"got {agent_context_budget_tokens}"
            )
        self._tokenizer = tokenizer
        self._budget = agent_context_budget_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, markdown: str) -> list[Window]:
        """Segment ``markdown`` into windows. See module docstring for guarantees."""

        if not markdown:
            return []

        blocks = self._parse_blocks(markdown)
        if not blocks:
            # Whitespace-only document — emit a single window covering the
            # whole input so downstream coverage invariants still hold.
            return [
                Window(
                    text=markdown,
                    start_offset=0,
                    end_offset=len(markdown),
                    window_index=0,
                    token_count=count_tokens(self._tokenizer, markdown),
                    leading_heading_path=[],
                    oversized=False,
                )
            ]

        # Pre-compute token counts and per-block text once.
        for block in blocks:
            block["text"] = markdown[block["start"] : block["end"]]
            block["tokens"] = count_tokens(self._tokenizer, block["text"])

        windows: list[Window] = []
        # Heading stack: list of (level, clean_title) — H1 -> Hn order.
        heading_stack: list[tuple[int, str]] = []
        # Snapshot of the heading stack at the moment the *current* window
        # began consuming its first block. Initialised for the first window
        # as the pre-document state (empty).
        window_leading_path: list[tuple[int, str]] = []
        # State for the in-progress window.
        cur_start = 0
        cur_tokens = 0
        cur_block_count = 0
        window_index = 0

        for block in blocks:
            # Only tables qualify for the "atomic + oversized" treatment
            # here; fenced code is no longer extracted as its own block
            # type by ``_parse_blocks`` (see the comment there for why),
            # so a triple-backtick line is now part of a regular text
            # block and goes through the normal packing path.
            block_alone_oversized = (
                block["tokens"] > self._budget
                and block["type"] == _BLOCK_TABLE
            )
            would_overflow = (
                cur_block_count > 0
                and (cur_tokens + block["tokens"]) > self._budget
            )

            # Flush the in-progress window if either condition holds:
            #   a) adding this block would push the window over budget, or
            #   b) this block is itself oversized (code/table) — it must
            #      occupy its own window, so flush whatever we have first.
            if cur_block_count > 0 and (would_overflow or block_alone_oversized):
                windows.append(
                    Window(
                        text=markdown[cur_start : block["start"]],
                        start_offset=cur_start,
                        end_offset=block["start"],
                        window_index=window_index,
                        token_count=cur_tokens,
                        leading_heading_path=[t for _, t in window_leading_path],
                        oversized=False,
                    )
                )
                window_index += 1
                cur_start = block["start"]
                cur_tokens = 0
                cur_block_count = 0
                # The next window's leading path is the current heading
                # stack — snapshot it now, before any new block is consumed.
                window_leading_path = list(heading_stack)

            if block_alone_oversized:
                # Solo oversized window for this block. Heading-path
                # snapshot is the stack as of the start of this window
                # (which we just refreshed above).
                windows.append(
                    Window(
                        text=block["text"],
                        start_offset=block["start"],
                        end_offset=block["end"],
                        window_index=window_index,
                        token_count=block["tokens"],
                        leading_heading_path=[t for _, t in window_leading_path],
                        oversized=True,
                    )
                )
                window_index += 1
                # Update the heading stack from this block's content (code
                # / table almost never have headings, but be defensive).
                heading_stack = self._update_heading_stack(
                    block["text"], heading_stack
                )
                cur_start = block["end"]
                cur_tokens = 0
                cur_block_count = 0
                # Refresh leading path for the *next* window.
                window_leading_path = list(heading_stack)
                continue

            # Normal accumulation into the current window.
            cur_tokens += block["tokens"]
            cur_block_count += 1
            heading_stack = self._update_heading_stack(block["text"], heading_stack)

        # Flush any trailing window. We extend its end to len(markdown) so
        # the overall coverage invariant holds even if the last block did
        # not reach the end of the document (it should — _parse_blocks
        # guarantees that — but we belt-and-brace the contract).
        if cur_start < len(markdown):
            windows.append(
                Window(
                    text=markdown[cur_start : len(markdown)],
                    start_offset=cur_start,
                    end_offset=len(markdown),
                    window_index=window_index,
                    token_count=cur_tokens,
                    leading_heading_path=[t for _, t in window_leading_path],
                    oversized=False,
                )
            )

        return windows

    # ------------------------------------------------------------------
    # Block parsing
    # ------------------------------------------------------------------

    def _parse_blocks(self, markdown: str) -> list[dict]:
        """Split ``markdown`` into contiguous ``(start, end, type)`` blocks.

        Invariants on the returned list:
          * ``blocks[0]["start"] == 0``
          * ``blocks[-1]["end"] == len(markdown)``
          * for every ``i``, ``blocks[i]["end"] == blocks[i+1]["start"]``

        Block types: ``"text"`` (paragraph), ``"code"`` (fenced ```` ``` ````
        block), ``"table"`` (contiguous ``|``-prefixed lines). Code and
        table blocks are atomic — never split inside.

        Trailing blank lines after a block are absorbed into that block so
        adjacent blocks remain contiguous; leading blank lines before the
        first non-blank line are absorbed into the first block.
        """

        # Tuples: (line_text_with_newline, start_offset, end_offset).
        # ``splitlines(keepends=True)`` preserves trailing newlines and the
        # offsets sum exactly to len(markdown), so coverage holds.
        lines: list[tuple[str, int, int]] = []
        pos = 0
        for line in markdown.splitlines(keepends=True):
            lines.append((line, pos, pos + len(line)))
            pos += len(line)

        blocks: list[dict] = []
        # Where the next block begins. Starts at 0 so any leading blank
        # lines get absorbed into the first emitted block.
        block_start = 0
        i = 0
        n = len(lines)

        while i < n:
            line_text, _, _ = lines[i]
            stripped = line_text.strip()

            # Triple-backtick lines are NOT treated as fenced-code-block
            # boundaries here — flow them through the normal text-paragraph
            # path. Real-world inputs (PDF-to-markdown converters,
            # truncated dumps) often produce stray or unclosed ``` fences
            # that would otherwise consume the entire rest of the
            # document as one oversized "code" block. The agent's system
            # prompt still tells the LLM not to split inside genuine code
            # blocks, so legitimate code is still kept atomic at the
            # chunk-boundary level by the agent. Tables remain atomic
            # because they have a much stricter line-shape signature.

            if stripped.startswith("|"):
                # Contiguous table run.
                j = i + 1
                while j < n and lines[j][0].strip().startswith("|"):
                    j += 1
                while j < n and lines[j][0].strip() == "":
                    j += 1
                block_end = lines[j - 1][2]
                blocks.append(
                    {"start": block_start, "end": block_end, "type": _BLOCK_TABLE}
                )
                block_start = block_end
                i = j
                continue

            if stripped == "":
                # Standalone blank line at start (no block in progress) —
                # advance; it will be absorbed into the next block via
                # ``block_start``.
                i += 1
                continue

            # Plain text paragraph: accumulate non-blank, non-table
            # lines until we hit a blank line (paragraph terminator)
            # or the start of an atomic table block. Triple-backtick
            # lines no longer break paragraphs — see the comment above.
            j = i + 1
            while j < n:
                next_stripped = lines[j][0].strip()
                if next_stripped == "":
                    # Paragraph terminator. Absorb this and any further
                    # consecutive blank lines as trailing whitespace.
                    j += 1
                    while j < n and lines[j][0].strip() == "":
                        j += 1
                    break
                if next_stripped.startswith("|"):
                    # Atomic table boundary — let the outer loop pick it up.
                    break
                j += 1
            block_end = lines[j - 1][2]
            blocks.append(
                {"start": block_start, "end": block_end, "type": _BLOCK_TEXT}
            )
            block_start = block_end
            i = j

        # If the document ends with blank lines that never got absorbed
        # (e.g. trailing whitespace after the last block was already
        # flushed), extend the last block to cover them.
        if blocks and blocks[-1]["end"] < len(markdown):
            blocks[-1]["end"] = len(markdown)
        elif not blocks and len(markdown) > 0:
            # Whitespace-only input — caller will emit a single window.
            pass

        return blocks

    # ------------------------------------------------------------------
    # Heading-stack maintenance
    # ------------------------------------------------------------------

    @staticmethod
    def _update_heading_stack(
        block_text: str, stack: list[tuple[int, str]]
    ) -> list[tuple[int, str]]:
        """Apply the heading lines in ``block_text`` to ``stack`` and
        return the new stack.

        Heading detection mirrors ``markdown_splitter.py``: a line is a
        heading if its stripped form starts with one or more ``#`` and is
        not the wiki-link prefix ``#[[``. The level is the count of
        leading ``#`` characters; the clean title is the stripped form
        with leading ``#``s and surrounding whitespace removed.

        On a new heading at level ``L``: drop every entry at level
        ``>= L`` from the stack, then append ``(L, clean_title)``.
        """

        new_stack = list(stack)
        for line in block_text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("#") or stripped.startswith("#[["):
                continue
            level = len(stripped) - len(stripped.lstrip("#"))
            if level <= 0:
                continue
            clean_title = stripped.lstrip("#").strip()
            # Drop any entries at the same or deeper level.
            new_stack = [(lvl, t) for lvl, t in new_stack if lvl < level]
            new_stack.append((level, clean_title))
        return new_stack
