"""Chunk assembler for the agentic chunking pipeline.

Implements Algorithm 5 from ``design.md``. Pure function: takes the
per-window raw-chunk lists plus their windows and produces a single flat
list of "linked" chunks with deterministic ``chunk_id``s, parent / prev
/ next / sibling links, ordered ``parent_titles`` maps, ``embedding_text``,
and ``token_count``.

Design decisions worth flagging:

* **Cross-window heading inheritance.** Each window carries
  ``leading_heading_path`` — the H1 -> Hn chain of clean heading titles
  active at the window's start, recorded by ``ParagraphPreSegmenter``.
  Before processing window N's chunks, the assembler seeds two heading
  stacks (parent-id stack for Phase C, title+id stack for Phase F) from
  this path. That is the mechanism that makes a chunk in window N
  inherit ancestor titles introduced by headings whose lines lived in
  earlier windows (Req 7.3).

* **Synthetic levels for seeded ancestors.** ``leading_heading_path`` is
  a flat list of titles — the originating real levels were dropped by
  the pre-segmenter. We assign positional synthetic levels ``1..N`` to
  the seeded entries. The pre-segmenter's ``_update_heading_stack`` only
  ever keeps strictly increasing real levels (deeper-or-equal entries
  are dropped on each new heading), so position in the path is monotone
  in the original level — synthetic levels preserve the relative
  ordering and keep the inheritance / drop logic sound. When a chunk in
  the window introduces its own heading at level ``L``, entries at
  synthetic level ``>= L`` are dropped, exactly as if the original real
  levels had been retained.

* **Deterministic ``chunk_id``.** ``uuid.uuid5(_CHUNK_ID_NAMESPACE,
  f"{doc_id}|{split_group_id}|{chunk_index}")`` where ``chunk_index`` is
  the chunk's position within its window — i.e. per-``(doc_id,
  split_group_id)`` zero-based sequence. Re-running the pipeline on the
  same ``(raw_chunks_per_window, windows)`` input yields the same
  ordered list of ``chunk_id``s (Req 7.2).

* **Sibling grouping by heading-section identity, not by window.** Two
  chunks are siblings iff they share the same
  ``(ancestor_chain_by_ordered_values, own_clean_title, own_level,
  chunk_type)``. ``split_group_id`` is preserved on every chunk for
  diagnostics but never participates in sibling decisions (Req 7.5,
  7.10). A heading section split across windows therefore produces one
  mutual sibling set covering all of its chunks.

* **Purity.** Every input chunk dict is deep-copied before any
  assignment; the input ``raw_chunks_per_window`` lists and ``windows``
  list are never mutated.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any

from agentic_chunking.data_models import Window
from agentic_chunking.tokenizer_factory import count_tokens


# Fixed UUID5 namespace for deterministic chunk-id derivation. We use the
# standard RFC 4122 DNS namespace UUID — any stable constant works as
# long as it never changes across runs (Req 7.2 / Qdrant idempotency).
_CHUNK_ID_NAMESPACE: uuid.UUID = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _clean_title(title: Any) -> str | None:
    """Normalise a title the way ``MarkdownSplitter._build_parent_titles`` does.

    Strips leading ``#`` characters and surrounding whitespace. Returns
    ``None`` if ``title`` is falsy (None / empty string) so callers can
    use truthiness to test "no heading on this chunk".
    """
    if title is None:
        return None
    if not isinstance(title, str):
        return None
    cleaned = title.lstrip("#").strip()
    return cleaned or None


def _filename_from_source(source: Any) -> str:
    """Return just the filename component of ``source`` (no directory).

    Used by ``ChunkAssembler`` to source-anchor each chunk's
    ``embedding_text``. Handles both POSIX (``/``) and Windows (``\\``)
    separators so the same chunks produced on either OS get the same
    anchor. Returns an empty string for falsy / non-string input so
    callers can drop the anchor with truthiness checks.
    """
    if not source or not isinstance(source, str):
        return ""
    name = source
    for sep in ("/", "\\"):
        if sep in name:
            name = name.rsplit(sep, 1)[-1]
    return name.strip()


class ChunkAssembler:
    """Build the final linked chunk list from per-window agent outputs.

    Stateless across calls — the same instance can be shared by every
    ``FileIngestor`` (and across threads, though ``assemble`` itself is
    only called from the main ingest thread after window fan-in).
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        raw_chunks_per_window: list[list[dict]],
        windows: list[Window],
    ) -> list[dict]:
        """Assemble linked chunks from per-window raw chunks.

        See module docstring and design.md Algorithm 5 for the full
        contract. Pure function — does not mutate ``raw_chunks_per_window``
        or ``windows``.
        """
        if len(raw_chunks_per_window) != len(windows):
            raise ValueError(
                "raw_chunks_per_window and windows must have equal length "
                f"(got {len(raw_chunks_per_window)} vs {len(windows)})"
            )

        # Phase A.0: deep-copy and flatten in window order. We track
        # which window each flat chunk came from so Phases A/C can re-seed
        # their heading stacks at window boundaries without re-iterating
        # the nested input.
        flat_chunks: list[dict] = []
        chunk_to_window: list[int] = []
        for w_idx, window_chunks in enumerate(raw_chunks_per_window):
            for ch in window_chunks:
                flat_chunks.append(copy.deepcopy(ch))
                chunk_to_window.append(w_idx)

        if not flat_chunks:
            return []

        # Phase B: deterministic chunk_id assignment. ``chunk_index`` is
        # the per-(doc_id, split_group_id) zero-based position — i.e. the
        # position within the window — so re-runs on the same windows
        # yield the same id sequence. Always overwrite — even if the
        # caller pre-set a chunk_id, the deterministic value is what
        # Qdrant's idempotent upsert relies on.
        per_window_index: dict[tuple[str, str], int] = {}
        for ch in flat_chunks:
            doc_id = str(ch.get("doc_id", "") or "")
            split_group_id = str(ch.get("split_group_id", "") or "")
            key = (doc_id, split_group_id)
            idx = per_window_index.get(key, 0)
            per_window_index[key] = idx + 1
            ch["chunk_id"] = uuid.uuid5(
                _CHUNK_ID_NAMESPACE,
                f"{doc_id}|{split_group_id}|{idx}",
            ).hex

        # Build ``(clean_title, level) -> last_chunk_id_in_group`` map by
        # walking the flat list once. "Group" = a maximal run of
        # contiguous chunks sharing the same ``(clean_title, level)``.
        # The last chunk of such a group is what Phase C registers as
        # the parent of deeper headings, and what Phase F associates
        # with that ancestor in ``parent_titles``. Mirrors
        # ``MarkdownSplitter._build_parent_titles``'s first pass.
        last_chunk_for_heading: dict[tuple[str, int], str] = {}
        i = 0
        n = len(flat_chunks)
        while i < n:
            ch = flat_chunks[i]
            ct = _clean_title(ch.get("title"))
            lvl = ch.get("level")
            if ct is not None and lvl:
                j = i + 1
                while j < n:
                    nct = _clean_title(flat_chunks[j].get("title"))
                    nlvl = flat_chunks[j].get("level")
                    if nct == ct and nlvl == lvl:
                        j += 1
                    else:
                        break
                last_chunk_for_heading[(ct, lvl)] = flat_chunks[j - 1]["chunk_id"]
                i = j
            else:
                i += 1

        # Phase C: parent_chunk_id assignment, mirroring
        # ``MarkdownSplitter._assign_parent_hierarchy`` but with the
        # parent stack re-seeded from ``window.leading_heading_path`` at
        # every window boundary so parent inheritance crosses windows.
        self._assign_parent_hierarchy(
            flat_chunks=flat_chunks,
            chunk_to_window=chunk_to_window,
            windows=windows,
            last_chunk_for_heading=last_chunk_for_heading,
        )

        # Phase F (parent_titles part): build the ordered
        # ``{ancestor_chunk_id: clean_title}`` map for every chunk by
        # walking the flat list with a (level -> (clean_title,
        # last_chunk_id)) stack, again seeding from
        # ``leading_heading_path`` at each window boundary.
        self._build_parent_titles(
            flat_chunks=flat_chunks,
            chunk_to_window=chunk_to_window,
            windows=windows,
            last_chunk_for_heading=last_chunk_for_heading,
        )

        # Phase D: single doubly-linked prev/next chain over the WHOLE
        # flattened list. The last chunk of window N points to the first
        # chunk of window N+1 — there are no None breaks at internal
        # window boundaries (Req 7.4).
        for idx, ch in enumerate(flat_chunks):
            ch["prev_chunk_id"] = (
                flat_chunks[idx - 1]["chunk_id"] if idx > 0 else None
            )
            ch["next_chunk_id"] = (
                flat_chunks[idx + 1]["chunk_id"] if idx < n - 1 else None
            )

        # Phase E: sibling grouping by heading-section identity. The
        # ancestor chain comparison uses ordered ``parent_titles``
        # values (clean titles) rather than ancestor chunk_ids — but in
        # practice chunks under the same section also share the same
        # ancestor chunk_ids, so the result is the same. ``split_group_id``
        # never enters the identity tuple (Req 7.5, 7.10).
        groups: dict[tuple, list[dict]] = {}
        for ch in flat_chunks:
            ancestor_chain = tuple(ch["parent_titles"].values())
            own_ct = _clean_title(ch.get("title"))
            identity = (
                ancestor_chain,
                own_ct,
                ch.get("level"),
                ch.get("chunk_type"),
            )
            groups.setdefault(identity, []).append(ch)

        for group in groups.values():
            if len(group) > 1:
                ids = [c["chunk_id"] for c in group]
                for c in group:
                    c["sibling_chunk_ids"] = [
                        cid for cid in ids if cid != c["chunk_id"]
                    ]
            else:
                # Ensure the key exists with the canonical empty-list value.
                group[0]["sibling_chunk_ids"] = []

        # Phase F (embedding_text + token_count). ``parent_titles`` values
        # are already in level-ascending insertion order from
        # ``_build_parent_titles``.
        #
        # Source-anchoring: prepend the document's filename (extracted
        # from ``ch["source"]``) so every chunk's dense vector carries
        # the document identity. Without this, two chunks from
        # different documents but the same heading section (e.g. two
        # "Education" sections from two resumes) end up with very
        # similar embeddings and the retriever cannot distinguish
        # them. Stripping to filename keeps the anchor short and
        # consistent across operating systems.
        #
        # Summary line: when the chunk carries a non-empty
        # ``summary``, slot it between the breadcrumb and the content
        # body. The summary is emitted inline by the agent in the same
        # tool call that picks chunk boundaries (see
        # ``tool_schema.py``) and acts as a high-signal preface — the
        # dense embedder weighs early tokens more heavily, so a
        # specific summary anchors the vector on the chunk's actual
        # entities rather than on generic structural words.
        # Fallback / oversized chunks have ``summary == ""`` and the
        # line is skipped — graceful degradation.
        for ch in flat_chunks:
            source_name = _filename_from_source(ch.get("source", ""))
            titles_chain = " > ".join(ch["parent_titles"].values())
            content = ch.get("content", "") or ""
            summary = (ch.get("summary") or "").strip()

            anchor_parts: list[str] = []
            if source_name:
                anchor_parts.append(source_name)
            if titles_chain:
                anchor_parts.append(titles_chain)
            anchor = " > ".join(anchor_parts)

            embedding_lines: list[str] = []
            if anchor:
                embedding_lines.append(anchor)
            if summary:
                embedding_lines.append(summary)
            embedding_lines.append(content)
            ch["embedding_text"] = "\n".join(embedding_lines)
            ch["token_count"] = count_tokens(self._tokenizer, ch["embedding_text"])

        return flat_chunks

    # ------------------------------------------------------------------
    # Phase C — parent_chunk_id assignment
    # ------------------------------------------------------------------

    def _assign_parent_hierarchy(
        self,
        flat_chunks: list[dict],
        chunk_to_window: list[int],
        windows: list[Window],
        last_chunk_for_heading: dict[tuple[str, int], str],
    ) -> None:
        """Mirror ``MarkdownSplitter._assign_parent_hierarchy`` over the
        flat list, with the parent stack re-seeded from
        ``window.leading_heading_path`` at each window boundary.

        ``parent_stack`` maps level -> last chunk_id of the heading
        group at that level. For seeded entries, levels are synthetic
        (positional 1..N). For in-window heading chunks, the chunk's
        own ``level`` is used.
        """
        parent_stack: dict[int, str] = {}
        i = 0
        n = len(flat_chunks)
        prev_window_idx = -1

        while i < n:
            cur_window_idx = chunk_to_window[i]
            if cur_window_idx != prev_window_idx:
                parent_stack = self._seed_parent_stack(
                    windows[cur_window_idx].leading_heading_path,
                    last_chunk_for_heading,
                )
                prev_window_idx = cur_window_idx

            ch = flat_chunks[i]
            ct = _clean_title(ch.get("title"))
            lvl = ch.get("level")

            if ct is not None and lvl:
                # Drop deeper-or-equal levels — this heading replaces them.
                for k in list(parent_stack.keys()):
                    if k >= lvl:
                        del parent_stack[k]

                # Find parent: nearest entry with strictly smaller level.
                parent_id: str | None = None
                for k in sorted(parent_stack.keys(), reverse=True):
                    if k < lvl:
                        parent_id = parent_stack[k]
                        break

                # Apply parent_id to every contiguous chunk in this
                # ``(clean_title, level)`` group **within the same window**.
                # We don't cross window boundaries here — the seeded
                # parent stack is correct only for chunks of the current
                # window; a heading group that spans windows is rare and
                # would already have been split by the agent.
                while i < n and chunk_to_window[i] == cur_window_idx:
                    nct = _clean_title(flat_chunks[i].get("title"))
                    nlvl = flat_chunks[i].get("level")
                    if nct == ct and nlvl == lvl:
                        flat_chunks[i]["parent_chunk_id"] = parent_id
                        i += 1
                    else:
                        break

                # Register the LAST chunk_id of the group we just walked.
                parent_stack[lvl] = flat_chunks[i - 1]["chunk_id"]
            else:
                # Chunk has no own heading. Its parent is the deepest
                # entry currently on the stack (the most recent ancestor)
                # — that gives chunks of an unheaded preamble inside a
                # section their natural parent. Mirrors what
                # ``MarkdownSplitter`` does in practice once you account
                # for the fact that its un-titled chunks only appear at
                # the very top of a document (where the stack is empty
                # and parent_id is therefore None anyway).
                if parent_stack:
                    deepest = max(parent_stack.keys())
                    ch["parent_chunk_id"] = parent_stack[deepest]
                else:
                    ch["parent_chunk_id"] = None
                i += 1

    # ------------------------------------------------------------------
    # Phase F — parent_titles assembly
    # ------------------------------------------------------------------

    def _build_parent_titles(
        self,
        flat_chunks: list[dict],
        chunk_to_window: list[int],
        windows: list[Window],
        last_chunk_for_heading: dict[tuple[str, int], str],
    ) -> None:
        """Build each chunk's ordered ``parent_titles`` map.

        Mirrors ``MarkdownSplitter._build_parent_titles``'s second pass:
        keep a level-keyed stack of ``(clean_title, last_chunk_id)``,
        update it on every heading chunk (using ``last_chunk_for_heading``
        for the chunk_id so siblings within a group share the same
        ancestor reference), and snapshot the stack into ``parent_titles``
        for every chunk.

        Cross-window: at each window boundary the stack is re-seeded from
        ``windows[window_idx].leading_heading_path`` with synthetic
        positional levels — same trick as ``_assign_parent_hierarchy``.
        """
        # Stack maps level -> (clean_title, chunk_id_or_None).
        heading_stack: dict[int, tuple[str, str | None]] = {}
        prev_window_idx = -1

        for i, ch in enumerate(flat_chunks):
            cur_window_idx = chunk_to_window[i]
            if cur_window_idx != prev_window_idx:
                heading_stack = self._seed_titles_stack(
                    windows[cur_window_idx].leading_heading_path,
                    last_chunk_for_heading,
                )
                prev_window_idx = cur_window_idx

            ct = _clean_title(ch.get("title"))
            lvl = ch.get("level")

            if ct is not None and lvl:
                cid = last_chunk_for_heading.get((ct, lvl), ch["chunk_id"])
                # Set first, then drop strictly deeper levels — matches
                # ``MarkdownSplitter._build_parent_titles`` exactly.
                heading_stack[lvl] = (ct, cid)
                for k in list(heading_stack.keys()):
                    if k > lvl:
                        del heading_stack[k]

            # Snapshot the current stack into parent_titles. Iterate
            # levels in ascending order so insertion order in the dict
            # mirrors H1 -> Hn order — that's the order ``embedding_text``
            # joins on. Skip seeded ancestors whose chunk_id we couldn't
            # resolve (cid is None) — they would otherwise inject a None
            # key, which Qdrant would not accept.
            parent_titles: dict[str, str] = {}
            for k in sorted(heading_stack.keys()):
                title, cid = heading_stack[k]
                if cid is None:
                    continue
                parent_titles[cid] = title
            ch["parent_titles"] = parent_titles

    # ------------------------------------------------------------------
    # Heading-stack seeding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_parent_stack(
        leading_heading_path: list[str],
        last_chunk_for_heading: dict[tuple[str, int], str],
    ) -> dict[int, str]:
        """Seed Phase C's parent stack from ``leading_heading_path``.

        Synthetic levels ``1..N`` are assigned positionally. For each
        path title, look up its chunk_id in ``last_chunk_for_heading``;
        if the synthetic level isn't a key, fall back to whatever
        ``(title, *)`` entry exists (titles are usually unique). Entries
        with no resolvable chunk_id are skipped — they'd produce a
        ``parent_chunk_id == None`` for in-window chunks, which is the
        same as having no ancestor.
        """
        stack: dict[int, str] = {}
        for syn_lvl, raw_title in enumerate(leading_heading_path, start=1):
            ct = _clean_title(raw_title)
            if ct is None:
                continue
            cid = last_chunk_for_heading.get((ct, syn_lvl))
            if cid is None:
                # Best-effort fallback: any (ct, *) match.
                for (mt, _ml), mid in last_chunk_for_heading.items():
                    if mt == ct:
                        cid = mid
                        break
            if cid is not None:
                stack[syn_lvl] = cid
        return stack

    @staticmethod
    def _seed_titles_stack(
        leading_heading_path: list[str],
        last_chunk_for_heading: dict[tuple[str, int], str],
    ) -> dict[int, tuple[str, str | None]]:
        """Seed Phase F's title stack from ``leading_heading_path``.

        Same synthetic-level scheme as ``_seed_parent_stack``, but the
        stack carries ``(clean_title, chunk_id_or_None)`` tuples so we
        can build ``parent_titles`` even when a chunk_id can't be
        resolved (the title still informs ``embedding_text`` formatting,
        though in practice we drop None-cid entries from
        ``parent_titles`` to avoid invalid keys).
        """
        stack: dict[int, tuple[str, str | None]] = {}
        for syn_lvl, raw_title in enumerate(leading_heading_path, start=1):
            ct = _clean_title(raw_title)
            if ct is None:
                continue
            cid: str | None = last_chunk_for_heading.get((ct, syn_lvl))
            if cid is None:
                for (mt, _ml), mid in last_chunk_for_heading.items():
                    if mt == ct:
                        cid = mid
                        break
            stack[syn_lvl] = (ct, cid)
        return stack
