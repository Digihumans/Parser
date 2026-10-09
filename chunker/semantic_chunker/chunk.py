"""Semantic chunker using an Ollama embedding model (qwen3-embedding).

Algorithm (embedding-distance breakpoints, a.k.a. "Kamradt" semantic split):

1. Parse markdown into heading sections. Inside each section, break the body
   into units: sentences for prose, one unit per list item, and atomic units
   for fenced code blocks and tables (never split semantically).
2. Embed every text unit once via ``ollama.Client().embed`` (batched).
3. For every gap between two text units, compare the mean embedding of the
   ``buffer_size`` units before the gap with the ``buffer_size`` units after
   it (cosine distance). Distances above the ``breakpoint_percentile`` of
   the document's distance distribution become chunk boundaries.
4. Enforce token limits: a chunk is also closed before it would exceed
   ``max_chunk_tokens``; afterwards undersized neighbours inside the same
   section are merged while they fit.
5. Linking (parent / siblings / prev-next / parent_titles / embedding_text)
   reuses the phases of the adaptive ``MarkdownSplitter`` so every chunker
   emits the same chunk schema.

Headings are always hard boundaries, so chunk titles stay accurate.

Env vars:
    OLLAMA_EMBED_MODEL   default "qwen3-embedding"
    OLLAMA_HOST          optional, default Ollama host
"""

from __future__ import annotations

import html
import os
import re
import uuid
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv

from chunker._schema import normalize_chunks
from chunker.adaptive_chunker.chunk import splitter_provider
from chunker.adaptive_chunker.markdown_splitter_kiro import MarkdownSplitter

load_dotenv()

PRODUCER = "semantic"

DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding")

# Sentence boundary: end punctuation followed by whitespace and a likely
# sentence start. Keeps "e.g. foo" / "3.14" together in most cases.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


# ──────────────────────────────────────────────
#  Internal data structures
# ──────────────────────────────────────────────

@dataclass
class _Unit:
    text: str
    kind: str = "text"        # "text" | "code" | "table"
    sep_before: str = ""      # how to join with the previous unit
    tokens: int = 0


@dataclass
class _Section:
    title: str | None
    level: int | None
    units: list[_Unit] = field(default_factory=list)


class SemanticChunker:
    def __init__(
        self,
        provider: str = "ollama",
        embed_model: str = DEFAULT_EMBED_MODEL,
        max_chunk_tokens: int = 800,
        min_chunk_tokens: int = 300,
        breakpoint_percentile: float = 90.0,
        buffer_size: int = 2,
        embed_batch_size: int = 32,
        ollama_host: str | None = None,
        splitter: MarkdownSplitter | None = None,
    ) -> None:
        import ollama

        self.embed_model = embed_model
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = max(1, buffer_size)
        self.embed_batch_size = max(1, embed_batch_size)
        self.client = ollama.Client(host=ollama_host or os.getenv("OLLAMA_HOST") or None)

        # Reused for tokenisation, table row splitting and the linking phases.
        self.splitter = splitter or MarkdownSplitter(
            provider=splitter_provider(provider),
            max_chunk_tokens=max_chunk_tokens,
            min_chunk_tokens=min_chunk_tokens,
        )

    # ──────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────

    def chunk(self, doc_id: str, source: str, markdown: str) -> list[dict]:
        filename = os.path.basename(source)
        sections = self._parse_sections(markdown)

        for section in sections:
            for unit in section.units:
                unit.tokens = self._count(unit.text)

        distances = self._section_distances(sections)
        threshold = self._threshold(distances)

        chunks: list[dict] = []
        for section, section_distances in zip(sections, distances):
            groups, boundaries = self._group_units(section.units, section_distances, threshold)
            groups = self._merge_small_groups(groups, boundaries)
            for group in groups:
                content = self._join(group)
                if content.strip():
                    chunks.append(
                        self._new_chunk(doc_id, filename, section, content, group[0].kind)
                    )

        # Same linking phases as the adaptive splitter (phases 4-7).
        self.splitter._assign_parent_hierarchy(chunks)
        self.splitter._assign_sequential_links(chunks)
        self.splitter._assign_sibling_links(chunks)
        self.splitter._build_parent_titles(chunks)

        return normalize_chunks(chunks, PRODUCER)

    # ──────────────────────────────────────────
    #  Step 1: parse markdown into sections + units
    # ──────────────────────────────────────────

    def _parse_sections(self, markdown: str) -> list[_Section]:
        sections: list[_Section] = [_Section(title=None, level=None)]
        paragraph: list[str] = []
        code: list[str] = []
        table: list[str] = []
        in_code = False
        pending_sep = ""

        def current() -> _Section:
            return sections[-1]

        def add_unit(text: str, kind: str, sep: str) -> None:
            nonlocal pending_sep
            units = current().units
            units.append(_Unit(text=text, kind=kind, sep_before=sep if units else ""))
            pending_sep = ""

        def flush_paragraph() -> None:
            nonlocal paragraph, pending_sep
            if not paragraph:
                return
            first = True
            for line in paragraph:
                if _LIST_ITEM_RE.match(line):
                    # One unit per list item keeps bullets intact.
                    add_unit(line.rstrip(), "text", pending_sep or ("\n" if not first else "\n\n"))
                else:
                    for i, sentence in enumerate(_SENTENCE_RE.split(line.strip())):
                        if not sentence.strip():
                            continue
                        if first and i == 0:
                            sep = pending_sep or "\n\n"
                        elif i == 0:
                            sep = "\n"
                        else:
                            sep = " "
                        add_unit(sentence.strip(), "text", sep)
                first = False
            paragraph = []
            pending_sep = "\n\n"

        def flush_table() -> None:
            nonlocal table
            if table:
                add_unit("\n".join(table), "table", "\n\n")
                table = []

        for line in markdown.splitlines():
            stripped = line.strip()

            if in_code:
                code.append(line)
                if stripped.startswith("```"):
                    add_unit("\n".join(code), "code", "\n\n")
                    code, in_code = [], False
                continue

            if stripped.startswith("```"):
                flush_paragraph()
                flush_table()
                code, in_code = [line], True
                continue

            if stripped.startswith("|"):
                flush_paragraph()
                table.append(line.rstrip())
                continue
            flush_table()

            if stripped.startswith("#") and not stripped.startswith("#[["):
                flush_paragraph()
                level = len(stripped) - len(stripped.lstrip("#"))
                sections.append(_Section(title=stripped, level=level))
                pending_sep = ""
                continue

            if not stripped:
                flush_paragraph()
                continue

            paragraph.append(line)

        # Unterminated code fence: keep whatever was collected.
        if in_code and code:
            add_unit("\n".join(code), "code", "\n\n")
        flush_paragraph()
        flush_table()

        return [s for s in sections if s.units]

    # ──────────────────────────────────────────
    #  Step 2-3: embeddings + distances
    # ──────────────────────────────────────────

    def _section_distances(self, sections: list[_Section]) -> list[list[float | None]]:
        """Per section, distance[i] = cosine distance between unit i and i+1.

        ``None`` marks a boundary next to a code/table unit (always a break).
        All text windows of the document are embedded in batched calls.
        """
        texts: list[str] = []
        index: list[list[int | None]] = []  # section -> unit -> row in texts

        for section in sections:
            rows: list[int | None] = []
            for unit in section.units:
                if unit.kind != "text":
                    rows.append(None)
                    continue
                rows.append(len(texts))
                texts.append(html.unescape(unit.text))
            index.append(rows)

        vectors = self._embed(texts) if texts else np.zeros((0, 0))

        distances: list[list[float | None]] = []
        for rows in index:
            section_d: list[float | None] = []
            for i in range(len(rows) - 1):
                if rows[i] is None or rows[i + 1] is None:
                    section_d.append(None)
                    continue
                left = self._window(vectors, rows, i, step=-1)
                right = self._window(vectors, rows, i + 1, step=1)
                section_d.append(float(1.0 - np.dot(left, right)))
            distances.append(section_d)
        return distances

    def _window(self, vectors: np.ndarray, rows: list[int | None], start: int, step: int) -> np.ndarray:
        """Mean (re-normalised) vector of up to ``buffer_size`` text units
        starting at ``start`` and walking in ``step`` direction, stopping at
        code/table units. Comparing non-overlapping left/right windows gives
        a sharper topic-shift signal than comparing overlapping windows."""
        picked = []
        j = start
        while 0 <= j < len(rows) and rows[j] is not None and len(picked) < self.buffer_size:
            picked.append(vectors[rows[j]])
            j += step
        mean = np.mean(picked, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm else mean

    def _embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.embed_batch_size):
            batch = texts[start : start + self.embed_batch_size]
            response = self.client.embed(model=self.embed_model, input=batch, truncate=True)
            out.extend(response.embeddings)
        matrix = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _threshold(self, distances: list[list[float | None]]) -> float:
        flat = [d for section in distances for d in section if d is not None]
        if not flat:
            return float("inf")
        return float(np.percentile(flat, self.breakpoint_percentile))

    # ──────────────────────────────────────────
    #  Step 4: grouping with token limits
    # ──────────────────────────────────────────

    def _group_units(
        self,
        units: list[_Unit],
        distances: list[float | None],
        threshold: float,
    ) -> tuple[list[list[_Unit]], list[float]]:
        """Return ``(groups, boundaries)``.

        ``boundaries[k]`` is the cosine distance across the cut between
        ``groups[k]`` and ``groups[k + 1]`` (``inf`` next to code/table
        units, ``0.0`` inside a word-split oversized sentence). The merge
        step uses it to merge small groups across the weakest cut.
        """
        groups: list[list[_Unit]] = []
        boundaries: list[float] = []
        current: list[_Unit] = []
        current_tokens = 0

        def close(boundary: float) -> None:
            nonlocal current, current_tokens
            if current:
                if groups:
                    boundaries.append(pending_boundary[0])
                groups.append(current)
                pending_boundary[0] = boundary
            current, current_tokens = [], 0

        # Distance across the cut *after* the most recently closed group.
        pending_boundary = [float("inf")]

        def emit(group: list[_Unit], boundary_after: float) -> None:
            if groups:
                boundaries.append(pending_boundary[0])
            groups.append(group)
            pending_boundary[0] = boundary_after

        for i, unit in enumerate(units):
            d = distances[i] if i < len(distances) else None
            d_after = float("inf") if d is None else d

            if unit.kind != "text":
                close(float("inf"))
                for piece in self._split_atomic(unit):
                    emit([piece], float("inf"))
                continue

            pieces = self._split_oversized_text(unit)
            for piece in pieces:
                if current and current_tokens + piece.tokens > self.max_chunk_tokens:
                    # Token-limit cut: inside a split sentence it is 0, else
                    # the distance from the previous unit.
                    prev_d = distances[i - 1] if 0 < i <= len(distances) else None
                    close(0.0 if piece is not pieces[0] else (prev_d if prev_d is not None else 0.0))
                current.append(piece)
                current_tokens += piece.tokens

            # Semantic breakpoint after this unit?
            if d is not None and d > threshold:
                close(d_after)

        close(float("inf"))
        return groups, boundaries

    def _split_atomic(self, unit: _Unit) -> list[_Unit]:
        """Code blocks stay whole; oversized tables are split by rows."""
        if unit.kind == "table" and unit.tokens > self.max_chunk_tokens:
            pieces = self.splitter._split_table_by_rows(unit.text, "")
            return [
                _Unit(text=p, kind="table", sep_before=unit.sep_before, tokens=self._count(p))
                for p in pieces
            ]
        return [unit]

    def _split_oversized_text(self, unit: _Unit) -> list[_Unit]:
        """A single sentence longer than max_chunk_tokens is split by words."""
        if unit.tokens <= self.max_chunk_tokens:
            return [unit]
        pieces: list[_Unit] = []
        words, buf = unit.text.split(), []
        for word in words:
            candidate = " ".join(buf + [word])
            if buf and self._count(candidate) > self.max_chunk_tokens:
                text = " ".join(buf)
                pieces.append(_Unit(text=text, sep_before=" ", tokens=self._count(text)))
                buf = [word]
            else:
                buf.append(word)
        if buf:
            text = " ".join(buf)
            pieces.append(_Unit(text=text, sep_before=" ", tokens=self._count(text)))
        pieces[0].sep_before = unit.sep_before
        return pieces

    def _merge_small_groups(
        self, groups: list[list[_Unit]], boundaries: list[float]
    ) -> list[list[_Unit]]:
        """Merge undersized text groups into a neighbour (same section).

        Repeatedly takes the smallest group below ``min_chunk_tokens`` and
        merges it with the adjacent text group across the *weaker* cut
        (smaller embedding distance), as long as the result fits
        ``max_chunk_tokens``. Code/table groups are never merged.
        """
        groups = [list(g) for g in groups]
        boundaries = list(boundaries)
        tokens = [sum(u.tokens for u in g) for g in groups]
        stuck: set[int] = set()  # ids of groups that cannot merge anywhere

        while True:
            candidates = [
                k for k, g in enumerate(groups)
                if g[0].kind == "text" and tokens[k] < self.min_chunk_tokens and id(g) not in stuck
            ]
            if not candidates:
                break
            k = min(candidates, key=lambda idx: tokens[idx])

            options = []
            for nb, cut in ((k - 1, k - 1), (k + 1, k)):
                if (
                    0 <= nb < len(groups)
                    and groups[nb][0].kind == "text"
                    and tokens[k] + tokens[nb] <= self.max_chunk_tokens
                    and boundaries[cut] != float("inf")
                ):
                    options.append((boundaries[cut], nb, cut))
            if not options:
                stuck.add(id(groups[k]))
                continue

            _, nb, cut = min(options)
            lo, hi = min(k, nb), max(k, nb)
            groups[lo] = groups[lo] + groups[hi]
            tokens[lo] += tokens[hi]
            del groups[hi], tokens[hi], boundaries[cut]

        return groups

    # ──────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────

    def _count(self, text: str) -> int:
        return self.splitter.count_tokens(text)

    @staticmethod
    def _join(group: list[_Unit]) -> str:
        parts = []
        for i, unit in enumerate(group):
            parts.append(unit.text if i == 0 else unit.sep_before + unit.text)
        return "".join(parts).strip() + "\n"

    @staticmethod
    def _new_chunk(doc_id: str, source: str, section: _Section, content: str, kind: str) -> dict:
        return {
            "doc_id": doc_id,
            "source": source,
            "chunk_id": uuid.uuid4().hex,
            "title": section.title,
            "level": section.level,
            "content": content,
            "chunk_type": kind,
            "parent_chunk_id": None,
            "parent_titles": {},
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "sibling_chunk_ids": [],
            "embedding_text": "",
            "token_count": 0,
            "kb_ids": None,
            "metadata": {},
        }
