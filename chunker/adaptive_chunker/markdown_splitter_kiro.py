"""
Markdown Splitter with Parent-Child Heading Hierarchy

Pipeline:
  1. Split by headings
  2. Sub-split by type (text / code / table)
  3. Token-aware overflow splitting (inline replacement)
  3.5. Merge undersized adjacent chunks within (title, level, type) groups
  4. Assign parent-child hierarchy (last chunk of parent heading = parent)
  5. Assign sequential prev/next across all chunks
  6. Assign sibling links within heading groups
  7. Build parent_titles {chunk_id: title} + embedding_text

All IDs in the final output reference chunks that exist in the list.
"""

import os
import re
import uuid
from pathlib import Path
from uuid import uuid4


class MarkdownSplitter:

    def __init__(
        self,
        provider: str = "ollama",
        tokenizer_name: str = "Qwen/Qwen3.5-4B",
        max_chunk_tokens: int | None = 800,
        min_chunk_tokens: int | None = 300,
    ):
        self.provider = provider
        self.max_chunk_tokens = max_chunk_tokens  # None = skip overflow splitting
        # Floor for the merge phase. Adjacent chunks under this size (and same
        # heading + type) get merged together up to max_chunk_tokens. None disables.
        self.min_chunk_tokens = min_chunk_tokens

        if provider == "azure":
            import tiktoken
            # use cl100k_base — the tokenizer for text-embedding-3-large
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            from transformers import AutoTokenizer
            tokenizer_path = str(Path(__file__).parent / "tokenizers" / tokenizer_name)
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                self.tokenizer.save_pretrained(tokenizer_path)

    def count_tokens(self, text: str) -> int:
        if self.provider == "azure":
            return len(self.tokenizer.encode(text))
        else:
            return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split(self, doc_id:str, source: str, markdown: str) -> list[dict]:
        """
        Full pipeline — all splitting first, then linking on the final flat list.
        """
        filename = os.path.basename(source)

        # Phase 1: split by headings
        heading_chunks = self._split_by_headings(doc_id, filename, markdown)

        # Phase 2: sub-split by type (text / code / table)
        typed_chunks = self._split_by_type(heading_chunks)

        # Phase 3: token-aware overflow splitting (skip if max_chunk_tokens is None)
        if self.max_chunk_tokens is not None:
            flat_chunks = self._split_oversized_chunks(typed_chunks)
        else:
            flat_chunks = typed_chunks

        # Phase 3.5: merge undersized adjacent chunks within (title, level, type)
        if self.min_chunk_tokens is not None and self.max_chunk_tokens is not None:
            flat_chunks = self._merge_small_chunks(flat_chunks)

        # Phase 4: parent-child hierarchy (last chunk of parent = parent_chunk_id)
        self._assign_parent_hierarchy(flat_chunks)

        # Phase 5: sequential prev/next
        self._assign_sequential_links(flat_chunks)

        # Phase 6: sibling links within each heading group
        self._assign_sibling_links(flat_chunks)

        # Phase 7: parent_titles + embedding_text + token_count
        self._build_parent_titles(flat_chunks)

        return flat_chunks

    # ──────────────────────────────────────────
    #  Phase 1: Split by headings
    # ──────────────────────────────────────────

    def _split_by_headings(self, doc_id: str, filename: str, markdown: str) -> list[dict]:
        chunks = []

        current = self._new_chunk(doc_id, filename, title=None, level=None)

        for line in markdown.splitlines():
            stripped = line.strip()

            # if stripped.startswith("<!--"):
                # continue

            if stripped.startswith("#") and not stripped.startswith("#[["):
                if current["content"].strip():
                    chunks.append(current)

                level = len(stripped) - len(stripped.lstrip("#"))
                current = self._new_chunk(doc_id, filename, title=stripped, level=level)
            else:
                current["content"] += line + "\n"

        if current["content"].strip():
            chunks.append(current)

        return chunks

    def _new_chunk(self, doc_id: str, source: str, title, level) -> dict:
        return {
            "doc_id": doc_id,
            "source": source,
            "chunk_id": uuid.uuid4().hex,
            "title": title,
            "level": level,
            "content": "",
            "chunk_type": "text",
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

    # ──────────────────────────────────────────
    #  Phase 2: Sub-split by type
    # ──────────────────────────────────────────

    def _split_by_type(self, heading_chunks: list[dict]) -> list[dict]:
        all_chunks = []

        for heading_chunk in heading_chunks:
            sub_chunks = self._extract_typed_blocks(heading_chunk)
            sub_chunks = [sc for sc in sub_chunks if sc["content"].strip()]

            if not sub_chunks:
                continue

            for sc in sub_chunks:
                sc["chunk_id"] = uuid.uuid4().hex

            all_chunks.extend(sub_chunks)

        return all_chunks

    def _extract_typed_blocks(self, heading_chunk: dict) -> list[dict]:
        sub_chunks = []
        text = ""
        code = ""
        table = ""
        is_code = False
        is_table = False

        for line in heading_chunk["content"].splitlines():
            stripped = line.strip()

            if stripped.startswith("```"):
                if not is_code:
                    if text.strip():
                        sub_chunks.append(self._make_sub(heading_chunk, text, "text"))
                        text = ""
                    if table.strip():
                        sub_chunks.append(self._make_sub(heading_chunk, table, "table"))
                        table = ""
                        is_table = False
                    is_code = True
                    code += line + "\n"
                    continue
                else:
                    code += line + "\n"
                    sub_chunks.append(self._make_sub(heading_chunk, code, "code"))
                    code = ""
                    is_code = False
                    continue

            if is_code:
                code += line + "\n"
                continue

            if stripped.startswith("|"):
                if text.strip():
                    sub_chunks.append(self._make_sub(heading_chunk, text, "text"))
                    text = ""
                is_table = True
                table += line + "\n"
                continue
            else:
                if is_table:
                    if table.strip():
                        sub_chunks.append(self._make_sub(heading_chunk, table, "table"))
                    table = ""
                    is_table = False

            text += line + "\n"

        if code.strip():
            sub_chunks.append(self._make_sub(heading_chunk, code, "code"))
        if table.strip():
            sub_chunks.append(self._make_sub(heading_chunk, table, "table"))
        if text.strip():
            sub_chunks.append(self._make_sub(heading_chunk, text, "text"))

        return sub_chunks

    def _make_sub(self, heading_chunk: dict, content: str, chunk_type: str) -> dict:
        return {
            "doc_id": heading_chunk["doc_id"],
            "source": heading_chunk["source"],
            "chunk_id": "",
            "title": heading_chunk["title"],
            "level": heading_chunk["level"],
            "content": content,
            "chunk_type": chunk_type,
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

    # ──────────────────────────────────────────
    #  Phase 3: Token-aware overflow splitting
    # ──────────────────────────────────────────

    def _split_oversized_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Check embedding_text token count. If oversized, split inline.
        We need the hierarchy prefix to measure accurately, but hierarchy
        isn't assigned yet. Use a temporary prefix from the title chain.
        """
        new_chunks = []

        for chunk in chunks:
            # build a temporary hierarchy prefix for token counting
            hierarchy_prefix = self._temp_hierarchy_prefix(chunk)
            embedding_text = hierarchy_prefix + "\n" + chunk["content"]
            token_count = self.count_tokens(embedding_text)

            if token_count <= self.max_chunk_tokens or chunk["chunk_type"] == "code":
                chunk["token_count"] = token_count
                new_chunks.append(chunk)
                continue

            if chunk["chunk_type"] == "table":
                pieces = self._split_table_by_rows(chunk["content"], hierarchy_prefix)
            else:
                pieces = self._split_text_by_sentences(chunk["content"], hierarchy_prefix)

            if not pieces:
                chunk["token_count"] = token_count
                new_chunks.append(chunk)
                continue

            for piece in pieces:
                new_chunk = {
                    "doc_id": chunk["doc_id"],
                    "source": chunk["source"],
                    "chunk_id": uuid.uuid4().hex,
                    "title": chunk["title"],
                    "level": chunk["level"],
                    "content": piece,
                    "chunk_type": chunk["chunk_type"],
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
                new_chunks.append(new_chunk)

        return new_chunks

    def _temp_hierarchy_prefix(self, chunk: dict) -> str:
        """Build a rough hierarchy prefix from the title for token counting.
        The real parent_titles are assigned later, but the title itself
        gives a close enough estimate."""
        if chunk.get("title"):
            return chunk["title"].lstrip("#").strip()
        return ""

    def _split_text_by_sentences(self, text: str, hierarchy_prefix: str) -> list[str]:
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())

        if len(sentences) <= 1 and self.count_tokens(hierarchy_prefix + "\n" + text) > self.max_chunk_tokens:
            sentences = [line for line in text.splitlines() if line.strip()]

        pieces = []
        current_piece = ""

        for sentence in sentences:
            candidate = (current_piece + " " + sentence).strip() if current_piece else sentence
            if self.count_tokens(hierarchy_prefix + "\n" + candidate) <= self.max_chunk_tokens:
                current_piece = candidate
            else:
                if current_piece:
                    pieces.append(current_piece)
                current_piece = sentence

        if current_piece:
            pieces.append(current_piece)

        return pieces

    def _split_table_by_rows(self, table_text: str, hierarchy_prefix: str) -> list[str]:
        lines = [l for l in table_text.split("\n") if l.strip()]

        if len(lines) <= 2:
            return [table_text]

        header = lines[:2]
        rows = lines[2:]

        pieces = []
        current_rows = header.copy()

        for row in rows:
            candidate = "\n".join(current_rows + [row])
            if self.count_tokens(hierarchy_prefix + "\n" + candidate) <= self.max_chunk_tokens:
                current_rows.append(row)
            else:
                if len(current_rows) > 2:
                    pieces.append("\n".join(current_rows))
                current_rows = header.copy() + [row]

        if len(current_rows) > 2:
            pieces.append("\n".join(current_rows))

        return pieces if pieces else [table_text]

    # ──────────────────────────────────────────
    #  Phase 3.5: Merge undersized adjacent chunks
    # ──────────────────────────────────────────

    def _merge_small_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Merge adjacent chunks that share (title, level, chunk_type) when their
        combined size stays under max_chunk_tokens. Each merge target tries to
        reach min_chunk_tokens but never breaks the type boundary.

        Linking fields (parent_chunk_id, prev/next, siblings, parent_titles,
        embedding_text) are reset here — the later phases rebuild them from
        the merged list.
        """
        if not chunks:
            return chunks

        # Step 1: split into runs of same (title, level, chunk_type)
        runs: list[list[dict]] = []
        current_run = [chunks[0]]
        for c in chunks[1:]:
            prev = current_run[-1]
            same_group = (
                c.get("title") == prev.get("title")
                and c.get("level") == prev.get("level")
                and c.get("chunk_type") == prev.get("chunk_type")
            )
            if same_group:
                current_run.append(c)
            else:
                runs.append(current_run)
                current_run = [c]
        runs.append(current_run)

        # Step 2: greedy merge inside each run
        merged: list[dict] = []
        for run in runs:
            merged.extend(self._merge_run(run))
        return merged

    def _merge_run(self, run: list[dict]) -> list[dict]:
        """
        Greedily merge a run of same-group chunks. We accumulate into a buffer
        and flush whenever:
          - adding the next chunk would exceed max_chunk_tokens, OR
          - the buffer is already at/above min_chunk_tokens AND the next chunk
            is also at/above min_chunk_tokens (no need to merge two healthy ones).
        """
        if len(run) == 1:
            return run

        # Code blocks: never merge. Cross-encoder treats code as a unit and
        # sticking two unrelated snippets together hurts more than it helps.
        if run[0].get("chunk_type") == "code":
            return run

        merged: list[dict] = []
        buffer: list[dict] = []
        buffer_tokens = 0

        def flush():
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            if len(buffer) == 1:
                merged.append(buffer[0])
            else:
                merged.append(self._combine_chunks(buffer))
            buffer = []
            buffer_tokens = 0

        for chunk in run:
            chunk_tokens = self._estimate_tokens(chunk)

            if not buffer:
                buffer.append(chunk)
                buffer_tokens = chunk_tokens
                continue

            combined_tokens = buffer_tokens + chunk_tokens
            buffer_full_enough = buffer_tokens >= self.min_chunk_tokens
            chunk_full_enough = chunk_tokens >= self.min_chunk_tokens

            # Skip merging if both sides are already healthy
            if buffer_full_enough and chunk_full_enough:
                flush()
                buffer.append(chunk)
                buffer_tokens = chunk_tokens
                continue

            # Skip merging if it would overflow
            if combined_tokens > self.max_chunk_tokens:
                flush()
                buffer.append(chunk)
                buffer_tokens = chunk_tokens
                continue

            buffer.append(chunk)
            buffer_tokens = combined_tokens

        flush()
        return merged

    def _estimate_tokens(self, chunk: dict) -> int:
        """Token estimate for a chunk during merging.
        We use hierarchy-prefix + content, matching what Phase 3 measured.
        Falls back to counting content alone if anything is missing."""
        prefix = self._temp_hierarchy_prefix(chunk)
        text = (prefix + "\n" + chunk["content"]) if prefix else chunk["content"]
        return self.count_tokens(text)

    def _combine_chunks(self, group: list[dict]) -> dict:
        """Combine a group of same-(title, level, type) chunks into one.
        Resets linking fields — they get rebuilt by phases 4–7."""
        first = group[0]
        combined_content = "".join(c["content"] for c in group)
        return {
            "doc_id": first["doc_id"],
            "source": first["source"],
            "chunk_id": uuid.uuid4().hex,
            "title": first["title"],
            "level": first["level"],
            "content": combined_content,
            "chunk_type": first["chunk_type"],
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

    # ──────────────────────────────────────────
    #  Phase 4: Parent-child hierarchy
    #  Last chunk of a heading section = parent
    # ──────────────────────────────────────────

    def _assign_parent_hierarchy(self, chunks: list[dict]):
        """
        Walk the flat list. For each heading group, find the LAST chunk
        in that group and use it as the parent for child headings.

        heading_stack: level → chunk_id of the LAST chunk in that heading's group
        """
        heading_stack = {}  # level → last_chunk_id

        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            title = chunk.get("title")
            level = chunk.get("level")

            if title and level:
                # pop deeper or equal levels
                for lvl in list(heading_stack.keys()):
                    if lvl >= level:
                        del heading_stack[lvl]

                # find parent: nearest heading with smaller level
                parent_id = None
                for lvl in sorted(heading_stack.keys(), reverse=True):
                    if lvl < level:
                        parent_id = heading_stack[lvl]
                        break

                # collect all contiguous chunks for this heading
                group_start = i
                while i < len(chunks) and chunks[i]["title"] == title and chunks[i]["level"] == level:
                    chunks[i]["parent_chunk_id"] = parent_id
                    i += 1

                # register the LAST chunk of this group
                heading_stack[level] = chunks[i - 1]["chunk_id"]
            else:
                chunk["parent_chunk_id"] = None
                i += 1

    # ──────────────────────────────────────────
    #  Phase 5: Sequential prev/next
    # ──────────────────────────────────────────

    def _assign_sequential_links(self, chunks: list[dict]):
        for i in range(len(chunks)):
            chunks[i]["prev_chunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else None
            chunks[i]["next_chunk_id"] = chunks[i + 1]["chunk_id"] if i < len(chunks) - 1 else None

    # ──────────────────────────────────────────
    #  Phase 6: Sibling links within heading groups
    # ──────────────────────────────────────────

    def _assign_sibling_links(self, chunks: list[dict]):
        """
        Group contiguous chunks that share the same (title, level).
        If a group has more than one chunk, they are siblings.
        """
        i = 0
        while i < len(chunks):
            title = chunks[i].get("title")
            level = chunks[i].get("level")

            # collect contiguous group
            group_start = i
            while (
                i < len(chunks)
                and chunks[i].get("title") == title
                and chunks[i].get("level") == level
            ):
                i += 1

            group = chunks[group_start:i]

            if len(group) > 1:
                all_ids = [c["chunk_id"] for c in group]
                for c in group:
                    c["sibling_chunk_ids"] = [cid for cid in all_ids if cid != c["chunk_id"]]

    # ──────────────────────────────────────────
    #  Phase 7: Parent titles + embedding text
    # ──────────────────────────────────────────

    def _build_parent_titles(self, chunks: list[dict]):
        """
        heading_stack: level → (clean_title, last_chunk_id)
        parent_titles stored as {chunk_id: title} for key-value retrieval.
        """
        # first pass: find the LAST chunk_id for each heading group
        last_chunk_for_heading = {}  # (title, level) → last chunk_id
        i = 0
        while i < len(chunks):
            title = chunks[i].get("title")
            level = chunks[i].get("level")
            if title and level:
                group_start = i
                while i < len(chunks) and chunks[i]["title"] == title and chunks[i]["level"] == level:
                    i += 1
                last_chunk_for_heading[(title, level)] = chunks[i - 1]["chunk_id"]
            else:
                i += 1

        # second pass: build parent_titles dict + embedding_text
        heading_stack = {}  # level → (clean_title, chunk_id)

        for chunk in chunks:
            title = chunk.get("title")
            level = chunk.get("level")

            if title and level:
                clean_title = title.lstrip("#").strip()
                chunk_id = last_chunk_for_heading.get((title, level), chunk["chunk_id"])
                heading_stack[level] = (clean_title, chunk_id)
                for lvl in list(heading_stack.keys()):
                    if lvl > level:
                        del heading_stack[lvl]

            chunk["parent_titles"] = {
                heading_stack[lvl][1]: heading_stack[lvl][0]
                for lvl in sorted(heading_stack.keys())
            }

            titles_list = list(chunk["parent_titles"].values())
            chunk["embedding_text"] = " > ".join(titles_list) + "\n" + chunk["content"]
            chunk["token_count"] = self.count_tokens(chunk["embedding_text"])


# ──────────────────────────────────────────────
#  TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import json

    splitter = MarkdownSplitter(max_chunk_tokens=1200, min_chunk_tokens=300)

    test_file = str(Path(__file__).parent / "test_sample.md")
    with open(test_file, "r", encoding="utf-8") as f:
        markdown = f.read()

    chunks = splitter.split("test_doc", test_file, markdown)

    for i, chunk in enumerate(chunks):
        print(f"\n{'='*60}")
        print(f"Chunk {i}")
        print(f"  chunk_id          : {chunk['chunk_id']}")
        print(f"  title             : {chunk['title']}")
        print(f"  level             : {chunk['level']}")
        print(f"  chunk_type        : {chunk['chunk_type']}")
        print(f"  token_count       : {chunk['token_count']}")
        print(f"  parent_chunk_id   : {chunk['parent_chunk_id']}")
        print(f"  parent_titles     : {chunk['parent_titles']}")
        print(f"  prev_chunk_id     : {chunk['prev_chunk_id']}")
        print(f"  next_chunk_id     : {chunk['next_chunk_id']}")
        print(f"  sibling_ids       : {chunk['sibling_chunk_ids']}")
        print(f"  content           : {chunk['content'].strip()[:100]}")

    # verify all referenced IDs exist
    all_ids = {c["chunk_id"] for c in chunks}
    errors = []
    for i, c in enumerate(chunks):
        for field in ["parent_chunk_id", "prev_chunk_id", "next_chunk_id"]:
            ref = c.get(field)
            if ref and ref not in all_ids:
                errors.append(f"Chunk {i} ({c['chunk_id'][:8]}): {field}={ref[:8]} NOT FOUND")
        for sid in c.get("sibling_chunk_ids", []):
            if sid not in all_ids:
                errors.append(f"Chunk {i} ({c['chunk_id'][:8]}): sibling {sid[:8]} NOT FOUND")

    if errors:
        print(f"\n{'!'*60}")
        print("BROKEN REFERENCES:")
        for e in errors:
            print(f"  {e}")
    else:
        print(f"\n\n✓ All {len(chunks)} chunks have valid references.")

    out_path = str(Path(__file__).parent / "splitter_test_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved to {out_path}")
