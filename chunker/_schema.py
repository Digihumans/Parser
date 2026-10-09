"""Shared chunk-schema helpers for every chunker in this package.

All chunkers (adaptive, semantic, agentic) must hand back chunk dicts with
the same payload keys so the downstream embed / Qdrant upsert stage does not
care which strategy produced them. The agentic pipeline already emits the
full schema; the rule-based chunkers (adaptive, semantic) are normalised here.
"""

from __future__ import annotations

# Mirrors agentic_chunking.orchestrator._REQUIRED_PAYLOAD_KEYS plus "chunk_id".
REQUIRED_CHUNK_KEYS: tuple[str, ...] = (
    "chunk_id",
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


def normalize_chunks(chunks: list[dict], producer: str) -> list[dict]:
    """Fill the payload keys the rule-based splitters don't set (in place).

    Existing values are never overwritten, so this is safe to call on
    chunks that already carry the agentic schema.
    """
    for chunk in chunks:
        chunk_type = chunk.get("chunk_type") or "text"
        chunk["chunk_type"] = chunk_type
        chunk.setdefault("primary_type", chunk_type)
        chunk.setdefault("has_text", chunk_type == "text")
        chunk.setdefault("has_code", chunk_type == "code")
        chunk.setdefault("has_table", chunk_type == "table")
        chunk.setdefault("split_group_id", None)
        chunk.setdefault("summary", "")
        chunk.setdefault("kb_ids", None)
        chunk.setdefault("metadata", {})
        chunk.setdefault("producer", producer)
    return chunks


def missing_keys(chunks: list[dict]) -> list[tuple[int, str]]:
    """Return (chunk_index, key) for every required key that is absent."""
    return [
        (i, key)
        for i, chunk in enumerate(chunks)
        for key in REQUIRED_CHUNK_KEYS
        if key not in chunk
    ]
