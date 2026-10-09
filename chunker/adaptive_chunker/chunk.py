"""Adaptive (structure-aware) chunker.

Thin wrapper over the rule-based ``MarkdownSplitter`` in
``markdown_splitter_kiro.py``: heading split -> text/code/table split ->
token-aware overflow split -> small-chunk merge -> parent / sibling /
prev-next linking. Best for documents with a real heading hierarchy,
code-heavy docs and table-heavy docs.
"""

from __future__ import annotations

from chunker._schema import normalize_chunks
from chunker.adaptive_chunker.markdown_splitter_kiro import MarkdownSplitter

PRODUCER = "adaptive"


def splitter_provider(provider: str) -> str:
    """MarkdownSplitter only knows "azure" (tiktoken) vs anything else (HF).

    AWS token accounting uses cl100k_base in the agentic pipeline, so map it
    to the tiktoken path here too.
    """
    return "azure" if provider in ("azure", "aws") else "ollama"


class AdaptiveChunker:
    def __init__(
        self,
        provider: str = "ollama",
        max_chunk_tokens: int = 800,
        min_chunk_tokens: int = 300,
        splitter: MarkdownSplitter | None = None,
    ) -> None:
        self.splitter = splitter or MarkdownSplitter(
            provider=splitter_provider(provider),
            max_chunk_tokens=max_chunk_tokens,
            min_chunk_tokens=min_chunk_tokens,
        )

    def chunk(self, doc_id: str, source: str, markdown: str) -> list[dict]:
        chunks = self.splitter.split(doc_id, source, markdown)
        return normalize_chunks(chunks, PRODUCER)
