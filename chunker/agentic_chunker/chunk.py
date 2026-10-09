"""Agentic (LLM-driven) chunker — the fallback strategy.

Wraps the top-level ``agentic_chunking`` pipeline:

    FileIngestor
      ├─ ParagraphPreSegmenter  (splits big docs into windows)
      ├─ AgenticChunker / ParagraphChunker  (one LLM tool call per window)
      ├─ FallbackSplitter       (rule-based MarkdownSplitter if a window fails)
      └─ ChunkAssembler         (ids, parent / sibling / prev-next links)

Configuration (.env), same names ``embed.py`` used:

    CHUNKER_PROVIDER   "aws" | "azure" | "ollama"   (falls back to LLM_PROVIDER)
    CHUNKER_STRATEGY   "char_offsets" (default)  -> AgenticChunker
                       "paragraph_ids"           -> ParagraphChunker
                                                    (for weaker Bedrock models
                                                     such as DeepSeek)
    AWS_BEDROCK_MODEL_ID / AWS_BEDROCK_REASONING_EFFORT   (aws, via aws_helpers)
    OLLAMA_CHAT_MODEL                                     (ollama)
    AGENTIC_CHUNKER_CONCURRENCY  max concurrent LLM calls   (default 4)
    AGENTIC_WINDOW_WORKERS       windows processed in parallel (default 4)

Token budgets and the tokenizer come from ``build_config(CHUNKER_PROVIDER)``,
so DeepSeek on Bedrock gets the Bedrock-sized budgets, not Ollama's.
"""

from __future__ import annotations

import logging
import os
import threading

from dotenv import load_dotenv

from chunker._schema import normalize_chunks
from chunker.adaptive_chunker.chunk import splitter_provider
from chunker.agentic_chunker import _compat

load_dotenv()

# Must run before any agentic_chunking module that imports
# ``markdown_splitter`` or ``aws_helpers``.
_compat.install()

from agentic_chunking.agentic_chunker import AgenticChunker as _CharOffsetChunker  # noqa: E402
from agentic_chunking.chunk_assembler import ChunkAssembler  # noqa: E402
from agentic_chunking.chunk_validator import ChunkValidator  # noqa: E402
from agentic_chunking.config import LLM_PROVIDER, PipelineConfig, build_config  # noqa: E402
from agentic_chunking.fallback_splitter import FallbackSplitter  # noqa: E402
from agentic_chunking.file_ingestor import FileIngestor  # noqa: E402
from agentic_chunking.paragraph_pre_segmenter import ParagraphPreSegmenter  # noqa: E402
from agentic_chunking.tokenizer_factory import build_tokenizer  # noqa: E402
from chunker.adaptive_chunker.markdown_splitter_kiro import MarkdownSplitter  # noqa: E402

logger = logging.getLogger(__name__)

PRODUCER = "agent"
STRATEGIES = ("char_offsets", "paragraph_ids")


def resolve_provider(provider: str | None = None) -> str:
    """Explicit arg > CHUNKER_PROVIDER > LLM_PROVIDER > "ollama"."""
    return (provider or os.getenv("CHUNKER_PROVIDER") or LLM_PROVIDER or "ollama").strip().lower()


def resolve_strategy(strategy: str | None = None) -> str:
    value = (strategy or os.getenv("CHUNKER_STRATEGY") or "char_offsets").strip().lower()
    if value not in STRATEGIES:
        logger.warning("unknown CHUNKER_STRATEGY %r, using 'char_offsets'", value)
        value = "char_offsets"
    return value


class AgenticChunker:
    """Build the agentic pipeline once, then call ``chunk`` per document."""

    def __init__(
        self,
        provider: str | None = None,
        strategy: str | None = None,
        model: str | None = None,
        config: PipelineConfig | None = None,
        max_retries: int = 3,
        max_concurrency: int | None = None,
        max_window_workers: int | None = None,
    ) -> None:
        self.provider = resolve_provider(provider)
        self.strategy = resolve_strategy(strategy)
        self.config = config or build_config(self.provider)
        cfg = self.config

        max_concurrency = max_concurrency or int(os.getenv("AGENTIC_CHUNKER_CONCURRENCY", "4"))
        max_window_workers = max_window_workers or int(os.getenv("AGENTIC_WINDOW_WORKERS", "4"))

        if self.strategy == "paragraph_ids":
            from agentic_chunking.paragraph_chunker import ParagraphChunker as chunker_cls
        else:
            chunker_cls = _CharOffsetChunker

        tokenizer = build_tokenizer(self.provider)
        validator = ChunkValidator(tokenizer, cfg.max_chunk_tokens, cfg.min_chunk_tokens)
        self.llm_chunker = chunker_cls(
            self.provider,
            model,
            cfg.max_chunk_tokens,
            cfg.min_chunk_tokens,
            max_retries,
            rate_limit_semaphore=threading.Semaphore(max_concurrency),
            validator=validator,
        )
        self.model = self.llm_chunker.model

        fallback = FallbackSplitter(
            MarkdownSplitter(
                provider=splitter_provider(self.provider),
                max_chunk_tokens=cfg.max_chunk_tokens,
                min_chunk_tokens=cfg.min_chunk_tokens,
            )
        )

        self.ingestor = FileIngestor(
            chunker=self.llm_chunker,
            pre_segmenter=ParagraphPreSegmenter(tokenizer, cfg.agent_context_budget_tokens),
            fallback=fallback,
            assembler=ChunkAssembler(tokenizer),
            tokenizer=tokenizer,
            paragraph_threshold_tokens=cfg.paragraph_threshold_tokens,
            agent_context_budget_tokens=cfg.agent_context_budget_tokens,
            max_window_workers=max_window_workers,
        )
        logger.info(
            "agentic chunker: provider=%s strategy=%s model=%s max/min tokens=%d/%d",
            self.provider, self.strategy, self.model, cfg.max_chunk_tokens, cfg.min_chunk_tokens,
        )

    def chunk(self, doc_id: str, source: str, markdown: str) -> list[dict]:
        """Return fully linked chunks.

        Per-window LLM failures fall back to the rule-based splitter inside
        ``FileIngestor`` (those chunks have ``producer="fallback"``);
        anything else (e.g. provider unreachable) raises.
        """
        linked_chunks, _ = self.ingestor.ingest(
            doc_id=doc_id, source=source, markdown=markdown
        )
        return normalize_chunks(linked_chunks, PRODUCER)
