"""Agentic chunking pipeline package.

Public types re-exported here are the contract surface used by the
``IngestionOrchestrator`` and any external caller. Implementation modules
(``paragraph_pre_segmenter``, ``agentic_chunker``, ``chunk_validator``,
``chunk_assembler``, ``fallback_splitter``, ``file_ingestor``,
``orchestrator``) import these from ``agentic_chunking.data_models``.
"""

from agentic_chunking.chunk_assembler import ChunkAssembler
from agentic_chunking.data_models import (
    EMPTY_METADATA,
    AgenticChunkerError,
    ChunkMetadata,
    RawChunk,
    ValidationResult,
    Window,
)

__all__ = [
    "Window",
    "RawChunk",
    "ChunkMetadata",
    "ValidationResult",
    "EMPTY_METADATA",
    "AgenticChunkerError",
    "ChunkAssembler",
]
