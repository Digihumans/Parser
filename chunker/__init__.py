"""Chunking strategies + upfront document classification.

Chunkers are imported lazily by ``DocumentRouter`` so importing this
package stays cheap.
"""

from chunker.document_classifier import (
    ChunkingStrategy,
    ClassificationResult,
    DocumentRouter,
    DocumentType,
    classify_document,
)

__all__ = [
    "ChunkingStrategy",
    "ClassificationResult",
    "DocumentRouter",
    "DocumentType",
    "classify_document",
]
