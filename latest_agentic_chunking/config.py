"""Provider-aware configuration for the agentic chunking pipeline.

Mirrors the env-loading pattern used by ``embed.py``: ``python-dotenv`` is
called at import time and ``LLM_PROVIDER`` is read from the environment
(default ``"ollama"``). Provider-specific defaults match the values
documented in ``design.md`` Requirements 13.3 and 13.4.

Embedding dimensions are computed locally rather than imported from
``embed.py`` because that module has heavy import-time side effects
(loads tokenizers, instantiates Qdrant / embedding clients, etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env exactly as embed.py does — same call shape, same default lookup.
load_dotenv()


# ──────────────────────────────────────────────
#  Provider selection (matches embed.py)
# ──────────────────────────────────────────────

#: Active provider from the environment. Defaults to "ollama" so behaviour
#: matches embed.py when the variable is unset.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")


# ──────────────────────────────────────────────
#  Embedding dimensions (mirrored from embed.py)
# ──────────────────────────────────────────────

#: Ollama qwen3-embedding output dimension.
OLLAMA_EMBED_DIMENSIONS: int = 4096

#: Azure embedding deployment env var resolves the dense vector size.
AZURE_EMBED_DEPLOYMENT: str = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)

#: text-embedding-3-large → 3072; text-embedding-3-small (or anything else) → 1536.
AZURE_EMBED_DIMENSIONS: int = (
    3072 if AZURE_EMBED_DEPLOYMENT == "text-embedding-3-large" else 1536
)


# ──────────────────────────────────────────────
#  PipelineConfig
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    """Tunable knobs for the agentic chunking pipeline.

    Frozen so it can be safely shared across the window-level worker pool.
    All fields are integers so equality and hashing are well-defined.
    """

    provider: str
    max_chunk_tokens: int
    min_chunk_tokens: int
    agent_context_budget_tokens: int
    paragraph_threshold_tokens: int
    vector_size: int


# ──────────────────────────────────────────────
#  Provider defaults (Requirements 13.3, 13.4)
# ──────────────────────────────────────────────

_AZURE_DEFAULTS: dict = dict(
    max_chunk_tokens=1200,
    min_chunk_tokens=200,
    agent_context_budget_tokens=8000,
    paragraph_threshold_tokens=6000,
    vector_size=AZURE_EMBED_DIMENSIONS,
)

_OLLAMA_DEFAULTS: dict = dict(
    max_chunk_tokens=500,
    min_chunk_tokens=150,
    agent_context_budget_tokens=4000,
    paragraph_threshold_tokens=3000,
    vector_size=OLLAMA_EMBED_DIMENSIONS,
)

# AWS Bedrock (gpt-oss-120b) — 128K context, matches Azure comfortably.
# Embeddings still run on Azure (``text-embedding-3-large``), so
# ``vector_size`` mirrors ``AZURE_EMBED_DIMENSIONS`` here.
_AWS_DEFAULTS: dict = dict(
    max_chunk_tokens=1200,
    min_chunk_tokens=200,
    agent_context_budget_tokens=8000,
    paragraph_threshold_tokens=6000,
    vector_size=AZURE_EMBED_DIMENSIONS,
)


def build_config(provider: str) -> PipelineConfig:
    """Build a ``PipelineConfig`` for ``provider``.

    Parameters
    ----------
    provider:
        Either ``"azure"``, ``"aws"``, or ``"ollama"`` (case-insensitive).
        Anything else raises ``ValueError``.

    Returns
    -------
    PipelineConfig
        Provider-specific defaults. ``vector_size`` is bound to the active
        embedding deployment (Azure for both the ``"azure"`` and ``"aws"``
        providers since embeddings remain on Azure).
    """

    normalized = (provider or "").strip().lower()

    if normalized == "azure":
        return PipelineConfig(provider="azure", **_AZURE_DEFAULTS)
    if normalized == "aws":
        return PipelineConfig(provider="aws", **_AWS_DEFAULTS)
    if normalized == "ollama":
        return PipelineConfig(provider="ollama", **_OLLAMA_DEFAULTS)

    raise ValueError(
        f"Unknown provider: {provider!r}. "
        f"Expected 'azure', 'aws', or 'ollama'."
    )
