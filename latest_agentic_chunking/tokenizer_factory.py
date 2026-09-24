"""
Tokenizer factory for the agentic chunking pipeline.

Loads provider-specific tokenizers once at module level so they are
process-shared and read-only across worker threads. Subsequent calls
to ``build_tokenizer`` for the same provider return the cached instance.

The local-cache convention mirrors ``markdown_splitter.py``:
  - Azure: ``tiktoken.get_encoding("cl100k_base")`` — the encoder
           used by ``text-embedding-3-large``.
  - Ollama: ``transformers.AutoTokenizer.from_pretrained`` with the
            local cache at ``<project-root>/tokenizers/<tokenizer_name>``.
            If the cache directory does not exist, the tokenizer is
            downloaded and saved there for next time.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

# Default Ollama tokenizer (matches markdown_splitter.py default).
DEFAULT_OLLAMA_TOKENIZER_NAME = "Qwen/Qwen3.5-4B"

# Project root — sibling of markdown_splitter.py's "tokenizers/" directory.
# tokenizer_factory.py lives at <root>/agentic_chunking/, so the parent's
# parent is the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Module-level cache: cache_key -> tokenizer instance.
# Loaded once per process; safe to share across threads (HuggingFace
# tokenizers and tiktoken Encodings are read-only after construction).
_tokenizer_cache: dict[str, Any] = {}
_tokenizer_lock = threading.Lock()


def build_tokenizer(
    provider: str,
    tokenizer_name: str = DEFAULT_OLLAMA_TOKENIZER_NAME,
) -> Any:
    """Return a process-shared, read-only tokenizer for ``provider``.

    The first call for a given ``(provider, tokenizer_name)`` loads the
    tokenizer; subsequent calls return the same instance.

    Args:
        provider: ``"azure"`` or ``"ollama"``.
        tokenizer_name: HuggingFace model id used only on the Ollama path.
            Ignored when ``provider == "azure"``.

    Raises:
        ValueError: If ``provider`` is not one of the supported values.
    """
    # ``aws`` piggybacks on the Azure cache key because we use the same
    # cl100k_base tokenizer for both — gpt-oss's real tokenizer is
    # different but cl100k is a close-enough proxy for our budget math
    # and avoids pulling a second tokenizer download.
    if provider in ("azure", "aws"):
        cache_key = "azure"
    else:
        cache_key = f"ollama:{tokenizer_name}"

    # Fast path — no lock when the tokenizer is already loaded.
    cached = _tokenizer_cache.get(cache_key)
    if cached is not None:
        return cached

    # Slow path — double-checked locking so concurrent first-callers
    # don't both pay the load cost.
    with _tokenizer_lock:
        cached = _tokenizer_cache.get(cache_key)
        if cached is not None:
            return cached

        if provider in ("azure", "aws"):
            import tiktoken

            # cl100k_base is the tokenizer for text-embedding-3-large
            # (matches markdown_splitter.py) — we reuse it for the aws
            # path since embeddings still run on Azure, so token
            # accounting for embeddings must use cl100k regardless of
            # which LLM produced the chunk. It's also a reasonable
            # proxy for gpt-oss context budgeting.
            tokenizer = tiktoken.get_encoding("cl100k_base")
        elif provider == "ollama":
            from transformers import AutoTokenizer

            tokenizer_path = str(_PROJECT_ROOT / "tokenizers" / tokenizer_name)
            if os.path.exists(tokenizer_path):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                tokenizer.save_pretrained(tokenizer_path)
        else:
            raise ValueError(
                f"Unknown provider: {provider!r} "
                f"(expected 'azure', 'aws', or 'ollama')"
            )

        _tokenizer_cache[cache_key] = tokenizer
        return tokenizer


def count_tokens(tokenizer: Any, text: str) -> int:
    """Count tokens in ``text`` using ``tokenizer`` with the right call shape.

    Detects the tokenizer family by the class's module path so the helper
    can be called with either a ``tiktoken`` Encoding or a HuggingFace
    ``PreTrainedTokenizer`` without the caller needing to know which.
    """
    # tiktoken's Encoding lives in the ``tiktoken`` module; its ``encode``
    # signature takes no ``add_special_tokens`` argument.
    if type(tokenizer).__module__.startswith("tiktoken"):
        return len(tokenizer.encode(text))

    # HuggingFace AutoTokenizer path. ``add_special_tokens=False`` matches
    # markdown_splitter.py — we don't want CLS/SEP tokens inflating counts
    # used for chunk-budget decisions.
    return len(tokenizer.encode(text, add_special_tokens=False))
