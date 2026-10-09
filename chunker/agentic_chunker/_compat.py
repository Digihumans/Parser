"""Import shims so the top-level ``agentic_chunking`` package can load.

``agentic_chunking`` imports two top-level modules that live elsewhere:

* ``markdown_splitter`` -> ``chunker/adaptive_chunker/markdown_splitter_kiro.py``
* ``aws_helpers``       -> ``chunker/aws_helpers.py``

``install()`` registers those files under the expected top-level names in
``sys.modules`` *only if* no real top-level module exists. A real
``markdown_splitter.py`` / ``aws_helpers.py`` on the import path always
takes precedence.
"""

from __future__ import annotations

import importlib.util
import sys

_installed = False


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def install() -> None:
    """Register the aliases once per process (idempotent)."""
    global _installed
    if _installed:
        return

    if not _module_available("markdown_splitter"):
        # FallbackSplitter needs the complete synchronous MarkdownSplitter.
        from chunker.adaptive_chunker import markdown_splitter_kiro

        sys.modules["markdown_splitter"] = markdown_splitter_kiro

    if not _module_available("aws_helpers"):
        # Bedrock Converse helpers; reads AWS_BEDROCK_MODEL_ID /
        # AWS_BEDROCK_REASONING_EFFORT from .env.
        from chunker import aws_helpers

        sys.modules["aws_helpers"] = aws_helpers

    _installed = True
