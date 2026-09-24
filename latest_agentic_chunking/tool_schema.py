"""Tool schema for the `extract_chunks` agent contract.

This module exports two values consumed by `AgenticChunker`:

* ``CHUNK_TOOL_SCHEMA`` — a JSON-Schema function-spec dict describing the
  tool the LLM must call to return chunk boundaries. The same dict can
  be wrapped as ``{"type": "function", "function": CHUNK_TOOL_SCHEMA}``
  for both Azure OpenAI (``tools=[...]`` with forced ``tool_choice``)
  and Ollama (``ollama.Client.chat(..., tools=[...])``).

* ``SYSTEM_PROMPT`` — the system message that forces tool-only responses
  and lays out the chunking rules: paragraph/heading boundaries, never
  split inside fenced code blocks or tables, return chunks in document
  order, cover every non-whitespace character of the window, call the
  tool exactly once.

The metadata sub-schema and metadata-generation responsibilities have
been removed from the agent contract — the retrieval stack does not
consume metadata, so producing it on every chunk was pure cost.
``EMPTY_METADATA`` is still attached to each chunk's payload by the
validator / fallback splitter so the Qdrant schema stays stable.
"""

from __future__ import annotations

from typing import Any, Dict


# ──────────────────────────────────────────────
#  TOOL SCHEMA (Agent Contract)
# ──────────────────────────────────────────────
# Required chunk fields:        start, end, title, chunk_type
# Optional chunk fields:        level, primary_type, has_text, has_code, has_table
CHUNK_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "extract_chunks",
    "description": (
        "Split the provided markdown window into semantically coherent chunks. "
        "Boundaries are character offsets into the window text. The union of "
        "chunks must cover all non-whitespace content of the window with no "
        "overlaps."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chunks": {
                "type": "array",
                "description": (
                    "Ordered list of chunks covering the window. Each chunk's "
                    "[start, end) is a character offset range into the window text."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        # Required fields
                        "start": {
                            "type": "integer",
                            "description": (
                                "Char offset (inclusive) within the window. "
                                "Must satisfy 0 <= start < end <= len(window_text)."
                            ),
                        },
                        "end": {
                            "type": "integer",
                            "description": (
                                "Char offset (exclusive) within the window. "
                                "Must satisfy start < end <= len(window_text)."
                            ),
                        },
                        "title": {
                            "type": "string",
                            "description": (
                                "Specific, content-reflective title for the chunk. "
                                "MUST describe what is actually in the chunk — name "
                                "the entities, topics, sections, dates, metrics, or "
                                "events present. Do NOT use generic document-level "
                                "labels like 'Full document', 'Full resume', "
                                "'Content', 'Information', 'Section', 'Chunk', or "
                                "'Document text'. If the chunk covers a whole short "
                                "document, the title should still be specific (e.g. "
                                "'Mahesh Kumar resume — software engineer at Acme', "
                                "not 'Full resume'). 5-12 words. Two different "
                                "chunks should not share the same title unless they "
                                "are siblings of the same heading section."
                            ),
                        },
                        "chunk_type": {
                            "type": "string",
                            "enum": ["text", "code", "table"],
                            "description": (
                                "Dominant content type of the chunk: prose, fenced "
                                "code block, or markdown table."
                            ),
                        },
                        "summary": {
                            "type": "string",
                            "description": (
                                "1-2 sentence summary of the chunk's specific "
                                "content. MUST mention concrete entities, names, "
                                "numbers, dates, or domain terms found in the "
                                "chunk so a retrieval system can match queries "
                                "about those specifics. Do NOT use generic "
                                "phrasing like 'this chunk discusses', 'an "
                                "overview of', 'information about'. Lead with "
                                "the entity or topic, not with framing words. "
                                "Aim for 15-40 words. Examples: 'Arsh Sahay's "
                                "MS Computer Science from IIT Delhi (2020-22), "
                                "with coursework in machine learning and "
                                "distributed systems.' / 'Acme Corp Q3 2023 "
                                "revenue of $1.2M, up 15% YoY, driven by SaaS "
                                "subscriptions in the EMEA region.'"
                            ),
                        },
                        # Optional fields (declared per design)
                        "level": {
                            "type": ["integer", "null"],
                            "description": (
                                "Heading level 1..6 if the chunk starts with a heading, "
                                "otherwise null."
                            ),
                        },
                        "primary_type": {
                            "type": "string",
                            "enum": ["text", "code", "table"],
                            "description": (
                                "Mirrors chunk_type; preserved for retrieval-stack "
                                "schema parity."
                            ),
                        },
                        "has_text": {
                            "type": "boolean",
                            "description": "True if the chunk contains prose text.",
                        },
                        "has_code": {
                            "type": "boolean",
                            "description": "True if the chunk contains a fenced code block.",
                        },
                        "has_table": {
                            "type": "boolean",
                            "description": "True if the chunk contains a markdown table.",
                        },
                    },
                    "required": ["start", "end", "title", "chunk_type", "summary"],
                },
            }
        },
        "required": ["chunks"],
    },
}


# ──────────────────────────────────────────────
#  SYSTEM PROMPT (forces tool-only responses)
# ──────────────────────────────────────────────
SYSTEM_PROMPT: str = (
    "You MUST call the tool extract_chunks.\n"
    "Do NOT respond with text.\n"
    "Do NOT return JSON in the message body.\n"
    "Only call the tool, exactly once.\n"
    "\n"
    "Chunking rules:\n"
    "- Split the provided window into semantically coherent chunks.\n"
    "- Boundaries are character offsets [start, end) into the window text.\n"
    "- Chunks MUST be returned in document order (ascending start).\n"
    "- Chunks MUST NOT overlap.\n"
    "- The union of chunks MUST cover every non-whitespace character of the window.\n"
    "- Prefer paragraph and heading boundaries.\n"
    "- NEVER split inside a fenced code block (triple backticks).\n"
    "- NEVER split inside a markdown table (contiguous lines starting with `|` "
    "OR a region where multiple lines contain several `|` separators inline). "
    "Treat any pipe-heavy region as an atomic table — split only between rows, "
    "never within a row.\n"
    "- Each chunk MUST stay at or below max_chunk_tokens. If a section is "
    "longer, split it at the next paragraph or sentence boundary even if "
    "that splits a logical idea.\n"
    "\n"
    "Title rules (the `title` field for each chunk):\n"
    "- Title MUST reflect the chunk's actual content — name the specific "
    "entities, topics, sections, dates, metrics, or events present.\n"
    "- NEVER use generic document-level titles like 'Full document', "
    "'Full resume', 'Full report', 'Document text', 'Content', "
    "'Information', 'Section', or 'Chunk'.\n"
    "- If the chunk covers a whole short document (e.g. a single-window "
    "resume), the title MUST still be specific. Bad: 'Full resume'. "
    "Good: 'Mahesh Kumar resume — software engineer at Acme'. Bad: "
    "'Full report'. Good: 'Martin & Bayley 2023 valuation — projected "
    "income statements'.\n"
    "- Aim for 5-12 words. Use specific names, numbers, dates, or "
    "domain terms over generic categories.\n"
    "- Two different chunks should not share the same title unless they "
    "are siblings of the same heading section.\n"
    "- If a chunk starts with a real markdown heading (`#`, `##`, ...), "
    "use that heading text as the title and set `level` accordingly.\n"
    "\n"
    "Summary rules (the `summary` field for each chunk):\n"
    "- Summary is a 1-2 sentence description of WHAT IS IN THE CHUNK. It "
    "is used to anchor the chunk's retrieval embedding and is shown to "
    "the reranker, so it must be specific.\n"
    "- Mention the concrete entities, names, numbers, dates, products, "
    "or domain terms found in the chunk content. The summary should be "
    "a high-signal preface that disambiguates this chunk from other "
    "chunks on similar topics from other documents.\n"
    "- NEVER start with framing words like 'This chunk discusses', 'An "
    "overview of', 'Information about', 'Details on', 'A summary of'. "
    "Lead with the entity or fact directly.\n"
    "- Aim for 15-40 words. One or two sentences.\n"
    "- Examples:\n"
    "  Good: 'Arsh Sahay's MS Computer Science from IIT Delhi (2020-22), "
    "with coursework in machine learning and distributed systems.'\n"
    "  Good: 'Acme Corp Q3 2023 revenue of $1.2M, up 15% YoY, driven "
    "by SaaS subscriptions in EMEA.'\n"
    "  Good: 'Steps to set up Zoho Vault: log in via the provided link, "
    "enter the OTP sent to the registered mobile, and confirm sign-in.'\n"
    "  Bad: 'This chunk discusses the user's education.' (generic, no "
    "specific entities)\n"
    "  Bad: 'Information about Q3 financials.' (generic, no concrete "
    "values)"
)


__all__ = ["CHUNK_TOOL_SCHEMA", "SYSTEM_PROMPT"]
