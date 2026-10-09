"""Shared helpers for AWS Bedrock Converse API.

Consumed by both ``retrieval_for_api.AnswerGenerator``/``QueryExpander``
and ``agentic_chunking.agentic_chunker.AgenticChunker`` so we don't
duplicate the OpenAI ⇄ AWS-Converse translation in two places.

Design notes
------------

* **No project imports.** Kept dependency-free so both the retrieval
  module and the chunking subpackage can pull from it without a
  circular import.
* **gpt-oss content shape.** OpenAI's gpt-oss models on Bedrock emit
  a ``reasoningContent`` block BEFORE the ``text``/``toolUse`` block.
  ``extract_aws_text`` and ``extract_aws_tool_use`` iterate every
  block and skip the reasoning one — never index ``[0]`` directly on
  Converse responses.
* **Reasoning effort.** gpt-oss burns output tokens on internal
  reasoning at the default ``medium`` setting; for structured
  extraction we drop it to ``low`` (env-overridable) or the tool call
  never gets emitted.
* **Region.** Not passed to ``boto3.client`` — resolves from the
  standard AWS env / profile chain per project convention.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv


# Load .env once at import time so downstream imports see the vars.
load_dotenv()


# ──────────────────────────────────────────────
#  Env-var constants
# ──────────────────────────────────────────────

#: The Bedrock model ID used across chunker / expander / chat when
#: ``LLM_PROVIDER=aws``. Env-overridable so we can swap models without
#: touching code.
AWS_MODEL_ID: str = os.getenv(
    "AWS_BEDROCK_MODEL_ID", "openai.gpt-oss-120b-1:0"
)

#: gpt-oss reasoning effort — "minimal" | "low" | "medium" | "high".
#: "low" is the sweet spot for structured extraction / rewrites; leave
#: room for the user to bump it for the chat model if needed.
AWS_REASONING_EFFORT: str = os.getenv(
    "AWS_BEDROCK_REASONING_EFFORT", "low"
)


# ──────────────────────────────────────────────
#  Tool schema conversion
# ──────────────────────────────────────────────

def openai_tool_to_aws(tool: dict) -> dict:
    """Convert an OpenAI function spec into an AWS Converse ``toolSpec``.

    Input shape (what ``tools[0]["function"]`` looks like in OpenAI /
    Azure and what ``CHUNK_TOOL_SCHEMA`` is in this repo)::

        {"name": ..., "description": ..., "parameters": {...}}

    Output shape (AWS Converse ``tools`` array item)::

        {"toolSpec": {
            "name": ...,
            "description": ...,
            "inputSchema": {"json": {...}}
        }}
    """
    return {
        "toolSpec": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "inputSchema": {"json": tool.get("parameters", {})},
        }
    }


def openai_tools_to_aws_config(
    tools: list[dict],
    *,
    force_tool_name: str | None = None,
) -> dict:
    """Build a full AWS ``toolConfig`` from a list of OpenAI function specs.

    Parameters
    ----------
    tools:
        List of OpenAI function specs (each with ``name`` / ``description`` /
        ``parameters``). If you have the OpenAI ``tools=[{"type":"function",
        "function":{...}}]`` shape, unwrap the ``function`` field first.
    force_tool_name:
        If set, the returned config uses ``toolChoice={"tool": {"name": ...}}``
        so the model MUST call that specific tool. If None, ``toolChoice``
        is omitted and the model decides.
    """
    config: dict = {"tools": [openai_tool_to_aws(t) for t in tools]}
    if force_tool_name is not None:
        config["toolChoice"] = {"tool": {"name": force_tool_name}}
    return config


# ──────────────────────────────────────────────
#  Message conversion
# ──────────────────────────────────────────────

def openai_messages_to_aws(
    messages: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split an OpenAI-shaped message list into AWS ``(system, messages)``.

    AWS Converse expects the system prompt(s) in a separate ``system``
    kwarg — a list of ``[{"text": "..."}]`` blocks — not inline in the
    ``messages`` array. Multiple consecutive system messages are
    concatenated with ``\\n\\n`` into a single block.

    ``user`` and ``assistant`` messages become
    ``{"role": ..., "content": [{"text": ...}]}``. Any ``tool`` role
    messages are DROPPED — AWS uses a different convention for tool
    results (``role: "user"`` with a ``toolResult`` content block),
    and the caller is expected to have already built those in the
    AWS shape.
    """
    system_blocks: list[dict] = []
    aws_messages: list[dict] = []
    system_parts: list[str] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "") or ""

        if role == "system":
            if content:
                system_parts.append(content)
            continue

        if role in ("user", "assistant"):
            aws_messages.append({
                "role": role,
                "content": [{"text": content}],
            })
            continue

        # Any other role (tool, function, developer, etc.) is skipped
        # here — Converse tool loops rebuild these on the AWS side.

    if system_parts:
        system_blocks.append({"text": "\n\n".join(system_parts)})

    return system_blocks, aws_messages


# ──────────────────────────────────────────────
#  Response block extraction
# ──────────────────────────────────────────────

def extract_aws_text(content_blocks: list[dict]) -> str:
    """Concatenate all ``text`` blocks in an AWS Converse response.

    Skips ``reasoningContent`` and ``toolUse`` blocks. Returns an empty
    string when no text block is present (e.g. the model went straight
    into a tool call).
    """
    parts: list[str] = []
    for block in content_blocks or []:
        if "text" in block:
            parts.append(block["text"])
    return "".join(parts)


def extract_aws_tool_use(content_blocks: list[dict]) -> dict | None:
    """Return the first ``toolUse`` block dict, or None.

    The returned dict has keys ``toolUseId``, ``name``, ``input``.
    """
    for block in content_blocks or []:
        if "toolUse" in block:
            return block["toolUse"]
    return None


def extract_aws_reasoning(content_blocks: list[dict]) -> str:
    """Concatenate ``reasoningContent`` text.

    Kept as a helper for debugging / logging; not used in the hot path
    since we don't surface reasoning to end users.
    """
    parts: list[str] = []
    for block in content_blocks or []:
        rc = block.get("reasoningContent")
        if not rc:
            continue
        rt = rc.get("reasoningText") or {}
        text = rt.get("text")
        if text:
            parts.append(text)
    return "\n".join(parts)


# ──────────────────────────────────────────────
#  Convenience: standard additionalModelRequestFields
# ──────────────────────────────────────────────

def default_additional_fields(
    reasoning_effort: str | None = None,
) -> dict:
    """Return the standard ``additionalModelRequestFields`` dict for gpt-oss.

    Currently only carries ``reasoning_effort``. Isolated here so the
    knob has a single source of truth and future additions (e.g. a
    ``verbosity`` field) can be added without touching every call site.
    """
    return {"reasoning_effort": reasoning_effort or AWS_REASONING_EFFORT}


__all__ = [
    "AWS_MODEL_ID",
    "AWS_REASONING_EFFORT",
    "openai_tool_to_aws",
    "openai_tools_to_aws_config",
    "openai_messages_to_aws",
    "extract_aws_text",
    "extract_aws_tool_use",
    "extract_aws_reasoning",
    "default_additional_fields",
]
