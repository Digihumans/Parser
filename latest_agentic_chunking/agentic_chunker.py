"""AgenticChunker — LLM-driven chunk boundary + metadata extraction.

Implements Algorithm 3 from ``design.md``. One LLM call per window via
the ``extract_chunks_with_metadata`` tool. Returns validated raw chunks
ready for ``ChunkAssembler``.

Design highlights
-----------------

* **Lazy provider clients.** Constructed once in ``__init__`` and reused
  across windows / threads. Mirrors ``embed.py``'s ``MetadataGenerator``.
* **Forced tool calls.** Azure path uses
  ``tool_choice={"type": "function", "function": {"name": "..."}}``;
  Ollama path uses ``tools=[CHUNK_TOOL_SCHEMA]`` with ``think=False`` and
  the configured ``num_ctx``.
* **Rate-limit semaphore.** Acquired before every provider call and
  released in ``finally`` so a raise still releases the permit.
* **Retry loop.** On no-tool-call, invalid JSON, validator-fail, or
  HTTP 429/500/503, append a corrective system message and sleep
  ``2 ** attempt`` seconds (1s, 2s, 4s — matches ``MetadataGenerator``)
  before retrying. Exhaustion raises ``AgenticChunkerError`` (the
  caller's signal to route to the fallback splitter).
* **Pure on input.** ``window`` is never mutated; ``content`` is always
  re-sliced from ``window.text`` and any value the model returns for
  ``content`` is ignored (Req 4.3).
* **Producer tag.** Every emitted chunk carries ``producer="agent"`` so
  ``FileIngestor`` can route metadata-generation correctly — only
  fallback chunks need ``MetadataGenerator``.
* **Oversized windows.** When ``window.oversized=True`` boundary
  detection is skipped — a single chunk covering the whole window is
  emitted with empty-defaults metadata. ``ChunkValidator`` relaxes its
  token bound for this case (Req 12.3) so the chunk is acceptable
  downstream.

The model parameter, when ``None``, defaults to the same env vars
``embed.py`` consults (``AZURE_OPENAI_FAST_DEPLOYMENT`` for Azure,
``OLLAMA_CHAT_MODEL`` for Ollama) so callers can use ``LLM_PROVIDER``
without hard-coding deployment names.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from typing import Any

from dotenv import load_dotenv

from agentic_chunking.chunk_validator import ChunkValidator
from agentic_chunking.data_models import (
    EMPTY_METADATA,
    AgenticChunkerError,
    Window,
)
from agentic_chunking.tool_schema import CHUNK_TOOL_SCHEMA, SYSTEM_PROMPT

# AWS Converse translation helpers — kept in a top-level module so
# both agentic_chunking and retrieval_for_api can share the code
# without a circular import.
from aws_helpers import (
    AWS_MODEL_ID as _AWS_MODEL_ID_DEFAULT,
    openai_tools_to_aws_config,
    extract_aws_tool_use,
    default_additional_fields,
)


# Load .env at import time. Mirrors embed.py so AzureOpenAI / Ollama env
# vars are available when the constructor lazily-instantiates the client.
load_dotenv()


# ──────────────────────────────────────────────
#  Provider configuration constants
#  (mirror embed.py — same env-var names + defaults)
# ──────────────────────────────────────────────

# Azure OpenAI client config.
_AZURE_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
_AZURE_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
_AZURE_API_VERSION: str = os.getenv(
    "AZURE_OPENAI_API_VERSION", "2025-03-01-preview"
)

# Default chat model when ``model`` is None on the Azure path.
_AZURE_FAST_DEPLOYMENT_DEFAULT: str = os.getenv(
    "AZURE_OPENAI_FAST_DEPLOYMENT", "gpt-5-mini"
)

# Default chat model when ``model`` is None on the Ollama path. embed.py
# defines OLLAMA_CHAT_MODEL as a hard-coded constant — we mirror the
# value but allow an env override so test/dev environments can swap in
# another tag without editing code.
_OLLAMA_CHAT_MODEL_DEFAULT: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:4b")

# Ollama context window for the chat tool call. Mirrors
# OLLAMA_CONTEXT_WINDOW in embed.py — large enough to fit
# agent_context_budget_tokens + system prompt overhead at the Ollama
# default budget (4000 tokens — see config.PipelineConfig).
_OLLAMA_NUM_CTX: int = 6144


# ──────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Tool schema wrappers (Azure / Ollama call shape)
# ──────────────────────────────────────────────

_TOOL_SPEC: dict = {"type": "function", "function": CHUNK_TOOL_SCHEMA}
_TOOL_NAME: str = CHUNK_TOOL_SCHEMA["name"]
_FORCED_TOOL_CHOICE: dict = {
    "type": "function",
    "function": {"name": _TOOL_NAME},
}

# AWS Converse ``toolConfig`` for the chunker. Built once at import
# time from the same ``CHUNK_TOOL_SCHEMA`` used by Azure/Ollama, then
# reused across every window / thread. Forced ``toolChoice`` so
# gpt-oss can't skip the tool call.
_AWS_TOOL_CONFIG: dict = openai_tools_to_aws_config(
    [CHUNK_TOOL_SCHEMA], force_tool_name=_TOOL_NAME,
)


# Substrings used to detect retryable HTTP errors. Mirrors the same
# pattern ``MetadataGenerator`` uses today in ``embed.py`` — keep these
# in sync if the upstream detection logic changes.
_RETRYABLE_ERROR_TOKENS: tuple[str, ...] = ("429", "500", "503")


def _is_retryable_provider_error(message: str) -> bool:
    """Return True for messages matching the retry policy in embed.py."""
    if not message:
        return False
    if "rate" in message.lower():
        return True
    return any(tok in message for tok in _RETRYABLE_ERROR_TOKENS)


class AgenticChunker:
    """Calls the LLM once per window to extract chunks + inline metadata."""

    def __init__(
        self,
        provider: str,
        model: str | None,
        max_chunk_tokens: int,
        min_chunk_tokens: int,
        max_retries: int = 3,
        *,
        rate_limit_semaphore: threading.Semaphore,
        validator: ChunkValidator,
    ) -> None:
        normalized = (provider or "").strip().lower()
        if normalized not in ("azure", "ollama", "aws"):
            raise ValueError(
                f"Unknown provider: {provider!r}. "
                f"Expected 'azure', 'ollama', or 'aws'."
            )
        self.provider: str = normalized
        self.max_chunk_tokens: int = int(max_chunk_tokens)
        self.min_chunk_tokens: int = int(min_chunk_tokens)
        self.max_retries: int = int(max_retries)
        self.rate_limit_semaphore: threading.Semaphore = rate_limit_semaphore
        self.validator: ChunkValidator = validator

        if self.max_retries <= 0:
            raise ValueError(
                f"max_retries must be positive, got {self.max_retries}"
            )

        # Resolve model default lazily from env so the same pattern works
        # across all provider paths.
        if model is None:
            if self.provider == "azure":
                self.model: str = _AZURE_FAST_DEPLOYMENT_DEFAULT
            elif self.provider == "aws":
                self.model = _AWS_MODEL_ID_DEFAULT
            else:
                self.model = _OLLAMA_CHAT_MODEL_DEFAULT
        else:
            self.model = model

        # Lazy-init provider client (constructed once, reused across
        # windows + threads). Imports are deferred so a deployment using
        # only one provider doesn't pay the import cost of the other.
        if self.provider == "azure":
            from openai import AzureOpenAI

            self.client = AzureOpenAI(
                azure_endpoint=_AZURE_ENDPOINT,
                api_key=_AZURE_API_KEY,
                api_version=_AZURE_API_VERSION,
            )
        elif self.provider == "aws":
            import boto3

            # No region_name — resolves from AWS env / profile chain.
            self.client = boto3.client("bedrock-runtime")
        else:
            import ollama

            self.client = ollama.Client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_window(
        self,
        window: Window,
        doc_id: str,
        source: str,
    ) -> list[dict]:
        """Run the agent on ``window`` and return validated raw chunks.

        Does not mutate ``window``. Tags every emitted chunk with
        ``producer="agent"`` and ``split_group_id=window.window_index``.

        Args:
            window: The pre-segmented window to chunk.
            doc_id: Document identifier propagated to each chunk.
            source: Source path / filename propagated to each chunk.

        Returns:
            A list of validated raw-chunk dicts ready for
            ``ChunkAssembler``. Always non-empty when this method
            returns successfully.

        Raises:
            AgenticChunkerError: When ``max_retries`` attempts fail to
                produce a valid response. The caller (``FileIngestor``)
                catches this and routes the window to the fallback
                splitter so one bad window does not abort the file.
        """
        # Oversized window short-circuit (Req 12.2): skip boundary
        # detection entirely and emit a single chunk covering the whole
        # window with empty-defaults metadata. ``ChunkValidator`` relaxes
        # its token bound for ``window.oversized=True`` (Req 12.3) so
        # this chunk is acceptable downstream.
        if window.oversized:
            return [self._build_oversized_chunk(window, doc_id, source)]

        feedback: str = ""

        for attempt in range(self.max_retries):
            messages = self._build_messages(window, feedback)

            # Provider call. Acquire/release happens in _call_provider
            # so a raise still releases the semaphore permit. HTTP
            # 429/500/503 are detected here and routed through the same
            # backoff path as logical failures.
            try:
                response = self._call_provider(messages)
            except Exception as exc:
                error_str = str(exc)
                if _is_retryable_provider_error(error_str):
                    reasons = [f"provider HTTP error: {error_str[:200]}"]
                    self._log_retry(window, attempt, reasons)
                    feedback = (
                        "The previous request failed with a provider "
                        "error (rate limit / 5xx). Try again."
                    )
                    self._sleep_for_attempt(attempt)
                    continue
                # Non-retryable — propagate so the caller sees the real
                # exception rather than a misleading retry-exhaustion.
                raise

            # 1. No tool call.
            tool_args_obj = self._extract_tool_arguments(response)
            if tool_args_obj is None:
                reasons = ["model did not call extract_chunks"]
                self._log_retry(window, attempt, reasons)
                feedback = (
                    "You did not call the tool. "
                    "You MUST call extract_chunks."
                )
                self._sleep_for_attempt(attempt)
                continue

            # 2. Invalid JSON / wrong shape.
            parsed = self._parse_tool_arguments(tool_args_obj)
            if parsed is None or not isinstance(parsed, dict):
                reasons = ["tool arguments were not valid JSON"]
                self._log_retry(window, attempt, reasons)
                feedback = (
                    "Your tool arguments were not valid JSON. "
                    "Return a single tool call whose arguments are a JSON "
                    "object with a 'chunks' array."
                )
                self._sleep_for_attempt(attempt)
                continue

            raw_chunks = parsed.get("chunks")
            if not isinstance(raw_chunks, list):
                reasons = ["tool arguments missing 'chunks' array"]
                self._log_retry(window, attempt, reasons)
                feedback = (
                    "Your tool arguments must include a 'chunks' array."
                )
                self._sleep_for_attempt(attempt)
                continue

            # 3. Slice content from window.text and tag pipeline fields.
            #    The model's content (if any) is ignored per Req 4.3 —
            #    the validator's auto-repair will also realign content
            #    to ``window.text[start:end]`` after any boundary trim.
            prepared, slice_error = self._prepare_chunks(
                raw_chunks, window, doc_id, source
            )
            if slice_error is not None:
                self._log_retry(window, attempt, [slice_error])
                feedback = "Each chunk must be a JSON object."
                self._sleep_for_attempt(attempt)
                continue

            # 4. Validate. The validator deepcopies ``prepared`` so our
            #    pipeline tags (split_group_id / doc_id / source) ride
            #    along through the auto-repair.
            validation = self.validator.validate(prepared, window)
            if validation.is_valid:
                final_chunks = validation.repaired_chunks
                # Defensive: validator contract says repaired_chunks is
                # non-null on success but fall back to ``prepared`` if
                # an alternative implementation returns None.
                if final_chunks is None:
                    final_chunks = prepared
                for ch in final_chunks:
                    ch["producer"] = "agent"
                return final_chunks

            reasons = list(validation.error_reasons)
            self._log_retry(window, attempt, reasons)
            feedback = (
                "Your boundaries were invalid: " + "; ".join(reasons)
            )
            self._sleep_for_attempt(attempt)

        # Exhausted retries — caller falls back.
        raise AgenticChunkerError(
            f"max retries reached for window {window.window_index}"
        )

    # ------------------------------------------------------------------
    # Provider call — wraps semaphore acquire/release
    # ------------------------------------------------------------------

    def _call_provider(self, messages: list[dict]) -> Any:
        """Wrap a single provider call with the rate-limit semaphore.

        The semaphore permit is released in ``finally`` so a raise
        (network, auth, rate-limit) still gives the permit back. The
        caller (``chunk_window``) is responsible for catching the
        exception and routing it to the retry path.
        """
        self.rate_limit_semaphore.acquire()
        try:
            if self.provider == "azure":
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=[_TOOL_SPEC],
                    tool_choice=_FORCED_TOOL_CHOICE,
                )
            if self.provider == "aws":
                # Split system out of messages into AWS's separate
                # ``system`` kwarg. We don't use the general
                # ``openai_messages_to_aws`` helper here because we
                # know the shape statically — messages is always
                # ``[system, user]``.
                system_text = messages[0]["content"]
                user_text = messages[1]["content"]
                # ``maxTokens=16000`` explicitly requests gpt-oss's full
                # per-response budget. Omitting the field ends up
                # capping the response at Bedrock's much smaller
                # default (~4K), which truncated the tool call
                # mid-way through the ``chunks`` array on long
                # windows and forced the validator to reject the
                # response with "uncovered range after the last
                # chunk" — pushing the window to fallback. Reasoning
                # effort stays at ``low`` (default from env) so the
                # model doesn't burn the enlarged budget on
                # deliberation before emitting the tool call.
                return self.client.converse(
                    modelId=self.model,
                    system=[{"text": system_text}],
                    messages=[{
                        "role": "user",
                        "content": [{"text": user_text}],
                    }],
                    inferenceConfig={
                        "temperature": 0.2,
                        "maxTokens": 16000,
                    },
                    toolConfig=_AWS_TOOL_CONFIG,
                    additionalModelRequestFields=default_additional_fields(),
                )
            # Ollama path
            return self.client.chat(
                model=self.model,
                messages=messages,
                tools=[_TOOL_SPEC],
                think=False,
                options={"num_ctx": _OLLAMA_NUM_CTX, "temperature": 0.4},
            )
        finally:
            self.rate_limit_semaphore.release()

    # ------------------------------------------------------------------
    # Tool-call extraction (provider-specific)
    # ------------------------------------------------------------------

    def _extract_tool_arguments(self, response: Any) -> Any:
        """Pull the tool-call arguments out of a provider response.

        Returns the raw arguments object (str for Azure, dict for
        Ollama) or ``None`` if no tool call was made / the response
        shape is unexpected. Catches attribute / index errors so a
        malformed response routes to the no-tool-call retry path
        rather than crashing the worker.
        """
        try:
            if self.provider == "azure":
                choice = response.choices[0]
                tool_calls = getattr(choice.message, "tool_calls", None)
                if not tool_calls:
                    return None
                return tool_calls[0].function.arguments
            if self.provider == "aws":
                # AWS Converse: iterate content blocks — the first is
                # usually ``reasoningContent`` for gpt-oss, the toolUse
                # block comes later. ``extract_aws_tool_use`` handles
                # this. Return the ``input`` dict directly — the
                # downstream ``_parse_tool_arguments`` accepts dicts.
                content_blocks = (
                    response.get("output", {})
                    .get("message", {})
                    .get("content", [])
                )
                tool_use = extract_aws_tool_use(content_blocks)
                if tool_use is None:
                    return None
                return tool_use.get("input")
            # Ollama path
            tool_calls = response.message.tool_calls
            if not tool_calls:
                return None
            return tool_calls[0].function.arguments
        except (IndexError, AttributeError, KeyError):
            return None

    @staticmethod
    def _parse_tool_arguments(args_obj: Any) -> Any:
        """Coerce the raw arguments object into a dict.

        Azure returns a JSON string; Ollama already returns a dict (or
        a Pydantic model exposing dict semantics). Returns ``None`` on
        parse failure so the caller can route to the invalid-JSON
        retry path.
        """
        if isinstance(args_obj, dict):
            return args_obj
        if isinstance(args_obj, str):
            try:
                return json.loads(args_obj)
            except (json.JSONDecodeError, ValueError):
                return None
        # Some clients wrap arguments in a Pydantic-like object — try a
        # final-resort dict coercion before giving up.
        if hasattr(args_obj, "model_dump"):
            try:
                value = args_obj.model_dump()
                return value if isinstance(value, dict) else None
            except Exception:  # noqa: BLE001 — best-effort coercion
                return None
        return None

    # ------------------------------------------------------------------
    # Chunk preparation + oversized-window short-circuit
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_chunks(
        raw_chunks: list[Any],
        window: Window,
        doc_id: str,
        source: str,
    ) -> tuple[list[dict], str | None]:
        """Slice ``content`` from ``window.text`` and tag pipeline fields.

        Returns ``(prepared, error)``. ``error`` is ``None`` on success;
        otherwise a string suitable for the corrective feedback message
        (and the retry log entry) and ``prepared`` is empty.

        We deep-copy each chunk so the validator's downstream deepcopy
        is the only modifier of these dicts — but more importantly so
        any mutable defaults the model echoed back (e.g. shared empty
        lists) don't alias across chunks. The agent's ``content`` value
        is overwritten unconditionally per Req 4.3.
        """
        prepared: list[dict] = []
        text = window.text
        text_len = len(text)
        for ch in raw_chunks:
            if not isinstance(ch, dict):
                return [], "one or more chunks are not objects"
            ch = copy.deepcopy(ch)
            start = ch.get("start")
            end = ch.get("end")
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= text_len
            ):
                ch["content"] = text[start:end]
            # If start/end are out of range we leave ``content`` whatever
            # the model returned — the validator will emit the right
            # range-violation diagnostic, so we do NOT crash here.
            ch["split_group_id"] = window.window_index
            ch["doc_id"] = doc_id
            ch["source"] = source
            prepared.append(ch)
        return prepared, None

    @staticmethod
    def _build_oversized_chunk(
        window: Window,
        doc_id: str,
        source: str,
    ) -> dict:
        """Construct the single chunk emitted for an oversized window.

        Oversized windows by definition contain a fenced code block or a
        markdown table whose token count alone exceeds the agent
        context budget (Req 12.1). We cannot fit the content into the
        model's context, so boundary detection is skipped and we emit
        one chunk covering the whole window with empty-defaults
        metadata. ``chunk_type`` is inferred from the content shape so
        the retrieval-stack flags (``has_code`` / ``has_table``) are
        still correct.
        """
        chunk_type = AgenticChunker._infer_oversized_chunk_type(window.text)
        title = (
            window.leading_heading_path[-1]
            if window.leading_heading_path
            else "Oversized block"
        )
        return {
            "start": 0,
            "end": len(window.text),
            "title": title,
            "summary": "",
            "level": None,
            "chunk_type": chunk_type,
            "primary_type": chunk_type,
            "has_text": chunk_type == "text",
            "has_code": chunk_type == "code",
            "has_table": chunk_type == "table",
            "content": window.text,
            # Fresh deep copy so empty list defaults can't alias across
            # chunks — same precaution as ``FallbackSplitter`` takes.
            "metadata": copy.deepcopy(EMPTY_METADATA),
            "split_group_id": window.window_index,
            "doc_id": doc_id,
            "source": source,
            "producer": "agent",
        }

    @staticmethod
    def _infer_oversized_chunk_type(text: str) -> str:
        """Best-effort detection of an oversized window's chunk type.

        The pre-segmenter only flags a window oversized when it contains
        a fenced code block or a markdown table on its own, so the two
        cheap heuristics below cover every realistic input:

        * Fenced code block — the leading non-whitespace token is the
          fence delimiter ``` ``` ```.
        * Markdown table — at least half of the non-empty lines start
          with ``|``.

        Falls back to ``"text"`` for anything else (defensive — should
        not happen in practice).
        """
        stripped = text.lstrip()
        if stripped.startswith("```"):
            return "code"
        non_empty_lines = [l.strip() for l in text.splitlines() if l.strip()]
        if non_empty_lines:
            pipe_lines = sum(
                1 for l in non_empty_lines if l.startswith("|")
            )
            if pipe_lines * 2 >= len(non_empty_lines):
                return "table"
        return "text"

    # ------------------------------------------------------------------
    # Prompt + retry helpers
    # ------------------------------------------------------------------

    def _build_messages(self, window: Window, feedback: str) -> list[dict]:
        """Build the chat messages for one provider call.

        The system message stacks the canonical ``SYSTEM_PROMPT``, a
        per-call note about the chunk-token budget, and (on retries)
        the corrective feedback identifying what went wrong on the
        previous attempt. The user message carries the window's
        heading-path context so the model can title chunks consistently
        with earlier windows, and the window text itself wrapped in
        BEGIN/END markers so character offsets are unambiguous.
        """
        sys_parts: list[str] = [SYSTEM_PROMPT]
        sys_parts.append(
            f"Each chunk should target between {self.min_chunk_tokens} and "
            f"{self.max_chunk_tokens} tokens of content where the "
            f"semantics allow it."
        )
        if feedback:
            sys_parts.append(feedback)

        heading_chain = (
            " > ".join(window.leading_heading_path)
            if window.leading_heading_path
            else "(none)"
        )

        user_content = (
            f"Heading context active at this window's start: "
            f"{heading_chain}\n"
            f"Window text length (characters): {len(window.text)}\n"
            f"Return character offsets [start, end) into the window text "
            f"shown between the BEGIN/END markers below.\n"
            f"--- BEGIN WINDOW TEXT ---\n"
            f"{window.text}\n"
            f"--- END WINDOW TEXT ---"
        )

        return [
            {"role": "system", "content": "\n\n".join(sys_parts)},
            {"role": "user", "content": user_content},
        ]

    def _log_retry(
        self,
        window: Window,
        attempt: int,
        error_reasons: list[str],
    ) -> None:
        """Emit the retry log entry mandated by Req 16.3.

        ``attempt`` here is the zero-indexed iteration counter; the log
        reports ``attempt + 1`` so attempt numbers in messages start at
        1, matching the requirement's wording.
        """
        logger.warning(
            "AgenticChunker retry: window_index=%d attempt=%d reasons=%s",
            window.window_index,
            attempt + 1,
            error_reasons,
        )

    @staticmethod
    def _sleep_for_attempt(attempt: int) -> None:
        """Sleep ``2 ** attempt`` seconds between attempts.

        ``attempt`` is the zero-based index of the iteration that just
        failed, so the wait sequence is 1s (after attempt 0), 2s (after
        attempt 1), 4s (after attempt 2). Matches the pattern
        ``MetadataGenerator`` uses today (Req 4.9).
        """
        time.sleep(2 ** attempt)
