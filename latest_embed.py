import json
import os
import threading
import time
from pathlib import Path
from dotenv import load_dotenv

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    SparseVectorParams,
    SparseVector
)
from fastembed import SparseTextEmbedding

from markdown_splitter import MarkdownSplitter
from latest_agentic_chunking.agentic_chunker import AgenticChunker
from latest_agentic_chunking.chunk_assembler import ChunkAssembler
from latest_agentic_chunking.chunk_validator import ChunkValidator
from latest_agentic_chunking.config import build_config
from latest_agentic_chunking.fallback_splitter import FallbackSplitter
from latest_agentic_chunking.file_ingestor import FileIngestor
from latest_agentic_chunking.orchestrator import IngestionOrchestrator
from latest_agentic_chunking.paragraph_pre_segmenter import ParagraphPreSegmenter
from latest_agentic_chunking.tokenizer_factory import build_tokenizer

load_dotenv()

print("Initiate")
print(__name__)


# ──────────────────────────────────────────────
#  PROVIDER CONFIG
# ──────────────────────────────────────────────

# Toggle: "ollama" or "azure"
PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Ollama settings
OLLAMA_CHAT_MODEL = "qwen3.5:4b"
OLLAMA_EMBED_MODEL = "qwen3-embedding"
OLLAMA_EMBED_DIMENSIONS = 4096
OLLAMA_CONTEXT_WINDOW = 6144
OLLAMA_MAX_CHUNK_TOKENS = 500  # needed to fit tool-calling overhead
OLLAMA_MIN_CHUNK_TOKENS = 200  # merge fragments below this when possible

# Azure OpenAI settings
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-12-01-preview")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5-mini")
AZURE_FAST_DEPLOYMENT = os.getenv("AZURE_OPENAI_FAST_DEPLOYMENT", "gpt-5-mini")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
AZURE_EMBED_DIMENSIONS = 3072 if AZURE_EMBED_DEPLOYMENT=="text-embedding-3-large" else 1536 # text-embedding-3-large default
AZURE_MAX_CHUNK_TOKENS = 1200  # ceiling — bigger than this hurts rerank precision
AZURE_MIN_CHUNK_TOKENS = 300   # floor — merge fragments below this when adjacent


# ──────────────────────────────────────────────
#  METADATA GENERATOR
# ──────────────────────────────────────────────

class MetadataGenerator:
    """
    Generates structured metadata per chunk via tool-calling.
    Supports Ollama and Azure OpenAI backends.
    """

    TOOL_SCHEMA = {
        "name": "extract_chunk_metadata",
        "description": "Extract structured metadata from a documentation chunk",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A one-line summary of the chunk content",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relevant search keywords",
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Broader thematic categories",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Named entities: people, organizations, products",
                },
                "references": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific laws, sections, acts, circulars, form numbers, standards",
                },
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Hypothetical user queries this chunk could answer",
                },
                "document_type": {
                    "type": "string",
                    "description": "Type of document, e.g. training guide, audit checklist, legal advisory, employee handbook",
                },
                "action_items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Instructions, steps, or action items found in the chunk",
                },
            },
            "required": [
                "summary", "keywords", "topics", "entities",
                "references", "questions", "document_type", "action_items",
            ],
        },
    }

    EMPTY_METADATA = {
        "summary": "",
        "keywords": [],
        "topics": [],
        "entities": [],
        "references": [],
        "questions": [],
        "document_type": "",
        "action_items": [],
    }

    SYSTEM_PROMPT = (
        "You MUST call the tool extract_chunk_metadata.\n"
        "Do NOT respond with text.\n"
        "Do NOT return JSON.\n"
        "Only call the tool."
    )

    def __init__(self, provider: str = "ollama"):
        # ``aws`` is treated as an Azure alias inside this class. The
        # metadata-generation path is dead code in the new pipeline
        # (metadata was removed from ``extract_chunks`` — only
        # ``summary`` remains, and it is emitted by AgenticChunker /
        # FallbackSplitter directly). We keep the class + module-scope
        # instance around only so imports don't break; if it ever
        # ends up being called on the aws path, an Azure client is
        # a reasonable fallback.
        self.provider = "azure" if provider == "aws" else provider

        if self.provider == "ollama":
            import ollama
            self.client = ollama.Client()
        elif self.provider == "azure":
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                api_version=AZURE_API_VERSION,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, chunk: dict, context: int | None = 6144, max_retries: int = 3) -> dict:
        title = chunk.get("title", "") or ""
        content = chunk.get("content", "")
        parent_titles = chunk.get("parent_titles", {})

        hierarchy = " > ".join(parent_titles.values())
        embedding_text = f"{hierarchy}\n{title}\n{content}".strip()

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"The text is provided below:\n{embedding_text}"},
        ]

        for attempt in range(max_retries):
            try:
                if self.provider == "ollama":
                    return self._generate_ollama(messages, context or 6144)
                else:
                    return self._generate_azure(messages)
            except Exception as e:
                error_str = str(e)
                # retry on rate limit (429) or server errors (5xx)
                if "429" in error_str or "rate" in error_str.lower() or "500" in error_str or "503" in error_str:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    print(f"  Retry {attempt + 1}/{max_retries} after {wait}s — {error_str[:100]}")
                    time.sleep(wait)
                else:
                    print(f"  Metadata generation failed: {error_str[:200]}")
                    return self.EMPTY_METADATA.copy()

        print(f"  Metadata generation failed after {max_retries} retries")
        return self.EMPTY_METADATA.copy()

    def _generate_ollama(self, messages: list, context: int) -> dict:
        ollama_tool = {
            "type": "function",
            "function": self.TOOL_SCHEMA,
        }

        response = self.client.chat(
            model=OLLAMA_CHAT_MODEL,
            messages=messages,
            think=False,
            tools=[ollama_tool],
            options={"num_ctx": context, "temperature": 0.4},
        )

        tool_calls = response.message.tool_calls
        if tool_calls:
            metadata = tool_calls[0].function.arguments
            return self._ensure_keys(metadata)

        print(f"No tool called — falling back to empty metadata")
        print(f"  Response: {response.message.content[:200]}")
        return self.EMPTY_METADATA.copy()

    def _generate_azure(self, messages: list) -> dict:
        azure_tool = {
            "type": "function",
            "function": self.TOOL_SCHEMA,
        }

        response = self.client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
            messages=messages,
            tools=[azure_tool],
            tool_choice={"type": "function", "function": {"name": "extract_chunk_metadata"}},
        )

        choice = response.choices[0]
        if choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            metadata = json.loads(tool_call.function.arguments)
            return self._ensure_keys(metadata)

        print(f"No tool called — falling back to empty metadata")
        print(f"  Response: {choice.message.content[:200] if choice.message.content else 'None'}")
        return self.EMPTY_METADATA.copy()

    def _ensure_keys(self, metadata: dict) -> dict:
        for key, default in self.EMPTY_METADATA.items():
            if key not in metadata:
                metadata[key] = default
        return metadata


# ──────────────────────────────────────────────

class SparseEmbeddingGenerator:
    """
    Generates sparse embeddings (BM25) using fastembed for Hybrid Search.
    """
    def __init__(self, model_name="Qdrant/bm25"):
        self.model = SparseTextEmbedding(model_name=model_name)

    def embed(self, chunks: list[dict]) -> list[dict]:
        texts = [c["embedding_text"] for c in chunks]
        embeddings = list(self.model.embed(texts))
        for chunk, embedding in zip(chunks, embeddings):
            chunk["sparse_embedding"] = SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist()
            )
        return chunks

# ──────────────────────────────────────────────
#  EMBEDDING GENERATOR
# ──────────────────────────────────────────────

class EmbeddingGenerator:
    """
    Generates embeddings per chunk.
    Supports Ollama and Azure OpenAI backends.
    """

    def __init__(self, provider: str = "ollama"):
        # Embeddings stay on Azure (``text-embedding-3-large``)
        # regardless of ``LLM_PROVIDER`` — the chat / chunker / expander
        # can run on AWS Bedrock while the vector space remains
        # consistent with existing indexed chunks. Alias ``aws`` to
        # ``azure`` inside this class so the module-scope
        # ``EmbeddingGenerator(provider=PROVIDER)`` still works when
        # ``LLM_PROVIDER=aws``.
        self.provider = "azure" if provider == "aws" else provider

        if self.provider == "ollama":
            import ollama
            self.client = ollama.Client()
        elif self.provider == "azure":
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                api_version=AZURE_EMBEDDING_API_VERSION,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def embed(self, chunks: list[dict], context: int | None = 6144) -> list[dict]:
        if self.provider == "ollama":
            return self._embed_ollama(chunks, context or 6144)
        else:
            return self._embed_azure(chunks)

    def _embed_ollama(self, chunks: list[dict], context: int) -> list[dict]:
        for chunk in tqdm(chunks, desc="embedding generation"):
            response = self.client.embed(
                model=OLLAMA_EMBED_MODEL,
                input=chunk["embedding_text"],
                options={"num_ctx": context},
            )
            chunk["embedding"] = response.embeddings[0]
        return chunks

    def _embed_azure(self, chunks: list[dict]) -> list[dict]:
        batch_size = 16
        max_retries = 3
        # text-embedding-3-large hard limit is 8192 tokens; leave a
        # 192-token safety margin for tokenizer drift between cl100k_base
        # locally and Azure's server-side count.
        embed_token_cap = 8000

        # Lazy-load the cl100k_base encoder once for truncation. Imported
        # inside the method so a deployment using only Ollama doesn't pay
        # the import cost.
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")

        def _truncate(text: str) -> str:
            tokens = encoder.encode(text)
            if len(tokens) <= embed_token_cap:
                return text
            return encoder.decode(tokens[:embed_token_cap])

        for i in tqdm(range(0, len(chunks), batch_size), desc="embedding generation"):
            batch = chunks[i:i + batch_size]
            texts = [_truncate(c["embedding_text"]) for c in batch]

            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=AZURE_EMBED_DEPLOYMENT,
                        input=texts,
                    )

                    for j, embedding_data in enumerate(response.data):
                        batch[j]["embedding"] = embedding_data.embedding
                    break  # success
                except Exception as e:
                    error_str = str(e)
                    if ("429" in error_str or "rate" in error_str.lower() or "500" in error_str) and attempt < max_retries - 1:
                        wait = 2 ** attempt
                        print(f"\n  Embed retry {attempt + 1}/{max_retries} after {wait}s — {error_str[:100]}")
                        time.sleep(wait)
                    else:
                        print(f"\n  Embedding failed for batch {i}: {error_str[:200]}")
                        # fill with empty vectors so pipeline doesn't crash
                        for chunk in batch:
                            if "embedding" not in chunk:
                                chunk["embedding"] = [0.0] * AZURE_EMBED_DIMENSIONS
                        break

        return chunks


# ──────────────────────────────────────────────
#  QDRANT
# ──────────────────────────────────────────────

class Qdrant:

    def __init__(self):
        url=os.getenv("QDRANT_URL")
        api_key=os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url, api_key=api_key)

    def upsert(self, collection_name: str, chunks: list[dict], vector_size: int = 4096) -> tuple[bool, str]:
        try:
            if not self.client.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    sparse_vectors_config={"text": SparseVectorParams()}
                )

                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="doc_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="kb_ids",
                    field_schema=PayloadSchemaType.KEYWORD,
                )



        except Exception as e:
            return True, f"Collection creation failed: {e}"

        points = []
        for chunk in chunks:
            vector_payload = {"": chunk["embedding"]}
            if "sparse_embedding" in chunk:
                vector_payload["text"] = chunk["sparse_embedding"]

            points.append(
                PointStruct(
                    id=chunk["chunk_id"],
                    vector=vector_payload,
                    payload={
                        "doc_id": chunk["doc_id"],
                        "source": chunk["source"],
                        "title": chunk["title"],
                        "level": chunk["level"],
                        "chunk_type": chunk["chunk_type"],
                        "primary_type": chunk.get("primary_type", chunk["chunk_type"]),
                        "has_text": chunk.get("has_text", chunk["chunk_type"] == "text"),
                        "has_code": chunk.get("has_code", chunk["chunk_type"] == "code"),
                        "has_table": chunk.get("has_table", chunk["chunk_type"] == "table"),
                        "split_group_id": chunk.get("split_group_id"),
                        "content": chunk["content"],
                        "token_count": chunk["token_count"],
                        "parent_chunk_id": chunk["parent_chunk_id"],
                        "parent_titles": chunk["parent_titles"],
                        "prev_chunk_id": chunk["prev_chunk_id"],
                        "next_chunk_id": chunk["next_chunk_id"],
                        "sibling_chunk_ids": chunk["sibling_chunk_ids"],
                        "kb_ids": chunk["kb_ids"],
                        "summary": chunk.get("summary", ""),
                    },
                )
            )

        try:
            # batch upsert to stay under Qdrant's 32MB payload limit
            batch_size = 250
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(collection_name=collection_name, points=batch, timeout=90)
            return False, f"success — {len(points)} points in {(len(points) - 1) // batch_size + 1} batches"
        except Exception as e:
            return True, str(e)

# ──────────────────────────────────────────────
#  Provider-specific runtime settings
# ──────────────────────────────────────────────

if PROVIDER == "ollama":
    context_window = OLLAMA_CONTEXT_WINDOW
    max_chunk_tokens = OLLAMA_MAX_CHUNK_TOKENS
    min_chunk_tokens = OLLAMA_MIN_CHUNK_TOKENS
    vector_size = OLLAMA_EMBED_DIMENSIONS
else:
    context_window = None  # not needed for Azure
    max_chunk_tokens = AZURE_MAX_CHUNK_TOKENS
    min_chunk_tokens = AZURE_MIN_CHUNK_TOKENS
    vector_size = AZURE_EMBED_DIMENSIONS

# ``MarkdownSplitter`` is still required — ``FallbackSplitter`` wraps it
# for windows where the agent exhausts retries (Req 14.4).
splitter = MarkdownSplitter(
    provider=PROVIDER,
    max_chunk_tokens=max_chunk_tokens,
    min_chunk_tokens=min_chunk_tokens,
)
# ``MetadataGenerator`` is now used only on fallback chunks (Req 14.5).
metadata_gen = MetadataGenerator(provider=PROVIDER)
embedder = EmbeddingGenerator(provider=PROVIDER)
sparse_embedder = SparseEmbeddingGenerator()
qdrant = Qdrant()

print(f"Provider         : {PROVIDER}")
print(f"Min chunk tokens : {min_chunk_tokens or 'disabled (no merging)'}")
print(f"Max chunk tokens : {max_chunk_tokens or 'disabled (no overflow splitting)'}")
print(f"Vector size      : {vector_size}")

# ──────────────────────────────────────────────
#  Agentic chunking pipeline wiring (built once)
# ──────────────────────────────────────────────

# Pipeline-tuning knobs from agentic_chunking.config — single source of
# truth for paragraph_threshold_tokens / agent_context_budget_tokens etc.
_pipeline_config = build_config(PROVIDER)

# Tokenizer is process-shared and read-only (Req 11.6). Built via the
# factory so cl100k_base / HuggingFace caching matches markdown_splitter.
_tokenizer = build_tokenizer(PROVIDER)

# Concurrency cap on simultaneous LLM calls. Defaults to 32 — sized
# against the Azure deployment's 9.75K RPM / 9.75M TPM headroom. One
# semaphore is shared across every AgenticChunker call site so the
# cap applies process-wide. Override via env (AGENTIC_CHUNKER_CONCURRENCY)
# if you start seeing 429s.
_concurrency_limit = int(os.getenv("AGENTIC_CHUNKER_CONCURRENCY", "32"))
_rate_limit_semaphore = threading.Semaphore(_concurrency_limit)

_chunk_validator = ChunkValidator(
    tokenizer=_tokenizer,
    max_chunk_tokens=_pipeline_config.max_chunk_tokens,
    min_chunk_tokens=_pipeline_config.min_chunk_tokens,
)

# Chunker provider — decoupled from ``LLM_PROVIDER`` so chunking can
# stay on Azure (gpt-5-mini handles the forced-tool-call contract
# reliably) even when chat / query expansion move to a different
# provider like AWS Bedrock. Falls back to ``LLM_PROVIDER`` when the
# override is unset so single-provider deployments keep working.
_CHUNKER_PROVIDER = os.getenv("CHUNKER_PROVIDER", PROVIDER).strip().lower()

# Chunker STRATEGY — orthogonal to provider. Selects between the
# character-offset chunker (works well on gpt-5-mini) and the
# paragraph-ID chunker (works on weaker Bedrock models like DeepSeek,
# GLM, Mistral, gpt-oss). Defaults to char_offsets so single-flip
# ``CHUNKER_PROVIDER=aws`` doesn't silently change chunking behaviour.
_CHUNKER_STRATEGY = os.getenv("CHUNKER_STRATEGY", "char_offsets").strip().lower()
if _CHUNKER_STRATEGY not in ("char_offsets", "paragraph_ids"):
    print(
        f"Chunker strategy: unknown value {_CHUNKER_STRATEGY!r} — "
        f"falling back to 'char_offsets'"
    )
    _CHUNKER_STRATEGY = "char_offsets"

print(f"Chunker provider : {_CHUNKER_PROVIDER}")
print(f"Chunker strategy : {_CHUNKER_STRATEGY}")

# Pick the chunker class based on the strategy. Both classes expose
# the same ``__init__`` signature and the same ``chunk_window`` return
# shape so ``FileIngestor`` never has to care.
if _CHUNKER_STRATEGY == "paragraph_ids":
    from latest_agentic_chunking.paragraph_chunker import ParagraphChunker as _ChunkerClass
else:
    _ChunkerClass = AgenticChunker

_agentic_chunker = _ChunkerClass(
    provider=_CHUNKER_PROVIDER,
    model=None,  # falls back to env-resolved default deployment / model
    max_chunk_tokens=_pipeline_config.max_chunk_tokens,
    min_chunk_tokens=_pipeline_config.min_chunk_tokens,
    rate_limit_semaphore=_rate_limit_semaphore,
    validator=_chunk_validator,
)

_paragraph_pre_segmenter = ParagraphPreSegmenter(
    tokenizer=_tokenizer,
    agent_context_budget_tokens=_pipeline_config.agent_context_budget_tokens,
)

_fallback_splitter = FallbackSplitter(splitter=splitter)

_chunk_assembler = ChunkAssembler(tokenizer=_tokenizer)

# Per-file window-level worker count. Defaults to 16 (well below the
# Azure 9.75K RPM / 9.75M TPM ceiling for gpt-5-mini) and is
# overridable via env. Should stay <= _concurrency_limit so the
# semaphore is the global cap, not the bottleneck.
_max_window_workers = int(os.getenv("AGENTIC_WINDOW_WORKERS", "16"))

_file_ingestor = FileIngestor(
    chunker=_agentic_chunker,
    pre_segmenter=_paragraph_pre_segmenter,
    fallback=_fallback_splitter,
    assembler=_chunk_assembler,
    tokenizer=_tokenizer,
    paragraph_threshold_tokens=_pipeline_config.paragraph_threshold_tokens,
    agent_context_budget_tokens=_pipeline_config.agent_context_budget_tokens,
    max_window_workers=_max_window_workers,
)

orchestrator = IngestionOrchestrator(
    ingestor=_file_ingestor,
    embedder=embedder,
    sparse_embedder=sparse_embedder,
    qdrant=qdrant,
    vector_size=_pipeline_config.vector_size,
)


def start_embed(user_id: str, id: str, source: str, markdown: str):
    """Public ingest entry point — delegates to ``IngestionOrchestrator``.

    Signature is unchanged from the previous rule-based implementation
    so ``api.py`` callers do not need updating. The orchestrator returns
    ``(success: bool, message: str)`` directly, matching the contract.
    """
    return orchestrator.ingest_file(user_id, id, source, markdown)

# ──────────────────────────────────────────────
#  MAIN PIPELINE — local file harness
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parent_path = str(Path(__file__).parent)
    folder = os.path.join(parent_path, "files")

    from tqdm import tqdm
    print(folder)
    print(f"Provider         : {PROVIDER}")
    print(f"Vector size      : {_pipeline_config.vector_size}")

    for root, _dirs, files in os.walk(folder):
        for file in tqdm(files):
            source = os.path.join(root, file)
            print(f"\n── Processing: {source}")

            with open(source, "r", encoding="utf-8") as fh:
                markdown = fh.read()

            success, msg = orchestrator.ingest_file(
                user_id="arsh",
                doc_id=file,
                source=os.path.basename(source),
                markdown=markdown,
            )
            print(f"  Ingest: success={success} msg={msg}")
