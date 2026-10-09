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

# Chunking: classify each document, then route to adaptive / semantic /
# agentic (agentic = fallback). Agentic provider + strategy come from
# CHUNKER_PROVIDER / CHUNKER_STRATEGY in .env.
from chunker.orchestrator import ChunkingOrchestrator

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
# Each setting accepts both the names used in .env (AZURE_ENDPOINT, ...)
# and the older AZURE_OPENAI_* names.
def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default

AZURE_ENDPOINT = _env("AZURE_ENDPOINT", "AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = _env("AZURE_API_KEY", "AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = _env("AZURE_LLM_API_VERSION", "AZURE_OPENAI_API_VERSION", default="2025-03-01-preview")
AZURE_EMBEDDING_API_VERSION = _env("AZURE_EMBEDDING_API_VERSION", "AZURE_OPENAI_EMBEDDING_API_VERSION", default="2024-12-01-preview")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5-mini")
AZURE_FAST_DEPLOYMENT = os.getenv("AZURE_OPENAI_FAST_DEPLOYMENT", "gpt-5-mini")
AZURE_EMBED_DEPLOYMENT = _env("AZURE_EMBEDDING_DEPLOYMENT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", default="text-embedding-3-large")
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
                        # Fail the document instead of storing zero vectors:
                        # a zero vector can never match under COSINE search.
                        raise RuntimeError(
                            f"Azure embedding failed for batch {i}: {error_str[:200]}"
                        ) from e

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

# Chunk-size budgets for the rule-based chunkers still follow the provider.
if PROVIDER == "ollama":
    max_chunk_tokens = OLLAMA_MAX_CHUNK_TOKENS
    min_chunk_tokens = OLLAMA_MIN_CHUNK_TOKENS
else:
    max_chunk_tokens = AZURE_MAX_CHUNK_TOKENS
    min_chunk_tokens = AZURE_MIN_CHUNK_TOKENS

# Final chunk embeddings ALWAYS use Azure (AZURE_EMBEDDING_DEPLOYMENT,
# text-embedding-3-large) regardless of LLM_PROVIDER, so stored vectors
# live in the same space as the query embeddings used at retrieval time.
# (The local Ollama embedding model is only used inside the semantic
# chunker to find topic boundaries; those vectors are never stored.)
EMBEDDING_PROVIDER = "azure"
context_window = None  # Ollama-only option, unused for Azure
vector_size = AZURE_EMBED_DIMENSIONS

# Classifier + router over the adaptive / semantic / agentic chunkers.
# ``max/min_chunk_tokens`` size the rule-based chunkers (adaptive,
# semantic); the agentic chunker sizes itself from CHUNKER_PROVIDER.
chunking = ChunkingOrchestrator(
    provider=PROVIDER,
    max_chunk_tokens=max_chunk_tokens,
    min_chunk_tokens=min_chunk_tokens,
)
# Kept for backwards compatibility; not called by the ingest pipeline.
metadata_gen = MetadataGenerator(provider=PROVIDER)
embedder = EmbeddingGenerator(provider=EMBEDDING_PROVIDER)
sparse_embedder = SparseEmbeddingGenerator()
qdrant = Qdrant()

print(f"Provider         : {PROVIDER}")
print(f"Min chunk tokens : {min_chunk_tokens or 'disabled (no merging)'}")
print(f"Max chunk tokens : {max_chunk_tokens or 'disabled (no overflow splitting)'}")
print(f"Vector size      : {vector_size}")

print(f"Chunker provider : {os.getenv('CHUNKER_PROVIDER') or PROVIDER} (agentic only)")
print(f"Chunker strategy : {os.getenv('CHUNKER_STRATEGY') or 'char_offsets'} (agentic only)")

# ──────────────────────────────────────────────
#  Ingest pipeline
# ──────────────────────────────────────────────

def start_embed(user_id: str, id: str, source: str, markdown: str) -> tuple[bool, str]:
    """Chunk -> dense + sparse embed -> Qdrant upsert for one document.

    Chunking goes through ``ChunkingOrchestrator``: the document is
    classified and sent to the adaptive, semantic or agentic chunker
    (agentic only when classified for it, or as a fallback).

    Returns ``(success, message)``.
    """
    if not user_id:
        return False, "user_id required"
    if not id:
        return False, "doc_id required"

    result = chunking.chunk_text(markdown, source=source, doc_id=id)
    if not result.success:
        return False, f"Error while chunking: {result.error}"
    print(f"  {result.summary()}")

    chunks = result.chunks
    for chunk in chunks:
        # Real values are set later when a knowledge base is attached;
        # the key must exist so the Qdrant payload index stays consistent.
        chunk["kb_ids"] = None

    try:
        embedder.embed(chunks, context=context_window)
        sparse_embedder.embed(chunks)
    except Exception as e:
        return False, f"Error while embedding: {e}"

    try:
        error, msg = qdrant.upsert(user_id, chunks, vector_size=vector_size)
    except Exception as e:
        return False, f"Error while upserting: {e}"

    if error:
        return False, msg
    return True, f"{msg} (strategy={result.strategy_used}, type={result.document_type})"

# ──────────────────────────────────────────────
#  MAIN PIPELINE — local file harness
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parent_path = str(Path(__file__).parent)
    folder = os.path.join(parent_path, "files")

    from tqdm import tqdm
    print(folder)
    print(f"Provider         : {PROVIDER}")
    print(f"Vector size      : {vector_size}")

    for root, _dirs, files in os.walk(folder):
        for file in tqdm(files):
            source = os.path.join(root, file)
            print(f"\n── Processing: {source}")

            with open(source, "r", encoding="utf-8") as fh:
                markdown = fh.read()

            success, msg = start_embed(
                user_id="arsh",
                id=file,
                source=os.path.basename(source),
                markdown=markdown,
            )
            print(f"  Ingest: success={success} msg={msg}")
