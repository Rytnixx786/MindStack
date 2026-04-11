"""Core RAG retrieval and answer generation pipeline."""

from __future__ import annotations

import importlib
import json
import os
import pickle
import re
import sys
import time
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, cast

try:
    from .config import settings
    from .models import ChunkResult, QueryRequest, RAGResponse
    from .reranker import CrossEncoderReranker
except ImportError:  # pragma: no cover
    from config import settings
    from models import ChunkResult, QueryRequest, RAGResponse
    from reranker import CrossEncoderReranker


COLLECTION_NAME = "rag_documents"
MODEL_NAME = "gpt-4o-mini"
BM25_INDEX_PATH = Path(__file__).resolve().parent.parent / "bm25_index.pkl"
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag_system_prompt.yaml"

_VECTORSTORE: Optional[Any] = None
_BM25_PAYLOAD: Dict[str, Any] = {}
_RERANKER: CrossEncoderReranker = CrossEncoderReranker()
_PROMPT_CONFIG: Optional[Dict[str, Any]] = None

# In-memory query cache with FIFO eviction
_QUERY_CACHE_MAX_SIZE = 100
_QUERY_CACHE: OrderedDict[str, RAGResponse] = OrderedDict()
_QUERY_CACHE_LOCK = threading.Lock()

# Generation tuning knobs (override via env when needed).
_MAX_CHUNK_CHARS = int(os.getenv("RAG_MAX_CHUNK_CHARS", "900"))
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
_OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "180"))
_OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
_OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "15m")
_ENABLE_EXTRACTIVE_FAST_PATH = os.getenv("RAG_EXTRACTIVE_FAST_PATH", "true").lower() == "true"
_EXTRACTIVE_OVERLAP_THRESHOLD = float(os.getenv("RAG_EXTRACTIVE_OVERLAP_THRESHOLD", "0.18"))
_REFUSAL_CONFIDENCE_ENABLED = bool(getattr(settings, "refusal_confidence_enabled", True))
_REFUSAL_CONFIDENCE_THRESHOLD = float(getattr(settings, "refusal_confidence_threshold", 0.18))
_FORCE_FALLBACK_ON_MODEL_REFUSAL = bool(getattr(settings, "force_fallback_on_model_refusal", False))
_FORCE_FALLBACK_CONFIDENCE_THRESHOLD = float(
    getattr(settings, "force_fallback_confidence_threshold", 0.55)
)

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "should",
    "tell",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
}


def _get_chroma_persist_dir() -> str:
    return str(
        getattr(
            settings,
            "CHROMA_PERSIST_DIR",
            getattr(settings, "chroma_persist_dir", "./chroma_db"),
        )
    )


def _initialize_resources() -> None:
    """Load vector store and BM25 index once at module startup."""
    global _VECTORSTORE, _BM25_PAYLOAD

    print("Initializing retrieval resources...")

    try:
        embeddings_module = importlib.import_module("langchain_community.embeddings")
        vectorstores_module = importlib.import_module("langchain_community.vectorstores")
        huggingface_embeddings = getattr(embeddings_module, "HuggingFaceEmbeddings")
        chroma = getattr(vectorstores_module, "Chroma")

        embeddings = huggingface_embeddings(model_name="all-MiniLM-L6-v2")
        _VECTORSTORE = chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=_get_chroma_persist_dir(),
        )
        print("ChromaDB collection loaded.")
    except Exception as exc:  # pylint: disable=broad-except
        _VECTORSTORE = None
        print(f"Warning: failed to initialize ChromaDB collection: {exc}")

    try:
        if BM25_INDEX_PATH.exists():
            with BM25_INDEX_PATH.open("rb") as file_obj:
                payload = pickle.load(file_obj)
                _BM25_PAYLOAD = payload if isinstance(payload, dict) else {}
            print("BM25 index loaded from pickle.")
        else:
            _BM25_PAYLOAD = {}
            print("Warning: BM25 index file not found.")
    except Exception as exc:  # pylint: disable=broad-except
        _BM25_PAYLOAD = {}
        print(f"Warning: failed to load BM25 index pickle: {exc}")


def refresh_retrieval_resources() -> None:
    """Reload retrieval indexes after ingestion and clear stale cache."""
    _initialize_resources()
    _QUERY_CACHE.clear()


def _load_system_prompt() -> str:
    """Load system prompt text from YAML file."""
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")

    yaml_module = importlib.import_module("yaml")
    safe_load = getattr(yaml_module, "safe_load")
    with PROMPT_PATH.open("r", encoding="utf-8") as file_obj:
        data = safe_load(file_obj) or {}

    prompt = str(data.get("system_prompt", "")).strip()
    if not prompt:
        raise ValueError("system_prompt is missing in rag_system_prompt.yaml")
    return prompt


def _normalize_query(query: str) -> str:
    """Normalize query for cache key (lowercase, contractions, punctuation, spacing)."""
    normalized = query.lower().strip()

    # Normalize common contractions to improve cache hit rate on equivalent queries.
    replacements = {
        "what's": "what is",
        "whats": "what is",
        "can't": "cannot",
        "dont": "do not",
        "don't": "do not",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    # Remove punctuation and collapse whitespace so semantically same phrasing maps together.
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return " ".join(normalized.split())


def _trim_chunk_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> str:
    """Trim chunk text to reduce prompt size and speed up generation."""
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def _tokenize_meaningful_terms(text: str) -> List[str]:
    """Tokenize text while removing generic stop words."""
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOP_WORDS]


def _has_relevant_support(query: str, chunks: List[Dict[str, Any]]) -> bool:
    """Return True when the retrieved chunks are meaningfully related to the query."""
    query_terms = set(_tokenize_meaningful_terms(query))
    if not query_terms:
        return False

    for chunk in chunks:
        chunk_terms = set(_tokenize_meaningful_terms(str(chunk.get("text", ""))))
        if not chunk_terms:
            continue

        overlap = query_terms.intersection(chunk_terms)
        if not overlap:
            continue

        overlap_ratio = len(overlap) / max(len(query_terms), 1)
        reranker_score = float(chunk.get("reranker_score", chunk.get("score", 0.0)))
        if overlap_ratio >= 0.2 or reranker_score > 0.0:
            return True

    return False


def _evidence_confidence(query: str, chunks: List[Dict[str, Any]]) -> float:
    """Estimate evidence quality for answer-vs-refusal decisions.

    Returns a normalized score in [0, 1]. Higher is stronger evidence.
    """
    if not chunks:
        return 0.0

    query_terms = set(_tokenize_meaningful_terms(query))
    if not query_terms:
        return 0.0

    best_overlap = 0.0
    best_reranker = -20.0

    for chunk in chunks:
        chunk_terms = set(_tokenize_meaningful_terms(str(chunk.get("text", ""))))
        if chunk_terms:
            overlap = len(query_terms.intersection(chunk_terms)) / max(len(query_terms), 1)
            if overlap > best_overlap:
                best_overlap = overlap

        best_reranker = max(best_reranker, float(chunk.get("reranker_score", chunk.get("score", -20.0))))

    # Cross-encoder scores are unbounded and model-dependent. Normalize conservatively.
    reranker_component = max(0.0, min((best_reranker + 2.0) / 8.0, 1.0))
    overlap_component = max(0.0, min(best_overlap, 1.0))
    return (0.65 * overlap_component) + (0.35 * reranker_component)


def _extractive_fast_answer(query: str, chunks: List[Dict[str, Any]]) -> Optional[str]:
    """Return a grounded extractive answer quickly when overlap is high."""
    if not chunks:
        return None

    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not query_terms:
        return None

    best_chunk: Optional[Dict[str, Any]] = None
    best_overlap = 0.0
    for chunk in chunks:
        chunk_text = str(chunk.get("text", "")).lower()
        chunk_terms = set(re.findall(r"[a-z0-9]+", chunk_text))
        if not chunk_terms:
            continue
        overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk = chunk

    # Conservative threshold to avoid returning irrelevant text.
    if best_chunk is None or best_overlap < _EXTRACTIVE_OVERLAP_THRESHOLD:
        return None

    text = str(best_chunk.get("text", "")).strip()
    if not text:
        return None

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return text[:280]

    query_terms_lower = {t.lower() for t in query_terms}
    matched = [
        sentence
        for sentence in sentences
        if query_terms_lower & set(re.findall(r"[a-z0-9]+", sentence.lower()))
    ]
    selected = matched[:2] if matched else sentences[:2]
    return " ".join(selected).strip()


def _build_cache_key(request: QueryRequest) -> str:
    """Build a stable cache key including parameters that affect output."""
    normalized_query = _normalize_query(request.query)
    return f"q={normalized_query}|tkR={int(request.top_k_retrieval)}|tkX={int(request.top_k_rerank)}"


def _get_cached_response(cache_key: str) -> Optional[RAGResponse]:
    """Retrieve a cached response by key, returning a copy safe to mutate."""
    with _QUERY_CACHE_LOCK:
        if cache_key not in _QUERY_CACHE:
            return None
        _QUERY_CACHE.move_to_end(cache_key)
        cached = _QUERY_CACHE[cache_key]

    # Never return the shared instance: callers may tweak latency/timestamp flags.
    return cached.model_copy(deep=True)


def _cache_response(cache_key: str, response: RAGResponse) -> None:
    """Store response in cache with LRU eviction when capacity exceeded."""
    with _QUERY_CACHE_LOCK:
        if cache_key in _QUERY_CACHE:
            _QUERY_CACHE.move_to_end(cache_key)

        if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX_SIZE and cache_key not in _QUERY_CACHE:
            _QUERY_CACHE.popitem(last=False)

        _QUERY_CACHE[cache_key] = response


def load_prompt_config() -> Dict[str, Any]:
    """Load and cache full prompt config from YAML (requires restart to reload)."""
    global _PROMPT_CONFIG

    if _PROMPT_CONFIG is not None:
        assert isinstance(_PROMPT_CONFIG, dict)
        return _PROMPT_CONFIG

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")

    yaml_module = importlib.import_module("yaml")
    safe_load = getattr(yaml_module, "safe_load")
    with PROMPT_PATH.open("r", encoding="utf-8") as file_obj:
        config = safe_load(file_obj) or {}

    _PROMPT_CONFIG = config
    print(f"Prompt config loaded (version {config.get('version', 'unknown')})")
    return cast(Dict[str, Any], _PROMPT_CONFIG)


def retrieve_chunks(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks from ChromaDB vector search."""
    if not query.strip() or top_k <= 0:
        return []

    if _VECTORSTORE is None:
        print("Warning: vector store is not initialized. Returning no chunks.")
        return []

    try:
        results = _VECTORSTORE.similarity_search_with_score(query, k=top_k)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Vector retrieval failed: {exc}")
        return []

    chunks: List[Dict[str, Any]] = []
    for doc, score in results:
        metadata = doc.metadata or {}
        chunks.append(
            {
                "text": doc.page_content,
                "source": str(metadata.get("source", "unknown")),
                "chunk_index": int(metadata.get("chunk_index", 0)),
                "score": float(score),
                "retrieval_method": "vector",
            }
        )
    return chunks


def bm25_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Run BM25 keyword search over the prebuilt index."""
    if not query.strip() or top_k <= 0:
        return []

    bm25 = _BM25_PAYLOAD.get("bm25") if isinstance(_BM25_PAYLOAD, dict) else None
    documents = _BM25_PAYLOAD.get("documents", []) if isinstance(_BM25_PAYLOAD, dict) else []

    if bm25 is None or not documents:
        print("Warning: BM25 index not available. Returning no BM25 results.")
        return []

    tokens = [token for token in query.lower().split() if token]
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

    results: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(ranked[: max(top_k, 0)], start=1):
        doc = documents[idx]
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        results.append(
            {
                "text": doc.get("page_content", ""),
                "source": str(metadata.get("source", "unknown")),
                "chunk_index": int(metadata.get("chunk_index", 0)),
                "score": float(score),
                "retrieval_method": "bm25",
            }
        )

    return results


def hybrid_retrieval(query: str, top_k_each: int = 10) -> List[Dict[str, Any]]:
    """Combine vector and BM25 results via Reciprocal Rank Fusion (RRF)."""
    vector_results = retrieve_chunks(query, top_k_each)
    bm25_results = bm25_search(query, top_k_each)

    rrf_scores: Dict[str, Dict[str, Any]] = {}

    def accumulate(results: List[Dict[str, Any]], method_label: str) -> None:
        for rank, item in enumerate(results, start=1):
            key = item.get("text", "")
            if key not in rrf_scores:
                rrf_scores[key] = {
                    "text": item.get("text", ""),
                    "source": item.get("source", "unknown"),
                    "chunk_index": item.get("chunk_index", 0),
                    "rrf_score": 0.0,
                }
            rrf_scores[key]["rrf_score"] += 1.0 / (rank + 60)

    accumulate(vector_results, "vector")
    accumulate(bm25_results, "bm25")

    fused = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    return [
        {
            "text": item["text"],
            "source": item["source"],
            "chunk_index": int(item["chunk_index"]),
            "score": float(item["rrf_score"]),
            "retrieval_method": "hybrid",
        }
        for item in fused
    ]


def generate_answer(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a final answer from retrieved chunks using Ollama."""
    prompt_config = load_prompt_config()
    system_prompt = str(prompt_config.get("system_prompt", "")).strip()
    refusal_token = str(prompt_config.get("refusal_token", "INSUFFICIENT_CONTEXT")).strip()
    prompt_version = str(prompt_config.get("version", "unknown")).strip()

    if not system_prompt:
        raise ValueError("system_prompt is missing in prompt config")

    class RefusalChecker:
        """Controls when refusal is allowed based on retrieved evidence quality."""

        @staticmethod
        def chunk_score(chunk: Dict[str, Any]) -> float:
            if "reranker_score" in chunk:
                return float(chunk.get("reranker_score", 0.0))
            return float(chunk.get("score", 0.0))

        @staticmethod
        def should_refuse(citations: List[Dict[str, Any]]) -> bool:
            # Cross-encoder scores are often relative and may be negative even for useful chunks.
            # Refuse only when no evidence chunks are available.
            return not citations

        @staticmethod
        def model_refused(answer: str, token: str) -> bool:
            lower = answer.lower()
            return token.lower() in lower or "cannot find sufficient information" in lower

    if RefusalChecker.should_refuse(chunks):
        return {
            "answer": refusal_token,
            "citations": [],
            "chunks_retrieved": len(chunks),
            "model_used": "mistral (ollama)",
            "answer_grounded": False,
            "prompt_version": prompt_version,
        }

    if not _has_relevant_support(query, chunks):
        return {
            "answer": refusal_token,
            "citations": [],
            "chunks_retrieved": 0,
            "model_used": "relevance-gate",
            "answer_grounded": False,
            "prompt_version": prompt_version,
            "llm_called": False,
        }

    evidence_confidence = _evidence_confidence(query, chunks)
    if _REFUSAL_CONFIDENCE_ENABLED and evidence_confidence < _REFUSAL_CONFIDENCE_THRESHOLD:
        return {
            "answer": refusal_token,
            "citations": [],
            "chunks_retrieved": 0,
            "model_used": "confidence-gate",
            "answer_grounded": False,
            "prompt_version": prompt_version,
            "llm_called": False,
        }

    context_sections: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        context_sections.append(
            f"[Chunk {i}] Source: {chunk.get('source', 'unknown')}, Index: {chunk.get('chunk_index', 0)}\n"
            f"{_trim_chunk_text(str(chunk.get('text', '')))}"
        )

    context_text = "\n\n".join(context_sections) if context_sections else "No context chunks retrieved."
    user_message = (
        f"Question:\n{query}\n\n"
        f"Context Chunks:\n{context_text}\n\n"
        "Use only these chunks. Refuse only if evidence is clearly insufficient or unrelated."
    )

    answer_text = ""
    model_used = f"{_OLLAMA_MODEL} (ollama)"
    llm_called = True

    if _ENABLE_EXTRACTIVE_FAST_PATH:
        fast_answer = _extractive_fast_answer(query, chunks)
        if fast_answer:
            answer_text = fast_answer
            model_used = "extractive-fast-path"
            llm_called = False

    if not answer_text:
        try:
            ollama_module = importlib.import_module("ollama")

            ollama_host = os.getenv(
                "OLLAMA_BASE_URL",
                str(getattr(settings, "ollama_host", "http://localhost:11434")),
            )
            client = ollama_module.Client(host=ollama_host)
            response = client.chat(
                model=_OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                options={
                    "num_predict": _OLLAMA_NUM_PREDICT,
                    "num_ctx": _OLLAMA_NUM_CTX,
                    "temperature": 0.0,
                },
                keep_alive=_OLLAMA_KEEP_ALIVE,
                stream=False,
            )
            answer_text = str(response.get("message", {}).get("content", "")).strip()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Ollama generation failed: {exc}")
            answer_text = refusal_token

    model_refused = RefusalChecker.model_refused(answer_text, refusal_token)
    if model_refused and chunks and _FORCE_FALLBACK_ON_MODEL_REFUSAL and (
        evidence_confidence >= _FORCE_FALLBACK_CONFIDENCE_THRESHOLD
    ):
        # Force fallback only when confidence is high enough.
        best_chunk = max(chunks, key=RefusalChecker.chunk_score)
        answer_text = str(best_chunk.get("text", "")).strip() or refusal_token
        answer_grounded = answer_text != refusal_token
    elif model_refused:
        answer_text = refusal_token
        answer_grounded = False
    else:
        answer_grounded = True

    citations = [
        {
            "source": str(chunk.get("source", "unknown")),
            "chunk_index": int(chunk.get("chunk_index", 0)),
        }
        for chunk in chunks
    ]

    return {
        "answer": answer_text,
        "citations": citations,
        "chunks_retrieved": len(chunks),
        "model_used": model_used,
        "answer_grounded": answer_grounded,
        "prompt_version": prompt_version,
        "llm_called": llm_called,
    }


async def run_rag_query_stream(request: QueryRequest) -> AsyncGenerator[str, None]:
    """Run retrieval pipeline and stream NDJSON events for progressive rendering."""
    start = time.perf_counter()

    query = request.query
    top_k_retrieval = request.top_k_retrieval
    top_k_rerank = request.top_k_rerank
    cache_key = _build_cache_key(request)

    cached_response = _get_cached_response(cache_key)
    if cached_response is not None:
        latency_ms = (time.perf_counter() - start) * 1000
        cached_response.latency_ms = latency_ms
        cached_response.cached = True
        cached_response.llm_called = False

        cached_citations = [
            {
                "source": citation.source,
                "chunk_index": citation.chunk_index,
                "text": citation.text,
                "reranker_score": citation.reranker_score,
                "retrieval_method": citation.retrieval_method,
            }
            for citation in cached_response.citations
        ]
        citations_payload = {
            "event": "citations",
            "data": {
                "citations": cached_citations,
                "chunks_retrieved": int(cached_response.chunks_retrieved),
                "answer_grounded": bool(cached_response.answer_grounded),
                "reranker_applied": bool(cached_response.reranker_applied),
                "prompt_version": str(cached_response.prompt_version),
            },
        }
        yield json.dumps(citations_payload) + "\n"

        if cached_response.answer:
            yield json.dumps({"event": "token", "data": {"token": cached_response.answer}}) + "\n"

        done_payload = {
            "event": "done",
            "data": {
                "answer": cached_response.answer,
                "answer_grounded": bool(cached_response.answer_grounded),
                "latency_ms": latency_ms,
                "cached": True,
                "model_used": str(cached_response.model_used),
                "llm_called": False,
                "chunks_retrieved": int(cached_response.chunks_retrieved),
                "prompt_version": str(cached_response.prompt_version),
                "reranker_applied": bool(cached_response.reranker_applied),
                "citations": cached_citations,
            },
        }
        yield json.dumps(done_payload) + "\n"
        return

    chunks = hybrid_retrieval(query=query, top_k_each=top_k_retrieval)
    reranked_chunks = _RERANKER.rerank(query=query, chunks=chunks, top_k=top_k_rerank)

    prompt_config = load_prompt_config()
    prompt_version = str(prompt_config.get("version", "1.1")).strip() or "1.1"
    refusal_token = str(prompt_config.get("refusal_token", "INSUFFICIENT_CONTEXT"))

    if not reranked_chunks:
        latency_ms = (time.perf_counter() - start) * 1000
        response = RAGResponse(
            query=query,
            answer=refusal_token,
            answer_grounded=False,
            citations=[],
            chunks_retrieved=0,
            latency_ms=latency_ms,
            model_used="relevance-gate",
            prompt_version=prompt_version,
            reranker_applied=True,
            llm_called=False,
            cached=False,
        )
        _cache_response(cache_key, response)

        done_payload = {
            "event": "done",
            "data": {
                "answer": refusal_token,
                "answer_grounded": False,
                "latency_ms": latency_ms,
                "cached": False,
                "model_used": "relevance-gate",
                "llm_called": False,
                "chunks_retrieved": 0,
                "prompt_version": prompt_version,
                "reranker_applied": True,
                "citations": [],
            },
        }
        yield json.dumps(done_payload) + "\n"
        return

    citations = [
        {
            "source": str(chunk.get("source", "unknown")),
            "chunk_index": int(chunk.get("chunk_index", 0)),
            "text": str(chunk.get("text", "")),
            "reranker_score": float(chunk.get("reranker_score", 0.0)),
            "retrieval_method": str(chunk.get("retrieval_method", "hybrid")),
        }
        for chunk in reranked_chunks
    ]

    citations_payload = {
        "event": "citations",
        "data": {
            "citations": citations,
            "chunks_retrieved": len(reranked_chunks),
            "answer_grounded": True,
            "reranker_applied": True,
            "prompt_version": prompt_version,
        },
    }
    yield json.dumps(citations_payload) + "\n"

    citation_objects = [
        ChunkResult(
            text=str(chunk.get("text", "")),
            source=str(chunk.get("source", "unknown")),
            chunk_index=int(chunk.get("chunk_index", 0)),
            reranker_score=float(chunk.get("reranker_score", 0.0)),
            retrieval_method=str(chunk.get("retrieval_method", "hybrid")),
        )
        for chunk in reranked_chunks
    ]

    if _ENABLE_EXTRACTIVE_FAST_PATH:
        fast_answer = _extractive_fast_answer(query, reranked_chunks)
        if fast_answer:
            yield json.dumps({"event": "token", "data": {"token": fast_answer}}) + "\n"

            latency_ms = (time.perf_counter() - start) * 1000
            response = RAGResponse(
                query=query,
                answer=fast_answer,
                answer_grounded=True,
                citations=citation_objects,
                chunks_retrieved=len(reranked_chunks),
                latency_ms=latency_ms,
                model_used="extractive-fast-path",
                prompt_version=prompt_version,
                reranker_applied=True,
                llm_called=False,
                cached=False,
            )
            _cache_response(cache_key, response)

            done_payload = {
                "event": "done",
                "data": {
                    "answer": fast_answer,
                    "answer_grounded": True,
                    "latency_ms": latency_ms,
                    "cached": False,
                    "model_used": "extractive-fast-path",
                    "llm_called": False,
                    "chunks_retrieved": len(reranked_chunks),
                    "prompt_version": prompt_version,
                    "reranker_applied": True,
                    "citations": citations,
                },
            }
            yield json.dumps(done_payload) + "\n"
            return

    context_sections: List[str] = []
    for i, chunk in enumerate(reranked_chunks, start=1):
        context_sections.append(
            f"[Chunk {i}] Source: {chunk.get('source', 'unknown')}, Index: {chunk.get('chunk_index', 0)}\n"
            f"{_trim_chunk_text(str(chunk.get('text', '')))}"
        )

    context_text = "\n\n".join(context_sections) if context_sections else "No context chunks retrieved."
    user_message = (
        f"Question:\n{query}\n\n"
        f"Context Chunks:\n{context_text}\n\n"
        "Use only these chunks. Refuse only if evidence is clearly insufficient or unrelated."
    )

    full_answer_text = ""
    try:
        ollama_module = importlib.import_module("ollama")
        ollama_host = os.getenv(
            "OLLAMA_BASE_URL",
            str(getattr(settings, "ollama_host", "http://localhost:11434")),
        )
        client = ollama_module.Client(host=ollama_host)
        stream_iter = client.chat(
            model=_OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": str(prompt_config.get("system_prompt", "")).strip()},
                {"role": "user", "content": user_message},
            ],
            options={
                "num_predict": _OLLAMA_NUM_PREDICT,
                "num_ctx": _OLLAMA_NUM_CTX,
                "temperature": 0.0,
            },
            keep_alive=_OLLAMA_KEEP_ALIVE,
            stream=True,
        )

        for chunk in stream_iter:
            token = ""
            if isinstance(chunk, dict):
                token = str(chunk.get("message", {}).get("content", "") or chunk.get("response", ""))
            else:
                token = str(chunk)

            if not token:
                continue

            full_answer_text += token

            token_payload = {"event": "token", "data": {"token": token}}
            yield json.dumps(token_payload) + "\n"

        latency_ms = (time.perf_counter() - start) * 1000
        final_answer = full_answer_text.strip() or refusal_token
        final_grounded = final_answer != refusal_token
        done_payload = {
            "event": "done",
            "data": {
                "answer": final_answer,
                "answer_grounded": final_grounded,
                "latency_ms": latency_ms,
                "cached": False,
                "model_used": "mistral",
                "llm_called": True,
                "chunks_retrieved": len(reranked_chunks),
                "prompt_version": prompt_version,
                "reranker_applied": True,
                "citations": citations,
            },
        }

        response = RAGResponse(
            query=query,
            answer=final_answer,
            answer_grounded=final_grounded,
            citations=citation_objects if final_grounded else [],
            chunks_retrieved=len(reranked_chunks),
            latency_ms=latency_ms,
            model_used="mistral",
            prompt_version=prompt_version,
            reranker_applied=True,
            llm_called=True,
            cached=False,
        )
        _cache_response(cache_key, response)

        yield json.dumps(done_payload) + "\n"
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Ollama streaming failed: {exc}")
        latency_ms = (time.perf_counter() - start) * 1000

        response = RAGResponse(
            query=query,
            answer=refusal_token,
            answer_grounded=False,
            citations=[],
            chunks_retrieved=len(reranked_chunks),
            latency_ms=latency_ms,
            model_used="mistral",
            prompt_version=prompt_version,
            reranker_applied=True,
            llm_called=False,
            cached=False,
        )
        _cache_response(cache_key, response)

        done_payload = {
            "event": "done",
            "data": {
                "answer": refusal_token,
                "answer_grounded": False,
                "latency_ms": latency_ms,
                "cached": False,
                "model_used": "mistral",
                "llm_called": False,
                "chunks_retrieved": len(reranked_chunks),
                "prompt_version": prompt_version,
                "reranker_applied": True,
                "citations": citations,
            },
        }
        yield json.dumps(done_payload) + "\n"


def run_rag_query(request: QueryRequest) -> RAGResponse:
    """Run retrieval (hybrid → rerank) and generation while measuring end-to-end latency.
    
    Includes caching: identical queries (normalized) return cached responses instantly.
    """
    start = time.perf_counter()

    query = request.query
    cache_key = _build_cache_key(request)

    cached_response = _get_cached_response(cache_key)
    if cached_response is not None:
        cache_latency_ms = (time.perf_counter() - start) * 1000
        cached_response.latency_ms = cache_latency_ms
        cached_response.cached = True
        cached_response.llm_called = False
        return cached_response

    top_k_retrieval = request.top_k_retrieval
    top_k_rerank = request.top_k_rerank

    chunks = hybrid_retrieval(query=query, top_k_each=top_k_retrieval)
    
    reranked_chunks = _RERANKER.rerank(query=query, chunks=chunks, top_k=top_k_rerank)
    
    prompt_config = load_prompt_config()
    prompt_version = str(prompt_config.get("version", "unknown")).strip()

    if not reranked_chunks:
        latency_ms = (time.perf_counter() - start) * 1000
        response = RAGResponse(
            query=query,
            answer=str(prompt_config.get("refusal_token", "INSUFFICIENT_CONTEXT")),
            answer_grounded=False,
            citations=[],
            chunks_retrieved=0,
            latency_ms=latency_ms,
            model_used=f"{_OLLAMA_MODEL} (ollama)",
            prompt_version=prompt_version,
            reranker_applied=True,
            llm_called=False,
            cached=False,
        )
        _cache_response(cache_key, response)
        return response

    result_dict = generate_answer(query=query, chunks=reranked_chunks)

    latency_ms = (time.perf_counter() - start) * 1000
    
    citation_objects = []
    if result_dict.get("answer_grounded", False):
        citation_objects = [
            ChunkResult(
                text=chunk.get("text", ""),
                source=chunk.get("source", "unknown"),
                chunk_index=chunk.get("chunk_index", 0),
                reranker_score=chunk.get("reranker_score", 0.0),
                retrieval_method=chunk.get("retrieval_method", "hybrid"),
            )
            for chunk in reranked_chunks
        ]

    response = RAGResponse(
        query=query,
        answer=result_dict.get("answer", ""),
        answer_grounded=result_dict.get("answer_grounded", False),
        citations=citation_objects,
        chunks_retrieved=result_dict.get("chunks_retrieved", 0),
        latency_ms=latency_ms,
        model_used=str(result_dict.get("model_used", f"{_OLLAMA_MODEL} (ollama)")),
        prompt_version=result_dict.get("prompt_version", ""),
        reranker_applied=True,
        llm_called=bool(result_dict.get("llm_called", True)),
        cached=False,
    )
    
    # Store in cache for future identical queries
    _cache_response(cache_key, response)
    return response


class Retriever:
    """Compatibility adapter for API module; delegates to vector retrieval."""

    def __init__(self, chunks: List[Dict[str, object]]):  # noqa: ARG002
        self._unused_chunks = chunks

    def search(self, query: str, top_k: int) -> List[Dict[str, object]]:
        """Return top-k chunk records in legacy format expected by API layer."""
        results = retrieve_chunks(query, top_k)
        legacy: List[Dict[str, object]] = []
        for item in results:
            legacy.append(
                {
                    "source": item["source"],
                    "chunk_id": item["chunk_index"],
                    "content": item["text"],
                    "score": item["score"],
                }
            )
        return legacy


def _main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.retrieval "What is the refund policy?"')

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        raise SystemExit("Query must not be empty.")

    request = QueryRequest(query=query)
    result = run_rag_query(request)
    print(result.model_dump_json(indent=2))


_initialize_resources()


if __name__ == "__main__":
    _main()
