"""FastAPI service for retrieval and reranking."""

import json
import os
from pathlib import Path
from typing import Any, List

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

try:
    from .config import settings
    from .db import get_metrics_summary, get_metrics_trend, init_db, insert_query_result
    from .ingestion import ingest_documents, load_and_chunk_documents
    from .models import ChunkResult, QueryRequest, RAGResponse
    from .retrieval import refresh_retrieval_resources
    from .reranker import Reranker
    from .retrieval import Retriever, run_rag_query, run_rag_query_stream
except ImportError:  # pragma: no cover
    from config import settings
    from db import get_metrics_summary, get_metrics_trend, init_db, insert_query_result
    from ingestion import ingest_documents, load_and_chunk_documents
    from models import ChunkResult, QueryRequest, RAGResponse
    from retrieval import refresh_retrieval_resources
    from reranker import Reranker
    from retrieval import Retriever, run_rag_query, run_rag_query_stream


app = FastAPI(title="RAG System API", version="1.0.0")


def _require_admin(
    x_admin_api_key: str | None = Header(default=None, alias="X-Admin-Api-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    """Protect mutating admin endpoints (/upload, /ingest) for public deployments."""
    if bool(getattr(settings, "enable_unauth_admin", False)):
        return

    expected = str(getattr(settings, "admin_api_key", "")).strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Admin endpoints are disabled. Set ADMIN_API_KEY to enable /upload and /ingest.",
        )

    token = (x_admin_api_key or "").strip()
    if not token and authorization:
        lower = authorization.lower()
        if lower.startswith("bearer "):
            token = authorization[7:].strip()

    if token != expected:
        raise HTTPException(status_code=403, detail="Forbidden")


def _get_allowed_origins() -> list[str]:
    """Load allowed CORS origins from ALLOWED_ORIGINS or use safe defaults."""
    raw_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001,http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
    )
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_chunks = load_and_chunk_documents("data")
_retriever = Retriever(_chunks)
_reranker = Reranker()
init_db()

LOGS_DIR = Path("logs")
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md", ".doc", ".docx"}


def _ensure_logs_dir() -> None:
    """Create logs directory if it does not exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _append_query_log(response: RAGResponse) -> None:
    """Append query result to logs/query_log.jsonl."""
    try:
        _ensure_logs_dir()
        log_entry = {
            "timestamp": response.timestamp.isoformat(),
            "query": response.query,
            "answer_grounded": response.answer_grounded,
            "latency_ms": response.latency_ms,
            "prompt_version": response.prompt_version,
            "chunks_retrieved": response.chunks_retrieved,
        }
        with (LOGS_DIR / "query_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: failed to write query log: {exc}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
async def trigger_ingestion(_auth: None = Depends(_require_admin)):
    try:
        summary = ingest_documents(str(DATA_DIR))
        refresh_retrieval_resources()
        return {"status": "success", "summary": summary}
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    _auth: None = Depends(_require_admin),
) -> dict[str, Any]:
    """Upload source files, ingest them, and refresh retrieval resources."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    for upload in files:
        filename = Path(upload.filename or "").name
        if not filename:
            raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type: {suffix or 'none'}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))}"
                ),
            )

        destination = UPLOAD_DIR / filename
        content = await upload.read()
        destination.write_bytes(content)
        saved_files.append(filename)

    try:
        summary = ingest_documents(str(DATA_DIR))
        refresh_retrieval_resources()
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Upload succeeded but ingestion failed: {exc}") from exc

    return {
        "status": "success",
        "files_uploaded": saved_files,
        "summary": summary,
    }


@app.post("/query", response_model=RAGResponse)
def query(payload: QueryRequest) -> RAGResponse:
    """Execute RAG query with hybrid retrieval and reranking."""
    response = run_rag_query(payload)
    
    # Keep JSONL logging for backward compatibility.
    _append_query_log(response)
    try:
        insert_query_result(response)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: failed to persist query metrics to SQLite: {exc}")
    
    return response


@app.post("/query/stream")
async def query_stream(payload: QueryRequest) -> StreamingResponse:
    """Stream query responses as newline-delimited JSON events."""

    async def event_generator():
        token_parts: list[str] = []
        citations_data: list[dict[str, Any]] = []
        chunks_retrieved = 0
        prompt_version = "1.1"
        reranker_applied = True
        done_data: dict[str, Any] = {}

        async for line in run_rag_query_stream(payload):
            try:
                event = json.loads(line)
                event_type = event.get("event")
                data = event.get("data", {}) if isinstance(event, dict) else {}

                if event_type == "citations" and isinstance(data, dict):
                    citations_data = data.get("citations", []) or []
                    chunks_retrieved = int(data.get("chunks_retrieved", 0) or 0)
                    prompt_version = str(data.get("prompt_version", "1.1") or "1.1")
                    reranker_applied = bool(data.get("reranker_applied", True))
                elif event_type == "token" and isinstance(data, dict):
                    token_parts.append(str(data.get("token", "")))
                elif event_type == "done" and isinstance(data, dict):
                    done_data = data
            except Exception:  # pylint: disable=broad-except
                pass

            yield line

        answer_text = "".join(token_parts).strip()
        if not answer_text and "answer" in done_data:
            answer_text = str(done_data.get("answer", ""))

        answer_grounded = bool(done_data.get("answer_grounded", bool(citations_data and answer_text != "INSUFFICIENT_CONTEXT")))
        llm_called = bool(done_data.get("llm_called", bool(token_parts)))
        latency_ms = float(done_data.get("latency_ms", 0.0) or 0.0)
        model_used = str(done_data.get("model_used", "mistral"))
        cached = bool(done_data.get("cached", False))

        citation_objects = [
            ChunkResult(
                text=str(item.get("text", "")),
                source=str(item.get("source", "unknown")),
                chunk_index=int(item.get("chunk_index", 0)),
                reranker_score=float(item.get("reranker_score", 0.0)),
                retrieval_method=str(item.get("retrieval_method", "hybrid")),
            )
            for item in citations_data
            if isinstance(item, dict)
        ]

        full_response = RAGResponse(
            query=payload.query,
            answer=answer_text,
            answer_grounded=answer_grounded,
            citations=citation_objects,
            chunks_retrieved=chunks_retrieved,
            latency_ms=latency_ms,
            model_used=model_used,
            prompt_version=prompt_version,
            reranker_applied=reranker_applied,
            llm_called=llm_called,
            cached=cached,
        )

        _append_query_log(full_response)
        try:
            insert_query_result(full_response)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: failed to persist stream query metrics to SQLite: {exc}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/metrics")
def metrics() -> dict:
    """Return query metrics from SQLite storage."""
    try:
        return get_metrics_summary()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: failed to read metrics from SQLite: {exc}")
        return {
            "total_queries": 0,
            "grounded_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "queries_last_24h": 0,
            "grounded_rate_7d": 0.0,
            "latency_breakdown": {
                "cached": {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0},
                "extractive_fast_path": {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0},
                "full_generation": {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0},
            },
            "latency_note": "No full generation queries logged yet.",
        }


@app.get("/metrics/trend")
def metrics_trend() -> List[dict[str, Any]]:
    """Return daily metrics trend for charting (last 30 days)."""
    try:
        return get_metrics_trend(days=30)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: failed to read metrics trend from SQLite: {exc}")
        return []
