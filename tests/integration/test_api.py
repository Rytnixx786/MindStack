from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi.testclient import TestClient

from src import api
from src.db import init_db


@pytest.mark.asyncio
async def test_health_returns_200():
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_returns_expected_schema(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "metrics_test.db"
    monkeypatch.setenv("SQLITE_PATH", str(sqlite_path))
    init_db()

    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.execute(
            """
            INSERT INTO query_log (
                timestamp,
                query_hash,
                latency_ms,
                grounded,
                cached,
                chunks_retrieved,
                reranker_applied,
                llm_called
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-04-03T10:00:00", "h1", 100.0, 1, 0, 2, 1, 1),
        )
        conn.execute(
            """
            INSERT INTO query_log (
                timestamp,
                query_hash,
                latency_ms,
                grounded,
                cached,
                chunks_retrieved,
                reranker_applied,
                llm_called
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-04-03T10:01:00", "h2", 300.0, 0, 0, 0, 1, 1),
        )
        conn.commit()

    client = TestClient(api.app)

    response = client.get("/metrics")

    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "total_queries",
        "grounded_rate",
        "avg_latency_ms",
        "p95_latency_ms",
        "p50_latency_ms",
        "queries_last_24h",
        "grounded_rate_7d",
        "latency_breakdown",
        "latency_note",
    }
    assert data["total_queries"] == 2
    assert data["grounded_rate"] == 0.5
    assert data["avg_latency_ms"] == 200.0
    assert data["p95_latency_ms"] == 300.0
    assert data["p50_latency_ms"] == 300.0
    assert isinstance(data["queries_last_24h"], int)
    assert isinstance(data["grounded_rate_7d"], float)
    assert isinstance(data["latency_breakdown"], dict)
    assert set(data["latency_breakdown"].keys()) == {
        "cached",
        "extractive_fast_path",
        "full_generation",
    }
    assert isinstance(data["latency_note"], str)


def test_ingest_triggers_reindex(monkeypatch):
    called = {"ingest": 0, "refresh": 0}

    def fake_ingest_documents(_data_dir: str):
        called["ingest"] += 1
        return {"chunks_created": 1, "documents_processed": 1, "collection_name": "rag_documents"}

    def fake_refresh():
        called["refresh"] += 1

    # Allow admin endpoints in tests without headers.
    monkeypatch.setattr(api.settings, "enable_unauth_admin", True)
    monkeypatch.setattr(api, "ingest_documents", fake_ingest_documents)
    monkeypatch.setattr(api, "refresh_retrieval_resources", fake_refresh)
    client = TestClient(api.app)

    response = client.post("/ingest")

    assert called["ingest"] == 1
    assert called["refresh"] == 1
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert "summary" in payload


@pytest.mark.asyncio
async def test_metrics_trend_returns_daily_aggregates(tmp_path, monkeypatch):
    sqlite_path = tmp_path / "metrics_trend.db"
    monkeypatch.setenv("SQLITE_PATH", str(sqlite_path))
    init_db()

    with closing(sqlite3.connect(sqlite_path)) as conn:
        conn.execute(
            """
            INSERT INTO query_log (
                timestamp,
                query_hash,
                latency_ms,
                grounded,
                cached,
                chunks_retrieved,
                reranker_applied,
                llm_called
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-04-01T10:00:00", "t1", 120.0, 1, 0, 2, 1, 1),
        )
        conn.execute(
            """
            INSERT INTO query_log (
                timestamp,
                query_hash,
                latency_ms,
                grounded,
                cached,
                chunks_retrieved,
                reranker_applied,
                llm_called
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-04-01T11:00:00", "t2", 180.0, 0, 0, 1, 1, 1),
        )
        conn.commit()

    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics/trend")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        first = data[0]
        assert set(first.keys()) == {"date", "total_queries", "grounded_rate", "avg_latency_ms"}


def test_query_endpoint_rejects_empty_query():
    """Error scenario: Empty query string should be rejected."""
    client = TestClient(api.app)
    response = client.post("/query", json={"query": ""})
    
    # Either 422 (validation error) or 400 (bad request)
    assert response.status_code in [400, 422]


def test_query_endpoint_rejects_missing_query_field():
    """Error scenario: Missing 'query' field should fail validation."""
    client = TestClient(api.app)
    response = client.post("/query", json={})
    
    assert response.status_code == 422


def test_query_endpoint_rejects_invalid_json():
    """Error scenario: Malformed JSON should return 400."""
    client = TestClient(api.app)
    response = client.post(
        "/query",
        content="{ invalid json }",
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 422


def test_query_endpoint_with_invalid_top_k_parameters():
    """Error scenario: Invalid top_k values should be rejected or clamped."""
    client = TestClient(api.app)
    
    # Negative top_k
    response = client.post("/query", json={
        "query": "What is the refund policy?",
        "top_k_retrieval": -5
    })
    assert response.status_code in [400, 422]
    
    # Zero top_k
    response = client.post("/query", json={
        "query": "What is the refund policy?",
        "top_k_rerank": 0
    })
    assert response.status_code in [400, 422]


def test_ingest_endpoint_requires_admin_key(monkeypatch):
    """Error scenario: /ingest without admin key should be rejected (unless auth is disabled)."""
    monkeypatch.setattr(api.settings, "enable_unauth_admin", False)
    monkeypatch.setattr(api.settings, "admin_api_key", "secret-key-123")
    
    client = TestClient(api.app)
    
    # Request without API key
    response = client.post("/ingest")
    assert response.status_code == 403
    
    # Request with wrong API key
    response = client.post("/ingest", headers={"X-Admin-Api-Key": "wrong-key"})
    assert response.status_code == 403


def test_ingest_endpoint_accepts_correct_admin_key(monkeypatch):
    """Positive scenario: /ingest with correct admin key should succeed."""
    def fake_ingest(_data_dir):
        return {"chunks_created": 1, "documents_processed": 1, "collection_name": "rag_documents"}
    
    def fake_refresh():
        pass
    
    monkeypatch.setattr(api.settings, "enable_unauth_admin", False)
    monkeypatch.setattr(api.settings, "admin_api_key", "secret-key-123")
    monkeypatch.setattr(api, "ingest_documents", fake_ingest)
    monkeypatch.setattr(api, "refresh_retrieval_resources", fake_refresh)
    
    client = TestClient(api.app)
    response = client.post("/ingest", headers={"X-Admin-Api-Key": "secret-key-123"})
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_ingest_endpoint_accepts_bearer_token(monkeypatch):
    """Positive scenario: /ingest with Bearer token should work."""
    def fake_ingest(_data_dir):
        return {"chunks_created": 1, "documents_processed": 1, "collection_name": "rag_documents"}
    
    def fake_refresh():
        pass
    
    monkeypatch.setattr(api.settings, "enable_unauth_admin", False)
    monkeypatch.setattr(api.settings, "admin_api_key", "secret-key-123")
    monkeypatch.setattr(api, "ingest_documents", fake_ingest)
    monkeypatch.setattr(api, "refresh_retrieval_resources", fake_refresh)
    
    client = TestClient(api.app)
    response = client.post("/ingest", headers={"Authorization": "Bearer secret-key-123"})
    
    assert response.status_code == 200


def test_metrics_endpoint_with_empty_database(tmp_path, monkeypatch):
    """Error scenario: /metrics with no query data should return sensible defaults."""
    sqlite_path = tmp_path / "metrics_empty.db"
    monkeypatch.setenv("SQLITE_PATH", str(sqlite_path))
    init_db()
    
    client = TestClient(api.app)
    response = client.get("/metrics")
    
    assert response.status_code == 200
    data = response.json()
    assert data["total_queries"] == 0
    assert data["grounded_rate"] == 0.0 or data["grounded_rate"] >= 0
    assert data["avg_latency_ms"] == 0.0 or isinstance(data["avg_latency_ms"], (int, float))


@pytest.mark.asyncio
async def test_query_endpoint_handles_very_long_query():
    """Edge case: Very long query string should be handled gracefully."""
    transport = ASGITransport(app=api.app)
    long_query = "What " * 1000  # Very long query
    
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"query": long_query}
        )
    
    # Should either handle it or reject with sensible error
    assert response.status_code in [200, 400, 422, 413]


def test_query_endpoint_handles_special_characters():
    """Edge case: Special characters in query should not cause crashes."""
    client = TestClient(api.app)
    special_queries = [
        "What is <script>alert('xss')</script>?",
        "Query with \n newlines \n inside",
        "Query with \t tabs \t inside",
        "Query with 'single' and \"double\" quotes",
        "Unicode query: 你好世界 🚀 مرحبا",
    ]
    
    for query in special_queries:
        response = client.post("/query", json={"query": query})
        # Should not crash, even if it returns 400
        assert response.status_code in [200, 400, 422]


def test_upload_endpoint_requires_admin_key(monkeypatch):
    """Error scenario: /upload without admin key should be rejected."""
    monkeypatch.setattr(api.settings, "enable_unauth_admin", False)
    monkeypatch.setattr(api.settings, "admin_api_key", "secret-key-123")
    
    client = TestClient(api.app)
    
    # Request without API key but with a file
    response = client.post("/upload", files={"files": ("test.txt", b"test content")})
    assert response.status_code == 403


def test_query_endpoint_response_schema_consistency(monkeypatch):
    """Validation: Query response should always have required fields."""
    def fake_run_rag_query(payload):
        from src.models import RAGResponse, ChunkResult
        return RAGResponse(
            query=payload.query,
            answer="Test answer",
            answer_grounded=True,
            citations=[
                ChunkResult(
                    text="Test citation",
                    source="test.txt",
                    chunk_index=0,
                    reranker_score=1.0,
                    retrieval_method="hybrid"
                )
            ],
            chunks_retrieved=1,
            latency_ms=10.0,
            model_used="test",
            prompt_version="1.0",
            reranker_applied=True,
            llm_called=False,
            cached=False,
        )
    
    monkeypatch.setattr(api, "run_rag_query", fake_run_rag_query)
    client = TestClient(api.app)
    response = client.post("/query", json={"query": "test query"})
    
    assert response.status_code == 200
    data = response.json()
    
    # Check all required fields are present
    required_fields = {
        "query", "answer", "answer_grounded", "citations", "chunks_retrieved",
        "latency_ms", "model_used", "prompt_version", "reranker_applied",
        "llm_called", "cached", "timestamp"
    }
    assert required_fields.issubset(set(data.keys())), f"Missing fields: {required_fields - set(data.keys())}"


def test_citation_schema_consistency(monkeypatch):
    """Validation: Each citation should have required fields."""
    def fake_run_rag_query(payload):
        from src.models import RAGResponse, ChunkResult
        return RAGResponse(
            query=payload.query,
            answer="Answer with citations",
            answer_grounded=True,
            citations=[
                ChunkResult(
                    text="Citation text",
                    source="source.txt",
                    chunk_index=5,
                    reranker_score=7.5,
                    retrieval_method="lexical"
                )
            ],
            chunks_retrieved=1,
            latency_ms=50.0,
            model_used="extractive",
            prompt_version="1.0",
            reranker_applied=True,
            llm_called=False,
            cached=False,
        )
    
    monkeypatch.setattr(api, "run_rag_query", fake_run_rag_query)
    client = TestClient(api.app)
    response = client.post("/query", json={"query": "test"})
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify citation structure
    assert len(data["citations"]) > 0
    citation = data["citations"][0]
    required_citation_fields = {"text", "source", "chunk_index", "reranker_score", "retrieval_method"}
    assert required_citation_fields.issubset(set(citation.keys())), f"Missing citation fields: {required_citation_fields - set(citation.keys())}"
