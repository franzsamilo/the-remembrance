"""API endpoint tests."""
import json

import pytest


def test_health(client):
    """Health check returns online status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert "version" in data


def test_stats_returns_json(client):
    """Stats endpoint returns JSON (may fail if Neo4j unavailable)."""
    resp = client.get("/stats")
    # 200 if Neo4j ok, 500 if not
    assert resp.headers["content-type"].startswith("application/json")


def test_delete_documents_path_traversal_blocked(client):
    """Delete documents endpoint blocks path traversal in filename."""
    resp = client.delete("/documents/../../../etc/passwd")
    assert resp.status_code == 403


def test_chat_stream_returns_sse(client):
    """Chat stream endpoint returns SSE with chunk/done/error events."""
    resp = client.post(
        "/chat/stream",
        json={"query": "What are the key materials?", "explain": False},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    # Consume stream and check we get valid SSE
    lines = resp.text.strip().split("\n\n")
    assert len(lines) >= 1
    for line in lines:
        if line.startswith("data: "):
            data = json.loads(line[6:])
            assert "type" in data
            assert data["type"] in ("chunk", "done", "error", "grounding_error")
