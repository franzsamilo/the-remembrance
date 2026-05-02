"""Pytest fixtures for The Remembrance backend tests."""
import os
import sys

# Set minimal env BEFORE any src imports (config validates at load)
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

import pytest

# Ensure backend src is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _isolate_compgcn_checkpoints(tmp_path, monkeypatch):
    """Redirect CompGCN checkpoint paths to a per-test tmp dir.

    Without this, tests that exercise run_audit() with monkey-patched mock
    data overwrite the real backend/run_logs/compgcn_best.pt with synthetic
    weights — a regression that has already polluted production checkpoints
    twice (Run 7's guardrail test, Run 8's adv_temp tests).

    Autouse so every test gets isolated paths regardless of whether it
    explicitly opts in.
    """
    try:
        import src.gnn_module as gnn_module
    except ImportError:
        return  # gnn_module not on path for some tests; nothing to isolate
    monkeypatch.setattr(gnn_module, "CHECKPOINT_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(gnn_module, "CHECKPOINT_PATH", str(tmp_path / "compgcn_best.pt"), raising=False)
    monkeypatch.setattr(gnn_module, "CHECKPOINT_META_PATH", str(tmp_path / "compgcn_best_meta.json"), raising=False)


@pytest.fixture
def client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)
