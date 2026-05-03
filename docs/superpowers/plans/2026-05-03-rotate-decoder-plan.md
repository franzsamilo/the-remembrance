# RotatE Decoder Implementation Plan (Run 9)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-coded DistMult decoder in `CompGCNAuditModel` with a runtime-dispatched decoder that supports both `distmult` (Runs 1–8 default, byte-identical) and `rotate` (Run 9, Sun et al. 2019). Encoder, training loop, loss, sampling, and Neo4j sync paths are unchanged. Decoder choice persisted in checkpoint meta JSON and `AuditRun` Neo4j node for ablation reporting.

**Architecture:** A new `COMPGCN_DECODER` env var (default `distmult`) gates the decoder choice. The model class gains a `decoder` parameter; when `decoder == "rotate"` it instantiates a separate `rel_phase` embedding for phase angles in `[-π, π]^128` (split halves of the 256-dim real encoder output). The `edge_logits` method dispatches on `self.decoder` — DistMult is unchanged; RotatE computes `-||h ∘ r - t||₂` in real arithmetic. The encoder's `rel_emb` (used for CompGCN message passing) is independent of the decoder's `rel_phase` (used only for scoring) — per Vashishth+ 2020 Table 4 encoder-decoder separation.

**Tech Stack:** Python 3.9 / PyTorch / PyTorch Geometric / Neo4j / pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-03-rotate-decoder-design.md`

---

## File Inventory

| File | Action | Lines (approx) |
|------|--------|----------------|
| `backend/src/config.py` | Modify (add 1 config flag) | +6 |
| `backend/src/gnn_module.py` | Modify (model __init__, edge_logits dispatch, _rotate_logits helper, run_audit model construction, checkpoint meta, AuditRun Cypher in run_audit and recover_from_checkpoint) | +50 |
| `backend/run_logs/rotate_audit.py` | Create (Run 9 launcher) | ~55 |
| `backend/tests/test_gnn_rotate.py` | Create (8 unit/wiring tests) | ~250 |
| `backend/TUNING_LOG.md` | Append Run 9 section after the run | ~100 |

---

## Task 1: Add `COMPGCN_DECODER` Config Flag

**Files:**
- Modify: `backend/src/config.py` (after `COMPGCN_ADV_TEMP` line)
- Create: `backend/tests/test_gnn_rotate.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_gnn_rotate.py` with this initial content:

```python
"""RotatE decoder tests."""
from __future__ import annotations

import math
import os
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

import torch
import torch.nn.functional as F


def test_config_exposes_decoder_with_distmult_default():
    """COMPGCN_DECODER must default to 'distmult' so Runs 1-8 reproduce.

    Verified in a subprocess so reloading the config module does not
    replace the in-process Config class — that would break monkeypatching
    for downstream tests in this file (lesson learned in Run 8).
    """
    from src.config import Config
    assert hasattr(Config, "COMPGCN_DECODER"), \
        "Config must expose COMPGCN_DECODER"

    import subprocess
    import sys
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if k != "COMPGCN_DECODER"}
    env.setdefault("NEO4J_URI", "bolt://localhost:7687")
    env.setdefault("NEO4J_PASSWORD", "test")
    env.setdefault("GOOGLE_API_KEY", "test-key")
    result = subprocess.run(
        [sys.executable, "-c",
         "from src.config import Config; print(repr(Config.COMPGCN_DECODER))"],
        env=env, capture_output=True, text=True, cwd=backend_dir,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    assert result.stdout.strip() == "'distmult'", \
        f"Default Config.COMPGCN_DECODER must be 'distmult', got: {result.stdout!r}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_gnn_rotate.py::test_config_exposes_decoder_with_distmult_default -v`
Expected: FAIL with `AssertionError: Config must expose COMPGCN_DECODER`.

- [ ] **Step 3: Add the config flag**

In `backend/src/config.py`, immediately after the `COMPGCN_ADV_TEMP` line, insert:

```python
    # Decoder choice for CompGCN. "distmult" (Runs 1-8 default) computes
    # s(h,r,t) = sum(h * r * t). "rotate" computes -||h o r - t||_2 with
    # relations parameterized as phase angles in [-pi, pi] over 128 complex
    # dimensions (split halves of the 256-dim real encoder output).
    # Reference: Sun et al. 2019, "RotatE: Knowledge Graph Embedding by
    # Relational Rotation in Complex Space".
    COMPGCN_DECODER = os.getenv("COMPGCN_DECODER", "distmult").lower()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_gnn_rotate.py::test_config_exposes_decoder_with_distmult_default -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/config.py backend/tests/test_gnn_rotate.py
git commit -m "feat(gnn): add COMPGCN_DECODER config flag (default distmult)"
```

---

## Task 2: RotatE Math Reference Implementation + Properties

These tests lock the formula at the math level before any model wiring.

**Files:**
- Modify: `backend/tests/test_gnn_rotate.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_gnn_rotate.py`:

```python
def _rotate_score(h_re, h_im, t_re, t_im, theta):
    """Reference RotatE score: -||h o r - t||_2 with r = cos(theta) + i sin(theta).

    h_re, h_im, t_re, t_im: (B, k) real tensors
    theta: (B, k) phase angles
    Returns: (B,) negative-distance scores in (-inf, 0].
    """
    r_re, r_im = torch.cos(theta), torch.sin(theta)
    hr_re = h_re * r_re - h_im * r_im
    hr_im = h_re * r_im + h_im * r_re
    diff_re = hr_re - t_re
    diff_im = hr_im - t_im
    d_squared = torch.sum(diff_re * diff_re + diff_im * diff_im, dim=1)
    return -torch.sqrt(d_squared + 1e-9)


def test_rotate_zero_phase_equals_translation_distance():
    """At theta=0 for all i, r = (1, 0), so h o r = h.
    Score reduces to -||h - t||_2 (pure translation distance, no rotation)."""
    torch.manual_seed(0)
    B, k = 4, 8
    h_re, h_im = torch.randn(B, k), torch.randn(B, k)
    t_re, t_im = torch.randn(B, k), torch.randn(B, k)
    theta = torch.zeros(B, k)

    actual = _rotate_score(h_re, h_im, t_re, t_im, theta)
    expected_d2 = ((h_re - t_re) ** 2 + (h_im - t_im) ** 2).sum(dim=1)
    expected = -torch.sqrt(expected_d2 + 1e-9)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_rotate_phase_pi_negates_real_part():
    """At theta=pi for all i, r = (-1, 0), so h o r = -h.
    Score equals -||-h - t|| = -||h + t||."""
    torch.manual_seed(1)
    B, k = 3, 6
    h_re, h_im = torch.randn(B, k), torch.randn(B, k)
    t_re, t_im = torch.randn(B, k), torch.randn(B, k)
    theta = torch.full((B, k), math.pi)

    actual = _rotate_score(h_re, h_im, t_re, t_im, theta)
    # h o e^{i pi} = h * (-1) (since cos(pi) = -1, sin(pi) = 0 — well, ~ 1.2e-16)
    expected_d2 = ((-h_re - t_re) ** 2 + (-h_im - t_im) ** 2).sum(dim=1)
    expected = -torch.sqrt(expected_d2 + 1e-9)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_rotate_score_always_non_positive():
    """Score = -sqrt(d^2 + eps) is always <= 0 (with floor near -sqrt(eps))."""
    torch.manual_seed(2)
    B, k = 32, 16
    h_re, h_im = torch.randn(B, k), torch.randn(B, k)
    t_re, t_im = torch.randn(B, k), torch.randn(B, k)
    theta = torch.empty(B, k).uniform_(-math.pi, math.pi)
    score = _rotate_score(h_re, h_im, t_re, t_im, theta)
    assert (score <= 0).all(), f"all scores must be <= 0, got max={score.max()}"


def test_rotate_score_zero_when_h_equals_rotated_t():
    """When t = h o r (perfect translation match), score should be near 0.
    (Exactly 0 minus eps-floor: -sqrt(1e-9) ~ -3.16e-5.)"""
    torch.manual_seed(3)
    B, k = 4, 8
    h_re, h_im = torch.randn(B, k), torch.randn(B, k)
    theta = torch.empty(B, k).uniform_(-math.pi, math.pi)
    r_re, r_im = torch.cos(theta), torch.sin(theta)
    # Construct t = h o r exactly so the difference is zero before the eps.
    t_re = h_re * r_re - h_im * r_im
    t_im = h_re * r_im + h_im * r_re

    score = _rotate_score(h_re, h_im, t_re, t_im, theta)
    # All scores should be near -sqrt(eps) = -sqrt(1e-9) ~ -3.16e-5
    assert (score > -1e-3).all(), "perfect match scores should be near 0"
    assert (score <= 0).all(), "scores still must be <= 0 by definition"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_gnn_rotate.py -v -k "rotate_zero or rotate_phase or rotate_score"`
Expected: 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_gnn_rotate.py
git commit -m "test(gnn): RotatE math reference (theta=0 reduces to translation, theta=pi negates real, scores are non-positive, perfect match gives near-zero)"
```

---

## Task 3: `CompGCNAuditModel` Decoder Dispatch

**Files:**
- Modify: `backend/src/gnn_module.py:211-246` (CompGCNAuditModel class)
- Modify: `backend/tests/test_gnn_rotate.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_gnn_rotate.py`:

```python
def test_compgcn_audit_model_default_decoder_is_distmult():
    """Without an explicit decoder kwarg, CompGCNAuditModel defaults to DistMult."""
    from src.gnn_module import CompGCNAuditModel
    model = CompGCNAuditModel(in_channels=64, hidden_channels=32, num_relations=3)
    assert hasattr(model, "decoder"), "model must expose .decoder attribute"
    assert model.decoder == "distmult"


def test_compgcn_audit_model_distmult_logits_unchanged_at_seed():
    """Default DistMult path produces the same edge_logits as the pre-Run-9 code.

    Reference: same seed, hand-rolled DistMult sum(x[src] * rel * x[dst]).
    """
    from src.gnn_module import CompGCNAuditModel
    torch.manual_seed(42)
    model = CompGCNAuditModel(in_channels=8, hidden_channels=16, num_relations=3, dropout=0.0)
    model.eval()  # disable dropout

    x = torch.randn(5, 16)  # already-encoded node embeddings
    edge_index = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 2], dtype=torch.long)

    with torch.no_grad():
        actual = model.edge_logits(x, edge_index, edge_type)
        # Reference DistMult
        rel = model.rel_emb(edge_type)
        expected = torch.sum(x[edge_index[0]] * rel * x[edge_index[1]], dim=1)
    assert torch.allclose(actual, expected, atol=1e-7)


def test_compgcn_audit_model_rotate_decoder_constructs_phase_embedding():
    """When decoder='rotate', model must instantiate self.rel_phase as
    Embedding(num_relations, hidden//2) with phases in [-pi, pi]."""
    from src.gnn_module import CompGCNAuditModel
    import torch.nn as nn
    torch.manual_seed(0)
    model = CompGCNAuditModel(
        in_channels=8, hidden_channels=16, num_relations=3,
        decoder="rotate",
    )
    assert model.decoder == "rotate"
    assert hasattr(model, "rel_phase"), "rotate decoder must add rel_phase embedding"
    assert isinstance(model.rel_phase, nn.Embedding)
    assert model.rel_phase.num_embeddings == 3
    assert model.rel_phase.embedding_dim == 8  # hidden_channels // 2

    # Phases initialised in [-pi, pi]
    phases = model.rel_phase.weight.detach()
    assert (phases.abs() <= math.pi + 1e-6).all(), "phases must be in [-pi, pi]"


def test_compgcn_audit_model_rotate_logits_match_reference_formula():
    """edge_logits with decoder='rotate' must equal the reference
    _rotate_score formula computed independently from the same weights."""
    from src.gnn_module import CompGCNAuditModel
    torch.manual_seed(7)
    hidden = 16
    k = hidden // 2  # 8
    model = CompGCNAuditModel(
        in_channels=8, hidden_channels=hidden, num_relations=3,
        decoder="rotate", dropout=0.0,
    )
    model.eval()

    x = torch.randn(5, hidden)
    edge_index = torch.tensor([[0, 1, 2, 3], [3, 4, 0, 1]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    with torch.no_grad():
        actual = model.edge_logits(x, edge_index, edge_type)

        # Reference RotatE
        h, t = x[edge_index[0]], x[edge_index[1]]
        h_re, h_im = h[:, :k], h[:, k:]
        t_re, t_im = t[:, :k], t[:, k:]
        theta = model.rel_phase(edge_type)
        expected = _rotate_score(h_re, h_im, t_re, t_im, theta)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_compgcn_audit_model_rotate_and_distmult_differ():
    """Sanity: at the same seed and inputs, the two decoders produce different
    edge_logits. If they're identical, the dispatch is broken."""
    from src.gnn_module import CompGCNAuditModel
    torch.manual_seed(11)
    model_dm = CompGCNAuditModel(in_channels=8, hidden_channels=16, num_relations=3,
                                 decoder="distmult", dropout=0.0)
    torch.manual_seed(11)
    model_rt = CompGCNAuditModel(in_channels=8, hidden_channels=16, num_relations=3,
                                 decoder="rotate", dropout=0.0)
    model_dm.eval()
    model_rt.eval()

    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [3, 0, 1]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 2], dtype=torch.long)

    with torch.no_grad():
        out_dm = model_dm.edge_logits(x, edge_index, edge_type)
        out_rt = model_rt.edge_logits(x, edge_index, edge_type)
    assert not torch.allclose(out_dm, out_rt, atol=1e-3), \
        "DistMult and RotatE must produce distinct logits at identical inputs"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_gnn_rotate.py -v`
Expected: 5 of these new tests FAIL (`...default_decoder_is_distmult`, `...rotate_decoder_constructs...`, `...rotate_logits_match_reference...`, `...rotate_and_distmult_differ`, plus possibly `...distmult_logits_unchanged...`). The 4 math-reference tests from Task 2 still pass.

- [ ] **Step 3: Modify `CompGCNAuditModel` to support decoder dispatch**

In `backend/src/gnn_module.py`, replace the `CompGCNAuditModel` class (lines 211-246) with:

```python
class CompGCNAuditModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_relations: int,
        dropout: float = 0.0,
        decoder: str = "distmult",
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.node_projection = nn.Linear(in_channels, hidden_channels)
        # Encoder relation embeddings — used by CompGCN message passing.
        # Independent of the decoder's relation parameters (per Vashishth+ 2020
        # encoder-decoder split: Table 4 evaluates DistMult / TransE / ConvE
        # decoders on top of the same CompGCN encoder).
        self.rel_emb = nn.Embedding(num_relations, hidden_channels)
        self.layer1 = CompGCNLayer(hidden_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.layer2 = CompGCNLayer(hidden_channels, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.layer3 = CompGCNLayer(hidden_channels, hidden_channels)

        self.decoder = decoder
        if decoder == "rotate":
            # RotatE relation embeddings are PHASE ANGLES, not vectors.
            # Half the hidden dim because we treat the encoder's real-valued
            # output as a complex vector by splitting halves: h_re = x[:k], h_im = x[k:].
            # Reference: Sun et al. 2019, "RotatE: Knowledge Graph Embedding by
            # Relational Rotation in Complex Space", eq. 14.
            self.rel_phase = nn.Embedding(num_relations, hidden_channels // 2)
            nn.init.uniform_(self.rel_phase.weight, -math.pi, math.pi)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.node_projection(x)
        x = self.dropout(F.relu(self.norm1(self.layer1(x, edge_index, edge_type, self.rel_emb.weight))))
        x = self.dropout(F.relu(self.norm2(self.layer2(x, edge_index, edge_type, self.rel_emb.weight))))
        x = self.layer3(x, edge_index, edge_type, self.rel_emb.weight)
        return self.dropout(x)

    def edge_logits(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        if self.decoder == "rotate":
            return self._rotate_logits(x, edge_index, edge_type)
        # DistMult (Runs 1-8 default) — unchanged
        src, dst = edge_index
        rel = self.rel_emb(edge_type)
        return torch.sum(x[src] * rel * x[dst], dim=1)

    def _rotate_logits(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """RotatE composition: -||h o r - t||_2 in complex space.

        Encoder output x is real-valued (B, hidden). Treat it as a complex
        vector by splitting halves: h_re = x[src][:, :k], h_im = x[src][:, k:].
        Relations are phase angles theta in [-pi, pi]; r = cos(theta) + i sin(theta).
        Score = -sqrt(sum_i (hr_re_i - t_re_i)^2 + (hr_im_i - t_im_i)^2 + eps).

        Eps prevents NaN gradient when distance -> 0 (perfect-match positive triple).
        """
        src, dst = edge_index
        h, t = x[src], x[dst]
        k = self.rel_phase.embedding_dim
        h_re, h_im = h[:, :k], h[:, k:]
        t_re, t_im = t[:, :k], t[:, k:]
        theta = self.rel_phase(edge_type)
        r_re, r_im = torch.cos(theta), torch.sin(theta)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        d_squared = torch.sum(diff_re * diff_re + diff_im * diff_im, dim=1)
        return -torch.sqrt(d_squared + 1e-9)

    def edge_scores(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.edge_logits(x, edge_index, edge_type))
```

Also add `import math` near the top of `backend/src/gnn_module.py` if not already imported.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_gnn_rotate.py -v`
Expected: All 9 tests PASS (1 config + 4 math + 4 model wiring).

- [ ] **Step 5: Run existing GNN tests for regression check**

Run: `cd backend && python -m pytest tests/test_gnn_self_adversarial.py tests/test_gnn_neg_sampling.py tests/test_gnn_loader_labels.py -v`
Expected: All 17 existing tests PASS — encoder is unchanged, default decoder is distmult, no behavioural change at default config.

- [ ] **Step 6: Commit**

```bash
git add backend/src/gnn_module.py backend/tests/test_gnn_rotate.py
git commit -m "feat(gnn): add RotatE decoder via runtime dispatch (Sun et al. 2019)"
```

---

## Task 4: Wire `Config.COMPGCN_DECODER` into `run_audit` and `recover_from_checkpoint`

**Files:**
- Modify: `backend/src/gnn_module.py` (run_audit model construction, checkpoint meta dump, AuditRun MERGE; recover_from_checkpoint model construction, AuditRun MERGE)
- Modify: `backend/tests/test_gnn_rotate.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_gnn_rotate.py`:

```python
def _stub_audit_data():
    """Shared MagicMock data + loader patch for run_audit tests."""
    from unittest.mock import MagicMock
    data = MagicMock()
    data.x = torch.randn(6, 4)
    data.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    data.edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    data.node_type = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    data.edge_rel_id = ["r0", "r1", "r2", "r3"]
    fake_loader = MagicMock()
    fake_loader.fetch_graph_data.return_value = (
        data, {"USES": 0, "PROPOSES": 1}, {}, {"A": 0, "B": 1}
    )
    return data, fake_loader


def _patch_neo4j_session(monkeypatch):
    from unittest.mock import MagicMock
    session = MagicMock()
    session.__enter__ = lambda self_: self_
    session.__exit__ = lambda *a: False
    driver = MagicMock()
    driver.session.return_value = session
    from src.db import DatabaseManager
    monkeypatch.setattr(DatabaseManager, "refresh", staticmethod(lambda: driver))
    return session


def test_run_audit_uses_rotate_when_configured(monkeypatch, tmp_path):
    """When Config.COMPGCN_DECODER='rotate', the model constructed inside
    run_audit must have decoder='rotate' and a rel_phase embedding."""
    import src.gnn_module as gnn_module
    from src.config import Config

    monkeypatch.setattr(gnn_module, "CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr(gnn_module, "CHECKPOINT_PATH", str(tmp_path / "best.pt"))
    monkeypatch.setattr(gnn_module, "CHECKPOINT_META_PATH", str(tmp_path / "best.json"))

    _, fake_loader = _stub_audit_data()
    import src.gnn_loader as gnn_loader_module
    monkeypatch.setattr(gnn_loader_module, "GNNLoader", lambda: fake_loader, raising=False)
    monkeypatch.setattr(gnn_module, "_evaluate_auc", lambda *a, **kw: 0.97)
    monkeypatch.setattr(gnn_module, "_evaluate_mrr", lambda *a, **kw: 0.93)
    monkeypatch.setattr(Config, "COMPGCN_EPOCHS", 1, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_PATIENCE", 5, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_LOSS", "bpr", raising=False)
    monkeypatch.setattr(Config, "COMPGCN_DECODER", "rotate", raising=False)
    monkeypatch.setattr(Config, "COMPGCN_AUC_GUARDRAIL", 0.95, raising=False)
    _patch_neo4j_session(monkeypatch)

    # Spy on CompGCNAuditModel to capture the decoder kwarg used.
    captured = {}
    real_init = gnn_module.CompGCNAuditModel.__init__

    def spy_init(self, *args, **kwargs):
        captured["decoder"] = kwargs.get("decoder", "distmult")
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(gnn_module.CompGCNAuditModel, "__init__", spy_init)

    gnn_module.run_audit()

    assert captured.get("decoder") == "rotate", \
        f"run_audit must construct model with decoder='rotate', got {captured!r}"


def test_run_audit_checkpoint_meta_records_decoder(monkeypatch, tmp_path):
    """Checkpoint meta JSON must include 'decoder' field for recovery attribution."""
    import json
    import src.gnn_module as gnn_module
    from src.config import Config

    monkeypatch.setattr(gnn_module, "CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr(gnn_module, "CHECKPOINT_PATH", str(tmp_path / "best.pt"))
    monkeypatch.setattr(gnn_module, "CHECKPOINT_META_PATH", str(tmp_path / "best.json"))

    _, fake_loader = _stub_audit_data()
    import src.gnn_loader as gnn_loader_module
    monkeypatch.setattr(gnn_loader_module, "GNNLoader", lambda: fake_loader, raising=False)
    monkeypatch.setattr(gnn_module, "_evaluate_auc", lambda *a, **kw: 0.97)
    monkeypatch.setattr(gnn_module, "_evaluate_mrr", lambda *a, **kw: 0.93)
    monkeypatch.setattr(Config, "COMPGCN_EPOCHS", 2, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_PATIENCE", 5, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_LOSS", "bpr", raising=False)
    monkeypatch.setattr(Config, "COMPGCN_DECODER", "rotate", raising=False)
    monkeypatch.setattr(Config, "COMPGCN_AUC_GUARDRAIL", 0.95, raising=False)
    _patch_neo4j_session(monkeypatch)

    gnn_module.run_audit()

    meta = json.loads((tmp_path / "best.json").read_text())
    assert meta.get("decoder") == "rotate", \
        f"checkpoint meta must record decoder, got {meta}"


def test_audit_run_node_records_decoder(monkeypatch, tmp_path):
    """The AuditRun MERGE in run_audit must SET run.decoder = $decoder."""
    import src.gnn_module as gnn_module
    from src.config import Config

    monkeypatch.setattr(gnn_module, "CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr(gnn_module, "CHECKPOINT_PATH", str(tmp_path / "best.pt"))
    monkeypatch.setattr(gnn_module, "CHECKPOINT_META_PATH", str(tmp_path / "best.json"))

    _, fake_loader = _stub_audit_data()
    import src.gnn_loader as gnn_loader_module
    monkeypatch.setattr(gnn_loader_module, "GNNLoader", lambda: fake_loader, raising=False)
    monkeypatch.setattr(gnn_module, "_evaluate_auc", lambda *a, **kw: 0.97)
    monkeypatch.setattr(gnn_module, "_evaluate_mrr", lambda *a, **kw: 0.93)
    monkeypatch.setattr(Config, "COMPGCN_EPOCHS", 1, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_PATIENCE", 5, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_LOSS", "bpr", raising=False)
    monkeypatch.setattr(Config, "COMPGCN_DECODER", "rotate", raising=False)
    monkeypatch.setattr(Config, "COMPGCN_AUC_GUARDRAIL", 0.95, raising=False)
    session = _patch_neo4j_session(monkeypatch)

    gnn_module.run_audit()

    merge_calls = [c for c in session.run.call_args_list if "MERGE" in str(c.args[0])]
    assert merge_calls, "AuditRun MERGE must be issued"
    matching = [c for c in merge_calls
                if "decoder" in str(c.args[0]) and "decoder" in c.kwargs]
    assert matching, \
        "AuditRun Cypher must SET run.decoder and pass decoder= kwarg"
    for c in matching:
        assert c.kwargs.get("decoder") == "rotate"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_gnn_rotate.py -v -k "uses_rotate or checkpoint_meta_records_decoder or audit_run_node_records_decoder"`
Expected: 3 FAILS.

- [ ] **Step 3: Pass `decoder` kwarg in `run_audit` model construction**

In `backend/src/gnn_module.py`, find `run_audit`'s `model = CompGCNAuditModel(...)` block. Update to:

```python
    model = CompGCNAuditModel(
        in_channels=data.x.size(1),
        hidden_channels=Config.COMPGCN_HIDDEN_CHANNELS,
        num_relations=num_rels,
        dropout=Config.COMPGCN_DROPOUT,
        decoder=Config.COMPGCN_DECODER,
    )
    logger.info(
        "CompGCN training loss=%s (margin=%s) decoder=%s",
        loss_mode, bpr_margin, Config.COMPGCN_DECODER,
    )
```

The existing `logger.info("CompGCN training loss=%s (margin=%s)", loss_mode, bpr_margin)` line two lines below should be deleted to avoid duplicate logging.

- [ ] **Step 4: Add `decoder` to checkpoint meta JSON in run_audit**

Update the existing `json.dump({...}, f)` block (added in Run 8) to include `decoder`:

```python
                with open(CHECKPOINT_META_PATH, "w") as f:
                    json.dump({
                        "best_epoch": best_epoch_idx,
                        "best_auc": float(best_auc),
                        "train_loss": final_train_loss,
                        "hidden_channels": Config.COMPGCN_HIDDEN_CHANNELS,
                        "loss_mode": loss_mode,
                        "adv_temp": float(Config.COMPGCN_ADV_TEMP),
                        "decoder": Config.COMPGCN_DECODER,
                        "num_relations": num_rels,
                        "in_channels": int(data.x.size(1)),
                        "saved_at": _utc_now_iso(),
                    }, f)
```

- [ ] **Step 5: Add `decoder` to run_audit's AuditRun MERGE**

Find the MERGE block in `run_audit`. Append `run.decoder = $decoder` to the SET clause and `decoder=Config.COMPGCN_DECODER` to the kwargs:

```python
            session.run(
                f"""
                MERGE (run:{Config.AUDIT_RUN_LABEL} {{run_id: $run_id}})
                SET run.status = $status,
                    run.audit_mode = 'compgcn',
                    run.audit_model = 'CompGCNAuditModel',
                    run.started_at = $started_at,
                    run.completed_at = $completed_at,
                    run.audited_relationships = $audited_relationships,
                    run.graph_nodes = $graph_nodes,
                    run.graph_relationship_types = $graph_relationship_types,
                    run.hidden_channels = $hidden_channels,
                    run.epochs = $epochs,
                    run.learning_rate = $learning_rate,
                    run.weight_decay = $weight_decay,
                    run.validation_split = $validation_split,
                    run.patience = $patience,
                    run.dropout = $dropout,
                    run.label_smoothing = $label_smoothing,
                    run.grad_clip = $grad_clip,
                    run.neg_ratio = $neg_ratio,
                    run.loss = $loss,
                    run.bpr_margin = $bpr_margin,
                    run.neg_sampling = $neg_sampling,
                    run.label_pool_sizes = $label_pool_sizes,
                    run.auc_guardrail_min = $auc_guardrail_min,
                    run.auc_roc = $auc_roc,
                    run.mrr = $mrr,
                    run.mrr_uniform = $mrr_uniform,
                    run.mrr_type_aware = $mrr_type_aware,
                    run.train_loss = $train_loss,
                    run.adv_temp = $adv_temp,
                    run.decoder = $decoder
                """,
                run_id=audit_run_id,
                status=audit_status,
                started_at=audit_started_at,
                completed_at=audit_completed_at,
                audited_relationships=int(data.edge_index.size(1)),
                graph_nodes=int(data.x.size(0)),
                graph_relationship_types=int(num_rels),
                hidden_channels=Config.COMPGCN_HIDDEN_CHANNELS,
                epochs=Config.COMPGCN_EPOCHS,
                learning_rate=Config.COMPGCN_LEARNING_RATE,
                weight_decay=Config.COMPGCN_WEIGHT_DECAY,
                validation_split=Config.COMPGCN_VALIDATION_SPLIT,
                patience=Config.COMPGCN_PATIENCE,
                dropout=Config.COMPGCN_DROPOUT,
                label_smoothing=Config.COMPGCN_LABEL_SMOOTHING,
                grad_clip=Config.COMPGCN_GRAD_CLIP,
                neg_ratio=neg_ratio,
                loss=loss_mode,
                bpr_margin=bpr_margin,
                neg_sampling=neg_sampling_mode,
                label_pool_sizes=json.dumps(pool_sizes),
                auc_guardrail_min=auc_guardrail,
                auc_roc=final_auc,
                mrr=final_mrr,
                mrr_uniform=mrr_uniform,
                mrr_type_aware=mrr_type_aware,
                train_loss=final_train_loss,
                adv_temp=float(Config.COMPGCN_ADV_TEMP),
                decoder=Config.COMPGCN_DECODER,
            )
```

- [ ] **Step 6: Update `recover_from_checkpoint` to thread `decoder` through**

In `recover_from_checkpoint`, the model is constructed from meta. Update:

```python
    model = CompGCNAuditModel(
        in_channels=data.x.size(1),
        hidden_channels=meta.get("hidden_channels", Config.COMPGCN_HIDDEN_CHANNELS),
        num_relations=num_rels,
        dropout=Config.COMPGCN_DROPOUT,
        decoder=meta.get("decoder", "distmult"),
    )
```

Then update the recovery's AuditRun MERGE to include `decoder`:

```python
        session.run(
            f"""
            MERGE (run:{Config.AUDIT_RUN_LABEL} {{run_id: $run_id}})
            SET run.status = 'recovered_checkpoint',
                run.audit_mode = 'compgcn',
                run.audit_model = 'CompGCNAuditModel',
                run.started_at = $saved_at,
                run.completed_at = $completed_at,
                run.best_epoch = $best_epoch,
                run.auc_roc = $auc_roc,
                run.mrr = $mrr,
                run.mrr_uniform = $mrr_uniform,
                run.mrr_type_aware = $mrr_type_aware,
                run.loss = $loss,
                run.adv_temp = $adv_temp,
                run.decoder = $decoder
            """,
            run_id=audit_run_id,
            saved_at=meta.get("saved_at"),
            completed_at=audit_completed_at,
            best_epoch=meta.get("best_epoch"),
            auc_roc=final_auc,
            mrr=final_mrr,
            mrr_uniform=mrr_uniform,
            mrr_type_aware=mrr_type_aware,
            loss=meta.get("loss_mode"),
            adv_temp=float(meta.get("adv_temp", 0.0)),
            decoder=meta.get("decoder", "distmult"),
        )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_gnn_rotate.py -v`
Expected: All 12 tests PASS.

- [ ] **Step 8: Run full backend test suite for regression check**

Run: `cd backend && python -m pytest 2>&1 | tail -10`
Expected: Same 20 passing + 2 pre-existing api failures as before. No new failures.

- [ ] **Step 9: Commit**

```bash
git add backend/src/gnn_module.py backend/tests/test_gnn_rotate.py
git commit -m "feat(gnn): persist decoder in checkpoint meta and AuditRun node; thread through run_audit + recover"
```

---

## Task 5: Run 9 Launcher Script

**Files:**
- Create: `backend/run_logs/rotate_audit.py`

- [ ] **Step 1: Write the launcher**

Create `backend/run_logs/rotate_audit.py`:

```python
"""Run 9: BPR + self-adversarial alpha=1.0 + RotatE decoder.

Sets COMPGCN_LOSS=bpr, COMPGCN_NEG_SAMPLING=uniform, COMPGCN_ADV_TEMP=1.0,
COMPGCN_DECODER=rotate and runs the full audit + Neo4j sync. Targets
MRR > 0.95 by replacing DistMult's symmetric scoring with RotatE's
relational rotation in complex space (Sun et al. 2019).

Encoder is unchanged from Run 8 (3-layer CompGCN + LayerNorm). Loss is
unchanged from Run 8 (BPR + self-adversarial alpha=1.0). Only the decoder
swaps DistMult -> RotatE.

Rollback to Run 8 (DistMult decoder): re-run backend/run_logs/self_adversarial_audit.py.
"""
from __future__ import annotations

import os
import sys
import time

# Running from backend/run_logs/; backend/ must be on sys.path for `src` imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must set before importing src.config
os.environ["COMPGCN_LOSS"] = "bpr"
os.environ["COMPGCN_BPR_MARGIN"] = os.environ.get("COMPGCN_BPR_MARGIN", "0.0")
os.environ["COMPGCN_NEG_SAMPLING"] = os.environ.get("COMPGCN_NEG_SAMPLING", "uniform")
os.environ["COMPGCN_ADV_TEMP"] = os.environ.get("COMPGCN_ADV_TEMP", "1.0")
os.environ["COMPGCN_DECODER"] = os.environ.get("COMPGCN_DECODER", "rotate")
os.environ["COMPGCN_AUC_GUARDRAIL"] = os.environ.get("COMPGCN_AUC_GUARDRAIL", "0.95")


def main():
    from src.gnn_module import run_audit, get_training_history
    from src.evaluation import persist_gnn_metrics

    t0 = time.time()
    print("ROTATE_AUDIT_BEGIN", flush=True)
    print(
        f"  loss=bpr neg_sampling={os.environ['COMPGCN_NEG_SAMPLING']} "
        f"adv_temp={os.environ['COMPGCN_ADV_TEMP']} "
        f"decoder={os.environ['COMPGCN_DECODER']} "
        f"auc_guardrail={os.environ['COMPGCN_AUC_GUARDRAIL']}",
        flush=True,
    )
    try:
        run_audit()
        th = get_training_history()
        persist_gnn_metrics(th)
        print(
            f"ROTATE_AUDIT_DONE elapsed_min={(time.time()-t0)/60.0:.2f} "
            f"best_epoch={th.get('best_epoch')} "
            f"auc={th.get('final_auc_roc')} "
            f"mrr_uniform={th.get('mrr_uniform')} "
            f"mrr_type_aware={th.get('mrr_type_aware')}",
            flush=True,
        )
    except Exception as e:
        import traceback
        print(f"ROTATE_AUDIT_FAILED {type(e).__name__} {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check launcher syntax**

Run: `cd backend && python -c "import ast; ast.parse(open('run_logs/rotate_audit.py').read()); print('SYNTAX OK')"`
Expected: `SYNTAX OK`.

- [ ] **Step 3: Commit**

```bash
git add backend/run_logs/rotate_audit.py
git commit -m "chore(run_logs): add rotate_audit.py launcher (Run 9)"
```

---

## Task 6: DistMult Reproducibility Check at HEAD

Verify that with `COMPGCN_DECODER=distmult` (default), Run 9 produces the same epoch-1 AUC as Run 8.

**Files:**
- Generates: `backend/run_logs/repro_check_distmult_default.log`

- [ ] **Step 1: Run a 2-epoch dry run with the default decoder**

Run:
```bash
cd backend && COMPGCN_LOSS=bpr COMPGCN_ADV_TEMP=1.0 COMPGCN_EPOCHS=2 COMPGCN_NEG_SAMPLING=uniform python -c "from src.gnn_module import run_audit; run_audit()" 2>&1 | tee run_logs/repro_check_distmult_default.log | grep -E "epoch 1/|epoch 2/|decoder="
```
Expected: epoch 1 AUC matches Run 8's epoch 1 (0.6554 from `audit_self_adversarial.log`). The log line should also confirm `decoder=distmult`.

- [ ] **Step 2: Confirm reproducibility**

Run: `grep "epoch 1/2" backend/run_logs/repro_check_distmult_default.log backend/run_logs/audit_self_adversarial.log`
Expected: identical AUC values within ±0.0001 FP tolerance for the first epoch.

If values differ by more than ±0.0001: STOP. The decoder dispatch may have inadvertently altered the DistMult code path. Investigate before continuing.

- [ ] **Step 3: Commit the verification log**

```bash
git add -f backend/run_logs/repro_check_distmult_default.log
git commit -m "chore(run_logs): DistMult reproducibility check at HEAD (Run 9 default)"
```

---

## Task 7: Execute Run 9 Audit (Full RotatE)

**Files:**
- Generates: `backend/run_logs/audit_rotate.log`
- Generates: `backend/run_logs/eval_chain_rotate.log`
- Modifies: `backend/run_logs/compgcn_best.pt` + `compgcn_best_meta.json`
- Modifies: Neo4j (writes new plausibility scores; AuditRun node)

- [ ] **Step 1: Verify Neo4j is reachable**

Run: `cd backend && python -c "from src.db import DatabaseManager; d = DatabaseManager.refresh(); d.verify_connectivity(); print('NEO4J_OK')"`
Expected: `NEO4J_OK`. If `ValueError: Cannot resolve address ...`: Aura instance is paused; resume from console.neo4j.io and wait ~60s.

- [ ] **Step 2: Run the audit**

Run:
```bash
cd backend && python run_logs/rotate_audit.py 2>&1 | tee run_logs/audit_rotate.log
```
Expected runtime: ~2.5–3 min (RotatE adds ~10% per epoch). Final line `ROTATE_AUDIT_DONE` with `auc=0.XX mrr_uniform=0.XX`.

If `ROTATE_AUDIT_FAILED`: capture the traceback, do NOT proceed, treat as a debugging task.

- [ ] **Step 3: Verify the AUC guardrail did not trip**

Run: `grep -E "guardrail|aborted" backend/run_logs/audit_rotate.log`
Expected: no matches. If guardrail tripped (`final_auc < 0.95`), Neo4j scores were NOT synced — investigate why before continuing. RotatE on small graphs can underperform DistMult; this is a documented risk in the spec §Risks.

- [ ] **Step 4: Verify Neo4j scores were synced**

Run:
```bash
cd backend && python -c "
from src.db import DatabaseManager
d = DatabaseManager.refresh()
with d.session() as s:
    r = s.run('MATCH ()-[r]->() WHERE r.plausibility_score IS NOT NULL RETURN count(r) AS n, avg(r.plausibility_score) AS avg, min(r.plausibility_score) AS min, max(r.plausibility_score) AS max').single()
    print(r)
"
```
Expected: `n=6419`. RotatE-trained scores are bounded by sigmoid(0)=0.5 above (since logits are <=0), so `max` should be ~0.5 and `avg` likely 0.3–0.45. This is the **calibration shift** documented in spec §3.6.

- [ ] **Step 5: Run the post-eval chain**

Run:
```bash
cd backend && python run_logs/post_audit_eval.py 2>&1 | tee run_logs/eval_chain_rotate.log
```
Expected runtime: ~20 min (5 queries × 4 thresholds + ablation). Final output `EVAL_DONE elapsed_min=XX.X`.

- [ ] **Step 6: If τ=0.95 returns no triplets, run a finer threshold sweep**

If the threshold sweep at τ=0.95 reports `null` or `no triplets pass`, RotatE's compressed score range requires recalibration. Run a manual finer sweep:

```bash
cd backend && python -c "
import asyncio
from src.evaluation import run_grounding_evaluation
async def main():
    for tau in [0.20, 0.30, 0.35, 0.40, 0.45, 0.49]:
        r = await run_grounding_evaluation(mode='full_stack', threshold=tau, persist_to_ablation=False)
        print(f'tau={tau} G={r.get(\"grounding_score\")} F={r.get(\"faithfulness_score\")}')
asyncio.run(main())
" 2>&1 | tee run_logs/rotate_finer_sweep.log
```
Expected: identifies the τ at which Grounding peaks (the **canonical RotatE τ**, analogous to Run 5's τ=0.30 calibration for BCE).

- [ ] **Step 7: Record headline numbers**

Open `backend/run_logs/audit_rotate.log` and `backend/run_logs/eval_chain_rotate.log`. Record on a scratch note:
- Best epoch (training loop)
- Final AUC-ROC
- MRR uniform eval
- MRR type-aware eval
- Score distribution buckets + max/avg/min
- Grounding @ canonical τ
- Faithfulness @ canonical τ
- Per-query G/F at canonical τ
- Threshold sweep table
- Wall-clock training time

These feed Task 8.

- [ ] **Step 8: Do NOT commit yet**

The logs and checkpoint are not committed in this task — Task 8 commits them with the TUNING_LOG update for one atomic narrative.

---

## Task 8: Append Run 9 Section to `TUNING_LOG.md`

**Files:**
- Modify: `backend/TUNING_LOG.md` (append after Run 8 section, before Overall Scoreboard)

- [ ] **Step 1: Locate the insertion point**

Run: `grep -n "## Overall Scoreboard" backend/TUNING_LOG.md`
Expected: single match at the line where `## Overall Scoreboard (as of 2026-05-03)` begins. The Run 9 section is inserted on the line before this.

- [ ] **Step 2: Write the Run 9 section**

Insert the following block immediately before the Overall Scoreboard heading. Replace `<DATE>` with today's date and `<...>` placeholders with Task 7's recorded numbers:

```markdown
## Run 9 — <DATE>: BPR + Self-Adversarial + RotatE Decoder (MRR Closure Attempt #2)

**Goal.** Close the last open paper KPI. Run 8 (BPR + self-adv α=1.0 + DistMult decoder) hit MRR 0.9119, AUC 0.9786, Grounding 0.9884, Faithfulness 0.9714 — three of four paper KPIs cleared. MRR remained 0.038 short, diagnosed as corpus-density-bound. This run swaps the **decoder** from DistMult to RotatE (Sun et al. 2019, eq. 14): relations as rotations in complex space, captured natively for asymmetric/inverse relations. Encoder, loss, sampling all unchanged from Run 8.

**Config delta vs Run 8.**

| Parameter | Run 8 | Run 9 |
|-----------|-------|-------|
| `COMPGCN_LOSS` | bpr | bpr |
| `COMPGCN_NEG_SAMPLING` | uniform | uniform |
| `COMPGCN_ADV_TEMP` | 1.0 | 1.0 |
| `COMPGCN_DECODER` | distmult (implicit) | **rotate** |
| All other hyperparams | identical | identical |

**Decoder formula.**

DistMult (Run 8): `s(h, r, t) = Σ h_i · r_i · t_i` (symmetric in (h,t) when r is symmetric)

RotatE (Run 9): `s(h, r, t) = -||h ∘ r - t||_2` where `r_i = e^{i θ_i} = cos θ_i + i sin θ_i`

Real-valued implementation: 256-dim encoder output split into 128-dim real + 128-dim imaginary halves. `rel_phase: Embedding(num_relations, 128)` stores phase angles in [-π, π].

**GNN Metrics.**

| Metric | Run 6 (DistMult) | Run 8 (DistMult + self-adv) | **Run 9 (RotatE + self-adv)** | Δ vs Run 8 | Target | Status |
|--------|------------------|------------------------------|-------------------------------|------------|--------|--------|
| AUC-ROC | 0.9688 | 0.9786 | <FILL> | <FILL> | > 0.95 | <FILL> |
| MRR (uniform eval) | 0.8860 | 0.9119 | <FILL> | <FILL> | > 0.95 | <FILL> |
| MRR (type-aware eval) | — | 0.8998 | <FILL> | <FILL> | > 0.95 | <FILL> |
| Grounding @ canonical τ | 0.987 (τ=0.95) | 0.9884 (τ=0.95) | <FILL> (τ=<FILL>) | <FILL> | > 0.98 | <FILL> |
| Faithfulness @ canonical τ | 0.979 (τ=0.95) | 0.9714 (τ=0.95) | <FILL> (τ=<FILL>) | <FILL> | high | <FILL> |
| Best epoch | 168/300 | 158/300 | <FILL> | — | — | — |
| Training wall-clock | 1.8 min | 1.68 min | <FILL> | — | — | — |

**Score distribution (RotatE plausibility = sigmoid(-distance) ∈ (0, 0.5]).**

| Bucket | Run 8 (DistMult+self-adv) | **Run 9 (RotatE+self-adv)** |
|--------|---------------------------|------------------------------|
| < 0.10 | 0 | <FILL> |
| 0.10 – 0.20 | 0 | <FILL> |
| 0.20 – 0.30 | 0 | <FILL> |
| 0.30 – 0.40 | 0 | <FILL> |
| 0.40 – 0.50 | 0 | <FILL> |
| 0.50 – 0.85 | 159 (2.5%) | <FILL> |
| 0.85 – 0.95 | 531 (8.3%) | <FILL> |
| ≥ 0.95 | 5,711 (89.0%) | <FILL> |
| max / avg / min | 1.0000 / 0.9770 / 0.0561 | <FILL> |

RotatE's compressed score range is by design: `sigmoid(score)` where `score = -distance ≤ 0`, so plausibility scores are bounded above by 0.5. **The canonical τ for RotatE is therefore not 0.95** — it is recalibrated post-hoc from the threshold sweep, analogous to Run 5's τ=0.30 calibration for BCE+label-smoothing.

**Threshold sweep.**

| τ | Grounding | Faithfulness |
|---|-----------|--------------|
| 0.30 | <FILL> | <FILL> |
| 0.40 | <FILL> | <FILL> |
| 0.45 | <FILL> | <FILL> |
| 0.49 | <FILL> | <FILL> |

**Per-query at canonical τ=<FILL>.**

| Query | Grounding | Faithfulness |
|-------|-----------|--------------|
| What are the key findings? | <FILL> | <FILL> |
| Who are the main researchers? | <FILL> | <FILL> |
| What methods were used? | <FILL> | <FILL> |
| What are the main results? | <FILL> | <FILL> |
| What datasets or concepts are discussed? | <FILL> | <FILL> |

**Interpretation.**

<FILL — narrate one of:
- If MRR ≥ 0.95: "H2 paper target ACHIEVED. RotatE decoder closes the last open KPI. The DistMult symmetric ceiling identified in Run 8's post-mortem is confirmed; RotatE's complex-space rotations capture the asymmetric structure of legal relations (USES, PROPOSES, EXTENDS) that DistMult cannot model expressively."
- If 0.92 ≤ MRR < 0.95: "Meaningful lift over Run 8 (+X.X pts MRR). RotatE outperforms DistMult on this corpus, confirming the decoder choice matters even at low density. Falls short of canonical 0.95 — corpus density (1.24 edges/node, 32× below FB15k) remains the dominant bound. Future work: corpus expansion combined with RotatE."
- If MRR < 0.92 or AUC regressed: "RotatE underperforms DistMult on this corpus. At ~5,000-node graph density, the complex-space expressivity advantage requires more data than is available; the simpler DistMult bilinear form is a better fit. Run 9 is reported as a documented decoder ablation; recommended defense config remains Run 8 (DistMult + self-adversarial). This is a defensible negative result — Vashishth+ 2020 reports decoder-choice sensitivity in their Table 4."
- Address any AUC drop, faithfulness shift, score-distribution change, calibration finding for τ.>

**Headline.** <FILL — one-line summary of the run outcome.>

**Run 9b (RotatE + 256-dim complex / wider hidden): SKIPPED unless** Run 9 lifts MRR but stalls below 0.95. The plan's gating rule is symmetric to Run 8b — only escalate ablations when the marginal lift is plausible.

**Operational notes.**
- DistMult reproducibility check at HEAD produced identical epoch-1 AUC to Run 8 within ±0.0001 (with `COMPGCN_DECODER=distmult` default). Verifies that the dispatch refactor does not alter the existing DistMult code path. Reproducibility log: `backend/run_logs/repro_check_distmult_default.log`.
- 12 new unit tests cover: config default, RotatE math properties (θ=0 reduces to translation, θ=π negates real part, scores ≤ 0, perfect-match score → 0), model dispatch (default DistMult unchanged, RotatE constructs rel_phase), wiring (run_audit uses Config.COMPGCN_DECODER), persistence (checkpoint meta + AuditRun MERGE both record decoder).
- AuditRun Neo4j node now has `run.decoder` property; checkpoint meta JSON gains `decoder` field. Recovery preserves attribution via `meta.get("decoder", "distmult")`.
- Training wall-clock: <FILL> min (Run 8: 1.68 min). RotatE adds ~10% per epoch (trig + sqrt) but converges similarly.

**Thesis Chapter 4 ablation extension (decoder column).**

| Run | Loss | Sampling | Adv. temp | Decoder | AUC | MRR (uniform) | Grounding | Faithfulness |
|-----|------|----------|-----------|---------|-----|---------------|-----------|--------------|
| 4 (BCE baseline) | BCE | Uniform | — | DistMult | 0.9502 | 0.8134 | — | — |
| 6 (BPR) | BPR | Uniform | 0 | DistMult | 0.9688 | 0.8860 | 0.987 | 0.979 |
| 7 (BPR + type-aware) | BPR | Same-label | 0 | DistMult | 0.9662 | 0.8873 | 0.988 | 0.91–0.95 |
| 8 (BPR + self-adv) | BPR | Uniform | 1.0 | DistMult | 0.9786 | 0.9119 | 0.9884 | 0.9714 |
| **9 (BPR + self-adv + RotatE)** | BPR | Uniform | 1.0 | **RotatE** | <FILL> | <FILL> | <FILL> | <FILL> |

This row fills the canonical decoder-ablation slot (cf. Vashishth+ 2020 Table 4 reports DistMult / TransE / ConvE separately).

**Reproduction.**
```bash
cd backend && python run_logs/rotate_audit.py
cd backend && python run_logs/post_audit_eval.py
```
```

- [ ] **Step 3: Replace all `<FILL>` and `<DATE>` markers**

Walk through every `<FILL>` and `<DATE>` in the inserted block; replace with the actual numbers from Task 7 Step 7. Pick the appropriate interpretation paragraph based on the observed MRR.

- [ ] **Step 4: Verify no markers remain**

Run: `grep -nE "<FILL>|<DATE>" backend/TUNING_LOG.md`
Expected: no matches.

- [ ] **Step 5: Update the Overall Scoreboard**

Find the `## Overall Scoreboard (as of 2026-05-03)` table. Update:
- Append a `BPR + self-adv + RotatE` column to the existing row table
- Update "Recommended configuration for thesis defense" — if MRR ≥ 0.95, set `COMPGCN_DECODER=rotate`; otherwise leave Run 8 recommended and add a "Run 9 attempted, decoder kept as DistMult" footnote
- Append a Run 9 finding to the Key Findings list (10th bullet)
- Append the Run 9 launcher to the Reproduction section

- [ ] **Step 6: Commit**

```bash
git add backend/TUNING_LOG.md backend/run_logs/audit_rotate.log backend/run_logs/eval_chain_rotate.log backend/run_logs/compgcn_best.pt backend/run_logs/compgcn_best_meta.json
[ -f backend/run_logs/rotate_finer_sweep.log ] && git add -f backend/run_logs/rotate_finer_sweep.log
git commit -m "docs(tuning): Run 9 — BPR + self-adv + RotatE decoder; MRR <FILL>"
```
(Replace `<FILL>` in commit message with achieved MRR rounded to 4 decimals.)

---

## Task 9: Update PAPER_TECHNICAL_INVENTORY.md

**Files:**
- Modify: `docs/paper/PAPER_TECHNICAL_INVENTORY.md`

- [ ] **Step 1: Append Run 9 row to §4.2 Run-by-Run Tuning Campaign**

Locate `## 4.2 Run-by-Run Tuning Campaign (Runs 1–8)`. Update the heading to "Runs 1–9" and append a Run 9 row to the table:

```markdown
| 9 | **<DATE>** | BPR | Uniform | **RotatE** | 1.0 | 3-layer + LN | <AUC> | <MRR> | <G> (τ=<canonical>) | <F> (τ=<canonical>) | Decoder ablation; <outcome summary> |
```

- [ ] **Step 2: Update §4.10 Final Scoreboard**

If MRR ≥ 0.95, update the H2 row from `−0.038 short` to `**PASS**` and the "Best Achieved" cell to the Run 9 MRR. Otherwise leave it but append a footnote: "Run 9 (RotatE decoder) attempted; MRR <X> — best in campaign but corpus-density-bound."

- [ ] **Step 3: Update §5.5.1 Future Work — Decoder upgrade**

Currently describes Run 9 as proposed. Update to "Run 9 — completed <DATE>" with the actual outcome:
- If MRR ≥ 0.95: "H2 closure achieved by RotatE decoder. Recommended defense config updated."
- Otherwise: "Empirical lift +<delta> MRR (best in campaign). Falls below 0.95; corpus density confirmed as principal bound."

- [ ] **Step 4: Append Run 9 commits to §11 Commit Reference**

Add the new commits at the top of the table:

```markdown
| <commit> | <DATE> | docs(tuning): Run 9 — BPR + self-adv + RotatE decoder |
| <commit> | <DATE> | chore(run_logs): DistMult reproducibility check at HEAD (Run 9 default) |
| <commit> | <DATE> | chore(run_logs): add rotate_audit.py launcher (Run 9) |
| <commit> | <DATE> | feat(gnn): persist decoder in checkpoint meta and AuditRun node |
| <commit> | <DATE> | feat(gnn): add RotatE decoder via runtime dispatch (Sun et al. 2019) |
| <commit> | <DATE> | test(gnn): RotatE math reference |
| <commit> | <DATE> | feat(gnn): add COMPGCN_DECODER config flag (default distmult) |
| <commit> | <DATE> | docs: Run 9 spec — RotatE decoder for CompGCN |
```

Replace `<commit>` with actual short SHAs from `git log --oneline -10`.

- [ ] **Step 5: Update Document version footer**

At the bottom of the file, update:

```markdown
**Document version:** 1.2 (Run 9 complete)
**Total length:** ~1,500+ lines
**Next update:** As needed for Run 10 (corpus expansion or RotatE + 256-complex).
```

- [ ] **Step 6: Commit**

```bash
git add docs/paper/PAPER_TECHNICAL_INVENTORY.md
git commit -m "docs(paper): incorporate Run 9 (RotatE decoder) into technical inventory"
```

---

## Task 10: Update Memory + Final Verification

**Files:**
- Create: `~/.claude/projects/C--Users-Franz-Samilo-Desktop-the-remembrance/memory/project_tuning_session_may3_run9.md`
- Modify: `~/.claude/projects/.../memory/MEMORY.md`

- [ ] **Step 1: Confirm full test suite passes**

Run: `cd backend && python -m pytest tests/test_gnn_self_adversarial.py tests/test_gnn_neg_sampling.py tests/test_gnn_loader_labels.py tests/test_gnn_rotate.py tests/test_utils.py -v 2>&1 | tail -10`
Expected: All GNN-related tests PASS (no regression).

- [ ] **Step 2: Confirm git working tree is clean of Run 9 work**

Run: `git status --short`
Expected: clean tree (modulo the pre-existing modifications already noted in `evaluation.py`, frontend files, etc. — those predate Run 9 and are not touched).

- [ ] **Step 3: Confirm commit chain**

Run: `git log --oneline -10`
Expected: Run 9 commits in correct order (config → tests → dispatch → persistence → launcher → repro check → tuning log → inventory).

- [ ] **Step 4: Write Run 9 memory entry**

Create `C:\Users\Franz Samilo\.claude\projects\C--Users-Franz-Samilo-Desktop-the-remembrance\memory\project_tuning_session_may3_run9.md` with this template (replace placeholders with Task 7's recorded numbers):

```markdown
---
name: GNN tuning session 2026-05-03 Run 9 (RotatE decoder)
description: <one-line outcome summary — H2 closed | partial lift | RotatE underperforms>
type: project
---

Run 9 — Replaced DistMult decoder with RotatE (Sun et al. 2019, eq. 14)
behind COMPGCN_DECODER='rotate'. Encoder, loss, sampling unchanged from Run 8.

**Why:** Run 8 closed ~2/3 of the MRR gap with self-adversarial weighting but
left 0.038 short of the canonical >0.95 paper target. Diagnosis: DistMult's
symmetric scoring is the expressivity ceiling for directed legal relations.
RotatE models relations as rotations in complex space — the canonical
asymmetric-relation decoder per Vashishth+ 2020 Table 4 ablation.

**How to apply:** <If MRR>=0.95: Recommended thesis defense config now
includes COMPGCN_DECODER=rotate. Architecture diagram updates to "CompGCN
with RotatE scoring". | Otherwise: Defense config remains Run 8 (DistMult);
Run 9 is documented as the canonical decoder ablation. Chapter 5 future
work emphasizes corpus expansion as the dominant lever.>

Headline numbers (<canonical τ for RotatE>):
- AUC-ROC: <FILL> (Run 8: 0.9786)
- MRR uniform eval: <FILL> (Run 8: 0.9119)
- MRR type-aware eval: <FILL>
- Grounding: <FILL>
- Faithfulness: <FILL>
- Best epoch <FILL>/300, training <FILL> min
- Score distribution: max <FILL> (RotatE bounded by 0.5 via sigmoid(-distance))

<Outcome paragraph: H2 closed | partial lift | regression — see TUNING_LOG.md
Run 9 section for full discussion.>

12 new unit tests in test_gnn_rotate.py cover config default, RotatE math
properties, model dispatch, wiring, persistence. AuditRun Neo4j node and
checkpoint meta JSON now record `decoder` field.

DistMult reproducibility verified — at COMPGCN_DECODER=distmult the dispatch
refactor preserves Run 8's epoch-1 AUC byte-identically.
```

- [ ] **Step 5: Update memory index**

In `~/.claude/projects/C--Users-Franz-Samilo-Desktop-the-remembrance/memory/MEMORY.md`, append a line under the existing tuning sessions:

```markdown
- [GNN tuning May 3 (Run 9)](project_tuning_session_may3_run9.md) — RotatE decoder swap. <one-line outcome summary>
```

(Memory updates are not git-committed — the memory directory is outside the repo.)

- [ ] **Step 6: Final smoke verification**

Run: `cd backend && python -m pytest 2>&1 | tail -5`
Expected: 22 + 12 = 34 tests; 32 passing + 2 pre-existing api failures (unrelated).

If anything fails: STOP. Investigate before declaring Run 9 complete.

---

## Self-Review (Plan-vs-Spec Coverage)

**Spec coverage:**
- §1 Algorithm: ✅ Tasks 2 + 3 (math reference + model dispatch)
- §2 Components §1 (Config flag): ✅ Task 1
- §2 Components §2 (Model class refactor): ✅ Task 3
- §2 Components §3 (run_audit construction): ✅ Task 4 Step 3
- §2 Components §4 (Checkpoint meta + AuditRun): ✅ Task 4 Steps 4-6
- §2 Components §5 (Launcher): ✅ Task 5
- §2 Components §6 (Threshold recalibration): ✅ Task 7 Step 6
- §2 Components §7 (TUNING_LOG.md Run 9 section): ✅ Task 8
- Testing requirements (8 unit tests): ✅ Tasks 1-4 (12 tests total — 4 more than spec required, covering math properties)
- Reproducibility check (DistMult byte-identity): ✅ Task 6
- Full audit run: ✅ Task 7

**Placeholder scan:** `<FILL>` and `<DATE>` markers appear only in the runtime templates (Task 8 TUNING_LOG section, Task 9 inventory updates, Task 10 memory entry) and are explicitly verified-removed in Task 8 Step 4 before commit. No placeholders in the executable plan steps.

**Type/name consistency:** `Config.COMPGCN_DECODER` (string), `model.decoder` (string), `model.rel_phase` (Embedding), `meta["decoder"]` (string), AuditRun `run.decoder` (string), Cypher kwarg `decoder=...` (string). Consistent throughout.

**Acceptance criteria for the whole plan:**
- All 12 unit tests pass
- DistMult reproducibility holds (epoch-1 AUC ±0.0001 vs Run 8)
- Full backend test suite has no new failures (only the 2 pre-existing api ones)
- Run 9 audit completes, AUC guardrail held or correctly tripped
- Post-eval chain completes; canonical τ identified
- TUNING_LOG.md Run 9 section written with all numbers filled
- PAPER_TECHNICAL_INVENTORY.md updated with Run 9 row + commits
- Memory entry written
