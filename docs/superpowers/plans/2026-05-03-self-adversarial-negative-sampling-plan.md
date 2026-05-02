# Self-Adversarial Negative Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add self-adversarial negative weighting (RotatE Sun+ 2019) to the BPR loss in CompGCN training, gated by `COMPGCN_ADV_TEMP` env var, default 0.0 for byte-for-byte Run 6 reproduction. Run 8 audit launcher sets α=1.0. Target: lift MRR from 0.886 → ≥ 0.95.

**Architecture:** One-line `if adv_temp > 0` branch added to the existing BPR loss block in `run_audit`. Detached softmax over K=neg_ratio negatives per positive reweights `-logsigmoid(diff)`. Sampling distribution, model architecture, optimizer, scheduler, and eval pipeline all unchanged. Checkpoint meta JSON gains an `adv_temp` field so recovered runs preserve attribution. AuditRun Neo4j node gains an `adv_temp` property for ablation reporting. New launcher `self_adversarial_audit.py` under `backend/run_logs/` executes the audit + post-eval chain end-to-end. After the run, `TUNING_LOG.md` gets a Run 8 section.

**Tech Stack:** Python 3.9 / PyTorch / PyTorch Geometric / Neo4j / pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-03-self-adversarial-negative-sampling-design.md`

---

## File Inventory

| File | Action | Lines (approx) |
|------|--------|----------------|
| `backend/src/config.py` | Modify (add 1 config flag) | +5 |
| `backend/src/gnn_module.py` | Modify (BPR branch, meta JSON, AuditRun Cypher, recovery Cypher) | +20 |
| `backend/run_logs/self_adversarial_audit.py` | Create (launcher) | ~50 |
| `backend/tests/test_gnn_self_adversarial.py` | Create (unit tests) | ~140 |
| `backend/TUNING_LOG.md` | Append Run 8 section after run | ~80 |

---

## Task 1: Add `COMPGCN_ADV_TEMP` Config Flag

**Files:**
- Modify: `backend/src/config.py:69-73`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_gnn_self_adversarial.py` with this initial content:

```python
"""Self-adversarial negative-weighting tests."""
from __future__ import annotations

import os
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

import torch
import torch.nn.functional as F


def test_config_exposes_adv_temp_with_zero_default():
    """COMPGCN_ADV_TEMP must default to 0.0 so Runs 6/7 reproduce."""
    # Ensure no env override leaks from CI shells.
    os.environ.pop("COMPGCN_ADV_TEMP", None)

    # Re-import config inside the test so the env-pop takes effect for the
    # value read at module load. We reload the module rather than relying
    # on a fresh interpreter to avoid Python import caching surprises.
    import importlib
    import src.config as config_module
    importlib.reload(config_module)

    assert hasattr(config_module.Config, "COMPGCN_ADV_TEMP"), \
        "Config must expose COMPGCN_ADV_TEMP"
    assert config_module.Config.COMPGCN_ADV_TEMP == 0.0, \
        "Default α must be 0.0 (uniform-mean BPR fallback)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_gnn_self_adversarial.py::test_config_exposes_adv_temp_with_zero_default -v`
Expected: FAIL with `AttributeError: type object 'Config' has no attribute 'COMPGCN_ADV_TEMP'`

- [ ] **Step 3: Add the config flag**

In `backend/src/config.py`, immediately after the `COMPGCN_AUC_GUARDRAIL` line (current line 73), insert:

```python
    # Self-adversarial negative weighting (RotatE Sun+ 2019, eq. 5). For each
    # positive, weight its K=neg_ratio negatives by softmax(α * neg_score).
    # α=0 disables (uniform-mean BPR — Run 6/7 reproduction). α=1.0 is the
    # RotatE canonical value; Run 8 uses 1.0.
    COMPGCN_ADV_TEMP = float(os.getenv("COMPGCN_ADV_TEMP", 0.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_gnn_self_adversarial.py::test_config_exposes_adv_temp_with_zero_default -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/config.py backend/tests/test_gnn_self_adversarial.py
git commit -m "feat(gnn): add COMPGCN_ADV_TEMP config flag (default 0.0)"
```

---

## Task 2: Pure-Math Test for Self-Adversarial Weight Shape

This test is independent of the model — exercises the softmax+detach math directly so we can lock the formula before touching `run_audit`.

**Files:**
- Modify: `backend/tests/test_gnn_self_adversarial.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_gnn_self_adversarial.py`:

```python
def _self_adversarial_loss(pos_logits, neg_logits, neg_ratio, adv_temp, margin=0.0):
    """Reference implementation of the formula. Tasks 3+ assert the
    in-codebase implementation produces identical values to this."""
    pos_expanded = pos_logits.repeat(neg_ratio)
    diff = pos_expanded - neg_logits - margin
    if adv_temp > 0.0:
        num_pos = pos_logits.size(0)
        neg_reshaped = neg_logits.view(neg_ratio, num_pos)
        weights = F.softmax(adv_temp * neg_reshaped, dim=0).detach()
        diff_reshaped = diff.view(neg_ratio, num_pos)
        return -(weights * F.logsigmoid(diff_reshaped)).sum(dim=0).mean()
    return -F.logsigmoid(diff).mean()


def test_adv_temp_zero_matches_uniform_mean_bpr():
    """α=0 path must equal the existing -logsigmoid(diff).mean() formula."""
    torch.manual_seed(0)
    pos = torch.randn(8)            # 8 positives
    neg = torch.randn(8 * 5)        # K=5 negatives each → 40 negs
    expected = -F.logsigmoid(pos.repeat(5) - neg).mean()
    actual = _self_adversarial_loss(pos, neg, neg_ratio=5, adv_temp=0.0)
    assert torch.allclose(expected, actual, atol=1e-7)


def test_adv_temp_positive_concentrates_weight_on_hard_negatives():
    """At α=1.0, the highest-scoring negative for each positive must carry
    > 1/K of the softmax weight; the lowest must carry < 1/K."""
    K = 4
    num_pos = 3
    # Hand-crafted neg logits: per positive, deliberate score spread.
    neg_logits = torch.tensor([
        # rep 0: scores per positive
        -2.0, -1.0,  0.0,
        # rep 1
         0.0,  0.0,  0.0,
        # rep 2
         1.0,  1.0,  1.0,
        # rep 3 (the "hard" negative for each positive)
         3.0,  2.5,  2.0,
    ])
    weights = F.softmax(neg_logits.view(K, num_pos), dim=0)
    uniform = 1.0 / K
    # The hardest neg (rep 3) carries the most weight for every positive.
    assert (weights[3] > uniform).all(), "hard negatives must dominate softmax"
    # The easiest neg (rep 0) carries the least.
    assert (weights[0] < uniform).all(), "easy negatives must be down-weighted"


def test_adv_temp_weights_have_no_gradient():
    """RotatE eq. 5: softmax weights must be detached. If gradient flowed
    through the weights themselves, the model could trivially game the loss
    by suppressing all negative scores."""
    pos = torch.randn(4, requires_grad=True)
    neg = torch.randn(4 * 3, requires_grad=True)
    loss = _self_adversarial_loss(pos, neg, neg_ratio=3, adv_temp=1.0)
    loss.backward()
    # Both pos and neg get gradient (through logsigmoid), but the softmax
    # weights themselves carry no gradient. Verify by re-running with the
    # weights NOT detached and asserting the gradients differ.
    pos2 = pos.detach().clone().requires_grad_(True)
    neg2 = neg.detach().clone().requires_grad_(True)
    pos_expanded = pos2.repeat(3)
    diff = pos_expanded - neg2
    weights_no_detach = F.softmax(neg2.view(3, 4), dim=0)  # NOT detached
    diff_reshaped = diff.view(3, 4)
    loss2 = -(weights_no_detach * F.logsigmoid(diff_reshaped)).sum(dim=0).mean()
    loss2.backward()
    # Detached and non-detached should give different gradients on neg.
    assert not torch.allclose(neg.grad, neg2.grad, atol=1e-6), \
        "with-detach and without-detach paths must produce different gradients"


def test_adv_temp_neg_ratio_one_equals_logsigmoid_diff():
    """K=1: softmax over a single value is 1.0, so the loss reduces to
    plain -logsigmoid(diff)."""
    pos = torch.tensor([0.5, -0.3, 1.2])
    neg = torch.tensor([0.1, 0.4, -0.2])  # K=1 → length matches pos
    expected = -F.logsigmoid(pos - neg).mean()
    actual = _self_adversarial_loss(pos, neg, neg_ratio=1, adv_temp=1.0)
    assert torch.allclose(expected, actual, atol=1e-7)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_gnn_self_adversarial.py -v -k "adv_temp"`
Expected: 4 tests PASS (zero_matches, positive_concentrates, weights_have_no_gradient, neg_ratio_one).

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_gnn_self_adversarial.py
git commit -m "test(gnn): self-adversarial weight math (zero-default, hard-neg concentration, detach, K=1)"
```

---

## Task 3: Wire Self-Adversarial Loss into `run_audit`

**Files:**
- Modify: `backend/src/gnn_module.py:443-448` (BPR loss branch)
- Modify: `backend/tests/test_gnn_self_adversarial.py` (add wiring test)

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_gnn_self_adversarial.py`:

```python
def test_run_audit_bpr_branch_uses_adv_temp(monkeypatch):
    """run_audit's BPR branch must compute self-adversarial loss when
    Config.COMPGCN_ADV_TEMP > 0. We assert this by monkeypatching the
    F.softmax call inside gnn_module and checking it was invoked with the
    expected α coefficient."""
    from unittest.mock import MagicMock
    import src.gnn_module as gnn_module
    from src.config import Config

    # Trivial graph
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
    import src.gnn_loader as gnn_loader_module
    monkeypatch.setattr(gnn_loader_module, "GNNLoader", lambda: fake_loader, raising=False)

    monkeypatch.setattr(gnn_module, "_evaluate_auc", lambda *a, **kw: 0.96)
    monkeypatch.setattr(gnn_module, "_evaluate_mrr", lambda *a, **kw: 0.92)
    monkeypatch.setattr(Config, "COMPGCN_EPOCHS", 2, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_PATIENCE", 5, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_LOSS", "bpr", raising=False)
    monkeypatch.setattr(Config, "COMPGCN_ADV_TEMP", 1.0, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_AUC_GUARDRAIL", 0.95, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_NEG_SAMPLING", "uniform", raising=False)

    # Mock Neo4j driver so sync calls don't hit a real DB.
    session = MagicMock()
    session.__enter__ = lambda self_: self_
    session.__exit__ = lambda *a: False
    driver = MagicMock()
    driver.session.return_value = session
    from src.db import DatabaseManager
    monkeypatch.setattr(DatabaseManager, "refresh", staticmethod(lambda: driver))

    # Spy on F.softmax to verify it's invoked with the adv-temp scale.
    softmax_calls = []
    real_softmax = torch.nn.functional.softmax
    def spy_softmax(input, dim=None, *args, **kwargs):
        softmax_calls.append((input.shape, dim, float(input.abs().max())))
        return real_softmax(input, dim=dim, *args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "softmax", spy_softmax)

    gnn_module.run_audit()

    # At least one softmax call must have shape (neg_ratio, num_pos) and dim=0.
    # neg_ratio is Config.COMPGCN_NEG_RATIO; num_pos is roughly len(train_idx).
    matching = [c for c in softmax_calls
                if len(c[0]) == 2 and c[1] == 0]
    assert matching, \
        "self-adversarial branch must call softmax(neg_logits.view(K, num_pos), dim=0)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_gnn_self_adversarial.py::test_run_audit_bpr_branch_uses_adv_temp -v`
Expected: FAIL — softmax never called with `dim=0` because the BPR branch still uses uniform mean.

- [ ] **Step 3: Modify the BPR branch**

In `backend/src/gnn_module.py`, replace lines 443-448:

```python
        if loss_mode == "bpr":
            # Pairwise: neg_edges = neg_ratio repetitions of pos ordering
            # (see _sample_negative_edges: list-cat preserves [pos_0..pos_N] grouped by rep)
            pos_expanded = pos_logits.repeat(neg_ratio)
            diff = pos_expanded - neg_logits - bpr_margin
            loss = -F.logsigmoid(diff).mean()
```

with:

```python
        if loss_mode == "bpr":
            # Pairwise: neg_edges = neg_ratio repetitions of pos ordering
            # (see _sample_negative_edges: list-cat preserves [pos_0..pos_N] grouped by rep)
            pos_expanded = pos_logits.repeat(neg_ratio)
            diff = pos_expanded - neg_logits - bpr_margin
            adv_temp = Config.COMPGCN_ADV_TEMP
            if adv_temp > 0.0:
                # Self-adversarial weighting (RotatE Sun+ 2019, eq. 5):
                # weight each negative by softmax over its K siblings. Hard
                # negatives (high score) dominate gradient. Detach so the
                # weights themselves carry no gradient.
                num_pos = pos_logits.size(0)
                neg_reshaped = neg_logits.view(neg_ratio, num_pos)
                weights = F.softmax(adv_temp * neg_reshaped, dim=0).detach()
                diff_reshaped = diff.view(neg_ratio, num_pos)
                loss = -(weights * F.logsigmoid(diff_reshaped)).sum(dim=0).mean()
            else:
                loss = -F.logsigmoid(diff).mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_gnn_self_adversarial.py::test_run_audit_bpr_branch_uses_adv_temp -v`
Expected: PASS

- [ ] **Step 5: Run full self-adversarial test suite**

Run: `cd backend && pytest tests/test_gnn_self_adversarial.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 6: Run existing GNN tests to confirm no regression**

Run: `cd backend && pytest tests/test_gnn_neg_sampling.py tests/test_gnn_loader_labels.py -v`
Expected: All existing tests still PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/src/gnn_module.py backend/tests/test_gnn_self_adversarial.py
git commit -m "feat(gnn): self-adversarial weighting in BPR loss (RotatE Sun+ 2019)"
```

---

## Task 4: Persist `adv_temp` in Checkpoint Meta + AuditRun Neo4j Property

**Files:**
- Modify: `backend/src/gnn_module.py:485-495` (checkpoint meta JSON write inside training loop)
- Modify: `backend/src/gnn_module.py` (run_audit's AuditRun Cypher write — find via grep)
- Modify: `backend/src/gnn_module.py` (recover_from_checkpoint's AuditRun Cypher write)
- Modify: `backend/tests/test_gnn_self_adversarial.py`

- [ ] **Step 1: Locate the AuditRun writes**

Run: `cd backend && grep -n "AUDIT_RUN_LABEL\|AuditRun" src/gnn_module.py`
Expected: two MERGE blocks in `run_audit` and `recover_from_checkpoint` plus optional one for the aborted-guardrail path.

Note the line ranges. They include `SET run.... = $...` lines. The new property should be added inside both SET blocks.

- [ ] **Step 2: Write the failing test**

Append to `backend/tests/test_gnn_self_adversarial.py`:

```python
def test_checkpoint_meta_records_adv_temp(tmp_path, monkeypatch):
    """When training saves a checkpoint, the meta JSON must include adv_temp."""
    import json
    from unittest.mock import MagicMock
    import src.gnn_module as gnn_module
    from src.config import Config

    # Redirect checkpoint paths to tmp.
    monkeypatch.setattr(gnn_module, "CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr(gnn_module, "CHECKPOINT_PATH", str(tmp_path / "best.pt"))
    monkeypatch.setattr(gnn_module, "CHECKPOINT_META_PATH", str(tmp_path / "best.json"))

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
    import src.gnn_loader as gnn_loader_module
    monkeypatch.setattr(gnn_loader_module, "GNNLoader", lambda: fake_loader, raising=False)

    monkeypatch.setattr(gnn_module, "_evaluate_auc", lambda *a, **kw: 0.97)
    monkeypatch.setattr(gnn_module, "_evaluate_mrr", lambda *a, **kw: 0.93)
    monkeypatch.setattr(Config, "COMPGCN_EPOCHS", 2, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_PATIENCE", 5, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_LOSS", "bpr", raising=False)
    monkeypatch.setattr(Config, "COMPGCN_ADV_TEMP", 1.0, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_AUC_GUARDRAIL", 0.95, raising=False)

    session = MagicMock()
    session.__enter__ = lambda self_: self_
    session.__exit__ = lambda *a: False
    driver = MagicMock()
    driver.session.return_value = session
    from src.db import DatabaseManager
    monkeypatch.setattr(DatabaseManager, "refresh", staticmethod(lambda: driver))

    gnn_module.run_audit()

    meta_path = tmp_path / "best.json"
    assert meta_path.exists(), "checkpoint meta JSON must be written"
    meta = json.loads(meta_path.read_text())
    assert "adv_temp" in meta, "checkpoint meta must record adv_temp for attribution"
    assert meta["adv_temp"] == 1.0


def test_audit_run_node_records_adv_temp(monkeypatch):
    """The AuditRun MERGE in run_audit must SET run.adv_temp = $adv_temp."""
    from unittest.mock import MagicMock
    import src.gnn_module as gnn_module
    from src.config import Config

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
    import src.gnn_loader as gnn_loader_module
    monkeypatch.setattr(gnn_loader_module, "GNNLoader", lambda: fake_loader, raising=False)

    monkeypatch.setattr(gnn_module, "_evaluate_auc", lambda *a, **kw: 0.97)
    monkeypatch.setattr(gnn_module, "_evaluate_mrr", lambda *a, **kw: 0.93)
    monkeypatch.setattr(Config, "COMPGCN_EPOCHS", 1, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_PATIENCE", 5, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_LOSS", "bpr", raising=False)
    monkeypatch.setattr(Config, "COMPGCN_ADV_TEMP", 1.0, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_AUC_GUARDRAIL", 0.95, raising=False)

    session = MagicMock()
    session.__enter__ = lambda self_: self_
    session.__exit__ = lambda *a: False
    driver = MagicMock()
    driver.session.return_value = session
    from src.db import DatabaseManager
    monkeypatch.setattr(DatabaseManager, "refresh", staticmethod(lambda: driver))

    gnn_module.run_audit()

    audit_run_calls = [c for c in session.run.call_args_list
                       if "AuditRun" in str(c) or "MERGE" in str(c)]
    assert audit_run_calls, "AuditRun MERGE must be issued"
    # The Cypher and the kwargs both have to mention adv_temp.
    matching = [c for c in audit_run_calls
                if "adv_temp" in str(c.args[0]) and "adv_temp" in c.kwargs]
    assert matching, \
        "AuditRun Cypher must SET run.adv_temp and pass adv_temp= kwarg"
    # Value passed must equal the configured α.
    for c in matching:
        assert c.kwargs.get("adv_temp") == 1.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_gnn_self_adversarial.py -v -k "checkpoint_meta or audit_run_node"`
Expected: 2 FAILS.

- [ ] **Step 4: Add `adv_temp` to checkpoint meta JSON**

In `backend/src/gnn_module.py`, locate the `json.dump({...})` block inside `run_audit` (the one inside `try:` after `torch.save(best_state, CHECKPOINT_PATH)`). Add `"adv_temp": float(Config.COMPGCN_ADV_TEMP),` to the dict — alongside `"loss_mode": loss_mode,`. The block becomes:

```python
                with open(CHECKPOINT_META_PATH, "w") as f:
                    json.dump({
                        "best_epoch": best_epoch_idx,
                        "best_auc": float(best_auc),
                        "train_loss": final_train_loss,
                        "hidden_channels": Config.COMPGCN_HIDDEN_CHANNELS,
                        "loss_mode": loss_mode,
                        "adv_temp": float(Config.COMPGCN_ADV_TEMP),
                        "num_relations": num_rels,
                        "in_channels": int(data.x.size(1)),
                        "saved_at": _utc_now_iso(),
                    }, f)
```

- [ ] **Step 5: Add `adv_temp` to `run_audit`'s AuditRun MERGE**

Find the `run_audit` MERGE that writes the completed `AuditRun` (search for `MERGE (run:` then look for the `SET run.` block). Add `run.adv_temp = $adv_temp` to the SET clause and pass `adv_temp=Config.COMPGCN_ADV_TEMP` in the kwargs.

Locate the run that the test exercises — there are two MERGE writes for AuditRun in `run_audit` (one for the regular completion path, one for the guardrail-aborted path). Update both.

Example (for the completion-path MERGE):

```python
            session.run(
                f"""
                MERGE (run:{Config.AUDIT_RUN_LABEL} {{run_id: $run_id}})
                SET run.status = 'completed',
                    run.audit_mode = 'compgcn',
                    run.audit_model = 'CompGCNAuditModel',
                    run.started_at = $started_at,
                    run.completed_at = $completed_at,
                    run.best_epoch = $best_epoch,
                    run.auc_roc = $auc_roc,
                    run.mrr = $mrr,
                    run.mrr_uniform = $mrr_uniform,
                    run.mrr_type_aware = $mrr_type_aware,
                    run.loss = $loss,
                    run.adv_temp = $adv_temp
                """,
                run_id=audit_run_id,
                started_at=audit_started_at,
                completed_at=audit_completed_at,
                best_epoch=best_epoch_idx,
                auc_roc=final_auc,
                mrr=final_mrr,
                mrr_uniform=mrr_uniform,
                mrr_type_aware=mrr_type_aware,
                loss=loss_mode,
                adv_temp=float(Config.COMPGCN_ADV_TEMP),
            )
```

Apply the same `run.adv_temp = $adv_temp` and kwarg to the guardrail-aborted MERGE block (status `'aborted_auc_guardrail'`).

- [ ] **Step 6: Add `adv_temp` to `recover_from_checkpoint`'s AuditRun MERGE**

In `recover_from_checkpoint` (around line 823 onward), locate the MERGE block. Update to:

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
                run.adv_temp = $adv_temp
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
        )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_gnn_self_adversarial.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 8: Run full backend test suite**

Run: `cd backend && pytest -v`
Expected: All tests PASS (no regression in `test_api.py`, `test_utils.py`, `test_gnn_neg_sampling.py`, `test_gnn_loader_labels.py`).

- [ ] **Step 9: Commit**

```bash
git add backend/src/gnn_module.py backend/tests/test_gnn_self_adversarial.py
git commit -m "feat(gnn): persist adv_temp in checkpoint meta and AuditRun node"
```

---

## Task 5: Run 8 Launcher Script

**Files:**
- Create: `backend/run_logs/self_adversarial_audit.py`

- [ ] **Step 1: Write the launcher**

Create `backend/run_logs/self_adversarial_audit.py`:

```python
"""Run 8: BPR + self-adversarial negative weighting (α=1.0).

Sets COMPGCN_LOSS=bpr, COMPGCN_NEG_SAMPLING=uniform, COMPGCN_ADV_TEMP=1.0 and
runs the full audit + Neo4j sync. Targets MRR > 0.95 by reweighting BPR
gradients toward hard negatives (RotatE Sun+ 2019, eq. 5).

Rollback to Run 6 (BPR uniform-mean): re-run backend/run_logs/bpr_audit.py.
"""
from __future__ import annotations

import os
import sys
import time

# Running from backend/run_logs/; backend/ must be on sys.path for `src` imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must set before importing src.config (load_dotenv runs at import-time, but
# os.environ[...] takes precedence over .env when load_dotenv uses override=False)
os.environ["COMPGCN_LOSS"] = "bpr"
os.environ["COMPGCN_BPR_MARGIN"] = os.environ.get("COMPGCN_BPR_MARGIN", "0.0")
os.environ["COMPGCN_NEG_SAMPLING"] = os.environ.get("COMPGCN_NEG_SAMPLING", "uniform")
os.environ["COMPGCN_ADV_TEMP"] = os.environ.get("COMPGCN_ADV_TEMP", "1.0")
os.environ["COMPGCN_AUC_GUARDRAIL"] = os.environ.get("COMPGCN_AUC_GUARDRAIL", "0.95")


def main():
    from src.gnn_module import run_audit, get_training_history
    from src.evaluation import persist_gnn_metrics

    t0 = time.time()
    print("SELF_ADV_AUDIT_BEGIN", flush=True)
    print(
        f"  loss=bpr neg_sampling={os.environ['COMPGCN_NEG_SAMPLING']} "
        f"adv_temp={os.environ['COMPGCN_ADV_TEMP']} "
        f"auc_guardrail={os.environ['COMPGCN_AUC_GUARDRAIL']}",
        flush=True,
    )
    try:
        run_audit()
        th = get_training_history()
        persist_gnn_metrics(th)
        print(
            f"SELF_ADV_AUDIT_DONE elapsed_min={(time.time()-t0)/60.0:.2f} "
            f"best_epoch={th.get('best_epoch')} "
            f"auc={th.get('final_auc_roc')} "
            f"mrr_uniform={th.get('mrr_uniform')} "
            f"mrr_type_aware={th.get('mrr_type_aware')}",
            flush=True,
        )
    except Exception as e:
        import traceback
        print(f"SELF_ADV_AUDIT_FAILED {type(e).__name__} {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check the launcher (no execution yet)**

Run: `cd backend && python -c "import ast; ast.parse(open('run_logs/self_adversarial_audit.py').read())"`
Expected: No output (silent = parse OK). The actual training run is gated under Task 7.

- [ ] **Step 3: Commit**

```bash
git add backend/run_logs/self_adversarial_audit.py
git commit -m "chore(run_logs): add self_adversarial_audit.py launcher (Run 8)"
```

---

## Task 6: Reproducibility Sanity Check (α=0 vs Run 6 First-Epoch AUC)

Before committing the full Run 8 audit, verify that `COMPGCN_ADV_TEMP=0.0` produces a first-epoch AUC matching Run 6 (which we know from `TUNING_LOG.md` Run 6: epoch 1 AUC was emitted in the per-epoch log; we need the same number with the new code).

**Files:**
- No new files. This is a verification step.

- [ ] **Step 1: Run a 2-epoch dry run with α=0**

Run:
```bash
cd backend && COMPGCN_LOSS=bpr COMPGCN_ADV_TEMP=0.0 COMPGCN_EPOCHS=2 COMPGCN_NEG_SAMPLING=uniform python -c "from src.gnn_module import run_audit; run_audit()" 2>&1 | tee run_logs/repro_check_alpha_zero.log
```

Expected: training completes 2 epochs. The log shows `CompGCN epoch 1/2 loss=... auc_roc=0.XXXX`. Record this AUC.

- [ ] **Step 2: Compare against Run 6 epoch-1 AUC**

The `TUNING_LOG.md` Run 6 section reports BPR per-epoch AUC values implicitly via "best epoch 168" — but the deterministic baseline is from Run 5 (BCE) where epoch 1 AUC = 0.4508. For BPR (Run 6) the comparable value is the first epoch from the prior `audit_bpr.log`. Open `backend/run_logs/audit_bpr.log` and find the `epoch 1/300` line.

Run: `grep "epoch 1/300\|epoch 1/2" backend/run_logs/audit_bpr.log backend/run_logs/repro_check_alpha_zero.log`
Expected: The α=0 dry run's epoch-1 AUC matches `audit_bpr.log`'s epoch-1 AUC within ±0.0001.

If the values match: reproducibility is confirmed; proceed to Task 7.
If the values differ by more than ±0.0001: STOP. Either the BPR branch has been semantically altered (bug in Task 3) or the RNG order has shifted. Investigate before continuing.

- [ ] **Step 3: Commit the verification log**

```bash
git add backend/run_logs/repro_check_alpha_zero.log
git commit -m "chore(run_logs): α=0 reproducibility check vs Run 6 (epoch-1 AUC match)"
```

---

## Task 7: Execute Run 8 (Full Self-Adversarial Audit)

**Files:**
- Generates: `backend/run_logs/audit_self_adversarial.log` (runtime output)
- Generates: `backend/run_logs/eval_chain_self_adversarial.log` (post-eval output)
- Modifies: `backend/run_logs/compgcn_best.pt` and `compgcn_best_meta.json` (best checkpoint)
- Modifies: Neo4j (writes new plausibility scores; `AuditRun` node)

- [ ] **Step 1: Verify Neo4j is reachable**

Run: `cd backend && python -c "from src.db import DatabaseManager; d = DatabaseManager.refresh(); print(d.verify_connectivity())"`
Expected: No exception. If you get an `AuthError` or `ServiceUnavailable`, fix `.env` Neo4j credentials before continuing.

- [ ] **Step 2: Run the audit**

Run:
```bash
cd backend && python run_logs/self_adversarial_audit.py 2>&1 | tee run_logs/audit_self_adversarial.log
```

Expected runtime: ~2–3 minutes (similar to Run 6's 1.8 min). Final line should be `SELF_ADV_AUDIT_DONE` with `auc=0.XX mrr_uniform=0.XX mrr_type_aware=0.XX`.

If `SELF_ADV_AUDIT_FAILED` appears: capture the full traceback in the log, do NOT proceed to Step 3, and treat as a debugging task.

- [ ] **Step 3: Verify the AUC guardrail did not trip**

Run: `grep -E "guardrail|aborted" backend/run_logs/audit_self_adversarial.log`
Expected: No matches (guardrail did not fire). If `final_auc < 0.95` triggered the guardrail, the run is aborted and Neo4j scores were not synced — investigate why AUC dropped before continuing.

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
Expected: `n=6419` (matches Run 6's edge count). Score distribution: avg should be in [0.5, 1.0] for BPR-style spread. If avg < 0.4 the score range collapsed — investigate.

- [ ] **Step 5: Run the post-eval chain**

Run:
```bash
cd backend && python run_logs/post_audit_eval.py 2>&1 | tee run_logs/eval_chain_self_adversarial.log
```

Expected: full chain runs (Neo4j verify → full-stack G/F at τ=0.95 → threshold sweep → prompt-only ablation). Final output includes G/F at τ=0.95.

- [ ] **Step 6: Record headline numbers**

Open `backend/run_logs/audit_self_adversarial.log` and `backend/run_logs/eval_chain_self_adversarial.log`. Record on a scratch note:

- Best epoch (training loop)
- Final AUC-ROC (validation)
- MRR uniform eval
- MRR type-aware eval (if computed)
- Grounding @ τ=0.95
- Faithfulness @ τ=0.95
- Per-query G/F (5 queries)
- Threshold sweep table (τ=0.30/0.50/0.85/0.95)
- Wall-clock training time

These feed Task 8.

- [ ] **Step 7: Do NOT commit yet**

The logs and checkpoint changes are not committed in this task. Task 9 commits them along with the `TUNING_LOG.md` update so a single commit captures the run + audit narrative.

---

## Task 8: Append Run 8 Section to `TUNING_LOG.md`

**Files:**
- Modify: `backend/TUNING_LOG.md` (append after the Run 7 section, before "## Overall Scoreboard")

- [ ] **Step 1: Locate the insertion point**

Run: `grep -n "## Overall Scoreboard" backend/TUNING_LOG.md`
Expected: Single match. The Run 8 section is inserted on the line before this.

- [ ] **Step 2: Write the Run 8 section**

Edit `backend/TUNING_LOG.md` and insert the following block immediately before the `## Overall Scoreboard (as of …)` heading. Replace `<DATE>` with today's date and the bracketed `<…>` placeholders with the recorded numbers from Task 7 Step 6:

```markdown
## Run 8 — <DATE>: BPR + Self-Adversarial Negative Weighting (MRR Closure Attempt)

**Goal.** Close the last open paper KPI. Run 6 (BPR uniform) hit AUC 0.9688 / MRR 0.886; Run 7 (BPR + type-aware) showed type-defined hardness does not lift MRR on this corpus (54% Concept dominates). This run swaps in score-defined hardness (RotatE Sun+ 2019, eq. 5): for each positive, weight its K negatives by softmax(α · neg_score). Hard negatives dominate gradient; easy negatives near zero. Sampling distribution (uniform), loss family (BPR), architecture (3-layer CompGCN + LayerNorm), all hyperparameters except α stay identical to Run 6.

**Config delta vs Run 6.**

| Parameter | Run 6 | Run 8 |
|-----------|-------|-------|
| `COMPGCN_LOSS` | bpr | bpr |
| `COMPGCN_NEG_SAMPLING` | uniform | uniform |
| `COMPGCN_ADV_TEMP` | 0.0 (implicit) | **1.0** |
| All other hyperparams | identical | identical |

**Loss formula.**

$$\mathcal{L} = -\frac{1}{|B|} \sum_{(h,r,t)} \sum_{k=1}^{K} w_k \cdot \log \sigma(s(h,r,t) - s(h,r,t'_k) - \gamma)$$

with $w_k = \text{softmax}(\alpha \cdot s(h,r,t'_k))$ over $k \in [1,K]$, detached.

**GNN Metrics.**

| Metric | Run 6 (BPR uniform) | Run 7 (BPR + type-aware) | Run 8 (BPR + self-adv α=1.0) | Δ vs Run 6 | Target | Status |
|--------|---------------------|---------------------------|------------------------------|------------|--------|--------|
| AUC-ROC | 0.9688 | 0.9662 | <FILL> | <FILL> | > 0.95 | <FILL> |
| MRR (uniform eval) | 0.8860 | 0.8873 | <FILL> | <FILL> | > 0.95 | <FILL> |
| MRR (type-aware eval) | — | 0.8755 | <FILL> | <FILL> | > 0.95 | <FILL> |
| Grounding @ τ=0.95 | 0.987 | 0.988 | <FILL> | <FILL> | > 0.98 | <FILL> |
| Faithfulness @ τ=0.95 | 0.979 | 0.91–0.95 | <FILL> | <FILL> | high | <FILL> |
| Best epoch | 168 | 163 | <FILL> | — | — | — |
| Training wall-clock | 1.8 min | 4.04 min | <FILL> | — | — | — |

**Score distribution (BPR-trained plausibility, all 6,419 edges).**

| Bucket | Run 6 | Run 8 |
|--------|-------|-------|
| < 0.50 | 19 (0.3%) | <FILL> |
| 0.50 – 0.85 | 59 (0.9%) | <FILL> |
| 0.85 – 0.95 | 181 (2.8%) | <FILL> |
| 0.95 – 0.99 | 700 (10.9%) | <FILL> |
| ≥ 0.99 | 5460 (85.1%) | <FILL> |
| max / avg / min | 1.000 / 0.9895 / 0.0421 | <FILL> |

**Threshold sweep.**

| τ | Grounding | Faithfulness |
|---|-----------|--------------|
| 0.30 | <FILL> | <FILL> |
| 0.50 | <FILL> | <FILL> |
| 0.85 | <FILL> | <FILL> |
| **0.95** | **<FILL>** | **<FILL>** |

**Per-query (τ=0.95).**

| Query | Grounding | Faithfulness |
|-------|-----------|--------------|
| What are the key findings? | <FILL> | <FILL> |
| Who are the main researchers? | <FILL> | <FILL> |
| What methods were used? | <FILL> | <FILL> |
| What are the main results? | <FILL> | <FILL> |
| What datasets or concepts are discussed? | <FILL> | <FILL> |

**Interpretation.**

<FILL — narrate whichever outcome was observed:
- If MRR ≥ 0.95: "H2 paper target achieved. Self-adversarial weighting is the closing intervention."
- If 0.92 ≤ MRR < 0.95: "Meaningful lift over Run 6 (+X.X pts). Falls short of canonical 0.95 — corpus density (5.2k nodes / 6.4k edges) appears to cap RotatE-style hard mining; FB15k benchmarks have ~100× more edges per node. Future work: combine with type-aware (Run 8b), or longer training with larger neg_ratio."
- If MRR < 0.92: "Self-adversarial weighting did not lift MRR on this corpus. Score-defined hardness does not produce contrasting examples beyond what uniform sampling already supplies, plausibly because the graph density is too low for hard negatives to differ materially from easy ones. Three of four paper KPIs remain met (AUC, Grounding, Faithfulness). MRR closure deferred to future work; thesis discussion frames the gap as a corpus-property finding."
- Address any AUC drop, faithfulness shift, score-distribution change.>

**Headline.** <FILL — one-line summary of the run outcome.>

**Reproduction.**
```bash
cd backend && python run_logs/self_adversarial_audit.py
cd backend && python run_logs/post_audit_eval.py
```
```

- [ ] **Step 3: Replace all `<FILL>` markers**

Walk through every `<FILL>` in the inserted block and replace with the actual numbers recorded in Task 7 Step 6. The interpretation paragraph picks the appropriate sub-bullet (≥0.95 / 0.92–0.95 / <0.92) based on the observed MRR.

- [ ] **Step 4: Verify no `<FILL>` markers remain**

Run: `grep "<FILL>" backend/TUNING_LOG.md`
Expected: No matches. If matches remain, finish populating before continuing.

- [ ] **Step 5: Update the Overall Scoreboard table**

In the same `TUNING_LOG.md`, find `## Overall Scoreboard (as of 2026-04-19)`. Update:
- Heading date: `## Overall Scoreboard (as of <DATE>)`
- Add a `Run 8 (BPR + self-adv)` column to the existing scoreboard table.
- Update the "Recommended configuration for thesis defense" section: if MRR ≥ 0.95, recommend `COMPGCN_ADV_TEMP=1.0`; otherwise leave Run 6 recommended and add a "Run 8 attempted" footnote.
- Append a "Run 8 finding" bullet to the Key Findings list.

- [ ] **Step 6: Commit**

```bash
git add backend/TUNING_LOG.md backend/run_logs/audit_self_adversarial.log backend/run_logs/eval_chain_self_adversarial.log backend/run_logs/compgcn_best.pt backend/run_logs/compgcn_best_meta.json
git commit -m "docs(tuning): Run 8 — BPR + self-adversarial α=1.0 (RotatE eq. 5); MRR <FILL>"
```

(Replace `<FILL>` in the commit message with the achieved MRR rounded to 4 decimals.)

---

## Task 9: Optional Run 8b — Self-Adversarial + Type-Aware (Combined)

This task is **gated**: only execute if Run 8's MRR landed in [0.92, 0.95). If MRR ≥ 0.95, skip — H2 is closed. If MRR < 0.92, skip — additional intervention unlikely to close such a wide gap, and the cleaner paper story is to write up Run 8's negative finding.

**Files:**
- Generates: `backend/run_logs/audit_self_adversarial_type_aware.log`
- Generates: `backend/run_logs/eval_chain_self_adversarial_type_aware.log`
- Modifies: `backend/TUNING_LOG.md` (append Run 8b sub-section)

- [ ] **Step 1: Confirm gating condition**

If Task 7's recorded MRR (uniform eval) is < 0.92 or ≥ 0.95, **stop here**. Mark this task complete-with-skip in the plan file and proceed to Task 10.

- [ ] **Step 2: Run the combined audit**

Run:
```bash
cd backend && COMPGCN_NEG_SAMPLING=type_aware python run_logs/self_adversarial_audit.py 2>&1 | tee run_logs/audit_self_adversarial_type_aware.log
```

Expected: ~3–5 min runtime (type-aware adds ~2 min per Run 7 baseline). `SELF_ADV_AUDIT_DONE` final line.

- [ ] **Step 3: Run post-eval**

Run:
```bash
cd backend && python run_logs/post_audit_eval.py 2>&1 | tee run_logs/eval_chain_self_adversarial_type_aware.log
```

- [ ] **Step 4: Append Run 8b sub-section to `TUNING_LOG.md`**

Insert immediately after the Run 8 section (before Overall Scoreboard). Mirror the Run 8 structure with config column showing `COMPGCN_NEG_SAMPLING=type_aware`. Same FILL → numbers workflow.

- [ ] **Step 5: Commit**

```bash
git add backend/TUNING_LOG.md backend/run_logs/audit_self_adversarial_type_aware.log backend/run_logs/eval_chain_self_adversarial_type_aware.log backend/run_logs/compgcn_best.pt backend/run_logs/compgcn_best_meta.json
git commit -m "docs(tuning): Run 8b — self-adv + type-aware combined; MRR <FILL>"
```

---

## Task 10: Final Verification

**Files:** None new; verification only.

- [ ] **Step 1: Confirm full test suite passes**

Run: `cd backend && pytest -v`
Expected: All tests PASS, including new `test_gnn_self_adversarial.py` and existing files.

- [ ] **Step 2: Confirm git working tree is clean**

Run: `git status`
Expected: clean tree (or only the pre-existing modifications listed at session start: `backend/src/evaluation.py`, `frontend/app/evidence/page.tsx`, `frontend/app/globals.css`, `frontend/components/DetectiveBoard.tsx`, `.claude/settings.local.json`, deleted docx/md files. None of those are touched by Run 8.).

- [ ] **Step 3: Confirm commits are in correct order**

Run: `git log --oneline -10`
Expected: Recent commits in order:
1. `docs(tuning): Run 8 — BPR + self-adversarial α=1.0 …`
2. `chore(run_logs): α=0 reproducibility check vs Run 6 …`
3. `chore(run_logs): add self_adversarial_audit.py launcher (Run 8)`
4. `feat(gnn): persist adv_temp in checkpoint meta and AuditRun node`
5. `feat(gnn): self-adversarial weighting in BPR loss (RotatE Sun+ 2019)`
6. `test(gnn): self-adversarial weight math …`
7. `feat(gnn): add COMPGCN_ADV_TEMP config flag (default 0.0)`

(Plus optional Run 8b commit at the top if executed.)

- [ ] **Step 4: Confirm `MEMORY.md` update is needed**

Add a new memory entry at `~/.claude/projects/C--Users-Franz-Samilo-Desktop-the-remembrance/memory/project_tuning_session_may3.md` summarizing Run 8 outcome. Update `MEMORY.md` index. (This is a memory-system task, not a code commit — no git commit needed.)
