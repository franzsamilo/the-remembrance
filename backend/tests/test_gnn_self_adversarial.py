"""Self-adversarial negative-weighting tests."""
from __future__ import annotations

import os
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

import torch
import torch.nn.functional as F


def test_config_exposes_adv_temp_with_zero_default():
    """COMPGCN_ADV_TEMP must default to 0.0 so Runs 6/7 reproduce.

    Verified in a subprocess so reloading the config module does not
    replace the in-process Config class — that would break monkeypatching
    for downstream tests in this file (see _stub_audit_data tests below).
    """
    from src.config import Config
    assert hasattr(Config, "COMPGCN_ADV_TEMP"), \
        "Config must expose COMPGCN_ADV_TEMP"

    import subprocess
    import sys
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if k != "COMPGCN_ADV_TEMP"}
    env.setdefault("NEO4J_URI", "bolt://localhost:7687")
    env.setdefault("NEO4J_PASSWORD", "test")
    env.setdefault("GOOGLE_API_KEY", "test-key")
    result = subprocess.run(
        [sys.executable, "-c",
         "from src.config import Config; print(repr(Config.COMPGCN_ADV_TEMP))"],
        env=env, capture_output=True, text=True, cwd=backend_dir,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    assert result.stdout.strip() == "0.0", \
        f"Default Config.COMPGCN_ADV_TEMP must be 0.0, got: {result.stdout!r}"


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
        # rep 0: scores per positive (the "easy" neg for each)
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


def test_run_audit_bpr_branch_uses_adv_temp(monkeypatch):
    """run_audit's BPR branch must compute self-adversarial loss when
    Config.COMPGCN_ADV_TEMP > 0. We assert this by spying on F.softmax
    and checking it was invoked with shape (K, num_pos), dim=0."""
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

    monkeypatch.setattr(gnn_module, "_evaluate_auc", lambda *a, **kw: 0.96)
    monkeypatch.setattr(gnn_module, "_evaluate_mrr", lambda *a, **kw: 0.92)
    monkeypatch.setattr(Config, "COMPGCN_EPOCHS", 2, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_PATIENCE", 5, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_LOSS", "bpr", raising=False)
    monkeypatch.setattr(Config, "COMPGCN_ADV_TEMP", 1.0, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_AUC_GUARDRAIL", 0.95, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_NEG_SAMPLING", "uniform", raising=False)

    session = MagicMock()
    session.__enter__ = lambda self_: self_
    session.__exit__ = lambda *a: False
    driver = MagicMock()
    driver.session.return_value = session
    from src.db import DatabaseManager
    monkeypatch.setattr(DatabaseManager, "refresh", staticmethod(lambda: driver))

    softmax_calls = []
    real_softmax = torch.nn.functional.softmax

    def spy_softmax(input, dim=None, *args, **kwargs):
        softmax_calls.append((tuple(input.shape), dim))
        return real_softmax(input, dim=dim, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "softmax", spy_softmax)

    gnn_module.run_audit()

    # At least one softmax call must have a 2-D shape and dim=0 — that's the
    # self-adversarial weight computation. Eval-path softmax (if any) is on
    # different shapes / dims and does not match.
    matching = [c for c in softmax_calls
                if len(c[0]) == 2 and c[1] == 0]
    assert matching, (
        "self-adversarial branch must call softmax(neg_logits.view(K, num_pos), dim=0). "
        f"All softmax calls observed: {softmax_calls}"
    )


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
    """Mock Neo4j driver/session and return the session for assertion."""
    from unittest.mock import MagicMock
    session = MagicMock()
    session.__enter__ = lambda self_: self_
    session.__exit__ = lambda *a: False
    driver = MagicMock()
    driver.session.return_value = session
    from src.db import DatabaseManager
    monkeypatch.setattr(DatabaseManager, "refresh", staticmethod(lambda: driver))
    return session


def test_checkpoint_meta_records_adv_temp(tmp_path, monkeypatch):
    """When training saves a checkpoint, the meta JSON must include adv_temp."""
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
    monkeypatch.setattr(Config, "COMPGCN_ADV_TEMP", 1.0, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_AUC_GUARDRAIL", 0.95, raising=False)
    _patch_neo4j_session(monkeypatch)

    gnn_module.run_audit()

    meta_path = tmp_path / "best.json"
    assert meta_path.exists(), "checkpoint meta JSON must be written"
    meta = json.loads(meta_path.read_text())
    assert "adv_temp" in meta, "checkpoint meta must record adv_temp for attribution"
    assert meta["adv_temp"] == 1.0


def test_audit_run_node_records_adv_temp(monkeypatch):
    """The AuditRun MERGE in run_audit must SET run.adv_temp = $adv_temp."""
    import src.gnn_module as gnn_module
    from src.config import Config

    _, fake_loader = _stub_audit_data()
    import src.gnn_loader as gnn_loader_module
    monkeypatch.setattr(gnn_loader_module, "GNNLoader", lambda: fake_loader, raising=False)

    monkeypatch.setattr(gnn_module, "_evaluate_auc", lambda *a, **kw: 0.97)
    monkeypatch.setattr(gnn_module, "_evaluate_mrr", lambda *a, **kw: 0.93)
    monkeypatch.setattr(Config, "COMPGCN_EPOCHS", 1, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_PATIENCE", 5, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_LOSS", "bpr", raising=False)
    monkeypatch.setattr(Config, "COMPGCN_ADV_TEMP", 1.0, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_AUC_GUARDRAIL", 0.95, raising=False)
    session = _patch_neo4j_session(monkeypatch)

    gnn_module.run_audit()

    audit_run_calls = [c for c in session.run.call_args_list
                       if "AuditRun" in str(c) or "MERGE" in str(c)]
    assert audit_run_calls, "AuditRun MERGE must be issued"
    matching = [c for c in audit_run_calls
                if "adv_temp" in str(c.args[0]) and "adv_temp" in c.kwargs]
    assert matching, \
        "AuditRun Cypher must SET run.adv_temp and pass adv_temp= kwarg"
    for c in matching:
        assert c.kwargs.get("adv_temp") == 1.0


def test_audit_run_records_adv_temp_when_guardrail_trips(monkeypatch):
    """The AuditRun MERGE handles both completion and aborted paths via
    status=$status — adv_temp must be present in the guardrail-aborted
    write too."""
    import src.gnn_module as gnn_module
    from src.config import Config

    _, fake_loader = _stub_audit_data()
    import src.gnn_loader as gnn_loader_module
    monkeypatch.setattr(gnn_loader_module, "GNNLoader", lambda: fake_loader, raising=False)

    # Force AUC below guardrail to trip the abort path.
    monkeypatch.setattr(gnn_module, "_evaluate_auc", lambda *a, **kw: 0.50)
    monkeypatch.setattr(gnn_module, "_evaluate_mrr", lambda *a, **kw: 0.10)
    monkeypatch.setattr(Config, "COMPGCN_EPOCHS", 1, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_PATIENCE", 5, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_LOSS", "bpr", raising=False)
    monkeypatch.setattr(Config, "COMPGCN_ADV_TEMP", 1.0, raising=False)
    monkeypatch.setattr(Config, "COMPGCN_AUC_GUARDRAIL", 0.95, raising=False)
    session = _patch_neo4j_session(monkeypatch)

    gnn_module.run_audit()

    # Plausibility-sync UNWIND must NOT have been issued.
    sync_calls = [c for c in session.run.call_args_list
                  if "plausibility_score" in str(c.args[0])]
    assert len(sync_calls) == 0, "guardrail must block plausibility sync"

    # AuditRun MERGE still issued, with adv_temp.
    merge_calls = [c for c in session.run.call_args_list
                   if "MERGE" in str(c.args[0])]
    assert merge_calls, "AuditRun MERGE must still record the aborted run"
    for c in merge_calls:
        assert "adv_temp" in str(c.args[0]), \
            "guardrail-aborted MERGE must also set run.adv_temp"
        assert c.kwargs.get("adv_temp") == 1.0
