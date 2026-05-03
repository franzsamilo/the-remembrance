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
    # h o e^{i pi} = h * (-1) (cos(pi) = -1, sin(pi) ~ 1.2e-16)
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
