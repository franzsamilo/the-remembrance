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
