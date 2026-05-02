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
