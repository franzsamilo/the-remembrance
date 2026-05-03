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
