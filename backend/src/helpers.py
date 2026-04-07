"""Shared utilities for the Knowledge Graph Framework."""

import re
from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


_SAFE_LABEL_RE = re.compile(r"^[A-Za-z_]\w*$")


def validate_neo4j_label(label: str, name: str) -> None:
    """Raise ValueError if label contains unsafe characters for Cypher interpolation."""
    if not _SAFE_LABEL_RE.match(label):
        raise ValueError(
            f"Config {name}={label!r} is not a safe Neo4j label. "
            f"Must match ^[A-Za-z_]\\w*$"
        )
