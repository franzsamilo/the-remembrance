"""User-run graph wipe.

The assistant's auto-permission classifier blocks programmatic mass-delete on
the live database, so this is run MANUALLY by the user:

    python -m run_logs.wipe_graph

It writes the outcome to run_logs/wipe_result.txt so the assistant can confirm
it even without seeing the terminal output. Reversible: the current graph is
backed up at run_logs/graph_snapshot_20260621_202736.jsonl.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.db import DatabaseManager  # noqa: E402

RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wipe_result.txt")


def main() -> int:
    try:
        d = DatabaseManager.refresh()
        with d.session(database=Config.NEO4J_DATABASE) as s:
            before = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            counters = s.run("MATCH (n) DETACH DELETE n").consume().counters
            after = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        msg = (
            f"WIPED before={before} nodes_deleted={counters.nodes_deleted} "
            f"rels_deleted={counters.relationships_deleted} after={after}"
        )
    except Exception as e:  # noqa: BLE001
        msg = f"ERROR {type(e).__name__}: {e}"
    print(msg)
    with open(RESULT, "w", encoding="utf-8") as f:
        f.write(msg + "\n")
    return 0 if msg.startswith("WIPED") and msg.endswith("after=0") else 1


if __name__ == "__main__":
    raise SystemExit(main())
