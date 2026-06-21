"""Restore a JSONL graph snapshot (from graph_backup.py) into Neo4j.

INTENDED FOR AN EMPTY DATABASE. Refuses against a non-empty DB unless --force,
to avoid silently duplicating data. Recreates every node (labels + props) and
relationship (type + props), preserving original connectivity via a temporary
`_backup_eid` property + `_Restore` label that are both removed at the end.

Caveat: Neo4j temporal/spatial values were stringified at backup time, so they
return as strings. Structure + DistilBERT embeddings restore exactly.

Usage (from backend/):
    python -m run_logs.graph_restore <snapshot.jsonl> [--force]
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.db import DatabaseManager  # noqa: E402

LABEL_RE = re.compile(r"^[A-Za-z_]\w*$")
BATCH = 500


def _safe(name: str) -> str:
    if not LABEL_RE.match(name):
        raise ValueError(f"unsafe label/relationship type in snapshot: {name!r}")
    return name


def _batched(items: list, size: int = BATCH):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("usage: python -m run_logs.graph_restore <snapshot.jsonl> [--force]", flush=True)
        return 2
    path = args[0]
    force = "--force" in args[1:]
    if not os.path.exists(path):
        print(f"FATAL snapshot not found: {path}", flush=True)
        return 2

    nodes_by_labels: dict[tuple, list] = defaultdict(list)
    rels_by_type: dict[str, list] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o["kind"] == "node":
                nodes_by_labels[tuple(o["labels"])].append(
                    {"eid": o["eid"], "props": o["props"]}
                )
            else:
                rels_by_type[o["type"]].append(
                    {"start": o["start"], "end": o["end"], "props": o["props"]}
                )

    n_nodes = sum(len(v) for v in nodes_by_labels.values())
    n_rels = sum(len(v) for v in rels_by_type.values())
    print(f"RESTORE_BEGIN file={path} nodes={n_nodes} rels={n_rels}", flush=True)

    driver = DatabaseManager.refresh()
    t0 = time.time()
    with driver.session(database=Config.NEO4J_DATABASE) as s:
        existing = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        if existing and not force:
            print(
                f"REFUSE database not empty (nodes={existing}); pass --force to override",
                flush=True,
            )
            return 3
        s.run(
            "CREATE INDEX backup_eid IF NOT EXISTS FOR (n:_Restore) ON (n._backup_eid)"
        )

        made = 0
        for labels, rows in nodes_by_labels.items():
            label_suffix = "".join(f":`{_safe(l)}`" for l in labels)
            cypher = (
                f"UNWIND $rows AS row CREATE (n{label_suffix}:_Restore) "
                "SET n += row.props, n._backup_eid = row.eid"
            )
            for chunk in _batched(rows):
                s.run(cypher, rows=chunk)
                made += len(chunk)
                print(f"RESTORE_NODES {made}/{n_nodes}", flush=True)

        made = 0
        for rtype, rows in rels_by_type.items():
            cypher = (
                "UNWIND $rows AS row "
                "MATCH (a:_Restore {_backup_eid: row.start}), "
                "(b:_Restore {_backup_eid: row.end}) "
                f"CREATE (a)-[r:`{_safe(rtype)}`]->(b) SET r += row.props"
            )
            for chunk in _batched(rows):
                s.run(cypher, rows=chunk)
                made += len(chunk)
                print(f"RESTORE_RELS {made}/{n_rels}", flush=True)

        # Strip restore scaffolding.
        while True:
            removed = s.run(
                "MATCH (n:_Restore) WITH n LIMIT 5000 "
                "REMOVE n:_Restore REMOVE n._backup_eid RETURN count(n) AS c"
            ).single()["c"]
            if not removed:
                break

    print(
        f"RESTORE_DONE nodes={n_nodes} rels={n_rels} elapsed_sec={time.time()-t0:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
