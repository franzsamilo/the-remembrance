"""Export the full Neo4j graph to a restorable JSONL snapshot.

The 2026-06-20 incident (graph wiped, no backup, paper numbers unreproducible)
happened because there was no graph backup. This dumps EVERY node (labels + all
properties, including DistilBERT embeddings) and EVERY relationship (type +
endpoints + properties) to a JSONL file that run_logs/graph_restore.py can
replay into an empty database.

Format: one JSON object per line.
  {"kind":"node","eid":..,"labels":[..],"props":{..}}
  {"kind":"rel","eid":..,"type":..,"start":<node eid>,"end":<node eid>,"props":{..}}

Neo4j temporal/spatial values are stringified (default=str); the GNN does not
use them, and structure + embeddings (the reproducibility-critical data) are
preserved exactly.

Usage (from backend/):
    python -m run_logs.graph_backup [output_path]
Default output: run_logs/graph_snapshot_<timestamp>.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.db import DatabaseManager  # noqa: E402

BATCH = 1000


def main() -> int:
    out = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"graph_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
        )
    )
    driver = DatabaseManager.refresh()
    n_nodes = n_rels = 0
    t0 = time.time()
    with driver.session(database=Config.NEO4J_DATABASE) as s, open(
        out, "w", encoding="utf-8"
    ) as f:
        total_nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        total_rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(
            f"BACKUP_BEGIN nodes={total_nodes} rels={total_rels} out={out}",
            flush=True,
        )

        skip = 0
        while True:
            rows = s.run(
                "MATCH (n) RETURN elementId(n) AS eid, labels(n) AS labels, "
                "properties(n) AS props ORDER BY elementId(n) SKIP $skip LIMIT $lim",
                skip=skip,
                lim=BATCH,
            ).data()
            if not rows:
                break
            for r in rows:
                f.write(
                    json.dumps(
                        {"kind": "node", "eid": r["eid"], "labels": r["labels"], "props": r["props"]},
                        default=str,
                    )
                    + "\n"
                )
            n_nodes += len(rows)
            skip += BATCH
            print(f"BACKUP_NODES {n_nodes}/{total_nodes}", flush=True)

        skip = 0
        while True:
            rows = s.run(
                "MATCH (a)-[r]->(b) RETURN elementId(r) AS eid, type(r) AS type, "
                "elementId(a) AS start, elementId(b) AS end, properties(r) AS props "
                "ORDER BY elementId(r) SKIP $skip LIMIT $lim",
                skip=skip,
                lim=BATCH,
            ).data()
            if not rows:
                break
            for r in rows:
                f.write(
                    json.dumps(
                        {
                            "kind": "rel",
                            "eid": r["eid"],
                            "type": r["type"],
                            "start": r["start"],
                            "end": r["end"],
                            "props": r["props"],
                        },
                        default=str,
                    )
                    + "\n"
                )
            n_rels += len(rows)
            skip += BATCH
            print(f"BACKUP_RELS {n_rels}/{total_rels}", flush=True)

    size_mb = os.path.getsize(out) / 1e6
    print(
        f"BACKUP_DONE nodes={n_nodes} rels={n_rels} size_mb={size_mb:.1f} "
        f"elapsed_sec={time.time()-t0:.1f} out={out}",
        flush=True,
    )
    if n_nodes != total_nodes or n_rels != total_rels:
        print("BACKUP_WARN count mismatch -- snapshot may be incomplete!", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
