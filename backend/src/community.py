"""
Leiden community detection and PageRank computation.
Pulls the graph from Neo4j, runs Leiden + PageRank in Python via igraph,
and writes `community` and `pagerank` properties back to nodes.
"""
from __future__ import annotations

import igraph as ig
import leidenalg

from src.config import Config, logger
from src.db import DatabaseManager
from src.helpers import utc_now_iso as _utc_now_iso


def run_community_detection(resolution: float | None = None) -> dict:
    """
    Run Leiden community detection and PageRank on the knowledge graph.
    Writes `community` (int) and `pagerank` (float) properties to every node
    that participates in non-FROM_CHUNK edges.

    Args:
        resolution: Leiden resolution parameter γ. Higher values yield smaller,
                    denser communities. Default from Config.LEIDEN_RESOLUTION.

    Returns:
        dict with community count, node count, and modularity.
    """
    if resolution is None:
        resolution = Config.LEIDEN_RESOLUTION

    driver = DatabaseManager.get_driver()

    with driver.session(database=Config.NEO4J_DATABASE) as session:
        # 1. Fetch all non-FROM_CHUNK edges
        logger.info("Fetching graph for community detection...")
        rel_records = list(session.run("""
            MATCH (s)-[r]->(t)
            WHERE type(r) <> 'FROM_CHUNK'
            RETURN elementId(s) AS source, elementId(t) AS target
        """))

        if not rel_records:
            logger.warning("No relationships found for community detection.")
            return {"communities": 0, "nodes": 0, "modularity": None}

        # 2. Build node ID mapping
        node_ids = set()
        for rec in rel_records:
            node_ids.add(rec["source"])
            node_ids.add(rec["target"])

        node_list = sorted(node_ids)
        node_to_idx = {nid: i for i, nid in enumerate(node_list)}

        # 3. Build igraph graph (undirected for Leiden)
        edges = []
        for rec in rel_records:
            src_idx = node_to_idx[rec["source"]]
            tgt_idx = node_to_idx[rec["target"]]
            if src_idx != tgt_idx:
                edges.append((src_idx, tgt_idx))

        g = ig.Graph(n=len(node_list), edges=edges, directed=False)
        g.simplify()  # remove duplicate edges and self-loops

        # 4. Run Leiden
        logger.info(
            "Running Leiden community detection (resolution=%.2f) on %s nodes, %s edges...",
            resolution, g.vcount(), g.ecount(),
        )
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=Config.COMPGCN_SEED,
        )
        communities = partition.membership
        modularity = partition.modularity
        num_communities = len(set(communities))

        logger.info(
            "Leiden complete: %s communities, modularity=%.4f",
            num_communities, modularity,
        )

        # 5. Run PageRank
        pagerank_scores = g.pagerank(directed=False)

        # 6. Write community and pagerank back to Neo4j
        logger.info("Writing community and pagerank properties to Neo4j...")
        updates = []
        for i, nid in enumerate(node_list):
            updates.append({
                "node_id": nid,
                "community": communities[i],
                "pagerank": pagerank_scores[i],
            })

        batch_size = 500
        updated_at = _utc_now_iso()
        for batch_start in range(0, len(updates), batch_size):
            batch = updates[batch_start:batch_start + batch_size]
            session.run(
                """
                UNWIND $updates AS u
                MATCH (n)
                WHERE elementId(n) = u.node_id
                SET n.community = u.community,
                    n.pagerank = u.pagerank,
                    n.community_updated_at = $updated_at
                """,
                updates=batch,
                updated_at=updated_at,
            )

        logger.info(
            "Community detection complete: %s nodes updated, %s communities.",
            len(updates), num_communities,
        )

        return {
            "communities": num_communities,
            "nodes": len(updates),
            "modularity": float(modularity),
            "resolution": resolution,
            "completed_at": updated_at,
        }
