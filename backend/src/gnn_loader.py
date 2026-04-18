import torch
from torch_geometric.data import Data
import numpy as np
from src.config import Config, logger
from src.db import DatabaseManager

class GNNLoader:
    def __init__(self):
        self.driver = DatabaseManager.get_driver()

    def fetch_graph_data(self):
        """Fetches nodes, relationships, and schema labels for GNN processing.

        Returns (data, rel_types, node_id_map, label_to_id) where
        data.node_type[i] is the integer id of node i's primary schema label
        (sentinel -1 when no label matches Config.LEGAL_NODE_TYPES).
        """
        dim = Config.EMBEDDING_DIMENSION
        zero_vec = [0.0] * dim

        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                logger.info("Fetching relationships for GNN...")
                rel_query = """
                MATCH (s)-[r]->(t)
                WHERE type(r) <> 'FROM_CHUNK'
                RETURN elementId(r) as rel_id, elementId(s) as source, elementId(t) as target, type(r) as type
                """
                rel_records = list(session.run(rel_query))

                if not rel_records:
                    logger.warning("No relationships found for GNN audit.")
                    return None

                node_ids = set()
                for rec in rel_records:
                    node_ids.add(rec["source"])
                    node_ids.add(rec["target"])

                logger.info("Fetching node embeddings for %s nodes...", len(node_ids))
                emb_query = """
                MATCH (n)
                WHERE elementId(n) IN $node_ids
                  AND n.embedding IS NOT NULL
                  AND n.embedding_model = $embedding_model
                  AND coalesce(n.embedding_dimension, 0) = $embedding_dimension
                RETURN elementId(n) as id, n.embedding as embedding
                """
                emb_result = session.run(
                    emb_query,
                    node_ids=list(node_ids),
                    embedding_model=Config.DISTILBERT_MODEL,
                    embedding_dimension=dim,
                )
                id_to_emb = {r["id"]: r["embedding"] for r in emb_result}

                logger.info("Fetching node labels for %s nodes...", len(node_ids))
                label_query = """
                MATCH (n)
                WHERE elementId(n) IN $node_ids
                RETURN elementId(n) as id, labels(n) as labels
                """
                label_result = session.run(label_query, node_ids=list(node_ids))
                id_to_labels = {r["id"]: list(r["labels"] or []) for r in label_result}

                # Deterministic label→id mapping (alphabetical) so run-to-run
                # pool indices are comparable across audits.
                schema_labels = sorted(set(Config.LEGAL_NODE_TYPES))
                label_to_id = {name: i for i, name in enumerate(schema_labels)}

                node_id_map = {}
                nodes = []
                node_type_list: list = []
                for i, nid in enumerate(sorted(node_ids)):
                    node_id_map[nid] = i
                    nodes.append(id_to_emb.get(nid, zero_vec))
                    # Pick first label that's in the schema. Sentinel -1 when
                    # no match — sampler falls back to uniform for those.
                    raw_labels = id_to_labels.get(nid, [])
                    primary = next((lbl for lbl in raw_labels if lbl in label_to_id), None)
                    node_type_list.append(label_to_id[primary] if primary is not None else -1)

                rel_types = {}
                rel_type_counter = 0
                edge_index = [[], []]
                edge_type_list = []
                edge_rel_id = []

                for rec in rel_records:
                    src = rec["source"]
                    tgt = rec["target"]
                    rtype = rec["type"]
                    if src not in node_id_map or tgt not in node_id_map:
                        continue
                    edge_index[0].append(node_id_map[src])
                    edge_index[1].append(node_id_map[tgt])
                    if rtype not in rel_types:
                        rel_types[rtype] = rel_type_counter
                        rel_type_counter += 1
                    edge_type_list.append(rel_types[rtype])
                    edge_rel_id.append(rec["rel_id"])

                if not edge_type_list:
                    logger.warning("No valid edges for GNN audit.")
                    return None

                x = torch.tensor(nodes, dtype=torch.float)
                norms = x.norm(dim=1, keepdim=True).clamp_min(1e-8)
                x = x / norms
                edge_index_t = torch.tensor(edge_index, dtype=torch.long)
                edge_type_t = torch.tensor(edge_type_list, dtype=torch.long)
                node_type_t = torch.tensor(node_type_list, dtype=torch.long)

                embedded_count = sum(1 for nid in node_ids if nid in id_to_emb)
                unlabeled = int((node_type_t == -1).sum().item())
                logger.info(
                    "Loaded graph: %s nodes (%s embedded, %s unlabeled), %s edges.",
                    len(nodes), embedded_count, unlabeled, len(edge_type_list),
                )

                data = Data(x=x, edge_index=edge_index_t, edge_type=edge_type_t)
                data.node_type = node_type_t
                data.edge_rel_id = edge_rel_id
                return data, rel_types, node_id_map, label_to_id

        except Exception as e:
            logger.error(f"Error loading GNN data: {str(e)}")
            return None

if __name__ == "__main__":
    loader = GNNLoader()
    result = loader.fetch_graph_data()
    if result:
        data, rel_types, node_id_map, label_to_id = result
        logger.info("Graph Data: %s", data)
        logger.info("Rel Types: %s", rel_types)
        logger.info("Label map: %s", label_to_id)
