import torch
from torch_geometric.data import Data
from neo4j import GraphDatabase
import numpy as np
from src.config import Config, logger

class GNNLoader:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )

    def fetch_graph_data(self):
        """Fetches nodes and relationships from Neo4j for GNN processing.
        Maximizes audit coverage by including all non-FROM_CHUNK edges; nodes without
        embeddings receive zero vectors so CompGCN can score every relationship.
        """
        dim = Config.EMBEDDING_DIMENSION
        zero_vec = [0.0] * dim

        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # 1. Fetch all non-FROM_CHUNK relationships
                logger.info("Fetching relationships for GNN...")
                rel_query = """
                MATCH (s)-[r]->(t)
                WHERE type(r) <> 'FROM_CHUNK'
                RETURN id(r) as rel_id, id(s) as source, id(t) as target, type(r) as type
                """
                rel_records = list(session.run(rel_query))

                if not rel_records:
                    logger.warning("No relationships found for GNN audit.")
                    return None

                # 2. Collect unique node IDs from edges
                node_ids = set()
                for rec in rel_records:
                    node_ids.add(rec["source"])
                    node_ids.add(rec["target"])

                # 3. Fetch embeddings for those nodes (or use zero for missing)
                logger.info("Fetching node embeddings for %s nodes...", len(node_ids))
                emb_query = """
                MATCH (n)
                WHERE id(n) IN $node_ids
                  AND n.embedding IS NOT NULL
                  AND n.embedding_model = $embedding_model
                  AND coalesce(n.embedding_dimension, 0) = $embedding_dimension
                RETURN id(n) as id, n.embedding as embedding
                """
                emb_result = session.run(
                    emb_query,
                    node_ids=list(node_ids),
                    embedding_model=Config.DISTILBERT_MODEL,
                    embedding_dimension=dim,
                )
                id_to_emb = {r["id"]: r["embedding"] for r in emb_result}

                # 4. Build node list and id map (order by id for stable indexing)
                node_id_map = {}
                nodes = []
                for i, nid in enumerate(sorted(node_ids)):
                    node_id_map[nid] = i
                    nodes.append(id_to_emb.get(nid, zero_vec))

                # 5. Build edge tensors
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
                # L2 normalize: zero vectors stay zero; non-zero get unit norm for stable training
                norms = x.norm(dim=1, keepdim=True).clamp_min(1e-8)
                x = x / norms
                edge_index_t = torch.tensor(edge_index, dtype=torch.long)
                edge_type_t = torch.tensor(edge_type_list, dtype=torch.long)
                edge_rel_id_t = torch.tensor(edge_rel_id, dtype=torch.long)

                embedded_count = sum(1 for nid in node_ids if nid in id_to_emb)
                logger.info(
                    "Loaded graph: %s nodes (%s embedded), %s edges.",
                    len(nodes), embedded_count, len(edge_type_list),
                )

                return Data(x=x, edge_index=edge_index_t, edge_type=edge_type_t, edge_rel_id=edge_rel_id_t), rel_types, node_id_map

        except Exception as e:
            logger.error(f"Error loading GNN data: {str(e)}")
            return None
        finally:
            self.driver.close()

if __name__ == "__main__":
    loader = GNNLoader()
    data = loader.fetch_graph_data()
    if data:
        logger.info("Graph Data: %s", data[0])
        logger.info("Rel Types: %s", data[1])
