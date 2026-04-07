"""
Hybrid vector + graph retrieval for the Knowledge Graph Framework.
Fetches seed nodes via embedding similarity, expands via graph traversal,
and returns context triplets plus community-based discovery leads.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import Config, logger
from src.db import DatabaseManager


class GraphRetriever:
    def __init__(self):
        self.driver = DatabaseManager.get_driver()
        # Using the same embedder for consistency
        self.model = SentenceTransformer(Config.DISTILBERT_MODEL)

    def retrieve(self, query, top_k=5, max_hops=2):
        """
        Hybrid Vector + Graph Traversal Retrieval.
        1. Find best matching entities via vector similarity.
        2. Expand context to N hops.
        """
        query_vector = self.model.encode(query).tolist()
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # 1. Vector Search using Cypher similarity on 'embedding' property
                # Since Neo4j Aura might not have vector index by default in free tier, 
                # we use the gds or simple cosine similarity if available, 
                # but for simplicity we'll fetch candidate nodes and rank in python or 
                # use a direct cypher similarity if the DB supports it.
                
                # Optimized Query: Find top seed nodes then expand
                # We use apoc.number.cosineSimilarity or simple dot product
                
                logger.info(f"Retrieving context for query: {query}")
                
                # Step A: Vector Match (Candidate Generation)
                candidate_query = """
                MATCH (n) WHERE n.embedding IS NOT NULL
                WITH n, gds.similarity.cosine(n.embedding, $query_vec) as score
                ORDER BY score DESC
                LIMIT $limit
                RETURN elementId(n) as id, n.name as name, labels(n)[0] as label, n.description as desc
                """
                
                # Check if GDS is available, if not fallback to manual top-N
                # Neo4j Aura Free usually doesn't have GDS. We'll use a standard cypher pattern
                # or fetch embeddings and compute locally.
                
                # Fallback for Aura Free:
                seeds_query = """
                MATCH (n)
                WHERE any(label IN labels(n) WHERE label IN $retrievable_labels)
                  AND n.embedding IS NOT NULL
                  AND n.embedding_model = $embedding_model
                  AND coalesce(n.embedding_dimension, 0) = $embedding_dimension
                RETURN elementId(n) as id, n.name as name, n.embedding as emb, n.description as desc
                LIMIT 100
                """
                
                seed_results = session.run(
                    seeds_query,
                    retrievable_labels=list(Config.LEGAL_NODE_TYPES),
                    embedding_model=Config.DISTILBERT_MODEL,
                    embedding_dimension=Config.EMBEDDING_DIMENSION,
                )
                candidates = []
                query_norm = np.linalg.norm(query_vector)
                if query_norm == 0:
                    return "Query could not be embedded.", [], []
                for record in seed_results:
                    emb = np.array(record["emb"])
                    emb_norm = np.linalg.norm(emb)
                    if emb.size == 0 or emb_norm == 0:
                        continue
                    score = np.dot(emb, query_vector) / (emb_norm * query_norm)
                    candidates.append({
                        "id": record["id"],
                        "name": record["name"],
                        "score": score,
                        "desc": record["desc"]
                    })
                
                # Sort and take top_k
                candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
                seed_ids = [c["id"] for c in candidates]
                
                if not seed_ids:
                    return "No relevant knowledge found in the graph.", [], []

                # Step B: Graph Expansion (Discovery)
                # Fetch paths and relationships with provenance for legal citation
                discovery_query = """
                MATCH (s)-[r]->(t)
                WHERE (elementId(s) IN $seeds OR elementId(t) IN $seeds)
                  AND type(r) <> 'FROM_CHUNK'
                RETURN s.name as source, type(r) as rel, t.name as target,
                       coalesce(r.plausibility_score, r.audit_score) as audit,
                       r.audit_status as audit_status,
                       r.description as rel_desc,
                       s.source_document as s_doc,
                       s.source_documents as s_docs,
                       t.source_document as t_doc,
                       t.source_documents as t_docs
                LIMIT 20
                """
                
                discovery_results = session.run(discovery_query, seeds=seed_ids)
                triplets = []
                for record in discovery_results:
                    s_docs_raw = record.get("s_docs") or []
                    s_doc_single = record.get("s_doc")
                    if s_docs_raw and isinstance(s_docs_raw, (list, tuple)):
                        source_docs = list(dict.fromkeys(str(x) for x in s_docs_raw if x))
                    elif s_doc_single:
                        source_docs = [str(s_doc_single)]
                    else:
                        source_docs = []
                    
                    t_docs_raw = record.get("t_docs") or []
                    t_doc_single = record.get("t_doc")
                    if t_docs_raw and isinstance(t_docs_raw, (list, tuple)):
                        target_docs = list(dict.fromkeys(str(x) for x in t_docs_raw if x))
                    elif t_doc_single:
                        target_docs = [str(t_doc_single)]
                    else:
                        target_docs = []
                    
                    cross_document = (
                        len(source_docs) > 0
                        and len(target_docs) > 0
                        and not (set(source_docs) & set(target_docs))
                    )
                    
                    triplets.append({
                        "source": record["source"],
                        "relation": record["rel"],
                        "target": record["target"],
                        "audit": record["audit"],
                        "audit_status": record["audit_status"],
                        "description": record["rel_desc"],
                        "source_docs": source_docs,
                        "target_docs": target_docs,
                        "cross_document": cross_document,
                    })
                
                # Step C: Community Detection (leads)
                # Find the "Leiden Context" of the top match
                top_seed_id = seed_ids[0]
                leads = []
                
                community_query = """
                MATCH (n) WHERE elementId(n) = $id
                WITH n.community as comm
                WHERE comm IS NOT NULL
                MATCH (m) WHERE m.community = comm AND m.name <> n.name
                RETURN m.name as name, m.description as desc, m.pagerank as rank
                ORDER BY m.pagerank DESC LIMIT 5
                """
                
                # Try-catch for missing schema/properties
                try:
                    comm_results = session.run(community_query, id=top_seed_id)
                    for record in comm_results:
                        desc_text = f": {record['desc']}" if record['desc'] else ""
                        leads.append(f"{record['name']}{desc_text}")
                except Exception as e:
                    logger.warning(f"Community expansion failed (schema likely missing): {e}")

                # Format into context string
                context_parts = []
                for t in triplets:
                    context_parts.append(f"({t['source']})-[{t['relation']}]->({t['target']})")
                
                return "\n".join(context_parts), triplets, leads

        except Exception as e:
            logger.error(f"Retrieval Error: {str(e)}")
            return "Internal Retrieval Error.", [], []

if __name__ == "__main__":
    retriever = GraphRetriever()
    context, triplets, leads = retriever.retrieve("What are the key materials?")
    logger.info("Context:\n%s", context)
    logger.info("Triplets: %s", len(triplets))
    logger.info("Leads: %s", len(leads))
