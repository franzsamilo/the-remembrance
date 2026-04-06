import asyncio
import sys
from datetime import datetime, timezone

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import Config, logger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _select_text_and_field(properties: dict) -> tuple[str | None, str | None]:
    prioritized_fields = ("description", "summary", "text", "content", "excerpt", "name")
    for field_name in prioritized_fields:
        value = properties.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip(), field_name
    return None, None

async def embed_nodes():
    """Align all retrievable nodes onto one DistilBERT embedding space."""

    logger.info("Loading DistilBERT model %s", Config.DISTILBERT_MODEL)
    try:
        model = SentenceTransformer(Config.DISTILBERT_MODEL)
    except Exception as exc:
        logger.error("Failed to load embedding model: %s", exc)
        return

    logger.info("Connecting to Neo4j at %s", Config.NEO4J_URI)
    try:
        driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD),
        )
    except Exception as exc:
        logger.error("Failed to connect to Neo4j: %s", exc)
        return

    try:
        with driver.session(database=Config.NEO4J_DATABASE) as session:
            logger.info("Fetching retrievable nodes that need DistilBERT embeddings")
            query = """
                MATCH (n)
                WHERE any(label IN labels(n) WHERE label IN $retrievable_labels)
                  AND (
                    n.embedding IS NULL
                    OR n.embedding_model IS NULL
                    OR n.embedding_model <> $embedding_model
                    OR coalesce(n.embedding_dimension, 0) <> $embedding_dimension
                  )
                RETURN id(n) as node_id, labels(n) as labels, n as properties
            """
            result = session.run(
                query,
                retrievable_labels=list(Config.LEGAL_NODE_TYPES),
                embedding_model=Config.DISTILBERT_MODEL,
                embedding_dimension=Config.EMBEDDING_DIMENSION,
            )

            nodes = []
            for record in result:
                props = record["properties"]
                text_to_embed, source_field = _select_text_and_field(props)
                if text_to_embed:
                    nodes.append({
                        "node_id": record["node_id"],
                        "text": text_to_embed,
                        "text_field": source_field,
                    })

            if not nodes:
                logger.info("No nodes found that require re-embedding.")
                return

            logger.info("Total nodes to process for embeddings: %s", len(nodes))
            batch_size = Config.EMBEDDING_BATCH_SIZE
            for i in tqdm(range(0, len(nodes), batch_size), desc="Embedding batches"):
                batch = nodes[i : i + batch_size]
                texts = [n["text"] for n in batch]

                logger.info(
                    "Embedding batch %s/%s",
                    i // batch_size + 1,
                    (len(nodes) + batch_size - 1) // batch_size,
                )
                embeddings = model.encode(texts, normalize_embeddings=True)

                with session.begin_transaction() as tx:
                    for node, embedding in zip(batch, embeddings):
                        vector = embedding.tolist()
                        tx.run("""
                            MATCH (n)
                            WHERE id(n) = $node_id
                            SET n.embedding = $vector,
                                n.embedding_model = $embedding_model,
                                n.embedding_dimension = $embedding_dimension,
                                n.embedding_provider = $embedding_provider,
                                n.embedding_status = 'Feature-Complete',
                                n.embedding_text_field = $text_field,
                                n.embedding_updated_at = $updated_at,
                                n.status = 'Feature-Complete'
                        """,
                            node_id=node["node_id"],
                            vector=vector,
                            embedding_model=Config.DISTILBERT_MODEL,
                            embedding_dimension=Config.EMBEDDING_DIMENSION,
                            embedding_provider=Config.EMBEDDING_PROVIDER,
                            text_field=node["text_field"],
                            updated_at=_utc_now_iso(),
                        )

                logger.info("Completed embedding batch %s", i // batch_size + 1)

        logger.info("Node embedding alignment completed successfully.")

    except Exception as exc:
        logger.exception("Embedding alignment failed: %s", exc)
    finally:
        driver.close()

if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    asyncio.run(embed_nodes())
