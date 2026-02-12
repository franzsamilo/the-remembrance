import os
import asyncio
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import sys

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

async def embed_nodes():
    # Initialize DistilBERT model
    # 'distilbert-base-nli-stsb-mean-tokens' is a standard model for semantic similarity
    print("Loading DistilBERT model (sentence-transformers)...")
    try:
        model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Connect to Neo4j
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return
    
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            # 1. Fetch nodes that need embeddings
            # We look for nodes with '__Entity__' or 'Entity' label
            print("Fetching nodes from Neo4j...")
            query = """
                MATCH (n)
                WHERE (n:__Entity__ OR n:Entity) AND n.embedding IS NULL
                RETURN id(n) as node_id, n as properties
            """
            result = session.run(query)
            
            nodes = []
            for record in result:
                props = record["properties"]
                # Try to find a property to embed. Order: description, name, id, text
                text_to_embed = (
                    props.get("description") or 
                    props.get("name") or 
                    props.get("id") or 
                    props.get("text") or
                    props.get("summary")
                )
                
                if text_to_embed:
                    nodes.append({
                        "node_id": record["node_id"],
                        "text": text_to_embed
                    })
            
            if not nodes:
                print("No nodes found that require embedding.")
                return

            print(f"Total nodes to process: {len(nodes)}")
            
            # 2. Batch processing to avoid timeouts and high memory usage
            batch_size = 50
            for i in tqdm(range(0, len(nodes), batch_size), desc="Embedding batches"):
                batch = nodes[i : i + batch_size]
                texts = [n["text"] for n in batch]
                
                print(f"Embedding batch {i // batch_size + 1}/{ (len(nodes) + batch_size - 1) // batch_size}...")
                embeddings = model.encode(texts)
                
                # 3. Update Neo4j
                # Using a single transaction per batch for efficiency
                with session.begin_transaction() as tx:
                    for node, embedding in zip(batch, embeddings):
                        # Convert numpy array to list for Neo4j compatibility
                        vector = embedding.tolist()
                        tx.run("""
                            MATCH (n)
                            WHERE id(n) = $node_id
                            SET n.embedding = $vector, n.status = 'Feature-Complete'
                        """, node_id=node["node_id"], vector=vector)
                
                print(f"Batch {i // batch_size + 1} completed and saved to Neo4j.")

        print("\nNode embedding process 'Cold Start' completed successfully!")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    # Ensure UTF-8 output for Windows terminals if needed
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    asyncio.run(embed_nodes())
