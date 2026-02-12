import sys
import os
import asyncio
import logging

# Add the backend directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import logger
from src.ingestion import process_documents
from src.embed_nodes import embed_nodes

async def run_full_pipeline():
    try:
        logger.info("Starting Full Research Ingestion Pipeline...")
        
        # Step 1: Ingest PDFs and Build Graph
        logger.info("--- Phase 1.1: Document Ingestion ---")
        await process_documents()
        
        # Step 2: Generate Vector Embeddings (Cold Start)
        logger.info("--- Phase 1.2: Node Feature Engineering (DistilBERT) ---")
        await embed_nodes()
        
        logger.info("Pipeline completed successfully! The graph is GNN-Ready.")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_full_pipeline())
