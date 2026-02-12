import os
import sys
import asyncio
import logging
import shutil
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Ensure project structure is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config, logger
from src.ingestion import process_documents
from src.embed_nodes import embed_nodes

app = FastAPI(
    title="The Remembrance Framework API",
    description="Backend API bridge for Knowledge Ingestion and GNN feature engineering.",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Neo4j Driver Management
class DatabaseManager:
    _driver = None

    @classmethod
    def get_driver(cls):
        if cls._driver is None:
            cls._driver = GraphDatabase.driver(
                Config.NEO4J_URI, 
                auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
            )
        return cls._driver

    @classmethod
    def close(cls):
        if cls._driver:
            cls._driver.close()
            cls._driver = None

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Neo4j connectivity...")
    DatabaseManager.get_driver()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Closing Neo4j connectivity...")
    DatabaseManager.close()

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "online", "version": "1.0.0"}

@app.get("/stats")
async def get_stats():
    """Returns real-time graph statistics from Neo4j Aura."""
    try:
        driver = DatabaseManager.get_driver()
        with driver.session(database=Config.NEO4J_DATABASE) as session:
            query = """
            MATCH (n)
            WITH count(n) as total_nodes
            OPTIONAL MATCH (e:__Entity__)
            WITH total_nodes, count(e) as entity_nodes
            OPTIONAL MATCH (f) WHERE f.status = 'Feature-Complete'
            WITH total_nodes, entity_nodes, count(f) as feature_complete
            OPTIONAL MATCH ()-[r]->()
            RETURN total_nodes, entity_nodes, feature_complete, count(r) as total_rels
            """
            result = session.run(query)
            record = result.single()
            
            if not record:
                return {"status": "empty", "message": "No data found in database."}

            stats = {
                "status": "healthy",
                "nodes": record["total_nodes"],
                "entities": record["entity_nodes"],
                "feature_complete": record["feature_complete"],
                "relationships": record["total_rels"],
                "embedding_progress": min(100, (record["feature_complete"] / record["entity_nodes"] * 100)) if record["entity_nodes"] > 0 else 0
            }
            logger.info(f"Stats retrieved: {stats['nodes']} nodes, {stats['relationships']} rels.")
            return stats
    except Exception as e:
        logger.error(f"Failed to fetch stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF document to the research source library."""
    docs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "documents")
    os.makedirs(docs_path, exist_ok=True)
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_path = os.path.join(docs_path, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File uploaded: {file.filename}")
        return {"filename": file.filename, "status": "uploaded"}
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_graph():
    """Wipes the entire Neo4j database to allow for a fresh framework start."""
    try:
        driver = DatabaseManager.get_driver()
        with driver.session(database=Config.NEO4J_DATABASE) as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("FRAMEWORK RESET: Database has been wiped.")
            return {"status": "success", "message": "Graph database has been cleared."}
    except Exception as e:
        logger.error(f"Reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """Triggers the full ingestion and embedding pipeline in the background."""
    async def run_pipeline():
        try:
            logger.info("Starting background ingestion pipeline...")
            await process_documents()
            logger.info("Ingestion complete. Starting embedding cold-start...")
            await embed_nodes()
            logger.info("Pipeline execution finished successfully.")
        except Exception as e:
            logger.error(f"Pipeline failure: {str(e)}")

    background_tasks.add_task(run_pipeline)
    return {"message": "Pipeline triggered. Monitor status via /stats."}

@app.get("/documents")
async def list_documents():
    """Lists available PDF source documents."""
    docs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "documents")
    if not os.path.exists(docs_path):
        return {"documents": []}
    files = [f for f in os.listdir(docs_path) if f.lower().endswith(".pdf")]
    return {"documents": files}

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on port {Config.PORT} (Debug={Config.DEBUG})...")
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)
