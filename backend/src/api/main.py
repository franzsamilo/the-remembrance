import json
import os
import sys
import asyncio
import logging
import shutil
import time
from collections import defaultdict
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

# Ensure project structure is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config, logger
from src.db import DatabaseManager
from src.ingestion import process_documents
from src.embed_nodes import embed_nodes
from src.gnn_module import run_audit
from src.generator import DiscoveryGenerator
from src.evaluation import run_grounding_evaluation
from src.aura_agent_client import invoke_agent

# Live Status Tracking
SYSTEM_STATUS = "Idle"

from contextlib import asynccontextmanager

def _derive_graph_state(stats_record) -> str:
    if stats_record["total_nodes"] == 0:
        return "empty_graph"

    if stats_record["retrievable_nodes"] == 0 or stats_record["source_documents"] == 0:
        return "partial_graph"

    if stats_record["embedded_nodes"] < stats_record["retrievable_nodes"]:
        return "partial_graph"

    if stats_record["provenance_covered_nodes"] < Config.GRAPH_READY_MIN_PROVENANCE_LINKS:
        return "partial_graph"

    if stats_record["latest_ingestion_status"] in {"failed", "partial", "empty"}:
        return "partial_graph"

    return "evidence_ready_graph"


def _derive_audit_state(audit_record) -> str:
    if not audit_record or not audit_record["latest_audit_status"]:
        return "absent"

    if audit_record["audited_relationships"] == 0:
        return "absent"

    if audit_record["latest_audit_status"] == "trained_experimental":
        return "ready"

    return "partial"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Neo4j connectivity
    logger.info("Initializing Neo4j connectivity...")
    DatabaseManager.get_driver()
    yield
    # Shutdown: Close Neo4j connectivity
    logger.info("Closing Neo4j connectivity...")
    DatabaseManager.close()

app = FastAPI(
    title=Config.API_TITLE,
    description="Backend API bridge for Knowledge Ingestion and GNN feature engineering.",
    version="1.0.0",
    lifespan=lifespan
)

# Rate limiting (in-memory, per IP)
_rate_limit_store = defaultdict(list)
_rate_limit_window = Config.RATE_LIMIT_WINDOW_SEC
_rate_limit_max = Config.RATE_LIMIT_REQUESTS


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/"):
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        now = time.time()
        # Prune old entries
        _rate_limit_store[client] = [t for t in _rate_limit_store[client] if now - t < _rate_limit_window]
        if len(_rate_limit_store[client]) >= _rate_limit_max:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
            )
        _rate_limit_store[client].append(now)
        return await call_next(request)


# Enable CORS for frontend communication
_cors_origins = Config.CORS_ORIGINS.split(",") if Config.CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "online", "version": "1.0.0"}

def _stats_research_kpis(latest_auc_roc, latest_mrr):
    """Build research KPIs for dashboard: GNN AUC/MRR + grounding/faithfulness from evaluation_results.json."""
    grounding_score = None
    faithfulness_score = None
    eval_path = Config.EVALUATION_RESULTS_PATH
    if eval_path and os.path.exists(eval_path):
        try:
            with open(eval_path, "r") as f:
                data = json.load(f)
            grounding_score = data.get("grounding_score")
            faithfulness_score = data.get("faithfulness_score")
        except Exception as e:
            logger.warning("Failed to read evaluation results for research KPIs: %s", e)
    return {
        "gnn_auc_roc": float(latest_auc_roc) if latest_auc_roc is not None else None,
        "gnn_mrr": float(latest_mrr) if latest_mrr is not None else None,
        "grounding_score": grounding_score,
        "faithfulness_score": faithfulness_score,
    }


def _empty_stats_response(status: str = "unavailable", message: str = "Could not fetch statistics."):
    """Return a minimal stats structure when Neo4j is unavailable or query fails."""
    return {
        "status": status,
        "message": message,
        "nodes": 0,
        "entities": 0,
        "feature_complete": 0,
        "relationships": 0,
        "embedding_progress": 0,
        "current_task": SYSTEM_STATUS,
        "graph_state": "empty",
        "graph_readiness": {
            "source_documents": 0,
            "provenance_covered_nodes": 0,
            "latest_ingestion_status": "unknown",
            "latest_documents_processed": 0,
            "latest_documents_failed": 0,
            "latest_completed_at": None,
            "embedding_model": Config.DISTILBERT_MODEL,
            "embedding_dimension": Config.EMBEDDING_DIMENSION,
        },
        "audit_readiness": {
            "state": "absent",
            "audited_relationships": 0,
            "total_relationships": 0,
            "latest_audit_status": "unknown",
            "latest_audit_mode": None,
            "latest_audit_completed_at": None,
            "latest_auc_roc": None,
        },
        "research_kpis": {"gnn_auc_roc": None, "gnn_mrr": None, "grounding_score": None, "faithfulness_score": None},
    }


@app.get("/stats")
async def get_stats():
    """Returns graph statistics plus readiness metadata for grounded serving."""
    try:
        driver = DatabaseManager.get_driver()
        with driver.session(database=Config.NEO4J_DATABASE) as session:
            query = """
            MATCH (n)
            WITH count(n) AS total_nodes,
                 count(CASE WHEN any(label IN labels(n) WHERE label IN $retrievable_labels) THEN 1 END) AS retrievable_nodes,
                 count(
                     CASE
                         WHEN any(label IN labels(n) WHERE label IN $retrievable_labels)
                          AND n.embedding_model = $embedding_model
                          AND coalesce(n.embedding_dimension, 0) = $embedding_dimension
                         THEN 1
                     END
                 ) AS embedded_nodes
            OPTIONAL MATCH ()-[r]->()
            WITH total_nodes, retrievable_nodes, embedded_nodes, count(r) AS total_rels
            OPTIONAL MATCH (p)
            WHERE any(label IN labels(p) WHERE label IN $retrievable_labels)
            WITH total_nodes, retrievable_nodes, embedded_nodes, total_rels,
                 count(CASE WHEN p.source_document IS NOT NULL OR size(coalesce(p.source_documents, [])) > 0 THEN 1 END) AS provenance_covered_nodes
            OPTIONAL MATCH (doc:SourceDocument)
            WITH total_nodes, retrievable_nodes, embedded_nodes, total_rels, provenance_covered_nodes, count(doc) AS source_documents
            OPTIONAL MATCH (run:IngestionRun)
            WITH total_nodes, retrievable_nodes, embedded_nodes, total_rels, provenance_covered_nodes, source_documents, run
            ORDER BY run.started_at DESC
            RETURN total_nodes,
                   retrievable_nodes,
                   embedded_nodes,
                   total_rels,
                   provenance_covered_nodes,
                   source_documents,
                   run.status AS latest_ingestion_status,
                   run.documents_processed AS latest_documents_processed,
                   run.documents_failed AS latest_documents_failed,
                   run.completed_at AS latest_completed_at
            LIMIT 1
            """
            result = session.run(
                query,
                retrievable_labels=list(Config.LEGAL_NODE_TYPES),
                embedding_model=Config.DISTILBERT_MODEL,
                embedding_dimension=Config.EMBEDDING_DIMENSION,
            )
            record = result.single()
            audit_result = session.run(
                """
                MATCH ()-[r]->()
                WITH count(r) AS total_relationships,
                     count(CASE WHEN r.audit_updated_at IS NOT NULL THEN 1 END) AS audited_relationships
                OPTIONAL MATCH (run:AuditRun)
                WITH total_relationships, audited_relationships, run
                ORDER BY run.completed_at DESC
                RETURN total_relationships,
                       audited_relationships,
                       run.status AS latest_audit_status,
                       run.completed_at AS latest_audit_completed_at,
                       run.audit_mode AS latest_audit_mode,
                       run.auc_roc AS latest_auc_roc,
                       run.mrr AS latest_mrr
                LIMIT 1
                """
            ).single()
            
            if not record:
                return _empty_stats_response("empty", "No data found in database.")

            graph_state = _derive_graph_state(record)
            audit_state = _derive_audit_state(audit_result)
            stats = {
                "status": "healthy",
                "nodes": record["total_nodes"],
                "entities": record["retrievable_nodes"],
                "feature_complete": record["embedded_nodes"],
                "relationships": record["total_rels"],
                "embedding_progress": min(100, (record["embedded_nodes"] / record["retrievable_nodes"] * 100)) if record["retrievable_nodes"] > 0 else 0,
                "current_task": SYSTEM_STATUS,
                "graph_state": graph_state,
                "graph_readiness": {
                    "source_documents": record["source_documents"],
                    "provenance_covered_nodes": record["provenance_covered_nodes"],
                    "latest_ingestion_status": record["latest_ingestion_status"] or "unknown",
                    "latest_documents_processed": record["latest_documents_processed"] or 0,
                    "latest_documents_failed": record["latest_documents_failed"] or 0,
                    "latest_completed_at": record["latest_completed_at"],
                    "embedding_model": Config.DISTILBERT_MODEL,
                    "embedding_dimension": Config.EMBEDDING_DIMENSION,
                },
                "audit_readiness": {
                    "state": audit_state,
                    "audited_relationships": audit_result["audited_relationships"] if audit_result else 0,
                    "total_relationships": audit_result["total_relationships"] if audit_result else 0,
                    "latest_audit_status": audit_result["latest_audit_status"] if audit_result else "unknown",
                    "latest_audit_mode": audit_result["latest_audit_mode"] if audit_result else None,
                    "latest_audit_completed_at": audit_result["latest_audit_completed_at"] if audit_result else None,
                    "latest_auc_roc": audit_result["latest_auc_roc"] if audit_result else None,
                    "latest_mrr": audit_result["latest_mrr"] if audit_result else None,
                },
                "research_kpis": _stats_research_kpis(
                    audit_result["latest_auc_roc"] if audit_result else None,
                    audit_result["latest_mrr"] if audit_result else None,
                ),
            }
            logger.info(f"Stats retrieved: {stats['nodes']} nodes, {stats['relationships']} rels, task: {SYSTEM_STATUS}")
            return stats
    except Exception as e:
        logger.error(f"Failed to fetch stats: {str(e)}")
        out = _empty_stats_response("error", str(e))
        return out

MAX_UPLOAD_BYTES = Config.UPLOAD_MAX_SIZE_MB * 1024 * 1024


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF document to the research source library."""
    docs_path = Config.DOCS_DIR
    os.makedirs(docs_path, exist_ok=True)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Sanitize filename to prevent path traversal
    safe_name = os.path.basename(file.filename)
    if safe_name != file.filename or ".." in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = os.path.join(docs_path, safe_name)
    try:
        total = 0
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    os.remove(file_path) if os.path.exists(file_path) else None
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {Config.UPLOAD_MAX_SIZE_MB}MB.",
                    )
                buffer.write(chunk)
        logger.info(f"File uploaded: {safe_name}")
        return {"filename": safe_name, "status": "uploaded"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass  # Ignore remove failure; file may not exist
        raise HTTPException(status_code=500, detail="Upload failed. Please try again.")

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
        raise HTTPException(status_code=500, detail="Database reset failed. Please try again.")

@app.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """Triggers the full ingestion and embedding pipeline in the background."""
    async def run_pipeline():
        global SYSTEM_STATUS
        try:
            SYSTEM_STATUS = "Extracting Concepts..."
            logger.info("Starting background ingestion pipeline...")
            manifest = await process_documents()

            if not manifest or manifest.get("documents_processed", 0) == 0:
                SYSTEM_STATUS = "Idle"
                logger.warning("Ingestion produced no processable documents; skipping embedding stage.")
                return

            SYSTEM_STATUS = "Embedding Nodes..."
            logger.info("Ingestion complete. Starting embedding cold-start...")
            await embed_nodes()

            SYSTEM_STATUS = "Idle"
            logger.info(
                "Pipeline execution finished with status=%s processed=%s failed=%s",
                manifest.get("status"),
                manifest.get("documents_processed"),
                manifest.get("documents_failed"),
            )
        except Exception as e:
            SYSTEM_STATUS = f"Error: {str(e)}"
            logger.error(f"Pipeline failure: {str(e)}")

    background_tasks.add_task(run_pipeline)
    return {"message": "Pipeline triggered. Monitor status via /stats."}

from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    explain: bool = False
    mode: str = "graph"  # "graph" = full stack, "prompt_only" = chunk RAG ablation

GROUNDING_ERROR_RESPONSE = {
    "narrative_text": "Grounding Error: I cannot answer this because there is no validated evidence in the Knowledge Graph.",
    "triplets": [],
    "leads": [],
    "context_summary": "",
    "grounding_status": "FAILED - No Validated Triples Found",
}

def _is_grounding_error(result: dict) -> bool:
    """True if the result has no validated evidence to support an answer."""
    triplets = result.get("triplets") or []
    return len(triplets) == 0

def _chat_result_to_response(result: dict, grounding_status: str = "OK - Local Graph") -> dict:
    """Build chat response dict from generator result."""
    result["grounding_status"] = grounding_status
    return result


@app.post("/chat/stream")
async def chat_discovery_stream(request: ChatRequest):
    """Streams chat response: narrative text in chunks, then metadata."""
    async def generate():
        try:
            generator = DiscoveryGenerator()
            if request.mode == "prompt_only":
                result = await generator.generate_answer_prompt_only(request.query)
                grounding_status = "OK - Prompt Only (Ablation)"
            else:
                result = await generator.generate_answer(request.query, explain=request.explain)
                if _is_grounding_error(result):
                    yield f"data: {json.dumps({'type': 'error', **GROUNDING_ERROR_RESPONSE})}\n\n"
                    return
                grounding_status = "OK - Local Graph"
            narrative = result.get("narrative_text", "")
            chunk_size = 40
            for i in range(0, len(narrative), chunk_size):
                chunk = narrative[i : i + chunk_size]
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'type': 'done', 'triplets': result.get('triplets', []), 'leads': result.get('leads', []), 'suggested_actions': result.get('suggested_actions', []), 'grounding_status': grounding_status})}\n\n"
        except Exception as e:
            logger.error(f"Stream chat failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'narrative_text': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat")
@app.post("/query")
async def chat_discovery(request: ChatRequest):
    """Answers research queries. Uses local graph by default; Aura Agent when configured and not preferring local."""
    # Path 0: Prompt-only ablation - bypasses graph and Aura
    if request.mode == "prompt_only":
        generator = DiscoveryGenerator()
        result = await generator.generate_answer_prompt_only(request.query)
        result["grounding_status"] = "OK - Prompt Only (Ablation)"
        return result

    # Path 1: Aura Agent (Module 3) - skipped when PREFER_LOCAL_GRAPH=true
    if Config.aura_configured() and not Config.PREFER_LOCAL_GRAPH:
        try:
            aura_result = await invoke_agent(request.query)
            if not aura_result.get("grounding_ok"):
                return {**GROUNDING_ERROR_RESPONSE, "grounding_status": aura_result.get("error") or GROUNDING_ERROR_RESPONSE["grounding_status"]}
            # Aura success
            if request.explain:
                # Hybrid: Aura answer + local triplets for Graph View and per-edge explanations
                generator = DiscoveryGenerator()
                local_result = await generator.generate_answer(request.query, explain=True)
                return {
                    "narrative_text": aura_result.get("answer", ""),
                    "triplets": local_result.get("triplets", []),
                    "leads": local_result.get("leads", []),
                    "context_summary": local_result.get("context_summary", ""),
                    "suggested_actions": local_result.get("suggested_actions", []),
                    "grounding_status": "OK - Aura Agent + Local Graph",
                }
            return {
                "narrative_text": aura_result.get("answer", ""),
                "triplets": [],
                "leads": [],
                "context_summary": "",
                "grounding_status": "OK - Aura Agent",
            }
        except Exception as e:
            logger.warning("Aura Agent invocation failed, falling back to local retrieval: %s", e)
            # Fall through to local path

    # Path 2: Local retrieval + synthesis
    generator = DiscoveryGenerator()
    result = await generator.generate_answer(request.query, explain=request.explain)
    if _is_grounding_error(result):
        return GROUNDING_ERROR_RESPONSE
    result["grounding_status"] = "OK - Local Graph"
    return result

@app.post("/audit")
async def trigger_audit(background_tasks: BackgroundTasks):
    """Triggers the GNN-based topological integrity audit, then runs grounding/faithfulness evaluation."""
    def run_gnn_audit_then_eval():
        global SYSTEM_STATUS
        try:
            SYSTEM_STATUS = "Running GNN Audit..."
            logger.info("Starting GNN Topological Audit...")
            run_audit()
            logger.info("Audit complete. Running grounding/faithfulness evaluation...")
            SYSTEM_STATUS = "Running Evaluation..."
            asyncio.run(run_grounding_evaluation())
            SYSTEM_STATUS = "Idle"
            logger.info("Evaluation complete.")
        except Exception as e:
            SYSTEM_STATUS = f"Audit Error: {str(e)}"
            logger.error(f"Audit failure: {str(e)}")

    background_tasks.add_task(run_gnn_audit_then_eval)
    return {"message": "GNN Audit triggered. Evaluation will run automatically after audit."}


@app.post("/evaluate")
async def trigger_evaluation(background_tasks: BackgroundTasks):
    """Triggers grounding and faithfulness evaluation (LLM-as-judge)."""
    async def run_eval():
        global SYSTEM_STATUS
        try:
            SYSTEM_STATUS = "Running Evaluation..."
            logger.info("Starting grounding/faithfulness evaluation...")
            await run_grounding_evaluation()
            SYSTEM_STATUS = "Idle"
            logger.info("Evaluation complete.")
        except Exception as e:
            SYSTEM_STATUS = f"Evaluation Error: {str(e)}"
            logger.error(f"Evaluation failure: {str(e)}")

    background_tasks.add_task(run_eval)
    return {"message": "Evaluation triggered. Results will be written to evaluation_results.json."}


@app.get("/documents")
async def list_documents():
    """Lists available PDF source documents."""
    docs_path = Config.DOCS_DIR
    if not os.path.exists(docs_path):
        return {"documents": []}
    files = [f for f in os.listdir(docs_path) if f.lower().endswith(".pdf")]
    logger.info(f"Listing documents in {docs_path}: {files}")
    return {"documents": files}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Deletes a source PDF document by name."""
    docs_path = Config.DOCS_DIR
    file_path = os.path.join(docs_path, filename)
    abs_file_path = os.path.abspath(file_path)
    abs_docs_path = os.path.abspath(docs_path)
    
    logger.info(f"Attempting to delete: {abs_file_path}")
    
    # Security check: ensure the file is actually in the documents folder
    if not abs_file_path.startswith(abs_docs_path):
        logger.warning(f"Security: Blocked attempt to delete outside docs: {abs_file_path}")
        raise HTTPException(status_code=403, detail="Access denied.")
        
    if os.path.exists(abs_file_path):
        os.remove(abs_file_path)
        logger.info(f"File successfully deleted: {filename}")
        return {"status": "success", "message": f"Deleted {filename}"}
    else:
        logger.error(f"Delete failed: {abs_file_path} not found.")
        raise HTTPException(status_code=404, detail="File not found.")

def _build_evaluation_section(latest_auc_roc, latest_mrr, audit_completed_at):
    """Build evaluation KPIs section for research panel."""
    eval_path = Config.EVALUATION_RESULTS_PATH
    grounding_score = None
    faithfulness_score = None
    gen_completed_at = None
    if eval_path and os.path.exists(eval_path):
        try:
            with open(eval_path, "r") as f:
                data = json.load(f)
            grounding_score = data.get("grounding_score")
            faithfulness_score = data.get("faithfulness_score")
            gen_completed_at = data.get("completed_at")
        except Exception as e:
            logger.warning("Failed to read evaluation results: %s", e)

    target_auc = 0.95
    target_mrr = 0.95
    target_grounding = 0.98

    return {
        "gnn": {
            "auc_roc": float(latest_auc_roc) if latest_auc_roc is not None else None,
            "mrr": float(latest_mrr) if latest_mrr is not None else None,
            "target_auc": target_auc,
            "target_mrr": target_mrr,
            "auc_pass": latest_auc_roc >= target_auc if latest_auc_roc is not None else None,
            "mrr_pass": latest_mrr >= target_mrr if latest_mrr is not None else None,
            "completed_at": audit_completed_at,
        },
        "generative": {
            "grounding_score": grounding_score,
            "faithfulness_score": faithfulness_score,
            "target_grounding": target_grounding,
            "grounding_pass": grounding_score >= target_grounding if grounding_score is not None else None,
            "completed_at": gen_completed_at,
        },
    }


@app.get("/config")
async def get_configuration():
    """Returns comprehensive backend configuration and technical details for showcase."""
    # Initialize with empty data in case Neo4j is unavailable
    relationship_distribution = []
    audit_results = []
    neo4j_status = "Disconnected"
    latest_auc_roc = None
    latest_mrr = None
    audit_completed_at = None

    # Try to fetch live data from Neo4j, but don't fail if unavailable
    try:
        driver = DatabaseManager.get_driver()

        with driver.session(database=Config.NEO4J_DATABASE) as session:
            # Get latest AuditRun metadata for evaluation KPIs
            audit_run_result = session.run(
                f"""
                MATCH (run:{Config.AUDIT_RUN_LABEL})
                RETURN run.auc_roc AS auc_roc, run.mrr AS mrr, run.completed_at AS completed_at
                ORDER BY run.completed_at DESC
                LIMIT 1
                """
            ).single()
            if audit_run_result:
                latest_auc_roc = audit_run_result.get("auc_roc")
                latest_mrr = audit_run_result.get("mrr")
                audit_completed_at = audit_run_result.get("completed_at")

            # Get relationship type counts
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            ORDER BY count DESC
            """
            rel_result = session.run(rel_query)
            for record in rel_result:
                relationship_distribution.append({
                    "type": record["type"],
                    "count": record["count"]
                })
            
            # Get audit results (edges with scores)
            audit_query = """
            MATCH (s)-[r]->(t)
            WHERE coalesce(r.plausibility_score, r.audit_score) IS NOT NULL
            RETURN s.name as source, t.name as target, type(r) as relation, 
                   coalesce(r.plausibility_score, r.audit_score) as score
            ORDER BY coalesce(r.plausibility_score, r.audit_score) ASC
            LIMIT 50
            """
            audit_result = session.run(audit_query)
            for record in audit_result:
                audit_results.append({
                    "source": record["source"] or "Unknown",
                    "target": record["target"] or "Unknown",
                    "relation": record["relation"],
                    "score": float(record["score"]) if record["score"] else 0.0
                })
            
            neo4j_status = "Connected"
            logger.info("Successfully fetched live data from Neo4j for config endpoint")
            
    except Exception as e:
        logger.warning(f"Neo4j unavailable for config endpoint, returning static config: {str(e)}")
        # Continue with empty relationship_distribution and audit_results
        
    try:
        # Build comprehensive config response
        config_data = {
            "models": {
                "llm": {
                    "name": "Google Gemini",
                    "model": Config.GEMINI_MODEL,
                    "purpose": "Knowledge extraction from PDFs and narrative synthesis",
                    "provider": "Google AI",
                    "hyperparameters": {
                        "temperature": 0.3,
                        "top_p": 1.0,
                        "top_k": 40,
                        "max_output_tokens": 8192,
                        "stop_sequences": None
                    },
                    "extraction_config": {
                        "schema_guided": True,
                        "entity_types": list(Config.LEGAL_NODE_TYPES),
                        "relationship_types": list(Config.LEGAL_RELATIONSHIP_TYPES),
                        "context_window": "Full document",
                        "chunking": "Automatic via SimpleKGPipeline"
                    },
                    "synthesis_config": {
                        "temperature": 0.3,
                        "persona": Config.SYNTHESIS_PERSONA,
                        "response_structure": "Analytical Briefing",
                        "max_context_triplets": 20,
                        "max_community_leads": 5
                    },
                    "capabilities": ["Entity extraction", "Relationship detection", "Narrative generation", "Schema-guided parsing"],
                    "rate_limiting": {
                        "max_retries": Config.INGESTION_LLM_MAX_RETRIES,
                        "base_delay": f"{Config.INGESTION_LLM_BASE_DELAY} seconds",
                        "backoff_strategy": "Exponential (2^n)"
                    }
                },
                "embedder": {
                    "name": "DistilBERT",
                    "model": Config.DISTILBERT_MODEL,
                    "full_model_name": "distilbert-base-nli-stsb-mean-tokens",
                    "purpose": "High-dimensional semantic vector encoding for graph nodes",
                    "architecture": {
                        "type": "Transformer-based sentence encoder",
                        "base_model": "DistilBERT",
                        "distillation_source": "BERT-base",
                        "layers": 6,
                        "attention_heads": 12,
                        "hidden_size": 768,
                        "intermediate_size": 3072,
                        "max_sequence_length": 512
                    },
                    "training": {
                        "pretrained_on": ["NLI (Natural Language Inference)", "STS-B (Semantic Textual Similarity)"],
                        "fine_tuning": "Sentence similarity tasks",
                        "pooling_strategy": "Mean pooling"
                    },
                    "hyperparameters": {
                        "output_dim": Config.EMBEDDING_DIMENSION,
                        "normalize_embeddings": True,
                        "batch_size": Config.EMBEDDING_BATCH_SIZE,
                        "device": "CPU (auto-detect GPU if available)"
                    },
                    "framework": "sentence-transformers",
                    "performance": {
                        "inference_speed": "~50 sentences/second (CPU)",
                        "memory_footprint": "~250MB"
                    }
                },
                "gnn": {
                    "name": "CompGCN Integrity Validator",
                    "architecture": "CompGCN-style relational message passing",
                    "purpose": "Relationship plausibility scoring and integrity validation",
                    "framework": "PyTorch Geometric",
                    "network_structure": {
                        "input_channels": Config.EMBEDDING_DIMENSION,
                        "hidden_channels": Config.COMPGCN_HIDDEN_CHANNELS,
                        "output_channels": Config.COMPGCN_HIDDEN_CHANNELS,
                        "num_layers": 2,
                        "layer_details": [
                            {
                                "layer": 1,
                                "type": "CompGCNLayer",
                                "in_channels": Config.COMPGCN_HIDDEN_CHANNELS,
                                "out_channels": Config.COMPGCN_HIDDEN_CHANNELS,
                                "activation": "ReLU",
                                "num_relations": "Dynamic (based on graph)",
                                "aggr": "mean"
                            },
                            {
                                "layer": 2,
                                "type": "CompGCNLayer",
                                "in_channels": Config.COMPGCN_HIDDEN_CHANNELS,
                                "out_channels": Config.COMPGCN_HIDDEN_CHANNELS,
                                "activation": "None (final layer)",
                                "num_relations": "Dynamic (based on graph)",
                                "aggr": "mean"
                            }
                        ]
                    },
                    "scoring": {
                        "method": "DistMult",
                        "formula": "score = sigmoid(sum(h_src ⊙ h_rel ⊙ h_dst))",
                        "relation_embeddings": {
                            "dim": Config.COMPGCN_HIDDEN_CHANNELS,
                            "learnable": True,
                            "initialization": "Trainable embedding"
                        },
                        "output_range": "[0, 1]",
                        "interpretation": "Higher score = stronger learned plausibility"
                    },
                    "training": {
                        "mode": "Local training with link prediction objective",
                        "weights": "Trained per audit run",
                        "optimization": "Adam + BCEWithLogitsLoss + ReduceLROnPlateau",
                        "epochs": Config.COMPGCN_EPOCHS,
                        "evaluation": "AUC-ROC, MRR",
                        "early_stopping": f"patience={Config.COMPGCN_PATIENCE}",
                        "note": "Scores remain experimental until calibration and governance thresholds are approved"
                    },
                    "hyperparameters": {
                        "dropout": Config.COMPGCN_DROPOUT,
                        "learning_rate": Config.COMPGCN_LEARNING_RATE,
                        "weight_decay": Config.COMPGCN_WEIGHT_DECAY,
                        "validation_split": Config.COMPGCN_VALIDATION_SPLIT,
                        "neg_ratio": Config.COMPGCN_NEG_RATIO,
                        "patience": Config.COMPGCN_PATIENCE,
                        "grad_clip": Config.COMPGCN_GRAD_CLIP,
                        "label_smoothing": Config.COMPGCN_LABEL_SMOOTHING,
                        "seed": Config.COMPGCN_SEED
                    },
                    "computational": {
                        "device": "CPU (auto-detect GPU)",
                        "precision": "float32",
                        "memory_efficient": True
                    }
                },
                "retriever": {
                    "name": "Hybrid Graph Retriever",
                    "type": "Vector + Graph Traversal",
                    "purpose": "Context retrieval for knowledge discovery queries",
                    "strategy": {
                        "stage_1": "Vector similarity search (cosine)",
                        "stage_2": "Graph expansion (N-hop traversal)",
                        "stage_3": "Community detection (Leiden-based leads)"
                    },
                    "hyperparameters": {
                        "top_k_seeds": 5,
                        "max_hops": 2,
                        "max_triplets": 20,
                        "max_leads": 5,
                        "similarity_metric": "Cosine similarity",
                        "candidate_pool_size": 100
                    },
                    "embedder": {
                        "model": Config.DISTILBERT_MODEL,
                        "same_as_node_embedder": True
                    },
                    "graph_queries": {
                        "seed_selection": "Vector similarity on node embeddings",
                        "expansion": "1-hop neighbors of seed nodes",
                        "community_detection": "Leiden community property (if available)"
                    }
                },
                "ingestion": {
                    "name": "Knowledge Graph Builder",
                    "pipeline": "SimpleKGPipeline (neo4j-graphrag)",
                    "purpose": "PDF → Knowledge Graph transformation",
                    "components": {
                        "pdf_parser": {
                            "library": "Built-in (neo4j-graphrag)",
                            "text_extraction": "Automatic",
                            "chunking": "Automatic (paragraph-based)"
                        },
                        "llm_extractor": {
                            "model": Config.GEMINI_MODEL,
                            "temperature": 0.0,
                            "schema_guided": True,
                            "entity_types": list(Config.LEGAL_NODE_TYPES),
                            "relationship_types": list(Config.LEGAL_RELATIONSHIP_TYPES)
                        },
                        "graph_writer": {
                            "database": "Neo4j Aura",
                            "batch_size": "Auto",
                            "conflict_resolution": "MERGE (avoid duplicates)",
                            "on_error": "IGNORE"
                        }
                    },
                    "hyperparameters": {
                        "from_pdf": True,
                        "max_retries": Config.INGESTION_LLM_MAX_RETRIES,
                        "retry_delay": f"{Config.INGESTION_LLM_BASE_DELAY} seconds (exponential backoff)",
                        "parallel_processing": False,
                        "embedding_on_ingestion": False
                    },
                    "post_processing": {
                        "embedding_step": "Separate (embed_nodes.py)",
                        "embedding_model": Config.DISTILBERT_MODEL,
                        "batch_size": Config.EMBEDDING_BATCH_SIZE
                    }
                }
            },
            "pipeline": {
                "stages": [
                    {"order": 1, "name": "PDF Ingestion", "component": "SimpleKGPipeline", "description": "Extracts text from PDF documents"},
                    {"order": 2, "name": "Entity Extraction", "component": "Gemini LLM", "description": "Identifies entities and relationships using schema-guided extraction"},
                    {"order": 3, "name": "Graph Storage", "component": "Neo4j", "description": "Stores entities and relationships in graph database"},
                    {"order": 4, "name": "Vector Embedding", "component": "DistilBERT", "description": "Generates 768-dim semantic embeddings for all entities"},
                    {"order": 5, "name": "Integrity Validation", "component": "CompGCN", "description": "Trains a local graph model and computes plausibility scores (AUC-ROC, MRR)"},
                    {"order": 6, "name": "Synthesis", "component": "Gemini + Retriever", "description": "Generates analytical briefings from graph context"},
                    {"order": 7, "name": "Evaluation", "component": "LLM-as-Judge", "description": "Grounding and faithfulness evaluation for research KPIs"}
                ],
                "schema": {
                    "node_types": list(Config.LEGAL_NODE_TYPES),
                    "relationship_types": list(Config.LEGAL_RELATIONSHIP_TYPES)
                },
                "retry_strategy": {
                    "max_retries": Config.INGESTION_LLM_MAX_RETRIES,
                    "base_delay": f"{Config.INGESTION_LLM_BASE_DELAY} seconds",
                    "backoff": "Exponential (2^n)",
                    "on_error": "IGNORE"
                },
                "data_flow": [
                    {"stage": 1, "input": "Raw PDFs", "process": "Text extraction, chunking", "output": "Chunks, text", "effect": "Unstructured → structured text"},
                    {"stage": 2, "input": "Chunks", "process": "Gemini LLM (schema-guided)", "output": "Entities, relationships", "effect": "Text → semantic triples"},
                    {"stage": 3, "input": "Triples", "process": "Neo4j MERGE", "output": "Nodes, edges", "effect": "Persistent knowledge graph"},
                    {"stage": 4, "input": "Nodes", "process": "DistilBERT (768-d)", "output": "Node embeddings", "effect": "Enables similarity search"},
                    {"stage": 5, "input": "Graph + embeddings", "process": "CompGCN link prediction", "output": "Plausibility scores", "effect": "Flags semantic anomalies"},
                    {"stage": 6, "input": "Query + graph", "process": "Retriever → Gemini", "output": "Narrative + triplets", "effect": "Grounded answers"},
                    {"stage": 7, "input": "Test queries", "process": "LLM-as-judge", "output": "AUC-ROC, MRR, Grounding, Faithfulness", "effect": "Research KPIs"}
                ]
            },
            "synthesis": {
                "persona": Config.SYNTHESIS_PERSONA,
                "temperature": 0.3,
                "response_structure": "Analytical Briefing",
                "strategy": "Discovery-Driven Intelligence",
                "features": ["Direct answers", "Community leads", "Contextual analysis", "Evidence-based reasoning"]
            },
            "neo4j": {
                "uri": Config.NEO4J_URI.replace("neo4j+s://", "neo4j+s://***@").split("@")[-1] if Config.NEO4J_URI else "Not configured",
                "database": Config.NEO4J_DATABASE,
                "driver_status": neo4j_status,
                "type": "Neo4j Aura (Cloud)"
            },
            "connections": {
                "relationship_distribution": relationship_distribution,
                "total_types": len(relationship_distribution)
            },
            "audit": {
                "results": audit_results,
                "total_audited": len(audit_results),
                "low_confidence_threshold": 0.95,
                "auc_roc": float(latest_auc_roc) if latest_auc_roc is not None else None,
                "mrr": float(latest_mrr) if latest_mrr is not None else None,
                "completed_at": audit_completed_at,
            },
            "evaluation": _build_evaluation_section(latest_auc_roc, latest_mrr, audit_completed_at),
            "tech_stack": [
                {"name": "Python", "version": "3.11+", "role": "Backend runtime"},
                {"name": "FastAPI", "version": "Latest", "role": "REST API framework"},
                {"name": "PyTorch", "version": "Latest", "role": "Deep learning framework"},
                {"name": "PyTorch Geometric", "version": "Latest", "role": "Graph neural networks"},
                {"name": "Neo4j", "version": "Aura", "role": "Graph database"},
                {"name": "Google Gemini", "version": Config.GEMINI_MODEL, "role": "LLM for extraction & synthesis"},
                {"name": "LangChain", "version": "Latest", "role": "LLM orchestration"},
                {"name": "sentence-transformers", "version": "Latest", "role": "Embedding generation"},
                {"name": "Next.js", "version": "16", "role": "Frontend framework"},
                {"name": "Tailwind CSS", "version": "4", "role": "UI styling"},
                {"name": "Framer Motion", "version": "12", "role": "Animations"}
            ],
            "api_endpoints": [
                {"method": "GET", "path": "/health", "description": "Health check"},
                {"method": "GET", "path": "/stats", "description": "Real-time graph statistics (includes research_kpis)"},
                {"method": "GET", "path": "/documents", "description": "List source PDFs"},
                {"method": "GET", "path": "/config", "description": "Backend configuration showcase"},
                {"method": "POST", "path": "/upload", "description": "Upload PDF document"},
                {"method": "POST", "path": "/ingest", "description": "Trigger ingestion pipeline"},
                {"method": "POST", "path": "/chat", "description": "Knowledge discovery (body: query, explain, mode)"},
                {"method": "POST", "path": "/chat/stream", "description": "SSE stream chat (body: query, explain, mode=graph|prompt_only)"},
                {"method": "POST", "path": "/audit", "description": "Run GNN audit (triggers evaluation after)"},
                {"method": "POST", "path": "/evaluate", "description": "Run grounding/faithfulness evaluation"},
                {"method": "POST", "path": "/reset", "description": "Wipe graph database"},
                {"method": "DELETE", "path": "/documents/{filename}", "description": "Delete source PDF"}
            ],
            "environment": {
                "port": Config.PORT,
                "debug": Config.DEBUG,
                "docs_dir": Config.DOCS_DIR
            },
            "ui": {
                "app_name": Config.FRONTEND_APP_NAME,
                "api_title": Config.API_TITLE,
                "show_dashboard_stats": Config.SHOW_DASHBOARD_STATS
            }
        }
        
        logger.info("Configuration data retrieved for showcase")
        return config_data
        
    except Exception as e:
        logger.error(f"Failed to fetch configuration: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load configuration.")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on port {Config.PORT} (Debug={Config.DEBUG})...")
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)
