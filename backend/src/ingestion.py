from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
import warnings
from typing import List, Optional, Sequence, Union

from sentence_transformers import SentenceTransformer

# Suppress Google API warnings about Python 3.9
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.auth")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.oauth2")

# Append parent directory to sys.path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import DatabaseManager
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import (
    Config,
    GOOGLE_API_KEY,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    logger,
)


from src.helpers import utc_now_iso as _utc_now_iso


class DistilBertEmbedder(Embedder):
    """Use the same sentence-transformer embedding space across ingestion and retrieval."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.internal_embedder = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * Config.EMBEDDING_DIMENSION

        embedding = self.internal_embedder.encode(text, normalize_embeddings=True)
        return embedding.tolist()


class GeminiLLM(LLMInterface):
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        super().__init__(model_name=model_name, model_params={})
        self.internal_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
        )

    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        max_retries = Config.INGESTION_LLM_MAX_RETRIES
        base_delay = Config.INGESTION_LLM_BASE_DELAY
        for i in range(max_retries):
            try:
                result = self.internal_llm.invoke(input)
                return LLMResponse(content=str(result.content))
            except Exception as exc:
                if "429" in str(exc) and i < max_retries - 1:
                    wait_time = base_delay * (2 ** i)
                    logger.warning(
                        "LLM rate limited. Retrying in %ss (attempt %s/%s).",
                        wait_time,
                        i + 1,
                        max_retries,
                    )
                    time.sleep(wait_time)
                else:
                    raise LLMGenerationError(exc) from exc

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        max_retries = Config.INGESTION_LLM_MAX_RETRIES
        base_delay = Config.INGESTION_LLM_BASE_DELAY
        for i in range(max_retries):
            try:
                result = await self.internal_llm.ainvoke(input)
                return LLMResponse(content=str(result.content))
            except Exception as exc:
                if "429" in str(exc) and i < max_retries - 1:
                    wait_time = base_delay * (2 ** i)
                    logger.warning(
                        "Async LLM rate limited. Retrying in %ss (attempt %s/%s).",
                        wait_time,
                        i + 1,
                        max_retries,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise LLMGenerationError(exc) from exc

    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ):
        raise NotImplementedError("Tool calling not implemented for this wrapper.")

    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ):
        raise NotImplementedError("Tool calling not implemented for this wrapper.")


def _count_graph(session) -> tuple[int, int]:
    record = session.run(
        """
        MATCH (n)
        WITH count(n) AS total_nodes
        OPTIONAL MATCH ()-[r]->()
        RETURN total_nodes, count(r) AS total_relationships
        """
    ).single()
    return int(record["total_nodes"]), int(record["total_relationships"])


def _snapshot_graph_ids(session) -> tuple[set[str], set[str]]:
    node_records = session.run("MATCH (n) RETURN elementId(n) AS id")
    rel_records = session.run("MATCH ()-[r]->() RETURN elementId(r) AS id")
    return (
        {record["id"] for record in node_records},
        {record["id"] for record in rel_records},
    )


def _upsert_source_document(session, run_id: str, filename: str, file_path: str, status: str, error: str | None = None) -> None:
    timestamp = _utc_now_iso()
    session.run(
        f"""
        MERGE (doc:{Config.SOURCE_DOCUMENT_LABEL} {{filename: $filename}})
        ON CREATE SET doc.created_at = $timestamp
        SET doc.path = $file_path,
            doc.status = $status,
            doc.last_run_id = $run_id,
            doc.last_updated_at = $timestamp,
            doc.error = $error
        """,
        filename=filename,
        file_path=file_path,
        status=status,
        run_id=run_id,
        timestamp=timestamp,
        error=error,
    )


def _tag_document_provenance(
    session,
    run_id: str,
    filename: str,
    file_path: str,
    new_node_ids: list[str],
    new_rel_ids: list[str],
) -> None:
    timestamp = _utc_now_iso()
    if new_node_ids:
        session.run(
            """
            MATCH (n)
            WHERE elementId(n) IN $node_ids
            SET n.source_document = $filename,
                n.source_path = $file_path,
                n.ingestion_run_id = $run_id,
                n.provenance_status = 'document_level',
                n.provenance_updated_at = $timestamp,
                n.source_documents = CASE
                    WHEN $filename IN coalesce(n.source_documents, []) THEN coalesce(n.source_documents, [])
                    ELSE coalesce(n.source_documents, []) + $filename
                END
            """,
            node_ids=new_node_ids,
            filename=filename,
            file_path=file_path,
            run_id=run_id,
            timestamp=timestamp,
        )

    if new_rel_ids:
        session.run(
            """
            MATCH ()-[r]->()
            WHERE elementId(r) IN $rel_ids
            SET r.source_document = $filename,
                r.source_path = $file_path,
                r.ingestion_run_id = $run_id,
                r.provenance_status = 'document_level',
                r.provenance_updated_at = $timestamp,
                r.source_documents = CASE
                    WHEN $filename IN coalesce(r.source_documents, []) THEN coalesce(r.source_documents, [])
                    ELSE coalesce(r.source_documents, []) + $filename
                END
            """,
            rel_ids=new_rel_ids,
            filename=filename,
            file_path=file_path,
            run_id=run_id,
            timestamp=timestamp,
        )

        session.run(
            """
            MATCH (s)-[r]->(t)
            WHERE elementId(r) IN $rel_ids
            SET s.source_documents = CASE
                    WHEN $filename IN coalesce(s.source_documents, []) THEN coalesce(s.source_documents, [])
                    ELSE coalesce(s.source_documents, []) + $filename
                END,
                s.provenance_updated_at = $timestamp,
                t.source_documents = CASE
                    WHEN $filename IN coalesce(t.source_documents, []) THEN coalesce(t.source_documents, [])
                    ELSE coalesce(t.source_documents, []) + $filename
                END,
                t.provenance_updated_at = $timestamp
            """,
            rel_ids=new_rel_ids,
            filename=filename,
            timestamp=timestamp,
        )


def _persist_ingestion_run(session, manifest: dict) -> None:
    session.run(
        f"""
        MERGE (run:{Config.INGESTION_RUN_LABEL} {{run_id: $run_id}})
        ON CREATE SET run.created_at = $started_at
        SET run.started_at = $started_at,
            run.completed_at = $completed_at,
            run.status = $status,
            run.files_discovered = $files_discovered,
            run.documents_processed = $documents_processed,
            run.documents_failed = $documents_failed,
            run.nodes_before = $nodes_before,
            run.nodes_after = $nodes_after,
            run.relationships_before = $relationships_before,
            run.relationships_after = $relationships_after,
            run.nodes_created = $nodes_created,
            run.relationships_created = $relationships_created,
            run.embedding_model = $embedding_model,
            run.embedding_dimension = $embedding_dimension,
            run.documents_json = $documents_json
        """,
        run_id=manifest["run_id"],
        started_at=manifest["started_at"],
        completed_at=manifest["completed_at"],
        status=manifest["status"],
        files_discovered=manifest["files_discovered"],
        documents_processed=manifest["documents_processed"],
        documents_failed=manifest["documents_failed"],
        nodes_before=manifest["nodes_before"],
        nodes_after=manifest["nodes_after"],
        relationships_before=manifest["relationships_before"],
        relationships_after=manifest["relationships_after"],
        nodes_created=manifest["nodes_created"],
        relationships_created=manifest["relationships_created"],
        embedding_model=Config.DISTILBERT_MODEL,
        embedding_dimension=Config.EMBEDDING_DIMENSION,
        documents_json=json.dumps(manifest["documents"]),
    )


async def process_documents():
    """Ingest documents and persist run metadata for graph readiness checks."""

    run_id = str(uuid.uuid4())
    manifest = {
        "run_id": run_id,
        "started_at": _utc_now_iso(),
        "completed_at": None,
        "status": "running",
        "files_discovered": 0,
        "documents_processed": 0,
        "documents_failed": 0,
        "nodes_before": 0,
        "nodes_after": 0,
        "relationships_before": 0,
        "relationships_after": 0,
        "nodes_created": 0,
        "relationships_created": 0,
        "documents": [],
    }

    driver = DatabaseManager.get_driver()

    try:
        documents_folder = Config.DOCS_DIR
        if not os.path.exists(documents_folder):
            manifest["status"] = "failed"
            manifest["completed_at"] = _utc_now_iso()
            logger.error("Documents directory not found: %s", documents_folder)
            return manifest

        pdf_files = sorted(f for f in os.listdir(documents_folder) if f.lower().endswith(".pdf"))
        manifest["files_discovered"] = len(pdf_files)

        if not pdf_files:
            manifest["status"] = "empty"
            manifest["completed_at"] = _utc_now_iso()
            logger.warning("No PDF documents found in %s", documents_folder)
            return manifest

        llm = GeminiLLM(api_key=GOOGLE_API_KEY, model_name=Config.GEMINI_MODEL)
        embedder = DistilBertEmbedder(Config.DISTILBERT_MODEL)
        schema_config = {
            "node_types": list(Config.LEGAL_NODE_TYPES),
            "relationship_types": [rel for rel in Config.LEGAL_RELATIONSHIP_TYPES if rel != "FROM_CHUNK"],
        }

        logger.info("Initializing knowledge graph pipeline for run %s", run_id)
        on_error = Config.INGESTION_ON_ERROR if Config.INGESTION_ON_ERROR in ("RAISE", "IGNORE") else "RAISE"
        kg_pipeline = SimpleKGPipeline(
            llm=llm,
            driver=driver,
            embedder=embedder,
            schema=schema_config,
            neo4j_database=NEO4J_DATABASE,
            on_error=on_error,
            from_pdf=True,
        )

        with driver.session(database=NEO4J_DATABASE) as session:
            manifest["nodes_before"], manifest["relationships_before"] = _count_graph(session)

        for pdf_file in pdf_files:
            file_path = os.path.join(documents_folder, pdf_file)
            logger.info("Processing source document %s", pdf_file)

            with driver.session(database=NEO4J_DATABASE) as session:
                _upsert_source_document(session, run_id, pdf_file, file_path, "processing")
                node_ids_before, rel_ids_before = _snapshot_graph_ids(session)

            try:
                await kg_pipeline.run_async(file_path=file_path)
                manifest["documents_processed"] += 1
                manifest["documents"].append({"filename": pdf_file, "status": "processed"})

                with driver.session(database=NEO4J_DATABASE) as session:
                    node_ids_after, rel_ids_after = _snapshot_graph_ids(session)
                    new_node_ids = sorted(node_ids_after - node_ids_before)
                    new_rel_ids = sorted(rel_ids_after - rel_ids_before)
                    _tag_document_provenance(
                        session,
                        run_id,
                        pdf_file,
                        file_path,
                        new_node_ids,
                        new_rel_ids,
                    )
                    _upsert_source_document(session, run_id, pdf_file, file_path, "processed")
                    manifest["documents"][-1]["new_nodes"] = len(new_node_ids)
                    manifest["documents"][-1]["new_relationships"] = len(new_rel_ids)

            except Exception as exc:
                manifest["documents_failed"] += 1
                manifest["documents"].append(
                    {"filename": pdf_file, "status": "failed", "error": str(exc)}
                )
                logger.exception("Failed to process %s", pdf_file)
                with driver.session(database=NEO4J_DATABASE) as session:
                    _upsert_source_document(session, run_id, pdf_file, file_path, "failed", str(exc))

        with driver.session(database=NEO4J_DATABASE) as session:
            manifest["nodes_after"], manifest["relationships_after"] = _count_graph(session)
            manifest["nodes_created"] = max(0, manifest["nodes_after"] - manifest["nodes_before"])
            manifest["relationships_created"] = max(
                0, manifest["relationships_after"] - manifest["relationships_before"]
            )
            manifest["completed_at"] = _utc_now_iso()
            if manifest["documents_processed"] and not manifest["documents_failed"]:
                manifest["status"] = "success"
            elif manifest["documents_processed"]:
                manifest["status"] = "partial"
            else:
                manifest["status"] = "failed"

            _persist_ingestion_run(session, manifest)

        logger.info(
            "Ingestion run %s completed with status=%s processed=%s failed=%s",
            run_id,
            manifest["status"],
            manifest["documents_processed"],
            manifest["documents_failed"],
        )
        return manifest


if __name__ == "__main__":
    asyncio.run(process_documents())
