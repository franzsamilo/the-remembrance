# The Remembrance Vault

Archival Knowledge Repository — a framework for ingesting documents into a knowledge graph, auditing relationships with GNN, and answering queries with grounded, evidence-based responses.

## Quick Start

### Backend

1. Create a virtual environment and install dependencies:

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in required values:

   ```bash
   cp .env.example .env
   ```

3. Required environment variables:
   - `NEO4J_URI` — Neo4j connection string (e.g. `neo4j+s://...`)
   - `NEO4J_PASSWORD` — Neo4j password
   - `GOOGLE_API_KEY` — Google AI (Gemini) API key

4. Run the API server:

   ```bash
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend

1. Install dependencies and run:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

2. Open [http://localhost:3000](http://localhost:3000).

## Architecture

- **Ingestion**: PDF documents → Neo4j GraphRAG pipeline → entities and relationships
- **Embedding**: DistilBERT embeddings for vector similarity
- **Audit**: CompGCN-based GNN scores relationship plausibility
- **Chat**: Local retrieval + synthesis (Gemini) or optional Aura Agent

## API Documentation

When the backend is running, OpenAPI docs are available at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Frontend

The included frontend is a **proof-of-concept reference implementation** that demonstrates the backend Knowledge Graph Framework API. Consumers can build their own UIs against the API; the reference UI shows upload, ingestion, audit, and chat flows. Configuration and framework-internal stats are available on the Backend Config page.

## Evaluation

See [EVALUATION.md](EVALUATION.md) for success criteria (Grounding ≥ 0.9, AUC ≥ 0.75) and ablation baselines (Prompt Only vs. Graph vs. Full Stack). Run `POST /evaluate` to compute grounding/faithfulness; GNN audit runs evaluation automatically.

## Configuration

- `PREFER_LOCAL_GRAPH=true` — Use local retrieval by default (recommended if Aura Agent schema differs)
- `CORS_ORIGINS` — Restrict CORS in production (e.g. `http://localhost:3000`)
- `UPLOAD_MAX_SIZE_MB` — Max PDF upload size (default 50)
