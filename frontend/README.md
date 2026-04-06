# Reference UI for Knowledge Graph Framework API

This is the **reference implementation** frontend for the Knowledge Graph Framework API. It demonstrates:

- Document upload and ingestion
- Semantic audit (GNN-based)
- Knowledge discovery chat with evidence trails
- Backend configuration and system status

Consumers can build their own UIs against the API. This frontend serves as a proof-of-concept and development reference.

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Ensure the backend API is running (default: `http://localhost:8000`).

## Configuration

Set `NEXT_PUBLIC_API_URL` to point to your backend API (default: `http://localhost:8000`).
