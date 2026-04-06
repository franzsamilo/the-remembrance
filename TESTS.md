# The Remembrance — Test Guide

## Backend Tests (pytest)

### Prerequisites

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Run All Backend Tests

```bash
cd backend
pytest tests -v
```

### Individual Test Suites

| Test | Command | Description |
|------|---------|-------------|
| API | `pytest tests/test_api.py -v` | Health, stats, path traversal, chat stream |
| Utils | `pytest tests/test_utils.py -v` | Format helpers |

### Test Coverage

- **test_health**: GET /health returns 200 and `status: "online"`
- **test_stats_returns_json**: GET /stats returns JSON (200 or 500)

- **test_delete_documents_path_traversal_blocked**: DELETE /documents/../../../etc/passwd returns 403

- **test_chat_stream_returns_sse**: POST /chat/stream returns SSE with chunk/done/error events

- **test_format_auc_roc_logic**: AUC-ROC formatting logic

## Frontend Tests (Manual)

### Build

```bash
cd frontend
npm run build
```

### Lint

```bash
cd frontend
npm run lint
```

## Manual Verification Checklist

### 1. Chat Streaming

1. Start backend and frontend.
2. Open Knowledge Discovery.
3. Ask a question with "Explain Connections" enabled.
4. Confirm: answer text appears progressively (chunk by chunk), not all at once.

### 2. Graph Zoom/Pan

1. Open Evidence Board for a message with triplets.
2. Expand "Graph View".
3. **Zoom**: Scroll wheel over graph — zoom in/out (0.4x–3x).
4. **Pan**: Click and drag on empty graph area — graph moves.

### 3. Critical Fixes

- **C1**: Config page — no invalid Tailwind classes (e.g. `text-[#8B1A1A]400`).
- **C2**: Main page empty state — text uses `text-[#2B2B2B]/40` not `-400`.
- **C3**: Retriever — `__main__` unpacks `context, triplets, leads`.

### 4. Backend Security

- **CORS**: Set `CORS_ORIGINS=http://localhost:3000` in prod.
- **Upload**: Try uploading a file > 50MB — should get 413.
- **Rate limit**: 60+ requests in 1 min to /stats — should get 429.

### 5. Accessibility

- **Config tabs**: Tab key moves between tabs; Enter/Space activates.
- **Icon buttons**: Trash2, RefreshCw have `aria-label`.
