from __future__ import annotations

"""
Chunk-based retrieval for ablation (prompt-only mode).
Loads PDFs from documents dir, chunks text, embeds with SentenceTransformer,
returns top-k chunks by cosine similarity. No graph, no GNN audit.
"""
import os
import numpy as np

from sentence_transformers import SentenceTransformer

from src.config import Config, logger
from src.utils.pdf_loader import extract_text_from_pdf, chunk_text

# Module-level cached model to avoid reloading ~400MB on every query
_cached_model: SentenceTransformer | None = None
# Cache parsed chunks and embeddings keyed by docs_dir path
_cached_chunks: dict[str, list[tuple[str, str, list[float]]]] = {}
_cached_chunks_mtime: dict[str, float] = {}


def _get_model() -> SentenceTransformer:
    global _cached_model
    if _cached_model is None:
        _cached_model = SentenceTransformer(Config.DISTILBERT_MODEL)
    return _cached_model


def _get_dir_mtime(docs_path: str) -> float:
    """Get latest modification time across all PDFs in directory."""
    latest = 0.0
    for f in os.listdir(docs_path):
        if f.lower().endswith(".pdf"):
            latest = max(latest, os.path.getmtime(os.path.join(docs_path, f)))
    return latest


def retrieve_chunks(query: str, top_k: int = 5, docs_dir: str | None = None) -> tuple[str, list[dict]]:
    """
    Retrieve top-k text chunks by semantic similarity.
    Returns (context_string, list of {text, source, score}).
    """
    docs_path = docs_dir or Config.DOCS_DIR
    if not os.path.exists(docs_path):
        logger.warning("Documents dir not found: %s", docs_path)
        return "No documents found.", []

    model = _get_model()
    query_vec = model.encode(query, normalize_embeddings=True)

    # Use cached chunks if directory hasn't changed
    dir_mtime = _get_dir_mtime(docs_path)
    if docs_path in _cached_chunks and _cached_chunks_mtime.get(docs_path) == dir_mtime:
        all_chunks = _cached_chunks[docs_path]
    else:
        all_chunks: list[tuple[str, str, list[float]]] = []
        for filename in sorted(os.listdir(docs_path)):
            if not filename.lower().endswith(".pdf"):
                continue
            file_path = os.path.join(docs_path, filename)
            try:
                text = extract_text_from_pdf(file_path)
                if not text or not text.strip():
                    continue
                chunks = chunk_text(text, chunk_size=512, overlap=100)
                for chunk in chunks:
                    if len(chunk.strip()) < 50:
                        continue
                    emb = model.encode(chunk, normalize_embeddings=True)
                    all_chunks.append((chunk.strip(), filename, emb.tolist()))
            except Exception as e:
                logger.warning("Failed to load %s: %s", filename, e)
        _cached_chunks[docs_path] = all_chunks
        _cached_chunks_mtime[docs_path] = dir_mtime

    if not all_chunks:
        return "No document chunks available.", []

    embeds = np.array([c[2] for c in all_chunks])
    scores = np.dot(embeds, query_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]

    context_parts = []
    result_chunks = []
    for i in top_indices:
        text, source, _ = all_chunks[i]
        score = float(scores[i])
        context_parts.append(f"[{source}] {text}")
        result_chunks.append({"text": text, "source": source, "score": score})

    context_str = "\n\n---\n\n".join(context_parts)
    return context_str, result_chunks
