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


def retrieve_chunks(query: str, top_k: int = 5, docs_dir: str | None = None) -> tuple[str, list[dict]]:
    """
    Retrieve top-k text chunks by semantic similarity.
    Returns (context_string, list of {text, source, score}).
    """
    docs_path = docs_dir or Config.DOCS_DIR
    if not os.path.exists(docs_path):
        logger.warning("Documents dir not found: %s", docs_path)
        return "No documents found.", []

    model = SentenceTransformer(Config.DISTILBERT_MODEL)
    query_vec = model.encode(query, normalize_embeddings=True)

    all_chunks: list[tuple[str, str, list[float]]] = []  # (text, source_filename, embedding)

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
