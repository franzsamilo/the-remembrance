"""
Legacy PDF loader utilities. Not used by the main ingestion pipeline,
which uses SimpleKGPipeline (neo4j-graphrag) instead. Kept for potential
standalone or alternative use cases.
"""
import fitz  # PyMuPDF
import os

from src.config import logger


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a single PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_documents_from_folder(folder_path: str) -> list[str]:
    """Loads all PDF files from the specified folder."""
    texts = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) # Ensure folder exists
        return []
        
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                text = extract_text_from_pdf(file_path)
                texts.append(text)
                logger.info("Loaded: %s", filename)
            except Exception as e:
                logger.warning("Error loading %s: %s", filename, e)
    return texts

def chunk_text(text: str, chunk_size=1024, overlap=200) -> list[str]:
    """Chunks text into segments with overlap."""
    # Simple character/word based chunking for now, could be improved with tiktoken
    words = text.split() 
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks
