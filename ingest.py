# Import standard libraries
import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict
import re

# Import third-party libraries
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ingest.log"),
        logging.StreamHandler()
    ]
)

# -----------------------------
# Define directories
# -----------------------------
SOURCE_DIR = Path("data/source")
PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")
EMBEDDINGS_DIR = Path("data/embeddings")
INDEX_DIR = Path("data/index")

for d in [PROCESSED_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load embedding model
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

# -----------------------------
# Utility functions
# -----------------------------
def normalize_text(text: str) -> str:
    """
    Normalize text but KEEP sentence boundaries.
    """
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" >", ">")
    return text.strip()


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages)

    elif path.suffix.lower() == ".docx":
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif path.suffix.lower() in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    else:
        raise ValueError(f"Unsupported file type: {path}")


def save_processed(path: Path, text: str):
    out = PROCESSED_DIR / f"{path.stem}.txt"
    out.write_text(text, encoding="utf-8")


# -----------------------------
# Sentence-based semantic chunking
# -----------------------------
def chunk_text(text: str, max_chars=350) -> List[str]:
    """
    Build semantic chunks from sentences.
    Each chunk represents a logical step / idea.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for s in sentences:
        if not s.strip():
            continue

        if len(current) + len(s) <= max_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current.strip():
        chunks.append(current.strip())

    return chunks


def build_chunks(path: Path, text: str) -> List[Dict]:
    raw_chunks = chunk_text(text)

    # 🔥 Remove empty / junk chunks
    clean_chunks = [
        c for c in raw_chunks
        if c.strip() and len(c.strip()) > 20
    ]

    return [
        {
            "chunk_id": f"{path.stem}_{i}",
            "source_path": str(path),
            "doc_type": path.suffix.lower(),
            "text": chunk
        }
        for i, chunk in enumerate(clean_chunks)
    ]


def save_chunks(path: Path, chunks: List[Dict]):
    out = CHUNKS_DIR / f"{path.stem}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def embed_chunks(chunks: List[Dict]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    return embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


def save_embeddings(path: Path, embeddings: np.ndarray):
    np.save(EMBEDDINGS_DIR / f"{path.stem}.npy", embeddings)


def build_index(all_chunks: List[Dict], all_embeddings: np.ndarray):
    if len(all_embeddings) == 0:
        raise RuntimeError("No embeddings to index")

    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(all_embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(all_chunks, indent=2),
        encoding="utf-8"
    )


# -----------------------------
# Main ingestion pipeline
# -----------------------------
def ingest():
    all_chunks = []
    all_embeddings = []

    for file in SOURCE_DIR.iterdir():
        if not file.is_file():
            continue

        t0 = time.time()
        try:
            logging.info(f"Processing {file.name}...")

            raw_text = extract_text(file)
            clean_text = normalize_text(raw_text)
            save_processed(file, clean_text)

            chunks = build_chunks(file, clean_text)
            save_chunks(file, chunks)
            all_chunks.extend(chunks)

            embeddings = embed_chunks(chunks)
            save_embeddings(file, embeddings)
            all_embeddings.extend(embeddings)

            logging.info(
                f"{file.name}: {len(chunks)} chunks created in {time.time() - t0:.2f}s"
            )

        except Exception as e:
            logging.error(f"Failed to process {file.name}: {e}")

    all_embeddings = np.array(all_embeddings)
    build_index(all_chunks, all_embeddings)

    logging.info("Ingestion complete. Index and chunks rebuilt.")


if __name__ == "__main__":
    ingest()
