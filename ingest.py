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
# Trash cleaner
# -----------------------------
def clean_trash(text: str) -> str:
    """
    Remove junk lines like '|' bars, repeated dashes, and UI noise.
    Keeps meaningful content, collapses symbol spam.
    """
    lines = text.splitlines()
    cleaned_lines = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"[| \-]+", s):
            continue
        symbols = sum(1 for c in s if not c.isalnum())
        if symbols / max(1, len(s)) > 0.7:
            continue
        s = re.sub(r"(\|[ \|]*)+", "|", s)
        s = re.sub(r"(-{2,})", "-", s)
        s = re.sub(r"\s+", " ", s)
        cleaned_lines.append(s)
    return "\n".join(cleaned_lines)

# -----------------------------
# Text normalization
# -----------------------------
def normalize_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<!\n)([-*•]\s)", r"\n\1", text)
    text = re.sub(r"(?<!\n)(\d+\.\s)", r"\n\1", text)
    return text.strip()

# -----------------------------
# Extraction functions (PDF, DOCX, TXT only)
# -----------------------------
def extract_pdf_text(path: Path) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text = clean_trash(text)
            pages.append(text)
    return "\n".join(pages)

def extract_docx_text(path: Path) -> str:
    doc = docx.Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return clean_trash(text)

def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path)
    elif suffix == ".docx":
        return extract_docx_text(path)
    elif suffix in [".txt", ".md"]:
        return clean_trash(path.read_text(encoding="utf-8", errors="ignore"))
    else:
        raise ValueError(f"Unsupported file type: {path}")

def save_processed(path: Path, text: str):
    out = PROCESSED_DIR / f"{path.stem}.txt"
    out.write_text(text, encoding="utf-8")

# -----------------------------
# Chunking
# -----------------------------
def _split_blocks(text: str) -> List[str]:
    lines = [l.strip() for l in text.split("\n")]
    blocks, current = [], []
    def flush():
        if current:
            blocks.append("\n".join(current).strip())
            current.clear()
    for ln in lines:
        if not ln:
            flush()
            continue
        if re.match(r"^(\d+\.\s|[-*•]\s)", ln) or ln.startswith("mvn ") or ln.startswith("-D"):
            flush()
            blocks.append(ln)
        else:
            current.append(ln)
    flush()
    return [b for b in blocks if b and len(b) > 1]

def chunk_text(text: str, max_chars=450) -> List[str]:
    blocks = _split_blocks(text)
    chunks, buf = [], ""
    for b in blocks:
        if len(b) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', b)
            cur = ""
            for s in sentences:
                if len(cur) + len(s) <= max_chars:
                    cur += (" " if cur else "") + s
                else:
                    if cur:
                        chunks.append(cur.strip())
                    cur = s
            if cur.strip():
                chunks.append(cur.strip())
            continue
        if len(buf) + len(b) <= max_chars:
            buf += ("\n" if buf else "") + b
        else:
            if buf.strip():
                chunks.append(buf.strip())
            buf = b
    if buf.strip():
        chunks.append(buf.strip())
    uniq, seen = [], set()
    for c in chunks:
        c_norm = re.sub(r"\s+", " ", c).strip()
        if len(c_norm) < 25:
            continue
        if c_norm.lower() in {"not mentioned in the document", "not relevant to the attached documents"}:
            continue
        if c_norm not in seen:
            uniq.append(c_norm)
            seen.add(c_norm)
    return uniq

def build_chunks(path: Path, text: str) -> List[Dict]:
    raw_chunks = chunk_text(text)
    return [
        {
            "chunk_id": f"{path.stem}_{i}",
            "source_path": str(path),
            "doc_type": path.suffix.lower(),
            "text": chunk,
            "has_ocr": False  # no OCR anymore
        }
        for i, chunk in enumerate(raw_chunks)
    ]

def save_chunks(path: Path, chunks: List[Dict]):
    out = CHUNKS_DIR / f"{path.stem}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

# -----------------------------
# Embeddings + index
# -----------------------------
def embed_chunks(chunks: List[Dict]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def save_embeddings(path: Path, embeddings: np.ndarray):
    np.save(EMBEDDINGS_DIR / f"{path.stem}.npy", embeddings)

def build_index(all_chunks: List[Dict], all_embeddings: np.ndarray):
    if len(all_embeddings) == 0:
        logging.error("No embeddings generated. Nothing to index.")
        return
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(all_embeddings)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    (INDEX_DIR / "meta.json").write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")

# -----------------------------
# Main ingestion pipeline
# -----------------------------
def ingest():
    all_chunks, all_embeddings = [], []
    for file in SOURCE_DIR.iterdir():
        if not file.is_file():
            continue
        t0 = time.time()
        try:
            logging.info(f"Processing {file.name}...")
            raw_text = extract_text(file)
            clean_text = normalize_text(raw_text)
            clean_text = clean_trash(clean_text)
            save_processed(file, clean_text)
            chunks = build_chunks(file, clean_text)
            save_chunks(file, chunks)
            all_chunks.extend(chunks)
            embeddings = embed_chunks(chunks)
            save_embeddings(file, embeddings)
            all_embeddings.extend(embeddings)
            logging.info(f"{file.name}: {len(chunks)} chunks created in {time.time() - t0:.2f}s")
        except Exception as e:
            logging.error(f"Failed to process {file.name}: {e}")
            continue
    if len(all_embeddings) == 0:
        logging.error("No embeddings generated. Skipping index build.")
        return
    all_embeddings = np.array(all_embeddings)
    build_index(all_chunks, all_embeddings)
    logging.info("Ingestion complete. Index and chunks rebuilt.")

if __name__ == "__main__":
    ingest()
