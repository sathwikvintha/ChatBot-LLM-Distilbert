import os
from typing import Dict
import pickle
import json, time, logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from generate import generate_answer, init_qa_router
from qdrant_client import QdrantClient
import numpy as np
from config import QDRANT_PATH, COLLECTION_NAME

# -----------------------------
# Configuration (Enterprise Ready)
# -----------------------------
DOC_EMBEDDINGS_CACHE = Path("data/doc_embeddings.pkl")
SCROLL_BATCH_SIZE = 1000

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Paths
# -----------------------------
TRAIN_FILE = Path("domain_data_flat.json")

# -----------------------------
# Connect to Qdrant (embedded mode)
# -----------------------------
t0 = time.time()
logging.info("Connecting to Qdrant (embedded mode)...")

client = QdrantClient(path=QDRANT_PATH)
collections = client.get_collections().collections
if not any(c.name == COLLECTION_NAME for c in collections):
    logging.error(f"Collection '{COLLECTION_NAME}' does not exist.")
    raise RuntimeError("Qdrant collection not found. Run ingestion first.")

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

logging.info(f"Qdrant ready in {time.time() - t0:.2f}s")

# -----------------------------
# Precompute full-document embeddings
# -----------------------------
logging.info("Preparing document embeddings...")

def build_doc_embeddings(points):
    doc_texts = {}

    for point in points:
        payload = point.payload
        src = payload["source_path"]
        text = payload["text"]

        doc_texts.setdefault(src, []).append(text)

    doc_embeddings = {}

    for src, texts in doc_texts.items():
        full_text = "\n".join(texts)
        emb = embed_model.encode(
            [full_text],
            normalize_embeddings=True
        )[0]
        doc_embeddings[src] = emb

    return doc_embeddings

def load_all_chunks():
    logging.info("Loading chunks from Qdrant (paginated)...")

    offset = None
    all_points = []

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=SCROLL_BATCH_SIZE,
            offset=offset,
            with_payload=True
        )
        if not points:
            break
        all_points.extend(points)
        if offset is None:
            break
    logging.info(f"Loaded {len(all_points)} total chunks.")
    return all_points
# -----------------------------
# Load training QAs once
# -----------------------------
logging.info("Loading training QA data...")
training_data = json.loads(TRAIN_FILE.read_text(encoding="utf-8"))

# Initialize router (JSON lookup + semantic embeddings)
init_qa_router(training_data)

# -----------------------------
# Request schema
# -----------------------------
class Query(BaseModel):
    question: str
    top_k: int = 8
    class Config:
        min_anystr_length = 3

# -----------------------------
# Retrieval (now includes scores)
# -----------------------------
def retrieve(question: str, k: int):

    if k <= 0:
        raise ValueError("top_k must be greater than 0")

    t_start = time.time()
    q_emb = embed_model.encode(
        [question],
        normalize_embeddings=True
    )[0]
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_emb.tolist(),
        limit=k,
        with_payload=True
    )
    final_results = []

    for hit in results:
        payload = hit.payload.copy()
        payload["text"] = payload.get("text", "")
        payload["relevance_score"] = float(hit.score)
        final_results.append(payload)

    logging.info(
        f"Retrieved {len(final_results)} chunks "
        f"in {time.time() - t_start:.2f}s"
    )
    return final_results

# -----------------------------
# Load or Build Document Embeddings
# -----------------------------
if DOC_EMBEDDINGS_CACHE.exists():
    logging.info("Loading cached document embeddings...")
    with open(DOC_EMBEDDINGS_CACHE, "rb") as f:
        DOC_EMBEDDINGS = pickle.load(f)
else:
    logging.info("Cache not found. Loading chunks to build embeddings...")
    points = load_all_chunks()  # <-- THIS WAS MISSING
    DOC_EMBEDDINGS = build_doc_embeddings(points)

    with open(DOC_EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(DOC_EMBEDDINGS, f)

logging.info(f"Document embeddings ready: {len(DOC_EMBEDDINGS)} docs")

def compute_confidence(chunks):
    if not chunks:
        return 0.0

    scores = [c["relevance_score"] for c in chunks]
    return float(np.mean(scores))

# -----------------------------
# API
# -----------------------------
@app.post("/chat")
def chat(q: Query):
    t_total = time.time()
    logging.info("Received /chat request")

    try:
        chunks = retrieve(q.question, q.top_k)
        answer, citations = generate_answer(
            q.question,
            chunks,
            DOC_EMBEDDINGS
        )

        confidence = compute_confidence(chunks)

        logging.info(f"/chat processed in {time.time() - t_total:.2f}s")

        return {
            "answer": answer,
            "citations": citations,
            "confidence_score": confidence
        }

    except Exception as e:
        logging.exception("Error during /chat processing")
        return {
            "answer": "An internal error occurred.",
            "citations": [],
            "confidence_score": 0.0
        }
    
@app.on_event("startup")
def startup_event():
    logging.info("Application startup complete.")

@app.get("/health")
def health():
    return {"status": "ok"}