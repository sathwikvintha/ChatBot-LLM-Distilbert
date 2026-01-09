import json, time, logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss

from generate import generate_answer, init_qa_router

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
INDEX_DIR = Path("data/index")
CHUNK_DIR = Path("data/chunks")
TRAIN_FILE = Path("domain_data_flat.json")

# -----------------------------
# Load FAISS + embeddings
# -----------------------------
t0 = time.time()
logging.info("Loading FAISS index and metadata...")

index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
metas = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

logging.info(f"FAISS loaded in {time.time() - t0:.2f}s")

# -----------------------------
# Precompute full-document embeddings
# -----------------------------
logging.info("Precomputing document embeddings...")

def get_chunk_text(chunk_id):
    for jf in CHUNK_DIR.glob("*.jsonl"):
        for line in jf.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            if obj.get("chunk_id") == chunk_id:
                return obj.get("text", "")
    return ""

# Group chunks by document
doc_texts = {}
for meta in metas:
    src = meta["source_path"]
    text = get_chunk_text(meta["chunk_id"])
    if text:
        if src not in doc_texts:
            doc_texts[src] = []
        doc_texts[src].append(text)

# Embed full documents
DOC_EMBEDDINGS = {}
for src, texts in doc_texts.items():
    full_text = "\n".join(texts)
    emb = embed_model.encode([full_text], normalize_embeddings=True)[0]
    DOC_EMBEDDINGS[src] = emb

logging.info(f"Precomputed embeddings for {len(DOC_EMBEDDINGS)} documents")

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

# -----------------------------
# Retrieval (now includes scores)
# -----------------------------
def retrieve(question, k):
    t_start = time.time()
    q_emb = embed_model.encode([question], normalize_embeddings=True)
    scores, idxs = index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        meta = metas[idx].copy()
        meta["text"] = get_chunk_text(meta["chunk_id"])
        meta["relevance_score"] = float(score)
        if meta["text"]:
            results.append(meta)

    logging.info(f"Retrieved {len(results)} chunks in {time.time() - t_start:.2f}s")
    return results

# -----------------------------
# API
# -----------------------------
@app.post("/chat")
def chat(q: Query):
    t_total = time.time()
    logging.info("Received /chat request")

    chunks = retrieve(q.question, q.top_k)
    answer, citations = generate_answer(q.question, chunks, DOC_EMBEDDINGS)

    logging.info(f"/chat processed in {time.time() - t_total:.2f}s")
    return {"answer": answer, "citations": citations}