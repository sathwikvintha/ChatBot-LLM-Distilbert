import torch, re, os
from sentence_transformers import SentenceTransformer
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast

# -----------------------------
# MODELS
# -----------------------------
qa_model = DistilBertForQuestionAnswering.from_pretrained("./distilbert-finetuned")
qa_tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert-finetuned")

device = "cuda" if torch.cuda.is_available() else "cpu"
qa_model.to(device).eval()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# TRAINING QA STORAGE
# -----------------------------
QUESTION_LOOKUP = {}
TRAIN_QUESTIONS = []
TRAIN_ANSWERS = []
TRAIN_EMBEDDINGS = None

def init_qa_router(training_data):
    global QUESTION_LOOKUP, TRAIN_QUESTIONS, TRAIN_ANSWERS, TRAIN_EMBEDDINGS

    QUESTION_LOOKUP.clear()
    TRAIN_QUESTIONS.clear()
    TRAIN_ANSWERS.clear()

    for q in training_data:
        QUESTION_LOOKUP[q["question"].lower()] = q["answers"][0]["text"]
        TRAIN_QUESTIONS.append(q["question"])
        TRAIN_ANSWERS.append(q["answers"][0]["text"])

    TRAIN_EMBEDDINGS = embed_model.encode(
        TRAIN_QUESTIONS,
        normalize_embeddings=True
    )

# -----------------------------
# HELPERS
# -----------------------------
FACTUAL = ["what is", "which", "where", "url", "token"]
HOW = ["how", "setup", "steps", "process", "guide"]
SEQUENCE = ["final", "last", "after", "verification"]

STOPWORDS = {
    "what", "where", "can", "you", "to", "for", "in", "the",
    "is", "are", "of", "on", "and", "a", "an"
}

def tokenize(text):
    return [
        w for w in re.findall(r"\w+", text.lower())
        if w not in STOPWORDS and len(w) > 2
    ]


def progressive_match_score(user_q, train_q):
    """
    Progressive keyword overlap score.
    """
    user_tokens = tokenize(user_q)
    train_tokens = set(tokenize(train_q))

    if not user_tokens or not train_tokens:
        return 0.0, 0

    matched = 0
    for t in user_tokens:
        if t in train_tokens:
            matched += 1

    score = matched / len(user_tokens)
    return score, matched


def dedupe_chunks(chunks):
    """
    Deduplicate chunks by source_path (UI-compatible).
    """
    seen = set()
    deduped = []

    for c in chunks:
        src = c.get("source_path")
        if src and src not in seen:
            deduped.append(c)
            seen.add(src)

    return deduped


def qa_extract(question, context):
    enc = qa_tokenizer(
        question,
        context,
        return_overflowing_tokens=True,
        max_length=512,
        stride=128,
        truncation=True,
        padding="max_length"
    )

    best, score = "", -1e9

    for i in range(len(enc["input_ids"])):
        ids = torch.tensor([enc["input_ids"][i]]).to(device)
        mask = torch.tensor([enc["attention_mask"][i]]).to(device)

        with torch.no_grad():
            out = qa_model(ids, attention_mask=mask)

        s = out.start_logits[0].argmax().item()
        e = out.end_logits[0].argmax().item()
        if e < s:
            continue

        ans = qa_tokenizer.decode(
            ids[0][s:e + 1],
            skip_special_tokens=True
        ).strip()

        sc = (out.start_logits[0][s] + out.end_logits[0][e]).item()

        if sc > score and ans:
            best, score = ans, sc

    return best


def extract_last_sentence(chunks):
    sentences = []
    for c in chunks:
        sentences.extend(
            re.split(r'(?<=[.!?])\s+', c["text"])
        )
    return sentences[-1].strip() if sentences else ""


def extract_steps(chunks):
    steps = []
    for c in chunks:
        for s in re.split(r'(?<=[.!?])\s+', c["text"]):
            s_clean = s.strip()
            if any(v in s_clean.lower() for v in ["go to", "click", "open", "run", "check"]):
                if len(s_clean) > 10 and s_clean not in steps:
                    steps.append(s_clean)
    return steps

# -----------------------------
# MAIN ROUTER
# -----------------------------
def generate_answer(question, chunks):
    q = question.lower()
    citations = dedupe_chunks(chunks)

    # 1️⃣ EXACT MATCH (JSON)
    if q in QUESTION_LOOKUP:
        return QUESTION_LOOKUP[q], citations

    # 2️⃣ SEMANTIC MATCH (JSON)
    if TRAIN_EMBEDDINGS is not None and len(TRAIN_EMBEDDINGS) > 0:
        q_emb = embed_model.encode([question], normalize_embeddings=True)
        sims = (q_emb @ TRAIN_EMBEDDINGS.T)[0]
        best = sims.argmax()

        if sims[best] > 0.85:
            return TRAIN_ANSWERS[best], citations

    # 3️⃣ PROGRESSIVE KEYWORD MATCH (JSON)
    best_score = 0.0
    best_idx = None
    best_matched = 0

    for i, tq in enumerate(TRAIN_QUESTIONS):
        score, matched = progressive_match_score(question, tq)
        if score > best_score:
            best_score = score
            best_idx = i
            best_matched = matched

    # Require at least 2 keyword matches to avoid false positives
    if best_score >= 0.35 and best_matched >= 2:
        return TRAIN_ANSWERS[best_idx], citations

    # 4️⃣ SEQUENCE QUESTIONS
    if any(w in q for w in SEQUENCE):
        answer = extract_last_sentence(chunks)
        if answer:
            return answer, citations

    # 5️⃣ HOW / SETUP QUESTIONS
    if any(w in q for w in HOW):
        steps = extract_steps(chunks)
        if steps:
            return "\n".join(f"- {s}" for s in steps), citations

    # 6️⃣ FACTUAL QA FALLBACK
    combined = " ".join(c["text"] for c in chunks)
    ans = qa_extract(question, combined)
    if ans:
        return ans, citations

    return "I could not find an answer in the documents.", citations