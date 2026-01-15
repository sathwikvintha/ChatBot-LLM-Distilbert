import torch, re, os, logging
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
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

    for q in training_data:  # ✅ FIXED SYNTAX ERROR
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
HOW = ["how", "setup", "steps", "process", "guide", "configure", "install"]
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
    user_tokens = tokenize(user_q)
    train_tokens = set(tokenize(train_q))
    if not user_tokens or not train_tokens:
        return 0.0, 0
    matched = sum(1 for t in user_tokens if t in train_tokens)
    score = matched / len(user_tokens)
    return score, matched

def dedupe_chunks(chunks):
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
        ans = qa_tokenizer.decode(ids[0][s:e + 1], skip_special_tokens=True).strip()
        sc = (out.start_logits[0][s] + out.end_logits[0][e]).item()
        if sc > score and ans:
            best, score = ans, sc
    return best

def extract_last_sentence(chunks):
    sentences = []
    for c in chunks:
        sentences.extend(re.split(r'(?<=[.!?])\s+', c["text"]))
    return sentences[-1].strip() if sentences else ""

def extract_steps(chunks):
    steps = []
    step_indicators = ["go to", "click", "open", "run", "check", "login", "enter", "replace", "download", "set up", "setup"]
    for c in chunks:
        text = c["text"]
        sentences = re.split(r'[\n•\-]|\.\s+(?=[A-Z])', text)
        for s in sentences:
            s_clean = s.strip().rstrip('.').strip()
            if len(s_clean) < 10:
                continue
            s_lower = s_clean.lower()
            if any(ind in s_lower for ind in step_indicators):
                if s_clean not in steps:
                    steps.append(s_clean)
    return steps

# -----------------------------
# QWEN INTEGRATION
# -----------------------------
_qwen_loaded = False
def _ensure_qwen():
    global _qwen_loaded, qwen2_answer, qwen2_check_relevance
    if not _qwen_loaded:
        from qwen2_handler import qwen2_answer, qwen2_check_relevance
        _qwen_loaded = True

def qwen_generate_answer(question, chunks):
    """Generate answer using the retrieved chunks."""
    _ensure_qwen()
    context_text = "\n\n".join(c["text"] for c in chunks)
    if not context_text.strip():
        return "I could not find an answer in the documents."
    
    # Use a directive prompt to enforce grounding
    answer = qwen2_answer(
        context=context_text,
        question=(
            f"Answer the following question using ONLY the information above.\n"
            f"Explain the answer clearly in your own words, using a concise but helpful tone.\n"
            f"If the question asks for steps, provide them in a logical step-by-step manner.\n"
            f"Do NOT add any information that is not explicitly present in the context.\n"
            f"Do NOT assume or invent details.\n"
            f"If the answer is not present, respond exactly: "
            f"'I could not find an answer in the documents.'\n\n"
            f"Question: {question}"
        )
    )
    return answer

# -----------------------------
# PROGRESSION LOGGER
# -----------------------------
def _log_progress(progression):
    logging.info("---- Layer Progression ----")
    for layer, status in progression.items():
        logging.info(f"{layer} {status}")
    logging.info("---------------------------")

# -----------------------------
# MAIN ROUTER
# -----------------------------
# Modify generate_answer signature
def generate_answer(question, chunks, doc_embeddings=None):
    q = question.lower()

    # 🔑 NEW: Semantic document selection
    def select_best_document_chunks(chunks, question, doc_embeddings):
        if not chunks or not doc_embeddings:
            return chunks
        
        # Get question embedding
        q_emb = embed_model.encode([question], normalize_embeddings=True)[0]
        
        # Consider only docs that appear in retrieved chunks
        candidate_docs = set(
            c["source_path"] for c in chunks 
            if c.get("source_path") in doc_embeddings
        )
        if not candidate_docs:
            return chunks

        # Find most similar document
        best_doc, best_sim = None, -1
        for doc in candidate_docs:
            sim = util.cos_sim(q_emb, doc_embeddings[doc]).item()
            if sim > best_sim:
                best_sim, best_doc = sim, doc

        return [c for c in chunks if c["source_path"] == best_doc]

    focused_chunks = select_best_document_chunks(chunks, question, doc_embeddings)
    citations = dedupe_chunks(focused_chunks)

    # ... rest of your logic unchanged (use focused_chunks everywhere)
    # (I've kept your existing layers below for completeness)
    
    progression = {
        "Layer 1: Exact Match": "❌",
        "Layer 2: Semantic Match": "❌",
        "Layer 3: Keyword Match (gated)": "❌",
        "Layer 4: Steps Fallback": "❌",
        "Layer 5: Sequence": "❌",
        "Layer 6: Qwen": "❌",
        "Layer 7: DistilBERT": "❌"
    }

    if q in QUESTION_LOOKUP:
        progression["Layer 1: Exact Match"] = "✅"
        _log_progress(progression)
        return QUESTION_LOOKUP[q], citations

    if TRAIN_EMBEDDINGS is not None and len(TRAIN_EMBEDDINGS) > 0:
        q_emb = embed_model.encode([question], normalize_embeddings=True)
        sims = (q_emb @ TRAIN_EMBEDDINGS.T)[0]
        best = sims.argmax()
        if sims[best] > 0.85:
            progression["Layer 2: Semantic Match"] = "✅"
            _log_progress(progression)
            return TRAIN_ANSWERS[best], citations

    _ensure_qwen()
    context_text = "\n\n".join(c.get("text", "") for c in focused_chunks)
    if context_text.strip():
        verdict = qwen2_check_relevance(context=context_text, question=question)
        if verdict.strip().upper() == "RELEVANT":
            best_score, best_idx, best_matched = 0.0, None, 0
            candidate_answers = []
            for i, tq in enumerate(TRAIN_QUESTIONS):
                score, matched = progressive_match_score(question, tq)
                if matched >= 2 and score >= 0.35:
                    candidate_answers.append((score, matched, i, tq))
            
            if candidate_answers:
                candidate_answers.sort(key=lambda x: (x[0], x[1]), reverse=True)
                best_score, best_matched, best_idx, best_tq = candidate_answers[0]
                
                answer_text = TRAIN_ANSWERS[best_idx]
                is_how_question = any(w in q for w in HOW)
                is_factual_answer = any(f in answer_text.lower() for f in ["http", ".zip", "=", "true", "false", "sqp_", "mvn "])
                
                if is_how_question and is_factual_answer:
                    pass
                else:
                    progression["Layer 3: Keyword Match (gated)"] = "✅"
                    _log_progress(progression)
                    return answer_text, citations

    if any(w in q for w in SEQUENCE):
        answer = extract_last_sentence(focused_chunks)
        if answer:
            progression["Layer 5: Sequence"] = "✅"
            _log_progress(progression)
            return answer, citations

    is_procedural = any(w in q for w in HOW)
    if is_procedural:
        steps = extract_steps(focused_chunks)
        if steps:
            progression["Layer 4: Steps Fallback"] = "✅"
            _log_progress(progression)
            return "\n".join(f"- {s}" for s in steps), citations

    qwen_ans = qwen_generate_answer(question, focused_chunks)
    if qwen_ans:
        progression["Layer 6: Qwen"] = "✅"
        _log_progress(progression)
        if "could not find an answer" in qwen_ans or "not mentioned" in qwen_ans.lower():
            pass
        else:
            return qwen_ans, citations

    combined = " ".join(c["text"] for c in focused_chunks)
    ans = qa_extract(question, combined)
    if ans:
        progression["Layer 7: DistilBERT"] = "✅"
        _log_progress(progression)
        return ans, citations

    _log_progress(progression)
    return "I could not find an answer in the documents.", []