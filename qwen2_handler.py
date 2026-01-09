from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype=torch.float32
)

model.eval()

def qwen2_check_relevance(context: str, question: str) -> str:
    """
    Returns 'RELEVANT' if the question is about the content in context,
    otherwise returns 'NOT RELEVANT'.
    """
    prompt = f"""
You are a strict relevance classifier.
Decide if the QUESTION is about the CONTENT provided in Context.

Rules:
- Reply with exactly one word: RELEVANT or NOT RELEVANT.
- If the question cannot be answered using the context, reply NOT RELEVANT.
- Do not explain.

Context:
{context}

Question:
{question}

Answer (RELEVANT or NOT RELEVANT):
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=4,
            temperature=0.0,
            do_sample=False
        )
    verdict = tokenizer.decode(output[0], skip_special_tokens=True)\
        .split("Answer")[-1].strip()
    verdict = verdict.strip().split()[-1].upper()
    if verdict not in {"RELEVANT", "NOT", "NOTRELEVANT", "NOT_RELEVANT"}:
        verdict = "RELEVANT" if any(tok in context.lower() for tok in question.lower().split()) else "NOT RELEVANT"
    if verdict in {"NOT", "NOTRELEVANT", "NOT_RELEVANT"}:
        verdict = "NOT RELEVANT"
    return verdict

def qwen2_answer(context: str, question: str):
    """
    Strictly grounded generation:
    - If the question is not relevant to the context, return 'Not relevant to the attached documents'.
    - If relevant but the answer is not found in the context, return 'Not mentioned in the document'.
    - Otherwise, answer using only the context.
    """
    rel = qwen2_check_relevance(context=context, question=question).upper()
    if rel != "RELEVANT":
        return "Not relevant to the attached documents"

    prompt = f"""
You are a document-based assistant.
You must answer ONLY using the provided context.
If the answer is not in the context, you MUST reply exactly with:
Not mentioned in the document.

Do not use outside knowledge.
Do not guess.
Do not invent.

Context:
{context}

Question:
{question}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.0,
            do_sample=False
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)\
        .split("Answer:")[-1].strip()

    # Guardrail: if answer not in context, force fallback
    if answer and answer.lower() != "not mentioned in the document":
        if answer not in context:
            return "Not mentioned in the document"

    if not answer:
        return "Not mentioned in the document"

    return answer
