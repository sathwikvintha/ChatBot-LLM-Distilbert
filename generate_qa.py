import json
import os
import time
import sys
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

# ---------------- CONFIG ----------------

CHUNKS_DIR = Path("data/chunks")
OUTPUT_FILE = Path("domain_data_flat.json")

QUESTIONS_PER_CHUNK = 20      # change to 15 if needed
MAX_QAS_PER_CALL = 8
MAX_CALLS_PER_CHUNK = 4

MODEL_NAME = "qwen/qwen-2.5-7b-instruct"
# Alternative:
# MODEL_NAME = "deepseek/deepseek-chat"

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

SLEEP_BETWEEN_CALLS = 1

# --------------------------------------


# -------------------------------------------------
# Loaders
# -------------------------------------------------

def load_chunks():
    chunks = []
    for jsonl_file in CHUNKS_DIR.glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def load_existing_qas():
    if not OUTPUT_FILE.exists():
        return []

    content = OUTPUT_FILE.read_text(encoding="utf-8").strip()
    if not content:
        return []

    try:
        data = json.loads(content)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def group_qas_by_chunk(qas):
    grouped = defaultdict(list)
    for qa in qas:
        grouped[qa["chunk_id"]].append(qa)
    return grouped


# -------------------------------------------------
# LLM logic
# -------------------------------------------------

def build_prompt(chunk_text, remaining):
    return f"""
You are generating training data for an extractive
question-answering model (DistilBERT style).

Context:
\"\"\"{chunk_text}\"\"\"

TASK:
Generate up to {remaining} DISTINCT question-answer pairs.

STRICT RULES:
- Questions must be answerable ONLY from the context
- Answers must be exact substrings from the context
- Each answer MUST include answer_start
- No explanations
- No markdown
- Output ONLY valid JSON ARRAY

OUTPUT FORMAT:
[
  {{
    "question": "...",
    "answers": [
      {{
        "text": "...",
        "answer_start": 0
      }}
    ]
  }}
]
""".strip()


def call_llm(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a precise dataset generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    text = response.choices[0].message.content
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        return []

    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def generate_questions_for_chunk(chunk, existing_qas):
    collected = existing_qas.copy()
    seen_questions = {qa["question"] for qa in collected}

    calls = 0
    total = QUESTIONS_PER_CHUNK

    while len(collected) < total and calls < MAX_CALLS_PER_CHUNK:
        remaining = min(MAX_QAS_PER_CALL, total - len(collected))
        calls += 1

        prompt = build_prompt(chunk["text"], remaining)
        new_qas = call_llm(prompt)

        for qa in new_qas:
            question = qa.get("question")
            answers = qa.get("answers")

            if not question or not answers:
                continue
            if question in seen_questions:
                continue

            seen_questions.add(question)
            collected.append({
                "id": f"{chunk['chunk_id']}_q{len(collected) + 1}",
                "chunk_id": chunk["chunk_id"],
                "question": question,
                "context": chunk["text"],
                "answers": answers
            })

            # 🔥 LIVE COUNTER UPDATE
            sys.stdout.write(
                f"\r🧠 {chunk['chunk_id']} → {len(collected)}/{total}"
            )
            sys.stdout.flush()

        time.sleep(SLEEP_BETWEEN_CALLS)

    print()  # newline after finishing chunk
    return collected


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    print("🔹 Loading chunks...")
    chunks = load_chunks()
    print(f"🔹 Loaded {len(chunks)} chunks")

    print("🔹 Loading existing questions...")
    existing_qas = load_existing_qas()
    grouped_qas = group_qas_by_chunk(existing_qas)

    new_dataset = existing_qas.copy()
    added_count = 0

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        existing_for_chunk = grouped_qas.get(chunk_id, [])

        if len(existing_for_chunk) >= QUESTIONS_PER_CHUNK:
            print(f"⏭️  Skipping {chunk_id} (already {len(existing_for_chunk)}/{QUESTIONS_PER_CHUNK})")
            continue

        print(f"🧠 Processing {chunk_id} ({len(existing_for_chunk)}/{QUESTIONS_PER_CHUNK})")

        updated_qas = generate_questions_for_chunk(chunk, existing_for_chunk)

        # Remove old entries for this chunk
        new_dataset = [
            qa for qa in new_dataset if qa["chunk_id"] != chunk_id
        ]

        # Add updated QAs
        new_dataset.extend(updated_qas)
        added_count += max(0, len(updated_qas) - len(existing_for_chunk))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, indent=2)

    print("\n✅ Added", added_count, "new questions")
    print("✅ Total questions now:", len(new_dataset))


if __name__ == "__main__":
    main()
