from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator
)

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("json", data_files="domain_data_flat.json")


def normalize(text: str) -> str:
    return " ".join(text.split())


# -----------------------------
# Tokenizer & model
# -----------------------------
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_batch(batch):
    questions = [normalize(q) for q in batch["question"]]
    contexts = [normalize(c) for c in batch["context"]]
    answers_list = batch["answers"]

    encodings = tokenizer(
        questions,
        contexts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True
    )

    input_ids_list = encodings["input_ids"]
    attn_mask_list = encodings["attention_mask"]
    offsets_list = encodings["offset_mapping"]

    start_positions_list = []
    end_positions_list = []
    valid_indices = []

    for i in range(len(questions)):
        ans_objs = answers_list[i]
        if not ans_objs:
            valid_indices.append(False)
            start_positions_list.append(0)
            end_positions_list.append(0)
            continue

        ans_text = normalize(ans_objs[0]["text"])

        # Find new answer_start AFTER normalization
        start_char = contexts[i].find(ans_text)
        if start_char == -1:
            valid_indices.append(False)
            start_positions_list.append(0)
            end_positions_list.append(0)
            continue

        end_char = start_char + len(ans_text)

        offsets = offsets_list[i]
        seq_ids = encodings.sequence_ids(i)

        start_tok = end_tok = None

        for t, s in enumerate(seq_ids):
            if s == 1:
                start, end = offsets[t]
                if start <= start_char < end:
                    start_tok = t
                if start < end_char <= end:
                    end_tok = t

        if start_tok is None or end_tok is None:
            valid_indices.append(False)
            start_positions_list.append(0)
            end_positions_list.append(0)
            continue

        valid_indices.append(True)
        start_positions_list.append(start_tok)
        end_positions_list.append(end_tok)

    filtered = {
        "input_ids": [],
        "attention_mask": [],
        "start_positions": [],
        "end_positions": []
    }

    for i, ok in enumerate(valid_indices):
        if ok:
            filtered["input_ids"].append(input_ids_list[i])
            filtered["attention_mask"].append(attn_mask_list[i])
            filtered["start_positions"].append(start_positions_list[i])
            filtered["end_positions"].append(end_positions_list[i])

    return filtered


tokenized = dataset["train"].map(
    preprocess_batch,
    batched=True,
    remove_columns=dataset["train"].column_names
)

if tokenized.num_rows == 0:
    raise RuntimeError("No valid training rows after alignment.")

# -----------------------------
# Training
# -----------------------------
args = TrainingArguments(
    output_dir="./distilbert-finetuned",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    save_steps=50,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

trainer.train()
model.save_pretrained("./distilbert-finetuned")
tokenizer.save_pretrained("./distilbert-finetuned")

print("Training complete.")
