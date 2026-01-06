import json

def flatten_squad(input_file="domain_data.json", output_file="domain_data_flat.json"):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                entry = {
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answers": qa["answers"]  # already has text + answer_start
                }
                flat.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(flat, f, indent=2, ensure_ascii=False)

    print(f"✅ Flattened {len(flat)} QA pairs into {output_file}")

if __name__ == "__main__":
    flatten_squad()
