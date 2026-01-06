# Domain-Specific QA Chatbot (Internal Knowledge Library)

This project implements a **domain-specific Question Answering (QA) chatbot** designed to act as a **searchable internal knowledge library**.
Users can ask questions in simple words and get accurate, document-backed answers with citations.

The solution is built to be **KT-friendly**, so that any new joiner can understand the flow, add documents, and use the system with minimal dependency.

---

## 🧠 What This Application Does

- Ingests internal documents (Word / PDF / Text)
- Splits documents into searchable chunks
- Creates semantic embeddings and indexes them using FAISS
- Answers user questions using a layered QA approach
- Always provides document-level citations
- Supports multiple documents and domains
- Includes a UI for easy interaction

---

## 📁 Project Structure

```text
chatbot_distilbert/
│
├── app.py                      # FastAPI backend
├── generate.py                 # Answer routing & QA logic
├── ingest.py                   # Document ingestion & indexing
├── train_distilbert.py         # Fine-tuning DistilBERT
├── flatten.py                  # Converts domain_data.json → flat format
│
├── domain_data.json            # Structured domain Q&A data
├── domain_data_flat.json       # Flattened Q&A data used for training
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── ingest.log                  # Ingestion logs
│
├── distilbert-finetuned/       # Trained QA model (local only, not pushed to Git)
│
├── data/
│   ├── source/                 # Place input documents here
│   ├── processed/              # Cleaned text output
│   ├── chunks/                 # Chunked document data
│   ├── embeddings/             # Generated embeddings
│   └── index/                  # FAISS index + metadata
│
└── org-docs-chatbot/            # UI (Frontend application)
```

---

## 🧩 Prerequisites

- Python **3.9+**
- Git
- Virtual environment support

---

## ⚙️ Environment Setup

### 1️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / Mac**

```bash
source .venv/bin/activate
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt` includes:

```text
fastapi
uvicorn
torch
transformers
sentence-transformers
faiss-cpu
pydantic
numpy
pdfplumber
python-docx
datasets
```

---

## 📄 Adding Documents

1. Place all domain documents inside:

   ```
   data/source/
   ```

2. Supported formats:

   - `.docx`
   - `.pdf`
   - `.txt`

3. You can add **multiple documents** at the same time.

---

## ⚙️ Running Document Ingestion

Run the ingestion pipeline:

```bash
python ingest.py
```

### What happens during ingestion:

- Documents are read from `data/source/`
- Text is extracted and cleaned
- Content is split into meaningful chunks
- Embeddings are generated
- FAISS index and metadata are created

After this step, the system is ready to answer questions from the documents.

---

## 🧠 Domain Q&A Data (Training Data)

### Files involved:

- `domain_data.json` → structured Q&A format
- `domain_data_flat.json` → flattened format used for training

If you update `domain_data.json`, run:

```bash
python flatten.py
```

This regenerates `domain_data_flat.json`.

---

## 🧪 Training the QA Model

Train / retrain the DistilBERT QA model using curated domain Q&A data:

```bash
python train_distilbert.py
```

### Output:

- Trained model is saved to:

  ```
  distilbert-finetuned/
  ```

> ⚠️ Note:
>
> - Trained models are **not committed** to GitHub.
> - They are generated locally as required.

---

## 🚀 Running the Backend (FastAPI)

Start the backend server:

```bash
uvicorn app:app --reload --port 8000
```

Backend will be available at:

```
http://127.0.0.1:8000
```

### Main API Endpoint

```http
POST /chat
```

Example request:

```json
{
  "question": "How to setup EXT API Sonar?",
  "top_k": 8
}
```

Example response:

```json
{
  "answer": "...",
  "confidence": "medium",
  "source_type": "semantic",
  "citations": [
    {
      "source_path": "data/source/extapi_sonar_setup.docx"
    }
  ]
}
```

---

## 🖥️ Running the UI (Frontend)

The UI is available under:

```
org-docs-chatbot/
```

### Steps:

1. Navigate to the UI folder:

   ```bash
   cd org-docs-chatbot
   ```

2. Install frontend dependencies (as per UI setup)
3. Start the UI application
4. UI communicates with the FastAPI backend via `/chat` API

---

## 📘 Knowledge Transfer (KT Notes)

For new joiners:

- Start with `README.md`
- Understand `ingest.py` → document flow
- Understand `generate.py` → answer logic
- Always run ingestion after adding new documents
- Use the system as a **searchable help library**

---

## 🔮 Future Enhancements

- Better handling of unseen questions
- Controlled answer generation for low-confidence queries
- Logging unanswered questions for improvement
- Performance tuning for large document sets
- Enhanced UI experience

---

## ✅ Best Practices

- ❌ Do not commit trained models or embeddings
- ❌ Do not push FAISS index to GitHub
- ✅ Always re-run ingestion after adding documents
- ✅ Keep domain Q&A data reviewed and clean

---

## 👤 Maintainer

**Sathwik Vintha**

---

This project is intended to simplify access to internal knowledge and improve productivity by providing a centralized, searchable reference system.
