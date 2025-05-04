# 🧠 Semantic AI Lab — RAG Prototype

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using sentence chunking, text embeddings, and cosine similarity.

---

## 📁 Project Structure

semantic-ai-lab/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── sample_texts/         # Example documents to embed or retrieve
│
├── notebooks/
│   ├── 01_embedding_vs_chunking.ipynb
│   ├── 02_cosine_similarity_exploration.ipynb
│   ├── 03_vector_search_faiss.ipynb
│   └── 04_rag_pipeline_intro.ipynb
│
├── src/
│   ├── embeddings/
│   │   ├── encoder.py        # Wraps SentenceTransformer, OpenAI, etc.
│   │   └── utils.py
│   │
│   ├── chunking/
│   │   └── splitter.py       # Sentence, paragraph, token-based splitters
│   │
│   ├── retrieval/
│   │   ├── search.py         # FAISS, Pinecone, etc.
│   │   └── ranker.py         # Optional reranking logic
│   │
│   ├── rag/
│   │   └── pipeline.py       # Basic RAG pipeline (embed → retrieve → generate)
│   │
│   └── evaluation/
│       └── metrics.py        # Similarity, precision@k, recall, etc.
│
└── tests/
    └── test_embeddings.py    # Unit tests


---

## ⚙️ Setup (with [`uv`](https://github.com/astral-sh/uv))

### 1. Install `uv`

Install using shell script:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via Homebrew (macOS):

```bash
brew install astral-sh/astral/uv
```

---

### 2. Create Virtual Environment + Install Dependencies

```bash
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
# or
uv sync
```

> ✅ This will:
>
> * Create a `.venv` folder
> * Activate the virtual environment
> * Install packages using `uv pip` (significantly faster than `pip`)

---

### 3. Run Examples

Now you're ready to run scripts or notebooks:

```bash
uv run python scripts/compare_embedding_similarity.py
```

or

```bash
uv run jupyter notebook notebooks/02_rag_pipeline_and_eval.ipynb
```

---
