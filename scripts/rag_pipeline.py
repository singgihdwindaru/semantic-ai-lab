import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from embeddings.encoder import TextEncoder
from rag.pipeline import RetrievalAugmentedGenerator
from evaluation.metrics import average_precision_at_k

# Sample document corpus
documents = [
    "Artificial intelligence is the simulation of human intelligence by machines.",
    "These machines are capable of learning and problem-solving.",
    "Applications include language models, recommendation systems, and robotics.",
]

query = "How do machines simulate intelligence?"
relevant_answer = "Artificial intelligence is the simulation of human intelligence by machines."

# Initialize RAG
encoder = TextEncoder()
rag = RetrievalAugmentedGenerator(documents, encoder)

# Run RAG
print("🔍 Query:", query)
answer = rag.generate_answer(query)
print("🧠 Generated Answer:\n", answer)

# Evaluation
query_emb = encoder.encode_single(query)
chunk_embs = encoder.encode(rag.chunks)

# Manually identify which chunk is relevant (e.g. index 0)
relevant_idx = rag.chunks.index(relevant_answer)

# Score
ap_score = average_precision_at_k(query_emb, chunk_embs, [relevant_idx], k=3)
print(f"\n📊 AP@3: {ap_score:.4f}")
