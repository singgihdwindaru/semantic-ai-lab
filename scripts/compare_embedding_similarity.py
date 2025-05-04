import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from embeddings.encoder import TextEncoder
from chunking.splitter import sentence_split
from sklearn.metrics.pairwise import cosine_similarity


def cosine(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]


def main():
    text = """
    Artificial intelligence is the simulation of human intelligence in machines.
    These machines are programmed to think like humans and mimic their actions.
    AI has many applications in today's society including search engines, chatbots, and self-driving cars.
    """

    query = "What is artificial intelligence?"

    encoder = TextEncoder()
    chunks = sentence_split(text)
    full_emb = encoder.encode_single(text)
    chunk_embs = encoder.encode(chunks)
    query_emb = encoder.encode_single(query)

    print("🔵 Full Text Similarity:")
    print(f"   {cosine(query_emb, full_emb):.4f}\n")

    print("🟢 Chunk Similarities:")
    for i, (chunk, chunk_emb) in enumerate(zip(chunks, chunk_embs)):
        score = cosine(query_emb, chunk_emb)
        print(f"   [{i}] {score:.4f} - {chunk.strip()}")

    best_score = max([cosine(query_emb, e) for e in chunk_embs])
    print(f"\n⭐ Best chunk score: {best_score:.4f}")


if __name__ == "__main__":
    main()
