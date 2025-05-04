from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def cosine_score(vec1, vec2) -> float:
    return cosine_similarity([vec1], [vec2])[0][0]


def top_k_accuracy(
    query_embedding, chunk_embeddings: List[List[float]], relevant_idx: int, k: int = 3
) -> bool:
    """
    Check if the relevant chunk is in the top-k most similar chunks.
    """
    scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
    top_indices = np.argsort(scores)[-k:]
    return relevant_idx in top_indices


def average_precision_at_k(
    query_embedding, chunk_embeddings: List[List[float]], relevant_indices: List[int], k: int = 5
) -> float:
    """
    Compute average precision at k (AP@k).
    """
    scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:k]

    hits = 0
    score = 0.0
    for i, idx in enumerate(top_indices):
        if idx in relevant_indices:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant_indices), k)
