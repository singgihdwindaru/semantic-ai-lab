from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SimilarityRanker:
    def __init__(self):
        pass

    def rank(
        self, query_embedding: List[float], candidate_embeddings: List[List[float]], candidates: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Return candidates sorted by similarity to query_embedding.
        """
        scores = cosine_similarity([query_embedding], candidate_embeddings)[0]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked
