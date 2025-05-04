from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DenseRetriever:
    def __init__(self, embeddings: List[List[float]], documents: List[str]):
        """
        :param embeddings: Precomputed vector representations of documents
        :param documents: Corresponding raw text chunks
        """
        self.embeddings = embeddings
        self.documents = documents

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """
        Return top-k most similar documents to the query_embedding.
        """
        scores = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
