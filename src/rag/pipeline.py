from typing import List, Callable
from embeddings.encoder import TextEncoder
from chunking.splitter import sentence_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class RetrievalAugmentedGenerator:
    def __init__(
        self,
        documents: List[str],
        encoder: TextEncoder,
        chunking_fn: Callable[[str], List[str]] = sentence_split,
    ):
        self.encoder = encoder
        self.chunking_fn = chunking_fn
        self.chunks = [chunk for doc in documents for chunk in chunking_fn(doc)]
        self.chunk_embeddings = self.encoder.encode(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_emb = self.encoder.encode_single(query)
        scores = cosine_similarity([query_emb], self.chunk_embeddings)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]

    def generate_answer(self, query: str, top_k: int = 3) -> str:
        """
        Dummy answer generator — concatenate top-k chunks.
        """
        retrieved_chunks = self.retrieve(query, top_k=top_k)
        return " ".join(retrieved_chunks)
