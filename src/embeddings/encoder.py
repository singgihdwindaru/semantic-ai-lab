from typing import List
from sentence_transformers import SentenceTransformer

class TextEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of text strings into embedding vectors.
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def encode_single(self, text: str) -> List[float]:
        """
        Encode a single text string.
        """
        return self.encode([text])[0]
