from typing import List
import re
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

def sentence_split(text: str) -> List[str]:
    """
    Split text into sentences using NLTK's sentence tokenizer.
    """
    return sent_tokenize(text)

def paragraph_split(text: str) -> List[str]:
    """
    Split text into paragraphs (double line break).
    """
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def fixed_token_split(text: str, max_tokens: int = 50) -> List[str]:
    """
    Naive fixed-length chunking by whitespace tokens.
    (Does not count model-specific tokens)
    """
    words = text.split()
    return [
        " ".join(words[i : i + max_tokens])
        for i in range(0, len(words), max_tokens)
    ]
