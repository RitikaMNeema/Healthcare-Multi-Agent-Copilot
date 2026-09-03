"""Real sentence embeddings for the vector half of hybrid retrieval.

Uses a small local Sentence-Transformers model (~90MB) - no API key, no
per-query cost, deterministic given fixed weights. The model is downloaded
from the Hugging Face Hub once and cached under `~/.cache/huggingface`;
every run after that is fully offline. This is a real, if modest, embedding
model - not a hash-based stand-in - so cosine similarity between its vectors
reflects actual learned semantics, not lexical overlap by another name.
"""
import os

DEFAULT_MODEL_NAME = os.environ.get("COPILOT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        get_dim = getattr(self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension)
        self.dimensions = get_dim()

    def encode(self, texts: list[str]):
        """Returns L2-normalized float32 embeddings, shape (len(texts), dimensions).
        Normalized so a FAISS inner-product index computes cosine similarity."""
        import numpy as np

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms


_singleton: Embedder | None = None


def get_embedder() -> Embedder:
    global _singleton
    if _singleton is None:
        _singleton = Embedder()
    return _singleton
