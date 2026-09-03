"""A small, dependency-free TF-IDF retriever over the local knowledge base.

Deliberately avoids embedding-model or vector-DB dependencies so the whole
project runs offline. Swap this for a real embedding index (e.g. pgvector,
FAISS, Chroma) without touching any caller - the `search(query, k)` contract
is the only thing that matters to the rest of the app.
"""
import math
import os
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class Retriever:
    def __init__(self, kb_dir: str):
        self.kb_dir = kb_dir
        self.chunks: list[dict] = []
        self._load(kb_dir)
        self._build_index()

    def _load(self, kb_dir: str) -> None:
        if not os.path.isdir(kb_dir):
            return
        for fname in sorted(os.listdir(kb_dir)):
            if not fname.endswith(".md"):
                continue
            with open(os.path.join(kb_dir, fname), encoding="utf-8") as f:
                text = f.read()
            paragraphs = (p.strip() for p in text.split("\n\n") if p.strip())
            # Skip markdown headings as their own chunk - a short, high-frequency-
            # normalized title otherwise outranks the substantive paragraph it labels.
            body_paragraphs = [p for p in paragraphs if not p.startswith("#")]
            for i, para in enumerate(body_paragraphs):
                self.chunks.append({"id": f"{fname}#{i}", "source": fname, "text": para})

    def _build_index(self) -> None:
        doc_tokens = [tokenize(c["text"]) for c in self.chunks]
        df: Counter = Counter()
        for toks in doc_tokens:
            for term in set(toks):
                df[term] += 1
        n_docs = len(doc_tokens)
        self.idf = {term: math.log((n_docs + 1) / (freq + 1)) + 1 for term, freq in df.items()}
        self.doc_vectors = [self._vectorize(toks) for toks in doc_tokens]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        vec = {term: (count / len(tokens)) * self.idf.get(term, 0.0) for term, count in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        return sum(v * b.get(term, 0.0) for term, v in a.items())

    def search(self, query: str, k: int = 3) -> list[dict]:
        q_vec = self._vectorize(tokenize(query))
        scored = [(self._cosine(q_vec, dv), chunk) for dv, chunk in zip(self.doc_vectors, self.chunks)]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"source": chunk["source"], "text": chunk["text"], "score": round(score, 4)}
            for score, chunk in scored[:k]
            if score > 0
        ]


_singleton: Retriever | None = None


def get_retriever() -> Retriever:
    global _singleton
    if _singleton is None:
        from copilot.config import KB_DIR

        _singleton = Retriever(KB_DIR)
    return _singleton
