"""A hybrid retriever over the local knowledge base: BM25 + real embeddings.

Two independent signals feed candidate generation, then a fused score reranks:

  1. BM25 (Okapi) over exact tokens - precise, and the primary channel: any
     document with a real term hit is a candidate.
  2. Real sentence embeddings (Sentence-Transformers) indexed in FAISS,
     cosine similarity via inner product on normalized vectors - a genuine
     semantic signal (paraphrase, synonymy) that BM25's exact-token matching
     misses, used as a secondary recall path (past a threshold) rather than
     folded into a full rank-fusion score, so it can surface a paraphrase
     BM25 missed entirely without being trusted enough to bury a document
     BM25 was confident about.

An earlier version of this used a hashed character-trigram "vector" instead
of real embeddings - a dependency-free stand-in that avoided any model
download, but was noisy enough (see `eval/retrieval_benchmark.py`) that it
had to be capped at 35% fusion weight and gated behind conservative
thresholds tuned to its specific noise floor. Real embeddings measurably
improve retrieval quality (run the benchmark to see the numbers) and don't
need those same guards, but the candidate-gating architecture - BM25 decides
who's in the running, the vector channel can only add or reorder, never
veto - is kept because it's the right design independent of vector quality:
a semantic signal should widen recall, not override a confirmed lexical hit.

FAISS does exact search (`IndexFlatIP`) here because the corpus is a few
dozen chunks; swap in an IVF/HNSW index without changing `search()`'s
contract (`{"source", "text", "score"}`) if the corpus grows to a scale
where exact search stops being instant.
"""
import math
import os
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
BM25_WEIGHT = 0.65
VECTOR_WEIGHT = 0.35
VECTOR_CANDIDATE_THRESHOLD = 0.45
RELATIVE_SCORE_CUTOFF = 0.6
# A tiny, uncurated policy corpus can give a common function word a spuriously
# high IDF just by chance (e.g. "does" appearing in only one paragraph) -
# without this filter, stopwords in the query can outweigh the actual content
# terms. Skipped only when *scoring* BM25, not when building the vector index.
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "and", "or", "but",
    "if", "then", "else", "of", "to", "in", "on", "at", "for", "with", "by", "from", "as",
    "that", "this", "these", "those", "it", "its", "i", "you", "he", "she", "we", "they",
    "what", "which", "who", "whom", "how", "why", "when", "where", "do", "does", "did",
    "doing", "have", "has", "had", "having", "can", "could", "will", "would", "shall",
    "should", "may", "might", "must", "not", "no", "so", "such", "than", "too", "very",
    "just", "about", "into", "over", "under", "again", "further", "once", "there", "here",
})


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class Retriever:
    def __init__(self, kb_dir: str, k1: float = 1.5, b: float = 0.75, embedder=None):
        self.kb_dir = kb_dir
        self.k1 = k1
        self.b = b
        self.chunks: list[dict] = []
        self._load(kb_dir)
        self.embedder = embedder
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
        self.doc_tokens = [tokenize(c["text"]) for c in self.chunks]
        self.doc_term_counts = [Counter(toks) for toks in self.doc_tokens]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.n_docs = len(self.chunks)

        self.df: Counter = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                self.df[term] += 1

        self.faiss_index = None
        if self.chunks:
            import faiss

            if self.embedder is None:
                from copilot.rag.embeddings import get_embedder

                self.embedder = get_embedder()
            doc_embeddings = self.embedder.encode([c["text"] for c in self.chunks])
            self.faiss_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
            self.faiss_index.add(doc_embeddings)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def _bm25_score(self, query_tokens: list[str], doc_index: int) -> float:
        term_counts = self.doc_term_counts[doc_index]
        dl = self.doc_len[doc_index]
        score = 0.0
        for term in query_tokens:
            if term in STOPWORDS:
                continue
            f = term_counts.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += idf * (f * (self.k1 + 1)) / (denom or 1)
        return score

    def _vector_scores(self, query: str) -> list[float]:
        if self.faiss_index is None:
            return [0.0] * self.n_docs
        query_embedding = self.embedder.encode([query])
        similarities, indices = self.faiss_index.search(query_embedding, self.n_docs)
        scores = [0.0] * self.n_docs
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0:
                scores[idx] = float(sim)
        return scores

    def _rerank(self, query: str, query_tokens: list[str], candidates: list[tuple[float, int]]) -> list[tuple[float, int]]:
        query_lower = query.lower()
        query_term_set = set(query_tokens)
        reranked = []
        for fused_score, idx in candidates:
            text_lower = self.chunks[idx]["text"].lower()
            phrase_bonus = 0.15 if len(query_lower) > 8 and query_lower in text_lower else 0.0
            doc_terms = set(self.doc_tokens[idx])
            coverage = (len(query_term_set & doc_terms) / len(query_term_set)) if query_term_set else 0.0
            reranked.append((fused_score + phrase_bonus + 0.1 * coverage, idx))
        reranked.sort(key=lambda pair: pair[0], reverse=True)
        return reranked

    def search(self, query: str, k: int = 3, candidate_pool: int = 10) -> list[dict]:
        if not self.chunks:
            return []

        query_tokens = tokenize(query)
        bm25_scores = [self._bm25_score(query_tokens, i) for i in range(self.n_docs)]
        vector_scores = self._vector_scores(query)

        # Candidate generation: BM25 hits are the primary channel (any real
        # term overlap qualifies); the vector channel is a secondary recall
        # path that can surface a paraphrase/synonym match BM25 missed
        # entirely, but only past a threshold. Gating candidacy this way -
        # rather than folding vector into a full-corpus rank-fusion score -
        # means the vector signal can never bury a document BM25 was
        # confident about; it can only add documents BM25 missed, or nudge
        # order within the confirmed set.
        candidate_idxs = {i for i in range(self.n_docs) if bm25_scores[i] > 0}
        candidate_idxs |= {i for i in range(self.n_docs) if vector_scores[i] > VECTOR_CANDIDATE_THRESHOLD}
        if not candidate_idxs:
            return []

        max_bm25 = max(bm25_scores[i] for i in candidate_idxs) or 1.0
        max_vector = max(vector_scores[i] for i in candidate_idxs) or 1.0
        fused = [
            (BM25_WEIGHT * (bm25_scores[i] / max_bm25) + VECTOR_WEIGHT * (vector_scores[i] / max_vector), i)
            for i in candidate_idxs
        ]
        fused.sort(key=lambda pair: pair[0], reverse=True)

        reranked = self._rerank(query, query_tokens, fused[:candidate_pool])
        top_k = reranked[:k]

        # A relative-score cutoff, not just the k cap: when the top hit clearly
        # dominates (a single-fact question with one strong match), a much
        # weaker runner-up shouldn't ride along just to fill k slots. Multiple
        # genuinely close hits (comparable scores) still all come through.
        if top_k:
            score_floor = top_k[0][0] * RELATIVE_SCORE_CUTOFF
            top_k = [(score, i) for score, i in top_k if score >= score_floor]

        return [
            {"source": self.chunks[i]["source"], "text": self.chunks[i]["text"], "score": round(score, 4)}
            for score, i in top_k
        ]


_singleton: Retriever | None = None


def get_retriever() -> Retriever:
    global _singleton
    if _singleton is None:
        from copilot.config import KB_DIR

        _singleton = Retriever(KB_DIR)
    return _singleton
