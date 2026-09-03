"""A small, dependency-free hybrid retriever over the local knowledge base.

Two independent signals feed candidate generation, then a fused score reranks:

  1. BM25 (Okapi) over exact tokens - precise, and the primary channel: any
     document with a real term hit is a candidate.
  2. A hashed character-trigram "vector" (cosine similarity) - a crude,
     dependency-free stand-in for a real sentence embedding that tolerates
     partial/sub-word overlap BM25 misses, used as a secondary recall path
     (only past a threshold well above its noise floor) so it can surface a
     paraphrase BM25 missed entirely, without being trusted enough to bury a
     document BM25 was confident about.

Within that candidate set, BM25 and vector scores are max-normalized and
combined with a weighted sum (BM25-dominant), then a cheap rerank pass boosts
exact-phrase containment and query-term coverage - a feature a single
bi-encoder-style score doesn't capture on its own. An earlier version of this
combined the two signals with rank-based Reciprocal Rank Fusion (RRF) instead
of gating candidacy by BM25; that let the noisy vector ranking veto documents
BM25 had ranked first, which is why candidacy and fusion are split apart here.

Deliberately avoids embedding-model or vector-DB dependencies so the whole
project runs offline. Swap `_hash_vector` for a real embedding call and
`search()`'s contract (`{"source", "text", "score"}`) doesn't need to change
for any caller.
"""
import math
import os
import re
import zlib
from collections import Counter

TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
VECTOR_DIMS = 1024
BM25_WEIGHT = 0.65
RELATIVE_SCORE_CUTOFF = 0.6
VECTOR_WEIGHT = 0.35
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


def _char_trigrams(text: str) -> list[str]:
    padded = f"  {text.lower()}  "
    return [padded[i:i + 3] for i in range(len(padded) - 2)]


def _hash_vector(text: str, dims: int = VECTOR_DIMS) -> list[float]:
    vec = [0.0] * dims
    for trigram in _char_trigrams(text):
        vec[zlib.crc32(trigram.encode("utf-8")) % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class Retriever:
    def __init__(self, kb_dir: str, k1: float = 1.5, b: float = 0.75):
        self.kb_dir = kb_dir
        self.k1 = k1
        self.b = b
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
        self.doc_tokens = [tokenize(c["text"]) for c in self.chunks]
        self.doc_term_counts = [Counter(toks) for toks in self.doc_tokens]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.n_docs = len(self.chunks)

        self.df: Counter = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                self.df[term] += 1

        self.doc_vectors = [_hash_vector(c["text"]) for c in self.chunks]

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
        query_vector = _hash_vector(query)

        bm25_scores = [self._bm25_score(query_tokens, i) for i in range(self.n_docs)]
        vector_scores = [_cosine(query_vector, self.doc_vectors[i]) for i in range(self.n_docs)]

        # Candidate generation: BM25 hits are the primary channel (any real
        # term overlap qualifies); the vector channel is a secondary recall
        # path that can surface a paraphrase/morphological match BM25 missed
        # entirely, but only past a threshold well above the hashed vector's
        # noise floor on a corpus this small (~0.1-0.2 for unrelated text).
        # Gating candidacy this way - rather than folding vector into a
        # full-corpus rank-fusion score - means the noisy vector signal can
        # never bury a document BM25 was confident about; it can only add
        # documents BM25 missed, or nudge order within the confirmed set.
        candidate_idxs = {i for i in range(self.n_docs) if bm25_scores[i] > 0}
        candidate_idxs |= {i for i in range(self.n_docs) if vector_scores[i] > 0.35}
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
        # weaker runner-up shouldn't ride along just to fill k slots - that's
        # how an unrelated paragraph that happens to share one word with the
        # query ends up quoted in an answer that never needed it. Multiple
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
