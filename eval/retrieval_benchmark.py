"""Retrieval quality benchmark - proves the embedding swap actually helped,
rather than just changing the architecture.

Compares four retrieval configurations against a labeled query -> relevant-chunk
dataset (`eval/retrieval_labels.jsonl`, 30 queries, one per knowledge-base
paragraph, roughly split between queries that share exact vocabulary with
their target paragraph ("lexical") and paraphrases that deliberately share
none ("paraphrase") - the split matters because it's exactly where a lexical
method (BM25) and a semantic method (embeddings) should diverge:

  - bm25_only               - the lexical signal alone
  - legacy_hashed_vector    - the original dependency-free hashed-trigram
                               "vector" this project used before the upgrade
  - real_embedding_vector   - Sentence-Transformers + FAISS alone
  - hybrid (production)     - BM25 + real embeddings, fused and reranked
                               (what `Retriever.search()` actually runs)

Metrics: Recall@1, Recall@3, and MRR (mean reciprocal rank), overall and
broken out by query style - the breakout is what shows *why* the upgrade
matters: a purely lexical method should lag specifically on paraphrases.

Run: python -m eval.retrieval_benchmark
"""
import json
import math
import zlib
from pathlib import Path

from copilot.config import KB_DIR
from copilot.rag.retriever import Retriever, tokenize

LABELS_PATH = Path(__file__).parent / "retrieval_labels.jsonl"
REPORT_PATH = Path(__file__).parent / "retrieval_benchmark_report.json"

LEGACY_VECTOR_DIMS = 1024


def load_labels() -> list[dict]:
    with open(LABELS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# The retired hashed-trigram vector, reproduced standalone here purely so the
# benchmark can show what it used to score - production code no longer has
# any use for it.
# --------------------------------------------------------------------------

def _legacy_hash_vector(text: str, dims: int = LEGACY_VECTOR_DIMS) -> list[float]:
    padded = f"  {text.lower()}  "
    trigrams = [padded[i:i + 3] for i in range(len(padded) - 2)]
    vec = [0.0] * dims
    for trigram in trigrams:
        vec[zlib.crc32(trigram.encode("utf-8")) % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def rank_bm25_only(retriever: Retriever, query: str) -> list[int]:
    query_tokens = tokenize(query)
    scored = [(retriever._bm25_score(query_tokens, i), i) for i in range(retriever.n_docs)]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [i for _, i in scored]


def rank_real_vector_only(retriever: Retriever, query: str) -> list[int]:
    scores = retriever._vector_scores(query)
    return sorted(range(retriever.n_docs), key=lambda i: scores[i], reverse=True)


def rank_legacy_hash_vector_only(retriever: Retriever, legacy_doc_vectors: list[list[float]], query: str) -> list[int]:
    query_vector = _legacy_hash_vector(query)
    scores = [_cosine(query_vector, doc_vector) for doc_vector in legacy_doc_vectors]
    return sorted(range(retriever.n_docs), key=lambda i: scores[i], reverse=True)


def rank_legacy_hybrid(retriever: Retriever, legacy_doc_vectors: list[list[float]], query: str) -> list[int]:
    """The exact fusion logic `rank_hybrid` uses, but with the legacy hashed
    vector in place of real embeddings - i.e. what this project's hybrid
    retriever actually scored *before* the embedding swap. This is the most
    direct before/after comparison: everything else held constant."""
    from copilot.rag.retriever import BM25_WEIGHT, VECTOR_WEIGHT

    query_tokens = tokenize(query)
    bm25_scores = [retriever._bm25_score(query_tokens, i) for i in range(retriever.n_docs)]
    query_vector = _legacy_hash_vector(query)
    vector_scores = [_cosine(query_vector, doc_vector) for doc_vector in legacy_doc_vectors]

    legacy_vector_threshold = 0.35  # the threshold this project used with the legacy vector
    candidate_idxs = {i for i in range(retriever.n_docs) if bm25_scores[i] > 0}
    candidate_idxs |= {i for i in range(retriever.n_docs) if vector_scores[i] > legacy_vector_threshold}

    if not candidate_idxs:
        return sorted(range(retriever.n_docs), key=lambda i: bm25_scores[i], reverse=True)

    max_bm25 = max(bm25_scores[i] for i in candidate_idxs) or 1.0
    max_vector = max(vector_scores[i] for i in candidate_idxs) or 1.0
    fused = [
        (BM25_WEIGHT * (bm25_scores[i] / max_bm25) + VECTOR_WEIGHT * (vector_scores[i] / max_vector), i)
        for i in candidate_idxs
    ]
    reranked = retriever._rerank(query, query_tokens, fused)
    ranked_idxs = [i for _, i in reranked]

    remainder = sorted(
        (i for i in range(retriever.n_docs) if i not in candidate_idxs),
        key=lambda i: bm25_scores[i], reverse=True,
    )
    return ranked_idxs + remainder


def rank_hybrid(retriever: Retriever, query: str) -> list[int]:
    """Reproduces `Retriever.search()`'s candidate-generation + fusion + rerank,
    but returns a full ranking (not the end-user top-k with a relative-score
    cutoff applied) - what a ranking benchmark needs. Any document that never
    became a candidate (no BM25 hit and below the vector threshold) is appended
    at the end in BM25 order, so it still costs Recall@k if it was the target."""
    from copilot.rag.retriever import BM25_WEIGHT, VECTOR_CANDIDATE_THRESHOLD, VECTOR_WEIGHT

    query_tokens = tokenize(query)
    bm25_scores = [retriever._bm25_score(query_tokens, i) for i in range(retriever.n_docs)]
    vector_scores = retriever._vector_scores(query)

    candidate_idxs = {i for i in range(retriever.n_docs) if bm25_scores[i] > 0}
    candidate_idxs |= {i for i in range(retriever.n_docs) if vector_scores[i] > VECTOR_CANDIDATE_THRESHOLD}

    if not candidate_idxs:
        return sorted(range(retriever.n_docs), key=lambda i: bm25_scores[i], reverse=True)

    max_bm25 = max(bm25_scores[i] for i in candidate_idxs) or 1.0
    max_vector = max(vector_scores[i] for i in candidate_idxs) or 1.0
    fused = [
        (BM25_WEIGHT * (bm25_scores[i] / max_bm25) + VECTOR_WEIGHT * (vector_scores[i] / max_vector), i)
        for i in candidate_idxs
    ]
    reranked = retriever._rerank(query, query_tokens, fused)
    ranked_idxs = [i for _, i in reranked]

    remainder = sorted(
        (i for i in range(retriever.n_docs) if i not in candidate_idxs),
        key=lambda i: bm25_scores[i], reverse=True,
    )
    return ranked_idxs + remainder


def evaluate(name: str, rank_fn, retriever: Retriever, labels: list[dict], id_to_index: dict) -> dict:
    per_style: dict[str, list[dict]] = {"lexical": [], "paraphrase": []}
    all_ranks = []

    for label in labels:
        target_idx = id_to_index[label["id"]]
        ranking = rank_fn(retriever, label["query"])
        rank_position = ranking.index(target_idx) + 1  # 1-indexed
        all_ranks.append(rank_position)
        per_style[label["style"]].append({"query": label["query"], "rank": rank_position})

    def summarize(ranks: list[int]) -> dict:
        n = len(ranks)
        if n == 0:
            return {"n": 0, "recall_at_1": None, "recall_at_3": None, "mrr": None}
        return {
            "n": n,
            "recall_at_1": round(sum(1 for r in ranks if r <= 1) / n, 3),
            "recall_at_3": round(sum(1 for r in ranks if r <= 3) / n, 3),
            "mrr": round(sum(1 / r for r in ranks) / n, 3),
        }

    return {
        "name": name,
        "overall": summarize(all_ranks),
        "lexical": summarize([e["rank"] for e in per_style["lexical"]]),
        "paraphrase": summarize([e["rank"] for e in per_style["paraphrase"]]),
    }


def main() -> None:
    labels = load_labels()
    retriever = Retriever(KB_DIR)
    id_to_index = {chunk["id"]: i for i, chunk in enumerate(retriever.chunks)}

    legacy_doc_vectors = [_legacy_hash_vector(chunk["text"]) for chunk in retriever.chunks]

    configs = [
        ("bm25_only", lambda r, q: rank_bm25_only(r, q)),
        ("legacy_hashed_vector_only", lambda r, q: rank_legacy_hash_vector_only(r, legacy_doc_vectors, q)),
        ("real_embedding_vector_only", lambda r, q: rank_real_vector_only(r, q)),
        ("hybrid_legacy (old prod)", lambda r, q: rank_legacy_hybrid(r, legacy_doc_vectors, q)),
        ("hybrid_production (now)", lambda r, q: rank_hybrid(r, q)),
    ]

    results = [evaluate(name, fn, retriever, labels, id_to_index) for name, fn in configs]

    print(f"\n{'=' * 88}")
    print(f"Retrieval benchmark - {len(labels)} labeled queries "
          f"({sum(1 for l in labels if l['style'] == 'lexical')} lexical, "
          f"{sum(1 for l in labels if l['style'] == 'paraphrase')} paraphrase)")
    print("=" * 88)
    header = f"{'config':<28}{'R@1':>8}{'R@3':>8}{'MRR':>8}   |{'lex R@1':>10}{'lex MRR':>10}   |{'para R@1':>10}{'para MRR':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        o, lex, para = r["overall"], r["lexical"], r["paraphrase"]
        print(
            f"{r['name']:<28}{o['recall_at_1']:>8}{o['recall_at_3']:>8}{o['mrr']:>8}   |"
            f"{lex['recall_at_1']:>10}{lex['mrr']:>10}   |"
            f"{para['recall_at_1']:>10}{para['mrr']:>10}"
        )
    print("=" * 88)
    print("R@1/R@3 = Recall@1/@3, MRR = mean reciprocal rank. 'lex'/'para' columns split by query style -")
    print("a lexical-only method should lag specifically on paraphrases if embeddings are adding real value.\n")

    REPORT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Full report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
