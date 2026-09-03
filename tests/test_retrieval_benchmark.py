from copilot.config import KB_DIR
from copilot.rag.retriever import Retriever
from eval.retrieval_benchmark import (
    evaluate,
    load_labels,
    rank_bm25_only,
    rank_hybrid,
    rank_real_vector_only,
)


def test_labels_reference_real_chunk_ids():
    retriever = Retriever(KB_DIR)
    valid_ids = {chunk["id"] for chunk in retriever.chunks}
    labels = load_labels()
    assert labels
    for label in labels:
        assert label["id"] in valid_ids
        assert label["style"] in ("lexical", "paraphrase")


def test_hybrid_ranking_beats_bm25_only_on_paraphrases():
    retriever = Retriever(KB_DIR)
    id_to_index = {chunk["id"]: i for i, chunk in enumerate(retriever.chunks)}
    labels = [label for label in load_labels() if label["style"] == "paraphrase"]

    bm25_result = evaluate("bm25_only", rank_bm25_only, retriever, labels, id_to_index)
    hybrid_result = evaluate("hybrid", rank_hybrid, retriever, labels, id_to_index)

    assert hybrid_result["overall"]["mrr"] >= bm25_result["overall"]["mrr"]


def test_real_embeddings_produce_a_full_ranking():
    retriever = Retriever(KB_DIR)
    ranking = rank_real_vector_only(retriever, "What is the timely filing deadline for Medicare claims?")
    assert sorted(ranking) == list(range(retriever.n_docs))
