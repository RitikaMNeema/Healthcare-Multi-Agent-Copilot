from copilot.config import KB_DIR
from copilot.rag.retriever import Retriever


def test_retriever_finds_relevant_doc():
    retriever = Retriever(KB_DIR)
    results = retriever.search("What is the timely filing deadline for Medicare claims?", k=3)
    assert results
    # "Timely filing" is genuinely discussed in both appeals_policy.md and
    # claims_submission_policy.md, so either may rank first - what matters is
    # that the Medicare-specific fact is actually surfaced somewhere in top-k.
    assert any("365 days" in r["text"] for r in results)


def test_retriever_ranks_more_relevant_doc_higher():
    retriever = Retriever(KB_DIR)
    results = retriever.search("What is the HIPAA minimum necessary standard?", k=3)
    assert results[0]["source"] == "hipaa_privacy_policy.md"


def test_retriever_returns_empty_for_unrelated_query():
    retriever = Retriever(KB_DIR)
    results = retriever.search("zzz qqq nonexistent gibberish term", k=3)
    assert results == []


def test_relative_score_cutoff_drops_weak_runner_up():
    retriever = Retriever(KB_DIR)
    results = retriever.search("What is the HIPAA breach notification deadline?", k=3)
    # A much weaker, only-incidentally-related paragraph should not ride along
    # just because k allows more slots - see RELATIVE_SCORE_CUTOFF.
    assert all(r["source"] == "hipaa_privacy_policy.md" for r in results)
    assert not any("export of raw" in r["text"] for r in results)


def test_bm25_not_dominated_by_stopwords_in_small_corpus():
    # A common function word can get a spuriously high IDF in a tiny corpus
    # (see STOPWORDS) - this regression-tests that the actual CO-197 policy
    # paragraph outranks paragraphs that merely share stopwords with the query.
    retriever = Retriever(KB_DIR)
    results = retriever.search("What does denial code CO-197 mean and how do I fix it?", k=3)
    assert any("CO-197 means" in r["text"] for r in results)
