from copilot.config import KB_DIR
from copilot.rag.retriever import Retriever


def test_retriever_finds_relevant_doc():
    retriever = Retriever(KB_DIR)
    results = retriever.search("production deployment approvals", k=3)
    assert results
    assert any("deploy" in r["source"].lower() for r in results)


def test_retriever_ranks_more_relevant_doc_higher():
    retriever = Retriever(KB_DIR)
    results = retriever.search("Sev1 incident escalation on-call", k=3)
    assert results[0]["source"] == "incident_response.md"


def test_retriever_returns_empty_for_unrelated_query():
    retriever = Retriever(KB_DIR)
    results = retriever.search("zzz qqq nonexistent gibberish term", k=3)
    assert results == []
