from copilot.rag.retriever import get_retriever


def search_kb(query: str, top_k: int = 3) -> list[dict]:
    return get_retriever().search(query, k=top_k)
