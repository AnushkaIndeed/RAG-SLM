"""
BASIC RAG FLOW (Step 1 — no planner/orchestration yet)

    User query --> Retriever --> top-k chunks --> single generator --> answer

This is the baseline everything else in this project builds on top of.
The generator call is mocked (marked below) since no real model API is
reachable from this sandbox — swap it for a real small model call
(e.g., Qwen3 via Ollama) and nothing else in this file changes.
"""

from retriever import Retriever


def mock_generate(query: str, context_chunks: list) -> str:
    # >>> REPLACE WITH REAL MODEL CALL <<<
    # Real version: send `query` + `context_chunks` to a small language
    # model (e.g., Qwen3-1.7B) as a standard RAG prompt and return its
    # generated answer.
    context_text = " ".join(c["text"] for c in context_chunks)
    return (
        f"[Answer based on retrieved context]\n"
        f"Regarding '{query}': {context_text[:300]}..."
    )


def basic_rag(query: str, top_k: int = 2) -> dict:
    retriever = Retriever()
    chunks = retriever.retrieve(query, top_k=top_k)
    answer = mock_generate(query, chunks)
    return {"query": query, "retrieved_chunks": chunks, "answer": answer}


if __name__ == "__main__":
    result = basic_rag("How do solar panels work and what do they cost?")
    print("=" * 70)
    print("BASIC RAG FLOW — DEMO")
    print("=" * 70)
    print(f"Query: {result['query']}\n")
    print("Retrieved chunks:")
    for c in result["retrieved_chunks"]:
        print(f"  [{c['id']}] (score={c['score']}) {c['text'][:80]}...")
    print(f"\nFinal answer:\n{result['answer']}")
