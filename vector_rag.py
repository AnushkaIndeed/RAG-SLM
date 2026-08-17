"""
RAG using the real vector store (FAISS), running RIGHT NOW with
TfidfEmbedder (since that's what works without internet access here).

To get REAL semantic retrieval on your own machine, change ONE line
(marked below) -- nothing else in this file, vector_store.py, or the
rest of the pipeline (planner.py, slm_agents.py) needs to change.
"""

from knowledge_base import DOCUMENTS
from embeddings import SentenceTransformerEmbedder, TfidfEmbedder
from vector_store import VectorStore


def build_vector_retriever():
    from embeddings import SentenceTransformerEmbedder
    embedder = SentenceTransformerEmbedder()
    # >>> ON YOUR OWN MACHINE, SWAP THE LINE ABOVE FOR: <<<
    # from embeddings import SentenceTransformerEmbedder
    # embedder = SentenceTransformerEmbedder()

    store = VectorStore(embedder)
    store.build(DOCUMENTS)
    return store


if __name__ == "__main__":
    print("=" * 70)
    print("VECTOR-DB RAG RETRIEVAL — DEMO (FAISS, real index, running now)")
    print("=" * 70)

    store = build_vector_retriever()

    queries = [
        "How do solar panels work?",
        "What is the cost of installing solar panels?",
        "How long do panels last and are they worth it?",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        results = store.search(q, top_k=2)
        for r in results:
            print(f"  [{r['id']}] score={r['score']}  {r['text'][:80]}...")
