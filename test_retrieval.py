from embeddings import SentenceTransformerEmbedder
from vector_store import VectorStore

embedder = SentenceTransformerEmbedder()
store = VectorStore(embedder)
store.load("./my_index")

test_queries = [
    "What did the Amsterdam paper find about orchestrator reasoning vs sub-agent size?",
    "What is the MAST failure taxonomy?",
    "Why did MapCoder-Lite's naive small-model approach collapse?",
]

for q in test_queries:
    print(f"\nQuery: {q}")
    results = store.search(q, top_k=3)
    for r in results:
        print(f"  [{r['id']}] score={r['score']}: {r['text'][:100]}...")
# --- Retrying the MapCoder-Lite query that returned weak results ---
print("\n--- Retry 1: higher top_k ---")
results = store.search("Why did MapCoder-Lite's naive small-model approach collapse?", top_k=6)
for r in results:
    print(f"  [{r['id']}] score={r['score']}: {r['text'][:100]}...")

print("\n--- Retry 2: rephrased query ---")
results = store.search("small model role assignment failure without distillation", top_k=5)
for r in results:
    print(f"  [{r['id']}] score={r['score']}: {r['text'][:100]}...")