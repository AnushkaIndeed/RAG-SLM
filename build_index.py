"""
One-time setup script: sanity-checks the PDFs, chunks them, builds the
FAISS index, and saves it to disk. Run this once, then re-run only if
you add/change PDFs in ./data/.
"""

from documentLoader import load_documents, load_and_chunk_documents
from embeddings import SentenceTransformerEmbedder
from vector_store import VectorStore

# --- Step 2: sanity check extraction quality ---
print("=" * 60)
print("STEP 2: Checking PDF extraction quality")
print("=" * 60)
docs = load_documents("./data")
for d in docs:
    print(f"{d['id']}: {len(d['text'])} characters extracted")
    print(d['text'][:200], "...\n")

# --- Step 3: chunk ---
print("=" * 60)
print("STEP 3: Chunking documents")
print("=" * 60)
documents = load_and_chunk_documents("./data", chunk_size=300, overlap=75)
print(f"{len(documents)} chunks created from your PDFs\n")

# --- Step 4: build + save the vector index ---
print("=" * 60)
print("STEP 4: Building and saving the FAISS index")
print("=" * 60)
embedder = SentenceTransformerEmbedder()
store = VectorStore(embedder, index_type="flat")
store.build(documents)
store.save("./my_index")
print("Index saved to ./my_index")