# Vector Databases and How They're Used in RAG Systems

---

## Part 1: What a Vector Database Actually Is

A vector database stores pieces of data as **numeric vectors** (long lists of numbers, typically 384–1536 numbers per vector) and lets you search by **similarity** instead of by exact keyword match.

The core operation every vector DB is built around:

> Given a query vector, find the K stored vectors that are *closest* to it in the vector space.

"Closest" is measured with a distance/similarity metric — most commonly **cosine similarity** (angle between vectors) or **dot product**. Two vectors that are close together are assumed to represent text with *similar meaning* — not similar spelling, similar meaning.

### Why this matters for search

A traditional keyword search (like the TF-IDF method used in your project so far) matches based on **shared words**. It has no idea that "car" and "automobile" mean the same thing, or that "cost of installing solar panels" and "solar panels typically cost between $12,000 and $18,000" are talking about the same thing if the exact words don't overlap enough.

A vector database, combined with a **neural embedding model**, matches based on **meaning**. The embedding model reads a sentence and outputs a vector that captures what the sentence is *about* — so semantically related text ends up close together in vector space even with completely different wording.

---

## Part 2: How This Fits Into a RAG System

RAG = Retrieval-Augmented Generation. The "retrieval" half is exactly where the vector database lives.

```
                     ┌─────────────────────┐
                     │   Document corpus    │
                     └──────────┬───────────┘
                                │
                    (one-time, done ahead of query)
                                │
                                v
                   ┌─────────────────────────┐
                   │  Embedding model         │
                   │  (turns text -> vector)  │
                   └──────────┬───────────────┘
                                │
                                v
                   ┌─────────────────────────┐
                   │  Vector database         │
                   │  (stores + indexes all   │
                   │   document vectors)      │
                   └──────────┬───────────────┘
                                │
   User query ──> Embedding model ──> query vector
                                │              │
                                └──────┬───────┘
                                        v
                        Vector DB similarity search
                       (find top-K closest document
                              vectors to the query)
                                        │
                                        v
                         Top-K relevant text chunks
                                        │
                                        v
                    Fed as CONTEXT into the LLM/SLM prompt
                                        │
                                        v
                              Final generated answer
```

**Two distinct phases:**

1. **Indexing phase** (happens once, or whenever documents change): every document/chunk is embedded and stored in the vector DB ahead of time. This is the expensive, slow-ish step — but it only has to happen once per document.
2. **Query phase** (happens every time a user asks something): the query itself gets embedded the same way, then the vector DB does a fast similarity search against everything already indexed. This step needs to be fast, since it happens on every request.

---

## Part 3: The Three Pieces People Often Confuse

| Piece | What it does | Examples |
|---|---|---|
| **Embedding model** | Turns text into a vector | `all-MiniLM-L6-v2`, OpenAI `text-embedding-3-small`, Cohere Embed 
| **Vector database / index** | Stores vectors, finds nearest neighbors fast | FAISS, Chroma, Pinecone, Weaviate, Qdrant, Milvus |
| **Similarity metric** | How "closeness" is measured | Cosine similarity, dot product, Euclidean distance |

A common beginner mistake is thinking "vector database" refers to the whole pipeline. It doesn't — the embedding model is a completely separate component you choose independently. The vector DB doesn't understand language at all; it just stores numbers and finds nearby numbers quickly.

---

## Part 4: What Made This Project's Original Retriever Different

The RAG flow built earlier in this project used **TF-IDF + cosine similarity**, not a vector database with neural embeddings. Worth being precise about why that's a meaningfully different (and weaker) approach:

- TF-IDF vectors are based on **word frequency statistics** — how often a word appears in a document vs. across the whole corpus.
- They have **no understanding of meaning** — "solar panel cost" and "price of installing photovoltaic systems" would look almost unrelated to TF-IDF, even though they mean nearly the same thing.
- There's **no real indexing structure** for fast search at scale — with only 5 documents this doesn't matter, but with a million documents, computing cosine similarity against every single one on every query becomes far too slow.

A real vector database fixes both problems: neural embeddings capture meaning, and the database itself is built around fast nearest-neighbor search algorithms (like HNSW or IVF) that scale to millions or billions of vectors.

---

## Part 5: Popular Vector Databases, and Where Each One Fits

| Vector DB | Runs where | Good for |
|---|---|---|
| **FAISS** | Fully local, in your own code | Prototyping, learning, small-to-medium projects, full control |
| **Chroma** | Local, or lightweight self-hosted server | Small-to-medium apps, easiest to get started with |
| **Qdrant** | Self-hosted or cloud | Production apps needing filtering + scalability |
| **Weaviate** | Self-hosted or cloud | Production apps, built-in hybrid search |
| **Pinecone** | Fully managed cloud service | Production apps that don't want to manage infrastructure |
| **Milvus** | Self-hosted or cloud | Very large-scale (billions of vectors) production use |

For a student project or internship-scale build, **FAISS or Chroma are the right choice** — both are free, run entirely on your own machine, and need no account, API key, or hosted service.

---

## Part 6: Step-by-Step — Adding a Real Vector DB to This Project

This project's `vector_store.py`, `embeddings.py`, and `vector_rag.py` (already built and running with TF-IDF embeddings as a stand-in) are set up so **only one piece needs to change** to get real semantic search: the embedding function. Here's the exact procedure to run on your own machine.

### Step 1 — Install the real embedding model library
```bash
pip install sentence-transformers
```
This downloads a small, free, locally-run embedding model the first time you use it (no API key, no cost, no internet needed after the first download).

### Step 2 — Confirm FAISS is installed
```bash
pip install faiss-cpu
```
(Already installed and working in this project — same command works anywhere.)

### Step 3 — Switch the embedder in `vector_rag.py`
Open `vector_rag.py` and change:
```python
embedder = TfidfEmbedder([d["text"] for d in DOCUMENTS])
```
to:
```python
from embeddings import SentenceTransformerEmbedder
embedder = SentenceTransformerEmbedder()
```
That's the entire change. `vector_store.py` and everything downstream (the planner, the SLM agents) don't need to be touched — they only ever interact with the `.embed()` interface, not with which embedding method is behind it.

### Step 4 — Re-run and compare
```bash
python3 vector_rag.py
```
Run the same three test queries as before and compare the retrieved documents against the TF-IDF version's output. You should see the "cost of installing solar panels" query now correctly retrieve the actual cost document (doc2) — something the TF-IDF version missed, exactly the gap semantic embeddings are meant to close.

### Step 5 — (Optional) Scale it up realistically
Once this works on a 5-document toy corpus:
1. Replace `knowledge_base.py`'s hardcoded list with real documents loaded from files (PDF, text, or web pages) — chunk long documents into smaller pieces (~200-500 words each) before embedding, since embedding an entire long document as one vector loses detail.
2. If your corpus grows past a few thousand chunks, switch FAISS from `IndexFlatIP` (exact search) to `IndexIVFFlat` or `IndexHNSWFlat` (approximate search) — same `.add()`/`.search()` interface, but built for speed at scale. This is a one-line change in `vector_store.py`, marked in the code comments.
3. Persist the index to disk instead of rebuilding it every run (FAISS supports `faiss.write_index()` / `faiss.read_index()`; Chroma persists automatically if you give it a storage path).

### Step 6 — Reconnect it to the rest of your pipeline
Your `planner.py` and `slm_agents.py` from the earlier build don't need any changes — they consume retrieved context as plain text, regardless of which retrieval method produced it. Just point `full_pipeline.py`'s retrieval step at `vector_rag.py`'s `VectorStore` instead of the original `Retriever` class, and the whole RAG + planner + SLM system now runs on real semantic search.
