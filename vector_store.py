"""
Real vector store using FAISS -- genuinely installed and running in
this sandbox, not mocked. Takes whatever embedder you give it
(TfidfEmbedder now, SentenceTransformerEmbedder on your own machine
later) and builds a proper similarity-search index around it.
"""

import os
import faiss
import numpy as np


class VectorStore:
    def __init__(self, embedder, index_type: str = "flat"):
        """
        index_type:
          "flat" - IndexFlatIP. Exact search, checks every vector.
                   Fine up to a few thousand chunks. What we've used
                   so far in this project.
          "ivf"  - IndexIVFFlat. Approximate search using clustering
                   ("inverted file") -- much faster on large corpora
                   (tens of thousands+ chunks) at a small accuracy
                   cost. Needs a training step before use.
          "hnsw" - IndexHNSWFlat. Approximate search using a graph
                   structure -- typically the best speed/accuracy
                   tradeoff for large corpora, no training step
                   needed, but uses more memory than IVF.
        """
        self.embedder = embedder
        self.index_type = index_type
        self.index = None
        self.documents = []

    def build(self, documents: list):
        """documents: list of {"id": str, "text": str}"""
        self.documents = documents
        vectors = self.embedder.embed([d["text"] for d in documents])
        dim = vectors.shape[1]
        faiss.normalize_L2(vectors)

        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(vectors)

        elif self.index_type == "ivf":
            n_clusters = max(1, min(100, len(documents) // 10))
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            self.index.train(vectors)  # IVF needs a training pass to learn clusters
            self.index.add(vectors)

        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.add(vectors)

        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

    def search(self, query: str, top_k: int = 3):
        query_vec = self.embedder.embed([query])
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            results.append({"id": doc["id"], "text": doc["text"], "score": round(float(score), 3)})
        return results

    def save(self, folder_path: str):
        """Persists the FAISS index + document metadata to disk, so
        you don't have to re-embed and re-index every time you run
        the pipeline -- only re-run build() when documents change."""
        import json
        os.makedirs(folder_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
        with open(os.path.join(folder_path, "documents.json"), "w") as f:
            json.dump(self.documents, f)

    def load(self, folder_path: str):
        import json
        self.index = faiss.read_index(os.path.join(folder_path, "index.faiss"))
        with open(os.path.join(folder_path, "documents.json")) as f:
            self.documents = json.load(f)