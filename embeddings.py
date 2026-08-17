"""
Embedding functions for the vector DB retriever.

TWO implementations, same interface (`embed(texts) -> list[list[float]]`),
so the vector store code never needs to change when you swap one for
the other:

  1. TfidfEmbedder   - WORKS RIGHT NOW in this sandbox. Real, running
                        code. Not a semantic embedding (no meaning
                        understanding), just dense numeric vectors
                        derived from TF-IDF, dense-ified with SVD so
                        they behave like normal embedding vectors for
                        FAISS to index.

  2. SentenceTransformerEmbedder - REAL semantic embeddings. This is
                        what a production RAG system actually uses.
                        Requires downloading model weights from
                        HuggingFace, which needs an internet
                        connection this sandbox doesn't have -- run
                        this on your own machine instead. Code is
                        correct and ready to run as-is.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np


class TfidfEmbedder:
    """Working now. Dense vectors derived from word-frequency
    statistics -- captures word OVERLAP, not meaning. 'car' and
    'automobile' will NOT be considered similar by this embedder,
    which is exactly the limitation a real embedding model fixes."""

    def __init__(self, corpus_texts: list, n_components: int = 50):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        sparse = self.vectorizer.fit_transform(corpus_texts)
        # SVD compresses the sparse TF-IDF matrix into dense vectors,
        # the same SHAPE a real embedding model would output, so the
        # rest of the pipeline (FAISS indexing) doesn't need to know
        # or care which embedder produced them.
        n_components = min(n_components, sparse.shape[0] - 1, sparse.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=max(n_components, 2))
        self.svd.fit(sparse)

    def embed(self, texts: list) -> np.ndarray:
        sparse = self.vectorizer.transform(texts)
        dense = self.svd.transform(sparse)
        return dense.astype("float32")


class SentenceTransformerEmbedder:
    """REAL semantic embeddings -- run this on your own machine with
    internet access. Requires: pip install sentence-transformers

    This is the swap-in for production use. Nothing else in
    vector_store.py or vector_rag.py needs to change -- both embedders
    expose the same `.embed(texts) -> np.ndarray` interface."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # local import
        self.model = SentenceTransformer(model_name)  # downloads ~80MB once

    def embed(self, texts: list) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True).astype("float32")
