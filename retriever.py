"""
Basic retriever for the RAG flow.
Uses TF-IDF + cosine similarity — a real, working retrieval method
(no internet/API needed), standing in for a production embedding-based
retriever. Swap in a real embedding model (e.g., sentence-transformers,
or a hosted embedding API) by replacing `fit`/`retrieve` internals only
— the rest of the pipeline doesn't need to change.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_base import DOCUMENTS


class Retriever:
    def __init__(self, documents=DOCUMENTS):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(
            [d["text"] for d in documents]
        )

    def retrieve(self, query: str, top_k: int = 2):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix)[0]
        ranked = sorted(
            zip(self.documents, scores), key=lambda x: x[1], reverse=True
        )
        return [
            {"id": doc["id"], "text": doc["text"], "score": round(float(s), 3)}
            for doc, s in ranked[:top_k]
            if s > 0
        ]
