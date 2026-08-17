"""
Loads real documents from a folder (.txt and .pdf supported) and
chunks them into smaller pieces before embedding.

WHY CHUNKING MATTERS: embedding an entire long document as a single
vector compresses everything into one point in vector space -- a
10-page PDF and a 1-paragraph note end up represented with the same
amount of "space" in the vector, so long documents lose detail. Chunking
breaks documents into ~200-500 word pieces, each gets its own vector,
so retrieval can point to the SPECIFIC section that answers a query,
not just "somewhere in this huge document."
"""

import os
import re
from pathlib import Path


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf_file(path: str) -> str:
    from pypdf import PdfReader  # local import, only needed if PDFs are used
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(folder_path: str) -> list:
    """Returns [{"id": filename, "text": full_text}] for every .txt/.pdf
    file in the folder."""
    docs = []
    for path in Path(folder_path).glob("*"):
        if path.suffix.lower() == ".txt":
            docs.append({"id": path.name, "text": load_text_file(str(path))})
        elif path.suffix.lower() == ".pdf":
            docs.append({"id": path.name, "text": load_pdf_file(str(path))})
    return docs


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """Splits text into ~chunk_size-word pieces with `overlap` words
    shared between consecutive chunks (overlap prevents a sentence that
    straddles a chunk boundary from losing context on either side)."""
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # step forward, leaving `overlap` words repeated
    return chunks


def load_and_chunk_documents(folder_path: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """End-to-end: load every file in a folder, chunk each one, and
    return a flat list of {"id": "<filename>_chunk<N>", "text": ...}
    ready to hand to the embedder + vector store."""
    raw_docs = load_documents(folder_path)
    chunked = []
    for doc in raw_docs:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            chunked.append({"id": f"{doc['id']}_chunk{i}", "text": piece})
    return chunked


if __name__ == "__main__":
    # Quick self-test using in-memory text (no real files needed to
    # verify the chunking logic works correctly)
    sample = " ".join([f"word{i}" for i in range(700)])  # 700 fake words
    chunks = chunk_text(sample, chunk_size=300, overlap=50)
    print(f"700-word document -> {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        word_count = len(c.split())
        print(f"  Chunk {i}: {word_count} words, starts with '{c[:30]}...'")