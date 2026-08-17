"""
FINAL INTEGRATED PIPELINE: real vector DB retrieval + planner +
SLM agent pool, all connected.

Includes TWO guards, in order:
  1. Compound-query splitting: a query with multiple distinct parts
     (e.g. "What did X find, and what is Y?") gets split BEFORE
     retrieval, and each part is retrieved separately -- this fixes
     the observed failure where a single embedding for a two-part
     question let one sub-topic dominate and starved out the other
     entirely (Amsterdam content retrieved, MAST silently dropped).
  2. Retrieval-confidence guard: if the top score for ANY part is
     below MIN_RELIABLE_SCORE, that part is flagged as unanswered
     rather than silently omitted or answered from irrelevant context.
"""

import re

from embeddings import SentenceTransformerEmbedder
from vector_store import VectorStore
from planner import run_planned_pipeline

MIN_RELIABLE_SCORE = 0.30

embedder = SentenceTransformerEmbedder()
VECTOR_STORE = VectorStore(embedder)
VECTOR_STORE.load("./my_index")


def split_compound_query(query: str) -> list:
    """Naive but effective split on common compound-question
    connectors. Not a real model call (mocked, deliberately simple) --
    marked below for the real upgrade path.
    >>> REPLACE WITH REAL MODEL CALL <<<
    A real version would use a small model to properly split a
    compound question into independent sub-questions, handling cases
    this regex can't (e.g. no explicit "and", nested clauses).
    """
    parts = re.split(r",?\s+and\s+what\s+is\s+|,?\s+and\s+what\s+are\s+", query, flags=re.IGNORECASE)
    if len(parts) > 1:
        # Re-attach "what is"/"what are" phrasing lost by the split,
        # so each part still reads as a complete question
        parts = [parts[0]] + [f"What is {p}" if not p.lower().startswith("what") else p for p in parts[1:]]
    return [p.strip() for p in parts if p.strip()]


def retrieve_for_query(query: str, top_k: int = 3) -> dict:
    """Retrieves for ONE query part, applying the confidence guard
    to that part specifically."""
    chunks = VECTOR_STORE.search(query, top_k=top_k)
    top_score = chunks[0]["score"] if chunks else 0.0
    reliable = top_score >= MIN_RELIABLE_SCORE
    return {"query": query, "chunks": chunks, "top_score": top_score, "reliable": reliable}


def run_full_pipeline_v2(query: str, top_k: int = 3):
    sub_queries = split_compound_query(query)
    retrievals = [retrieve_for_query(q, top_k=top_k) for q in sub_queries]

    log = []
    if len(sub_queries) > 1:
        log.append(f"Compound query detected -> split into {len(sub_queries)} parts: {sub_queries}")

    reliable_retrievals = [r for r in retrievals if r["reliable"]]
    unreliable_parts = [r["query"] for r in retrievals if not r["reliable"]]

    if not reliable_retrievals:
        return {
            "query": query,
            "retrieved_chunks": [],
            "task_log": log + [f"RETRIEVAL WARNING: no part of this query found "
                                f"reliable context (threshold={MIN_RELIABLE_SCORE})."],
            "results": {},
            "final_answer": (
                "[No confident answer generated] None of the retrieved context "
                "was relevant enough to this question. This usually means the "
                "question is outside what's in the current knowledge base."
            ),
        }

    if unreliable_parts:
        log.append(f"WARNING: no reliable context found for: {unreliable_parts} "
                    f"-- these part(s) of the question will be skipped, not guessed at.")

    # KEY FIX: run planning + extraction SEPARATELY per sub-query, using
    # only that sub-query's own retrieved context. This stops content
    # from two different sources (e.g. two different papers) getting
    # flattened together and misattributed to one another -- each
    # sub-question gets its own clean, correctly-sourced answer.
    all_chunks = []
    per_part_answers = []
    for r in reliable_retrievals:
        part_context = " ".join(c["text"] for c in r["chunks"])
        all_chunks.extend(r["chunks"])
        part_result = run_planned_pipeline(r["query"], part_context)
        log.append(f"-- Sub-question: '{r['query']}' --")
        log.extend(part_result["task_log"])
        per_part_answers.append((r["query"], part_result["final_answer"]))

    if len(per_part_answers) > 1:
        final_answer = "\n\n".join(
            f"### Regarding: {q}\n{a}" for q, a in per_part_answers
        )
    else:
        final_answer = per_part_answers[0][1]

    return {
        "query": query,
        "retrieved_chunks": all_chunks,
        "task_log": log,
        "results": {q: a for q, a in per_part_answers},
        "final_answer": final_answer,
    }


if __name__ == "__main__":
    query = (
        "What did the Amsterdam paper find about orchestrator reasoning "
        "versus sub-agent size, and what is the MAST failure taxonomy?"
    )

    print("=" * 70)
    print("FINAL PIPELINE — Vector DB Retrieval + Planner + SLM Agents")
    print("=" * 70)

    result = run_full_pipeline_v2(query)

    print(f"\nQuery: {result['query']}\n")

    print("Step 1 - Retrieved chunks (via FAISS vector search, per sub-query):")
    for c in result["retrieved_chunks"]:
        print(f"  [{c['id']}] (score={c['score']}) {c['text'][:70]}...")

    print("\nStep 2 - Planner + routing log:")
    for entry in result["task_log"]:
        print(f"  - {entry}")

    print("\nStep 3 - Final synthesized answer:")
    print(result["final_answer"])