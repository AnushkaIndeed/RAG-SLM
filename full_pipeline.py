"""
FINAL INTEGRATED PIPELINE: real vector DB retrieval + planner +
SLM agent pool, all connected.

UPGRADED (this version):
  1. QUERY ANALYSIS is now a real model call, not a regex. The model
     itself decides whether a query is ONE topic (however long or
     detailed) or MULTIPLE distinct topics needing separate research
     -- this is "the planner knows which part to break into multiple
     parts and which to just answer as it is."
  2. Each part still gets its own retrieval + confidence guard + task
     planning/execution, exactly as before -- this is what keeps
     different sources from getting blended/misattributed.
  3. NEW: a SYNTHESIS step merges every part's answer into ONE
     coherent final answer via a real model call, instead of just
     concatenating "### Regarding: X" sections. The synthesis prompt
     is explicitly told not to add any new facts -- only reorganize
     and connect what's already been established, to avoid
     introducing ungrounded content at the merge step.
"""

import json

from embeddings import SentenceTransformerEmbedder
from vector_store import VectorStore
from planner import run_planned_pipeline, PLANNER_MODEL
from slm_agents import call_ollama

MIN_RELIABLE_SCORE = 0.30
SYNTHESIS_MODEL = "qwen3:4b"   

embedder = SentenceTransformerEmbedder()
VECTOR_STORE = VectorStore(embedder)
VECTOR_STORE.load("./my_index")


QUERY_ANALYSIS_PROMPT = """You analyze a user's question to decide how
to research it.

Decide: does this question contain MULTIPLE DISTINCT, INDEPENDENT
topics that would each need SEPARATE research/sources to answer well?
Or is it ONE cohesive question, even if it's long, detailed, or has
several clauses about the SAME underlying topic?

Rules:
- Do NOT split just because a question is long or has multiple
  clauses -- only split when the parts are genuinely about different
  subjects/sources (e.g. two different papers, two unrelated topics).
- If it's one topic, return exactly one part: the original question,
  unchanged.
- If it's multiple distinct topics, return each as its own complete,
  standalone question (re-attach words like "what is" if the split
  would otherwise leave a sentence fragment).

Respond with ONLY a JSON object, no explanation, no markdown fences.
Example (single topic): {"parts": ["What is the MAST failure taxonomy?"]}
Example (multiple topics): {"parts": ["What did the Amsterdam paper find about orchestrator reasoning?", "What is the MAST failure taxonomy?"]}

User question: """


def analyze_query(query: str) -> tuple:
    """Real model call replacing the old regex split. Returns
    (parts, raw_output) so the raw decision is visible in the debug
    trace -- same principle as the planner's own decomposition."""
    raw_output = call_ollama(PLANNER_MODEL, QUERY_ANALYSIS_PROMPT + query)
    try:
        cleaned = raw_output.strip().strip("`").replace("json\n", "")
        parsed = json.loads(cleaned)
        parts = [p.strip() for p in parsed.get("parts", []) if p.strip()]
        if parts:
            return parts, raw_output
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    # Fallback: if the model's output couldn't be parsed, treat the
    # whole query as a single part rather than crashing or guessing
    # at a split.
    return [query], raw_output + "\n[PARSE FAILED - treated as single topic]"


def retrieve_for_query(query: str, top_k: int = 3) -> dict:
    chunks = VECTOR_STORE.search(query, top_k=top_k)
    top_score = chunks[0]["score"] if chunks else 0.0
    reliable = top_score >= MIN_RELIABLE_SCORE
    return {"query": query, "chunks": chunks, "top_score": top_score, "reliable": reliable}


def synthesize_final_answer(original_query: str, per_part_answers: list) -> str:
    """Merges every part's answer into ONE coherent final answer.
    If there's only one part, skip the model call entirely -- nothing
    to merge, and it avoids an unnecessary extra call."""
    if len(per_part_answers) == 1:
        return per_part_answers[0][1]

    parts_text = "\n\n".join(f"[About: {q}]\n{a}" for q, a in per_part_answers)
    prompt = (
        f"The user asked: \"{original_query}\"\n\n"
        f"Below are separately-researched answers to the different "
        f"parts of this question. Merge them into ONE single, "
        f"coherent, well-organized answer that reads naturally and "
        f"addresses the full original question.\n\n"
        f"IMPORTANT - stay strictly grounded: do NOT add any new "
        f"facts, claims, or details that aren't already present in "
        f"the text below. Only reorganize, connect, and smooth the "
        f"transitions between the existing content. It is fine to use "
        f"headings or clear paragraph breaks per topic if that reads "
        f"better, but every fact must come from the material given.\n\n"
        f"{parts_text}"
    )
    return call_ollama(SYNTHESIS_MODEL, prompt)


def run_full_pipeline_v2(query: str, top_k: int = 3):
    log = []
    sub_queries, raw_analysis = analyze_query(query)
    log.append(f"[QUERY_ANALYSIS] ({PLANNER_MODEL}) raw output: {raw_analysis}")
    if len(sub_queries) > 1:
        log.append(f"[QUERY_ANALYSIS] Multiple distinct topics detected -> "
                    f"split into {len(sub_queries)} parts: {sub_queries}")
    else:
        log.append(f"[QUERY_ANALYSIS] Single cohesive topic -> answered as one question")

    retrievals = [retrieve_for_query(q, top_k=top_k) for q in sub_queries]
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

    all_chunks = []
    per_part_answers = []
    for r in reliable_retrievals:
        part_context = " ".join(c["text"] for c in r["chunks"])
        all_chunks.extend(r["chunks"])
        part_result = run_planned_pipeline(r["query"], part_context)
        log.append(f"-- Sub-question: '{r['query']}' --")
        log.extend(part_result["task_log"])
        per_part_answers.append((r["query"], part_result["final_answer"]))

    final_answer = synthesize_final_answer(query, per_part_answers)
    if len(per_part_answers) > 1:
        log.append(f"[SYNTHESIS] ({SYNTHESIS_MODEL}) merged {len(per_part_answers)} "
                    f"part-answers into one coherent final answer")

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