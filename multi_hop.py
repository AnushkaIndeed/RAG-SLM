"""
MULTI-HOP QUERY HANDLING :

    "Find Bihar's population, and using Bihar's total income,
     calculate the average income per person."

This is DIFFERENT from full_pipeline.py's compound-query splitting.
That splits INDEPENDENT topics (answerable separately, merged only at
the text level). This handles DEPENDENT facts: fact A and fact B may
live in completely different documents, but the FINAL ANSWER can't be
computed without both actual values.

Pipeline, per multi-hop query:

    1. detect_multi_hop()   -- real model call: is this independent
                                topics, a single simple question, or a
                                multi-hop computation needing 2+ facts
                                combined?
    2. For each needed fact: retrieve from the index (each fact's own
       phrasing naturally finds its OWN best-matching document --
       population question -> population doc, income question ->
       income doc, no manual document-routing needed)
    3. For each needed fact: run 'numeric_extraction' to pull ONE
       clean number out of that fact's retrieved context
    4. Feed the actual extracted numbers into CalculatorTool
       (deterministic -- math is never left to an SLM to guess, same
       principle used everywhere else in this project)
    5. Synthesize a final answer that states both extracted facts and
       the computed result, grounded in exactly what was extracted
"""

import json

from vector_store import VectorStore
from slm_agents import call_ollama, TEXT_AGENT_QWEN, CALCULATOR_AGENT

MIN_RELIABLE_SCORE = 0.30
MULTI_HOP_MODEL = "qwen3:8b"   

MULTI_HOP_DETECTION_PROMPT = """You analyze a user's question to decide
its TYPE, for a system that has access to a document index.

Decide ONE of:
  "simple"      - one self-contained question, answerable from one
                   piece of retrieved context.
  "independent" - multiple DISTINCT topics that can each be answered
                   SEPARATELY and don't depend on each other's answer.
  "multi_hop"   - the question requires extracting TWO OR MORE
                   specific numeric facts (possibly from DIFFERENT
                   documents) and then COMBINING them with a
                   calculation to answer the actual question. The
                   final answer is NOT computable without both facts.

For "simple", also output "parts": [the original question, unchanged].

For "independent", also output "parts": a list of each distinct
question as its own complete, standalone question (re-attach words
like "what is" if the split would otherwise leave a fragment). Do NOT
split just because a question is long or has multiple clauses about
the SAME topic -- only split genuinely different subjects/sources.

For "multi_hop" ONLY, also output:
  - "facts_needed": a list of objects, each with:
      "label": a short snake_case identifier for this fact
      "search_query": a natural question to retrieve/extract this
                        ONE fact (phrase it so it would match a
                        document specifically ABOUT this fact)
      "fact_description": a short description of what to extract
                           (used to prompt the extractor)
  - "operation": one of "add", "subtract", "multiply", "divide"
  - "numerator_label" / "denominator_label" (for divide), OR
    "value_a_label" / "value_b_label" (for add/subtract/multiply)
    -- must match one of the labels in facts_needed

WORKED EXAMPLES to anchor your judgment (use these as reference
points, not just the definitions above):

  simple: "What is the MAST failure taxonomy and why does it matter
  for debugging?" -> {"type": "simple", "parts": ["What is the MAST
  failure taxonomy and why does it matter for debugging?"]}

  independent: "What did the Amsterdam paper find about orchestrator
  reasoning, and what is the MAST failure taxonomy?" ->
  {"type": "independent", "parts": ["What did the Amsterdam paper find
  about orchestrator reasoning?", "What is the MAST failure
  taxonomy?"]}

  multi_hop: "Find the population of Bihar and using the total income
  of Bihar, calculate the average income per person." ->
  {"type": "multi_hop",
   "facts_needed": [
     {"label": "population_bihar", "search_query": "What is the population of Bihar?", "fact_description": "the total population number of Bihar"},
     {"label": "income_bihar", "search_query": "What is the total income of Bihar?", "fact_description": "the total income figure for Bihar"}
   ],
   "operation": "divide",
   "numerator_label": "income_bihar",
   "denominator_label": "population_bihar"}

  The key test for multi_hop specifically: could you even ATTEMPT to
  answer the second part without already knowing the numeric result
  of the first part? If yes -> multi_hop. If both parts are answerable
  independently in any order -> independent.

Respond with ONLY a JSON object, no explanation, no markdown fences.

User question: """


def detect_query_type(query: str) -> dict:
    """Real model call. Single unified decision replacing the old
    separate analyze_query()+detect_multi_hop() calls -- one judgment
    point instead of two that could disagree with each other. Falls
    back to {'type': 'simple', 'parts': [query]} if parsing fails."""
    raw = call_ollama(MULTI_HOP_MODEL, MULTI_HOP_DETECTION_PROMPT + query)
    try:
        cleaned = raw.strip().strip("`").replace("json\n", "")
        parsed = json.loads(cleaned)
        if parsed.get("type") in ("simple", "independent", "multi_hop"):
            parsed["_raw"] = raw
            return parsed
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return {"type": "simple", "parts": [query], "_raw": raw, "_parse_failed": True}


def _clean_number(raw_text: str):
    """Extraction agents are prompted to return ONLY a number, but
    small models sometimes add stray text anyway. Second line of
    defense: pull the first number-looking substring out with a
    simple scan, rather than trusting the prompt alone."""
    import re
    if "NOT_FOUND" in raw_text.upper():
        return None
    match = re.search(r"-?\d[\d,]*\.?\d*", raw_text)
    if not match:
        return None
    return match.group(0).replace(",", "")


def run_multi_hop_pipeline(query: str, plan: dict, vector_store, top_k: int = 3) -> dict:
    """Executes a detected multi-hop plan: retrieve + extract each
    needed fact (each from its own best-matching document), then
    compute deterministically, then synthesize a grounded answer."""
    log = [f"[MULTI_HOP_PLAN] {json.dumps(plan)}"]
    extracted_values = {}
    all_chunks = []
    fact_summaries = []

    for fact in plan["facts_needed"]:
        label = fact["label"]
        search_query = fact["search_query"]

        chunks = vector_store.search(search_query, top_k=top_k)
        top_score = chunks[0]["score"] if chunks else 0.0
        all_chunks.extend(chunks)

        if top_score < MIN_RELIABLE_SCORE:
            log.append(f"[MULTI_HOP_FACT_FAIL] '{label}' -> no reliable context "
                       f"found (top score {top_score:.2f}), cannot proceed")
            return {
                "query": query, "task_log": log, "retrieved_chunks": all_chunks,
                "results": {}, "final_answer":
                    f"[No confident answer generated] Could not find reliable "
                    f"information for '{search_query}' (needed to compute this "
                    f"answer). Top retrieval score was only {top_score:.2f}.",
            }

        context = " ".join(c["text"] for c in chunks)
        raw_extraction = TEXT_AGENT_QWEN.run("numeric_extraction", {
            "context": context, "fact_description": fact["fact_description"],
        })
        cleaned_value = _clean_number(raw_extraction)
        log.append(f"[MULTI_HOP_EXTRACT] '{label}' <- {raw_extraction.strip()} "
                   f"-> parsed as {cleaned_value}")

        if cleaned_value is None:
            log.append(f"[MULTI_HOP_FACT_FAIL] '{label}' -> extraction did not "
                       f"return a usable number, cannot proceed")
            return {
                "query": query, "task_log": log, "retrieved_chunks": all_chunks,
                "results": {}, "final_answer":
                    f"[No confident answer generated] Found context for "
                    f"'{search_query}' but could not extract a clean numeric "
                    f"value from it.",
            }

        extracted_values[label] = cleaned_value
        fact_summaries.append(f"{label} = {cleaned_value} (source: {chunks[0]['id']})")

    # Deterministic computation -- never left to an SLM to guess
    calc_payload = {"operation": plan["operation"]}
    if plan["operation"] == "divide":
        calc_payload["value_a"] = extracted_values[plan["numerator_label"]]
        calc_payload["value_b"] = extracted_values[plan["denominator_label"]]
        calc_payload["label_a"] = plan["numerator_label"]
        calc_payload["label_b"] = plan["denominator_label"]
    else:
        calc_payload["value_a"] = extracted_values[plan["value_a_label"]]
        calc_payload["value_b"] = extracted_values[plan["value_b_label"]]
        calc_payload["label_a"] = plan["value_a_label"]
        calc_payload["label_b"] = plan["value_b_label"]

    calc_result = CALCULATOR_AGENT.run("math", calc_payload)
    log.append(f"[MULTI_HOP_COMPUTE] {calc_result}")

    final_answer = (
        f"Based on the extracted facts:\n" + "\n".join(f"- {s}" for s in fact_summaries) +
        f"\n\n{calc_result}"
    )

    return {
        "query": query,
        "task_log": log,
        "retrieved_chunks": all_chunks,
        "results": {"multi_hop_computation": calc_result},
        "final_answer": final_answer,
    }