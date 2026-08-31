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
  - "facts_needed": a list of RAW facts to extract from documents,
    each with:
      "label": a short snake_case identifier for this fact
      "search_query": a natural question to retrieve/extract this
                        ONE fact (phrase it so it would match a
                        document specifically ABOUT this fact)
      "fact_description": a short description of what to extract
                           (used to prompt the extractor)
  - "computation_steps": an ORDERED list of calculation steps. Each
    step has:
      "label": a short snake_case identifier for THIS STEP'S result
               (later steps, or the final answer, can reference this
               label the same way they'd reference a raw fact)
      "operation": one of "add", "subtract", "multiply", "divide", "compare"
      "operand_a_label" / "operand_b_label": labels to combine -- each
        MUST be either a label from facts_needed, OR the label of an
        EARLIER step in this same computation_steps list (never a
        later one).
    Use MULTIPLE steps whenever the question needs more than one
    calculation before it's answerable (e.g. compute two derived
    values first, THEN compare them) -- do not try to force a
    multi-step calculation into a single operation.

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

  multi_hop (single step): "Find the population of Bihar and using
  the total income of Bihar, calculate the average income per
  person." ->
  {"type": "multi_hop",
   "facts_needed": [
     {"label": "population_bihar", "search_query": "What is the population of Bihar?", "fact_description": "the total population number of Bihar"},
     {"label": "income_bihar", "search_query": "What is the total income of Bihar?", "fact_description": "the total income figure for Bihar"}
   ],
   "computation_steps": [
     {"label": "avg_income_bihar", "operation": "divide", "operand_a_label": "income_bihar", "operand_b_label": "population_bihar"}
   ]}

  multi_hop (CHAINED, multiple steps -- e.g. a comparison between two
  states, where each side must FIRST be derived before comparing):
  "Compare the average income per person in Kerala versus Uttar
  Pradesh" ->
  {"type": "multi_hop",
   "facts_needed": [
     {"label": "population_kerala", "search_query": "What is the population of Kerala?", "fact_description": "the total population number of Kerala"},
     {"label": "income_kerala", "search_query": "What is the total income of Kerala?", "fact_description": "the total income figure for Kerala"},
     {"label": "population_up", "search_query": "What is the population of Uttar Pradesh?", "fact_description": "the total population number of Uttar Pradesh"},
     {"label": "income_up", "search_query": "What is the total income of Uttar Pradesh?", "fact_description": "the total income figure for Uttar Pradesh"}
   ],
   "computation_steps": [
     {"label": "avg_income_kerala", "operation": "divide", "operand_a_label": "income_kerala", "operand_b_label": "population_kerala"},
     {"label": "avg_income_up", "operation": "divide", "operand_a_label": "income_up", "operand_b_label": "population_up"},
     {"label": "comparison_result", "operation": "compare", "operand_a_label": "avg_income_kerala", "operand_b_label": "avg_income_up"}
   ]}

  The key test for multi_hop specifically: could you even ATTEMPT to
  answer the second part without already knowing the numeric result
  of the first part? If yes -> multi_hop. If both parts are answerable
  independently in any order -> independent. And: never ask a
  search_query for a value that would have to be CALCULATED (like
  "average income per person") unless that exact phrase is likely to
  appear verbatim in a document -- if it's derived, extract the RAW
  components instead (population, total income) and add a
  computation_step to derive it.

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
    needed fact (each from its own best-matching document), then runs
    an ORDERED CHAIN of computation steps -- later steps can consume
    earlier steps' results by label, which is what makes nested
    calculations (derive A, derive B, then compare A vs B) possible,
    not just a single flat operation."""
    log = [f"[MULTI_HOP_PLAN] {json.dumps(plan)}"]
    known_values = {}   # accumulates BOTH extracted facts AND computed step results
    all_chunks = []
    fact_summaries = []

    for fact in plan.get("facts_needed", []):
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

        known_values[label] = cleaned_value
        fact_summaries.append(f"{label} = {cleaned_value} (source: {chunks[0]['id']})")

    # Backward-compat: accept the old flat single-operation format too
    computation_steps = plan.get("computation_steps")
    if not computation_steps and plan.get("operation"):
        if plan["operation"] == "divide" and "numerator_label" in plan:
            computation_steps = [{"label": "result", "operation": "divide",
                                   "operand_a_label": plan["numerator_label"],
                                   "operand_b_label": plan["denominator_label"]}]
        elif "value_a_label" in plan:
            computation_steps = [{"label": "result", "operation": plan["operation"],
                                   "operand_a_label": plan["value_a_label"],
                                   "operand_b_label": plan["value_b_label"]}]

    if not computation_steps:
        log.append("[MULTI_HOP_FAIL] plan had no computation_steps and no "
                   "legacy operation fields -- nothing to compute")
        return {
            "query": query, "task_log": log, "retrieved_chunks": all_chunks,
            "results": {}, "final_answer":
                "[No confident answer generated] The plan didn't specify how "
                "to combine the extracted facts.",
        }

    step_results = []
    for step in computation_steps:
        label_a, label_b = step["operand_a_label"], step["operand_b_label"]
        if label_a not in known_values or label_b not in known_values:
            missing = [l for l in (label_a, label_b) if l not in known_values]
            log.append(f"[MULTI_HOP_STEP_FAIL] step '{step['label']}' references "
                       f"unknown label(s) {missing} -- plan referenced a fact/step "
                       f"that was never extracted or computed")
            return {
                "query": query, "task_log": log, "retrieved_chunks": all_chunks,
                "results": {}, "final_answer":
                    f"[No confident answer generated] The computation plan "
                    f"referenced {missing}, which was never extracted.",
            }

        calc_payload = {
            "operation": step["operation"],
            "value_a": known_values[label_a], "value_b": known_values[label_b],
            "label_a": label_a, "label_b": label_b,
        }
        display, numeric_result = CALCULATOR_AGENT.compute(calc_payload)
        log.append(f"[MULTI_HOP_COMPUTE] step '{step['label']}': {display}")
        step_results.append(display)

        if numeric_result is None:
            log.append(f"[MULTI_HOP_STEP_FAIL] step '{step['label']}' produced no "
                       f"usable numeric result -- cannot continue the chain")
            break
        known_values[step["label"]] = numeric_result   # available to LATER steps

    final_answer = (
        f"Based on the extracted facts:\n" + "\n".join(f"- {s}" for s in fact_summaries) +
        f"\n\nCalculation steps:\n" + "\n".join(f"- {s}" for s in step_results)
    )

    return {
        "query": query,
        "task_log": log,
        "retrieved_chunks": all_chunks,
        "results": {"multi_hop_computation": " | ".join(step_results)},
        "final_answer": final_answer,
    }