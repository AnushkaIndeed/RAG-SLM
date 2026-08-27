"""
TEST SUITE — validates query splitting, complexity classification, and
routing behavior against expected outcomes, across a fixed battery of
queries including compound and complex ones.

Two modes:
  1. `run_correctness_suite()` — runs each test case ONCE, checks the
     actual split/complexity/routing decisions against what's expected,
     reports PASS/FAIL per case.
  2. `run_consistency_check()` — runs the SAME query N times and
     reports whether the complexity classification is STABLE across
     runs. This directly diagnoses "is the vague/particular
     distinction actually working, or just inconsistent" -- if a
     query gets classified differently run to run, that's the root
     cause, not a routing bug.

Requires a live Ollama server with the models pulled (this is real
model output, not mocked) and a built vector index at ./my_index.
Run with: python3 test_suite.py
"""

import json
from collections import Counter

from full_pipeline import run_full_pipeline_v2, analyze_query
from planner import decompose_query


# ---------------------------------------------------------------------------
# Test battery -- mix of single-topic, compound, vague, and specific queries
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "name": "single_specific",
        "query": "What GAIA score did the 8B orchestrator achieve in the Amsterdam paper?",
        "expected_split_count": 1,
        "expected_complexities": ["simple"],
    },
    {
        "name": "single_vague",
        "query": "Give me a broad overview of why small language models matter for the future of AI agents",
        "expected_split_count": 1,
        "expected_complexities": ["complex"],
    },
    {
        "name": "two_distinct_topics",
        "query": "What did the Amsterdam paper find about orchestrator reasoning, and what is the MAST failure taxonomy?",
        "expected_split_count": 2,
        "expected_complexities": ["simple", "simple"],
    },
    {
        "name": "mixed_vague_and_specific",
        "query": "Give me a broad take on multi-agent system design philosophy, and what specifically does MAST say the verification-gap failure rate was?",
        "expected_split_count": 2,
        "expected_complexities": ["complex", "simple"],
    },
    {
        "name": "long_but_single_topic",
        "query": "What is the MAST failure taxonomy and why does it matter for debugging multi-agent systems?",
        "expected_split_count": 1,   # should NOT split just because it's long/multi-clause
        "expected_complexities": ["simple"],
    },
    {
        "name": "three_way_split",
        "query": "What is the MAST taxonomy, what did the Amsterdam paper find about orchestrator reasoning, and what does MapCoder-Lite say about small-model role collapse?",
        "expected_split_count": 3,
        "expected_complexities": ["simple", "simple", "simple"],
    },
    {
        "name": "atomic_task_with_math",
        "query": "Summarize the MAST paper and calculate the payback period for a $20000 system saving $2000/year",
        "expected_split_count": 1,
        "expected_task_types": ["summarization", "math"],  # must appear exactly once each
    },
]


def extract_debug_field(task_log: list, prefix: str) -> str:
    """Pulls the raw content of the first log line starting with a
    given tag, e.g. '[QUERY_ANALYSIS]'."""
    for line in task_log:
        if line.startswith(prefix):
            return line
    return ""


def run_correctness_suite():
    print("=" * 78)
    print("CORRECTNESS SUITE — one run per case")
    print("=" * 78)

    results = []
    for case in TEST_CASES:
        print(f"\n--- {case['name']} ---")
        print(f"Query: {case['query']}")

        result = run_full_pipeline_v2(case["query"])
        log = result["task_log"]

        # How many parts did it actually split into?
        analysis_line = extract_debug_field(log, "[QUERY_ANALYSIS] Multiple")
        actual_split_count = len(result["results"]) if result["results"] else 1
        # results dict is keyed by sub-question when split, by task type when not;
        # count sub-questions instead, more reliable:
        sub_question_lines = [l for l in log if l.startswith("-- Sub-question:")]
        actual_split_count = max(len(sub_question_lines), 1)

        split_pass = actual_split_count == case["expected_split_count"]
        print(f"  Split: expected {case['expected_split_count']}, got {actual_split_count} "
              f"{'✓' if split_pass else '✗ MISMATCH'}")

        # What complexity did each task actually get?
        actual_complexities = []
        for line in log:
            if line.startswith("[PLANNER_PARSED]"):
                # format: "[PLANNER_PARSED] N task(s): [('type', 'complexity'), ...]"
                try:
                    tuples_str = line.split(":", 1)[1].strip()
                    parsed_tuples = eval(tuples_str, {"__builtins__": {}})
                    actual_complexities.extend([c for _, c in parsed_tuples])
                except Exception:
                    pass

        complexity_pass = True
        if "expected_complexities" in case:
            complexity_pass = actual_complexities == case["expected_complexities"]
            print(f"  Complexity: expected {case['expected_complexities']}, got "
                  f"{actual_complexities} {'✓' if complexity_pass else '✗ MISMATCH'}")

        task_type_pass = True
        if "expected_task_types" in case:
            actual_types = list(result["results"].keys())
            task_type_pass = sorted(actual_types) == sorted(case["expected_task_types"])
            print(f"  Task types: expected {case['expected_task_types']}, got "
                  f"{actual_types} {'✓' if task_type_pass else '✗ MISMATCH'}")

        overall_pass = split_pass and complexity_pass and task_type_pass
        results.append({"name": case["name"], "pass": overall_pass})
        print(f"  RESULT: {'PASS' if overall_pass else 'FAIL'}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    passed = sum(1 for r in results if r["pass"])
    for r in results:
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['name']}")
    print(f"\n{passed}/{len(results)} passed")
    return results


def run_consistency_check(query: str, n_runs: int = 5):
    """Runs the SAME query n_runs times through the planner directly
    (skipping retrieval, using dummy context) and reports whether the
    complexity classification is stable. If it isn't, that's the real
    root cause of 'the vague/particular split doesn't seem to work' --
    it's not that the routing is broken, it's that the classification
    upstream of it isn't consistent."""
    print("\n" + "=" * 78)
    print(f"CONSISTENCY CHECK — running the same query {n_runs} times")
    print("=" * 78)
    print(f"Query: {query}\n")

    dummy_context = "Some representative context text the query would be answered against."
    complexity_votes = []

    for i in range(n_runs):
        tasks, raw = decompose_query(query, dummy_context)
        complexities = [t.get("complexity", "simple") for t in tasks]
        complexity_votes.append(tuple(complexities))
        print(f"  Run {i+1}: {complexities}")

    counts = Counter(complexity_votes)
    most_common, freq = counts.most_common(1)[0]
    stability = freq / n_runs * 100

    print(f"\n  Stability: {stability:.0f}% consistent "
          f"(most common result: {most_common}, seen {freq}/{n_runs} times)")
    if stability < 80:
        print("  ⚠ LOW STABILITY — the model is genuinely inconsistent on this "
              "query. Consider: more few-shot examples in the prompt, a lower "
              "temperature setting, or a larger planner model.")
    else:
        print("  ✓ Reasonably stable classification.")
    return stability


if __name__ == "__main__":
    run_correctness_suite()

    # Consistency check on the query type most likely to have been
    # ambiguous for the model -- a mixed vague+specific compound query
    run_consistency_check(
        "Give me a broad take on multi-agent system design philosophy, "
        "and what specifically does MAST say the verification-gap failure rate was?",
        n_runs=5,
    )