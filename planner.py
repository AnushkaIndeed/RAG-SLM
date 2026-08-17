"""
PLANNER / ORCHESTRATION LAYER — updated per meeting notes:

  1. Planning is now done by a REAL model call (qwen3:1.7b) instead of
     keyword matching -- this is the "see how SLM breaks the tasks"
     step. The model's raw output is logged so you can literally watch
     how a real small model decomposes a query, and compare that
     against the atomic-task rule to see if it agrees on its own or
     needs the hard rule to override it.

  2. Routing now walks TASK_REGISTRY instead of scanning AGENT_POOL by
     capability -- agents are looked up by name, in the registry's
     declared priority order, with automatic fallback to the next
     agent in that list (usually GeneralistSLM-8B) if the first fails
     or the task type isn't registered.

  3. Rule 2 (atomicity) is now checked against task_registry.is_atomic()
     as the single source of truth, instead of a hardcoded set living
     in this file.
"""

import json
import ollama

from slm_agents import AGENTS_BY_NAME, call_ollama
from task_registry import get_registration, is_atomic, get_agent_priority_list

PLANNER_MODEL = "qwen3:8b"
# Per meeting notes: "Planner agent 7b or 8b or 4b" -- to benchmark a
# different planner size, just change this one line. Nothing else in
# this file depends on the specific size.

PLANNER_SYSTEM_PROMPT = """You are a task planner for a multi-agent system.
Break the user's request into the MINIMUM set of task types needed.

Valid task types ONLY: extraction, summarization, keypoints, math, code,
qa, skill_extraction, skill_matching

RULES you must follow:
- summarization, keypoints, math, code, skill_extraction, and
  skill_matching are ATOMIC: each appears at most ONCE in your output,
  never split into smaller pieces (e.g. do NOT output "summarize
  paragraph 1", "summarize paragraph 2" - just one "summarization"
  entry covering everything).
- Only include a task type if the request actually needs it.
- If the request includes a calculation, extract the two relevant
  numbers into "cost" and "annual_savings" fields.
- For EACH task, set "complexity" to either "simple" or "complex".
  Mark a task "complex" when the request is VAGUE, OPEN-ENDED, or
  requires broad/general knowledge not likely to be narrowly covered
  by the provided context. Mark it "simple" when the request is
  PARTICULAR/SPECIFIC and clearly answerable from the given context.
  This decides which size of model handles the task, so get this right.
- If the request is about matching skills from a job description /
  group discussion (GD) topic against a resume, use "skill_extraction"
  first (pull required skills from the GD/JD text) and then
  "skill_matching" (compare those against the resume).

Respond with ONLY a JSON list, no explanation, no markdown fences.
Example: [{"type": "summarization", "complexity": "simple"},
{"type": "math", "complexity": "simple", "cost": 15000, "annual_savings": 1200}]
"""


def decompose_query(query: str, context: str) -> tuple:
    """Real model call. Returns (task_list, raw_model_output) so the
    raw output can be logged/inspected -- this is what lets you
    literally observe how the SLM breaks the task down."""
    prompt = f"{PLANNER_SYSTEM_PROMPT}\n\nUser request: {query}"
    raw_output = call_ollama(PLANNER_MODEL, prompt)

    try:
        # Strip common wrapping the model might add despite instructions
        cleaned = raw_output.strip().strip("`").replace("json\n", "")
        parsed = json.loads(cleaned)
        tasks = []
        for item in parsed:
            task_type = item.get("type")
            if not task_type:
                continue
            payload = {"context": context, "query": query}
            if task_type == "math":
                payload["cost"] = item.get("cost")
                payload["annual_savings"] = item.get("annual_savings")
            complexity = item.get("complexity", "simple")
            tasks.append({
                "type": task_type,
                "payload": payload,
                "complexity": complexity,
                "is_complex_or_vague": complexity == "complex",
            })
        if tasks:
            return tasks, raw_output
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # If parsing fails, fall back to a single extraction task so the
    # user still gets SOMETHING instead of a hard failure. This is
    # deliberately a "best effort" fallback, not a guaranteed answer.   
    fallback_tasks = [{"type": "extraction", "payload": {"context": context, "query": query}}]
    return fallback_tasks, raw_output + "\n[PARSE FAILED - fell back to single extraction task]"


def debug_step(log: list, stage: str, detail: str):
    """Structured debug trace (per meeting notes: 'add debugging
    section make the planner break steps and see them'). Every entry
    is tagged with a stage name so the trace can be grouped/filtered
    by whoever is reading it, not just a flat list of strings."""
    log.append({"stage": stage, "detail": detail})


def route_task(task: dict, log: list) -> str:
    task_type = task["type"]
    registration = get_registration(task_type)
    is_complex = task.get("is_complex_or_vague", False)

    if not registration:
        debug_step(log, "ROUTE_FAIL", f"'{task_type}' is not in the task registry at all")
        return f"[No registry entry for task type: {task_type}]"

    priority_list = get_agent_priority_list(task_type, complex_or_vague=is_complex)
    debug_step(log, "ROUTE_PLAN",
               f"'{task_type}' (complexity={task.get('complexity', 'simple')}) "
               f"-> priority list: {priority_list}")

    for agent_name in priority_list:
        agent = AGENTS_BY_NAME.get(agent_name)
        if agent is None:
            continue
        result = agent.run(task_type, task["payload"])
        if result.startswith("[ERROR]") or "[ERROR]" in result:
            debug_step(log, "ROUTE_RETRY", f"'{task_type}' -> {agent_name} FAILED, trying next")
            continue
        tag = "primary" if agent_name == priority_list[0] else "fallback"
        debug_step(log, "ROUTE_OK", f"'{task_type}' -> {agent_name} ({tag})")
        return result

    debug_step(log, "ROUTE_FAIL", f"all agents failed for '{task_type}': {priority_list}")
    return f"[All agents failed for task type: {task_type}]"


def run_planned_pipeline(query: str, context: str) -> dict:
    log = []
    tasks, raw_planner_output = decompose_query(query, context)
    debug_step(log, "PLANNER_RAW", f"({PLANNER_MODEL}) raw output: {raw_planner_output}")
    debug_step(log, "PLANNER_PARSED",
               f"{len(tasks)} task(s): "
               f"{[(t['type'], t.get('complexity', 'simple')) for t in tasks]}")

    # Second line of defense for Rule 2: even if the planner model
    # tries to over-decompose an atomic task type, duplicates collapse
    # into one.
    seen_atomic = set()
    filtered_tasks = []
    for task in tasks:
        if is_atomic(task["type"]):
            if task["type"] in seen_atomic:
                debug_step(log, "ATOMICITY_BLOCK",
                           f"planner tried to duplicate atomic task '{task['type']}' - skipped")
                continue
            seen_atomic.add(task["type"])
        filtered_tasks.append(task)

    results = {}
    for task in filtered_tasks:
        # Dependency handling for the resume/GD example: skill_matching
        # needs skill_extraction's output. This is a simple, explicit
        # special-case (not a general dependency graph) -- the two-step
        # "task 1 then task 2" pattern from the meeting notes doesn't
        # need more than this for now.
        if task["type"] == "skill_matching" and "skill_extraction" in results:
            task["payload"]["required_skills"] = results["skill_extraction"]
            debug_step(log, "DEPENDENCY",
                       "'skill_matching' received 'skill_extraction' output as input")
        results[task["type"]] = route_task(task, log)

    synthesis = "\n\n".join(f"-- {k.upper()} --\n{v}" for k, v in results.items())
    debug_trace = [f"[{e['stage']}] {e['detail']}" for e in log]  
    return {
        "query": query,
        "task_log": debug_trace,
        "debug_trace_structured": log,  
        "results": results,
        "final_answer": synthesis,
    }
