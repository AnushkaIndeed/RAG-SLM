"""
TASK REGISTRY — a declarative, single source of truth for which agent(s)
handle which task type, replacing the informal list-scan that used to
live inside planner.py's route_task() function.

Why this matters (per the meeting notes: "write registry for tasks"):
before, adding/removing an agent meant editing AGENT_POOL and hoping
the routing logic still worked. Now, the registry IS the routing
logic's source of truth — agents register their capabilities here
explicitly, in priority order, and routing just walks the registry.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskRegistration:
    task_type: str
    is_atomic: bool                # per meeting notes: some tasks must
                                    # NEVER be decomposed further
                                    # (summarization, keypoints, math, code)
    agents_in_priority_order: list = field(default_factory=list)
   
    complex_agents_in_priority_order: list = field(default_factory=list)
    
    notes: str = ""


TASK_REGISTRY = {
    "extraction": TaskRegistration(
        task_type="extraction",
        is_atomic=False,
        agents_in_priority_order=["TextProcessingSLM-qwen", "GeneralistSLM-8B"],
        complex_agents_in_priority_order=["GeneralistSLM-8B", "TextProcessingSLM-qwen"],
        notes="Merged into TextProcessingSLM per meeting notes.",
    ),
    "summarization": TaskRegistration(
        task_type="summarization",
        is_atomic=True,
        agents_in_priority_order=["TextProcessingSLM-qwen", "TextProcessingSLM-gemma", "GeneralistSLM-8B"],
        complex_agents_in_priority_order=["GeneralistSLM-8B", "TextProcessingSLM-qwen"],
        notes="ATOMIC: never split into per-sentence sub-tasks.",
    ),
    "keypoints": TaskRegistration(
        task_type="keypoints",
        is_atomic=True,
        agents_in_priority_order=["TextProcessingSLM-qwen", "TextProcessingSLM-gemma", "GeneralistSLM-8B"],
        complex_agents_in_priority_order=["GeneralistSLM-8B", "TextProcessingSLM-qwen"],
        notes="ATOMIC: never split into per-fact sub-tasks.",
    ),
    "skill_extraction": TaskRegistration(
        task_type="skill_extraction",
        is_atomic=True,
        agents_in_priority_order=["TextProcessingSLM-qwen", "GeneralistSLM-8B"],
        complex_agents_in_priority_order=["GeneralistSLM-8B", "TextProcessingSLM-qwen"],
        notes="ATOMIC. Extracts required skills from a GD/JD document. "
              "Step 1 of the resume-matching example from the meeting notes. "
              "GD topics are often broad/open-ended -> planner usually marks "
              "this 'complex', routing to the bigger generalist model first.",
    ),
    "skill_matching": TaskRegistration(
        task_type="skill_matching",
        is_atomic=True,
        agents_in_priority_order=["TextProcessingSLM-qwen", "GeneralistSLM-8B"],
        # Deliberately NO complex_agents list: matching against a
        # resume is a narrow, grounded comparison task even when the
        # GD topic itself was vague -- per meeting notes, this is the
        # "particular" half of the pipeline and should stay on the
        # smaller model to keep answers tightly grounded, not biased
        # by a bigger model's broader (and looser) associations.
        notes="ATOMIC. Compares extracted GD skills against resume skills. "
              "Depends on 'skill_extraction' output -- planner.py injects "
              "that result into this task's payload automatically. Must "
              "stay strongly grounded in the resume text (no invented skills).",
    ),
    "qa": TaskRegistration(
        task_type="qa",
        is_atomic=False,
        agents_in_priority_order=["GeneralistSLM-8B"],
        notes="No narrow specialist covers this on purpose - tests fallback routing.",
    ),
    "code": TaskRegistration(
        task_type="code",
        is_atomic=True,
        agents_in_priority_order=["CoderSLM", "GeneralistSLM-8B"],
        notes="ATOMIC: one full code-gen call, not split into per-function sub-tasks.",
    ),
    "math": TaskRegistration(
        task_type="math",
        is_atomic=True,
        agents_in_priority_order=["CalculatorTool"],
        notes="ATOMIC. Deliberately routed to a deterministic tool, not any SLM.",
    ),
}


def get_registration(task_type: str) -> Optional[TaskRegistration]:
    return TASK_REGISTRY.get(task_type)


def is_atomic(task_type: str) -> bool:
    reg = get_registration(task_type)
    return reg.is_atomic if reg else False


def get_agent_priority_list(task_type: str, complex_or_vague: bool = False) -> list:
    """Per meeting notes: a task marked complex/vague by the planner
    uses complex_agents_in_priority_order (biased toward the bigger
    generalist model) instead of the normal specialist-first order."""
    reg = get_registration(task_type)
    if not reg:
        return []
    if complex_or_vague and reg.complex_agents_in_priority_order:
        return reg.complex_agents_in_priority_order
    return reg.agents_in_priority_order