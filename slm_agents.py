"""
SLM agent pool, updated per meeting notes:
  - Extraction + summarization + keypoints merged into ONE agent
    (TextProcessingSLM), offered in TWO model-family variants (Qwen and
    Gemma) so they can be A/B compared on the same task -- this
    directly tests cross-model-family generalization, one of the open
    gaps flagged in the earlier research review.
  - Agent `.name` values now match the keys used in task_registry.py,
    so routing can look agents up by name instead of scanning
    capabilities.

REQUIRES Ollama with these models pulled:
    ollama pull qwen3:1.7b
    ollama pull qwen3:4b
    ollama pull qwen3:8b
    ollama pull gemma2:2b
"""

import ollama


def call_ollama(model: str, prompt: str) -> str:
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except ollama.ResponseError as e:
        return f"[ERROR] Ollama couldn't run '{model}': {e}\nFix: run `ollama pull {model}`."
    except Exception as e:
        return f"[ERROR] Couldn't reach Ollama for '{model}': {e}\nFix: make sure Ollama is running."


class BaseSLMAgent:
    name = "base"
    capabilities = []
    model_size = "unknown"
    ollama_model = None

    def can_handle(self, task_type: str) -> bool:
        return task_type in self.capabilities

    def run(self, task_type: str, payload: dict) -> str:
        raise NotImplementedError


class TextProcessingSLM(BaseSLMAgent):
    """MERGED agent (per meeting notes): handles extraction,
    summarization, AND keypoints in one class. Instantiated TWICE
    below with different underlying models (Qwen vs Gemma), so the
    planner can route the same task type to either variant and you
    can directly compare their outputs on identical input -- this is
    the cross-model-family test the earlier research review flagged
    as an open gap."""
    capabilities = ["extraction", "summarization", "keypoints",
                     "skill_extraction", "skill_matching"]

    def __init__(self, ollama_model: str, name: str, model_size: str):
        self.ollama_model = ollama_model
        self.name = name
        self.model_size = model_size

    def run(self, task_type: str, payload: dict) -> str:
        context = payload.get("context", "")
        if task_type == "extraction":
            prompt = (
                f"Extract the key facts from the following text, one "
                f"fact per line, no commentary.\n\nText:\n{context}"
            )
        elif task_type == "summarization":
            prompt = f"Summarize this: {context}"
        elif task_type == "keypoints":
            prompt = (
                f"Read the following text and list the key points as "
                f"short bullet points, one per line.\n\nText:\n{context}"
            )
        elif task_type == "skill_extraction":
            # Step 1 of the resume/GD example from the meeting notes:
            # pull the required skills out of a GD/JD document.
            prompt = (
                f"Read the following group discussion (GD) / job "
                f"description text and list ONLY the specific skills, "
                f"tools, or competencies it requires. One skill per "
                f"line, no commentary, no invented skills not present "
                f"in the text.\n\nText:\n{context}"
            )
        elif task_type == "skill_matching":
            # Step 2: depends on skill_extraction's output, injected
            # into payload["required_skills"] by planner.py.
            required_skills = payload.get("required_skills", "")
            prompt = (
                f"You are matching a candidate's resume against a list "
                f"of required skills.\n\n"
                f"Required skills (from the GD/JD):\n{required_skills}\n\n"
                f"Candidate's resume text:\n{context}\n\n"
                f"IMPORTANT - stay strictly grounded: list ONLY skills "
                f"that appear explicitly or are clearly and directly "
                f"evidenced in the resume text above. Do NOT invent, "
                f"assume, or infer skills the resume does not actually "
                f"support, even if they seem like a natural fit. For "
                f"each required skill, state whether the resume "
                f"supports it (Yes/No) and quote the specific resume "
                f"phrase that supports it, if any."
            )
        else:
            return f"[{self.name}] Unsupported task type: {task_type}"
        return f"[{self.name}] {call_ollama(self.ollama_model, prompt)}"


class CoderSLM(BaseSLMAgent):
    name = "CoderSLM"
    capabilities = ["code"]
    model_size = "4B"
    ollama_model = "qwen3:4b"

    def run(self, task_type: str, payload: dict) -> str:
        instructions = payload.get("instructions", payload.get("context", ""))
        prompt = (
            f"Write a short, correct code snippet for the following "
            f"request. Return only the code, no explanation.\n\n"
            f"Request: {instructions}"
        )
        return f"[Coder] {call_ollama(self.ollama_model, prompt)}"


class CalculatorTool(BaseSLMAgent):
    name = "CalculatorTool"
    capabilities = ["math"]
    model_size = "n/a (deterministic tool)"

    def run(self, task_type: str, payload: dict) -> str:
        expr = payload.get("expression")
        cost = payload.get("cost")
        annual_savings = payload.get("annual_savings")
        if cost and annual_savings:
            years = round(cost / annual_savings, 1)
            return f"[Calculator] Payback period = {cost} / {annual_savings} = {years} years"
        return f"[Calculator] Result: {eval(expr) if expr else 'N/A'}"


class GeneralistFallbackSLM(BaseSLMAgent):
    name = "GeneralistSLM-8B"
    capabilities = ["extraction", "qa", "summarization", "keypoints", "code"]
    model_size = "8B"
    ollama_model = "qwen3:8b"

    def run(self, task_type: str, payload: dict) -> str:
        context = payload.get("context", "")
        query = payload.get("query", "")
        prompt = (
            f"You are a general-purpose assistant handling a '{task_type}' "
            f"task no specialist was available for.\n\nQuery: {query}\n"
            f"Context: {context}\n\nRespond appropriately for a '{task_type}' task."
        )
        return f"[Generalist fallback - {task_type}] {call_ollama(self.ollama_model, prompt)}"


# Instantiate the two TextProcessingSLM variants, named to match
# task_registry.py's agent priority lists.
TEXT_AGENT_QWEN = TextProcessingSLM("qwen3:4b", "TextProcessingSLM-qwen", "4B")
TEXT_AGENT_GEMMA = TextProcessingSLM("gemma2:2b", "TextProcessingSLM-gemma", "2B")
CODER_AGENT = CoderSLM()
CALCULATOR_AGENT = CalculatorTool()
FALLBACK_AGENT = GeneralistFallbackSLM()

# Registry lookup by name -> agent instance, used by the router
AGENTS_BY_NAME = {
    agent.name: agent
    for agent in [TEXT_AGENT_QWEN, TEXT_AGENT_GEMMA, CODER_AGENT, CALCULATOR_AGENT, FALLBACK_AGENT]
}
