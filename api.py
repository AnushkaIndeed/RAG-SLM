"""
FastAPI backend, per meeting notes: "Convert into FastAPI backend and
make a frontend... Query endpoint and then fn calling."

Two endpoints:
  POST /query      - runs the full pipeline (retrieval + planner +
                      agents) on a user query, returns the answer plus
                      the full debug trace.
  GET  /functions   - lists every registered agent as a JSON-schema
                      "function" definition, the shape a real
                      function-calling model expects. This is the
                      "fn calling" piece from the notes: each agent's
                      capability is exposed as a callable tool schema,
                      matching how OpenAI/Ollama-style tool-calling
                      works, even though the routing logic itself
                      already does this matching internally via
                      task_registry.py.

Run with:
    uvicorn api:app --reload
Then POST to http://127.0.0.1:8000/query with {"query": "..."}
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from full_pipeline import run_full_pipeline_v2   # the REAL pipeline: retrieval +
                                                   # compound-query splitting +
                                                   # confidence guard + planner + agents
from task_registry import TASK_REGISTRY
from slm_agents import AGENTS_BY_NAME

app = FastAPI(title="SLM Multi-Agent Pipeline API")

# Allow the local frontend (served from a different port/file://) to
# call this API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    context: str = ""   # OPTIONAL OVERRIDE ONLY: if you pass this,
                         # it's used INSTEAD of real retrieval -- for
                         # manual testing without touching the vector
                         # store. Leave blank for real retrieval.


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    task_log: list
    results: dict
    retrieved_chunks: list = []


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """Runs the FULL production pipeline: compound-query splitting,
    real FAISS retrieval per sub-question, the confidence guard, then
    planner + agent execution per reliable sub-question. This is the
    same pipeline full_pipeline.py runs from the terminal -- the
    frontend now gets identical behavior, not a stripped-down version.

    If request.context is non-empty, it's used as a manual override
    instead of retrieval (useful for testing agents in isolation
    without needing the vector index built).
    """
    if request.context.strip():
        # Manual override path: skip retrieval, go straight to planning
        from planner import run_planned_pipeline
        result = run_planned_pipeline(request.query, request.context)
        result["retrieved_chunks"] = []
    else:
        # Normal path: full pipeline with real retrieval
        result = run_full_pipeline_v2(request.query)

    return QueryResponse(
        query=result["query"],
        final_answer=result["final_answer"],
        task_log=result["task_log"],
        results=result["results"],
        retrieved_chunks=result.get("retrieved_chunks", []),
    )


@app.get("/functions")
def list_functions():
    """Function-calling-style schema listing: every task type and
    which agent(s) would handle it, in the JSON-schema shape a
    function-calling model expects. Useful for debugging routing
    decisions and for eventually wiring this into a model that does
    its OWN function selection (rather than the current planner-model
    + registry approach)."""
    functions = []
    for task_type, registration in TASK_REGISTRY.items():
        agent_names = registration.agents_in_priority_order
        functions.append({
            "name": task_type,
            "description": registration.notes or f"Handles '{task_type}' tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "Text context for the task"},
                    "query": {"type": "string", "description": "The original user query"},
                },
                "required": ["context"],
            },
            "is_atomic": registration.is_atomic,
            "handled_by_primary": agent_names[0] if agent_names else None,
            "handled_by_fallback_chain": agent_names,
        })
    return {"functions": functions}


@app.get("/health")
def health_check():
    return {"status": "ok", "agents_loaded": list(AGENTS_BY_NAME.keys())}