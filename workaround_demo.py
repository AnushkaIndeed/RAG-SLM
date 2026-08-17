"""
Explicit demo of RULE 1 (capability workaround) actually triggering.
We manually inject a task type ('sentiment') that NO specialist agent
in the pool declares as a capability, to force the router to escalate
to the generalist fallback -- and show what happens if even the
fallback can't cover it either.
"""

from planner import route_task

log = []

print("=" * 70)
print("CAPABILITY-MISMATCH WORKAROUND — DEMO")
print("=" * 70)

# Case A: no SPECIALIST covers 'qa' -> should escalate to the generalist fallback
task_a = {"type": "qa", "payload": {"context": "Solar panels have a 25-30 year lifespan."}}
result_a = route_task(task_a, log)
print(f"\nTask: 'qa' (no specialist covers this -> should fall back to generalist)")
print(f"Result: {result_a}")

# Case B: a task type that was never classified as atomic/non-atomic at
# all -> the router's safety check should REFUSE to route it, rather
# than guessing or silently mishandling it. This is intended behavior,
# not a crash: an unrouted, unclassified task type should stop the
# pipeline and surface loudly instead of failing silently downstream.
task_b = {"type": "sentiment", "payload": {"context": "Reviews are very positive."}}
try:
    result_b = route_task(task_b, log)
    print(f"\nTask: 'sentiment'\nResult: {result_b}")
except AssertionError as e:
    print(f"\nTask: 'sentiment' (never classified by the planner's task-type rules)")
    print(f"Router correctly REFUSED to guess: {e}")
    log.append("Refused 'sentiment': unclassified task type, stopped rather than silently mishandling it")

print("\nFull routing log:")
for entry in log:
    print(f"  - {entry}")
