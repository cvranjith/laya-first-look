import time
from laya import Router

router = Router()
state = "We were billed twice for March. Please refund or we will cancel."

# warm up first (cold-start cost paid once, ignored)
router.predict(state, {"warm": {"type": "noul", "instructions": "placeholder"}})

def make_questions(n):
    qs = {}
    for i in range(n):
        qs[f"q{i}"] = {"type": "noul", "instructions": f"Is this question number {i} relevant?"}
    return qs

for n in (1, 2, 4, 8, 16, 32):
    qs = make_questions(n)
    # average over a few runs for stability
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        router.predict(state, qs)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times) * 1000
    print(f"{n:3d} questions: avg {avg:6.1f} ms")
