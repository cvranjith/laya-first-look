import time
from laya import Router

t0 = time.perf_counter()
router = Router()
t1 = time.perf_counter()
print(f"Router() construction: {t1 - t0:.3f}s")

state = "We were billed twice for March. Please refund or we will cancel."
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department handles this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, errors",
            "other": "everything else",
        },
    },
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
}

for i in range(8):
    t0 = time.perf_counter()
    result = router.predict(state, questions)
    dt = time.perf_counter() - t0
    print(f"call {i}: {dt*1000:.1f} ms")
