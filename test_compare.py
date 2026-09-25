"""Run the same inputs through Laya (local) and Jev (API) and print both side by side."""
import json

from laya import Router
from jev_client import predict as jev_predict

router = Router()

cases = [
    {
        "name": "Basic decision (billing/churn)",
        "state": "We were billed twice for March. Please refund or we will cancel.",
        "questions": {
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
        },
    },
    {
        "name": "Support ticket triage",
        "state": (
            "My laptop screen is flickering since the last update, and I'm on a "
            "business trip until Friday — need this fixed urgently."
        ),
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department handles this?",
                "criteria": {
                    "billing": "invoices, payments, refunds",
                    "technical": "bugs, outages, errors, hardware issues",
                    "other": "everything else",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": [
                    "not urgent at all, can wait indefinitely",
                    "low urgency, can wait a week or more",
                    "moderate urgency, should be handled within a few days",
                    "high urgency, needs attention within a day",
                    "extremely urgent, needs immediate attention",
                ],
            },
            "is_hardware_issue": {
                "type": "noul",
                "instructions": "Is this a hardware problem (as opposed to software)?",
            },
        },
    },
    {
        "name": "Multilingual (Hindi) refund request",
        "state": (
            "मेरा ऑर्डर दो हफ्ते पहले आना था और अभी तक नहीं आया। "
            "अगर जल्दी नहीं आया तो मैं पैसे वापस चाहता हूँ।"
        ),
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department handles this?",
                "criteria": {
                    "billing": "invoices, payments, refunds",
                    "shipping": "delivery, tracking, delays",
                    "other": "everything else",
                },
            },
            "wants_refund": {"type": "noul", "instructions": "Does the user want a refund?"},
        },
    },
]

for case in cases:
    print("=" * 80)
    print(case["name"])
    print("=" * 80)

    laya_result, laya_time = None, None
    import time as _time
    t0 = _time.perf_counter()
    laya_result = router.predict(case["state"], case["questions"])
    laya_time = _time.perf_counter() - t0

    try:
        jev_result, jev_time, jev_server_ms = jev_predict(case["state"], case["questions"])
    except Exception as e:
        jev_result, jev_time, jev_server_ms = {"error": str(e)}, None, None

    print(f"\n-- Laya (local, {laya_time*1000:.1f} ms) --")
    print(json.dumps(laya_result["answers"], indent=2, ensure_ascii=False))

    jt = f"{jev_time*1000:.1f} ms" if jev_time else "n/a"
    js = f"{jev_server_ms:.1f} ms" if jev_server_ms else "n/a"
    print(f"\n-- Jev (API, {jt} round trip, {js} server-side) --")
    print(json.dumps(jev_result.get("answers", jev_result), indent=2, ensure_ascii=False))
    print()
