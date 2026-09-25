from laya import Router

router = Router()

state = (
    "My laptop screen is flickering since the last update, and I'm on a "
    "business trip until Friday — need this fixed urgently."
)
questions = {
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
}

result = router.predict(state, questions)
print(result)
