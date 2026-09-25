from laya import Router

router = Router()

state = (
    "This is the third time your app has logged me out mid-purchase this "
    "month. I've wasted an hour on this. Fix it or I'm switching to a "
    "competitor and telling everyone why."
)

questions = {
    "category": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, integrations",
            "other": "everything else",
        },
    },
    "action": {
        "type": "choice",
        "instructions": "What should happen next?",
        "criteria": {
            "escalate": "needs a senior or specialist to step in",
            "reply": "a normal reply resolves it",
            "close": "no action needed",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["no time pressure", "needs attention soon", "blocking issue or hard deadline"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the customer suggest they may leave for a competitor?",
    },
    "needs_human": {
        "type": "noul",
        "instructions": "Does this require a human agent rather than an automated response?",
    },
}

result = router.predict(state, questions, model="typed-decisions")
print(result)
