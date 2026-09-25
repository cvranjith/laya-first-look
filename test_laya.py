from laya import Router

router = Router()  # downloads checkpoints on first use

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
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel?",
    },
}

result = router.predict(state, questions)
print(result)
