from laya import Router

router = Router()

state = (
    "मेरा ऑर्डर दो हफ्ते पहले आना था और अभी तक नहीं आया। "
    "अगर जल्दी नहीं आया तो मैं पैसे वापस चाहता हूँ।"
)  # "My order was supposed to arrive two weeks ago and still hasn't. If it
   #  doesn't arrive soon, I want a refund."

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department handles this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "shipping": "delivery, tracking, delays",
            "other": "everything else",
        },
    },
    "wants_refund": {
        "type": "noul",
        "instructions": "Does the user want a refund?",
    },
}

result = router.predict(state, questions)
print(result)
