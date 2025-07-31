def generate_prompt(credit_assessment: str, score: str, fields: dict) -> list:
    return [
        {
            "role": "system",
            "content": "You are a risk manager. Output the decision explicitly as 'Decision: Approved' or 'Decision: Denied'.",
        },
        {
            "role": "user",
            "content": f"""Credit Assessment:\n{credit_assessment}

Rating: {score}

Applicant Info:\n{fields}"""
        },
    ]
