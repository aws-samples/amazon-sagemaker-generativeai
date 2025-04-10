def generate_prompt(summary: str, fields: dict) -> list:
    return [
        {
            "role": "system",
            "content": "You are a credit analyst. Rate the applicant’s creditworthiness as Low, Medium, or High.",
        },
        {
            "role": "user",
            "content": f"""Loan Summary:\n{summary}\n\nDetails:\n{fields}""",
        },
    ]
