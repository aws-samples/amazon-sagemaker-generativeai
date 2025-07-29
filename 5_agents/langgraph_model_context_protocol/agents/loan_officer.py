def generate_prompt(data: dict) -> list:
    return [
        {
            "role": "system",
            "content": "You are a helpful Loan Officer who summarizes and extracts loan application details.",
        },
        {
            "role": "user",
            "content": f"""Summarize the following loan application:

Name: {data['name']}
Age: {data['age']}
Income: {data['income']}
Loan Amount: {data['loan_amount']}
Credit Score: {data['credit_score']}
Existing Liabilities: {data['existing_liabilities']}
Purpose: {data.get('purpose', 'N/A')}
""",
        },
    ]
