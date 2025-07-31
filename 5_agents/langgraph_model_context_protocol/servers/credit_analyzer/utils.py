from servers.credit_analyzer.model import CreditAnalyzerInput, CreditAnalyzerOutput
from common.sagemaker_client import chat_llm
from agents.credit_analyst import generate_prompt  # ✅ use agent

def evaluate_credit(input_data: CreditAnalyzerInput) -> CreditAnalyzerOutput:
    # ✅ generate prompt using agent
    prompt = generate_prompt(input_data.summary, input_data.fields)

    response = chat_llm.invoke(prompt)
    output_text = response.content.strip()

    # basic score parsing
    if "low" in output_text.lower():
        score = "Low"
    elif "high" in output_text.lower():
        score = "High"
    else:
        score = "Medium"

    return CreditAnalyzerOutput(
        credit_assessment=output_text,
        score=score
    )
