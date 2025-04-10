from servers.risk_assessor.model import RiskAssessorInput, RiskAssessorOutput
from common.sagemaker_client import chat_llm
from agents.risk_manager import generate_prompt  
import re

def assess_risk(input_data: RiskAssessorInput) -> RiskAssessorOutput:
    prompt = generate_prompt(
        credit_assessment=input_data.credit_assessment,
        score=input_data.score,
        fields=input_data.fields
    )

    response = chat_llm.invoke(prompt)
    full_text = response.content.strip()

    #decision = "Approved" if "approved" in full_text.lower() else "Denied"


    if "decision: approved" in full_text.lower():
        decision = "Approved"
    elif "decision: denied" in full_text.lower():
        decision = "Denied"
    else:
        decision = "Undetermined"


    return RiskAssessorOutput(
        decision=decision,
        reasoning=full_text
    )
