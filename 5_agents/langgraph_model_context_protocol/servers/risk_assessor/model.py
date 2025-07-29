from pydantic import BaseModel
from typing import Dict

class RiskAssessorInput(BaseModel):
    credit_assessment: str
    score: str  # Low / Medium / High
    fields: Dict  # Original parsed application

class RiskAssessorOutput(BaseModel):
    decision: str  # "Approved" or "Denied"
    reasoning: str
