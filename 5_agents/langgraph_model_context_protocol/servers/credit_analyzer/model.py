from pydantic import BaseModel
from typing import Dict, Optional

class CreditAnalyzerInput(BaseModel):
    summary: str  # summary from loan_parser
    fields: Dict  # raw structured fields from loan_parser

class CreditAnalyzerOutput(BaseModel):
    credit_assessment: str
    score: str  # optional, e.g., "Low", "Medium", "High"
    fields: Optional[Dict[str, str]] = None  # Add this line
