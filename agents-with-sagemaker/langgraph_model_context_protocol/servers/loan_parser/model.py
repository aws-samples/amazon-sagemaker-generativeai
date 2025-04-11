from pydantic import BaseModel
from typing import Optional

class LoanParserInput(BaseModel):
    name: str
    age: int
    income: float
    loan_amount: float
    credit_score: int
    existing_liabilities: float
    purpose: Optional[str] = None

class LoanParserOutput(BaseModel):
    summary: str
    fields: dict
