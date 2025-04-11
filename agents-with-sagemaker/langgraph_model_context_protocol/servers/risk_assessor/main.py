from fastapi import FastAPI
from servers.risk_assessor.model import RiskAssessorInput, RiskAssessorOutput
from servers.risk_assessor.utils import assess_risk

app = FastAPI()

@app.post("/process", response_model=RiskAssessorOutput)
async def process_risk(input_data: RiskAssessorInput):
    return assess_risk(input_data)
