from fastapi import FastAPI
from servers.credit_analyzer.model import CreditAnalyzerInput, CreditAnalyzerOutput
from servers.credit_analyzer.utils import evaluate_credit

app = FastAPI()

@app.post("/process", response_model=CreditAnalyzerOutput)
async def process_credit(input_data: CreditAnalyzerInput):
    result = evaluate_credit(input_data)

    updated_result = result.model_copy(update={"fields": input_data.fields})

    return updated_result
