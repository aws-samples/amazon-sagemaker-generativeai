from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Any, Dict
from servers.loan_parser.model import LoanParserInput, LoanParserOutput
from servers.loan_parser.utils import parse_application

app = FastAPI()

@app.post("/process", response_model=LoanParserOutput)
async def process_application(request: Request, input_data: LoanParserInput):
    parsed_output = parse_application(input_data)
    return parsed_output
