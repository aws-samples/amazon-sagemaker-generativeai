from servers.loan_parser.model import LoanParserInput, LoanParserOutput
from common.sagemaker_client import chat_llm
from agents.loan_officer import generate_prompt  # ✅ agent prompt builder

def parse_application(input_data: LoanParserInput) -> LoanParserOutput:
    # ✅ Use the agent to build the prompt
    prompt = generate_prompt(input_data.dict())

    # ✅ Send prompt to SageMaker-hosted LLM
    response = chat_llm.invoke(prompt)

    return LoanParserOutput(
        summary=response.content,
        fields=input_data.dict()
    )
