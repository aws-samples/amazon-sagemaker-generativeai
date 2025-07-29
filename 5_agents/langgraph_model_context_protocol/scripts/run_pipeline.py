from dotenv import load_dotenv
load_dotenv()

import asyncio
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph_flow.graph import build_graph

graph = build_graph()


loan_input = {
    "output": {
        "name": "Jane Doe",
        "age": 35,
        "income": 450000,
        "loan_amount": 450000,
        "credit_score": 720,
        "existing_liabilities": 15000,
        "purpose": "Home Renovation"
    }
}


from langsmith import traceable

@traceable(name="LoanUnderwriter::FullRun")
async def run():
    result = await graph.ainvoke(loan_input)
    print("✅ Final result:", result["output"])

if __name__ == "__main__":
    asyncio.run(run())
