# 🏦 Loan Underwriter: Agentic LLM Pipeline with MCP + LangGraph + SageMaker

This project implements an LLM-powered loan underwriting pipeline using:

- 🤖 **LangGraph** for multi-step orchestration
- 🛠 **MCP servers** (Model Context Protocol) as LLM-powered tools
- 🧠 **Agent prompt modules** for clean, reusable prompt engineering
- 📡 **Amazon SageMaker** for secure, scalable LLM hosting
- 🔍 **LangSmith** for deep observability & tracing

---

## 💡 Overview

This is a modular, agentic system that processes a loan application through 3 roles:

1. **Loan Officer** ➝ Summarizes the application
2. **Credit Analyst** ➝ Evaluates creditworthiness
3. **Risk Manager** ➝ Makes a final approval/denial decision

Each role is:
- Represented by an **agent** (in `agents/`)
- Deployed via a dedicated **MCP server** (in `servers/`)
- Orchestrated with **LangGraph** (in `langgraph_flow/`)

---

## 🧱 Project Structure

```
Agents-FSI-MCP/
├── agents/
│   ├── loan_officer.py
│   ├── credit_analyst.py
│   └── risk_manager.py
│
├── servers/
│   ├── loan_parser/
│   ├── credit_analyzer/
│   └── risk_assessor/
│
├── common/
│   └── sagemaker_client.py
│
├── langgraph_flow/
│   └── graph.py
│
├── scripts/
│   └── run_pipeline.py
│
├── .env
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install -U langchain-aws
```
Note: In case you get module not found error with Langchain
Run pip install -U langchain-aws

### 2. Deploy LLM Endpoint
python3 deploy_sm_endpoint.py

Copy the output of the endpoint name craeted after this job is successful for the next step to paste it in your.env file.

### 3. Set up `.env`file in the same root folder. Copy the code below.
Note: Langchain API key to access LangSmith is optional.

```env
AWS_REGION=YOUR REGION
SAGEMAKER_ENDPOINT=your-endpoint-name (created in Step2)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
LANGCHAIN_API_KEY=your-langsmith-key(optional)
LANGCHAIN_PROJECT=LoanUnderwriterFlow(optional)
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com(optional)
LANGCHAIN_TRACING_V2=true(optional)

LOAN_PARSER_URL=http://127.0.0.1:8002/process
CREDIT_ANALYZER_URL=http://127.0.0.1:8003/process
RISK_ASSESSOR_URL=http://127.0.0.1:8004/process
```

### 3. Start MCP servers
In case you are running this code in Amazon SageMaker JupyterLab. You would need to run the commands individually in three different terminals to start the servers.
Go to JupyterLab -> Click New Terminal -> Copy and paste uvicorn servers.loan_parser.main:app --port 8002
Go to JupyterLab -> Click New Terminal -> Copy and paste uvicorn servers.credit_analyzer.main:app --port 8003
Go to JupyterLab -> Click New Terminal -> Copy and paste uvicorn servers.risk_assessor.main:app --port 8004

Or you can use tmux( terminal multiplexer) to run all three servers if you have tmux installed
```bash
uvicorn servers.loan_parser.main:app --port 8002
uvicorn servers.credit_analyzer.main:app --port 8003
uvicorn servers.risk_assessor.main:app --port 8004
```

### 4. Run the pipeline
From JupyterLab-> Click new terminal-> copy and paste python scripts/run_pipeline.py
Note: make sure all the terminals are running.
```bash
python scripts/run_pipeline.py
```

Check results in:
- 🧠 Terminal (final output)
- 🔍 [LangSmith](https://smith.langchain.com) for full trace (Optional)

---

## ✨ Key Concepts

### ✅ MCP Servers
Each server exposes a single `/process` endpoint that:
- Receives structured input
- Uses an agent to generate a role-specific prompt
- Sends the prompt to the SageMaker-hosted LLM
- Returns structured output

### ✅ Agents
Each agent is a prompt builder that encapsulates role-specific logic. They're imported inside the MCP server `utils.py` files.

### ✅ LangGraph
Handles orchestration across:
- `LoanParser` ➝ `CreditAnalyzer` ➝ `RiskAssessor`
- Passes structured state between nodes
- Tagged with `agent:*` + `mcp:*` in LangSmith

### ✅ LangSmith
Observability tool that traces each LLM call. Every prompt, response, and tag is recorded for debugging and analysis.

---

## 📦 Possible Next Steps
- Docker Compose setup for all MCP servers
- Streamlit UI to submit applications
- Fine-tuned LLMs for better decision accuracy

---
