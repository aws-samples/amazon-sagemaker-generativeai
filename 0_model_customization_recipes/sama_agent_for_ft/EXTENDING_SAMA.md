# Extending SAMA

Build custom MCP servers, add reward functions, and expand the agentic fine-tuning system.

## System Architecture

```
sama-science-agents/
├── MCP_servers/                          # All MCP server packages
│   ├── sama-sft-model-helper-mcp-server/     # Model discovery (4 tools)
│   ├── sama-dataset-prep-mcp-server/         # Dataset prep (4 tools)
│   ├── sama-sft-recipe-generator-mcp-server/ # SFT recipes (2 tools)
│   ├── sama-grpo-recipe-generator-mcp-server/# GRPO recipes (2 tools)
│   ├── sama-sft-training-workflow-mcp-server/ # Script gen (2 tools)
│   └── sama-sagemaker-job-mcp-server/        # Job launch/monitor (4 tools)
├── sample.kiro/                          # Template .kiro config
│   ├── settings/mcp.json                     # MCP server config
│   ├── skills/                               # Workflow skills
│   └── steering/                             # Context steering files
├── install.sh                            # One-command setup
└── requirements.txt                      # All Python deps
```

## Creating a Custom MCP Server

### 1. Project Structure

```
MCP_servers/sama-my-custom-mcp-server/
├── sama_my_custom_mcp_server/
│   ├── __init__.py
│   └── server.py
├── pyproject.toml
└── README.md
```

### 2. pyproject.toml

```toml
[project]
name = "sama-my-custom-mcp-server"
version = "0.1.0"
description = "My custom SAMA MCP server"
requires-python = ">=3.10"
dependencies = ["fastmcp>=2.9.2", "pydantic>=2.10.6", "mcp[cli]>=1.6.0"]

[project.scripts]
"sama-my-custom-mcp-server" = "sama_my_custom_mcp_server.server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["sama_my_custom_mcp_server"]
```

### 3. server.py

```python
#!/usr/bin/env python3
import logging, os, sys, warnings
from typing import Any, Dict

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

from fastmcp import FastMCP

mcp = FastMCP("sama-my-custom-mcp-server")

@mcp.tool()
def my_tool(param: str) -> Dict[str, Any]:
    """Tool description. Include WORKFLOW position if part of a pipeline."""
    return {"status": "success", "result": f"Processed {param}"}

def main():
    # CRITICAL: show_banner=False prevents stdout pollution that kills MCP transport
    mcp.run(show_banner=False)

if __name__ == "__main__":
    main()
```

Key rules:
- `mcp.run(show_banner=False)` — FastMCP 3.x prints an ASCII banner to stdout that breaks MCP
- All logging goes to `sys.stderr` — stdout is reserved for JSON-RPC
- If importing `sagemaker` or other noisy libs, wrap in `sys.stdout = io.StringIO()` / `finally: sys.stdout = _saved`
- Return dicts with `status` field and `next_step` for workflow guidance

### 4. Install and Configure

```bash
pip install -e MCP_servers/sama-my-custom-mcp-server
```

Add to `.kiro/settings/mcp.json`:

```json
"my_custom": {
    "command": "/opt/conda/bin/python3",
    "args": ["-m", "sama_my_custom_mcp_server.server"],
    "env": { "FASTMCP_LOG_LEVEL": "ERROR" },
    "disabled": false,
    "autoApprove": ["my_tool"]
}
```

### 5. Test

```bash
# Should hang silently (waiting for JSON-RPC on stdin)
python3 -m sama_my_custom_mcp_server.server

# Quick import test
python3 -c "from sama_my_custom_mcp_server.server import my_tool; print(my_tool('test'))"
```

## Handling Noisy Libraries (SageMaker, etc.)

The SageMaker SDK prints config warnings to stdout. This kills the MCP stdio transport. Pattern:

```python
import io

def _import_sagemaker():
    """Lazy import to avoid stdout pollution at module load."""
    _o = sys.stdout
    sys.stdout = io.StringIO()
    try:
        import sagemaker
        return sagemaker
    finally:
        sys.stdout = _o

@mcp.tool()
def my_sagemaker_tool(param: str) -> dict:
    try:
        sagemaker = _import_sagemaker()
        _saved = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sess = sagemaker.Session()
            # ... all SageMaker SDK calls here ...
        finally:
            sys.stdout = _saved
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

## Adding Skills and Steering

### Skills (manual inclusion via `#` in chat)

Create `.kiro/skills/my-workflow.md`:

```markdown
---
inclusion: manual
---

# My Workflow

Prescribed steps for the agent to follow...
```

Invoke with `#my-workflow` in Kiro CLI chat.

### Steering (auto-included based on open files)

Create `.kiro/steering/my-context.md`:

```markdown
---
inclusion: fileMatch
fileMatchPattern: "my_directory/**"
---

# Context

Auto-loaded when files in my_directory/ are open...
```

## Custom Reward Functions (GRPO)

Add to `preference_optimization/grpo_rlvr/sagemaker_code/rewards/`:

```python
def my_reward(completions, **kwargs):
    tokenizer = kwargs.get('tokenizer')
    rewards = []
    for completion in completions:
        score = calculate_score(completion)
        rewards.append(score)
    return rewards
```

Reference it in the GRPO training launch:

```
tools_script: tools_funcs/my_tools.py
reward_fn: rewards/my_reward.py
```

## SageMaker vs Local Differences

| Aspect | SageMaker | Local |
|---|---|---|
| Python path | `/opt/conda/bin/python3` | Your venv path |
| AWS credentials | Auto (IAM role) | `aws configure` or env vars |
| SageMaker role | Auto-detected | Must set `role_arn` |
| Install | `bash install.sh` | `PYTHON=/path bash install.sh` |
| Job launch | Works directly | Needs AWS credentials + role |
| Dataset upload | S3 access built-in | Needs S3 permissions |

## Testing Checklist

1. Server starts cleanly: `python3 -m <module>.server` hangs silently
2. Import works: `python3 -c "import <module>; print('OK')"`
3. Tool returns dict: `python3 -c "from <module>.server import my_tool; print(my_tool('test'))"`
4. MCP handshake works: `echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | python3 -m <module>.server 2>/dev/null`
5. Kiro CLI loads: `kiro-cli chat` then `/mcp` shows green checkmark
