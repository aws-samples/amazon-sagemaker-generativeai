# SAMA Deployment MCP Server - Quick Start Guide

## TL;DR - Choose Your Mode

### 🚀 Q CLI Mode (Easiest)
```bash
# Just start Q CLI - servers launch automatically
q chat
# Then use natural language: "Deploy meta-textgeneration-llama-3-2-3b model"
```

### 🌐 Standalone HTTP Mode (For Multiple Agents)
```bash
# Install dependencies
pip install uvicorn fastapi

# Launch HTTP server
python http_wrapper.py --host 0.0.0.0 --port 8001

# Use REST API from any agent
curl -X POST "http://localhost:8001/deploy" \
  -H "Content-Type: application/json" \
  -d '{"model_id": "meta-textgeneration-llama-3-2-3b"}'
```

## When to Use Each Mode

| Mode | Best For | Pros | Cons |
|------|----------|------|------|
| **Q CLI** | Single user, interactive work | Easy setup, natural language | One user at a time |
| **HTTP Standalone** | Multiple agents, automation | Multiple clients, REST API | Requires setup |
| **Docker** | Production, isolation | Containerized, scalable | More complex |

## Configuration Files

### Q CLI Configuration
**Location**: `~/.aws/amazonq/mcp.json` or `.amazonq/mcp.json`

```json
{
  "mcpServers": {
    "sama-deployment-mcp-server": {
      "command": "sama-deployment-mcp-server",
      "args": ["--allow-write"],
      "disabled": false,
      "autoApprove": ["*"]
    }
  }
}
```

### Standalone Configuration
No config file needed - just command line arguments:

```bash
python http_wrapper.py --host 0.0.0.0 --port 8001
```

## Quick Examples

### Q CLI Natural Language
```
"Deploy meta-textgeneration-llama-3-2-3b on ml.g5.xlarge"
"List my endpoints"
"Delete endpoint jumpstart-llama-xyz"
"Get deployment recommendations for high-throughput"
```

### HTTP API Calls
```bash
# Deploy model
curl -X POST "http://localhost:8001/deploy" \
  -d '{"model_id": "meta-textgeneration-llama-3-2-3b"}'

# List endpoints
curl "http://localhost:8001/endpoints"

# Get recommendations
curl -X POST "http://localhost:8001/recommendations" \
  -d '{"model_id": "meta-textgeneration-llama-3-2-3b"}'
```

## Troubleshooting

### Check if servers are running
```bash
ps aux | grep sama-deployment-mcp-server
```

### Test AWS credentials
```bash
aws sts get-caller-identity
```

### View logs
```bash
# Q CLI mode - logs appear in Q CLI
# HTTP mode - logs in terminal where you started the server
```

## Next Steps

1. **Start with Q CLI mode** for interactive testing
2. **Move to HTTP mode** when you need multiple agents
3. **Use Docker** for production deployments
4. **Check the full README.md** for detailed configuration options

## Common Use Cases

### Development & Testing
- Use Q CLI mode
- Natural language commands
- Interactive debugging

### Multi-Agent Systems
- Use HTTP wrapper mode
- REST API integration
- Multiple concurrent clients

### Production Deployment
- Use Docker container
- Load balancer in front
- Monitoring and logging
