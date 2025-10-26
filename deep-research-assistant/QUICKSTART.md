# Quick Start Guide - MCP Architecture

## Prerequisites

1. Python 3.11+
2. PostgreSQL with pgvector extension
3. OpenAI API key

## Installation

```bash
# 1. Clone and navigate to project
cd deep-research-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Initialize database
# Make sure PostgreSQL is running with pgvector extension
```

## Starting the System

### Quick Start (Recommended)

```bash
start_servers.bat
```

This starts both:

- **MCP Server** on port 8001
- **FastAPI App** on port 8000

### Manual Start

**Terminal 1:**

```bash
python mcp_server.py
```

**Terminal 2:**

```bash
python -m app.main
```

## Testing

### 1. Check Health

```bash
curl http://localhost:8000/api/v1/health
```

### 2. Submit a Research Query

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Impact of COVID-19 on Indian economy?",
    "stream": false
  }'
```

### 3. Test with MCP Inspector

```bash
npx @modelcontextprotocol/inspector http://localhost:8001
```

## Architecture Overview

```
┌─────────────┐         ┌─────────────┐
│   Agents    │ ──────> │ MCP Client  │
│ (Clients)   │         │    Pool     │
└─────────────┘         └──────┬──────┘
                               │
                               │ HTTP/SSE
                               │
                        ┌──────▼──────┐
                        │ MCP Server  │
                        │  (Tools)    │
                        └─────────────┘
```

## Key Endpoints

### FastAPI (Port 8000)

- `POST /api/v1/chat` - Submit query
- `GET /api/v1/stream/{task_id}` - Stream progress
- `GET /api/v1/health` - Health check
- `GET /docs` - API documentation

### MCP Server (Port 8001)

- `GET /sse` - SSE connection
- `POST /messages` - MCP messages

## Configuration

Edit `app/config.py` or `.env`:

```python
MCP_SERVER_URL=http://localhost:8001
MCP_CLIENT_POOL_SIZE=5
```

## Troubleshooting

**MCP Server won't start:**

- Check port 8001 is free
- Verify database connection

**Agents can't connect:**

- Ensure MCP server started first
- Check MCP_SERVER_URL in config

**Tool execution fails:**

- Check MCP server logs
- Verify agent permissions
- Test with MCP Inspector

## Next Steps

1. Read [MCP_ARCHITECTURE.md](MCP_ARCHITECTURE.md) for detailed architecture
2. Explore tools via MCP Inspector
3. Check API docs at http://localhost:8000/docs
4. Review agent implementations in `app/agents.py`

## Support

For issues or questions, check the logs:

- MCP Server: Terminal 1 output
- FastAPI App: Terminal 2 output
