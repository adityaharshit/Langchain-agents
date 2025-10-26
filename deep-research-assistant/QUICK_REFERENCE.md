# Quick Reference - Streamable-HTTP MCP Architecture

## Start Servers

```bash
# Option 1: Use batch script
start_servers.bat

# Option 2: Manual start
# Terminal 1
python mcp_server.py

# Terminal 2
python -m app.main
```

## Test Connection

```bash
python test_mcp_connection.py
```

## Submit Query

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question?", "stream": false}'
```

## MCP Inspector

```bash
npx @modelcontextprotocol/inspector http://localhost:8001
```

## Architecture

```
Agents → MCP Client Pool → streamable-http → FastMCP Server → Tools
```

## Ports

- **8000**: FastAPI Application
- **8001**: MCP Server (FastMCP with streamable-http)

## Key Files

- `mcp_server.py` - FastMCP server with streamable-http
- `app/mcp_client.py` - Client using streamable_http_client
- `app/mcp.py` - Registry that routes to MCP client
- `app/agents.py` - Agents (no changes needed)
- `app/mcp_tools.py` - Tool definitions (no changes needed)

## Configuration

```python
# app/config.py
MCP_SERVER_URL = "http://localhost:8001"
MCP_CLIENT_POOL_SIZE = 5
```

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Test MCP connection
python test_mcp_connection.py

# Check health
curl http://localhost:8000/api/v1/health

# View API docs
# Open: http://localhost:8000/docs
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8001 in use | Kill process or change port in config |
| Can't connect to MCP | Ensure mcp_server.py is running |
| Tool execution fails | Check server logs, test with Inspector |
| Import errors | Run `pip install fastmcp mcp` |

## Transport Details

**Type**: streamable-http (FastMCP)  
**Protocol**: HTTP-based MCP  
**Client Import**: `from mcp.client.streamable_http import streamable_http`  
**Advantages**: Modern, simple, robust  
**Replaces**: Deprecated SSE transport

## Documentation

- [MCP_ARCHITECTURE.md](MCP_ARCHITECTURE.md) - Full architecture
- [QUICKSTART.md](QUICKSTART.md) - Getting started
- [STREAMABLE_HTTP_UPDATE.md](STREAMABLE_HTTP_UPDATE.md) - Transport details
- [CHANGES.md](CHANGES.md) - All changes made
