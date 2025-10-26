# Architecture Changes - MCP Client-Server Model

## Summary

The Deep Research Assistant has been refactored from a **direct tool access** model to an **MCP client-server architecture** where agents act as clients and access tools through an MCP server.

## What Changed

### Before (Direct Access)
```python
# Agents called tools directly
result = await tool_function(tool_input, context)
```

### After (MCP Client-Server)
```python
# Agents call tools via MCP protocol
result = await mcp_registry.execute_tool(
    tool_name, agent_name, tool_input, context
)
# ↓ (internally routes through MCP client)
# ↓ (sends request to MCP server via SSE)
# ↓ (server executes tool and returns result)
```

## New Files

1. **`mcp_server.py`** - MCP server using SSE transport (port 8001)
2. **`app/mcp_client.py`** - MCP client and client pool for agents
3. **`start_servers.bat`** - Convenience script to start both servers
4. **`MCP_ARCHITECTURE.md`** - Detailed architecture documentation
5. **`QUICKSTART.md`** - Quick start guide
6. **`CHANGES.md`** - This file

## Modified Files

1. **`app/mcp.py`**
   - `execute_tool()` now routes through MCP client instead of direct execution
   - Added import for `execute_tool_via_mcp`

2. **`app/main.py`**
   - Added MCP client pool initialization on startup
   - Added MCP client pool shutdown on shutdown

3. **`app/config.py`**
   - Added `MCP_SERVER_URL` configuration
   - Added `MCP_CLIENT_POOL_SIZE` configuration

4. **`requirements.txt`**
   - Added `mcp` package
   - Added `starlette` package

5. **`app/agents.py`**
   - No changes needed! Agents still use `mcp_registry.execute_tool()`
   - The routing to MCP server happens transparently

## Architecture Benefits

### 1. Separation of Concerns
- Tools are isolated in MCP server
- Agents don't need to know about tool implementation
- Clear boundary between client and server

### 2. Scalability
- Multiple agents can access tools concurrently
- Client pool manages connections efficiently
- Easy to scale MCP server independently

### 3. Monitoring & Debugging
- All tool calls go through MCP protocol
- Can log/monitor all tool executions
- MCP Inspector for testing and debugging

### 4. Security
- Tool access controlled at protocol level
- Agent permissions validated by server
- Centralized access control

### 5. Flexibility
- Easy to add/modify tools without changing agents
- Can swap MCP server implementation
- Tools can be tested independently

## How to Use

### Start Both Servers
```bash
start_servers.bat
```

Or manually:
```bash
# Terminal 1
python mcp_server.py

# Terminal 2
python -m app.main
```

### Test with MCP Inspector
```bash
npx @modelcontextprotocol/inspector http://localhost:8001
```

### Submit Research Query
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Your research question?", "stream": false}'
```

## Migration Notes

### For Developers

**No changes needed in agent code!** The agents still call:
```python
result = await mcp_registry.execute_tool(
    tool_name, agent_name, tool_input, context
)
```

The routing to MCP server happens transparently in the registry.

### For Deployment

**Two processes now required:**
1. MCP Server (port 8001)
2. FastAPI Application (port 8000)

Both must be running for the system to work.

### Configuration

Add to `.env` or `app/config.py`:
```bash
MCP_SERVER_URL=http://localhost:8001
MCP_CLIENT_POOL_SIZE=5
```

## Testing

### 1. Test MCP Server
```bash
python mcp_server.py
# Should show: "MCP Server ready on http://localhost:8001"
```

### 2. Test MCP Client Connection
```python
from app.mcp_client import MCPClient

client = MCPClient()
await client.connect()
print(client.get_available_tools())
```

### 3. Test End-to-End
```bash
# Start both servers
start_servers.bat

# Submit query
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query", "stream": false}'
```

## Troubleshooting

### "Connection refused" error
- Ensure MCP server is running on port 8001
- Check `MCP_SERVER_URL` in config

### "Tool not found" error
- Verify tool is registered in `app/mcp_tools.py`
- Check MCP server logs for tool registration

### "Agent not allowed" error
- Check tool's `allowed_agents` list
- Verify agent name matches exactly

## Performance

- **Client Pool**: 5 concurrent connections by default
- **Connection Reuse**: Clients are pooled and reused
- **Async Operations**: All tool calls are non-blocking
- **Overhead**: Minimal (~10-20ms per tool call for MCP protocol)

## Future Enhancements

1. **Load Balancing**: Multiple MCP server instances
2. **Caching**: Cache tool results at MCP level
3. **Rate Limiting**: Per-agent rate limits
4. **Metrics**: Prometheus metrics for tool usage
5. **Authentication**: Token-based auth for MCP clients

## Rollback

To rollback to direct tool access:

1. In `app/mcp.py`, change `execute_tool()` to call tool function directly:
```python
# Old direct execution
if asyncio.iscoroutinefunction(tool.function):
    result = await tool.function(tool_input, context or {})
else:
    result = tool.function(tool_input, context or {})
```

2. Remove MCP client initialization from `app/main.py`

3. Start only FastAPI application (no MCP server needed)

## Questions?

See:
- [MCP_ARCHITECTURE.md](MCP_ARCHITECTURE.md) - Detailed architecture
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- MCP Server logs - For debugging tool execution
- FastAPI logs - For debugging agent behavior
