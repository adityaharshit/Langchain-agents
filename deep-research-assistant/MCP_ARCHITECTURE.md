# MCP Architecture for Deep Research Assistant

## Overview

The Deep Research Assistant now uses a **Model Context Protocol (MCP)** architecture where:
- **MCP Server** exposes all tools via SSE (Server-Sent Events) transport
- **Agents** act as MCP clients and access tools through the server
- All tool execution goes through the MCP protocol

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│                      (Port 8000)                             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Research   │  │  WebScraper  │  │ DeepAnalysis │     │
│  │ Coordinator  │  │    Agent     │  │    Agent     │ ... │
│  │    Agent     │  │              │  │              │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                  │              │
│         └─────────────────┼──────────────────┘              │
│                           │                                 │
│                    ┌──────▼──────┐                          │
│                    │ MCP Client  │                          │
│                    │    Pool     │                          │
│                    └──────┬──────┘                          │
└───────────────────────────┼──────────────────────────────────┘
                            │
                            │ HTTP/SSE
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                      MCP Server                              │
│                      (Port 8001)                             │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Tool Registry                          │    │
│  │  • query_decomposer                                 │    │
│  │  • semantic_search                                  │    │
│  │  • web_scraper                                      │    │
│  │  • comprehensive_analysis                           │    │
│  │  • fact_checking                                    │    │
│  │  • ... (20+ tools)                                  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. MCP Server (`mcp_server.py`)
- Runs on port **8001**
- Uses **streamable-http** transport (FastMCP)
- Exposes all registered tools via MCP protocol
- Handles tool execution and returns results

**Transport:**
- Uses FastMCP with streamable-http for modern HTTP-based communication
- Automatically handles request/response streaming

### 2. MCP Client (`app/mcp_client.py`)
- **MCPClient**: Single client connection to MCP server
- **MCPClientPool**: Pool of clients for concurrent access
- Handles connection management and tool execution

### 3. Agents (`app/agents.py`)
- Act as MCP clients
- Access tools through `mcp_registry.execute_tool()`
- Registry internally uses MCP client to communicate with server

## Starting the System

### Option 1: Using the Batch Script (Recommended)
```bash
cd deep-research-assistant
start_servers.bat
```

This will:
1. Start MCP Server on port 8001
2. Wait 3 seconds
3. Start FastAPI Application on port 8000

### Option 2: Manual Start

**Terminal 1 - Start MCP Server:**
```bash
cd deep-research-assistant
python mcp_server.py
# Server will start on http://0.0.0.0:8001 with streamable-http transport
```

**Terminal 2 - Start FastAPI Application:**
```bash
cd deep-research-assistant
python -m app.main
# Application will start on http://0.0.0.0:8000
```

## Configuration

Edit `app/config.py` or set environment variables:

```python
# MCP Server Configuration
MCP_SERVER_URL: str = "http://localhost:8001"
MCP_CLIENT_POOL_SIZE: int = 5
```

Or in `.env`:
```bash
MCP_SERVER_URL=http://localhost:8001
MCP_CLIENT_POOL_SIZE=5
```

## Testing with MCP Inspector

To inspect and test tools using the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector http://localhost:8001
```

This opens a web interface where you can:
- Browse all available tools
- Test tool execution with custom inputs
- See tool schemas and allowed agents
- Debug tool responses

## Tool Execution Flow

1. **Agent** calls `mcp_registry.execute_tool(tool_name, agent_name, tool_input, context)`
2. **Registry** validates agent permissions
3. **Registry** calls `execute_tool_via_mcp()` from `mcp_client.py`
4. **MCP Client Pool** gets an available client
5. **MCP Client** sends request to MCP Server via SSE
6. **MCP Server** executes the tool function
7. **MCP Server** returns result via SSE
8. **MCP Client** parses and returns result
9. **Agent** receives the result

## Benefits of MCP Architecture

1. **Separation of Concerns**: Tools are isolated in the MCP server
2. **Scalability**: Multiple agents can access tools concurrently via client pool
3. **Monitoring**: All tool calls go through MCP protocol (can be logged/monitored)
4. **Testing**: Tools can be tested independently via MCP Inspector
5. **Security**: Tool access is controlled and validated at the protocol level
6. **Flexibility**: Easy to add new tools or modify existing ones

## Troubleshooting

### MCP Server Not Starting
- Check if port 8001 is available
- Verify database connection
- Check logs for errors

### Agents Can't Connect to MCP Server
- Ensure MCP server is running first
- Check `MCP_SERVER_URL` configuration
- Verify network connectivity

### Tool Execution Fails
- Check MCP server logs
- Verify agent has permission for the tool
- Test tool directly via MCP Inspector

## Development

### Adding a New Tool

1. Add tool function in `app/mcp_tools.py`:
```python
@mcp_tool(
    name="my_new_tool",
    description="Description of the tool",
    allowed_agents=["MyAgent"]
)
async def my_new_tool(tool_input: dict, context: dict) -> dict:
    # Tool implementation
    return {"status": "ok", "result": {...}}
```

2. Tool is automatically registered and available via MCP server

3. Agent can use it:
```python
result = await mcp_registry.execute_tool(
    "my_new_tool",
    "MyAgent",
    {"param": "value"},
    {"task_id": task_id}
)
```

### Testing a Tool

Use MCP Inspector:
```bash
npx @modelcontextprotocol/inspector http://localhost:8001
```

Or programmatically:
```python
from app.mcp_client import MCPClient

client = MCPClient()
await client.connect()
result = await client.execute_tool(
    "my_new_tool",
    "TestAgent",
    {"param": "value"}
)
```

## API Endpoints

### FastAPI Application (Port 8000)
- `POST /api/v1/chat` - Submit research query
- `GET /api/v1/stream/{task_id}` - Stream progress events
- `GET /api/v1/status/{task_id}` - Get task status
- `GET /api/v1/health` - Health check
- `GET /api/v1/stats` - System statistics
- `GET /docs` - API documentation

### MCP Server (Port 8001)
- Uses streamable-http transport (FastMCP)
- Automatically handles HTTP-based MCP protocol communication

## Performance Considerations

- **Client Pool Size**: Adjust `MCP_CLIENT_POOL_SIZE` based on concurrent agent needs
- **Connection Reuse**: Clients are pooled and reused for efficiency
- **Async Operations**: All tool calls are async for non-blocking execution
- **Error Handling**: Automatic retry and error recovery in client pool
