"""
MCP Server for Deep Research Assistant.
Exposes tools via Model Context Protocol for use with MCP Inspector and clients.
"""
import asyncio
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from app.config import config
from app.db.database import init_database, close_database
from app.mcp import mcp_registry

# Import all tools to register them
import app.mcp_tools  # noqa: F401

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Log to stderr to not interfere with MCP stdio
)
logger = logging.getLogger(__name__)

# Create MCP server
mcp_server = Server("deep-research-assistant")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available MCP tools."""
    tools = []
    
    for mcp_tool in mcp_registry.get_all_tools():
        # Convert to MCP Tool format
        tool = Tool(
            name=mcp_tool.name,
            description=mcp_tool.description,
            inputSchema={
                "type": "object",
                "properties": {
                    "tool_input": {
                        "type": "object",
                        "description": "Tool-specific input parameters"
                    },
                    "agent_name": {
                        "type": "string",
                        "description": f"Agent name (allowed: {', '.join(mcp_tool.allowed_agents)})"
                    }
                },
                "required": ["tool_input", "agent_name"]
            }
        )
        tools.append(tool)
    
    logger.info(f"Listed {len(tools)} tools")
    return tools


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute an MCP tool."""
    try:
        logger.info(f"Calling tool: {name}")
        
        # Extract parameters
        tool_input = arguments.get("tool_input", {})
        agent_name = arguments.get("agent_name", "MCPClient")
        context = arguments.get("context", {})
        
        # Execute tool through registry
        result = await mcp_registry.execute_tool(
            tool_name=name,
            agent_name=agent_name,
            tool_input=tool_input,
            context=context
        )
        
        # Format result as TextContent
        import json
        result_text = json.dumps(result, indent=2, default=str)
        
        return [TextContent(
            type="text",
            text=result_text
        )]
        
    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        import json
        error_result = {
            "status": "error",
            "error": str(e),
            "tool": name
        }
        return [TextContent(
            type="text",
            text=json.dumps(error_result, indent=2)
        )]


async def main():
    """Run the MCP server."""
    try:
        logger.info("Starting Deep Research Assistant MCP Server...")
        
        # Initialize database
        await init_database()
        logger.info("Database initialized")
        
        # Log registered tools
        all_tools = mcp_registry.get_all_tools()
        logger.info(f"Registered {len(all_tools)} tools:")
        for tool in all_tools:
            logger.info(f"  - {tool.name}: {tool.description[:80]}...")
        
        # Run the server
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP Server running on stdio")
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options()
            )
            
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        await close_database()
        logger.info("MCP Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
