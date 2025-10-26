"""
MCP Server for Deep Research Assistant.
Exposes tools via Model Context Protocol using streamable-http transport.
"""

import asyncio
import logging

from mcp.server.fastmcp import FastMCP

from app.db.database import init_database, close_database
from app.mcp import mcp_registry

# Import all tools to register them
import app.mcp_tools  # noqa: F401

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP("deep-research-assistant")


# Register all tools from the registry dynamically
def register_tools():
    """Register all tools from the MCP registry."""
    all_tools = mcp_registry.get_all_tools()
    logger.info(f"Registering {len(all_tools)} tools...")

    for mcp_tool in all_tools:
        # Create a closure to capture the current tool
        def create_tool_handler(tool):
            @mcp.tool(name=tool.name, description=tool.description)
            async def tool_handler(
                tool_input: dict = {}, agent_name: str = "MCPClient", context: dict = {}
            ) -> dict:
                """Execute the tool through direct function call."""
                try:
                    logger.info(f"Executing tool: {tool.name} for agent: {agent_name}")

                    # Execute the actual tool function (server-side execution)
                    if tool.function:
                        if asyncio.iscoroutinefunction(tool.function):
                            result = await tool.function(tool_input, context)
                        else:
                            result = tool.function(tool_input, context)
                        return result
                    else:
                        return {
                            "status": "error",
                            "error": f"Tool {tool.name} has no function",
                            "meta": {},
                        }

                except Exception as e:
                    logger.error(
                        f"Tool {tool.name} execution failed: {e}", exc_info=True
                    )
                    return {
                        "status": "error",
                        "error": str(e),
                        "tool": tool.name,
                        "meta": {},
                    }

            return tool_handler

        # Create and register the handler
        create_tool_handler(mcp_tool)
        logger.info(f"  ✓ Registered: {mcp_tool.name}")


async def initialize_server():
    """Initialize server resources."""
    logger.info("Starting Deep Research Assistant MCP Server...")

    # Initialize database
    await init_database()
    logger.info("Database initialized")

    # Register all tools
    register_tools()

    logger.info("MCP Server ready on http://localhost:8001")


async def cleanup_server():
    """Cleanup server resources."""
    logger.info("Shutting down MCP Server...")
    await close_database()
    logger.info("MCP Server stopped")


if __name__ == "__main__":
    # Initialize server resources
    asyncio.run(initialize_server())

    # Run with streamable-http transport
    logger.info("Starting MCP server with streamable-http transport...")
    mcp.run(transport="streamable-http")
