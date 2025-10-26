"""
Stable MCP Client for agents to connect to the MCP server.
Fixes async cancel-scope issues with AnyIO.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class MCPClient:
    """Resilient MCP Client for agent tool access."""

    def __init__(self, server_url: str = None):
        from app.config import config

        self.server_url = "http://localhost:8000/mcp"
        self.session: Optional[ClientSession] = None
        self.transport_cm = None
        self.read_stream = None
        self.write_stream = None
        self._connected = False
        self._available_tools: Dict[str, Any] = {}

    async def _safe_connect_once(self):
        """Perform a single connection attempt."""
        logger.info(f"Connecting to MCP server at {self.server_url}")
        self.transport_cm = streamablehttp_client(self.server_url)
        self.read_stream, self.write_stream, _ = await self.transport_cm.__aenter__()
        self.session = ClientSession(self.read_stream, self.write_stream)
        await self.session.__aenter__()
        await self.session.initialize()

        tools_result = await self.session.list_tools()
        self._available_tools = {t.name: t for t in tools_result.tools}
        self._connected = True
        logger.info(f"Connected to MCP server, tools available: {len(self._available_tools)}")

    async def connect(self, retries: int = 3, delay: float = 2.0):
        """Try to connect, with retries."""
        if self._connected:
            return
        for attempt in range(1, retries + 1):
            try:
                await self._safe_connect_once()
                return
            except Exception as e:
                logger.warning(f"MCP connection attempt {attempt}/{retries} failed: {e}")
                await asyncio.sleep(delay)
        raise ConnectionError(f"Could not connect to MCP after {retries} attempts")

    async def disconnect(self):
        """Clean up transport/session safely."""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if self.transport_cm:
                await self.transport_cm.__aexit__(None, None, None)
        except Exception as e:
            logger.error(f"MCP disconnect error: {e}", exc_info=True)
        finally:
            self.session = None
            self.transport_cm = None
            self._connected = False
            logger.info("MCP client disconnected")

    async def execute_tool(
        self,
        tool_name: str,
        agent_name: str,
        tool_input: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a tool via MCP."""
        if not self._connected:
            await self.connect()

        try:
            logger.info(f"[{agent_name}] Executing tool '{tool_name}'")
            args = {"tool_input": tool_input, "agent_name": agent_name, "context": context or {}}
            result = await self.session.call_tool(tool_name, args)

            if getattr(result, "content", None):
                import json
                return json.loads(result.content[0].text)

            return {"status": "error", "error": "Empty response", "meta": {}}
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "meta": {"agent": agent_name, "tool": tool_name},
            }


class MCPClientPool:
    """Manages a pool of MCPClient connections."""

    def __init__(self, server_url: str = None, pool_size: int = None):
        from app.config import config

        self.server_url = server_url or config.MCP_SERVER_URL
        self.pool_size = pool_size or config.MCP_CLIENT_POOL_SIZE
        self._clients: List[MCPClient] = []
        self._queue: asyncio.Queue[MCPClient] = asyncio.Queue()
        self._initialized = False

    async def initialize(self):
        """Lazy init clients on first use."""
        if self._initialized:
            return

        logger.info(f"Initializing MCP client pool ({self.pool_size})")
        for _ in range(self.pool_size):
            client = MCPClient(self.server_url)
            try:
                await client.connect()
                await self._queue.put(client)
                self._clients.append(client)
            except Exception as e:
                logger.error(f"Failed to connect MCP client: {e}")

        if self._clients:
            self._initialized = True
            logger.info(f"Initialized {len(self._clients)} MCP clients")
        else:
            logger.warning("No MCP clients available — all connections failed.")

    async def get_client(self) -> MCPClient:
        if not self._initialized:
            await self.initialize()
        return await self._queue.get()

    async def return_client(self, client: MCPClient):
        await self._queue.put(client)

    async def execute_tool(
        self, tool_name: str, agent_name: str, tool_input: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        client = await self.get_client()
        try:
            return await client.execute_tool(tool_name, agent_name, tool_input, context)
        finally:
            await self.return_client(client)

    async def shutdown(self):
        """Graceful shutdown of all clients."""
        logger.info("Shutting down MCP client pool...")
        for client in self._clients:
            await client.disconnect()
        self._clients.clear()
        self._initialized = False
        logger.info("MCP client pool shut down.")


# Singleton helpers
mcp_client_pool: Optional[MCPClientPool] = None


async def get_mcp_client_pool() -> MCPClientPool:
    global mcp_client_pool
    if mcp_client_pool is None:
        mcp_client_pool = MCPClientPool()
        # ⚠️ Don’t connect during FastAPI startup — lazy init is safer
    return mcp_client_pool


async def execute_tool_via_mcp(tool_name: str, agent_name: str, tool_input: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
    pool = await get_mcp_client_pool()
    return await pool.execute_tool(tool_name, agent_name, tool_input, context)
