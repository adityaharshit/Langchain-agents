"""
Model Context Protocol (MCP) implementation for Deep Research Assistant.
Provides tool decoration, registration, and agent-scoped tool discovery.
"""
import asyncio
import inspect
import logging
from typing import Dict, List, Any, Callable, Optional, Set
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)


class ToolStatus(Enum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MCPTool:
    """MCP Tool definition with metadata."""
    name: str
    description: str
    function: Callable
    allowed_agents: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    return_schema: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Extract function signature and docstring."""
        if not self.description and self.function.__doc__:
            self.description = self.function.__doc__.strip()
        
        # Extract parameter information from function signature
        sig = inspect.signature(self.function)
        for param_name, param in sig.parameters.items():
            if param_name not in ['tool_input', 'context']:
                self.parameters[param_name] = {
                    'type': param.annotation.__name__ if param.annotation != inspect.Parameter.empty else 'Any',
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'required': param.default == inspect.Parameter.empty
                }


class MCPRegistry:
    """Registry for MCP tools with agent-scoped discovery."""
    
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}
        self._agent_tools: Dict[str, Set[str]] = {}
        self._execution_log: List[Dict[str, Any]] = []
    
    def register_tool(self, tool: MCPTool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
        
        # Update agent-tool mappings
        for agent in tool.allowed_agents:
            if agent not in self._agent_tools:
                self._agent_tools[agent] = set()
            self._agent_tools[agent].add(tool.name)
        
        logger.info(f"Registered tool '{tool.name}' for agents: {tool.allowed_agents}")
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)
    
    def get_tools_for_agent(self, agent_name: str) -> List[MCPTool]:
        """Get all tools allowed for a specific agent."""
        if agent_name not in self._agent_tools:
            return []
        
        tool_names = self._agent_tools[agent_name]
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_all_tools(self) -> List[MCPTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def validate_tool_access(self, tool_name: str, agent_name: str) -> bool:
        """Validate if an agent can access a specific tool."""
        tool = self.get_tool(tool_name)
        if not tool:
            return False
        return agent_name in tool.allowed_agents
    
    async def execute_tool(self, tool_name: str, agent_name: str, tool_input: dict, context: dict = None) -> dict:
        """Execute a tool with validation and logging."""
        # Validate tool access
        if not self.validate_tool_access(tool_name, agent_name):
            error_msg = f"Agent '{agent_name}' not allowed to use tool '{tool_name}'"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "meta": {"agent": agent_name, "tool": tool_name}
            }
        
        tool = self.get_tool(tool_name)
        if not tool:
            error_msg = f"Tool '{tool_name}' not found"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "meta": {"agent": agent_name, "tool": tool_name}
            }
        
        # Log execution attempt
        execution_id = len(self._execution_log)
        log_entry = {
            "id": execution_id,
            "tool_name": tool_name,
            "agent_name": agent_name,
            "input": tool_input,
            "context": context or {},
            "status": ToolStatus.RUNNING.value,
            "started_at": asyncio.get_event_loop().time(),
            "result": None,
            "error": None
        }
        self._execution_log.append(log_entry)
        
        try:
            logger.info(f"Executing tool '{tool_name}' for agent '{agent_name}'")
            
            # Execute the tool function
            if asyncio.iscoroutinefunction(tool.function):
                result = await tool.function(tool_input, context or {})
            else:
                result = tool.function(tool_input, context or {})
            
            # Update log entry
            log_entry.update({
                "status": ToolStatus.COMPLETED.value,
                "result": result,
                "finished_at": asyncio.get_event_loop().time()
            })
            
            logger.info(f"Tool '{tool_name}' executed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"Tool '{tool_name}' execution failed: {e}", exc_info=True)
            
            # Update log entry
            log_entry.update({
                "status": ToolStatus.FAILED.value,
                "error": error_msg,
                "finished_at": asyncio.get_event_loop().time()
            })
            
            return {
                "status": "error",
                "error": error_msg,
                "meta": {"agent": agent_name, "tool": tool_name}
            }
    
    def get_execution_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get tool execution log."""
        if limit:
            return self._execution_log[-limit:]
        return self._execution_log.copy()
    
    def get_agent_stats(self) -> Dict[str, Dict[str, int]]:
        """Get execution statistics by agent."""
        stats = {}
        for entry in self._execution_log:
            agent = entry["agent_name"]
            if agent not in stats:
                stats[agent] = {"total": 0, "completed": 0, "failed": 0}
            
            stats[agent]["total"] += 1
            if entry["status"] == ToolStatus.COMPLETED.value:
                stats[agent]["completed"] += 1
            elif entry["status"] == ToolStatus.FAILED.value:
                stats[agent]["failed"] += 1
        
        return stats


# Global registry instance
mcp_registry = MCPRegistry()


def mcp_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    allowed_agents: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
):
    """
    Decorator to register a function as an MCP tool.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        allowed_agents: List of agent names allowed to use this tool
        tags: Optional tags for categorization
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__.strip() if func.__doc__ else "")
        tool_agents = allowed_agents or []
        tool_tags = tags or []
        
        # Create and register the tool
        tool = MCPTool(
            name=tool_name,
            description=tool_description,
            function=func,
            allowed_agents=tool_agents,
            tags=tool_tags
        )
        
        mcp_registry.register_tool(tool)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        # Add metadata to the wrapper function
        wrapper._mcp_tool = tool
        wrapper._mcp_name = tool_name
        
        return wrapper
    
    return decorator


# Utility functions for tool management
def get_tools_by_tag(tag: str) -> List[MCPTool]:
    """Get all tools with a specific tag."""
    return [tool for tool in mcp_registry.get_all_tools() if tag in tool.tags]


def get_tool_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a tool."""
    tool = mcp_registry.get_tool(tool_name)
    if not tool:
        return None
    
    return {
        "name": tool.name,
        "description": tool.description,
        "allowed_agents": tool.allowed_agents,
        "parameters": tool.parameters,
        "tags": tool.tags,
        "function_name": tool.function.__name__,
        "module": tool.function.__module__
    }


def list_all_tools() -> Dict[str, Dict[str, Any]]:
    """List all registered tools with their information."""
    return {
        tool.name: get_tool_info(tool.name)
        for tool in mcp_registry.get_all_tools()
    }


def validate_agent_tool_access(agent_name: str, tool_names: List[str]) -> Dict[str, bool]:
    """Validate agent access to multiple tools."""
    return {
        tool_name: mcp_registry.validate_tool_access(tool_name, agent_name)
        for tool_name in tool_names
    }