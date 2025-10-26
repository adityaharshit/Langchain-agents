"""
Test script to verify MCP server connection and tool availability.
Run this after starting the MCP server to verify everything is working.
"""
import asyncio
import sys
from app.mcp_client import MCPClient


async def test_connection():
    """Test MCP server connection and list available tools."""
    print("=" * 60)
    print("MCP Connection Test")
    print("=" * 60)
    print()
    
    client = MCPClient()
    
    try:
        print("1. Connecting to MCP server...")
        await client.connect()
        print("   ✓ Connected successfully!")
        print()
        
        print("2. Listing available tools...")
        tools = client.get_available_tools()
        print(f"   ✓ Found {len(tools)} tools:")
        print()
        
        for i, tool_name in enumerate(sorted(tools), 1):
            print(f"   {i:2d}. {tool_name}")
        print()
        
        print("3. Testing a simple tool (query_decomposer)...")
        result = await client.execute_tool(
            tool_name="query_decomposer",
            agent_name="TestAgent",
            tool_input={
                "query": "What is the impact of AI on healthcare?",
                "max_subqueries": 3
            },
            context={"task_id": "test-123"}
        )
        
        if result.get("status") == "ok":
            print("   ✓ Tool execution successful!")
            subqueries = result.get("result", {}).get("subqueries", [])
            print(f"   Generated {len(subqueries)} subqueries:")
            for sq in subqueries:
                print(f"     - {sq.get('subquery', 'N/A')}")
        else:
            print(f"   ✗ Tool execution failed: {result.get('error')}")
        print()
        
        print("=" * 60)
        print("✓ All tests passed! MCP server is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure MCP server is running: python mcp_server.py")
        print("2. Check if port 8001 is accessible")
        print("3. Verify MCP_SERVER_URL in config.py")
        sys.exit(1)
    
    finally:
        await client.disconnect()


if __name__ == "__main__":
    print()
    print("Make sure the MCP server is running before running this test!")
    print("Start it with: python mcp_server.py")
    print()
    input("Press Enter to continue...")
    print()
    
    asyncio.run(test_connection())
