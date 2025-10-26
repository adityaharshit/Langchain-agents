@echo off
REM Start MCP Server for Deep Research Assistant

echo Starting Deep Research Assistant MCP Server...
echo.
echo To use with MCP Inspector, run in another terminal:
echo   npx @modelcontextprotocol/inspector python mcp_server.py
echo.

python mcp_server.py
