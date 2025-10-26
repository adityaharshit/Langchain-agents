@echo off
REM Start both MCP Server and FastAPI Application

echo Starting Deep Research Assistant with MCP Architecture
echo ========================================================
echo.

REM Start MCP Server in a new window
echo Starting MCP Server on port 8001...
start "MCP Server" cmd /k "python mcp_server.py"

REM Wait a bit for MCP server to start
timeout /t 3 /nobreak > nul

REM Start FastAPI Application
echo Starting FastAPI Application on port 8000...
echo.
echo MCP Server: http://localhost:8001
echo FastAPI App: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.

python -m app.main
