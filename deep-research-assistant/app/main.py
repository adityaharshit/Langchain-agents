"""
FastAPI application for Deep Research Assistant.
Provides REST endpoints and SSE streaming for multi-agent research.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import config
from app.schemas import (
    ChatRequest, ChatResponse, TaskStatusResponse, 
    HealthCheckResponse, StatsResponse, ErrorResponse
)
from app.db.database import init_database, close_database, db_manager
from app.langgraph_agents import process_research_query
from app.mcp_tools import progress_queue
from app.workers.vector_store import vector_store

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Deep Research Assistant",
    description="Multi-agent orchestration system for intelligent research",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global task storage (in production, use Redis or database)
active_tasks: Dict[str, Dict[str, Any]] = {}
task_results: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    try:
        logger.info("Starting Deep Research Assistant...")
        
        # Initialize database
        await init_database()
        
        # Test database connection
        if not await db_manager.health_check():
            logger.error("Database health check failed")
            raise RuntimeError("Database connection failed")
        
        logger.info("Deep Research Assistant started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    try:
        logger.info("Shutting down Deep Research Assistant...")
        
        # Close database connections
        await close_database()
        
        logger.info("Deep Research Assistant shut down successfully")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks
) -> ChatResponse:
    """
    Main chat endpoint for research queries.
    
    Args:
        request: Chat request with query and options
        background_tasks: FastAPI background tasks
        
    Returns:
        Chat response with task ID and status
    """
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task tracking
        active_tasks[task_id] = {
            "query": request.query,
            "status": "pending",
            "created_at": datetime.utcnow(),
            "started_at": None,
            "finished_at": None,
            "progress_events": [],
            "stream": request.stream
        }
        
        logger.info(f"New research task {task_id}: {request.query[:50]}...")
        
        if request.stream:
            # For streaming, start background task and return immediately
            background_tasks.add_task(
                process_research_task,
                task_id,
                request.query,
                request.dict()
            )
            
            return ChatResponse(
                task_id=task_id,
                status="accepted",
                message="Research task started. Use /stream endpoint for progress.",
                result=None
            )
        else:
            # For non-streaming, process synchronously (not recommended for long tasks)
            result = await process_research_task(task_id, request.query, request.dict())
            
            return ChatResponse(
                task_id=task_id,
                status="completed" if result.get("success") else "failed",
                message="Research completed",
                result=result.get("result")
            )
            
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/v1/stream/{task_id}")
async def stream_endpoint(task_id: str):
    """
    Server-Sent Events streaming endpoint for real-time progress.
    
    Args:
        task_id: Task identifier
        
    Returns:
        SSE stream of progress events
    """
    try:
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        logger.info(f"Starting SSE stream for task {task_id}")
        
        async def event_generator():
            """Generate SSE events for the task."""
            try:
                # Send initial event
                yield f"data: {json.dumps({
                    'event': 'progress',
                    'step': 'stream_started',
                    'payload': {'task_id': task_id},
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                })}\n\n"
                
                # Monitor task progress
                while task_id in active_tasks:
                    task_info = active_tasks[task_id]
                    
                    # Check if task is completed
                    if task_info["status"] in ["completed", "failed"]:
                        # Send final result
                        if task_id in task_results:
                            result = task_results[task_id]
                            
                            if result.get("success"):
                                yield f"data: {json.dumps({
                                    'event': 'final',
                                    'step': 'complete',
                                    'payload': result["result"],
                                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                                    'task_id': task_id
                                })}\n\n"
                            else:
                                yield f"data: {json.dumps({
                                    'event': 'error',
                                    'step': 'failed',
                                    'payload': {'error': result.get("error", "Unknown error")},
                                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                                    'task_id': task_id
                                })}\n\n"
                        
                        break
                    
                    # Check for new progress events
                    try:
                        # Get progress events from queue (non-blocking)
                        while not progress_queue.empty():
                            try:
                                event = progress_queue.get_nowait()
                                if event.get("task_id") == task_id:
                                    yield f"data: {json.dumps(event)}\n\n"
                                    
                                    # Store event in task history
                                    task_info["progress_events"].append(event)
                            except asyncio.QueueEmpty:
                                break
                    except Exception as e:
                        logger.warning(f"Progress queue error: {e}")
                    
                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.5)
                
                # Send stream end event
                yield f"data: {json.dumps({
                    'event': 'progress',
                    'step': 'stream_ended',
                    'payload': {'task_id': task_id},
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                })}\n\n"
                
            except Exception as e:
                logger.error(f"SSE generator error: {e}")
                yield f"data: {json.dumps({
                    'event': 'error',
                    'step': 'stream_error',
                    'payload': {'error': str(e)},
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                })}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")


@app.get("/api/v1/status/{task_id}", response_model=TaskStatusResponse)
async def status_endpoint(task_id: str) -> TaskStatusResponse:
    """
    Get task status and results.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Task status and results
    """
    try:
        if task_id not in active_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_info = active_tasks[task_id]
        result = task_results.get(task_id)
        
        return TaskStatusResponse(
            task_id=task_id,
            status=task_info["status"],
            created_at=task_info["created_at"],
            started_at=task_info.get("started_at"),
            finished_at=task_info.get("finished_at"),
            result=result.get("result") if result and result.get("success") else None,
            error_message=result.get("error") if result and not result.get("success") else None,
            progress_events=task_info.get("progress_events", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")


@app.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint.
    
    Returns:
        System health status
    """
    try:
        # Check database connection
        db_healthy = await db_manager.health_check()
        
        # Check vector store
        vector_stats = await vector_store.get_vector_store_stats()
        vector_healthy = bool(vector_stats)
        
        # Overall health
        overall_healthy = db_healthy and vector_healthy
        
        return HealthCheckResponse(
            status="healthy" if overall_healthy else "unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            components={
                "database": {
                    "status": "healthy" if db_healthy else "unhealthy",
                    "details": "PostgreSQL connection"
                },
                "vector_store": {
                    "status": "healthy" if vector_healthy else "unhealthy",
                    "details": f"Documents: {vector_stats.get('total_documents', 0)}"
                },
                "active_tasks": {
                    "status": "healthy",
                    "details": f"Active: {len(active_tasks)}"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            components={
                "error": {
                    "status": "unhealthy",
                    "details": str(e)
                }
            }
        )


@app.get("/api/v1/stats", response_model=StatsResponse)
async def stats_endpoint() -> StatsResponse:
    """
    Get system statistics.
    
    Returns:
        System statistics and metrics
    """
    try:
        # Get database stats
        db_stats = await vector_store.get_vector_store_stats()
        
        # Get processing stats (would be from content processor)
        processing_stats = {
            "active_tasks": len(active_tasks),
            "completed_tasks": len(task_results),
            "total_tasks": len(active_tasks) + len(task_results)
        }
        
        # Get cache stats (would be from embedding cache)
        cache_stats = {
            "cache_enabled": True,
            "cache_size": 0  # Would get from actual cache
        }
        
        return StatsResponse(
            database_stats=db_stats,
            processing_stats=processing_stats,
            cache_stats=cache_stats,
            uptime_seconds=0.0  # Would calculate actual uptime
        )
        
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message=str(exc),
            timestamp=datetime.utcnow(),
            request_id=str(uuid.uuid4())
        ).dict()
    )


async def process_research_task(task_id: str, query: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background task to process research query.
    
    Args:
        task_id: Task identifier
        query: Research query
        options: Processing options
        
    Returns:
        Processing result
    """
    try:
        # Update task status
        if task_id in active_tasks:
            active_tasks[task_id]["status"] = "running"
            active_tasks[task_id]["started_at"] = datetime.utcnow()
        
        logger.info(f"Processing research task {task_id}")
        
        # Process query through multi-agent system
        result = await process_research_query(query, task_id)
        
        # Store result
        task_results[task_id] = result
        
        # Update task status
        if task_id in active_tasks:
            active_tasks[task_id]["status"] = "completed" if result.get("success") else "failed"
            active_tasks[task_id]["finished_at"] = datetime.utcnow()
        
        logger.info(f"Research task {task_id} completed: {result.get('success', False)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Research task {task_id} failed: {e}")
        
        # Store error result
        error_result = {
            "success": False,
            "error": str(e),
            "task_id": task_id
        }
        task_results[task_id] = error_result
        
        # Update task status
        if task_id in active_tasks:
            active_tasks[task_id]["status"] = "failed"
            active_tasks[task_id]["finished_at"] = datetime.utcnow()
        
        return error_result


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=config.LOG_LEVEL.lower()
    )