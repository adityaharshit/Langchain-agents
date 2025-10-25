"""
Database connection and session management for Deep Research Assistant.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    AsyncEngine, 
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy import text, event
from sqlalchemy.pool import NullPool
from sqlalchemy.engine import Engine

from app.config import config
from app.db.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize database connection and session factory."""
        if self._initialized:
            return
            
        logger.info(f"Initializing database connection to {config.DATABASE_URL}")
        
        # Create async engine with connection pooling
        self.engine = create_async_engine(
            config.DATABASE_URL,
            echo=config.LOG_LEVEL == "DEBUG",
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections after 1 hour
        )
        
        # Create session factory
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
        
        # Enable pgvector extension and create tables
        await self._setup_database()
        
        self._initialized = True
        logger.info("Database initialization completed")
    
    async def _setup_database(self) -> None:
        """Set up database schema and extensions."""
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        
        async with self.engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))  # For full-text search
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database schema created successfully")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session with automatic cleanup."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def get_session_direct(self) -> AsyncSession:
        """Get a direct session (caller responsible for cleanup)."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        return self.session_factory()
    
    async def close(self) -> None:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
    
    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions for common operations
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session (dependency injection helper)."""
    async with db_manager.get_session() as session:
        yield session


async def init_database() -> None:
    """Initialize the database (startup helper)."""
    await db_manager.initialize()


async def close_database() -> None:
    """Close database connections (shutdown helper)."""
    await db_manager.close()



@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log SQL queries in debug mode."""
    if config.LOG_LEVEL == "DEBUG":
        logger.debug(f"SQL Query: {statement}")
        if parameters:
            logger.debug(f"Parameters: {parameters}")


# Utility functions for database operations
async def execute_raw_sql(sql: str, params: Optional[dict] = None) -> None:
    """Execute raw SQL statement."""
    async with db_manager.get_session() as session:
        await session.execute(text(sql), params or {})


async def check_pgvector_extension() -> bool:
    """Check if pgvector extension is available."""
    try:
        async with db_manager.get_session() as session:
            result = await session.execute(
                text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            )
            return result.scalar()
    except Exception as e:
        logger.error(f"Failed to check pgvector extension: {e}")
        return False


async def create_vector_index_if_not_exists() -> None:
    """Create vector index for chunks table if it doesn't exist."""
    try:
        async with db_manager.get_session() as session:
            # Check if index exists
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE tablename = 'chunks' 
                    AND indexname = 'chunks_embedding_idx'
                )
            """))
            
            if not result.scalar():
                # Create HNSW index for vector similarity search
                await session.execute(text("""
                    CREATE INDEX chunks_embedding_idx ON chunks 
                    USING hnsw (embedding vector_cosine_ops) 
                    WITH (m = 16, ef_construction = 64)
                """))
                logger.info("Created vector index for chunks table")
            else:
                logger.info("Vector index already exists")
                
    except Exception as e:
        logger.error(f"Failed to create vector index: {e}")
        raise