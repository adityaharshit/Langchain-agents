"""
Configuration management for Deep Research Assistant.
Handles environment variables and system constants.
"""
import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration from environment variables."""
    
    # Database Configuration
    DATABASE_URL: str = "postgresql+asyncpg://vectoruser:vectorpass@localhost:5433/vectordb"
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    ANALYSIS_MODEL: str = "gpt-4o"
    REASONING_MODEL: str = "gpt-4o"
    DECOMPOSITION_MODEL: str = "gpt-4o"
    COLLECTION_NAME: str = "research_documents"
    
    # Chunking Configuration
    MIN_CHUNK_TOKENS: int = 150
    MAX_CHUNK_TOKENS: int = 800
    CHUNK_OVERLAP_RATIO: float = 0.2
    
    # Retrieval Configuration
    RETRIEVAL_K: int = 8
    CONFIDENCE_THRESHOLD: float = 0.6
    SIMILARITY_THRESHOLD: float = 0
    
    # Scraping Configuration
    MAX_CONCURRENT_SCRAPES: int = 5
    SCRAPE_RATE_LIMIT: float = 1.0  # seconds between requests
    REQUEST_TIMEOUT: int = 30
    
    # Intelligent URL Generation
    URL_GENERATION_MODEL: str = "gpt-4o"
    MAX_GENERATED_URLS: int = 10
    URL_GENERATION_TEMPERATURE: float = 0.3
    
    # Embedding Configuration
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_DIMENSION: int = 3072  # text-embedding-3-large dimension
    
    # System Configuration
    LOG_LEVEL: str = "INFO"
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 2.0
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.DATABASE_URL = os.getenv("DATABASE_URL", self.DATABASE_URL)
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", self.EMBEDDING_MODEL)
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL)
        
        # Validate required configuration
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")


# Global configuration instance
config = Config()