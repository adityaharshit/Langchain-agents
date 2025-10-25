"""
OpenAI embedding generation wrapper with batch processing and caching.
"""
import asyncio
import logging
import hashlib
from typing import List, Optional, Dict, Any
import time
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI

from app.config import config

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: List[float]
    token_count: int
    processing_time: float
    cached: bool = False


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, List[float]] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model."""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._get_cache_key(text, model)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        key = self._get_cache_key(text, model)
        
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = embedding
        self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_times.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


class EmbeddingGenerator:
    """
    OpenAI embedding generator with batch processing and caching.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,
        batch_size: int = None,
        max_retries: int = 3,
        enable_cache: bool = True
    ):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.EMBEDDING_MODEL
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        self.max_retries = max_retries
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Initialize cache
        self.cache = EmbeddingCache() if enable_cache else None
        
        # Rate limiting
        self.last_request_time = 0.0
        self.min_request_interval = 0.1  # 100ms between requests
        
        logger.info(f"EmbeddingGenerator initialized with model {self.model}, batch size {self.batch_size}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            return [0.0] * config.EMBEDDING_DIMENSION
        
        # Check cache first
        if self.cache:
            cached_embedding = self.cache.get(text, self.model)
            if cached_embedding:
                logger.debug("Retrieved embedding from cache")
                return cached_embedding
        
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Generate embedding
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Cache the result
            if self.cache:
                self.cache.set(text, self.model, embedding)
            
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * config.EMBEDDING_DIMENSION
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batch processing.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text and text.strip()]
        
        if not valid_texts:
            return [[0.0] * config.EMBEDDING_DIMENSION] * len(texts)
        
        # Check cache for existing embeddings
        embeddings = [None] * len(texts)
        texts_to_process = []
        
        if self.cache:
            for i, text in valid_texts:
                cached_embedding = self.cache.get(text, self.model)
                if cached_embedding:
                    embeddings[i] = cached_embedding
                else:
                    texts_to_process.append((i, text))
        else:
            texts_to_process = valid_texts
        
        # Process remaining texts in batches
        if texts_to_process:
            await self._process_texts_in_batches(texts_to_process, embeddings)
        
        # Fill in zero vectors for empty texts
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                embeddings[i] = [0.0] * config.EMBEDDING_DIMENSION
        
        logger.info(f"Generated embeddings for {len(texts)} texts ({len(texts_to_process)} new, {len(texts) - len(texts_to_process)} cached)")
        return embeddings
    
    async def _process_texts_in_batches(
        self, 
        texts_to_process: List[tuple], 
        embeddings: List[Optional[List[float]]]
    ) -> None:
        """Process texts in batches."""
        for i in range(0, len(texts_to_process), self.batch_size):
            batch = texts_to_process[i:i + self.batch_size]
            batch_texts = [text for _, text in batch]
            batch_indices = [idx for idx, _ in batch]
            
            try:
                batch_embeddings = await self._generate_batch_embeddings(batch_texts)
                
                # Store results
                for j, embedding in enumerate(batch_embeddings):
                    original_index = batch_indices[j]
                    embeddings[original_index] = embedding
                    
                    # Cache the result
                    if self.cache:
                        self.cache.set(batch_texts[j], self.model, embedding)
                
                logger.debug(f"Processed batch of {len(batch)} embeddings")
                
            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")
                # Fill with zero vectors
                for j in range(len(batch)):
                    original_index = batch_indices[j]
                    embeddings[original_index] = [0.0] * config.EMBEDDING_DIMENSION
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                await self._rate_limit()
                
                # Generate embeddings
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    encoding_format="float"
                )
                
                # Extract embeddings in order
                embeddings = [data.embedding for data in response.data]
                
                return embeddings
                
            except Exception as e:
                logger.warning(f"Batch embedding attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        # Should not reach here
        return [[0.0] * config.EMBEDDING_DIMENSION] * len(texts)
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def generate_embeddings_with_metadata(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings with detailed metadata.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of EmbeddingResult objects
        """
        results = []
        
        for text in texts:
            start_time = time.time()
            
            # Check cache
            cached = False
            if self.cache:
                embedding = self.cache.get(text, self.model)
                if embedding:
                    cached = True
                else:
                    embedding = await self.generate_embedding(text)
            else:
                embedding = await self.generate_embedding(text)
            
            processing_time = time.time() - start_time
            
            # Estimate token count (rough approximation)
            token_count = len(text.split()) * 1.3  # Rough tokens per word
            
            results.append(EmbeddingResult(
                text=text,
                embedding=embedding,
                token_count=int(token_count),
                processing_time=processing_time,
                cached=cached
            ))
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {"cache_enabled": False}
        
        return {
            "cache_enabled": True,
            "cache_size": self.cache.size(),
            "max_cache_size": self.cache.max_size,
            "cache_hit_ratio": "N/A"  # Would need to track hits/misses
        }
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")
    
    async def test_connection(self) -> bool:
        """Test OpenAI API connection."""
        try:
            test_embedding = await self.generate_embedding("test")
            return len(test_embedding) == config.EMBEDDING_DIMENSION
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False


# Global embedding generator instance
embedding_generator: Optional[EmbeddingGenerator] = None


async def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create global embedding generator."""
    global embedding_generator
    
    if embedding_generator is None:
        embedding_generator = EmbeddingGenerator()
        
        # Test connection
        if not await embedding_generator.test_connection():
            logger.warning("OpenAI embedding connection test failed")
    
    return embedding_generator


async def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using global generator.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector
    """
    generator = await get_embedding_generator()
    return await generator.generate_embedding(text)


async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts using global generator.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    generator = await get_embedding_generator()
    return await generator.generate_embeddings(texts)


def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score (-1 to 1)
    """
    if not embedding1 or not embedding2:
        return 0.0
    
    if len(embedding1) != len(embedding2):
        return 0.0
    
    try:
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Calculate magnitudes
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        # Clamp to [-1, 1] range
        return max(-1.0, min(1.0, similarity))
        
    except Exception as e:
        logger.error(f"Cosine similarity calculation failed: {e}")
        return 0.0


def find_most_similar(
    query_embedding: List[float], 
    candidate_embeddings: List[List[float]], 
    top_k: int = 5
) -> List[tuple]:
    """
    Find most similar embeddings to query.
    
    Args:
        query_embedding: Query embedding vector
        candidate_embeddings: List of candidate embedding vectors
        top_k: Number of top results to return
        
    Returns:
        List of (index, similarity_score) tuples, sorted by similarity
    """
    similarities = []
    
    for i, candidate in enumerate(candidate_embeddings):
        similarity = calculate_cosine_similarity(query_embedding, candidate)
        similarities.append((i, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


async def embed_and_store_texts(
    texts: List[str], 
    metadata_list: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Embed texts and prepare for storage.
    
    Args:
        texts: List of texts to embed
        metadata_list: Optional list of metadata dictionaries
        
    Returns:
        List of dictionaries with text, embedding, and metadata
    """
    embeddings = await generate_embeddings(texts)
    
    results = []
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        result = {
            "text": text,
            "embedding": embedding,
            "token_count": len(text.split()) * 1.3,  # Rough estimate
        }
        
        if metadata_list and i < len(metadata_list):
            result.update(metadata_list[i])
        
        results.append(result)
    
    return results