"""
Vector storage and retrieval operations for pgvector.
Implements semantic search with cosine similarity and confidence scoring.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import db_manager
from app.db.models import Document, Chunk
from app.workers.embedding import generate_embedding, generate_embeddings, calculate_cosine_similarity
from app.config import config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with similarity score and metadata."""
    chunk_id: int
    document_id: int
    chunk_text: str
    similarity_score: float
    document_title: str
    document_url: str
    chunk_meta: Dict[str, Any]
    token_count: int


@dataclass
class RetrievalResult:
    """Complete retrieval result with confidence assessment."""
    results: List[SearchResult]
    query: str
    total_results: int
    confidence_score: float
    retrieval_method: str
    processing_time: float


class VectorStore:
    """
    Vector storage and retrieval operations using pgvector.
    """
    
    def __init__(self):
        self.similarity_threshold = config.SIMILARITY_THRESHOLD
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.default_k = config.RETRIEVAL_K
    
    async def upsert_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        document_id: int
    ) -> List[int]:
        """
        Upsert chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            document_id: ID of the parent document
            
        Returns:
            List of created chunk IDs
        """
        if not chunks:
            return []
        
        try:
            # Generate embeddings for all chunks
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]
            embeddings = await generate_embeddings(chunk_texts)
            
            chunk_ids = []
            
            async with db_manager.get_session() as session:
                for chunk_data, embedding in zip(chunks, embeddings):
                    # Check if chunk already exists (by text hash or similar)
                    existing_chunk = await session.execute(
                        select(Chunk).where(
                            Chunk.document_id == document_id,
                            Chunk.chunk_text == chunk_data["chunk_text"]
                        )
                    )
                    existing_chunk = existing_chunk.scalar_one_or_none()
                    
                    if existing_chunk:
                        # Update existing chunk
                        existing_chunk.embedding = embedding
                        existing_chunk.token_count = chunk_data.get("token_count", 0)
                        existing_chunk.chunk_meta = chunk_data.get("chunk_meta", {})
                        chunk_ids.append(existing_chunk.id)
                    else:
                        # Create new chunk
                        chunk = Chunk(
                            document_id=document_id,
                            chunk_text=chunk_data["chunk_text"],
                            token_count=chunk_data.get("token_count", 0),
                            chunk_meta=chunk_data.get("chunk_meta", {}),
                            embedding=embedding
                        )
                        session.add(chunk)
                        await session.flush()  # Get the ID
                        chunk_ids.append(chunk.id)
            
            logger.info(f"Upserted {len(chunk_ids)} chunks for document {document_id}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Chunk upsert failed: {e}")
            return []
    
    async def semantic_search(
        self, 
        query: str, 
        k: int = None,
        similarity_threshold: float = None,
        document_ids: Optional[List[int]] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query: Search query text
            k: Number of results to return
            similarity_threshold: Minimum similarity score
            document_ids: Optional list of document IDs to search within
            
        Returns:
            List of search results ordered by similarity
        """
        k = k or self.default_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        try:
            # Generate query embedding
            query_embedding = await generate_embedding(query)
            
            if not query_embedding or all(x == 0 for x in query_embedding):
                logger.warning("Query embedding is empty or zero vector")
                return []
            
            # Build search query
            search_query = text("""
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    c.chunk_text,
                    c.token_count,
                    c.chunk_meta,
                    d.title as document_title,
                    d.url as document_url,
                    (c.embedding <=> :query_embedding) as distance,
                    (1 - (c.embedding <=> :query_embedding)) as similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                    AND d.language = 'en'
                    AND (1 - (c.embedding <=> :query_embedding)) >= :similarity_threshold
                    {document_filter}
                ORDER BY c.embedding <=> :query_embedding
                LIMIT :k
            """.format(
                document_filter="AND c.document_id = ANY(:document_ids)" if document_ids else ""
            ))
            
            params = {
                "query_embedding": query_embedding,
                "similarity_threshold": similarity_threshold,
                "k": k
            }
            
            if document_ids:
                params["document_ids"] = document_ids
            
            async with db_manager.get_session() as session:
                result = await session.execute(search_query, params)
                rows = result.fetchall()
            
            # Convert to SearchResult objects
            search_results = []
            for row in rows:
                search_results.append(SearchResult(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    chunk_text=row.chunk_text,
                    similarity_score=float(row.similarity),
                    document_title=row.document_title,
                    document_url=row.document_url,
                    chunk_meta=row.chunk_meta or {},
                    token_count=row.token_count
                ))
            
            logger.info(f"Semantic search returned {len(search_results)} results for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def hybrid_search(
        self, 
        query: str, 
        k: int = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query text
            k: Number of results to return
            semantic_weight: Weight for semantic search results
            keyword_weight: Weight for keyword search results
            
        Returns:
            List of search results with combined scores
        """
        k = k or self.default_k
        
        try:
            # Get semantic search results
            semantic_results = await self.semantic_search(query, k * 2)  # Get more for reranking
            
            # Get keyword search results
            keyword_results = await self.keyword_search(query, k * 2)
            
            # Combine and rerank results
            combined_results = self._combine_search_results(
                semantic_results, 
                keyword_results,
                semantic_weight,
                keyword_weight
            )
            
            # Return top k results
            return combined_results[:k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return await self.semantic_search(query, k)  # Fallback to semantic only
    
    async def keyword_search(
        self, 
        query: str, 
        k: int = None
    ) -> List[SearchResult]:
        """
        Perform keyword-based full-text search.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of search results ordered by text relevance
        """
        k = k or self.default_k
        
        try:
            search_query = text("""
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    c.chunk_text,
                    c.token_count,
                    c.chunk_meta,
                    d.title as document_title,
                    d.url as document_url,
                    ts_rank(to_tsvector('english', c.chunk_text), plainto_tsquery('english', :query)) as rank
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE to_tsvector('english', c.chunk_text) @@ plainto_tsquery('english', :query)
                    AND d.language = 'en'
                ORDER BY rank DESC
                LIMIT :k
            """)
            
            async with db_manager.get_session() as session:
                result = await session.execute(search_query, {"query": query, "k": k})
                rows = result.fetchall()
            
            # Convert to SearchResult objects
            search_results = []
            for row in rows:
                search_results.append(SearchResult(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    chunk_text=row.chunk_text,
                    similarity_score=float(row.rank),  # Use rank as similarity score
                    document_title=row.document_title,
                    document_url=row.document_url,
                    chunk_meta=row.chunk_meta or {},
                    token_count=row.token_count
                ))
            
            logger.info(f"Keyword search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _combine_search_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[SearchResult]:
        """Combine and rerank semantic and keyword search results."""
        # Create a map of chunk_id to results
        result_map = {}
        
        # Add semantic results
        for result in semantic_results:
            result_map[result.chunk_id] = {
                "result": result,
                "semantic_score": result.similarity_score,
                "keyword_score": 0.0
            }
        
        # Add keyword results
        for result in keyword_results:
            if result.chunk_id in result_map:
                result_map[result.chunk_id]["keyword_score"] = result.similarity_score
            else:
                result_map[result.chunk_id] = {
                    "result": result,
                    "semantic_score": 0.0,
                    "keyword_score": result.similarity_score
                }
        
        # Calculate combined scores
        combined_results = []
        for chunk_data in result_map.values():
            combined_score = (
                chunk_data["semantic_score"] * semantic_weight +
                chunk_data["keyword_score"] * keyword_weight
            )
            
            # Update the result with combined score
            result = chunk_data["result"]
            result.similarity_score = combined_score
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return combined_results
    
    async def retrieve_with_confidence(
        self, 
        query: str, 
        k: int = None,
        method: str = "semantic"
    ) -> RetrievalResult:
        """
        Retrieve results with confidence scoring.
        
        Args:
            query: Search query
            k: Number of results to return
            method: Search method ("semantic", "keyword", "hybrid")
            
        Returns:
            RetrievalResult with confidence assessment
        """
        import time
        start_time = time.time()
        
        k = k or self.default_k
        
        # Perform search based on method
        if method == "semantic":
            results = await self.semantic_search(query, k)
        elif method == "keyword":
            results = await self.keyword_search(query, k)
        elif method == "hybrid":
            results = await self.hybrid_search(query, k)
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(results, query)
        
        processing_time = time.time() - start_time
        
        return RetrievalResult(
            results=results,
            query=query,
            total_results=len(results),
            confidence_score=confidence_score,
            retrieval_method=method,
            processing_time=processing_time
        )
    
    def _calculate_confidence_score(self, results: List[SearchResult], query: str) -> float:
        """Calculate confidence score for retrieval results."""
        if not results:
            return 0.0
        
        # Factors for confidence calculation
        factors = {
            "result_count": 0.0,
            "top_similarity": 0.0,
            "average_similarity": 0.0,
            "similarity_consistency": 0.0
        }
        
        # Result count factor (more results = higher confidence, up to a point)
        factors["result_count"] = min(len(results) / self.default_k, 1.0) * 0.2
        
        # Top similarity factor
        factors["top_similarity"] = results[0].similarity_score * 0.4
        
        # Average similarity factor
        avg_similarity = sum(r.similarity_score for r in results) / len(results)
        factors["average_similarity"] = avg_similarity * 0.3
        
        # Similarity consistency factor (lower variance = higher confidence)
        if len(results) > 1:
            similarities = [r.similarity_score for r in results]
            variance = sum((s - avg_similarity) ** 2 for s in similarities) / len(similarities)
            consistency = max(0, 1 - variance)  # Lower variance = higher consistency
            factors["similarity_consistency"] = consistency * 0.1
        
        # Calculate total confidence
        confidence = sum(factors.values())
        
        return min(1.0, confidence)
    
    async def get_similar_chunks(
        self, 
        chunk_id: int, 
        k: int = 5
    ) -> List[SearchResult]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            k: Number of similar chunks to return
            
        Returns:
            List of similar chunks
        """
        try:
            async with db_manager.get_session() as session:
                # Get the reference chunk
                ref_chunk = await session.get(Chunk, chunk_id)
                if not ref_chunk or not ref_chunk.embedding:
                    return []
                
                # Search for similar chunks
                search_query = text("""
                    SELECT 
                        c.id as chunk_id,
                        c.document_id,
                        c.chunk_text,
                        c.token_count,
                        c.chunk_meta,
                        d.title as document_title,
                        d.url as document_url,
                        (1 - (c.embedding <=> :ref_embedding)) as similarity
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.id != :chunk_id
                        AND c.embedding IS NOT NULL
                        AND d.language = 'en'
                    ORDER BY c.embedding <=> :ref_embedding
                    LIMIT :k
                """)
                
                result = await session.execute(search_query, {
                    "ref_embedding": ref_chunk.embedding,
                    "chunk_id": chunk_id,
                    "k": k
                })
                rows = result.fetchall()
            
            # Convert to SearchResult objects
            similar_chunks = []
            for row in rows:
                similar_chunks.append(SearchResult(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    chunk_text=row.chunk_text,
                    similarity_score=float(row.similarity),
                    document_title=row.document_title,
                    document_url=row.document_url,
                    chunk_meta=row.chunk_meta or {},
                    token_count=row.token_count
                ))
            
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Similar chunks search failed: {e}")
            return []
    
    async def get_document_chunks(self, document_id: int) -> List[SearchResult]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of chunks for the document
        """
        try:
            async with db_manager.get_session() as session:
                query = select(Chunk, Document).join(Document).where(
                    Chunk.document_id == document_id
                ).order_by(Chunk.id)
                
                result = await session.execute(query)
                rows = result.fetchall()
            
            chunks = []
            for chunk, document in rows:
                chunks.append(SearchResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    chunk_text=chunk.chunk_text,
                    similarity_score=1.0,  # Not applicable for this query
                    document_title=document.title,
                    document_url=document.url,
                    chunk_meta=chunk.chunk_meta or {},
                    token_count=chunk.token_count
                ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Document chunks retrieval failed: {e}")
            return []
    
    async def delete_document_chunks(self, document_id: int) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Number of deleted chunks
        """
        try:
            async with db_manager.get_session() as session:
                # Count chunks before deletion
                count_query = select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
                result = await session.execute(count_query)
                chunk_count = result.scalar()
                
                # Delete chunks
                delete_query = text("DELETE FROM chunks WHERE document_id = :document_id")
                await session.execute(delete_query, {"document_id": document_id})
            
            logger.info(f"Deleted {chunk_count} chunks for document {document_id}")
            return chunk_count
            
        except Exception as e:
            logger.error(f"Chunk deletion failed: {e}")
            return 0
    
    async def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            async with db_manager.get_session() as session:
                # Count documents and chunks
                doc_count_query = select(func.count(Document.id))
                chunk_count_query = select(func.count(Chunk.id))
                embedded_chunk_query = select(func.count(Chunk.id)).where(Chunk.embedding.isnot(None))
                
                doc_count = (await session.execute(doc_count_query)).scalar()
                chunk_count = (await session.execute(chunk_count_query)).scalar()
                embedded_count = (await session.execute(embedded_chunk_query)).scalar()
                
                # Language distribution
                lang_query = text("""
                    SELECT d.language, COUNT(*) as count
                    FROM documents d
                    GROUP BY d.language
                    ORDER BY count DESC
                """)
                lang_result = await session.execute(lang_query)
                language_distribution = dict(lang_result.fetchall())
                
                # Average chunk size
                avg_tokens_query = select(func.avg(Chunk.token_count)).where(Chunk.token_count > 0)
                avg_tokens = (await session.execute(avg_tokens_query)).scalar() or 0
            
            return {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "embedded_chunks": embedded_count,
                "embedding_coverage": embedded_count / chunk_count if chunk_count > 0 else 0,
                "average_chunk_tokens": float(avg_tokens),
                "language_distribution": language_distribution,
                "similarity_threshold": self.similarity_threshold,
                "confidence_threshold": self.confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Vector store stats failed: {e}")
            return {}


# Global vector store instance
vector_store = VectorStore()


async def semantic_search(query: str, k: int = None) -> List[SearchResult]:
    """Perform semantic search using global vector store."""
    return await vector_store.semantic_search(query, k)


async def retrieve_with_confidence(query: str, k: int = None, method: str = "semantic") -> RetrievalResult:
    """Retrieve with confidence using global vector store."""
    return await vector_store.retrieve_with_confidence(query, k, method)