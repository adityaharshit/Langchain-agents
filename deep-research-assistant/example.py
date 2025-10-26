import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import re
from app.workers.embedding import generate_embedding, generate_embeddings, calculate_cosine_similarity
from app.mcp import mcp_tool
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
load_dotenv()

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
        similarity_threshold = self.similarity_threshold
        
        try:
            # Generate query embedding
            query_embedding = await generate_embedding(query)
            
            if not query_embedding or all(x == 0 for x in query_embedding):
                logger.warning("Query embedding is empty or zero vector")
                return []
            
            # Ensure query_embedding is a flat list
            if isinstance(query_embedding, list) and len(query_embedding) == 1 and isinstance(query_embedding[0], list):
                query_embedding = query_embedding[0]

            print(query_embedding)
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            
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
                "query_embedding": embedding_str,
                "similarity_threshold": similarity_threshold,
                "k": k
            }
            
            if document_ids:
                params["document_ids"] = document_ids
            
            async with db_manager.get_session() as session:
                result = await session.execute(search_query, params)
                rows = result.fetchall()
            print("SEMANTIC SEARCH RESULTS:" + rows)
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
    

if __name__ == "__main__":
    import asyncio
    import json

    async def main():
        # Example input for semantic search
        tool_input = {
            "query": "Who is the best batsman in the currect indian cricket team?",
            "k": 5,
            "collection_name": "research_documents",
            "similarity_threshold": 0.4
        }

        context = {}

        result = await semantic_search_tool(tool_input, context)
        print(json.dumps(result, indent=2))

    asyncio.run(main())
