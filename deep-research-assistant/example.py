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

from app.mcp import mcp_tool

logger = logging.getLogger(__name__)
load_dotenv()


async def semantic_search_tool(tool_input: dict, context: dict) -> dict:
    """
    Uses PGVector semantic search to find relevant content.
    
    Args:
        tool_input: {"query": str, "k": int, "collection_name": str, "similarity_threshold": float}
        context: Additional context information
    """
    try:
        from langchain_postgres import PGVector
        from langchain_openai import OpenAIEmbeddings
        from app.config import config
        import asyncio
        
        query = tool_input.get("query", "")
        k = tool_input.get("k", 8)
        collection_name = tool_input.get("collection_name", "research_documents")
        similarity_threshold = tool_input.get("similarity_threshold", config.SIMILARITY_THRESHOLD)
        
        if not query:
            return {"status": "error", "error": "Query is required", "meta": {}}
        
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        
        # Initialize PGVector
        connection_string = config.DATABASE_URL.replace("+asyncpg", "")
        vectorstore = PGVector(
            connection=connection_string,
            embeddings=embeddings,
            collection_name=collection_name
        )
        
        # Perform semantic search
        results = await asyncio.to_thread(
            vectorstore.similarity_search_with_score,
            query,
            k=k
        )
        
        # Filter by similarity threshold and format results
        relevant_results = []
        for doc, score in results:
            # Convert distance to similarity (PGVector returns distance, lower is better)
            similarity = 1 - score if score <= 1 else 1 / (1 + score)
            
            if similarity >= similarity_threshold and similarity <= 0.6:
                relevant_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": similarity,
                    "url": doc.metadata.get("url", ""),
                    "title": doc.metadata.get("title", ""),
                    "source_trust_score": doc.metadata.get("source_trust_score", 0.5)
                })
        
        # Calculate confidence based on results
        if relevant_results:
            avg_similarity = sum(r["similarity_score"] for r in relevant_results) / len(relevant_results)
            confidence = min(avg_similarity * 1.2, 1.0)  # Boost confidence slightly
        else:
            confidence = 0.0
        
        return {
            "status": "ok",
            "result": {
                "results": relevant_results,
                "total_results": len(relevant_results),
                "confidence_score": confidence,
                "query": query,
                "collection_name": collection_name
            },
            "meta": {
                "search_method": "pgvector_semantic",
                "embedding_model": config.EMBEDDING_MODEL,
                "similarity_threshold": similarity_threshold,
                "requested_k": k,
                "returned_k": len(relevant_results)
            }
        }
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return {
            "status": "error",
            "error": f"Semantic search failed: {str(e)}",
            "meta": {}
        }

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
