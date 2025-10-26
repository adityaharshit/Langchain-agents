"""
LangChain PGVector implementation for vector storage and retrieval.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document as LangChainDocument

from app.config import config

logger = logging.getLogger(__name__)


class LangChainVectorStore:
    """
    Vector store implementation using LangChain PGVector.
    """
    
    def __init__(self, collection_name: str = "research_documents"):
        self.collection_name = config.COLLECTION_NAME
        self.connection_string = config.DATABASE_URL.replace("+asyncpg", "")
        
        # Initialize OpenAI embeddings with text-embedding-3-large
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )
        
        # Initialize PGVector
        self.vectorstore = PGVector(
            connection=self.connection_string,
            embeddings=self.embeddings,
            collection_name=self.collection_name
        )
        
        logger.info(f"LangChainVectorStore initialized with collection '{collection_name}'")
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of document IDs
        """
        try:
            # Convert to LangChain Document format
            langchain_docs = []
            for doc in documents:
                content = doc.get("cleaned_text", doc.get("content", ""))
                if content:
                    langchain_doc = LangChainDocument(
                        page_content=content,
                        metadata={
                            "url": doc.get("url", ""),
                            "title": doc.get("title", ""),
                            "source_trust_score": doc.get("source_trust_score", 0.5),
                            "language": doc.get("language", "en"),
                            "extraction_method": doc.get("extraction_method", "unknown"),
                            "author": doc.get("author", ""),
                            "publish_date": str(doc.get("publish_date", "")),
                        }
                    )
                    langchain_docs.append(langchain_doc)
            
            if not langchain_docs:
                return []
            
            # Add documents to vector store
            doc_ids = await asyncio.to_thread(
                self.vectorstore.add_documents,
                langchain_docs
            )
            
            logger.info(f"Added {len(doc_ids)} documents to vector store")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            return []
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 8,
        similarity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search results with similarity scores
        """
        try:
            similarity_threshold =config.SIMILARITY_THRESHOLD
            
            # Perform similarity search with scores
            results = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_score,
                query,
                k=k
            )
            
            # Filter and format results
            formatted_results = []
            for doc, score in results:
                # Convert distance to similarity (lower distance = higher similarity)
                similarity = 1 - score if score <= 1 else 1 / (1 + score)
                
                if similarity >= similarity_threshold:
                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": similarity,
                        "url": doc.metadata.get("url", ""),
                        "title": doc.metadata.get("title", ""),
                        "source_trust_score": doc.metadata.get("source_trust_score", 0.5)
                    })
            
            logger.info(f"Similarity search returned {len(formatted_results)} results above threshold {similarity_threshold}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        try:
            # This would require custom SQL queries to get stats from PGVector
            # For now, return basic info
            return {
                "collection_name": self.collection_name,
                "embedding_model": config.EMBEDDING_MODEL,
                "embedding_dimension": config.EMBEDDING_DIMENSION,
                "similarity_threshold": config.SIMILARITY_THRESHOLD
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}


# Global vector store instance
langchain_vector_store: Optional[LangChainVectorStore] = None


async def get_langchain_vector_store(collection_name: str = "research_documents") -> LangChainVectorStore:
    """Get or create the global LangChain vector store."""
    global langchain_vector_store
    collection_name = config.COLLECTION_NAME
    if langchain_vector_store is None or langchain_vector_store.collection_name != collection_name:
        langchain_vector_store = LangChainVectorStore(collection_name)
    
    return langchain_vector_store


async def add_documents_to_vectorstore(documents: List[Dict[str, Any]], collection_name: str = "research_documents") -> List[str]:
    """Add documents to the vector store."""
    collection_name = config.COLLECTION_NAME
    vectorstore = await get_langchain_vector_store(collection_name)
    return await vectorstore.add_documents(documents)


async def search_vectorstore(
    query: str, 
    k: int = 8, 
    collection_name: str = "research_documents",
    similarity_threshold: float = None
) -> List[Dict[str, Any]]:
    """Search the vector store."""
    collection_name = config.COLLECTION_NAME
    vectorstore = await get_langchain_vector_store(collection_name)
    return await vectorstore.similarity_search(query, k, similarity_threshold)