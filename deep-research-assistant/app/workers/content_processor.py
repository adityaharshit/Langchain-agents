"""
Content processing pipeline for document cleaning, language detection, and metadata extraction.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

from app.workers.scraper_worker import DocumentData
from app.utils.text_cleaner import TextCleaner, calculate_content_quality_score
from app.utils.lang_detect import detect_language, filter_english_documents
from app.utils.chunker import SemanticChunker, ChunkData
from app.workers.embedding import generate_embeddings
from app.db.database import db_manager
from app.db.models import Document, Chunk
from sqlalchemy import select

logger = logging.getLogger(__name__)


class ContentProcessor:
    """
    Content processing pipeline for scraped documents.
    Handles cleaning, language detection, chunking, and storage preparation.
    """

    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.chunker = SemanticChunker()

        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "documents_filtered": 0,
            "chunks_created": 0,
            "processing_errors": 0,
        }

    async def process_documents(
        self, documents: List[DocumentData], store_in_db: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a list of scraped documents through the complete pipeline.

        Args:
            documents: List of DocumentData objects from scraping
            store_in_db: Whether to store results in database

        Returns:
            List of processed document dictionaries
        """
        if not documents:
            return []

        logger.info(f"Processing {len(documents)} documents through content pipeline")

        # Step 1: Filter English documents
        english_documents = await self._filter_english_documents(documents)

        # Step 2: Clean and enhance content
        cleaned_documents = await self._clean_and_enhance_documents(english_documents)

        # Step 3: Create semantic chunks
        chunked_documents = await self._create_document_chunks(cleaned_documents)

        # Step 4: Generate embeddings
        embedded_documents = await self._generate_document_embeddings(chunked_documents)

        # Step 5: Store in database if requested
        if store_in_db:
            stored_documents = await self._store_documents_in_db(embedded_documents)
        else:
            stored_documents = embedded_documents

        # Update statistics
        self.stats["documents_processed"] += len(documents)
        self.stats["documents_filtered"] += len(documents) - len(english_documents)
        self.stats["chunks_created"] += sum(
            len(doc.get("chunks", [])) for doc in stored_documents
        )

        logger.info(
            f"Content processing complete: {len(stored_documents)} documents, {sum(len(doc.get('chunks', [])) for doc in stored_documents)} chunks"
        )

        return stored_documents

    async def _filter_english_documents(
        self, documents: List[DocumentData]
    ) -> List[DocumentData]:
        """Filter documents to keep only English content."""
        english_docs = []

        for doc in documents:
            try:
                # Skip if already marked as non-English
                if doc.language and doc.language != "en":
                    logger.debug(
                        f"Skipping non-English document: {doc.url} ({doc.language})"
                    )
                    continue

                # Detect language if not already done
                if not doc.language or doc.language == "unknown":
                    if doc.cleaned_text:
                        lang_result = detect_language(doc.cleaned_text)
                        doc.language = lang_result["language"]

                        if not lang_result["is_english"]:
                            logger.info(
                                f"Filtered non-English document: {doc.url} (detected: {lang_result['language']})"
                            )
                            continue
                    else:
                        logger.warning(f"No content to analyze for language: {doc.url}")
                        continue

                english_docs.append(doc)

            except Exception as e:
                logger.error(f"Language filtering failed for {doc.url}: {e}")
                continue

        logger.info(
            f"Language filtering: {len(english_docs)} English documents out of {len(documents)}"
        )
        return english_docs

    async def _clean_and_enhance_documents(
        self, documents: List[DocumentData]
    ) -> List[Dict[str, Any]]:
        """Clean and enhance document content and metadata."""
        enhanced_docs = []

        for doc in documents:
            try:
                # Re-clean content if needed (scraper may have done basic cleaning)
                if doc.raw_html and (
                    not doc.cleaned_text or len(doc.cleaned_text) < 100
                ):
                    cleaned_data = self.text_cleaner.clean_html_content(
                        doc.raw_html, doc.url
                    )
                    doc.cleaned_text = cleaned_data.get(
                        "cleaned_text", doc.cleaned_text
                    )

                    # Update metadata if missing
                    if not doc.title:
                        doc.title = cleaned_data.get("title", "")
                    if not doc.description:
                        doc.description = cleaned_data.get("description", "")
                    if not doc.author:
                        doc.author = cleaned_data.get("author", "")

                # Calculate content quality score
                content_quality = calculate_content_quality_score(
                    {
                        "cleaned_text": doc.cleaned_text,
                        "title": doc.title,
                        "author": doc.author,
                        "publish_date": doc.publish_date,
                        "description": doc.description,
                        "extraction_method": doc.extraction_method,
                    }
                )

                # Skip low-quality content
                if content_quality < 0.3:
                    logger.info(
                        f"Skipping low-quality content: {doc.url} (quality: {content_quality:.2f})"
                    )
                    continue

                # Create enhanced document dictionary
                enhanced_doc = {
                    "url": doc.url,
                    "title": doc.title or "Untitled",
                    "cleaned_text": doc.cleaned_text,
                    "raw_html": doc.raw_html,
                    "language": doc.language,
                    "publish_date": self._parse_publish_date(doc.publish_date),
                    "source_trust_score": max(doc.source_trust_score, content_quality),
                    "license": doc.license,
                    "author": doc.author,
                    "description": doc.description,
                    "keywords": doc.keywords or [],
                    "extraction_method": doc.extraction_method,
                    "scraping_time": doc.scraping_time,
                    "content_quality": content_quality,
                    "word_count": len(doc.cleaned_text.split())
                    if doc.cleaned_text
                    else 0,
                    "char_count": len(doc.cleaned_text) if doc.cleaned_text else 0,
                    "processing_metadata": {
                        "processed_at": datetime.utcnow(),
                        "content_hash": self._calculate_content_hash(doc.cleaned_text),
                    },
                }

                enhanced_docs.append(enhanced_doc)

            except Exception as e:
                logger.error(f"Document enhancement failed for {doc.url}: {e}")
                self.stats["processing_errors"] += 1
                continue

        logger.info(f"Document enhancement: {len(enhanced_docs)} documents enhanced")
        return enhanced_docs

    async def _create_document_chunks(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create semantic chunks for each document."""
        chunked_docs = []

        for doc in documents:
            try:
                if not doc.get("cleaned_text"):
                    logger.warning(f"No content to chunk for {doc['url']}")
                    doc["chunks"] = []
                    chunked_docs.append(doc)
                    continue

                # Create chunks using semantic chunker
                chunk_metadata = {
                    "url": doc["url"],
                    "title": doc["title"],
                    "author": doc.get("author", ""),
                    "publish_date": doc.get("publish_date"),
                }

                chunks = await self.chunker.chunk_document(
                    doc["cleaned_text"], chunk_metadata
                )

                # Convert ChunkData objects to dictionaries
                chunk_dicts = []
                for i, chunk in enumerate(chunks):
                    chunk_dict = {
                        "chunk_text": chunk.chunk_text,
                        "token_count": chunk.token_count,
                        "chunk_meta": {
                            **chunk.chunk_meta,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        },
                        "embedding": None,  # Will be filled in next step
                    }
                    chunk_dicts.append(chunk_dict)

                doc["chunks"] = chunk_dicts
                doc["total_chunks"] = len(chunk_dicts)
                doc["total_tokens"] = sum(chunk.token_count for chunk in chunks)

                chunked_docs.append(doc)

            except Exception as e:
                logger.error(f"Chunking failed for {doc['url']}: {e}")
                doc["chunks"] = []
                doc["chunking_error"] = str(e)
                chunked_docs.append(doc)
                self.stats["processing_errors"] += 1

        total_chunks = sum(len(doc.get("chunks", [])) for doc in chunked_docs)
        logger.info(
            f"Document chunking: {total_chunks} chunks created from {len(documents)} documents"
        )

        return chunked_docs

    async def _generate_document_embeddings(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for all document chunks."""
        embedded_docs = []

        for doc in documents:
            try:
                chunks = doc.get("chunks", [])
                if not chunks:
                    embedded_docs.append(doc)
                    continue

                # Extract chunk texts for batch embedding
                chunk_texts = [chunk["chunk_text"] for chunk in chunks]

                # Generate embeddings in batch
                embeddings = await generate_embeddings(chunk_texts)

                # Assign embeddings to chunks
                for chunk, embedding in zip(chunks, embeddings):
                    chunk["embedding"] = embedding

                doc["embedding_generated"] = True
                embedded_docs.append(doc)

            except Exception as e:
                logger.error(f"Embedding generation failed for {doc['url']}: {e}")
                doc["embedding_generated"] = False
                doc["embedding_error"] = str(e)
                embedded_docs.append(doc)
                self.stats["processing_errors"] += 1

        total_embeddings = sum(
            len(doc.get("chunks", []))
            for doc in embedded_docs
            if doc.get("embedding_generated")
        )
        logger.info(f"Embedding generation: {total_embeddings} embeddings created")

        return embedded_docs

    async def _store_documents_in_db(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Store processed documents and chunks in the database."""
        stored_docs = []

        async with db_manager.get_session() as session:
            for doc in documents:
                try:
                    # Check if document already exists
                    existing_doc = await session.execute(
                        select(Document).where(Document.url == doc["url"])
                    )
                    existing_doc = existing_doc.scalar_one_or_none()

                    if existing_doc:
                        # Update existing document
                        existing_doc.title = doc["title"]
                        existing_doc.cleaned_text = doc["cleaned_text"]
                        existing_doc.raw_html = doc["raw_html"]
                        existing_doc.language = doc["language"]
                        existing_doc.publish_date = doc.get("publish_date")
                        existing_doc.source_trust_score = doc["source_trust_score"]
                        existing_doc.license = doc.get("license")

                        document_id = existing_doc.id

                        # Delete existing chunks
                        await session.execute(
                            select(Chunk).where(Chunk.document_id == document_id)
                        )
                        existing_chunks = await session.execute(
                            select(Chunk).where(Chunk.document_id == document_id)
                        )
                        for chunk in existing_chunks.scalars():
                            await session.delete(chunk)
                    else:
                        # Create new document
                        new_doc = Document(
                            url=doc["url"],
                            title=doc["title"],
                            publish_date=doc.get("publish_date"),
                            raw_html=doc["raw_html"],
                            cleaned_text=doc["cleaned_text"],
                            language=doc["language"],
                            source_trust_score=doc["source_trust_score"],
                            license=doc.get("license"),
                        )

                        session.add(new_doc)
                        await session.flush()  # Get the document ID
                        document_id = new_doc.id

                    # Store chunks
                    chunk_ids = []
                    for chunk_data in doc.get("chunks", []):
                        chunk = Chunk(
                            document_id=document_id,
                            chunk_text=chunk_data["chunk_text"],
                            token_count=chunk_data["token_count"],
                            chunk_meta=chunk_data["chunk_meta"],
                            embedding=chunk_data.get("embedding"),
                        )
                        session.add(chunk)
                        await session.flush()
                        chunk_ids.append(chunk.id)

                    # Update document with database info
                    doc["document_id"] = document_id
                    doc["chunk_ids"] = chunk_ids
                    doc["stored_in_db"] = True

                    stored_docs.append(doc)

                except Exception as e:
                    logger.error(f"Database storage failed for {doc['url']}: {e}")
                    doc["stored_in_db"] = False
                    doc["storage_error"] = str(e)
                    stored_docs.append(doc)
                    self.stats["processing_errors"] += 1

        successful_stores = sum(1 for doc in stored_docs if doc.get("stored_in_db"))
        logger.info(
            f"Database storage: {successful_stores} documents stored successfully"
        )

        return stored_docs

    def _parse_publish_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse publish date string to datetime object."""
        if not date_str:
            return None

        if isinstance(date_str, datetime):
            return date_str

        try:
            # Try common date formats
            date_formats = [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z",
            ]

            for fmt in date_formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

            return None

        except Exception:
            return None

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content for deduplication."""
        if not content:
            return ""

        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            "documents_processed": 0,
            "documents_filtered": 0,
            "chunks_created": 0,
            "processing_errors": 0,
        }


class BatchContentProcessor:
    """
    Batch processor for handling large numbers of documents efficiently.
    """

    def __init__(self, batch_size: int = 10, max_concurrent: int = 3):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.processor = ContentProcessor()

    async def process_document_batch(
        self,
        documents: List[DocumentData],
        progress_callback: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process documents in batches with progress tracking.

        Args:
            documents: List of documents to process
            progress_callback: Optional callback for progress updates

        Returns:
            List of processed documents
        """
        if not documents:
            return []

        # Split into batches
        batches = [
            documents[i : i + self.batch_size]
            for i in range(0, len(documents), self.batch_size)
        ]

        logger.info(f"Processing {len(documents)} documents in {len(batches)} batches")

        # Process batches with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_batch(
            batch_idx: int, batch: List[DocumentData]
        ) -> List[Dict[str, Any]]:
            async with semaphore:
                try:
                    result = await self.processor.process_documents(batch)

                    if progress_callback:
                        await progress_callback(
                            {
                                "batch_idx": batch_idx,
                                "batch_size": len(batch),
                                "processed_count": len(result),
                                "total_batches": len(batches),
                            }
                        )

                    return result

                except Exception as e:
                    logger.error(f"Batch {batch_idx} processing failed: {e}")
                    return []

        # Execute all batches
        batch_tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Combine results
        all_results = []
        for result in batch_results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Batch processing exception: {result}")

        logger.info(
            f"Batch processing complete: {len(all_results)} documents processed"
        )
        return all_results


# Global content processor instance
content_processor = ContentProcessor()


async def process_scraped_documents(
    documents: List[DocumentData], store_in_db: bool = True
) -> List[Dict[str, Any]]:
    """Process scraped documents using global processor."""
    return await content_processor.process_documents(documents, store_in_db)


async def process_documents_in_batches(
    documents: List[DocumentData],
    batch_size: int = 10,
    progress_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """Process documents in batches using batch processor."""
    batch_processor = BatchContentProcessor(batch_size=batch_size)
    return await batch_processor.process_document_batch(documents, progress_callback)
