"""
Semantic chunking implementation with adaptive sizing and overlap.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import tiktoken
from sentence_transformers import SentenceTransformer

from app.config import config

logger = logging.getLogger(__name__)


@dataclass
class ChunkData:
    """Data structure for document chunks."""
    chunk_text: str
    token_count: int
    chunk_meta: Dict[str, Any]
    embedding: Optional[List[float]] = None


class SemanticChunker:
    """
    Semantic chunking with adaptive sizing and overlap.
    Implements boundary detection, adaptive chunk sizing, and semantic overlap.
    """
    
    def __init__(
        self,
        min_tokens: int = None,
        max_tokens: int = None,
        overlap_ratio: float = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.min_tokens = min_tokens or config.MIN_CHUNK_TOKENS
        self.max_tokens = max_tokens or config.MAX_CHUNK_TOKENS
        self.overlap_ratio = overlap_ratio or config.CHUNK_OVERLAP_RATIO
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer(embedding_model)
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}. Using simple overlap.")
            self.sentence_model = None
        
        logger.info(f"SemanticChunker initialized: {self.min_tokens}-{self.max_tokens} tokens, {self.overlap_ratio:.1%} overlap")
    
    async def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[ChunkData]:
        """
        Creates semantic chunks with adaptive sizing and overlap.
        
        Args:
            text: Document text to chunk
            metadata: Document metadata (document_id, url, etc.)
            
        Returns:
            List of ChunkData objects
        """
        if not text or not text.strip():
            return []
        
        try:
            # Step 1: Detect semantic boundaries
            boundaries = self._detect_semantic_boundaries(text)
            
            # Step 2: Create adaptive chunks
            chunks = self._create_adaptive_chunks(text, boundaries, metadata)
            
            # Step 3: Apply semantic overlap
            overlapped_chunks = await self._apply_semantic_overlap(chunks)
            
            logger.info(f"Created {len(overlapped_chunks)} chunks from document")
            return overlapped_chunks
            
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            # Fallback to simple chunking
            return self._simple_chunk_fallback(text, metadata)
    
    def _detect_semantic_boundaries(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect semantic boundaries using headings and paragraph structure.
        
        Args:
            text: Document text
            
        Returns:
            List of boundary dictionaries with position and type
        """
        boundaries = []
        
        # Split text into lines for analysis
        lines = text.split('\n')
        current_position = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped:
                current_position += len(line) + 1
                continue
            
            boundary_type = None
            confidence = 0.0
            
            # Detect headings (various patterns)
            if self._is_heading(line_stripped):
                boundary_type = "heading"
                confidence = 0.9
            
            # Detect paragraph breaks (double newlines)
            elif i > 0 and not lines[i-1].strip():
                boundary_type = "paragraph"
                confidence = 0.6
            
            # Detect list items
            elif self._is_list_item(line_stripped):
                boundary_type = "list_item"
                confidence = 0.4
            
            # Detect section transitions (keywords)
            elif self._is_section_transition(line_stripped):
                boundary_type = "section_transition"
                confidence = 0.7
            
            if boundary_type:
                boundaries.append({
                    "position": current_position,
                    "type": boundary_type,
                    "confidence": confidence,
                    "text": line_stripped[:50],  # First 50 chars for reference
                    "line_number": i
                })
            
            current_position += len(line) + 1
        
        # Sort boundaries by position
        boundaries.sort(key=lambda x: x["position"])
        
        logger.debug(f"Detected {len(boundaries)} semantic boundaries")
        return boundaries
    
    def _is_heading(self, line: str) -> bool:
        """Check if line is likely a heading."""
        # Check for markdown-style headings
        if re.match(r'^#{1,6}\s+', line):
            return True
        
        # Check for all caps (likely heading)
        if len(line) > 3 and line.isupper() and not line.endswith('.'):
            return True
        
        # Check for numbered sections
        if re.match(r'^\d+\.?\s+[A-Z]', line):
            return True
        
        # Check for short lines that might be headings
        if len(line) < 60 and not line.endswith('.') and any(c.isupper() for c in line):
            return True
        
        return False
    
    def _is_list_item(self, line: str) -> bool:
        """Check if line is a list item."""
        # Bullet points
        if re.match(r'^[-*•]\s+', line):
            return True
        
        # Numbered lists
        if re.match(r'^\d+\.?\s+', line):
            return True
        
        # Lettered lists
        if re.match(r'^[a-zA-Z]\.?\s+', line):
            return True
        
        return False
    
    def _is_section_transition(self, line: str) -> bool:
        """Check if line indicates a section transition."""
        transition_keywords = [
            "introduction", "background", "methodology", "results", "discussion",
            "conclusion", "summary", "abstract", "overview", "analysis",
            "findings", "recommendations", "implications", "limitations"
        ]
        
        line_lower = line.lower()
        return any(keyword in line_lower for keyword in transition_keywords)
    
    def _create_adaptive_chunks(
        self, 
        text: str, 
        boundaries: List[Dict[str, Any]], 
        metadata: Dict[str, Any]
    ) -> List[ChunkData]:
        """
        Create chunks with adaptive sizing based on semantic boundaries.
        
        Args:
            text: Document text
            boundaries: Detected semantic boundaries
            metadata: Document metadata
            
        Returns:
            List of ChunkData objects
        """
        chunks = []
        
        if not boundaries:
            # No boundaries found, use simple splitting
            return self._simple_split_chunks(text, metadata)
        
        # Add document start and end boundaries
        all_boundaries = [{"position": 0, "type": "start", "confidence": 1.0}]
        all_boundaries.extend(boundaries)
        all_boundaries.append({"position": len(text), "type": "end", "confidence": 1.0})
        
        current_chunk_start = 0
        current_chunk_text = ""
        current_section = "introduction"
        current_heading = ""
        
        for i, boundary in enumerate(all_boundaries[1:], 1):  # Skip first boundary (start)
            # Extract text segment
            segment_text = text[current_chunk_start:boundary["position"]].strip()
            
            if not segment_text:
                continue
            
            # Calculate tokens for current chunk + new segment
            potential_chunk = current_chunk_text + "\n" + segment_text if current_chunk_text else segment_text
            token_count = len(self.tokenizer.encode(potential_chunk))
            
            # Update section and heading info
            if boundary["type"] == "heading":
                current_heading = boundary.get("text", "")
                current_section = self._extract_section_name(current_heading)
            
            # Decision logic for chunk boundaries
            should_split = False
            
            if token_count >= self.max_tokens:
                # Must split - chunk is too large
                should_split = True
            elif token_count >= self.min_tokens:
                # Can split - check if it's a good boundary
                if boundary["confidence"] >= 0.7:  # High confidence boundary
                    should_split = True
                elif boundary["type"] in ["heading", "section_transition"]:
                    should_split = True
            
            if should_split and current_chunk_text:
                # Create chunk from current content
                chunk_token_count = len(self.tokenizer.encode(current_chunk_text))
                
                chunk_meta = {
                    "section": current_section,
                    "heading": current_heading,
                    "position": len(chunks),
                    "boundary_type": boundary["type"],
                    "confidence": boundary["confidence"],
                    **metadata
                }
                
                chunks.append(ChunkData(
                    chunk_text=current_chunk_text.strip(),
                    token_count=chunk_token_count,
                    chunk_meta=chunk_meta
                ))
                
                # Start new chunk
                current_chunk_start = boundary["position"]
                current_chunk_text = segment_text
            else:
                # Add to current chunk
                current_chunk_text = potential_chunk
        
        # Handle remaining text
        if current_chunk_text.strip():
            chunk_token_count = len(self.tokenizer.encode(current_chunk_text))
            
            chunk_meta = {
                "section": current_section,
                "heading": current_heading,
                "position": len(chunks),
                "boundary_type": "end",
                "confidence": 1.0,
                **metadata
            }
            
            chunks.append(ChunkData(
                chunk_text=current_chunk_text.strip(),
                token_count=chunk_token_count,
                chunk_meta=chunk_meta
            ))
        
        logger.debug(f"Created {len(chunks)} adaptive chunks")
        return chunks
    
    async def _apply_semantic_overlap(self, chunks: List[ChunkData]) -> List[ChunkData]:
        """
        Apply 20% semantic overlap based on sentence-embedding similarity.
        
        Args:
            chunks: List of chunks to add overlap to
            
        Returns:
            List of chunks with semantic overlap applied
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.chunk_text
            
            # Add overlap from previous chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_text = await self._calculate_semantic_overlap(
                    prev_chunk.chunk_text, 
                    chunk_text, 
                    direction="forward"
                )
                if overlap_text:
                    chunk_text = overlap_text + "\n\n" + chunk_text
            
            # Add overlap to next chunk
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                overlap_text = await self._calculate_semantic_overlap(
                    chunk_text, 
                    next_chunk.chunk_text, 
                    direction="backward"
                )
                if overlap_text:
                    chunk_text = chunk_text + "\n\n" + overlap_text
            
            # Update token count
            new_token_count = len(self.tokenizer.encode(chunk_text))
            
            # Update chunk metadata
            updated_meta = chunk.chunk_meta.copy()
            updated_meta["has_overlap"] = i > 0 or i < len(chunks) - 1
            updated_meta["original_tokens"] = chunk.token_count
            
            overlapped_chunks.append(ChunkData(
                chunk_text=chunk_text,
                token_count=new_token_count,
                chunk_meta=updated_meta,
                embedding=chunk.embedding
            ))
        
        logger.debug(f"Applied semantic overlap to {len(overlapped_chunks)} chunks")
        return overlapped_chunks
    
    async def _calculate_semantic_overlap(
        self, 
        source_text: str, 
        target_text: str, 
        direction: str
    ) -> str:
        """
        Calculate semantic overlap between two text segments.
        
        Args:
            source_text: Source text to extract overlap from
            target_text: Target text to compare against
            direction: "forward" or "backward" overlap direction
            
        Returns:
            Overlap text or empty string
        """
        if not self.sentence_model:
            # Fallback to simple word-based overlap
            return self._simple_word_overlap(source_text, target_text, direction)
        
        try:
            # Split into sentences
            source_sentences = self._split_sentences(source_text)
            target_sentences = self._split_sentences(target_text)
            
            if not source_sentences or not target_sentences:
                return ""
            
            # Calculate target overlap size (20% of source)
            target_overlap_tokens = int(len(self.tokenizer.encode(source_text)) * self.overlap_ratio)
            
            # Get sentences for overlap based on direction
            if direction == "forward":
                # Take last sentences from source
                candidate_sentences = source_sentences[-3:]  # Last 3 sentences
            else:
                # Take first sentences from source
                candidate_sentences = source_sentences[:3]  # First 3 sentences
            
            # Find best semantic match
            best_overlap = ""
            best_similarity = 0.0
            
            for i in range(1, len(candidate_sentences) + 1):
                if direction == "forward":
                    overlap_sentences = candidate_sentences[-i:]
                else:
                    overlap_sentences = candidate_sentences[:i]
                
                overlap_text = " ".join(overlap_sentences)
                overlap_tokens = len(self.tokenizer.encode(overlap_text))
                
                # Check if overlap size is reasonable
                if overlap_tokens > target_overlap_tokens * 1.5:  # Too large
                    continue
                
                # Calculate semantic similarity
                similarity = await self._calculate_sentence_similarity(
                    overlap_text, 
                    " ".join(target_sentences[:2])  # Compare with first 2 sentences of target
                )
                
                if similarity > best_similarity and similarity > 0.3:  # Minimum similarity threshold
                    best_similarity = similarity
                    best_overlap = overlap_text
            
            return best_overlap
            
        except Exception as e:
            logger.warning(f"Semantic overlap calculation failed: {e}")
            return self._simple_word_overlap(source_text, target_text, direction)
    
    async def _calculate_sentence_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two text segments."""
        if not self.sentence_model or not text1.strip() or not text2.strip():
            return 0.0
        
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            # Calculate cosine similarity
            similarity = float(embeddings[0] @ embeddings[1] / 
                             (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Sentence similarity calculation failed: {e}")
            return 0.0
    
    def _simple_word_overlap(self, source_text: str, target_text: str, direction: str) -> str:
        """Simple word-based overlap fallback."""
        source_words = source_text.split()
        target_overlap_words = int(len(source_words) * self.overlap_ratio)
        
        if target_overlap_words == 0:
            return ""
        
        if direction == "forward":
            overlap_words = source_words[-target_overlap_words:]
        else:
            overlap_words = source_words[:target_overlap_words]
        
        return " ".join(overlap_words)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (could be enhanced with NLTK)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_section_name(self, heading: str) -> str:
        """Extract section name from heading text."""
        if not heading:
            return "content"
        
        # Clean heading text
        clean_heading = re.sub(r'^#{1,6}\s*', '', heading)  # Remove markdown
        clean_heading = re.sub(r'^\d+\.?\s*', '', clean_heading)  # Remove numbering
        clean_heading = clean_heading.lower().strip()
        
        # Map to standard section names
        section_mapping = {
            "introduction": "introduction",
            "background": "background", 
            "method": "methodology",
            "methodology": "methodology",
            "result": "results",
            "results": "results",
            "finding": "findings",
            "findings": "findings",
            "discussion": "discussion",
            "analysis": "analysis",
            "conclusion": "conclusion",
            "summary": "summary"
        }
        
        for key, section in section_mapping.items():
            if key in clean_heading:
                return section
        
        return "content"
    
    def _simple_split_chunks(self, text: str, metadata: Dict[str, Any]) -> List[ChunkData]:
        """Fallback simple chunking when no boundaries are found."""
        chunks = []
        words = text.split()
        
        current_chunk_words = []
        current_tokens = 0
        
        for word in words:
            word_tokens = len(self.tokenizer.encode(word))
            
            if current_tokens + word_tokens > self.max_tokens and current_chunk_words:
                # Create chunk
                chunk_text = " ".join(current_chunk_words)
                chunk_meta = {
                    "section": "content",
                    "heading": "",
                    "position": len(chunks),
                    "boundary_type": "token_limit",
                    "confidence": 0.5,
                    **metadata
                }
                
                chunks.append(ChunkData(
                    chunk_text=chunk_text,
                    token_count=current_tokens,
                    chunk_meta=chunk_meta
                ))
                
                # Start new chunk
                current_chunk_words = [word]
                current_tokens = word_tokens
            else:
                current_chunk_words.append(word)
                current_tokens += word_tokens
        
        # Handle remaining words
        if current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            chunk_meta = {
                "section": "content",
                "heading": "",
                "position": len(chunks),
                "boundary_type": "end",
                "confidence": 1.0,
                **metadata
            }
            
            chunks.append(ChunkData(
                chunk_text=chunk_text,
                token_count=current_tokens,
                chunk_meta=chunk_meta
            ))
        
        return chunks
    
    def _simple_chunk_fallback(self, text: str, metadata: Dict[str, Any]) -> List[ChunkData]:
        """Simple fallback chunking in case of errors."""
        try:
            return self._simple_split_chunks(text, metadata)
        except Exception as e:
            logger.error(f"Fallback chunking failed: {e}")
            # Ultimate fallback - single chunk
            return [ChunkData(
                chunk_text=text,
                token_count=len(self.tokenizer.encode(text)),
                chunk_meta={"section": "content", "error": "chunking_failed", **metadata}
            )]


# Import numpy for similarity calculations
try:
    import numpy as np
except ImportError:
    logger.warning("NumPy not available - semantic similarity will be limited")
    np = None