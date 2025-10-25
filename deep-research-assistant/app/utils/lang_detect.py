"""
Language detection filter for ensuring English-only content.
"""
import logging
from typing import Optional, Dict, Any
import re

try:
    from langdetect import detect, detect_langs, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    # Set seed for consistent results
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Language detection utility with multiple detection methods.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        
        # English indicators for heuristic detection
        self.english_indicators = {
            'common_words': [
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through',
                'during', 'before', 'after', 'above', 'below', 'between',
                'among', 'this', 'that', 'these', 'those', 'a', 'an'
            ],
            'patterns': [
                r'\b(is|are|was|were|be|been|being)\b',  # English "to be" verbs
                r'\b(have|has|had|having)\b',  # English "to have" verbs
                r'\b(do|does|did|doing|done)\b',  # English "to do" verbs
                r'\b(will|would|could|should|might|may|can)\b',  # Modal verbs
                r'\b\w+ing\b',  # -ing endings
                r'\b\w+ed\b',   # -ed endings
                r'\b\w+ly\b',   # -ly adverbs
            ]
        }
        
        # Non-English script patterns
        self.non_english_scripts = [
            r'[\u4e00-\u9fff]',  # Chinese
            r'[\u0400-\u04ff]',  # Cyrillic
            r'[\u0590-\u05ff]',  # Hebrew
            r'[\u0600-\u06ff]',  # Arabic
            r'[\u3040-\u309f]',  # Hiragana
            r'[\u30a0-\u30ff]',  # Katakana
            r'[\u0900-\u097f]',  # Devanagari
        ]
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of text using multiple methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detection results
        """
        if not text or not text.strip():
            return {
                "language": "unknown",
                "confidence": 0.0,
                "is_english": False,
                "method": "empty_text",
                "details": {}
            }
        
        # Clean text for analysis
        clean_text = self._clean_text_for_detection(text)
        
        if len(clean_text) < 10:
            return {
                "language": "unknown",
                "confidence": 0.0,
                "is_english": False,
                "method": "insufficient_text",
                "details": {"text_length": len(clean_text)}
            }
        
        # Try langdetect first (most accurate)
        if LANGDETECT_AVAILABLE:
            langdetect_result = self._detect_with_langdetect(clean_text)
            if langdetect_result["confidence"] >= self.confidence_threshold:
                return langdetect_result
        
        # Fallback to heuristic detection
        heuristic_result = self._detect_with_heuristics(clean_text)
        
        # Combine results if both available
        if LANGDETECT_AVAILABLE and langdetect_result["language"] != "unknown":
            return self._combine_detection_results(langdetect_result, heuristic_result)
        
        return heuristic_result
    
    def is_english(self, text: str, strict: bool = False) -> bool:
        """
        Check if text is in English.
        
        Args:
            text: Text to check
            strict: If True, require high confidence
            
        Returns:
            True if text is detected as English
        """
        result = self.detect_language(text)
        
        if strict:
            return result["is_english"] and result["confidence"] >= 0.8
        else:
            return result["is_english"]
    
    def filter_english_content(self, content_list: list) -> list:
        """
        Filter list of content to keep only English items.
        
        Args:
            content_list: List of text content or dictionaries with text
            
        Returns:
            Filtered list containing only English content
        """
        english_content = []
        
        for item in content_list:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or str(item)
            else:
                text = str(item)
            
            if self.is_english(text):
                english_content.append(item)
            else:
                logger.debug(f"Filtered out non-English content: {text[:50]}...")
        
        return english_content
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text for language detection."""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove numbers and special characters (keep basic punctuation)
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', ' ', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _detect_with_langdetect(self, text: str) -> Dict[str, Any]:
        """Detect language using langdetect library."""
        try:
            # Get detailed detection results
            lang_probs = detect_langs(text)
            
            if not lang_probs:
                return {
                    "language": "unknown",
                    "confidence": 0.0,
                    "is_english": False,
                    "method": "langdetect",
                    "details": {"error": "no_detection"}
                }
            
            # Get top detection
            top_lang = lang_probs[0]
            language = top_lang.lang
            confidence = top_lang.prob
            
            # Check for English variants
            is_english = language in ['en', 'en-us', 'en-gb', 'en-ca', 'en-au']
            
            return {
                "language": language,
                "confidence": confidence,
                "is_english": is_english,
                "method": "langdetect",
                "details": {
                    "all_detections": [(lang.lang, lang.prob) for lang in lang_probs[:3]],
                    "text_length": len(text)
                }
            }
            
        except LangDetectException as e:
            logger.warning(f"Langdetect failed: {e}")
            return {
                "language": "unknown",
                "confidence": 0.0,
                "is_english": False,
                "method": "langdetect",
                "details": {"error": str(e)}
            }
    
    def _detect_with_heuristics(self, text: str) -> Dict[str, Any]:
        """Detect language using heuristic methods."""
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return {
                "language": "unknown",
                "confidence": 0.0,
                "is_english": False,
                "method": "heuristic",
                "details": {"error": "no_words"}
            }
        
        # Check for non-English scripts
        for script_pattern in self.non_english_scripts:
            if re.search(script_pattern, text):
                return {
                    "language": "non-english",
                    "confidence": 0.9,
                    "is_english": False,
                    "method": "heuristic",
                    "details": {"reason": "non_english_script"}
                }
        
        # Calculate English indicators
        english_score = 0.0
        total_indicators = 0
        
        # Common English words
        common_word_count = sum(1 for word in words if word in self.english_indicators['common_words'])
        common_word_ratio = common_word_count / len(words)
        english_score += common_word_ratio * 0.4
        total_indicators += 0.4
        
        # English patterns
        pattern_matches = 0
        for pattern in self.english_indicators['patterns']:
            if re.search(pattern, text_lower):
                pattern_matches += 1
        
        pattern_score = min(pattern_matches / len(self.english_indicators['patterns']), 1.0)
        english_score += pattern_score * 0.3
        total_indicators += 0.3
        
        # Character distribution (English uses mostly ASCII)
        ascii_chars = sum(1 for char in text if ord(char) < 128)
        ascii_ratio = ascii_chars / len(text) if text else 0
        english_score += ascii_ratio * 0.3
        total_indicators += 0.3
        
        # Normalize score
        confidence = english_score / total_indicators if total_indicators > 0 else 0.0
        is_english = confidence >= 0.5
        
        return {
            "language": "en" if is_english else "unknown",
            "confidence": confidence,
            "is_english": is_english,
            "method": "heuristic",
            "details": {
                "common_word_ratio": common_word_ratio,
                "pattern_matches": pattern_matches,
                "ascii_ratio": ascii_ratio,
                "word_count": len(words)
            }
        }
    
    def _combine_detection_results(
        self, 
        langdetect_result: Dict[str, Any], 
        heuristic_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine results from multiple detection methods."""
        # Weight langdetect more heavily if it's confident
        if langdetect_result["confidence"] >= 0.8:
            primary_result = langdetect_result
            secondary_result = heuristic_result
            weight = 0.8
        else:
            # Use average if langdetect is less confident
            primary_result = langdetect_result
            secondary_result = heuristic_result
            weight = 0.6
        
        # Combine confidence scores
        combined_confidence = (
            primary_result["confidence"] * weight + 
            secondary_result["confidence"] * (1 - weight)
        )
        
        # Determine final language
        if primary_result["is_english"] and secondary_result["is_english"]:
            final_language = "en"
            is_english = True
        elif primary_result["is_english"] or secondary_result["is_english"]:
            # One method says English - use combined confidence
            final_language = "en" if combined_confidence >= 0.5 else "unknown"
            is_english = combined_confidence >= 0.5
        else:
            final_language = primary_result["language"]
            is_english = False
        
        return {
            "language": final_language,
            "confidence": combined_confidence,
            "is_english": is_english,
            "method": "combined",
            "details": {
                "langdetect": langdetect_result,
                "heuristic": heuristic_result,
                "weight": weight
            }
        }


# Global detector instance
language_detector = LanguageDetector()


def detect_language(text: str) -> Dict[str, Any]:
    """
    Detect language of text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with detection results
    """
    return language_detector.detect_language(text)


def is_english(text: str, strict: bool = False) -> bool:
    """
    Check if text is in English.
    
    Args:
        text: Text to check
        strict: If True, require high confidence
        
    Returns:
        True if text is detected as English
    """
    return language_detector.is_english(text, strict)


def filter_english_documents(documents: list) -> list:
    """
    Filter documents to keep only English content.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Filtered list of English documents
    """
    english_docs = []
    
    for doc in documents:
        # Check multiple text fields
        text_fields = ['content', 'cleaned_text', 'title', 'description']
        text_to_check = ""
        
        for field in text_fields:
            if field in doc and doc[field]:
                text_to_check += " " + str(doc[field])
        
        if text_to_check.strip():
            detection_result = detect_language(text_to_check)
            
            if detection_result["is_english"]:
                # Add language metadata to document
                doc["language_detection"] = detection_result
                doc["language"] = "en"
                english_docs.append(doc)
            else:
                logger.info(f"Filtered non-English document: {doc.get('title', 'Unknown')[:50]}...")
        else:
            logger.warning(f"Document has no text content to analyze: {doc.get('url', 'Unknown')}")
    
    return english_docs


def get_language_statistics(texts: list) -> Dict[str, Any]:
    """
    Get language detection statistics for a list of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary with language statistics
    """
    stats = {
        "total_texts": len(texts),
        "english_count": 0,
        "non_english_count": 0,
        "unknown_count": 0,
        "language_distribution": {},
        "average_confidence": 0.0,
        "detection_methods": {}
    }
    
    total_confidence = 0.0
    
    for text in texts:
        result = detect_language(text)
        
        # Count by English/non-English
        if result["is_english"]:
            stats["english_count"] += 1
        elif result["language"] == "unknown":
            stats["unknown_count"] += 1
        else:
            stats["non_english_count"] += 1
        
        # Language distribution
        lang = result["language"]
        stats["language_distribution"][lang] = stats["language_distribution"].get(lang, 0) + 1
        
        # Method distribution
        method = result["method"]
        stats["detection_methods"][method] = stats["detection_methods"].get(method, 0) + 1
        
        # Confidence tracking
        total_confidence += result["confidence"]
    
    # Calculate averages
    if texts:
        stats["average_confidence"] = total_confidence / len(texts)
        stats["english_percentage"] = (stats["english_count"] / len(texts)) * 100
    
    return stats