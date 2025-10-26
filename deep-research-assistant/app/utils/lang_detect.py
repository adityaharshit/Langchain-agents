"""
Language detection utilities for content filtering.
"""
import re
from typing import Optional


def is_english(text: str, threshold: float = 0.3) -> bool:
    """
    Simple English language detection based on common English patterns.
    
    Args:
        text: Text to analyze
        threshold: Minimum ratio of English indicators required
        
    Returns:
        True if text appears to be English
    """
    if not text or len(text.strip()) < 10:
        return False
    
    # Convert to lowercase for analysis
    text_lower = text.lower()
    
    # Common English words and patterns
    english_indicators = [
        # Common English words
        r'\bthe\b', r'\band\b', r'\bof\b', r'\bto\b', r'\ba\b', r'\bin\b',
        r'\bis\b', r'\bit\b', r'\byou\b', r'\bthat\b', r'\bhe\b', r'\bwas\b',
        r'\bfor\b', r'\bon\b', r'\bare\b', r'\bas\b', r'\bwith\b', r'\bhis\b',
        r'\bthey\b', r'\bbe\b', r'\bat\b', r'\bone\b', r'\bhave\b', r'\bthis\b',
        r'\bfrom\b', r'\bor\b', r'\bhad\b', r'\bby\b', r'\bword\b', r'\bbut\b',
        r'\bnot\b', r'\bwhat\b', r'\ball\b', r'\bwere\b', r'\bwe\b', r'\bwhen\b',
        r'\byour\b', r'\bcan\b', r'\bsaid\b', r'\bthere\b', r'\beach\b',
        r'\bwhich\b', r'\bdo\b', r'\bhow\b', r'\btheir\b', r'\bif\b',
        
        # English-specific patterns
        r'\b\w+ing\b',  # -ing endings
        r'\b\w+ed\b',   # -ed endings
        r'\b\w+ly\b',   # -ly endings
        r'\b\w+tion\b', # -tion endings
        r'\b\w+ness\b', # -ness endings
    ]
    
    # Count English indicators
    total_indicators = len(english_indicators)
    found_indicators = 0
    
    for pattern in english_indicators:
        if re.search(pattern, text_lower):
            found_indicators += 1
    
    # Calculate ratio
    english_ratio = found_indicators / total_indicators
    
    # Additional checks for non-English scripts
    # Check for non-Latin scripts (simplified)
    non_latin_chars = len(re.findall(r'[^\x00-\x7F]', text))
    total_chars = len(text.replace(' ', ''))
    
    if total_chars > 0:
        non_latin_ratio = non_latin_chars / total_chars
        # If more than 30% non-Latin characters, likely not English
        if non_latin_ratio > 0.3:
            return False
    
    return english_ratio >= threshold


def detect_language(text: str) -> Optional[str]:
    """
    Simple language detection (English vs other).
    
    Args:
        text: Text to analyze
        
    Returns:
        'en' for English, 'other' for non-English, None if undetermined
    """
    if not text or len(text.strip()) < 10:
        return None
    
    if is_english(text):
        return 'en'
    else:
        return 'other'