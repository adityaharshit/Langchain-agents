"""
Text cleaning utilities for HTML content processing.
Includes readability extraction, schema.org parsing, and content cleaning.
"""
import re
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment
from readability import Document

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Comprehensive text cleaning and content extraction utility.
    """
    
    def __init__(self):
        # Common navigation and advertisement selectors to remove
        self.noise_selectors = [
            'nav', 'header', 'footer', 'aside', 'sidebar',
            '.nav', '.navigation', '.menu', '.header', '.footer',
            '.sidebar', '.aside', '.advertisement', '.ad', '.ads',
            '.social', '.share', '.sharing', '.comments', '.comment',
            '.related', '.recommended', '.popup', '.modal',
            '.cookie', '.gdpr', '.newsletter', '.subscription'
        ]
        
        # Selectors for main content
        self.content_selectors = [
            'article', 'main', '.content', '.post', '.entry',
            '.article', '.story', '.text', '.body', '.main'
        ]
    
    def clean_html_content(self, html: str, url: str = "") -> Dict[str, Any]:
        """
        Clean HTML content and extract structured information.
        
        Args:
            html: Raw HTML content
            url: Source URL for context
            
        Returns:
            Dictionary with cleaned content and metadata
        """
        try:
            if not html or not html.strip():
                return self._empty_result()
            
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract metadata first
            metadata = self._extract_metadata(soup, url)
            
            # Use readability for main content extraction
            cleaned_content = self._extract_with_readability(html, url)
            
            # If readability fails, use manual extraction
            if not cleaned_content.get("content"):
                cleaned_content = self._manual_content_extraction(soup)
            
            # Combine results
            result = {
                **metadata,
                **cleaned_content,
                "url": url,
                "extraction_method": cleaned_content.get("method", "manual")
            }
            
            # Final text cleaning
            result["cleaned_text"] = self._clean_text(result.get("content", ""))
            
            return result
            
        except Exception as e:
            logger.error(f"HTML cleaning failed for {url}: {e}")
            return self._empty_result(error=str(e))
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML head and structured data."""
        metadata = {
            "title": "",
            "description": "",
            "author": "",
            "publish_date": None,
            "keywords": [],
            "language": "en",
            "schema_data": {}
        }
        
        try:
            # Title extraction
            title_tag = soup.find('title')
            if title_tag:
                metadata["title"] = title_tag.get_text().strip()
            
            # Meta description
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag:
                metadata["description"] = desc_tag.get('content', '').strip()
            
            # Author
            author_tag = soup.find('meta', attrs={'name': 'author'})
            if author_tag:
                metadata["author"] = author_tag.get('content', '').strip()
            
            # Keywords
            keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
            if keywords_tag:
                keywords = keywords_tag.get('content', '').strip()
                metadata["keywords"] = [k.strip() for k in keywords.split(',') if k.strip()]
            
            # Language
            html_tag = soup.find('html')
            if html_tag and html_tag.get('lang'):
                metadata["language"] = html_tag.get('lang')[:2].lower()
            
            # Open Graph metadata
            og_title = soup.find('meta', property='og:title')
            if og_title and not metadata["title"]:
                metadata["title"] = og_title.get('content', '').strip()
            
            og_desc = soup.find('meta', property='og:description')
            if og_desc and not metadata["description"]:
                metadata["description"] = og_desc.get('content', '').strip()
            
            # Publication date (various formats)
            metadata["publish_date"] = self._extract_publish_date(soup)
            
            # Schema.org structured data
            metadata["schema_data"] = self._extract_schema_data(soup)
            
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
        
        return metadata
    
    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract publication date from various HTML elements."""
        date_selectors = [
            ('meta', {'name': 'article:published_time'}),
            ('meta', {'property': 'article:published_time'}),
            ('meta', {'name': 'pubdate'}),
            ('meta', {'name': 'date'}),
            ('time', {'datetime': True}),
            ('.date', {}),
            ('.published', {}),
            ('.publish-date', {})
        ]
        
        for tag_name, attrs in date_selectors:
            try:
                if tag_name == 'meta':
                    tag = soup.find(tag_name, attrs=attrs)
                    if tag:
                        date_str = tag.get('content', '')
                else:
                    tag = soup.find(tag_name, attrs=attrs) if attrs else soup.select_one(tag_name)
                    if tag:
                        date_str = tag.get('datetime') or tag.get_text()
                
                if date_str:
                    return self._parse_date_string(date_str.strip())
                    
            except Exception:
                continue
        
        return None
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats."""
        date_formats = [
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S%z',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y'
        ]
        
        # Clean date string
        date_str = re.sub(r'[^\w\s:+-]', ' ', date_str)
        date_str = ' '.join(date_str.split())
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _extract_schema_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract Schema.org structured data."""
        schema_data = {}
        
        try:
            # JSON-LD structured data
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_ld_scripts:
                try:
                    import json
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        schema_data.update(data)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                schema_data.update(item)
                except json.JSONDecodeError:
                    continue
            
            # Microdata
            microdata_items = soup.find_all(attrs={'itemscope': True})
            for item in microdata_items:
                item_type = item.get('itemtype', '')
                if 'schema.org' in item_type:
                    schema_data[item_type] = self._extract_microdata_properties(item)
            
        except Exception as e:
            logger.warning(f"Schema data extraction failed: {e}")
        
        return schema_data
    
    def _extract_microdata_properties(self, element) -> Dict[str, Any]:
        """Extract microdata properties from an element."""
        properties = {}
        
        for prop_element in element.find_all(attrs={'itemprop': True}):
            prop_name = prop_element.get('itemprop')
            prop_value = prop_element.get('content') or prop_element.get_text().strip()
            
            if prop_name and prop_value:
                properties[prop_name] = prop_value
        
        return properties
    
    def _extract_with_readability(self, html: str, url: str) -> Dict[str, Any]:
        """Extract main content using readability algorithm."""
        try:
            doc = Document(html)
            
            return {
                "content": doc.content(),
                "title": doc.title(),
                "method": "readability"
            }
            
        except Exception as e:
            logger.warning(f"Readability extraction failed: {e}")
            return {"content": "", "method": "readability_failed"}
    
    def _manual_content_extraction(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Manual content extraction as fallback."""
        try:
            # Remove noise elements
            self._remove_noise_elements(soup)
            
            # Try to find main content
            main_content = self._find_main_content(soup)
            
            if main_content:
                content_text = self._extract_text_from_element(main_content)
            else:
                # Fallback to body text
                body = soup.find('body')
                content_text = self._extract_text_from_element(body) if body else soup.get_text()
            
            return {
                "content": content_text,
                "method": "manual"
            }
            
        except Exception as e:
            logger.error(f"Manual extraction failed: {e}")
            return {"content": soup.get_text() if soup else "", "method": "fallback"}
    
    def _remove_noise_elements(self, soup: BeautifulSoup) -> None:
        """Remove navigation, ads, and other noise elements."""
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Remove script and style tags
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        
        # Remove noise elements by selector
        for selector in self.noise_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Remove elements with noise-indicating attributes
        noise_patterns = [
            r'nav', r'menu', r'header', r'footer', r'sidebar',
            r'ad', r'advertisement', r'social', r'share', r'comment'
        ]
        
        for pattern in noise_patterns:
            for element in soup.find_all(attrs={'class': re.compile(pattern, re.I)}):
                element.decompose()
            for element in soup.find_all(attrs={'id': re.compile(pattern, re.I)}):
                element.decompose()
    
    def _find_main_content(self, soup: BeautifulSoup):
        """Find the main content element."""
        # Try content selectors in order of preference
        for selector in self.content_selectors:
            element = soup.select_one(selector)
            if element and self._has_substantial_text(element):
                return element
        
        # Look for the element with most text content
        candidates = soup.find_all(['div', 'section', 'article'])
        best_candidate = None
        max_text_length = 0
        
        for candidate in candidates:
            text_length = len(candidate.get_text().strip())
            if text_length > max_text_length:
                max_text_length = text_length
                best_candidate = candidate
        
        return best_candidate if max_text_length > 200 else None
    
    def _has_substantial_text(self, element) -> bool:
        """Check if element has substantial text content."""
        text = element.get_text().strip()
        return len(text) > 200 and len(text.split()) > 30
    
    def _extract_text_from_element(self, element) -> str:
        """Extract clean text from HTML element."""
        if not element:
            return ""
        
        # Get text with some structure preservation
        text_parts = []
        
        for child in element.descendants:
            if child.name in ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                text = child.get_text().strip()
                if text and text not in text_parts:
                    text_parts.append(text)
            elif child.name == 'br':
                text_parts.append('\n')
        
        if not text_parts:
            # Fallback to simple text extraction
            return element.get_text()
        
        return '\n'.join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """Final text cleaning and normalization."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove common web artifacts
        artifacts = [
            r'Cookie Policy',
            r'Privacy Policy',
            r'Terms of Service',
            r'Subscribe to newsletter',
            r'Follow us on',
            r'Share this article',
            r'Related articles',
            r'Advertisement'
        ]
        
        for artifact in artifacts:
            text = re.sub(artifact, '', text, flags=re.IGNORECASE)
        
        return text
    
    def _empty_result(self, error: str = None) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "title": "",
            "description": "",
            "author": "",
            "publish_date": None,
            "keywords": [],
            "language": "en",
            "schema_data": {},
            "content": "",
            "cleaned_text": "",
            "extraction_method": "failed",
            "error": error
        }


async def extract_semantic_content(html: str, url: str, extract_schema: bool = True) -> Dict[str, Any]:
    """
    Extract semantic content using schema.org and readability heuristics.
    
    Args:
        html: Raw HTML content
        url: Source URL
        extract_schema: Whether to extract schema.org data
        
    Returns:
        Dictionary with extracted content and metadata
    """
    cleaner = TextCleaner()
    result = cleaner.clean_html_content(html, url)
    
    if not extract_schema:
        result.pop("schema_data", None)
    
    return result


def estimate_reading_time(text: str) -> int:
    """
    Estimate reading time in minutes.
    
    Args:
        text: Text content
        
    Returns:
        Estimated reading time in minutes
    """
    if not text:
        return 0
    
    # Average reading speed: 200-250 words per minute
    word_count = len(text.split())
    reading_time = max(1, round(word_count / 225))
    
    return reading_time


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract key phrases from text using simple heuristics.
    
    Args:
        text: Text content
        max_phrases: Maximum number of phrases to return
        
    Returns:
        List of key phrases
    """
    if not text:
        return []
    
    # Simple key phrase extraction
    # In production, would use NLP libraries like spaCy or NLTK
    
    # Find capitalized phrases (potential proper nouns)
    capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    
    # Find quoted phrases
    quoted_phrases = re.findall(r'"([^"]+)"', text)
    
    # Combine and deduplicate
    phrases = list(set(capitalized_phrases + quoted_phrases))
    
    # Filter by length and relevance
    filtered_phrases = [
        phrase for phrase in phrases 
        if 2 <= len(phrase.split()) <= 4 and len(phrase) > 5
    ]
    
    return filtered_phrases[:max_phrases]


def calculate_content_quality_score(content_data: Dict[str, Any]) -> float:
    """
    Calculate a quality score for extracted content.
    
    Args:
        content_data: Extracted content data
        
    Returns:
        Quality score between 0.0 and 1.0
    """
    score = 0.0
    
    # Text length score (0.3 weight)
    text_length = len(content_data.get("cleaned_text", ""))
    if text_length > 1000:
        score += 0.3
    elif text_length > 500:
        score += 0.2
    elif text_length > 200:
        score += 0.1
    
    # Metadata completeness score (0.3 weight)
    metadata_fields = ["title", "author", "publish_date", "description"]
    filled_fields = sum(1 for field in metadata_fields if content_data.get(field))
    score += (filled_fields / len(metadata_fields)) * 0.3
    
    # Structure score (0.2 weight)
    if content_data.get("extraction_method") == "readability":
        score += 0.2
    elif content_data.get("extraction_method") == "manual":
        score += 0.1
    
    # Schema data score (0.2 weight)
    if content_data.get("schema_data"):
        score += 0.2
    
    return min(1.0, score)