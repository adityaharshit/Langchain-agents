"""
Asynchronous web scraping worker with rate limiting and robots.txt compliance.
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse, robots
from urllib.robotparser import RobotFileParser
from dataclasses import dataclass
import re

import aiohttp
from aiohttp import ClientTimeout, ClientError
from bs4 import BeautifulSoup

from app.config import config
from app.utils.text_cleaner import TextCleaner
from app.utils.lang_detect import detect_language

logger = logging.getLogger(__name__)

# Try to import Playwright for JavaScript-heavy pages
try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available - JavaScript-heavy pages may not be scraped properly")


@dataclass
class DocumentData:
    """Scraped document data structure."""
    url: str
    title: str
    cleaned_text: str
    raw_html: str
    language: str
    publish_date: Optional[str]
    source_trust_score: float
    license: Optional[str]
    author: Optional[str]
    description: Optional[str]
    keywords: List[str]
    extraction_method: str
    scraping_time: float
    error: Optional[str] = None


class RobotsChecker:
    """Robots.txt compliance checker with caching."""
    
    def __init__(self):
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.cache_expiry: Dict[str, float] = {}
        self.cache_duration = 3600  # 1 hour
    
    async def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL can be fetched according to robots.txt."""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            # Check cache
            current_time = time.time()
            if (robots_url in self.robots_cache and 
                robots_url in self.cache_expiry and 
                current_time < self.cache_expiry[robots_url]):
                
                rp = self.robots_cache[robots_url]
                return rp.can_fetch(user_agent, url)
            
            # Fetch and parse robots.txt
            rp = RobotFileParser()
            rp.set_url(robots_url)
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(robots_url, timeout=ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            robots_content = await response.text()
                            rp.set_url(robots_url)
                            # Parse robots content manually since RobotFileParser doesn't support async
                            for line in robots_content.split('\n'):
                                rp.read()  # This doesn't work async, so we'll do basic parsing
                        else:
                            # If robots.txt not found, assume allowed
                            return True
            except Exception:
                # If can't fetch robots.txt, assume allowed
                return True
            
            # Cache the result
            self.robots_cache[robots_url] = rp
            self.cache_expiry[robots_url] = current_time + self.cache_duration
            
            return rp.can_fetch(user_agent, url)
            
        except Exception as e:
            logger.warning(f"Robots.txt check failed for {url}: {e}")
            return True  # Default to allowing if check fails


class ScrapingWorker:
    """
    Asynchronous web scraping worker with ethical constraints.
    """
    
    def __init__(
        self,
        max_concurrent: int = None,
        rate_limit: float = None,
        request_timeout: int = None,
        user_agent: str = None
    ):
        self.max_concurrent = max_concurrent or config.MAX_CONCURRENT_SCRAPES
        self.rate_limit = rate_limit or config.SCRAPE_RATE_LIMIT
        self.request_timeout = request_timeout or config.REQUEST_TIMEOUT
        
        self.user_agent = user_agent or (
            "DeepResearchAssistant/1.0 (Educational Research Bot; "
            "Contact: research@example.com)"
        )
        
        # Initialize components
        self.robots_checker = RobotsChecker()
        self.text_cleaner = TextCleaner()
        
        # Rate limiting
        self.last_request_times: Dict[str, float] = {}
        
        # Playwright browser (lazy initialization)
        self.playwright_browser: Optional[Browser] = None
        
        logger.info(f"ScrapingWorker initialized: {self.max_concurrent} concurrent, {self.rate_limit}s rate limit")
    
    async def scrape_urls(self, urls: List[str]) -> List[DocumentData]:
        """
        Scrape multiple URLs with rate limiting and robots.txt compliance.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of DocumentData objects
        """
        if not urls:
            return []
        
        # Filter and validate URLs
        valid_urls = self._filter_valid_urls(urls)
        
        if not valid_urls:
            logger.warning("No valid URLs to scrape")
            return []
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create scraping tasks
        tasks = [
            self._scrape_single_url(url, semaphore)
            for url in valid_urls
        ]
        
        # Execute tasks and collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        scraped_documents = []
        for result in results:
            if isinstance(result, DocumentData):
                scraped_documents.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Scraping task failed: {result}")
        
        logger.info(f"Successfully scraped {len(scraped_documents)} out of {len(urls)} URLs")
        return scraped_documents
    
    async def _scrape_single_url(self, url: str, semaphore: asyncio.Semaphore) -> Optional[DocumentData]:
        """Scrape a single URL with all safety checks."""
        async with semaphore:
            try:
                start_time = time.time()
                
                # Check robots.txt
                if not await self.robots_checker.can_fetch(url, self.user_agent):
                    logger.info(f"Robots.txt disallows scraping: {url}")
                    return None
                
                # Apply rate limiting
                await self._apply_rate_limit(url)
                
                # Try aiohttp first
                document_data = await self._scrape_with_aiohttp(url)
                
                # If aiohttp fails or returns poor content, try Playwright
                if (not document_data or 
                    not document_data.cleaned_text or 
                    len(document_data.cleaned_text) < 200):
                    
                    if PLAYWRIGHT_AVAILABLE:
                        logger.info(f"Retrying with Playwright: {url}")
                        playwright_result = await self._scrape_with_playwright(url)
                        if playwright_result and len(playwright_result.cleaned_text) > len(document_data.cleaned_text if document_data else ""):
                            document_data = playwright_result
                
                if document_data:
                    document_data.scraping_time = time.time() - start_time
                    
                    # Language detection and filtering
                    if document_data.cleaned_text:
                        lang_result = detect_language(document_data.cleaned_text)
                        document_data.language = lang_result["language"]
                        
                        if not lang_result["is_english"]:
                            logger.info(f"Filtered non-English content: {url} (detected: {lang_result['language']})")
                            return None
                    
                    logger.info(f"Successfully scraped: {url} ({len(document_data.cleaned_text)} chars)")
                    return document_data
                else:
                    logger.warning(f"Failed to extract content from: {url}")
                    return None
                
            except Exception as e:
                logger.error(f"Scraping failed for {url}: {e}")
                return DocumentData(
                    url=url,
                    title="",
                    cleaned_text="",
                    raw_html="",
                    language="unknown",
                    publish_date=None,
                    source_trust_score=0.0,
                    license=None,
                    author=None,
                    description=None,
                    keywords=[],
                    extraction_method="failed",
                    scraping_time=0.0,
                    error=str(e)
                )
    
    async def _scrape_with_aiohttp(self, url: str) -> Optional[DocumentData]:
        """Scrape URL using aiohttp."""
        try:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            timeout = ClientTimeout(total=self.request_timeout)
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' not in content_type:
                        logger.warning(f"Non-HTML content type for {url}: {content_type}")
                        return None
                    
                    # Get content
                    html_content = await response.text()
                    
                    if not html_content or len(html_content) < 100:
                        logger.warning(f"Empty or too short content for {url}")
                        return None
                    
                    # Extract content using text cleaner
                    extracted_data = self.text_cleaner.clean_html_content(html_content, url)
                    
                    # Calculate trust score based on domain and content quality
                    trust_score = self._calculate_trust_score(url, extracted_data)
                    
                    return DocumentData(
                        url=url,
                        title=extracted_data.get("title", ""),
                        cleaned_text=extracted_data.get("cleaned_text", ""),
                        raw_html=html_content,
                        language=extracted_data.get("language", "en"),
                        publish_date=extracted_data.get("publish_date"),
                        source_trust_score=trust_score,
                        license=extracted_data.get("license"),
                        author=extracted_data.get("author", ""),
                        description=extracted_data.get("description", ""),
                        keywords=extracted_data.get("keywords", []),
                        extraction_method="aiohttp_" + extracted_data.get("extraction_method", "unknown"),
                        scraping_time=0.0  # Will be set by caller
                    )
                    
        except ClientError as e:
            logger.error(f"aiohttp client error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"aiohttp scraping failed for {url}: {e}")
            return None
    
    async def _scrape_with_playwright(self, url: str) -> Optional[DocumentData]:
        """Scrape URL using Playwright for JavaScript-heavy pages."""
        if not PLAYWRIGHT_AVAILABLE:
            return None
        
        try:
            # Initialize browser if needed
            if not self.playwright_browser:
                playwright = await async_playwright().start()
                self.playwright_browser = await playwright.chromium.launch(headless=True)
            
            # Create new page
            page = await self.playwright_browser.new_page()
            
            # Set user agent
            await page.set_extra_http_headers({
                'User-Agent': self.user_agent
            })
            
            # Navigate to page
            await page.goto(url, wait_until='networkidle', timeout=self.request_timeout * 1000)
            
            # Wait for content to load
            await page.wait_for_timeout(2000)  # 2 second wait
            
            # Get page content
            html_content = await page.content()
            
            # Close page
            await page.close()
            
            if not html_content or len(html_content) < 100:
                return None
            
            # Extract content using text cleaner
            extracted_data = self.text_cleaner.clean_html_content(html_content, url)
            
            # Calculate trust score
            trust_score = self._calculate_trust_score(url, extracted_data)
            
            return DocumentData(
                url=url,
                title=extracted_data.get("title", ""),
                cleaned_text=extracted_data.get("cleaned_text", ""),
                raw_html=html_content,
                language=extracted_data.get("language", "en"),
                publish_date=extracted_data.get("publish_date"),
                source_trust_score=trust_score,
                license=extracted_data.get("license"),
                author=extracted_data.get("author", ""),
                description=extracted_data.get("description", ""),
                keywords=extracted_data.get("keywords", []),
                extraction_method="playwright_" + extracted_data.get("extraction_method", "unknown"),
                scraping_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Playwright scraping failed for {url}: {e}")
            return None
    
    def _filter_valid_urls(self, urls: List[str]) -> List[str]:
        """Filter and validate URLs."""
        valid_urls = []
        
        for url in urls:
            if not url or not isinstance(url, str):
                continue
            
            url = url.strip()
            
            # Basic URL validation
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Parse URL
            try:
                parsed = urlparse(url)
                if not parsed.netloc:
                    continue
                
                # Skip certain file types
                if parsed.path.lower().endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')):
                    logger.info(f"Skipping document file: {url}")
                    continue
                
                # Skip certain domains (can be configured)
                blocked_domains = ['localhost', '127.0.0.1', '0.0.0.0']
                if parsed.netloc.lower() in blocked_domains:
                    continue
                
                valid_urls.append(url)
                
            except Exception as e:
                logger.warning(f"Invalid URL {url}: {e}")
                continue
        
        return valid_urls
    
    async def _apply_rate_limit(self, url: str) -> None:
        """Apply rate limiting per domain."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        current_time = time.time()
        
        if domain in self.last_request_times:
            time_since_last = current_time - self.last_request_times[domain]
            if time_since_last < self.rate_limit:
                sleep_time = self.rate_limit - time_since_last
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s for {domain}")
                await asyncio.sleep(sleep_time)
        
        self.last_request_times[domain] = time.time()
    
    def _calculate_trust_score(self, url: str, extracted_data: Dict[str, Any]) -> float:
        """Calculate trust score based on URL and content characteristics."""
        score = 0.5  # Base score
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Domain-based scoring
        trusted_domains = {
            'wikipedia.org': 0.9,
            'gov': 0.95,
            'edu': 0.85,
            'reuters.com': 0.9,
            'bbc.com': 0.85,
            'nature.com': 0.9,
            'science.org': 0.9,
            'who.int': 0.9,
            'cdc.gov': 0.9
        }
        
        for trusted_domain, domain_score in trusted_domains.items():
            if trusted_domain in domain:
                score = max(score, domain_score)
                break
        
        # Content quality indicators
        content_length = len(extracted_data.get("cleaned_text", ""))
        if content_length > 1000:
            score += 0.1
        elif content_length > 500:
            score += 0.05
        
        # Metadata completeness
        metadata_fields = ["title", "author", "publish_date", "description"]
        filled_fields = sum(1 for field in metadata_fields if extracted_data.get(field))
        score += (filled_fields / len(metadata_fields)) * 0.1
        
        # Schema.org data presence
        if extracted_data.get("schema_data"):
            score += 0.1
        
        # HTTPS bonus
        if parsed_url.scheme == 'https':
            score += 0.05
        
        return min(1.0, score)
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.playwright_browser:
            await self.playwright_browser.close()
            self.playwright_browser = None
        
        logger.info("ScrapingWorker closed")


# Global scraping worker instance
scraping_worker: Optional[ScrapingWorker] = None


async def get_scraping_worker() -> ScrapingWorker:
    """Get or create global scraping worker."""
    global scraping_worker
    
    if scraping_worker is None:
        scraping_worker = ScrapingWorker()
    
    return scraping_worker


async def scrape_urls(urls: List[str]) -> List[DocumentData]:
    """Scrape URLs using global scraping worker."""
    worker = await get_scraping_worker()
    return await worker.scrape_urls(urls)


async def scrape_single_url(url: str) -> Optional[DocumentData]:
    """Scrape a single URL using global scraping worker."""
    worker = await get_scraping_worker()
    results = await worker.scrape_urls([url])
    return results[0] if results else None