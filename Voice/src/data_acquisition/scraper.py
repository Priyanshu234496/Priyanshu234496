"""
Web scraper for earnings call transcripts and audio files
"""

import os
import re
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import yfinance as yf

from config.config import (
    SEEKING_ALPHA_BASE_URL, 
    YAHOO_FINANCE_BASE_URL,
    DEFAULT_COMPANIES,
    RAW_DATA_DIR
)

logger = logging.getLogger(__name__)

class EarningsCallScraper:
    """
    Scraper for earnings call transcripts and audio files from various sources
    """
    
    def __init__(self, headless: bool = True, download_dir: Optional[Path] = None):
        """
        Initialize the scraper
        
        Args:
            headless: Whether to run browser in headless mode
            download_dir: Directory to save downloaded files
        """
        self.download_dir = download_dir or RAW_DATA_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize web driver
        self.driver = self._setup_driver(headless)
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _setup_driver(self, headless: bool) -> webdriver.Chrome:
        """Setup Chrome web driver"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Set download preferences
        prefs = {
            "download.default_directory": str(self.download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        return webdriver.Chrome(options=chrome_options)
    
    def scrape_company_earnings_calls(self, 
                                    company_symbol: str, 
                                    max_calls: int = 10,
                                    start_date: Optional[datetime] = None) -> List[Dict]:
        """
        Scrape earnings calls for a specific company
        
        Args:
            company_symbol: Stock symbol (e.g., 'AAPL')
            max_calls: Maximum number of calls to scrape
            start_date: Start date for scraping (default: 2 years ago)
            
        Returns:
            List of earnings call data dictionaries
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)  # 2 years ago
        
        logger.info(f"Scraping earnings calls for {company_symbol} from {start_date}")
        
        calls_data = []
        
        try:
            # Get company info from Yahoo Finance
            company_info = self._get_company_info(company_symbol)
            if not company_info:
                logger.warning(f"Could not get company info for {company_symbol}")
                return calls_data
            
            # Scrape from Seeking Alpha
            seeking_alpha_calls = self._scrape_seeking_alpha(company_symbol, max_calls, start_date)
            calls_data.extend(seeking_alpha_calls)
            
            # Scrape from company's investor relations page
            ir_calls = self._scrape_investor_relations(company_info, max_calls, start_date)
            calls_data.extend(ir_calls)
            
            # Download audio files for calls with transcripts
            self._download_audio_files(calls_data)
            
            # Save metadata
            self._save_metadata(company_symbol, calls_data)
            
        except Exception as e:
            logger.error(f"Error scraping earnings calls for {company_symbol}: {e}")
        
        logger.info(f"Scraped {len(calls_data)} earnings calls for {company_symbol}")
        return calls_data
    
    def _get_company_info(self, symbol: str) -> Optional[Dict]:
        """Get company information from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'website': info.get('website', ''),
                'industry': info.get('industry', ''),
                'sector': info.get('sector', '')
            }
        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {e}")
            return None
    
    def _scrape_seeking_alpha(self, 
                            symbol: str, 
                            max_calls: int, 
                            start_date: datetime) -> List[Dict]:
        """Scrape earnings calls from Seeking Alpha"""
        calls = []
        
        try:
            # Search for earnings call transcripts
            search_url = f"{SEEKING_ALPHA_BASE_URL}/search?q={symbol}+earnings+call+transcript"
            
            self.driver.get(search_url)
            time.sleep(3)
            
            # Wait for search results
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "search-result"))
            )
            
            # Get search results
            results = self.driver.find_elements(By.CLASS_NAME, "search-result")
            
            for result in results[:max_calls]:
                try:
                    # Extract call data
                    title_elem = result.find_element(By.CSS_SELECTOR, "h3 a")
                    title = title_elem.text
                    url = title_elem.get_attribute("href")
                    
                    # Check if it's an earnings call transcript
                    if "earnings call" in title.lower() and "transcript" in title.lower():
                        call_data = self._extract_seeking_alpha_call(url, symbol)
                        if call_data:
                            calls.append(call_data)
                            
                except Exception as e:
                    logger.warning(f"Error processing Seeking Alpha result: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping Seeking Alpha for {symbol}: {e}")
        
        return calls
    
    def _extract_seeking_alpha_call(self, url: str, symbol: str) -> Optional[Dict]:
        """Extract earnings call data from Seeking Alpha article"""
        try:
            self.driver.get(url)
            time.sleep(2)
            
            # Extract transcript text
            transcript_elem = self.driver.find_element(By.CLASS_NAME, "article-content")
            transcript_text = transcript_elem.text
            
            # Extract date
            date_elem = self.driver.find_element(By.CLASS_NAME, "article-date")
            date_text = date_elem.text
            call_date = self._parse_date(date_text)
            
            # Extract speakers and their segments
            speakers = self._extract_speakers(transcript_text)
            
            # Look for audio link
            audio_url = self._find_audio_link()
            
            return {
                'symbol': symbol,
                'date': call_date,
                'source': 'seeking_alpha',
                'url': url,
                'transcript': transcript_text,
                'speakers': speakers,
                'audio_url': audio_url,
                'title': self.driver.title
            }
            
        except Exception as e:
            logger.warning(f"Error extracting call from {url}: {e}")
            return None
    
    def _scrape_investor_relations(self, 
                                 company_info: Dict, 
                                 max_calls: int, 
                                 start_date: datetime) -> List[Dict]:
        """Scrape from company's investor relations page"""
        calls = []
        
        try:
            website = company_info.get('website', '')
            if not website:
                return calls
            
            # Try to find investor relations page
            ir_urls = [
                f"{website}/investors",
                f"{website}/investor-relations",
                f"{website}/ir",
                f"{website}/investor"
            ]
            
            for ir_url in ir_urls:
                try:
                    response = self.session.get(ir_url, timeout=10)
                    if response.status_code == 200:
                        calls.extend(self._scrape_ir_page(ir_url, company_info, max_calls, start_date))
                        break
                except Exception as e:
                    logger.debug(f"Could not access {ir_url}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping investor relations for {company_info['symbol']}: {e}")
        
        return calls
    
    def _scrape_ir_page(self, 
                       ir_url: str, 
                       company_info: Dict, 
                       max_calls: int, 
                       start_date: datetime) -> List[Dict]:
        """Scrape earnings calls from investor relations page"""
        calls = []
        
        try:
            self.driver.get(ir_url)
            time.sleep(3)
            
            # Look for earnings call links
            earnings_links = self.driver.find_elements(
                By.XPATH, 
                "//a[contains(text(), 'earnings') or contains(text(), 'call') or contains(text(), 'transcript')]"
            )
            
            for link in earnings_links[:max_calls]:
                try:
                    href = link.get_attribute("href")
                    if href and "earnings" in href.lower():
                        call_data = self._extract_ir_call(href, company_info)
                        if call_data:
                            calls.append(call_data)
                except Exception as e:
                    logger.debug(f"Error processing IR link: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping IR page {ir_url}: {e}")
        
        return calls
    
    def _extract_ir_call(self, url: str, company_info: Dict) -> Optional[Dict]:
        """Extract earnings call data from IR page"""
        try:
            self.driver.get(url)
            time.sleep(2)
            
            # Extract transcript (this will vary by company)
            transcript_text = self._extract_transcript_from_page()
            
            if not transcript_text:
                return None
            
            # Extract date
            call_date = self._extract_date_from_page()
            
            # Extract speakers
            speakers = self._extract_speakers(transcript_text)
            
            # Look for audio
            audio_url = self._find_audio_link()
            
            return {
                'symbol': company_info['symbol'],
                'date': call_date,
                'source': 'investor_relations',
                'url': url,
                'transcript': transcript_text,
                'speakers': speakers,
                'audio_url': audio_url,
                'title': self.driver.title
            }
            
        except Exception as e:
            logger.warning(f"Error extracting IR call from {url}: {e}")
            return None
    
    def _extract_transcript_from_page(self) -> str:
        """Extract transcript text from current page"""
        try:
            # Try different selectors for transcript content
            selectors = [
                ".transcript", ".earnings-transcript", ".call-transcript",
                ".content", ".article-content", ".main-content",
                "div[class*='transcript']", "div[class*='earnings']"
            ]
            
            for selector in selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    return element.text
                except:
                    continue
            
            # Fallback: get all text content
            return self.driver.find_element(By.TAG_NAME, "body").text
            
        except Exception as e:
            logger.warning(f"Error extracting transcript: {e}")
            return ""
    
    def _extract_date_from_page(self) -> Optional[datetime]:
        """Extract call date from current page"""
        try:
            # Try different date patterns
            date_patterns = [
                r'(\d{1,2}/\d{1,2}/\d{4})',
                r'(\d{4}-\d{2}-\d{2})',
                r'(\w+ \d{1,2}, \d{4})',
                r'(\d{1,2} \w+ \d{4})'
            ]
            
            page_text = self.driver.page_source
            
            for pattern in date_patterns:
                match = re.search(pattern, page_text)
                if match:
                    return self._parse_date(match.group(1))
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting date: {e}")
            return None
    
    def _extract_speakers(self, transcript_text: str) -> List[Dict]:
        """Extract speakers and their segments from transcript"""
        speakers = []
        
        try:
            # Common speaker patterns
            speaker_patterns = [
                r'([A-Z][a-z]+ [A-Z][a-z]+):',
                r'([A-Z][A-Z]+):',
                r'([A-Z][a-z]+):',
                r'([A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+):'
            ]
            
            lines = transcript_text.split('\n')
            current_speaker = None
            current_text = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line contains speaker
                speaker_found = None
                for pattern in speaker_patterns:
                    match = re.match(pattern, line)
                    if match:
                        speaker_found = match.group(1)
                        break
                
                if speaker_found:
                    # Save previous speaker's text
                    if current_speaker and current_text:
                        speakers.append({
                            'name': current_speaker,
                            'text': ' '.join(current_text)
                        })
                    
                    # Start new speaker
                    current_speaker = speaker_found
                    current_text = [line.split(':', 1)[1].strip() if ':' in line else '']
                else:
                    # Continue current speaker's text
                    if current_speaker:
                        current_text.append(line)
            
            # Add last speaker
            if current_speaker and current_text:
                speakers.append({
                    'name': current_speaker,
                    'text': ' '.join(current_text)
                })
                
        except Exception as e:
            logger.warning(f"Error extracting speakers: {e}")
        
        return speakers
    
    def _find_audio_link(self) -> Optional[str]:
        """Find audio file link on current page"""
        try:
            # Look for audio links
            audio_selectors = [
                "a[href*='.mp3']", "a[href*='.wav']", "a[href*='.m4a']",
                "a[href*='audio']", "a[href*='recording']"
            ]
            
            for selector in audio_selectors:
                try:
                    links = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for link in links:
                        href = link.get_attribute("href")
                        if href and any(ext in href.lower() for ext in ['.mp3', '.wav', '.m4a']):
                            return href
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Error finding audio link: {e}")
            return None
    
    def _download_audio_files(self, calls_data: List[Dict]):
        """Download audio files for calls that have audio URLs"""
        for call in calls_data:
            audio_url = call.get('audio_url')
            if not audio_url:
                continue
            
            try:
                # Create filename
                symbol = call['symbol']
                date_str = call['date'].strftime('%Y%m%d') if call['date'] else 'unknown'
                filename = f"{symbol}_{date_str}_earnings_call"
                
                # Download audio file
                response = self.session.get(audio_url, stream=True)
                if response.status_code == 200:
                    # Determine file extension
                    content_type = response.headers.get('content-type', '')
                    if 'mp3' in content_type:
                        ext = '.mp3'
                    elif 'wav' in content_type:
                        ext = '.wav'
                    elif 'm4a' in content_type:
                        ext = '.m4a'
                    else:
                        ext = '.mp3'  # Default
                    
                    filepath = self.download_dir / f"{filename}{ext}"
                    
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    call['audio_file'] = str(filepath)
                    logger.info(f"Downloaded audio: {filepath}")
                    
            except Exception as e:
                logger.warning(f"Error downloading audio for {call['symbol']}: {e}")
    
    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse date from various formats"""
        try:
            # Common date formats
            formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%B %d, %Y',
                '%d %B %Y',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_text.strip(), fmt)
                except ValueError:
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Error parsing date '{date_text}': {e}")
            return None
    
    def _save_metadata(self, symbol: str, calls_data: List[Dict]):
        """Save metadata for scraped calls"""
        try:
            metadata_file = self.download_dir / f"{symbol}_metadata.json"
            
            # Convert datetime objects to strings for JSON serialization
            serializable_data = []
            for call in calls_data:
                call_copy = call.copy()
                if call_copy.get('date'):
                    call_copy['date'] = call_copy['date'].isoformat()
                serializable_data.append(call_copy)
            
            import json
            with open(metadata_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
            logger.info(f"Saved metadata: {metadata_file}")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def scrape_multiple_companies(self, 
                                companies: Optional[List[str]] = None,
                                max_calls_per_company: int = 10) -> Dict[str, List[Dict]]:
        """
        Scrape earnings calls for multiple companies
        
        Args:
            companies: List of company symbols (default: DEFAULT_COMPANIES)
            max_calls_per_company: Maximum calls per company
            
        Returns:
            Dictionary mapping company symbols to their earnings call data
        """
        companies = companies or DEFAULT_COMPANIES
        all_calls = {}
        
        for company in companies:
            logger.info(f"Scraping {company}...")
            calls = self.scrape_company_earnings_calls(company, max_calls_per_company)
            all_calls[company] = calls
            
            # Be respectful with delays
            time.sleep(5)
        
        return all_calls
    
    def close(self):
        """Close the web driver"""
        if self.driver:
            self.driver.quit()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
