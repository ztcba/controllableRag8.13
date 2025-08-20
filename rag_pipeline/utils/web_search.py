# rag_pipeline/utils/web_search.py

import requests
from typing import List, Dict, Optional
import time
import logging

logger = logging.getLogger(__name__)

class DuckDuckGoSearchTool:
    """
    A simple web search tool using DuckDuckGo's instant answer API.
    Provides structured search results without requiring API keys.
    """
    
    def __init__(self, max_results: int = 5, timeout: int = 10):
        """
        Initialize the search tool.
        
        Args:
            max_results: Maximum number of search results to return
            timeout: Request timeout in seconds
        """
        self.max_results = max_results
        self.timeout = timeout
        self.base_url = "https://api.duckduckgo.com/"
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Perform a web search using DuckDuckGo.
        
        Args:
            query: The search query string
            
        Returns:
            List of dictionaries containing search results with keys:
            - title: The title of the search result
            - snippet: Brief description/snippet
            - url: The URL of the result
        """
        try:
            # DuckDuckGo instant answer API parameters
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            logger.info(f"Searching DuckDuckGo for: {query}")
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract instant answer if available
            if data.get('Abstract'):
                results.append({
                    'title': data.get('AbstractText', 'DuckDuckGo Instant Answer'),
                    'snippet': data.get('Abstract', ''),
                    'url': data.get('AbstractURL', '')
                })
            
            # Extract related topics
            for topic in data.get('RelatedTopics', [])[:self.max_results-1]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else 'Related Topic',
                        'snippet': topic.get('Text', ''),
                        'url': topic.get('FirstURL', '')
                    })
            
            # If no results from above, try to use definition
            if not results and data.get('Definition'):
                results.append({
                    'title': 'Definition',
                    'snippet': data.get('Definition', ''),
                    'url': data.get('DefinitionURL', '')
                })
            
            logger.info(f"Found {len(results)} search results")
            return results[:self.max_results]
            
        except requests.RequestException as e:
            logger.error(f"Search request failed: {str(e)}")
            return self._create_error_result(f"网络搜索失败：{str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during search: {str(e)}")
            return self._create_error_result(f"搜索过程中出现错误：{str(e)}")
    
    def _create_error_result(self, error_msg: str) -> List[Dict[str, str]]:
        """Create a structured error result."""
        return [{
            'title': '搜索失败',
            'snippet': error_msg,
            'url': ''
        }]


def create_web_search_tool() -> DuckDuckGoSearchTool:
    """
    Factory function to create a web search tool instance.
    This follows the same pattern as other factory functions in the codebase.
    """
    return DuckDuckGoSearchTool(max_results=5, timeout=10)
