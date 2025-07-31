# 고급 웹 검색 시스템
# 실시간 검색, 다중 소스, 결과 요약 및 분석

import asyncio
import aiohttp
import json
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import quote, urljoin
import hashlib
from bs4 import BeautifulSoup
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    title: str
    url: str
    snippet: str
    source: str
    timestamp: str
    relevance_score: float = 0.0

class AdvancedWebSearcher:
    """고급 웹 검색 엔진"""
    
    def __init__(self):
        self.search_engines = {
            'google': self._search_google,
            'bing': self._search_bing,
            'duckduckgo': self._search_duckduckgo,
            'wikipedia': self._search_wikipedia,
            'reddit': self._search_reddit,
            'github': self._search_github,
            'stackoverflow': self._search_stackoverflow
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        self.session = None
        self.cache = {}
        self.cache_ttl = 3600  # 1시간 캐시
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=self.headers,
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, query: str, engine: str) -> str:
        """캐시 키 생성"""
        return hashlib.md5(f"{query}_{engine}".encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """캐시 유효성 검사"""
        return time.time() - cache_entry['timestamp'] < self.cache_ttl
    
    async def _search_google(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Google 검색 (스크래핑 방식)"""
        try:
            search_url = f"https://www.google.com/search?q={quote(query)}&num={max_results}&hl=ko"
            
            async with self.session.get(search_url) as response:
                if response.status != 200:
                    logger.warning(f"Google 검색 실패: {response.status}")
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                results = []
                search_results = soup.find_all('div', class_='g')
                
                for result in search_results[:max_results]:
                    try:
                        title_elem = result.find('h3')
                        link_elem = result.find('a')
                        snippet_elem = result.find('span', class_=['aCOpRe', 'st'])
                        
                        if title_elem and link_elem:
                            title = title_elem.get_text(strip=True)
                            url = link_elem.get('href', '')
                            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                            
                            if url.startswith('/url?q='):
                                url = url.split('/url?q=')[1].split('&')[0]
                            
                            results.append(SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source="Google",
                                timestamp=datetime.now().isoformat()
                            ))
                    except Exception as e:
                        logger.debug(f"Google 결과 파싱 오류: {e}")
                        continue
                
                logger.info(f"Google에서 {len(results)}개 결과 수집")
                return results
                
        except Exception as e:
            logger.error(f"Google 검색 오류: {e}")
            return []
    
    async def _search_bing(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Bing 검색"""
        try:
            search_url = f"https://www.bing.com/search?q={quote(query)}&count={max_results}"
            
            async with self.session.get(search_url) as response:
                if response.status != 200:
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                results = []
                search_results = soup.find_all('li', class_='b_algo')
                
                for result in search_results[:max_results]:
                    try:
                        title_elem = result.find('h2')
                        link_elem = title_elem.find('a') if title_elem else None
                        snippet_elem = result.find('p')
                        
                        if title_elem and link_elem:
                            title = title_elem.get_text(strip=True)
                            url = link_elem.get('href', '')
                            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                            
                            results.append(SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source="Bing",
                                timestamp=datetime.now().isoformat()
                            ))
                    except Exception as e:
                        continue
                
                logger.info(f"Bing에서 {len(results)}개 결과 수집")
                return results
                
        except Exception as e:
            logger.error(f"Bing 검색 오류: {e}")
            return []
    
    async def _search_duckduckgo(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """DuckDuckGo 검색"""
        try:
            # DuckDuckGo Instant Answer API 사용
            search_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
            
            async with self.session.get(search_url) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                results = []
                
                # Abstract 결과
                if data.get('Abstract'):
                    results.append(SearchResult(
                        title=data.get('Heading', query),
                        url=data.get('AbstractURL', ''),
                        snippet=data.get('Abstract', ''),
                        source="DuckDuckGo",
                        timestamp=datetime.now().isoformat()
                    ))
                
                # Related Topics
                for topic in data.get('RelatedTopics', [])[:max_results-1]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append(SearchResult(
                            title=topic.get('Text', '')[:100] + "...",
                            url=topic.get('FirstURL', ''),
                            snippet=topic.get('Text', ''),
                            source="DuckDuckGo",
                            timestamp=datetime.now().isoformat()
                        ))
                
                logger.info(f"DuckDuckGo에서 {len(results)}개 결과 수집")
                return results
                
        except Exception as e:
            logger.error(f"DuckDuckGo 검색 오류: {e}")
            return []
    
    async def _search_wikipedia(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Wikipedia 검색"""
        try:
            # Wikipedia API 사용
            search_url = f"https://ko.wikipedia.org/api/rest_v1/page/summary/{quote(query)}"
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    result = SearchResult(
                        title=data.get('title', ''),
                        url=data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                        snippet=data.get('extract', ''),
                        source="Wikipedia",
                        timestamp=datetime.now().isoformat()
                    )
                    
                    logger.info("Wikipedia에서 1개 결과 수집")
                    return [result]
                
                # 검색 API 사용
                search_url = f"https://ko.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(query)}&format=json&srlimit={max_results}"
                
                async with self.session.get(search_url) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    results = []
                    
                    for item in data.get('query', {}).get('search', []):
                        results.append(SearchResult(
                            title=item.get('title', ''),
                            url=f"https://ko.wikipedia.org/wiki/{quote(item.get('title', ''))}",
                            snippet=re.sub(r'<[^>]+>', '', item.get('snippet', '')),
                            source="Wikipedia",
                            timestamp=datetime.now().isoformat()
                        ))
                    
                    logger.info(f"Wikipedia에서 {len(results)}개 결과 수집")
                    return results
                
        except Exception as e:
            logger.error(f"Wikipedia 검색 오류: {e}")
            return []
    
    async def _search_reddit(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Reddit 검색"""
        try:
            search_url = f"https://www.reddit.com/search.json?q={quote(query)}&limit={max_results}&sort=relevance"
            
            async with self.session.get(search_url) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                results = []
                
                for post in data.get('data', {}).get('children', []):
                    post_data = post.get('data', {})
                    
                    results.append(SearchResult(
                        title=post_data.get('title', ''),
                        url=f"https://reddit.com{post_data.get('permalink', '')}",
                        snippet=post_data.get('selftext', '')[:200] + "..." if post_data.get('selftext') else "",
                        source=f"Reddit (r/{post_data.get('subreddit', '')})",
                        timestamp=datetime.now().isoformat()
                    ))
                
                logger.info(f"Reddit에서 {len(results)}개 결과 수집")
                return results
                
        except Exception as e:
            logger.error(f"Reddit 검색 오류: {e}")
            return []
    
    async def _search_github(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """GitHub 검색"""
        try:
            search_url = f"https://api.github.com/search/repositories?q={quote(query)}&sort=stars&order=desc&per_page={max_results}"
            
            async with self.session.get(search_url) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                results = []
                
                for repo in data.get('items', []):
                    results.append(SearchResult(
                        title=repo.get('full_name', ''),
                        url=repo.get('html_url', ''),
                        snippet=repo.get('description', '') or "설명 없음",
                        source=f"GitHub (⭐{repo.get('stargazers_count', 0)})",
                        timestamp=datetime.now().isoformat()
                    ))
                
                logger.info(f"GitHub에서 {len(results)}개 결과 수집")
                return results
                
        except Exception as e:
            logger.error(f"GitHub 검색 오류: {e}")
            return []
    
    async def _search_stackoverflow(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Stack Overflow 검색"""
        try:
            search_url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q={quote(query)}&site=stackoverflow&pagesize={max_results}"
            
            async with self.session.get(search_url) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                results = []
                
                for question in data.get('items', []):
                    results.append(SearchResult(
                        title=question.get('title', ''),
                        url=question.get('link', ''),
                        snippet=f"조회수: {question.get('view_count', 0)}, 답변: {question.get('answer_count', 0)}",
                        source="Stack Overflow",
                        timestamp=datetime.now().isoformat()
                    ))
                
                logger.info(f"Stack Overflow에서 {len(results)}개 결과 수집")
                return results
                
        except Exception as e:
            logger.error(f"Stack Overflow 검색 오류: {e}")
            return []
    
    async def search_multiple_engines(self, query: str, engines: List[str] = None, max_results_per_engine: int = 5) -> Dict[str, List[SearchResult]]:
        """다중 검색 엔진에서 검색"""
        if engines is None:
            engines = ['google', 'wikipedia', 'github']
        
        results = {}
        tasks = []
        
        for engine in engines:
            if engine in self.search_engines:
                # 캐시 확인
                cache_key = self._get_cache_key(query, engine)
                if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
                    results[engine] = self.cache[cache_key]['results']
                    continue
                
                # 비동기 검색 작업 추가
                task = asyncio.create_task(self.search_engines[engine](query, max_results_per_engine))
                tasks.append((engine, task))
        
        # 모든 검색 작업 완료 대기
        for engine, task in tasks:
            try:
                search_results = await task
                results[engine] = search_results
                
                # 캐시에 저장
                cache_key = self._get_cache_key(query, engine)
                self.cache[cache_key] = {
                    'results': search_results,
                    'timestamp': time.time()
                }
                
            except Exception as e:
                logger.error(f"{engine} 검색 오류: {e}")
                results[engine] = []
        
        return results
    
    def _calculate_relevance_score(self, result: SearchResult, query: str) -> float:
        """검색 결과 관련성 점수 계산"""
        score = 0.0
        query_lower = query.lower()
        
        # 제목에서 쿼리 단어 매칭
        title_lower = result.title.lower()
        query_words = query_lower.split()
        
        for word in query_words:
            if word in title_lower:
                score += 2.0
        
        # 스니펫에서 쿼리 단어 매칭
        snippet_lower = result.snippet.lower()
        for word in query_words:
            if word in snippet_lower:
                score += 1.0
        
        # 소스별 가중치
        source_weights = {
            'Wikipedia': 1.5,
            'GitHub': 1.3,
            'Stack Overflow': 1.2,
            'Google': 1.0,
            'Bing': 0.9,
            'DuckDuckGo': 0.8,
            'Reddit': 0.7
        }
        
        for source, weight in source_weights.items():
            if source in result.source:
                score *= weight
                break
        
        return score
    
    def merge_and_rank_results(self, search_results: Dict[str, List[SearchResult]], query: str, max_results: int = 10) -> List[SearchResult]:
        """검색 결과 병합 및 순위 매기기"""
        all_results = []
        seen_urls = set()
        
        # 모든 결과 수집 및 중복 제거
        for engine, results in search_results.items():
            for result in results:
                if result.url and result.url not in seen_urls:
                    result.relevance_score = self._calculate_relevance_score(result, query)
                    all_results.append(result)
                    seen_urls.add(result.url)
        
        # 관련성 점수로 정렬
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return all_results[:max_results]
    
    async def comprehensive_search(self, query: str, engines: List[str] = None, max_results: int = 10) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """종합 검색 수행"""
        start_time = time.time()
        
        if engines is None:
            # 쿼리 유형에 따라 적절한 엔진 선택
            if any(keyword in query.lower() for keyword in ['github', 'code', '코드', '프로그래밍']):
                engines = ['google', 'github', 'stackoverflow']
            elif any(keyword in query.lower() for keyword in ['reddit', '커뮤니티', '토론']):
                engines = ['google', 'reddit', 'wikipedia']
            else:
                engines = ['google', 'wikipedia', 'bing']
        
        # 다중 엔진 검색
        search_results = await self.search_multiple_engines(query, engines, max_results_per_engine=5)
        
        # 결과 병합 및 순위 매기기
        final_results = self.merge_and_rank_results(search_results, query, max_results)
        
        # 검색 통계
        search_stats = {
            'query': query,
            'engines_used': list(search_results.keys()),
            'total_results_found': sum(len(results) for results in search_results.values()),
            'final_results_count': len(final_results),
            'search_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"종합 검색 완료: '{query}' - {len(final_results)}개 결과, {search_stats['search_time']:.2f}초")
        
        return final_results, search_stats

class SearchResultFormatter:
    """검색 결과 포맷터"""
    
    @staticmethod
    def format_for_discord(results: List[SearchResult], stats: Dict[str, Any], max_length: int = 2000) -> str:
        """디스코드용 검색 결과 포맷팅"""
        if not results:
            return "🔍 검색 결과를 찾을 수 없습니다."
        
        query = stats.get('query', '')
        search_time = stats.get('search_time', 0)
        total_found = stats.get('total_results_found', 0)
        engines = ', '.join(stats.get('engines_used', []))
        
        # 헤더
        header = f"🔍 **'{query}' 검색 결과**\n"
        header += f"📊 {total_found}개 결과 발견, {len(results)}개 선별 ({search_time:.1f}초)\n"
        header += f"🌐 검색 엔진: {engines}\n\n"
        
        # 결과 목록
        results_text = ""
        current_length = len(header)
        
        for i, result in enumerate(results, 1):
            # 제목 길이 제한
            title = result.title[:80] + "..." if len(result.title) > 80 else result.title
            
            # 스니펫 길이 제한
            snippet = result.snippet[:150] + "..." if len(result.snippet) > 150 else result.snippet
            
            result_text = f"{i}. **{title}**\n"
            result_text += f"   🔗 {result.url}\n"
            result_text += f"   📰 {result.source}"
            
            if result.relevance_score > 0:
                result_text += f" (관련도: {result.relevance_score:.1f})"
            
            result_text += f"\n   💬 {snippet}\n\n"
            
            # 길이 제한 확인
            if current_length + len(result_text) > max_length - 100:
                results_text += f"... 및 {len(results) - i + 1}개 추가 결과\n"
                break
            
            results_text += result_text
            current_length += len(result_text)
        
        # 푸터
        footer = f"⏰ 검색 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return header + results_text + footer
    
    @staticmethod
    def format_summary(results: List[SearchResult], max_results: int = 3) -> str:
        """검색 결과 요약"""
        if not results:
            return "검색 결과가 없습니다."
        
        summary = "📋 **검색 결과 요약:**\n\n"
        
        for i, result in enumerate(results[:max_results], 1):
            summary += f"{i}. **{result.title}** ({result.source})\n"
            summary += f"   {result.snippet[:100]}...\n\n"
        
        return summary

# 전역 검색 인스턴스
web_searcher = None

async def initialize_web_search():
    """웹 검색 시스템 초기화"""
    global web_searcher
    try:
        web_searcher = AdvancedWebSearcher()
        logger.info("고급 웹 검색 시스템 초기화 완료")
    except Exception as e:
        logger.error(f"웹 검색 시스템 초기화 오류: {e}")

async def search_web(query: str, engines: List[str] = None, max_results: int = 10) -> str:
    """웹 검색 수행 및 결과 반환"""
    if not web_searcher:
        return "🔧 웹 검색 시스템이 초기화되지 않았습니다."
    
    try:
        async with web_searcher:
            results, stats = await web_searcher.comprehensive_search(query, engines, max_results)
            return SearchResultFormatter.format_for_discord(results, stats)
    except Exception as e:
        logger.error(f"웹 검색 오류: {e}")
        return f"🚨 웹 검색 중 오류가 발생했습니다: {str(e)}"

async def search_web_summary(query: str, engines: List[str] = None) -> str:
    """웹 검색 요약 결과 반환"""
    if not web_searcher:
        return "🔧 웹 검색 시스템이 초기화되지 않았습니다."
    
    try:
        async with web_searcher:
            results, stats = await web_searcher.comprehensive_search(query, engines, max_results=5)
            return SearchResultFormatter.format_summary(results)
    except Exception as e:
        logger.error(f"웹 검색 요약 오류: {e}")
        return f"🚨 웹 검색 요약 중 오류가 발생했습니다: {str(e)}"