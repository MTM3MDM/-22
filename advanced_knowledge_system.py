# 고급 지식 시스템 - AI/GPT/기술 뉴스 전문 처리
# 벡터 검색, 실시간 데이터 수집, 지능형 응답 생성

import asyncio
import aiohttp
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import pickle
from pathlib import Path
from types import TracebackType
from typing import Coroutine

# 벡터 검색 및 임베딩
try:
    import faiss  # type: ignore
except ImportError:
    faiss = None
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


# 뉴스 및 데이터 수집
import feedparser  # type: ignore
import requests
from bs4 import BeautifulSoup
try:
    import newspaper  # type: ignore
    from newspaper import Article  # type: ignore
except ImportError:
    newspaper = None

# 텍스트 처리
import re
from collections import defaultdict, Counter
import tiktoken

logger = logging.getLogger(__name__)

class TechNewsCollector:
    """최신 기술 뉴스 수집기"""
    
    def __init__(self):
        self.news_sources = {
            'ai_news_2025': [
                'https://feeds.feedburner.com/oreilly/radar',
                'https://techcrunch.com/category/artificial-intelligence/feed/',
                'https://www.theverge.com/ai-artificial-intelligence/rss/index.xml',
                'https://venturebeat.com/ai/feed/',
                'https://www.wired.com/feed/tag/ai/latest/rss',
                'https://spectrum.ieee.org/rss/blog/artificial-intelligence',
                'https://www.technologyreview.com/feed/',
                'https://www.artificialintelligence-news.com/feed/',
                'https://machinelearningmastery.com/feed/',
                'https://towardsdatascience.com/feed',
                'https://distill.pub/rss.xml',
            ],
            'gpt_llm_2025': [
                'https://openai.com/blog/rss.xml',
                'https://blog.google/technology/ai/rss/',
                'https://blogs.microsoft.com/ai/feed/',
                'https://www.anthropic.com/news/rss.xml',
                'https://huggingface.co/blog/feed.xml',
                'https://deepmind.google/discover/blog/rss.xml',
                'https://stability.ai/news/rss',
                'https://cohere.com/blog/rss.xml',
            ],
            'tech_startup_2025': [
                'https://feeds.feedburner.com/TechCrunch',
                'https://www.engadget.com/rss.xml',
                'https://arstechnica.com/feed/',
                'https://www.zdnet.com/news/rss.xml',
                'https://www.cnet.com/rss/news/',
                'https://techcrunch.com/category/startups/feed/',
                'https://www.producthunt.com/feed',
                'https://news.ycombinator.com/rss',
                'https://www.crunchbase.com/feed',
            ],
            'dev_programming_2025': [
                'https://github.blog/feed/',
                'https://stackoverflow.blog/feed/',
                'https://dev.to/feed',
                'https://www.freecodecamp.org/news/rss/',
                'https://css-tricks.com/feed/',
                'https://www.smashingmagazine.com/feed/',
                'https://blog.jetbrains.com/feed/',
            ],
            'korean_tech_2025': [
                'https://www.etnews.com/rss/S08.xml',
                'https://www.bloter.net/feed',
                'https://www.itworld.co.kr/rss/news.xml',
                'https://zdnet.co.kr/rss/news.xml',
                'https://www.boannews.com/rss/news.xml',
                'https://www.ddaily.co.kr/rss/S1N15.xml',
            ],
            'cloud_blockchain_2025': [
                'https://aws.amazon.com/blogs/aws/feed/',
                'https://cloud.google.com/blog/rss',
                'https://azure.microsoft.com/en-us/blog/feed/',
                'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'https://cointelegraph.com/rss',
                'https://decrypt.co/feed',
            ]
        }
        
        self.session = None
        self.collected_articles: List[Dict[str, Any]] = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        return self
        
    async def __aexit__(self, 
                      exc_type: Optional[type[BaseException]], 
                      exc_val: Optional[BaseException], 
                      exc_tb: Optional[TracebackType]):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def collect_from_rss(self, url: str, category: str) -> List[Dict[str, Any]]:
        """RSS 피드에서 뉴스 수집"""
        if not self.session or self.session.closed:
            logger.warning("Aiohttp 세션이 활성화되지 않았습니다. RSS 수집을 건너뜁니다.")
            return []
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    articles: List[Dict[str, Any]] = []
                    for entry in feed.entries[:10]:  # 최신 10개만
                        article: Dict[str, Any] = {
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'link': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'category': category,
                            'source': feed.feed.get('title', 'Unknown') if hasattr(feed, 'feed') else 'Unknown',
                            'collected_at': datetime.now().isoformat()
                        }
                        articles.append(article)
                    
                    logger.info(f"RSS에서 {len(articles)}개 기사 수집: {url}")
                    return articles
            return [] # response.status가 200이 아닌 경우
                    
        except Exception as e:
            logger.error(f"RSS 수집 오류 {url}: {e}")
            return []
    
    async def collect_all_news(self) -> List[Dict[str, Any]]:
        """모든 소스에서 뉴스 수집"""
        all_articles: List[Dict[str, Any]] = []
        
        # 새로운 세션 생성
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        ) as session:
            self.session = session
            
            tasks: List[Coroutine[Any, Any, List[Dict[str, Any]]]] = []
            for category, urls in self.news_sources.items():
                for url in urls:
                    tasks.append(self.collect_from_rss(url, category))
            
            results = await asyncio.gather(*tasks)
            for result in results:
                if result:
                    all_articles.extend(result)

        # 중복 제거 (제목 기준)
        seen_titles: set[str] = set()
        unique_articles: List[Dict[str, Any]] = []
        for article in all_articles:
            if article and article.get('title'):
                try:
                    title_hash = hashlib.md5(article['title'].encode()).hexdigest()
                    if title_hash not in seen_titles:
                        seen_titles.add(title_hash)
                        unique_articles.append(article)
                except Exception as e:
                    logger.debug(f"기사 중복 제거 처리 오류: {e}")
                    unique_articles.append(article)
        
        logger.info(f"총 {len(unique_articles)}개의 고유 기사 수집 완료")
        return unique_articles
    
    async def enhance_article_content(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """기사 내용 상세 수집"""
        if not self.session or self.session.closed:
            logger.warning("Aiohttp 세션이 활성화되지 않았습니다. 기사 내용 수집을 건너뜁니다.")
            return article

        try:
            link = article.get('link')
            if not link:
                return article
                
            async with self.session.get(link) as response:
                if response.status == 200:
                    html_content = await response.text()
                    
                    # newspaper3k로 본문 추출
                    if newspaper and Article:
                        try:
                            news_article = Article(link)
                            news_article.set_html(html_content)
                            news_article.parse()
                            
                            article['full_text'] = news_article.text
                            article['authors'] = news_article.authors
                            article['keywords'] = news_article.keywords
                            
                        except Exception as e:
                            logger.warning(f"본문 추출 실패 {link}: {e}")
                            
        except Exception as e:
            logger.error(f"기사 내용 수집 오류 {link}: {e}")
            
        return article

class VectorKnowledgeBase:
    """벡터 기반 지식베이스"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder: Optional[SentenceTransformer] = None
        self.index: Optional["faiss.Index"] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension = 384  # all-MiniLM-L6-v2의 차원
        
        # ChromaDB 설정
        self.chroma_client: Optional["chromadb.Client"] = None
        self.collection: Optional["chromadb.Collection"] = None
        
        self.data_dir = Path("knowledge_base")
        self.data_dir.mkdir(exist_ok=True)
        
    async def initialize(self):
        """벡터 검색 시스템 초기화"""
        if not faiss or not chromadb:
            logger.error("FAISS 또는 ChromaDB가 설치되지 않아 벡터 검색 시스템을 초기화할 수 없습니다.")
            return

        try:
            # SentenceTransformer 모델 로드
            self.encoder = SentenceTransformer(self.model_name)
            logger.info(f"임베딩 모델 로드 완료: {self.model_name}")
            
            # FAISS 인덱스 초기화
            if faiss:
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (코사인 유사도)
            
            # ChromaDB 초기화
            if chromadb:
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.data_dir / "chroma_db")
                )
                
                try:
                    if self.chroma_client:
                        self.collection = self.chroma_client.get_collection("tech_knowledge")
                        logger.info("기존 ChromaDB 컬렉션 로드")
                except Exception:
                    if self.chroma_client:
                        self.collection = self.chroma_client.create_collection(
                            name="tech_knowledge",
                            metadata={"description": "Tech news and AI knowledge base"}
                        )
                        logger.info("새 ChromaDB 컬렉션 생성")
                
            # 기존 데이터 로드
            await self.load_existing_data()
            
        except Exception as e:
            logger.error(f"벡터 검색 시스템 초기화 오류: {e}")
            raise
    
    async def load_existing_data(self):
        """기존 저장된 데이터 로드"""
        if not faiss:
            return
        try:
            # FAISS 인덱스 로드
            index_path = self.data_dir / "faiss_index.bin"
            if index_path.exists() and faiss:
                self.index = faiss.read_index(str(index_path))
                logger.info("기존 FAISS 인덱스 로드")
            
            # 메타데이터 로드
            metadata_path = self.data_dir / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"메타데이터 {len(self.metadata)}개 로드")
                
        except Exception as e:
            logger.error(f"기존 데이터 로드 오류: {e}")
    
    async def save_data(self):
        """데이터 저장"""
        if not faiss or not self.index:
            return
        try:
            # FAISS 인덱스 저장
            faiss.write_index(self.index, str(self.data_dir / "faiss_index.bin"))
            
            # 메타데이터 저장
            with open(self.data_dir / "metadata.pkl", 'wb') as f:
                pickle.dump(self.metadata, f)
                
            logger.info("벡터 데이터베이스 저장 완료")
            
        except Exception as e:
            logger.error(f"데이터 저장 오류: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not text:
            return ""
            
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수문자 정리
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 연속 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    async def add_articles(self, articles: List[Dict[str, Any]]):
        """기사들을 벡터 데이터베이스에 추가"""
        if not articles or not self.encoder or not self.index or not self.collection:
            logger.warning("시스템이 초기화되지 않았거나 추가할 기사가 없어 기사 추가를 건너뜁니다.")
            return
            
        try:
            texts_to_embed = []
            new_metadata = []
            
            for article in articles:
                if article is None:
                    continue
                # 텍스트 조합 (제목 + 요약 + 본문)
                combined_text = f"{article.get('title', '')} {article.get('summary', '')} {article.get('full_text', '')}"
                processed_text = self.preprocess_text(combined_text)
                
                if len(processed_text) < 50:
                    continue
                    
                texts_to_embed.append(processed_text)
                
                metadata: Dict[str, Any] = {
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'link': article.get('link', ''),
                    'category': article.get('category', ''),
                    'source': article.get('source', ''),
                    'published': article.get('published', ''),
                    'collected_at': article.get('collected_at', ''),
                    'text_length': len(processed_text)
                }
                new_metadata.append(metadata)
            
            if not texts_to_embed:
                logger.warning("임베딩할 텍스트가 없습니다")
                return
            
            # 임베딩 생성
            logger.info(f"{len(texts_to_embed)}개 텍스트 임베딩 생성 중...")
            embeddings = self.encoder.encode(texts_to_embed, show_progress_bar=True)
            
            # 정규화 (코사인 유사도를 위해)
            embeddings_np = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_np)
            
            # FAISS에 추가
            self.index.add(embeddings_np)
            self.metadata.extend(new_metadata)
            
            # ChromaDB에도 추가
            ids = [f"doc_{len(self.metadata) - len(new_metadata) + i}" for i in range(len(texts_to_embed))]
            
            self.collection.add(
                documents=texts_to_embed,
                metadatas=new_metadata,
                ids=ids
            )
            
            logger.info(f"벡터 데이터베이스에 {len(texts_to_embed)}개 문서 추가 완료")
            
            # 데이터 저장
            await self.save_data()
            
        except Exception as e:
            logger.error(f"기사 추가 오류: {e}")
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """벡터 검색 수행"""
        if not self.encoder or not self.index:
            logger.warning("인코더 또는 인덱스가 초기화되지 않아 검색을 수행할 수 없습니다.")
            return []
        try:
            if not query.strip():
                return []
                
            # 쿼리 임베딩
            query_embedding = self.encoder.encode([query])
            query_embedding_np = np.array(query_embedding, dtype=np.float32)
            faiss.normalize_L2(query_embedding_np)
            
            # FAISS 검색
            scores, indices = self.index.search(query_embedding_np, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(score)
                    result['rank'] = i + 1
                    results.append(result)
            
            logger.info(f"검색 완료: '{query}' -> {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"벡터 검색 오류: {e}")
            return []
    
    async def get_relevant_context(self, query: str, max_context_length: int = 2000) -> str:
        """쿼리와 관련된 컨텍스트 생성"""
        try:
            search_results = await self.search(query, top_k=3)
            
            if not search_results:
                return ""
            
            context_parts = []
            current_length = 0
            
            for result in search_results:
                title = result.get('title', '')
                summary = result.get('summary', '')
                source = result.get('source', '')
                
                part = f"[{source}] {title}\n{summary}\n"
                
                if current_length + len(part) > max_context_length:
                    break
                    
                context_parts.append(part)
                current_length += len(part)
            
            context = "\n".join(context_parts)
            logger.info(f"컨텍스트 생성 완료: {len(context)}자")
            
            return context
            
        except Exception as e:
            logger.error(f"컨텍스트 생성 오류: {e}")
            return ""

class IntelligentResponseGenerator:
    """지능형 응답 생성기"""
    
    def __init__(self, knowledge_base: VectorKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.tech_keywords = {
            'ai': ['인공지능', 'AI', 'artificial intelligence', '머신러닝', 'machine learning', '딥러닝', 'deep learning'],
            'gpt': ['GPT', 'ChatGPT', 'OpenAI', 'LLM', '대화형 AI', '언어모델'],
            'tech': ['기술', '테크', 'technology', '혁신', 'innovation', '스타트업', 'startup'],
            'programming': ['프로그래밍', 'programming', '개발', 'development', '코딩', 'coding'],
            'cloud': ['클라우드', 'cloud', 'AWS', 'Azure', 'GCP', '서버리스'],
            'blockchain': ['블록체인', 'blockchain', '암호화폐', 'cryptocurrency', 'NFT', 'DeFi']
        }
    
    def detect_tech_category(self, query: str) -> List[str]:
        """쿼리에서 기술 카테고리 감지"""
        query_lower = query.lower()
        detected_categories = []
        
        for category, keywords in self.tech_keywords.items():
            if any(keyword.lower() in query_lower for keyword in keywords):
                detected_categories.append(category)
        
        return detected_categories
    
    async def generate_enhanced_response(self, query: str, base_response: str) -> str:
        """기존 응답을 최신 정보로 강화"""
        try:
            # 기술 카테고리 감지
            categories = self.detect_tech_category(query)
            
            if not categories:
                return base_response
            
            # 관련 컨텍스트 수집
            context = await self.knowledge_base.get_relevant_context(query)
            
            if not context:
                return base_response
            
            # 응답 강화
            enhanced_response = f"""{base_response}

📰 **최신 관련 정보:**

{context}

💡 *위 정보는 최신 기술 뉴스를 바탕으로 제공되었습니다.*"""
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"응답 강화 오류: {e}")
            return base_response
    
    async def get_tech_news_summary(self, category: str = None) -> str:
        """기술 뉴스 요약 생성"""
        try:
            if category:
                query = f"{category} 최신 뉴스"
            else:
                query = "최신 기술 뉴스 AI GPT"
            
            search_results = await self.knowledge_base.search(query, top_k=5)
            
            if not search_results:
                return "요약할 최신 기술 뉴스를 찾지 못했어요."
            
            summary_parts = ["📰 **최신 기술 뉴스 요약**\n"]
            for result in search_results:
                title = result.get('title', '제목 없음')
                link = result.get('link', '#')
                source = result.get('source', '출처 불명')
                summary_parts.append(f"• **[{source}] {title}**\n  [링크]({link})")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"기술 뉴스 요약 생성 오류: {e}")
            return "기술 뉴스 요약을 생성하는 중 오류가 발생했어요."

# --- 모듈 API ---
# 이 시스템의 공개 인터페이스입니다. 다른 모듈에서는 이 함수들을 사용합니다.

knowledge_system_instance: Optional[VectorKnowledgeBase] = None
response_generator_instance: Optional[IntelligentResponseGenerator] = None
is_initialized = False

async def initialize_knowledge_system(update_on_start: bool = True):
    """
    고급 지식 시스템을 초기화합니다.
    - 벡터 데이터베이스를 설정합니다.
    - 응답 생성기를 준비합니다.
    - update_on_start가 True이면, 시작 시 최신 뉴스를 수집하고 데이터베이스에 추가합니다.
    """
    global knowledge_system_instance, response_generator_instance, is_initialized
    
    if is_initialized:
        logger.info("지식 시스템이 이미 초기화되었습니다.")
        return

    try:
        logger.info("고급 지식 시스템 초기화 시작...")
        
        knowledge_system_instance = VectorKnowledgeBase()
        await knowledge_system_instance.initialize()
        
        response_generator_instance = IntelligentResponseGenerator(knowledge_system_instance)
        
        if update_on_start:
            logger.info("시작 시 뉴스 업데이트를 수행합니다...")
            async with TechNewsCollector() as collector:
                articles = await collector.collect_all_news()
                if articles:
                    # 기사 내용 상세 수집 (선택적, 시간이 오래 걸릴 수 있음)
                    # enhanced_articles = await asyncio.gather(
                    #     *[collector.enhance_article_content(art) for art in articles[:20]] # 예: 20개만
                    # )
                    await knowledge_system_instance.add_articles(articles)
        
        is_initialized = True
        logger.info("고급 지식 시스템 초기화 완료.")
        
    except Exception as e:
        logger.critical(f"고급 지식 시스템 초기화에 치명적인 오류 발생: {e}", exc_info=True)
        # 초기화 실패 시, 관련 인스턴스를 None으로 설정하여 오류 전파를 막습니다.
        knowledge_system_instance = None
        response_generator_instance = None
        is_initialized = False

async def get_enhanced_response(query: str, base_response: str) -> str:
    """
    기본 응답에 최신 기술 정보를 추가하여 강화된 응답을 반환합니다.
    """
    if not is_initialized or not response_generator_instance:
        logger.warning("지식 시스템이 초기화되지 않아 응답 강화를 건너뜁니다.")
        return base_response
    
    return await response_generator_instance.generate_enhanced_response(query, base_response)

async def get_tech_news_summary(category: str = None) -> str:
    """
    지정된 카테고리 또는 전체 최신 기술 뉴스에 대한 요약을 반환합니다.
    """
    if not is_initialized or not response_generator_instance:
        logger.warning("지식 시스템이 초기화되지 않아 뉴스 요약을 건너뜁니다.")
        return "지식 시스템이 아직 준비되지 않았어요. 잠시 후 다시 시도해주세요."
        
    return await response_generator_instance.get_tech_news_summary(category)

async def search_knowledge_base(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    내부 지식 베이스에서 정보를 검색합니다.
    """
    if not is_initialized or not knowledge_system_instance:
        logger.warning("지식 시스템이 초기화되지 않아 지식 검색을 건너뜁니다.")
        return []
        
    return await knowledge_system_instance.search(query, top_k)

# --- 스케줄링된 작업 ---
async def scheduled_news_update():
    """
    정기적으로 최신 뉴스를 수집하고 벡터 데이터베이스를 업데이트합니다.
    """
    if not is_initialized:
        logger.info("스케줄된 뉴스 업데이트: 시스템이 아직 초기화되지 않았습니다.")
        return

    logger.info("스케줄된 뉴스 업데이트 시작...")
    try:
        async with TechNewsCollector() as collector:
            articles = await collector.collect_all_news()
            if articles and knowledge_system_instance:
                await knowledge_system_instance.add_articles(articles)
        logger.info("스케줄된 뉴스 업데이트 완료.")
    except Exception as e:
        logger.error(f"스케줄된 뉴스 업데이트 중 오류 발생: {e}", exc_info=True)