"""
지능형 웹 검색 및 기억 시스템
SerpAPI를 사용한 실시간 검색과 벡터 기반 기억 기능
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import aiohttp
import aiosqlite
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from serpapi import GoogleSearch
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentWebSearcher:
    """지능형 웹 검색 시스템"""
    
    def __init__(self, serpapi_key: str = "YOUR_SERPAPI_KEY_HERE", db_path: str = "search_memory.db"):
        self.serpapi_key = serpapi_key
        self.db_path = db_path
        
        # 임베딩 모델 초기화
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # FAISS 인덱스 초기화
        self.dimension = 384  # all-MiniLM-L6-v2의 차원
        self.index = faiss.IndexFlatIP(self.dimension)  # 코사인 유사도용
        self.search_memory = []  # 검색 기록 메타데이터
        
        # 검색 결과 캐시 (메모리 효율성을 위해)
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=6)  # 6시간 캐시
        
        # 2025년 최신 기술 키워드 (검색 강화용)
        self.tech_keywords_2025 = {
            'ai_models': ['GPT-5', 'Claude 3.5', 'Gemini 2.0', 'Llama 3', 'GPT-4o', 'o1-preview', 'o1-mini'],
            'ai_trends': ['AGI', 'ASI', 'multimodal AI', 'AI agents', 'reasoning models', 'AI safety'],
            'programming': ['GitHub Copilot', 'Cursor IDE', 'AI coding', 'automated programming', 'code generation'],
            'hardware': ['H100', 'H200', 'B200', 'AI chips', 'TPU v5', 'neuromorphic computing'],
            'companies': ['OpenAI', 'Anthropic', 'Google DeepMind', 'Meta AI', 'xAI', 'Perplexity'],
            'frameworks': ['LangChain', 'LlamaIndex', 'AutoGen', 'CrewAI', 'Semantic Kernel']
        }
        
        # 2025년 최신 검색 소스 우선순위
        self.priority_sources_2025 = [
            'openai.com', 'anthropic.com', 'deepmind.google', 'ai.meta.com',
            'github.com', 'arxiv.org', 'papers.nips.cc', 'techcrunch.com',
            'theverge.com', 'wired.com', 'technologyreview.com'
        ]
        
    async def initialize(self):
        """시스템 초기화"""
        await self.init_database()
        await self.load_search_memory()
        logger.info("지능형 웹 검색 시스템 초기화 완료! 이제 기억하면서 검색할 수 있어요~ ✨")
    
    async def init_database(self):
        """데이터베이스 초기화"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    query_hash TEXT UNIQUE,
                    search_results TEXT,
                    summary TEXT,
                    embedding BLOB,
                    search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    relevance_score REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS search_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_type TEXT,
                    response_time REAL,
                    result_count INTEGER,
                    user_satisfaction REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.commit()
    
    async def load_search_memory(self):
        """저장된 검색 기록을 메모리로 로드"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT query, search_results, summary, embedding, search_date, relevance_score
                    FROM search_history 
                    ORDER BY search_date DESC 
                    LIMIT 1000
                """) as cursor:
                    rows = await cursor.fetchall()
                    
                    embeddings = []
                    for row in rows:
                        query, results, summary, embedding_blob, date, score = row
                        
                        if embedding_blob:
                            # BLOB에서 numpy 배열로 변환
                            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                            embeddings.append(embedding)
                            
                            self.search_memory.append({
                                'query': query,
                                'results': json.loads(results) if results else [],
                                'summary': summary,
                                'date': date,
                                'relevance_score': score
                            })
                    
                    if embeddings:
                        embeddings_array = np.vstack(embeddings)
                        self.index.add(embeddings_array.astype('float32'))
                        
            logger.info(f"검색 기록 {len(self.search_memory)}개 로드 완료! 이전 기억들을 다 불러왔어요~ 🧠")
            
        except Exception as e:
            logger.error(f"검색 기록 로드 오류: {e}")
    
    def _clean_cache(self):
        """만료된 캐시 정리"""
        now = datetime.now()
        expired_keys = [
            key for key, expiry in self.cache_expiry.items() 
            if now > expiry
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.cache_expiry[key]
    
    def _get_query_hash(self, query: str) -> str:
        """쿼리 해시 생성"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    async def search_web(self, query: str, num_results: int = 5) -> Dict:
        """SerpAPI를 사용한 웹 검색 (2025년 최신 정보 강화)"""
        start_time = datetime.now()
        
        try:
            # 2025년 최신 정보를 위한 쿼리 강화
            enhanced_query = self.enhance_query_for_2025(query)
            
            # 캐시 확인 (원본 쿼리로)
            query_hash = self._get_query_hash(query)
            if query_hash in self.cache and datetime.now() < self.cache_expiry[query_hash]:
                logger.info(f"캐시에서 검색 결과 반환: {query}")
                return self.cache[query_hash]
            
            # SerpAPI 검색 (강화된 쿼리 사용)
            search_params = {
                "q": enhanced_query,
                "api_key": self.serpapi_key,
                "engine": "google",
                "num": num_results,
                "hl": "ko",  # 한국어
                "gl": "kr",   # 한국 지역
                "tbs": "qdr:y"  # 최근 1년 내 결과 우선
            }
            
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            # 결과 처리
            processed_results = self._process_search_results(results)
            
            # 캐시에 저장
            self._clean_cache()
            self.cache[query_hash] = processed_results
            self.cache_expiry[query_hash] = datetime.now() + self.cache_duration
            
            # 응답 시간 기록
            response_time = (datetime.now() - start_time).total_seconds()
            await self._log_search_analytics("web_search", response_time, len(processed_results.get('results', [])))
            
            return processed_results
            
        except Exception as e:
            logger.error(f"웹 검색 오류: {e}")
            return {
                'query': query,
                'results': [],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _process_search_results(self, raw_results: Dict) -> Dict:
        """검색 결과 처리 및 정리"""
        processed = {
            'results': [],
            'answer_box': None,
            'knowledge_graph': None,
            'related_questions': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # 일반 검색 결과
        if 'organic_results' in raw_results:
            for result in raw_results['organic_results'][:5]:
                processed['results'].append({
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', ''),
                    'position': result.get('position', 0)
                })
        
        # 답변 박스 (즉시 답변)
        if 'answer_box' in raw_results:
            answer_box = raw_results['answer_box']
            processed['answer_box'] = {
                'answer': answer_box.get('answer', ''),
                'title': answer_box.get('title', ''),
                'link': answer_box.get('link', '')
            }
        
        # 지식 그래프
        if 'knowledge_graph' in raw_results:
            kg = raw_results['knowledge_graph']
            processed['knowledge_graph'] = {
                'title': kg.get('title', ''),
                'description': kg.get('description', ''),
                'attributes': kg.get('attributes', {})
            }
        
        # 관련 질문
        if 'related_questions' in raw_results:
            for rq in raw_results['related_questions'][:3]:
                processed['related_questions'].append({
                    'question': rq.get('question', ''),
                    'snippet': rq.get('snippet', '')
                })
        
        return processed
    
    def enhance_query_for_2025(self, query: str) -> str:
        """2025년 최신 정보 검색을 위한 쿼리 강화"""
        enhanced_query = query
        
        # 2025년 키워드 추가
        if any(keyword in query.lower() for keyword in ['ai', 'gpt', 'chatgpt', 'llm', '인공지능']):
            enhanced_query += " 2025 latest"
        
        # 특정 모델명 감지 시 최신 버전으로 확장
        model_mappings = {
            'gpt': 'GPT-5 GPT-4o o1-preview',
            'claude': 'Claude 3.5 Sonnet',
            'gemini': 'Gemini 2.0 Pro',
            'llama': 'Llama 3.1 3.2'
        }
        
        for old_term, new_terms in model_mappings.items():
            if old_term in query.lower():
                enhanced_query += f" {new_terms}"
        
        # 최신 소스 우선 검색을 위한 site 필터 추가 (가끔씩)
        import random
        if random.random() < 0.3:  # 30% 확률로 우선 소스 검색
            priority_site = random.choice(self.priority_sources_2025)
            enhanced_query += f" site:{priority_site}"
        
        return enhanced_query
    
    async def generate_summary(self, search_results: Dict, query: str) -> str:
        """검색 결과를 바탕으로 요약 생성"""
        try:
            summary_parts = []
            
            # 답변 박스가 있으면 우선 사용
            if search_results.get('answer_box'):
                answer = search_results['answer_box'].get('answer', '')
                if answer:
                    summary_parts.append(f"📋 바로 답변드릴게요: {answer}")
            
            # 지식 그래프 정보
            if search_results.get('knowledge_graph'):
                kg = search_results['knowledge_graph']
                if kg.get('description'):
                    summary_parts.append(f"📚 간단히 설명하면: {kg['description']}")
            
            # 주요 검색 결과 요약
            if search_results.get('results'):
                summary_parts.append("🔍 찾아본 주요 내용들:")
                for i, result in enumerate(search_results['results'][:3], 1):
                    title = result.get('title', '')
                    snippet = result.get('snippet', '')
                    if title and snippet:
                        summary_parts.append(f"{i}. **{title}**\n   {snippet[:150]}...")
            
            # 관련 질문
            if search_results.get('related_questions'):
                summary_parts.append("\n❓ 이런 것도 궁금하실 것 같아요:")
                for rq in search_results['related_questions']:
                    question = rq.get('question', '')
                    if question:
                        summary_parts.append(f"• {question}")
            
            return "\n\n".join(summary_parts) if summary_parts else "어라? 검색 결과를 요약하기가 어려워요... 😅"
            
        except Exception as e:
            logger.error(f"요약 생성 오류: {e}")
            return "요약 생성 중 오류가 발생했습니다."
    
    async def save_search_to_memory(self, query: str, search_results: Dict, summary: str):
        """검색 결과를 장기 기억에 저장"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.encoder.encode([query])[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)  # 정규화
            
            # 데이터베이스에 저장
            query_hash = self._get_query_hash(query)
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO search_history 
                    (query, query_hash, search_results, summary, embedding, search_date, relevance_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    query,
                    query_hash,
                    json.dumps(search_results, ensure_ascii=False),
                    summary,
                    query_embedding.tobytes(),
                    datetime.now().isoformat(),
                    1.0
                ))
                await db.commit()
            
            # 메모리에도 추가
            self.search_memory.append({
                'query': query,
                'results': search_results,
                'summary': summary,
                'date': datetime.now().isoformat(),
                'relevance_score': 1.0
            })
            
            # FAISS 인덱스에 추가
            self.index.add(query_embedding.reshape(1, -1).astype('float32'))
            
            logger.info(f"검색 결과 저장 완료: {query} (이제 기억해뒀어요! 💾)")
            
        except Exception as e:
            logger.error(f"검색 결과 저장 오류: {e}")
    
    async def find_related_memory(self, query: str, top_k: int = 3, similarity_threshold: float = 0.7) -> List[Dict]:
        """관련된 과거 검색 기록 찾기"""
        try:
            if len(self.search_memory) == 0:
                return []
            
            # 쿼리 임베딩 생성
            query_embedding = self.encoder.encode([query])[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # FAISS로 유사한 검색 찾기
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                min(top_k, len(self.search_memory))
            )
            
            related_memories = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= similarity_threshold and idx < len(self.search_memory):
                    memory = self.search_memory[idx].copy()
                    memory['similarity_score'] = float(score)
                    related_memories.append(memory)
            
            return sorted(related_memories, key=lambda x: x['similarity_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"관련 기억 검색 오류: {e}")
            return []
    
    async def intelligent_search(self, query: str) -> Dict:
        """지능형 검색 - 기억 확인 후 필요시 새로 검색"""
        try:
            # 1. 관련된 과거 검색 기록 확인
            related_memories = await self.find_related_memory(query, top_k=3, similarity_threshold=0.8)
            
            # 2. 최근 관련 검색이 있으면 그것을 기반으로 답변
            if related_memories:
                recent_memory = related_memories[0]
                # 24시간 이내의 검색이면 재사용
                memory_date = datetime.fromisoformat(recent_memory['date'].replace('Z', '+00:00').replace('+00:00', ''))
                if datetime.now() - memory_date < timedelta(hours=24):
                    logger.info(f"기존 검색 기록 활용: {query} (아! 이거 전에 찾아봤던 거네요~)")
                    return {
                        'type': 'memory_based',
                        'query': query,
                        'answer': recent_memory['summary'],
                        'source': 'previous_search',
                        'similarity_score': recent_memory['similarity_score'],
                        'original_query': recent_memory['query'],
                        'search_date': recent_memory['date']
                    }
            
            # 3. 새로운 검색 수행
            logger.info(f"새로운 웹 검색 수행: {query} (새로 찾아볼게요! 🔍)")
            search_results = await self.search_web(query)
            
            if search_results.get('error'):
                return {
                    'type': 'error',
                    'query': query,
                    'error': search_results['error']
                }
            
            # 4. 요약 생성
            summary = await self.generate_summary(search_results, query)
            
            # 5. 기억에 저장
            await self.save_search_to_memory(query, search_results, summary)
            
            # 6. 관련 기억과 함께 결과 반환
            return {
                'type': 'new_search',
                'query': query,
                'answer': summary,
                'search_results': search_results,
                'related_memories': related_memories[:2] if related_memories else [],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"지능형 검색 오류: {e}")
            return {
                'type': 'error',
                'query': query,
                'error': str(e)
            }
    
    async def _log_search_analytics(self, query_type: str, response_time: float, result_count: int, satisfaction: float = 1.0):
        """검색 분석 데이터 로깅"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO search_analytics 
                    (query_type, response_time, result_count, user_satisfaction)
                    VALUES (?, ?, ?, ?)
                """, (query_type, response_time, result_count, satisfaction))
                await db.commit()
        except Exception as e:
            logger.error(f"분석 데이터 로깅 오류: {e}")
    
    async def get_search_stats(self) -> Dict:
        """검색 통계 조회"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 총 검색 수
                async with db.execute("SELECT COUNT(*) FROM search_history") as cursor:
                    total_searches = (await cursor.fetchone())[0]
                
                # 최근 24시간 검색 수
                async with db.execute("""
                    SELECT COUNT(*) FROM search_history 
                    WHERE search_date > datetime('now', '-1 day')
                """) as cursor:
                    recent_searches = (await cursor.fetchone())[0]
                
                # 평균 응답 시간
                async with db.execute("""
                    SELECT AVG(response_time) FROM search_analytics 
                    WHERE created_at > datetime('now', '-7 days')
                """) as cursor:
                    avg_response_time = (await cursor.fetchone())[0] or 0
                
                return {
                    'total_searches': total_searches,
                    'recent_searches': recent_searches,
                    'avg_response_time': round(avg_response_time, 2),
                    'memory_size': len(self.search_memory),
                    'cache_size': len(self.cache)
                }
        except Exception as e:
            logger.error(f"통계 조회 오류: {e}")
            return {}

# 전역 인스턴스
web_searcher = None

async def initialize_web_search(serpapi_key: str = "YOUR_SERPAPI_KEY_HERE"):
    """웹 검색 시스템 초기화"""
    global web_searcher
    web_searcher = IntelligentWebSearcher(serpapi_key)
    await web_searcher.initialize()
    return web_searcher

async def search_and_remember(query: str) -> Dict:
    """검색하고 기억하는 메인 함수"""
    if web_searcher is None:
        return {
            'type': 'error',
            'error': '웹 검색 시스템이 초기화되지 않았습니다.'
        }
    
    return await web_searcher.intelligent_search(query)

async def get_search_statistics() -> Dict:
    """검색 통계 조회"""
    if web_searcher is None:
        return {}
    
    return await web_searcher.get_search_stats()