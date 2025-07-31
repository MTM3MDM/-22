"""
실시간 자동 학습 시스템
- 주기적으로 최신 기술 뉴스 수집
- 다중 AI 모델을 통한 정보 분석 및 요약
- 벡터 데이터베이스에 저장하여 실시간 검색 가능
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import aiohttp
import aiosqlite
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from serpapi import GoogleSearch
import feedparser
import openai
from dataclasses import dataclass

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearnedInfo:
    """학습된 정보 구조"""
    title: str
    content: str
    summary: str
    source: str
    url: str
    timestamp: datetime
    category: str
    keywords: List[str]
    confidence: float

class MultiModelAnalyzer:
    """다중 AI 모델을 활용한 정보 분석기"""
    
    def __init__(self, openrouter_key: str):
        self.openrouter_key = openrouter_key
        self.base_url = "https://openrouter.ai/api/v1"
        
        # 사용할 모델들 (무료 모델들로 구성)
        self.models = {
            'llama': 'meta-llama/llama-3.2-3b-instruct:free',
            'gemma': 'google/gemma-2-9b-it:free',
            'qwen': 'qwen/qwen-2-7b-instruct:free',
            'phi': 'microsoft/phi-3-mini-128k-instruct:free',
            'mistral': 'mistralai/mistral-7b-instruct:free'
        }
        
        # OpenAI 클라이언트 설정 (OpenRouter 사용)
        self.client = openai.AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.openrouter_key,
        )
    
    async def analyze_with_multiple_models(self, content: str, task: str) -> Dict[str, str]:
        """여러 모델로 동시에 분석"""
        results = {}
        
        # 각 모델별 프롬프트 최적화
        prompts = {
            'llama': f"다음 기술 뉴스를 분석해서 핵심 내용을 한국어로 요약해주세요. 2025년 최신 정보인지 확인하고, 중요도를 1-10으로 평가해주세요:\n\n{content}",
            'gemma': f"이 기술 뉴스의 핵심 포인트와 영향도를 분석해주세요. 특히 AI/ML 분야의 최신 동향인지 판단해주세요:\n\n{content}",
            'qwen': f"다음 내용을 요약하고 관련 키워드를 추출해주세요. 2025년 기술 트렌드와의 연관성도 분석해주세요:\n\n{content}",
            'phi': f"이 뉴스가 개발자나 기술자에게 어떤 의미인지 실용적 관점에서 분석해주세요:\n\n{content}",
            'mistral': f"다음 기술 정보의 신뢰성과 정확성을 평가하고, 핵심 내용을 간단히 정리해주세요:\n\n{content}"
        }
        
        # API 사용량 절약을 위해 랜덤으로 2개 모델만 사용
        import random
        selected_models = random.sample(list(self.models.items()), min(2, len(self.models)))
        
        tasks = []
        for model_name, model_id in selected_models:
            if model_name in prompts:
                task = self._query_model(model_id, prompts[model_name], model_name)
                tasks.append(task)
        
        # 결과 수집
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (model_name, _) in enumerate(selected_models):
            if i < len(responses) and not isinstance(responses[i], Exception):
                results[model_name] = responses[i]
            else:
                logger.warning(f"{model_name} 모델 분석 실패")
        
        return results
    
    async def _query_model(self, model_id: str, prompt: str, model_name: str) -> str:
        """개별 모델 쿼리"""
        try:
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "당신은 2025년 최신 기술 정보를 분석하는 전문가입니다. 정확하고 간결한 한국어로 답변해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"{model_name} 모델 쿼리 오류: {e}")
            return ""
    
    async def synthesize_analysis(self, multi_results: Dict[str, str], original_content: str) -> LearnedInfo:
        """다중 모델 결과를 종합하여 최종 학습 정보 생성"""
        try:
            # 모든 분석 결과를 종합
            synthesis_prompt = f"""
다음은 여러 AI 모델이 같은 기술 뉴스를 분석한 결과들입니다:

원본 내용: {original_content[:500]}...

분석 결과들:
{json.dumps(multi_results, ensure_ascii=False, indent=2)}

이 결과들을 종합해서 다음 형식으로 정리해주세요:
1. 핵심 요약 (2-3문장)
2. 주요 키워드 (5개 이하)
3. 카테고리 (AI/ML, 프로그래밍, 스타트업, 하드웨어 등)
4. 중요도 (1-10)
5. 2025년 최신 정보 여부 (true/false)
"""
            
            # Llama로 최종 종합
            response = await self.client.chat.completions.create(
                model=self.models['llama'],
                messages=[
                    {"role": "system", "content": "당신은 기술 정보 종합 전문가입니다. 여러 분석을 종합해서 정확한 결론을 도출해주세요."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=300,
                temperature=0.2
            )
            
            synthesis = response.choices[0].message.content
            
            # 결과 파싱 (간단한 파싱)
            lines = synthesis.split('\n')
            summary = ""
            keywords = []
            category = "기술"
            confidence = 7.0
            
            for line in lines:
                if "요약" in line or "핵심" in line:
                    summary = line.split(':', 1)[-1].strip()
                elif "키워드" in line:
                    keyword_text = line.split(':', 1)[-1].strip()
                    keywords = [k.strip() for k in keyword_text.split(',')]
                elif "카테고리" in line:
                    category = line.split(':', 1)[-1].strip()
                elif "중요도" in line:
                    try:
                        confidence = float(line.split(':', 1)[-1].strip().split()[0])
                    except:
                        confidence = 7.0
            
            return LearnedInfo(
                title=original_content[:100] + "...",
                content=original_content,
                summary=summary or "최신 기술 정보",
                source="Multi-Model Analysis",
                url="",
                timestamp=datetime.now(),
                category=category,
                keywords=keywords,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"분석 종합 오류: {e}")
            return LearnedInfo(
                title="분석 실패",
                content=original_content,
                summary="분석 중 오류 발생",
                source="Error",
                url="",
                timestamp=datetime.now(),
                category="오류",
                keywords=[],
                confidence=0.0
            )

class RealtimeLearningSystem:
    """실시간 자동 학습 시스템"""
    
    def __init__(self, openrouter_key: str, serpapi_key: str):
        self.openrouter_key = openrouter_key
        self.serpapi_key = serpapi_key
        self.db_path = "realtime_learning.db"
        
        # 다중 모델 분석기
        self.analyzer = MultiModelAnalyzer(openrouter_key)
        
        # 임베딩 모델
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # FAISS 벡터 인덱스
        self.dimension = 384
        self.index = faiss.IndexFlatIP(self.dimension)
        self.learned_info = []  # 학습된 정보 메타데이터
        
        # 학습 소스들 (API 사용량 절약)
        self.learning_sources = {
            'search_queries': [
                'GPT-5 release news 2025',
                'Claude 3.5 updates',
                'OpenAI latest news 2025',
                'AI breakthrough 2025',
                'LLM model releases'
            ],
            'rss_feeds': [
                'https://openai.com/blog/rss.xml',
                'https://techcrunch.com/category/artificial-intelligence/feed/',
                'https://www.theverge.com/ai-artificial-intelligence/rss/index.xml',
                'https://www.technologyreview.com/feed/'
            ]
        }
        
        # 학습 주기 설정 (API 사용량 절약)
        self.learning_interval = 7200  # 2시간마다
        self.last_learning_time = None
        
    async def initialize(self):
        """시스템 초기화"""
        await self.init_database()
        await self.load_learned_info()
        logger.info("실시간 학습 시스템 초기화 완료")
    
    async def init_database(self):
        """데이터베이스 초기화"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS learned_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    content TEXT,
                    summary TEXT,
                    source TEXT,
                    url TEXT,
                    timestamp TEXT,
                    category TEXT,
                    keywords TEXT,
                    confidence REAL,
                    embedding BLOB
                )
            ''')
            await db.commit()
    
    async def load_learned_info(self):
        """저장된 학습 정보 로드"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('SELECT * FROM learned_info ORDER BY timestamp DESC LIMIT 1000') as cursor:
                rows = await cursor.fetchall()
                
                embeddings = []
                for row in rows:
                    info = LearnedInfo(
                        title=row[1],
                        content=row[2],
                        summary=row[3],
                        source=row[4],
                        url=row[5],
                        timestamp=datetime.fromisoformat(row[6]),
                        category=row[7],
                        keywords=json.loads(row[8]) if row[8] else [],
                        confidence=row[9]
                    )
                    self.learned_info.append(info)
                    
                    # 임베딩 로드
                    if row[10]:
                        embedding = np.frombuffer(row[10], dtype=np.float32)
                        embeddings.append(embedding)
                
                # FAISS 인덱스 구성
                if embeddings:
                    embeddings_array = np.array(embeddings)
                    self.index.add(embeddings_array)
                
                logger.info(f"학습된 정보 {len(self.learned_info)}개 로드 완료")
    
    async def start_continuous_learning(self):
        """지속적 학습 시작"""
        logger.info("실시간 자동 학습 시작")
        
        while True:
            try:
                current_time = datetime.now()
                
                # 학습 주기 체크
                if (self.last_learning_time is None or 
                    (current_time - self.last_learning_time).seconds >= self.learning_interval):
                    
                    logger.info("새로운 학습 사이클 시작")
                    await self.learn_from_all_sources()
                    self.last_learning_time = current_time
                
                # 1분마다 체크
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"지속적 학습 오류: {e}")
                await asyncio.sleep(300)  # 5분 대기 후 재시도
    
    async def learn_from_all_sources(self):
        """모든 소스에서 학습"""
        new_info_count = 0
        
        # 웹 검색을 통한 학습
        for query in self.learning_sources['search_queries']:
            try:
                search_results = await self.search_and_learn(query)
                new_info_count += len(search_results)
                await asyncio.sleep(2)  # API 제한 고려
            except Exception as e:
                logger.error(f"검색 학습 오류 ({query}): {e}")
        
        # RSS 피드를 통한 학습
        for rss_url in self.learning_sources['rss_feeds']:
            try:
                rss_results = await self.learn_from_rss(rss_url)
                new_info_count += len(rss_results)
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"RSS 학습 오류 ({rss_url}): {e}")
        
        logger.info(f"학습 완료: {new_info_count}개의 새로운 정보 획득")
    
    async def search_and_learn(self, query: str) -> List[LearnedInfo]:
        """웹 검색을 통한 학습"""
        try:
            # SerpAPI 검색
            search_params = {
                "q": f"{query} 2025",
                "api_key": self.serpapi_key,
                "engine": "google",
                "num": 5,
                "hl": "ko",
                "gl": "kr",
                "tbs": "qdr:m"  # 최근 1개월
            }
            
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            learned_items = []
            
            if 'organic_results' in results:
                for result in results['organic_results'][:2]:  # 상위 2개만 (API 절약)
                    title = result.get('title', '')
                    snippet = result.get('snippet', '')
                    link = result.get('link', '')
                    
                    # 중복 체크
                    if await self.is_duplicate_content(title, snippet):
                        continue
                    
                    # 다중 모델 분석
                    content = f"제목: {title}\n내용: {snippet}"
                    analysis_results = await self.analyzer.analyze_with_multiple_models(content, "summarize")
                    
                    # 종합 분석
                    learned_info = await self.analyzer.synthesize_analysis(analysis_results, content)
                    learned_info.url = link
                    learned_info.source = f"Search: {query}"
                    
                    # 저장
                    await self.save_learned_info(learned_info)
                    learned_items.append(learned_info)
            
            return learned_items
            
        except Exception as e:
            logger.error(f"검색 학습 오류: {e}")
            return []
    
    async def learn_from_rss(self, rss_url: str) -> List[LearnedInfo]:
        """RSS 피드를 통한 학습"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(rss_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        learned_items = []
                        
                        for entry in feed.entries[:2]:  # 최신 2개만 (API 절약)
                            title = entry.get('title', '')
                            summary = entry.get('summary', '')
                            link = entry.get('link', '')
                            
                            # 중복 체크
                            if await self.is_duplicate_content(title, summary):
                                continue
                            
                            # 최근 정보인지 확인 (1주일 이내)
                            pub_date = entry.get('published_parsed')
                            if pub_date:
                                pub_datetime = datetime(*pub_date[:6])
                                if (datetime.now() - pub_datetime).days > 7:
                                    continue
                            
                            # 다중 모델 분석
                            content = f"제목: {title}\n내용: {summary}"
                            analysis_results = await self.analyzer.analyze_with_multiple_models(content, "analyze")
                            
                            # 종합 분석
                            learned_info = await self.analyzer.synthesize_analysis(analysis_results, content)
                            learned_info.url = link
                            learned_info.source = f"RSS: {rss_url}"
                            
                            # 저장
                            await self.save_learned_info(learned_info)
                            learned_items.append(learned_info)
                        
                        return learned_items
        
        except Exception as e:
            logger.error(f"RSS 학습 오류: {e}")
            return []
    
    async def is_duplicate_content(self, title: str, content: str) -> bool:
        """중복 콘텐츠 체크"""
        # 간단한 중복 체크 (제목 기반)
        for info in self.learned_info[-100:]:  # 최근 100개만 체크
            if info.title and title:
                # 제목 유사도 체크 (간단한 방법)
                title_words = set(title.lower().split())
                info_words = set(info.title.lower().split())
                
                if len(title_words & info_words) / len(title_words | info_words) > 0.7:
                    return True
        
        return False
    
    async def save_learned_info(self, info: LearnedInfo):
        """학습된 정보 저장"""
        try:
            # 임베딩 생성
            text_for_embedding = f"{info.title} {info.summary}"
            embedding = self.encoder.encode([text_for_embedding])[0]
            
            # 데이터베이스 저장
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT INTO learned_info 
                    (title, content, summary, source, url, timestamp, category, keywords, confidence, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    info.title,
                    info.content,
                    info.summary,
                    info.source,
                    info.url,
                    info.timestamp.isoformat(),
                    info.category,
                    json.dumps(info.keywords, ensure_ascii=False),
                    info.confidence,
                    embedding.tobytes()
                ))
                await db.commit()
            
            # 메모리에 추가
            self.learned_info.append(info)
            
            # FAISS 인덱스에 추가
            self.index.add(np.array([embedding]))
            
            logger.info(f"새로운 정보 학습 완료: {info.title[:50]}...")
            
        except Exception as e:
            logger.error(f"정보 저장 오류: {e}")
    
    async def query_learned_info(self, question: str, top_k: int = 5) -> List[LearnedInfo]:
        """학습된 정보에서 관련 내용 검색"""
        try:
            # 질문 임베딩
            question_embedding = self.encoder.encode([question])[0]
            
            # FAISS 검색
            if self.index.ntotal > 0:
                scores, indices = self.index.search(np.array([question_embedding]), top_k)
                
                results = []
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(self.learned_info) and score > 0.3:  # 유사도 임계값
                        info = self.learned_info[idx]
                        info.confidence = float(score)  # 검색 점수로 업데이트
                        results.append(info)
                
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"학습 정보 검색 오류: {e}")
            return []
    
    async def get_latest_info_about(self, topic: str) -> Optional[str]:
        """특정 주제에 대한 최신 정보 반환"""
        try:
            # 먼저 학습된 정보에서 검색
            learned_results = await self.query_learned_info(topic, top_k=3)
            
            if learned_results:
                # 가장 관련성 높은 정보들을 종합
                summaries = []
                for info in learned_results:
                    age_days = (datetime.now() - info.timestamp).days
                    summaries.append(f"[{age_days}일 전] {info.summary}")
                
                return f"{topic}에 대한 최신 정보:\n" + "\n".join(summaries)
            
            # 학습된 정보가 없으면 실시간 검색
            search_results = await self.search_and_learn(f"{topic} latest news 2025")
            
            if search_results:
                return f"{topic}에 대한 실시간 검색 결과:\n{search_results[0].summary}"
            
            return f"{topic}에 대한 최신 정보를 찾을 수 없어요. 조금 더 구체적으로 질문해주시면 도움이 될 것 같아요."
            
        except Exception as e:
            logger.error(f"최신 정보 조회 오류: {e}")
            return "정보 조회 중 오류가 발생했어요."

# 전역 학습 시스템 인스턴스
learning_system = None

async def initialize_learning_system():
    """학습 시스템 초기화"""
    global learning_system
    
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    serpapi_key = os.getenv('SERPAPI_KEY')
    
    if not openrouter_key or not serpapi_key:
        logger.error("OpenRouter 또는 SerpAPI 키가 설정되지 않았습니다")
        return None
    
    learning_system = RealtimeLearningSystem(openrouter_key, serpapi_key)
    await learning_system.initialize()
    
    # 백그라운드에서 지속적 학습 시작
    asyncio.create_task(learning_system.start_continuous_learning())
    
    logger.info("실시간 학습 시스템 가동 시작!")
    return learning_system

async def get_smart_answer(question: str) -> str:
    """스마트 답변 생성 (학습된 정보 기반)"""
    global learning_system
    
    if not learning_system:
        return "학습 시스템이 아직 초기화되지 않았어요."
    
    return await learning_system.get_latest_info_about(question)