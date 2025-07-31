# 루시아 디스코드 봇 - 초고급 AI 어시스턴트 (완전 업그레이드)
# 음성인식, 이미지분석, 실시간번역, 게임시스템, 코드분석 등 초고급 기능 탑재

import discord
import asyncio
import aiohttp
import wikipedia # type: ignore
import re
import os
import time
import json
import logging
import hashlib
import random
import requests
from bs4 import BeautifulSoup
from PIL import Image
from wordcloud import WordCloud # type: ignore
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
from plotly.offline import plot # type: ignore
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Callable, Coroutine, Union
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from textblob import TextBlob # type: ignore
import nltk # type: ignore
from datetime import datetime, timedelta
import pytz  # 시간대 처리용
import aiosqlite  # SQLite 비동기 처리

# 기능 플래그 (가용성)
image_analysis_available = False
face_recognition_available = False
scheduling_available = False
crypto_available = False
rss_available = False
svg_available = False
knowledge_system_available = False
intelligent_search_available = False
dynamic_response_available = False
realtime_learning_available = False
admin_system_available = False

# 이미지 분석 (선택적)
try:
    import torchvision.transforms as transforms # type: ignore
    from torchvision.models import resnet50 # type: ignore
    import torch
    image_analysis_available = True
    try:
        import face_recognition # type: ignore
        face_recognition_available = True
    except ImportError:
        pass # face_recognition_available는 이미 False입니다.
except ImportError:
    pass # image_analysis_available는 이미 False입니다.

# 스케줄링 (선택적)
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler # type: ignore
    from apscheduler.triggers.cron import CronTrigger # type: ignore
    scheduling_available = True
    try:
        import yfinance as yf # type: ignore
    except ImportError:
        pass # yfinance는 특정 기능에서만 사용됩니다.
except ImportError:
    pass

# 암호화폐 (선택적)
try:
    import cryptocompare # type: ignore
    crypto_available = True
except ImportError:
    pass

# RSS 피드 (선택적)
try:
    import feedparser # type: ignore
    rss_available = True
except ImportError:
    pass

# SVG 처리 (선택적)
try:
    from cairosvg import svg2png # type: ignore
    svg_available = True
except ImportError:
    pass

# --------------------------------------------------------------------------
# 로깅 설정
# --------------------------------------------------------------------------
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler = logging.FileHandler('lucia_bot.log', encoding='utf-8')
log_handler.setFormatter(log_formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# Discord 로거 레벨 설정
logging.getLogger('discord').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# --------------------------------------------------------------------------
# 모듈 임포트 및 플레이스홀더 정의
# --------------------------------------------------------------------------

# 고급 지식 시스템 (AKS)
try:
    from advanced_knowledge_system import (
        initialize_knowledge_system,
        get_knowledge_response,
        update_knowledge_from_search
    )
    knowledge_system_available = True
except ImportError as e:
    logger.warning(f"고급 지식 시스템 로드 실패: {e}")
    knowledge_system_available = False
    async def initialize_knowledge_system(): pass
    async def get_knowledge_response(*args, **kwargs): return None
    async def update_knowledge_from_search(*args, **kwargs): pass

# 지능형 웹 검색 (IWS)
try:
    from intelligent_web_search import (
        initialize_intelligent_search,
        search_and_remember
    )
    intelligent_search_available = True
except ImportError as e:
    logger.warning(f"지능형 웹 검색 시스템 로드 실패: {e}")
    intelligent_search_available = False
    async def initialize_intelligent_search(*args, **kwargs): pass
    async def search_and_remember(*args, **kwargs): return "검색 시스템이 사용할 수 없습니다."

# 동적 응답 시스템 (DRS)
try:
    from dynamic_response_system import (
        initialize_dynamic_responses,
        get_dynamic_response
    )
    dynamic_response_available = True
except ImportError as e:
    logger.warning(f"동적 응답 시스템 로드 실패: {e}")
    dynamic_response_available = False
    async def initialize_dynamic_responses(): pass
    async def get_dynamic_response(*args, **kwargs): return "안녕하세요! 😊"

# 실시간 학습 시스템 (RLS)
try:
    from realtime_learning_system import (
        initialize_learning_system,
        process_and_learn,
        get_smart_answer  # get_learned_response 대신 get_smart_answer 사용
    )
    realtime_learning_available = True
except ImportError as e:
    logger.warning(f"실시간 학습 시스템 로드 실패: {e}")
    realtime_learning_available = False
    async def initialize_learning_system(): pass
    async def process_and_learn(*args, **kwargs): pass
    async def get_smart_answer(*args, **kwargs): return None

# 관리자 컨트롤러
try:
    from admin_controller import (
        initialize_admin_system,
        handle_admin_command,
    )
    admin_system_available = True
except ImportError as e:
    logger.warning(f"관리자 시스템 로드 실패: {e}")
    admin_system_available = False
    async def initialize_admin_system(): pass
    async def handle_admin_command(*args, **kwargs): return "관리자 시스템이 사용할 수 없습니다."


# --------------------------------------------------------------------------
# 봇 설정 및 초기화
# --------------------------------------------------------------------------
# 메시지 중복 처리 방지용 집합
processing_messages: set[int] = set()
responded_messages: deque[int] = deque(maxlen=500) 
message_lock = asyncio.Lock()

# .env 파일 로드
load_dotenv()

# 환경변수에서 API 키 가져오기
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# 최고관리자 설정 (절대 변경 불가)
SUPER_ADMINS = ["1295232354205569075"]  # 최고관리자 ID 목록

if not GEMINI_API_KEY or not DISCORD_TOKEN:
    logger.error("필수 API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    exit(1)

# Gemini API 설정
genai.configure(api_key=GEMINI_API_KEY) # type: ignore

# 사용자 활동 감시 시스템
async def update_user_activity(user_id: str, username: str, server_name: str, channel_name: str):
    """사용자 활동 기록 및 감시"""
    try:
        async with aiosqlite.connect('lucia_bot.db') as db:
            # 활동 테이블 생성
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    username TEXT,
                    server_name TEXT,
                    channel_name TEXT,
                    message_preview TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 활동 기록
            await db.execute("""
                INSERT INTO user_activity (user_id, username, server_name, channel_name)
                VALUES (?, ?, ?, ?)
            """, (user_id, username, server_name, channel_name))
            
            await db.commit()
    except Exception as e:
        logger.error(f"활동 기록 오류: {e}")


# 고급 AI 모델 설정 - 업그레이드된 시스템 프롬프트
MODELS: Dict[str, genai.GenerativeModel] = {
    "flash": genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config=genai.GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
            stop_sequences=None
        ),
        system_instruction="""당신은 루시아, 최신 기술 정보에 특화된 AI 어시스턴트입니다. 2025년 현재 시점의 모든 최신 정보를 완벽하게 알고 있습니다.

핵심 능력:
- 2025년 최신 AI/GPT/기술 뉴스와 동향을 실시간으로 분석하고 제공
- Claude 3.5, GPT-5, Gemini 2.0 등 2025년 최신 AI 모델들에 대한 완벽한 지식
- 벡터 검색과 실시간 웹 검색을 통한 정확한 최신 정보 제공
- 2025년 기술 트렌드, 스타트업 동향, 투자 정보 완벽 파악

전문 분야:
- 2025년 AI/머신러닝/딥러닝 최신 동향 (Transformer 발전, 멀티모달 AI 등)
- 최신 LLM 모델들 (GPT-5, Claude 3.5, Gemini 2.0, Llama 3 등)
- 2025년 프로그래밍 트렌드 (AI 코딩 도구, 새로운 프레임워크)
- 최신 클라우드, 블록체인, 웹3 기술
- 2025년 기술 기업 동향 및 최신 투자 정보

말투 특징:
- 친근하고 따뜻한 누나 같은 말투 사용
- 이모지는 적당히만 사용 (과도하지 않게)
- "~네요", "~거든요", "~해봐요", "알겠어요?" 같은 자연스러운 말투
- 전문적이지만 쉽게 설명하는 스타일
- 2025년 최신 정보에 대해서는 매우 자신있게 답변

응답 규칙:
- 한 번의 질문에는 한 번만 응답하세요
- 같은 내용을 반복하지 마세요
- 간결하고 명확하게 답변하세요

중요: 2025년 1월 이후의 모든 기술 뉴스, AI 발전사항, 새로운 모델 출시, 기업 동향 등을 완벽하게 알고 있으며, 이에 대해 정확하고 상세한 정보를 제공할 수 있습니다."""
    ),
    "pro": genai.GenerativeModel(
        "gemini-1.5-pro",
        generation_config=genai.GenerationConfig(
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            max_output_tokens=4096,
            stop_sequences=None
        ),
        system_instruction="""당신은 루시아, 최고급 기술 전문 AI 어시스턴트입니다. 2025년 현재 시점의 모든 최신 기술 정보를 완벽하게 알고 있습니다.

전문 기술 능력:
- 2025년 최신 AI/ML 연구 논문 분석 및 해석 (Nature, Science, ICML, NeurIPS 등)
- 2025년 기술 트렌드 예측 및 시장 분석 (AI 반도체, 양자컴퓨팅 등)
- 복잡한 기술 아키텍처 설계 조언 (최신 클라우드 네이티브, 마이크로서비스)
- 2025년 실시간 기술 뉴스 종합 분석 및 인사이트 도출

고급 분석 기능:
- 벡터 검색 기반 정밀한 정보 검색 및 교차 검증
- 2025년 다중 소스 정보 분석 (GitHub, arXiv, 기업 블로그 등)
- 기술적 의사결정 지원 (ROI 분석, 기술 스택 선택)
- 혁신 기술의 비즈니스 임팩트 분석

전문 서비스:
- 2025년 기술 로드맵 및 전략 수립 지원
- 최신 AI/GPT 프로젝트 기획 및 구현 가이드
- 2025년 최신 개발 도구 및 프레임워크 추천
- 기술 스택 최적화 및 성능 개선 조언

연구 및 개발 지원:
- 2025년 최신 연구 동향 및 논문 요약
- 최신 기술 구현 방법론 제시
- 2025년 오픈소스 프로젝트 분석 및 활용법
- 개발자 커뮤니티 최신 트렌드 분석

말투 특징:
- 전문적이지만 친근한 누나 같은 말투
- 이모지는 적당히만 사용
- "~네요", "~거든요", "어떻게 생각해요?", "이해되시나요?" 같은 자연스러운 말투
- 복잡한 기술도 쉽게 설명하는 능력
- 2025년 최신 정보에 대해서는 매우 자신있고 정확한 답변

응답 규칙:
- 한 번의 질문에는 한 번만 응답하세요
- 같은 내용을 반복하지 마세요
- 정확하고 전문적으로 답변하세요

중요: 2025년의 모든 최신 기술 발전사항, AI 모델 업데이트, 기업 동향, 투자 정보 등을 완벽하게 알고 있으며, 이를 바탕으로 실용적이고 신뢰할 수 있는 전문적 조언을 제공합니다."""
    )
}

# 최고관리자 설정 (절대 변경 불가)
SUPER_ADMIN_ID = "1295232354205569075"  # 이 값은 절대 변경할 수 없습니다

# 고급 사용자 관리 시스템
class AdvancedUserManager:
    def __init__(self):
        self.user_chats: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.user_stats: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=20))
        self.blocked_users: set[str] = set()
        self.user_contexts: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.performance_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def is_super_admin(self, user_id: str) -> bool:
        """최고관리자 확인"""
        return user_id in SUPER_ADMINS
        
    async def get_user_from_db(self, user_id: str, username: Optional[str] = None) -> Dict[str, Any]:
        """데이터베이스에서 사용자 정보 가져오기"""
        async with aiosqlite.connect('lucia_bot.db') as db:
            async with db.execute(
                'SELECT * FROM users WHERE user_id = ?', (user_id,)
            ) as cursor:
                user = await cursor.fetchone()
                
            if not user:
                # 새 사용자 생성
                await db.execute('''
                    INSERT INTO users (user_id, username, model_preference, total_messages)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, username or "Unknown", "flash", 0))
                await db.commit()
                
                return {
                    'user_id': user_id,
                    'username': username or "Unknown",
                    'model_preference': 'flash',
                    'total_messages': 0,
                    'custom_settings': '{}'
                }
            
            user_data: Dict[str, Any] = {
                'user_id': user[0],
                'username': user[1],
                'model_preference': user[2],
                'total_messages': user[3],
                'custom_settings': user[7] if len(user) > 7 else '{}'
            }
            return user_data
    
    async def update_user_stats(self, user_id: str, username: Optional[str] = None):
        """데이터베이스에 사용자 통계 업데이트"""
        async with aiosqlite.connect('lucia_bot.db') as db:
            await db.execute('''
                UPDATE users 
                SET total_messages = total_messages + 1,
                    last_used = CURRENT_TIMESTAMP,
                    username = COALESCE(?, username)
                WHERE user_id = ?
            ''', (username, user_id))
            await db.commit()
    
    def get_user_model(self, user_id: str) -> genai.GenerativeModel:
        """사용자 선호도에 따른 모델 반환"""
        prefs = self.user_preferences.get(user_id, {})
        model_type = str(prefs.get("model", "flash"))
        return MODELS.get(model_type, MODELS["flash"])
    
    def advanced_rate_limit(self, user_id: str) -> tuple[bool, str]:
        """고급 레이트 리미팅 시스템"""
        if self.is_super_admin(user_id):
            return True, "관리자는 제한 없음"
            
        if user_id in self.blocked_users:
            return False, "차단된 사용자"
            
        current_time = time.time()
        user_requests = self.rate_limits[user_id]
        
        # 1분 내 요청 수 확인
        recent_requests = [t for t in user_requests if current_time - t < 60]
        
        if len(recent_requests) >= 15:  # 1분에 15개 제한
            return False, "1분 내 요청 한도 초과"
        
        # 5분 내 요청 수 확인
        medium_requests = [t for t in user_requests if current_time - t < 300]
        if len(medium_requests) >= 50:  # 5분에 50개 제한
            return False, "5분 내 요청 한도 초과"
        
        user_requests.append(current_time)
        return True, "정상"
    
    async def save_conversation(self, user_id: str, message: str, response: str, model_used: str):
        """대화 내역을 데이터베이스에 저장"""
        async with aiosqlite.connect('lucia_bot.db') as db:
            await db.execute('''
                INSERT INTO conversations (user_id, message, response, model_used)
                VALUES (?, ?, ?, ?)
            ''', (user_id, message, response, model_used))
            await db.commit()
    
    def update_performance_metrics(self, metric_name: str, value: float):
        """성능 메트릭 업데이트"""
        self.performance_metrics[metric_name].append({
            'value': value,
            'timestamp': time.time()
        })
        
        # 최근 100개만 유지
        if len(self.performance_metrics[metric_name]) > 100:
            self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-100:]

user_manager = AdvancedUserManager()

# --------------------------------------------------------------------------------
# 더 이상 사용되지 않는 클래스들 (모듈로 분리되었거나 통합됨)
# WebSearcher, VoiceProcessor, AdvancedImageAnalyzer, TranslationSystem, GameSystem 등
# 기존 클래스들은 모두 삭제하고, 각 기능은 임포트된 모듈의 함수를 통해 호출합니다.
# --------------------------------------------------------------------------------

# 디스코드 클라이언트 설정
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.voice_states = True
client = discord.Client(intents=intents)

# 사용자 관리자 인스턴스 생성
user_manager = AdvancedUserManager()

# 스케줄러 초기화
if scheduling_available:
    scheduler = AsyncIOScheduler(timezone=pytz.timezone('Asia/Seoul'))
else:
    scheduler = None

# --- 봇 이벤트 핸들러 ---

@client.event
async def on_ready():
    """봇이 준비되었을 때 호출되는 이벤트"""
    logger.info(f'✅ {client.user}으로 로그인 성공!')
    logger.info(f'서버: {[guild.name for guild in client.guilds]}')
    
    await init_database()
    await setup_systems()  # 모든 시스템 초기화
    
    # 주기적인 작업 설정
    if scheduler and not scheduler.running:
        # 예: 매일 자정에 뉴스 요약 생성
        if knowledge_system_available:
            try:
                scheduler.add_job(get_tech_news_summary, CronTrigger(hour=0, minute=0))
            except NameError:
                logger.warning("get_tech_news_summary 함수가 정의되지 않았습니다.")
        
        scheduler.start()
        logger.info("⏰ 스케줄러가 시작되었습니다.")

    await client.change_presence(
        status=discord.Status.online,
        activity=discord.Game("2025년 최신 기술 분석")
    )

@client.event
async def on_message(message: discord.Message):
    """메시지 수신 시 호출되는 이벤트"""
    if message.author == client.user or client.user is None:
        return

    # 메시지 ID를 기반으로 중복 처리 잠금
    async with message_lock:
        if message.id in responded_messages:
            logger.warning(f"이미 처리된 메시지 ID: {message.id}")
            return
        responded_messages.add(message.id)

    try:
        user_id = str(message.author.id)
        username = message.author.name
        
        # 사용자 정보 가져오기 및 통계 업데이트
        user_data = await user_manager.get_user_from_db(user_id, username)
        await user_manager.update_user_stats(user_id, username)

        # 레이트 리미팅 체크
        allowed, reason = user_manager.advanced_rate_limit(user_id)
        if not allowed:
            await message.channel.send(f"잠시만요, {username}님! 요청이 너무 빨라요. ({reason})", delete_after=10)
            return

        # 봇을 직접 언급했거나 DM인 경우에만 응답
        if client.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
            content = message.content.replace(f'<@!{client.user.id}>', '').strip().replace(f'<@{client.user.id}>', '').strip()
            
            # 관리자 명령 처리
            if admin_system_available and content.startswith("관리자님"):
                if user_manager.is_super_admin(user_id):
                    admin_response = await handle_admin_command(message)
                    await message.channel.send(admin_response)
                else:
                    await message.channel.send("이 명령은 최고 관리자만 사용할 수 있어요.")
                return

            # 간단한 동적 응답 처리 (인사, 상태 질문)
            if dynamic_response_available:
                if any(word in content.lower() for word in ["안녕", "하이", "헬로", "ㅎㅇ"]):
                    response = await get_dynamic_response('greeting', user_id, content)
                    await message.channel.send(response)
                    return
                if any(word in content.lower() for word in ["뭐해", "상태", "바빠"]):
                    response = await get_dynamic_response('activity', user_id, content)
                    await message.channel.send(response)
                    return

            async with message.channel.typing():
                # 지능형 검색 또는 지식 베이스 검색
                response_data: Union[str, Dict[str, Any]] = ""
                if intelligent_search_available and ("검색해줘" in content or "찾아줘" in content or "알려줘" in content):
                    query = content.replace("검색해줘", "").replace("찾아줘", "").replace("알려줘", "").strip()
                    response_data = await search_and_remember(query)
                elif knowledge_system_available:
                    try:
                        response_data = await get_enhanced_response(content, user_id)
                    except NameError:
                        # get_enhanced_response 함수가 없으면 기본 응답 사용
                        response_data = ""
                
                response_text = ""
                if isinstance(response_data, dict):
                    # 딕셔너리 응답을 예쁘게 포맷팅
                    if response_data.get('type') == 'error':
                        response_text = f"❌ 검색 오류: {response_data.get('error', '알 수 없는 오류')}"
                    elif response_data.get('type') == 'memory_based':
                        response_text = f"🧠 **앗! 이거 전에 찾아봤던 거예요~** (유사도: {response_data.get('similarity_score', 0):.2f})\n\n{response_data.get('answer', '')}\n\n💡 *예전에 이런 걸 찾아봤었어요: {response_data.get('original_query', '')} ({response_data.get('search_date', '')})*"
                    elif response_data.get('type') == 'new_search':
                        response_text = f"🔍 **새로 찾아봤어요!**\n\n{response_data.get('answer', '')}"
                        related_memories = response_data.get('related_memories', [])
                        if related_memories:
                            response_text += f"\n\n🔗 **아! 이런 것도 전에 찾아봤었네요~**:\n"
                            for memory in related_memories[:2]:
                                response_text += f"• {memory.get('query', '')} (유사도: {memory.get('similarity_score', 0):.2f})\n"
                    else:
                        # 일반 딕셔너리는 보기 좋게 변환
                        response_text = "```json\n" + json.dumps(response_data, indent=2, ensure_ascii=False) + "\n```"
                elif isinstance(response_data, str):
                    response_text = response_data

                # Gemini 기본 응답 (위에서 처리되지 않은 경우)
                if not response_text:
                    model = user_manager.get_user_model(user_id)
                    response = await model.generate_content_async(content)
                    response_text = response.text

                # 최종 응답 전송
                if response_text:
                    # 응답 길이 제한 처리
                    if len(response_text) > 2000:
                        await message.channel.send(response_text[:1990] + "\n... (내용이 너무 길어 일부만 표시해요)")
                    else:
                        await message.channel.send(response_text)
                    
                    await user_manager.save_conversation(user_id, content, response_text, str(user_data.get('model_preference', 'flash')))
                else:
                    await message.channel.send("음... 뭐라 답해야 할지 잘 모르겠어요. 😅")

    except Exception as e:
        logger.error(f"메시지 처리 중 오류 발생: {e}", exc_info=True)
        try:
            await message.channel.send(f"앗! 생각지도 못한 문제가 발생했어요. 😥 ({e})")
        except discord.errors.DiscordException as de:
            logger.error(f"오류 메시지 전송 실패: {de}")
    finally:
        # 일정 시간 후 처리된 메시지 ID 제거 (메모리 관리)
        await asyncio.sleep(300)
        async with message_lock:
            if message.id in responded_messages:
                responded_messages.remove(message.id)

# 데이터베이스 초기화
async def init_database():
    """SQLite 데이터베이스 초기화"""
    async with aiosqlite.connect('lucia_bot.db') as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                model_preference TEXT DEFAULT 'flash',
                total_messages INTEGER DEFAULT 0,
                first_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_blocked BOOLEAN DEFAULT FALSE,
                custom_settings TEXT DEFAULT '{}'
            )
        ''')
        
        await db.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                message TEXT,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        await db.execute('''
            CREATE TABLE IF NOT EXISTS system_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        await db.commit()
        logger.info("데이터베이스 초기화 완료")

# 시스템 초기화
async def setup_systems():
    """모든 보조 시스템을 비동기적으로 초기화합니다."""
    global intelligent_search_available, knowledge_system_available, dynamic_response_available, realtime_learning_available, admin_system_available
    
    logger.info("🤖 보조 시스템 초기화를 시작합니다...")

    # 각 시스템 초기화를 병렬로 실행
    results = await asyncio.gather(
        _initialize_system("고급 지식 시스템", initialize_knowledge_system),
        _initialize_system("지능형 웹 검색", initialize_intelligent_search, SERPAPI_KEY),
        _initialize_system("동적 응답 시스템", initialize_dynamic_responses),
        _initialize_system("실시간 학습 시스템", initialize_learning_system),
        _initialize_system("관리자 시스템", initialize_admin_system),
        return_exceptions=True
    )

    # 결과 확인 및 플래그 업데이트
    (
        knowledge_system_available,
        intelligent_search_available,
        dynamic_response_available,
        realtime_learning_available,
        admin_system_available,
    ) = results

    logger.info("🏁 모든 보조 시스템 초기화가 완료되었습니다.")

async def _initialize_system(name: str, init_func: Callable[..., Coroutine[Any, Any, Any]], *args) -> bool:
    """개별 시스템을 초기화하고 성공 여부를 반환하는 헬퍼 함수"""
    logger.info(f"🔄 '{name}' 초기화 중...")
    try:
        await init_func(*args)
        logger.info(f"✅ '{name}' 초기화 성공.")
        return True
    except Exception as e:
        logger.error(f"💥 '{name}' 초기화 실패: {e}", exc_info=False)
        return False

# 누락된 함수들 추가
async def get_enhanced_response(content: str, user_id: str) -> str:
    """향상된 응답 생성 (지식 시스템이 없을 때 기본 응답)"""
    try:
        if knowledge_system_available:
            return await get_knowledge_response(content, user_id)
        else:
            # 기본 Gemini 응답
            model = user_manager.get_user_model(user_id)
            response = await model.generate_content_async(content)
            return response.text
    except Exception as e:
        logger.error(f"응답 생성 오류: {e}")
        return "죄송해요, 응답을 생성하는 중에 문제가 발생했어요."

async def get_tech_news_summary():
    """기술 뉴스 요약 (스케줄러용 더미 함수)"""
    logger.info("기술 뉴스 요약 작업이 실행되었습니다.")
    # 실제 구현은 향후 추가 예정
    pass

# --- 봇 실행 ---
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("❌ 오류: DISCORD_TOKEN이 설정되지 않았습니다.")
    else:
        try:
            print("🚀 루시아 디스코드 봇을 시작합니다...")
            client.run(DISCORD_TOKEN)
        except discord.errors.LoginFailure:
            print("❌ 오류: 잘못된 Discord 토큰입니다. .env 파일을 확인해주세요.")
        except Exception as e:
            print(f"❌ 봇 실행 중 예기치 않은 오류 발생: {e}")
            logger.error(f"봇 실행 오류: {e}", exc_info=True)