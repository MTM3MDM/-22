# 루시아 디스코드 봇 - 초고급 AI 어시스턴트 (완전 업그레이드)
# 음성인식, 이미지분석, 실시간번역, 게임시스템, 코드분석 등 초고급 기능 탑재

import discord
import google.generativeai as genai
import asyncio
import os
import time
import re
import json
import logging
import aiohttp
import aiosqlite
import hashlib
import random
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import base64
import sqlite3
from collections import defaultdict, deque
import psutil
import wikipedia
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import cv2
import speech_recognition as sr
import pyttsx3
from deep_translator import GoogleTranslator
import qrcode
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import tempfile
import subprocess
import ast
import pylint.lint
from pylint.reporters.text import TextReporter
from io import StringIO
import schedule
import threading
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from transformers import pipeline
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    face_recognition = None
import emoji
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

try:
    import cryptocompare
    CRYPTOCOMPARE_AVAILABLE = True
except ImportError:
    CRYPTOCOMPARE_AVAILABLE = False
    cryptocompare = None
import feedparser
import chess
import chess.engine
import chess.svg
try:
    from cairosvg import svg2png
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False
    svg2png = None



# 중복 응답 방지를 위한 메시지 추적 (강화)
processing_messages = set()
message_lock = asyncio.Lock()
responded_messages = set()  # 이미 응답한 메시지 추적

# 응답 캐시 시스템 (중복 응답 방지)
response_cache = {}
cache_lock = asyncio.Lock()

# 고급 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lucia_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



# 고급 지식 시스템 임포트
try:
    from advanced_knowledge_system import (
        initialize_knowledge_system,
        get_enhanced_response,
        get_tech_news_summary,
        search_knowledge_base
    )
    KNOWLEDGE_SYSTEM_AVAILABLE = True
    logger.info("고급 지식 시스템 모듈 로드 성공")
except ImportError as e:
    KNOWLEDGE_SYSTEM_AVAILABLE = False
    logger.warning(f"고급 지식 시스템 모듈 로드 실패: {e}")

# 고급 웹 검색 시스템 임포트
try:
    from advanced_web_search import (
        initialize_web_search,
        search_web,
        search_web_summary
    )
    WEB_SEARCH_AVAILABLE = True
    logger.info("고급 웹 검색 시스템 모듈 로드 성공")
except ImportError as e:
    WEB_SEARCH_AVAILABLE = False
    logger.warning(f"고급 웹 검색 시스템 모듈 로드 실패: {e}")

# 실시간 학습 시스템 임포트
try:
    from realtime_learning_system import (
        initialize_learning_system,
        get_smart_answer
    )
    REALTIME_LEARNING_AVAILABLE = True
    logger.info("실시간 학습 시스템 모듈 로드 성공")
except ImportError as e:
    REALTIME_LEARNING_AVAILABLE = False
    logger.warning(f"실시간 학습 시스템 모듈 로드 실패: {e}")

# .env 파일 로드
load_dotenv()

# 환경변수에서 API 키 가져오기
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# 최고관리자 설정 (절대 변경 불가)
SUPER_ADMINS = ["1295232354205569075"]  # 최고관리자 ID 목록

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")  # 웹 검색용 (선택적)

if not GEMINI_API_KEY or not DISCORD_TOKEN:
    logger.error("필수 API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    exit(1)

# Gemini API 설정
genai.configure(api_key=GEMINI_API_KEY)

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
MODELS = {
    "flash": genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config=genai.types.GenerationConfig(
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
        generation_config=genai.types.GenerationConfig(
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
        self.user_chats: Dict[str, any] = {}
        self.user_preferences: Dict[str, dict] = {}
        self.user_stats: Dict[str, dict] = {}
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.blocked_users: set = set()
        self.user_contexts: Dict[str, dict] = defaultdict(dict)
        self.performance_metrics = defaultdict(list)
        
    def is_super_admin(self, user_id: str) -> bool:
        """최고관리자 확인"""
        return user_id == SUPER_ADMIN_ID
        
    async def get_user_from_db(self, user_id: str, username: str = None) -> dict:
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
            
            return {
                'user_id': user[0],
                'username': user[1],
                'model_preference': user[2],
                'total_messages': user[3],
                'custom_settings': user[7] if len(user) > 7 else '{}'
            }
    
    async def update_user_stats(self, user_id: str, username: str = None):
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
        model_type = prefs.get("model", "flash")
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

# 고급 기능 클래스들
class WebSearcher:
    """고급 웹 검색 기능 - SerpAPI 및 기억 기능 포함"""
    
    def __init__(self):
        self.intelligent_searcher = None
    
    async def initialize_intelligent_search(self, serpapi_key: str = "YOUR_SERPAPI_KEY_HERE"):
        """지능형 검색 시스템 초기화"""
        try:
            from intelligent_web_search import initialize_web_search
            self.intelligent_searcher = await initialize_web_search(serpapi_key)
            logger.info("지능형 웹 검색 시스템 초기화 완료")
        except Exception as e:
            logger.error(f"지능형 검색 초기화 오류: {e}")
    
    @staticmethod
    async def search_wikipedia(query: str, lang: str = 'ko') -> str:
        """위키피디아 검색"""
        try:
            wikipedia.set_lang(lang)
            summary = wikipedia.summary(query, sentences=3)
            return f"📚 위키피디아 검색 결과:\n{summary}"
        except Exception as e:
            return f"위키피디아 검색 중 오류가 발생했습니다: {str(e)}"
    
    async def intelligent_web_search(self, query: str) -> str:
        """지능형 웹 검색 - 기억 기능 포함"""
        try:
            if self.intelligent_searcher is None:
                return "🔍 지능형 검색 시스템이 초기화되지 않았습니다. 관리자에게 문의하세요."
            
            from intelligent_web_search import search_and_remember
            result = await search_and_remember(query)
            
            if result.get('type') == 'error':
                return f"❌ 검색 오류: {result.get('error', '알 수 없는 오류')}"
            
            elif result.get('type') == 'memory_based':
                return f"🧠 **앗! 이거 전에 찾아봤던 거예요~** (유사도: {result.get('similarity_score', 0):.2f})\n\n{result.get('answer', '')}\n\n💡 *예전에 이런 걸 찾아봤었어요: {result.get('original_query', '')} ({result.get('search_date', '')})*"
            
            elif result.get('type') == 'new_search':
                response = f"🔍 **새로 찾아봤어요!**\n\n{result.get('answer', '')}"
                
                # 관련 기억이 있으면 추가
                related_memories = result.get('related_memories', [])
                if related_memories:
                    response += f"\n\n🔗 **아! 이런 것도 전에 찾아봤었네요~**:\n"
                    for memory in related_memories[:2]:
                        response += f"• {memory.get('query', '')} (유사도: {memory.get('similarity_score', 0):.2f})\n"
                
                return response
            
            else:
                return "🤔 어라? 검색 결과를 어떻게 보여드려야 할지 모르겠어요..."
                
        except Exception as e:
            logger.error(f"지능형 웹 검색 오류: {e}")
            return f"❌ 앗! 검색하다가 문제가 생겼어요: {str(e)}"
    
    async def web_search(self, query: str) -> str:
        """일반 웹 검색 (기존 호환성 유지)"""
        # 지능형 검색이 가능하면 사용, 아니면 기본 검색
        if self.intelligent_searcher is not None:
            return await self.intelligent_web_search(query)
        else:
            return await self._basic_web_search(query)
    
    async def _basic_web_search(self, query: str) -> str:
        """기본 웹 검색 (백업용)"""
        try:
            async with aiohttp.ClientSession() as session:
                search_url = f"https://www.google.com/search?q={query}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                
                async with session.get(search_url, headers=headers) as response:
                    if response.status == 200:
                        return f"🔍 '{query}'에 대한 기본 웹 검색을 수행했습니다. 더 자세한 정보가 필요하시면 구체적으로 질문해주세요!"
                    else:
                        return "웹 검색 중 문제가 발생했습니다."
        except Exception as e:
            return f"웹 검색 오류: {str(e)}"
    
    async def get_search_stats(self) -> str:
        """검색 통계 조회"""
        try:
            if self.intelligent_searcher is None:
                return "📊 지능형 검색 시스템이 초기화되지 않았습니다."
            
            from intelligent_web_search import get_search_statistics
            stats = await get_search_statistics()
            
            if not stats:
                return "📊 검색 통계를 가져올 수 없습니다."
            
            return f"""📊 **검색 시스템 통계**

🔍 총 검색 수: {stats.get('total_searches', 0)}개
📅 최근 24시간: {stats.get('recent_searches', 0)}개
⚡ 평균 응답 시간: {stats.get('avg_response_time', 0)}초
🧠 기억 저장소: {stats.get('memory_size', 0)}개
💾 캐시 크기: {stats.get('cache_size', 0)}개"""
            
        except Exception as e:
            logger.error(f"검색 통계 조회 오류: {e}")
            return f"❌ 통계 조회 오류: {str(e)}"

# WebSearcher 인스턴스 생성
web_searcher = WebSearcher()

class ImageGenerator:
    """이미지 생성 및 처리"""
    
    @staticmethod
    async def create_simple_chart(data: dict, title: str = "차트") -> io.BytesIO:
        """간단한 차트 생성"""
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(data.keys(), data.values())
            plt.title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            return buffer
        except Exception as e:
            logger.error(f"차트 생성 오류: {e}")
            return None
    
    @staticmethod
    async def create_status_image(stats: dict) -> io.BytesIO:
        """시스템 상태 이미지 생성"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # CPU 사용률
            cpu_percent = psutil.cpu_percent()
            ax1.pie([cpu_percent, 100-cpu_percent], labels=['사용중', '여유'], autopct='%1.1f%%')
            ax1.set_title('CPU 사용률')
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            ax2.pie([memory.percent, 100-memory.percent], labels=['사용중', '여유'], autopct='%1.1f%%')
            ax2.set_title('메모리 사용률')
            
            # 사용자 통계 (예시)
            user_data = stats.get('users', {'활성': 10, '비활성': 5})
            ax3.bar(user_data.keys(), user_data.values())
            ax3.set_title('사용자 통계')
            
            # 메시지 통계 (예시)
            msg_data = stats.get('messages', {'오늘': 50, '어제': 30, '그제': 20})
            ax4.plot(list(msg_data.keys()), list(msg_data.values()), marker='o')
            ax4.set_title('메시지 통계')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            return buffer
        except Exception as e:
            logger.error(f"상태 이미지 생성 오류: {e}")
            return None

class SystemMonitor:
    """시스템 모니터링"""
    
    @staticmethod
    def get_system_info() -> dict:
        """시스템 정보 수집"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).strftime('%Y-%m-%d %H:%M:%S'),
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"시스템 정보 수집 오류: {e}")
            return {}

# ===== 초고급 기능 클래스들 =====

class VoiceProcessor:
    """음성 인식 및 TTS 시스템"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.setup_tts()
    
    def setup_tts(self):
        """TTS 엔진 설정"""
        try:
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # 한국어 음성 선택 (가능한 경우)
                for voice in voices:
                    if 'korean' in voice.name.lower() or 'ko' in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.tts_engine.setProperty('rate', 150)  # 말하기 속도
            self.tts_engine.setProperty('volume', 0.8)  # 볼륨
        except Exception as e:
            logger.error(f"TTS 설정 오류: {e}")
    
    async def speech_to_text(self, audio_file_path: str) -> str:
        """음성을 텍스트로 변환"""
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language='ko-KR')
                return text
        except sr.UnknownValueError:
            return "음성을 인식할 수 없습니다."
        except sr.RequestError as e:
            logger.error(f"음성 인식 서비스 오류: {e}")
            return "음성 인식 서비스에 문제가 발생했습니다."
        except Exception as e:
            logger.error(f"음성 처리 오류: {e}")
            return "음성 처리 중 오류가 발생했습니다."
    
    async def text_to_speech(self, text: str) -> io.BytesIO:
        """텍스트를 음성으로 변환"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                self.tts_engine.save_to_file(text, temp_file.name)
                self.tts_engine.runAndWait()
                
                with open(temp_file.name, 'rb') as f:
                    audio_data = io.BytesIO(f.read())
                
                os.unlink(temp_file.name)
                return audio_data
        except Exception as e:
            logger.error(f"TTS 변환 오류: {e}")
            return None

class AdvancedImageAnalyzer:
    """고급 이미지 분석 및 생성 시스템"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.setup_models()
    
    def setup_models(self):
        """AI 모델 설정"""
        try:
            # 이미지 분류 모델 로드
            self.image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
            self.object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
        except Exception as e:
            logger.error(f"이미지 분석 모델 로드 오류: {e}")
            self.image_classifier = None
            self.object_detector = None
    
    async def analyze_image(self, image_path: str) -> dict:
        """종합적인 이미지 분석"""
        try:
            image = cv2.imread(image_path)
            pil_image = Image.open(image_path)
            
            analysis = {
                'basic_info': self.get_basic_info(image),
                'faces': self.detect_faces(image),
                'colors': self.analyze_colors(pil_image),
                'objects': await self.detect_objects(pil_image),
                'classification': await self.classify_image(pil_image),
                'text': self.extract_text(image),
                'quality': self.assess_quality(image)
            }
            
            return analysis
        except Exception as e:
            logger.error(f"이미지 분석 오류: {e}")
            return {"error": str(e)}
    
    def get_basic_info(self, image) -> dict:
        """기본 이미지 정보"""
        height, width, channels = image.shape
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'size': f"{width}x{height}",
            'aspect_ratio': round(width/height, 2)
        }
    
    def detect_faces(self, image) -> dict:
        """얼굴 감지"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            face_info = []
            for (x, y, w, h) in faces:
                face_info.append({
                    'position': {'x': int(x), 'y': int(y)},
                    'size': {'width': int(w), 'height': int(h)},
                    'confidence': 0.8  # 기본값
                })
            
            return {
                'count': len(faces),
                'faces': face_info
            }
        except Exception as e:
            logger.error(f"얼굴 감지 오류: {e}")
            return {'count': 0, 'faces': []}
    
    def analyze_colors(self, image: Image.Image) -> dict:
        """색상 분석"""
        try:
            # 주요 색상 추출
            colors = image.getcolors(maxcolors=256*256*256)
            if colors:
                # 가장 많이 사용된 색상들
                dominant_colors = sorted(colors, key=lambda x: x[0], reverse=True)[:5]
                color_info = []
                
                for count, color in dominant_colors:
                    if isinstance(color, tuple) and len(color) >= 3:
                        color_info.append({
                            'rgb': color[:3],
                            'hex': '#{:02x}{:02x}{:02x}'.format(*color[:3]),
                            'percentage': round((count / sum(c[0] for c in colors)) * 100, 2)
                        })
                
                return {
                    'dominant_colors': color_info,
                    'total_colors': len(colors)
                }
        except Exception as e:
            logger.error(f"색상 분석 오류: {e}")
        
        return {'dominant_colors': [], 'total_colors': 0}
    
    async def detect_objects(self, image: Image.Image) -> list:
        """객체 감지"""
        try:
            if self.object_detector:
                results = self.object_detector(image)
                objects = []
                for result in results:
                    objects.append({
                        'label': result['label'],
                        'confidence': round(result['score'], 3),
                        'box': result['box']
                    })
                return objects
        except Exception as e:
            logger.error(f"객체 감지 오류: {e}")
        
        return []
    
    async def classify_image(self, image: Image.Image) -> list:
        """이미지 분류"""
        try:
            if self.image_classifier:
                results = self.image_classifier(image)
                classifications = []
                for result in results[:3]:  # 상위 3개만
                    classifications.append({
                        'label': result['label'],
                        'confidence': round(result['score'], 3)
                    })
                return classifications
        except Exception as e:
            logger.error(f"이미지 분류 오류: {e}")
        
        return []
    
    def extract_text(self, image) -> str:
        """이미지에서 텍스트 추출 (OCR)"""
        try:
            # pytesseract가 설치되어 있다면 사용
            import pytesseract
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, lang='kor+eng')
            return text.strip()
        except ImportError:
            return "OCR 라이브러리가 설치되지 않았습니다."
        except Exception as e:
            logger.error(f"텍스트 추출 오류: {e}")
            return "텍스트 추출 실패"
    
    def assess_quality(self, image) -> dict:
        """이미지 품질 평가"""
        try:
            # 블러 정도 측정
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 밝기 분석
            brightness = np.mean(gray)
            
            # 대비 분석
            contrast = gray.std()
            
            return {
                'blur_score': round(blur_score, 2),
                'blur_level': 'sharp' if blur_score > 100 else 'blurry',
                'brightness': round(brightness, 2),
                'contrast': round(contrast, 2),
                'overall_quality': 'good' if blur_score > 100 and 50 < brightness < 200 else 'poor'
            }
        except Exception as e:
            logger.error(f"품질 평가 오류: {e}")
            return {'error': str(e)}
    
    async def generate_enhanced_image(self, image_path: str, enhancement_type: str) -> io.BytesIO:
        """이미지 향상 및 필터 적용"""
        try:
            image = Image.open(image_path)
            
            if enhancement_type == 'sharpen':
                image = image.filter(ImageFilter.SHARPEN)
            elif enhancement_type == 'blur':
                image = image.filter(ImageFilter.BLUR)
            elif enhancement_type == 'enhance_contrast':
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
            elif enhancement_type == 'enhance_brightness':
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.2)
            elif enhancement_type == 'enhance_color':
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.3)
            
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            return buffer
        except Exception as e:
            logger.error(f"이미지 향상 오류: {e}")
            return None

class TranslationSystem:
    """실시간 번역 시스템"""
    
    def __init__(self):
        self.translator = GoogleTranslator()
        self.supported_languages = {
            'ko': '한국어', 'en': '영어', 'ja': '일본어', 'zh': '중국어',
            'es': '스페인어', 'fr': '프랑스어', 'de': '독일어', 'ru': '러시아어',
            'it': '이탈리아어', 'pt': '포르투갈어', 'ar': '아랍어', 'hi': '힌디어'
        }
    
    async def translate_text(self, text: str, target_lang: str = 'en', source_lang: str = 'auto') -> dict:
        """텍스트 번역"""
        try:
            result = self.translator.translate(text, dest=target_lang, src=source_lang)
            
            return {
                'original_text': text,
                'translated_text': result.text,
                'source_language': result.src,
                'target_language': target_lang,
                'confidence': getattr(result, 'confidence', 0.9)
            }
        except Exception as e:
            logger.error(f"번역 오류: {e}")
            return {
                'error': str(e),
                'original_text': text
            }
    
    async def detect_language(self, text: str) -> dict:
        """언어 감지"""
        try:
            detection = self.translator.detect(text)
            return {
                'language': detection.lang,
                'language_name': self.supported_languages.get(detection.lang, detection.lang),
                'confidence': detection.confidence
            }
        except Exception as e:
            logger.error(f"언어 감지 오류: {e}")
            return {'error': str(e)}
    
    async def multi_translate(self, text: str, target_languages: list) -> dict:
        """다중 언어 번역"""
        results = {}
        for lang in target_languages:
            translation = await self.translate_text(text, lang)
            results[lang] = translation
        return results

class GameSystem:
    """고급 게임 시스템"""
    
    def __init__(self):
        self.active_games = {}
        self.game_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
        self.chess_boards = {}
    
    async def start_chess_game(self, user_id: str, opponent_id: str = None) -> dict:
        """체스 게임 시작"""
        try:
            board = chess.Board()
            game_id = f"{user_id}_{opponent_id or 'ai'}_{int(time.time())}"
            
            self.chess_boards[game_id] = {
                'board': board,
                'white_player': user_id,
                'black_player': opponent_id or 'ai',
                'current_turn': 'white',
                'start_time': datetime.now()
            }
            
            # 체스판 이미지 생성
            svg_board = chess.svg.board(board=board)
            png_board = svg2png(bytestring=svg_board.encode('utf-8'))
            
            return {
                'game_id': game_id,
                'board_image': io.BytesIO(png_board),
                'message': f"체스 게임이 시작되었습니다! 게임 ID: {game_id}",
                'current_turn': 'white'
            }
        except Exception as e:
            logger.error(f"체스 게임 시작 오류: {e}")
            return {'error': str(e)}
    
    async def make_chess_move(self, game_id: str, move: str, user_id: str) -> dict:
        """체스 수 두기"""
        try:
            if game_id not in self.chess_boards:
                return {'error': '게임을 찾을 수 없습니다.'}
            
            game = self.chess_boards[game_id]
            board = game['board']
            
            # 수 검증 및 실행
            try:
                chess_move = chess.Move.from_uci(move)
                if chess_move in board.legal_moves:
                    board.push(chess_move)
                    
                    # 게임 상태 확인
                    status = self.check_chess_game_status(board)
                    
                    # 체스판 이미지 생성
                    svg_board = chess.svg.board(board=board, lastmove=chess_move)
                    png_board = svg2png(bytestring=svg_board.encode('utf-8'))
                    
                    return {
                        'success': True,
                        'board_image': io.BytesIO(png_board),
                        'move': move,
                        'status': status,
                        'fen': board.fen()
                    }
                else:
                    return {'error': '유효하지 않은 수입니다.'}
            except ValueError:
                return {'error': '잘못된 수 형식입니다. (예: e2e4)'}
                
        except Exception as e:
            logger.error(f"체스 수 두기 오류: {e}")
            return {'error': str(e)}
    
    def check_chess_game_status(self, board: chess.Board) -> dict:
        """체스 게임 상태 확인"""
        if board.is_checkmate():
            winner = 'black' if board.turn == chess.WHITE else 'white'
            return {'game_over': True, 'result': 'checkmate', 'winner': winner}
        elif board.is_stalemate():
            return {'game_over': True, 'result': 'stalemate'}
        elif board.is_insufficient_material():
            return {'game_over': True, 'result': 'insufficient_material'}
        elif board.is_check():
            return {'game_over': False, 'check': True}
        else:
            return {'game_over': False}
    
    async def play_trivia_game(self, category: str = 'general') -> dict:
        """퀴즈 게임"""
        try:
            # 간단한 퀴즈 데이터베이스
            trivia_questions = {
                'general': [
                    {'question': '대한민국의 수도는?', 'answer': '서울', 'options': ['서울', '부산', '대구', '인천']},
                    {'question': '1 + 1은?', 'answer': '2', 'options': ['1', '2', '3', '4']},
                    {'question': '지구에서 가장 큰 대륙은?', 'answer': '아시아', 'options': ['아시아', '아프리카', '유럽', '북미']},
                ],
                'science': [
                    {'question': '물의 화학식은?', 'answer': 'H2O', 'options': ['H2O', 'CO2', 'O2', 'N2']},
                    {'question': '빛의 속도는 약 몇 km/s인가?', 'answer': '300,000', 'options': ['300,000', '150,000', '450,000', '600,000']},
                ]
            }
            
            questions = trivia_questions.get(category, trivia_questions['general'])
            question = random.choice(questions)
            
            return {
                'question': question['question'],
                'options': question['options'],
                'correct_answer': question['answer'],
                'category': category
            }
        except Exception as e:
            logger.error(f"퀴즈 게임 오류: {e}")
            return {'error': str(e)}
    
    async def word_chain_game(self, user_id: str, word: str) -> dict:
        """끝말잇기 게임"""
        try:
            if user_id not in self.active_games:
                self.active_games[user_id] = {
                    'type': 'word_chain',
                    'used_words': [],
                    'last_word': None,
                    'score': 0
                }
            
            game = self.active_games[user_id]
            
            # 첫 단어인 경우
            if not game['last_word']:
                game['last_word'] = word
                game['used_words'].append(word)
                return {
                    'success': True,
                    'message': f"'{word}'로 시작합니다! 다음 단어를 말해주세요.",
                    'score': game['score']
                }
            
            # 끝말잇기 규칙 검증
            last_char = game['last_word'][-1]
            first_char = word[0]
            
            if last_char != first_char:
                return {
                    'success': False,
                    'message': f"'{last_char}'로 시작하는 단어를 말해주세요!",
                    'score': game['score']
                }
            
            if word in game['used_words']:
                return {
                    'success': False,
                    'message': "이미 사용한 단어입니다!",
                    'score': game['score']
                }
            
            # 성공
            game['last_word'] = word
            game['used_words'].append(word)
            game['score'] += 1
            
            # AI 응답 생성
            ai_word = await self.generate_word_chain_response(word)
            if ai_word:
                game['last_word'] = ai_word
                game['used_words'].append(ai_word)
                
                return {
                    'success': True,
                    'message': f"좋습니다! 제가 '{ai_word}'로 이어가겠습니다.",
                    'ai_word': ai_word,
                    'score': game['score']
                }
            else:
                return {
                    'success': True,
                    'message': f"훌륭합니다! '{word[-1]}'로 시작하는 단어를 찾지 못했네요. 당신이 이겼습니다!",
                    'game_over': True,
                    'final_score': game['score']
                }
                
        except Exception as e:
            logger.error(f"끝말잇기 게임 오류: {e}")
            return {'error': str(e)}
    
    async def generate_word_chain_response(self, word: str) -> str:
        """끝말잇기 AI 응답 생성"""
        try:
            # 간단한 단어 데이터베이스
            word_db = {
                '가': ['가방', '가족', '가을', '가수'],
                '나': ['나무', '나비', '나라', '나침반'],
                '다': ['다리', '달', '닭', '단어'],
                '라': ['라면', '라디오', '라이터', '라벨'],
                '마': ['마음', '마우스', '마법', '마당'],
                '바': ['바다', '바람', '바나나', '바지'],
                '사': ['사과', '사람', '사자', '사진'],
                '아': ['아기', '아침', '아버지', '아이'],
                '자': ['자동차', '자전거', '자리', '자석'],
                '차': ['차', '책', '창문', '천장'],
                '카': ['카메라', '카드', '카페', '카레'],
                '타': ['타이어', '탁자', '태양', '터널'],
                '파': ['파도', '파일', '팬', '팔'],
                '하': ['하늘', '학교', '한국', '할머니']
            }
            
            last_char = word[-1]
            possible_words = word_db.get(last_char, [])
            
            if possible_words:
                return random.choice(possible_words)
            return None
            
        except Exception as e:
            logger.error(f"끝말잇기 응답 생성 오류: {e}")
            return None

class CodeAnalyzer:
    """AI 기반 코드 분석 및 생성 시스템"""
    
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'java', 'cpp', 'c', 'html', 'css', 'sql']
    
    async def analyze_code(self, code: str, language: str = 'python') -> dict:
        """코드 분석"""
        try:
            analysis = {
                'language': language,
                'line_count': len(code.split('\n')),
                'character_count': len(code),
                'syntax_check': await self.check_syntax(code, language),
                'complexity': await self.analyze_complexity(code, language),
                'suggestions': await self.get_improvement_suggestions(code, language),
                'security_issues': await self.check_security_issues(code, language)
            }
            
            return analysis
        except Exception as e:
            logger.error(f"코드 분석 오류: {e}")
            return {'error': str(e)}
    
    async def check_syntax(self, code: str, language: str) -> dict:
        """구문 검사"""
        try:
            if language == 'python':
                try:
                    ast.parse(code)
                    return {'valid': True, 'message': '구문이 올바릅니다.'}
                except SyntaxError as e:
                    return {
                        'valid': False,
                        'error': str(e),
                        'line': e.lineno,
                        'column': e.offset
                    }
            else:
                return {'valid': True, 'message': f'{language} 구문 검사는 지원되지 않습니다.'}
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def analyze_complexity(self, code: str, language: str) -> dict:
        """코드 복잡도 분석"""
        try:
            if language == 'python':
                # 간단한 복잡도 측정
                lines = code.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                
                # 제어 구조 카운트
                control_structures = 0
                for line in non_empty_lines:
                    if any(keyword in line for keyword in ['if', 'for', 'while', 'try', 'except']):
                        control_structures += 1
                
                # 함수 카운트
                function_count = len([line for line in non_empty_lines if line.strip().startswith('def ')])
                
                # 클래스 카운트
                class_count = len([line for line in non_empty_lines if line.strip().startswith('class ')])
                
                complexity_score = control_structures + function_count * 2 + class_count * 3
                
                return {
                    'lines_of_code': len(non_empty_lines),
                    'control_structures': control_structures,
                    'functions': function_count,
                    'classes': class_count,
                    'complexity_score': complexity_score,
                    'complexity_level': 'low' if complexity_score < 10 else 'medium' if complexity_score < 25 else 'high'
                }
            else:
                return {'message': f'{language} 복잡도 분석은 지원되지 않습니다.'}
        except Exception as e:
            return {'error': str(e)}
    
    async def get_improvement_suggestions(self, code: str, language: str) -> list:
        """코드 개선 제안"""
        suggestions = []
        
        try:
            if language == 'python':
                lines = code.split('\n')
                
                # 기본적인 개선 제안들
                for i, line in enumerate(lines, 1):
                    line_stripped = line.strip()
                    
                    # 긴 줄 검사
                    if len(line) > 100:
                        suggestions.append(f"라인 {i}: 줄이 너무 깁니다. (100자 초과)")
                    
                    # 하드코딩된 값 검사
                    if re.search(r'\b\d{3,}\b', line_stripped):
                        suggestions.append(f"라인 {i}: 하드코딩된 숫자를 상수로 정의하는 것을 고려해보세요.")
                    
                    # TODO/FIXME 주석 검사
                    if 'TODO' in line_stripped or 'FIXME' in line_stripped:
                        suggestions.append(f"라인 {i}: TODO/FIXME 주석이 있습니다.")
                    
                    # 빈 except 블록 검사
                    if line_stripped == 'except:' or line_stripped == 'except Exception:':
                        suggestions.append(f"라인 {i}: 구체적인 예외 타입을 지정하세요.")
                
                # 전체 코드 구조 제안
                if 'import *' in code:
                    suggestions.append("와일드카드 import(import *)는 피하는 것이 좋습니다.")
                
                if code.count('print(') > 5:
                    suggestions.append("많은 print 문이 있습니다. 로깅 시스템 사용을 고려해보세요.")
        
        except Exception as e:
            suggestions.append(f"분석 중 오류 발생: {str(e)}")
        
        return suggestions
    
    async def check_security_issues(self, code: str, language: str) -> list:
        """보안 이슈 검사"""
        security_issues = []
        
        try:
            if language == 'python':
                # 위험한 함수들 검사
                dangerous_functions = ['eval', 'exec', 'input', '__import__']
                for func in dangerous_functions:
                    if f'{func}(' in code:
                        security_issues.append(f"위험한 함수 '{func}' 사용 감지")
                
                # SQL 인젝션 위험 검사
                if 'execute(' in code and '%' in code:
                    security_issues.append("SQL 인젝션 위험: 문자열 포맷팅 대신 매개변수화된 쿼리를 사용하세요.")
                
                # 하드코딩된 비밀번호/키 검사
                if re.search(r'(password|key|secret)\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
                    security_issues.append("하드코딩된 비밀번호나 키가 감지되었습니다.")
        
        except Exception as e:
            security_issues.append(f"보안 검사 중 오류 발생: {str(e)}")
        
        return security_issues
    
    async def generate_code(self, description: str, language: str = 'python') -> dict:
        """코드 생성"""
        try:
            # 간단한 코드 템플릿들
            templates = {
                'python': {
                    'hello_world': 'print("Hello, World!")',
                    'function': '''def my_function(param):
    """함수 설명"""
    return param * 2''',
                    'class': '''class MyClass:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value''',
                    'file_read': '''with open('filename.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    print(content)''',
                    'api_request': '''import requests

response = requests.get('https://api.example.com/data')
if response.status_code == 200:
    data = response.json()
    print(data)'''
                }
            }
            
            # 키워드 기반 템플릿 선택
            description_lower = description.lower()
            
            if 'hello' in description_lower or '헬로' in description_lower:
                template_key = 'hello_world'
            elif 'function' in description_lower or '함수' in description_lower:
                template_key = 'function'
            elif 'class' in description_lower or '클래스' in description_lower:
                template_key = 'class'
            elif 'file' in description_lower or '파일' in description_lower:
                template_key = 'file_read'
            elif 'api' in description_lower or 'request' in description_lower:
                template_key = 'api_request'
            else:
                template_key = 'hello_world'
            
            code = templates.get(language, {}).get(template_key, f'# {description}에 대한 코드를 생성할 수 없습니다.')
            
            return {
                'generated_code': code,
                'language': language,
                'description': description,
                'template_used': template_key
            }
        except Exception as e:
            logger.error(f"코드 생성 오류: {e}")
            return {'error': str(e)}

class DataAnalyzer:
    """고급 데이터 분석 및 시각화 시스템"""
    
    def __init__(self):
        self.setup_plotting()
    
    def setup_plotting(self):
        """플롯 설정"""
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
    
    async def analyze_data(self, data: Union[dict, list], analysis_type: str = 'basic') -> dict:
        """데이터 분석"""
        try:
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                if all(isinstance(item, dict) for item in data):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame({'values': data})
            else:
                return {'error': '지원되지 않는 데이터 형식입니다.'}
            
            analysis = {
                'basic_stats': self.get_basic_statistics(df),
                'data_types': self.get_data_types(df),
                'missing_values': self.check_missing_values(df),
                'correlations': self.calculate_correlations(df) if analysis_type == 'advanced' else None
            }
            
            return analysis
        except Exception as e:
            logger.error(f"데이터 분석 오류: {e}")
            return {'error': str(e)}
    
    def get_basic_statistics(self, df: pd.DataFrame) -> dict:
        """기본 통계"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats = df[numeric_cols].describe().to_dict()
                return {
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'numeric_columns': len(numeric_cols),
                    'statistics': stats
                }
            else:
                return {
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'numeric_columns': 0,
                    'message': '숫자형 데이터가 없습니다.'
                }
        except Exception as e:
            return {'error': str(e)}
    
    def get_data_types(self, df: pd.DataFrame) -> dict:
        """데이터 타입 분석"""
        try:
            return {
                'dtypes': df.dtypes.astype(str).to_dict(),
                'memory_usage': df.memory_usage(deep=True).to_dict()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def check_missing_values(self, df: pd.DataFrame) -> dict:
        """결측값 검사"""
        try:
            missing = df.isnull().sum()
            return {
                'missing_count': missing.to_dict(),
                'missing_percentage': (missing / len(df) * 100).to_dict(),
                'total_missing': missing.sum()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_correlations(self, df: pd.DataFrame) -> dict:
        """상관관계 분석"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                return {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'strong_correlations': self.find_strong_correlations(corr_matrix)
                }
            else:
                return {'message': '상관관계 분석을 위한 충분한 숫자형 컬럼이 없습니다.'}
        except Exception as e:
            return {'error': str(e)}
    
    def find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> list:
        """강한 상관관계 찾기"""
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    strong_corrs.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': round(corr_value, 3)
                    })
        return strong_corrs
    
    async def create_visualization(self, data: Union[dict, list], chart_type: str = 'bar') -> io.BytesIO:
        """데이터 시각화"""
        try:
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                if all(isinstance(item, dict) for item in data):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame({'values': data})
            
            plt.figure(figsize=(10, 6))
            
            if chart_type == 'bar':
                if len(df.columns) == 1:
                    df.iloc[:, 0].plot(kind='bar')
                else:
                    df.plot(kind='bar')
            elif chart_type == 'line':
                df.plot(kind='line', marker='o')
            elif chart_type == 'scatter':
                if len(df.columns) >= 2:
                    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
                    plt.xlabel(df.columns[0])
                    plt.ylabel(df.columns[1])
            elif chart_type == 'histogram':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df[numeric_cols[0]].hist(bins=20)
            elif chart_type == 'box':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df[numeric_cols].boxplot()
            
            plt.title(f'{chart_type.title()} Chart')
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            return buffer
        except Exception as e:
            logger.error(f"시각화 생성 오류: {e}")
            return None
    
    async def generate_word_cloud(self, text: str) -> io.BytesIO:
        """워드클라우드 생성"""
        try:
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                font_path=None,  # 시스템 폰트 사용
                max_words=100,
                colormap='viridis'
            ).generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            return buffer
        except Exception as e:
            logger.error(f"워드클라우드 생성 오류: {e}")
            return None

class SmartScheduler:
    """스마트 알림 및 스케줄링 시스템"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.reminders = {}
        self.recurring_tasks = {}
        # 스케줄러는 나중에 시작
    
    async def set_reminder(self, user_id: str, message: str, remind_time: datetime, channel_id: str) -> dict:
        """알림 설정"""
        try:
            reminder_id = f"{user_id}_{int(time.time())}"
            
            self.scheduler.add_job(
                self.send_reminder,
                'date',
                run_date=remind_time,
                args=[user_id, message, channel_id, reminder_id],
                id=reminder_id
            )
            
            self.reminders[reminder_id] = {
                'user_id': user_id,
                'message': message,
                'remind_time': remind_time,
                'channel_id': channel_id,
                'created_at': datetime.now()
            }
            
            return {
                'success': True,
                'reminder_id': reminder_id,
                'message': f"알림이 설정되었습니다: {remind_time.strftime('%Y-%m-%d %H:%M')}"
            }
        except Exception as e:
            logger.error(f"알림 설정 오류: {e}")
            return {'error': str(e)}
    
    async def send_reminder(self, user_id: str, message: str, channel_id: str, reminder_id: str):
        """알림 전송"""
        try:
            channel = client.get_channel(int(channel_id))
            if channel:
                embed = discord.Embed(
                    title="⏰ 알림",
                    description=message,
                    color=0x00ff00,
                    timestamp=datetime.now()
                )
                embed.add_field(name="사용자", value=f"<@{user_id}>", inline=True)
                
                await channel.send(embed=embed)
                
                # 알림 기록에서 제거
                if reminder_id in self.reminders:
                    del self.reminders[reminder_id]
                    
        except Exception as e:
            logger.error(f"알림 전송 오류: {e}")
    
    async def set_recurring_task(self, user_id: str, message: str, cron_expression: str, channel_id: str) -> dict:
        """반복 작업 설정"""
        try:
            task_id = f"recurring_{user_id}_{int(time.time())}"
            
            # 크론 표현식 파싱
            trigger = CronTrigger.from_crontab(cron_expression)
            
            self.scheduler.add_job(
                self.send_recurring_reminder,
                trigger,
                args=[user_id, message, channel_id, task_id],
                id=task_id
            )
            
            self.recurring_tasks[task_id] = {
                'user_id': user_id,
                'message': message,
                'cron_expression': cron_expression,
                'channel_id': channel_id,
                'created_at': datetime.now()
            }
            
            return {
                'success': True,
                'task_id': task_id,
                'message': f"반복 작업이 설정되었습니다: {cron_expression}"
            }
        except Exception as e:
            logger.error(f"반복 작업 설정 오류: {e}")
            return {'error': str(e)}
    
    async def send_recurring_reminder(self, user_id: str, message: str, channel_id: str, task_id: str):
        """반복 알림 전송"""
        try:
            channel = client.get_channel(int(channel_id))
            if channel:
                embed = discord.Embed(
                    title="🔄 반복 알림",
                    description=message,
                    color=0x0099ff,
                    timestamp=datetime.now()
                )
                embed.add_field(name="사용자", value=f"<@{user_id}>", inline=True)
                
                await channel.send(embed=embed)
                
        except Exception as e:
            logger.error(f"반복 알림 전송 오류: {e}")
    
    async def list_reminders(self, user_id: str) -> dict:
        """사용자의 알림 목록"""
        user_reminders = {
            rid: reminder for rid, reminder in self.reminders.items()
            if reminder['user_id'] == user_id
        }
        
        user_tasks = {
            tid: task for tid, task in self.recurring_tasks.items()
            if task['user_id'] == user_id
        }
        
        return {
            'reminders': user_reminders,
            'recurring_tasks': user_tasks,
            'total_reminders': len(user_reminders),
            'total_tasks': len(user_tasks)
        }
    
    async def cancel_reminder(self, reminder_id: str, user_id: str) -> dict:
        """알림 취소"""
        try:
            if reminder_id in self.reminders and self.reminders[reminder_id]['user_id'] == user_id:
                self.scheduler.remove_job(reminder_id)
                del self.reminders[reminder_id]
                return {'success': True, 'message': '알림이 취소되었습니다.'}
            else:
                return {'error': '알림을 찾을 수 없거나 권한이 없습니다.'}
        except Exception as e:
            logger.error(f"알림 취소 오류: {e}")
            return {'error': str(e)}

class SecurityModerator:
    """고급 보안 및 모더레이션 시스템"""
    
    def __init__(self):
        self.user_warnings = defaultdict(int)
        self.banned_words = set()
        self.spam_tracker = defaultdict(lambda: {'count': 0, 'last_message': '', 'timestamps': deque()})
        self.load_banned_words()
    
    def load_banned_words(self):
        """금지 단어 로드"""
        # 기본 금지 단어들 (실제 운영시에는 파일에서 로드)
        default_banned = ['spam', 'hack', 'cheat', 'bot']
        self.banned_words.update(default_banned)
    
    async def moderate_message(self, message: discord.Message) -> dict:
        """메시지 모더레이션"""
        try:
            user_id = str(message.author.id)
            content = message.content.lower()
            
            # 최고관리자는 모더레이션 제외
            if user_manager.is_super_admin(user_id):
                return {'action': 'allow', 'reason': 'super_admin'}
            
            # 스팸 검사
            spam_check = self.check_spam(user_id, message.content)
            if spam_check['is_spam']:
                return {
                    'action': 'delete',
                    'reason': 'spam',
                    'details': spam_check
                }
            
            # 금지 단어 검사
            banned_word_check = self.check_banned_words(content)
            if banned_word_check['found']:
                return {
                    'action': 'warn',
                    'reason': 'banned_word',
                    'details': banned_word_check
                }
            
            # 링크 검사
            link_check = self.check_suspicious_links(content)
            if link_check['suspicious']:
                return {
                    'action': 'warn',
                    'reason': 'suspicious_link',
                    'details': link_check
                }
            
            # 대문자 남용 검사
            caps_check = self.check_excessive_caps(content)
            if caps_check['excessive']:
                return {
                    'action': 'warn',
                    'reason': 'excessive_caps',
                    'details': caps_check
                }
            
            return {'action': 'allow', 'reason': 'clean'}
            
        except Exception as e:
            logger.error(f"메시지 모더레이션 오류: {e}")
            return {'action': 'allow', 'reason': 'error'}
    
    def check_spam(self, user_id: str, content: str) -> dict:
        """스팸 검사"""
        try:
            now = time.time()
            tracker = self.spam_tracker[user_id]
            
            # 5분 이전 메시지는 제거
            while tracker['timestamps'] and now - tracker['timestamps'][0] > 300:
                tracker['timestamps'].popleft()
            
            # 동일한 메시지 반복 검사
            if tracker['last_message'] == content:
                tracker['count'] += 1
                if tracker['count'] >= 3:
                    return {
                        'is_spam': True,
                        'type': 'repeated_message',
                        'count': tracker['count']
                    }
            else:
                tracker['count'] = 1
                tracker['last_message'] = content
            
            # 메시지 빈도 검사
            tracker['timestamps'].append(now)
            if len(tracker['timestamps']) > 10:  # 5분 내 10개 이상 메시지
                return {
                    'is_spam': True,
                    'type': 'high_frequency',
                    'message_count': len(tracker['timestamps'])
                }
            
            return {'is_spam': False}
            
        except Exception as e:
            logger.error(f"스팸 검사 오류: {e}")
            return {'is_spam': False}
    
    def check_banned_words(self, content: str) -> dict:
        """금지 단어 검사"""
        try:
            found_words = []
            for word in self.banned_words:
                if word in content:
                    found_words.append(word)
            
            return {
                'found': len(found_words) > 0,
                'words': found_words,
                'count': len(found_words)
            }
        except Exception as e:
            logger.error(f"금지 단어 검사 오류: {e}")
            return {'found': False, 'words': []}
    
    def check_suspicious_links(self, content: str) -> dict:
        """의심스러운 링크 검사"""
        try:
            # URL 패턴 찾기
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, content)
            
            suspicious_domains = ['bit.ly', 'tinyurl.com', 'short.link']  # 예시
            suspicious_urls = []
            
            for url in urls:
                for domain in suspicious_domains:
                    if domain in url:
                        suspicious_urls.append(url)
                        break
            
            return {
                'suspicious': len(suspicious_urls) > 0,
                'urls': suspicious_urls,
                'total_urls': len(urls)
            }
        except Exception as e:
            logger.error(f"링크 검사 오류: {e}")
            return {'suspicious': False, 'urls': []}
    
    def check_excessive_caps(self, content: str) -> dict:
        """대문자 남용 검사"""
        try:
            if len(content) < 10:  # 짧은 메시지는 제외
                return {'excessive': False}
            
            caps_count = sum(1 for c in content if c.isupper())
            caps_ratio = caps_count / len(content)
            
            return {
                'excessive': caps_ratio > 0.7,  # 70% 이상 대문자
                'ratio': round(caps_ratio, 2),
                'caps_count': caps_count
            }
        except Exception as e:
            logger.error(f"대문자 검사 오류: {e}")
            return {'excessive': False}
    
    async def warn_user(self, user_id: str, reason: str) -> dict:
        """사용자 경고"""
        try:
            self.user_warnings[user_id] += 1
            warning_count = self.user_warnings[user_id]
            
            # 경고 누적에 따른 조치
            if warning_count >= 5:
                action = 'ban'
            elif warning_count >= 3:
                action = 'timeout'
            else:
                action = 'warn'
            
            return {
                'action': action,
                'warning_count': warning_count,
                'reason': reason,
                'message': f"경고 {warning_count}회: {reason}"
            }
        except Exception as e:
            logger.error(f"사용자 경고 오류: {e}")
            return {'error': str(e)}
    
    async def get_user_warnings(self, user_id: str) -> dict:
        """사용자 경고 조회"""
        return {
            'user_id': user_id,
            'warning_count': self.user_warnings[user_id],
            'status': 'clean' if self.user_warnings[user_id] == 0 else 'warned'
        }

# 초고급 기능 인스턴스 생성
voice_processor = VoiceProcessor()
image_analyzer = AdvancedImageAnalyzer()
translation_system = TranslationSystem()
game_system = GameSystem()
code_analyzer = CodeAnalyzer()
data_analyzer = DataAnalyzer()
smart_scheduler = SmartScheduler()
security_moderator = SecurityModerator()

# 봇 설정
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

client = discord.Client(intents=intents)

# 보안 및 유틸리티 함수들
def is_spam_or_abuse(text: str, user_id: str = None) -> bool:
    """향상된 스팸 및 악용 감지 (최고관리자는 제외)"""
    if user_id and user_manager.is_super_admin(user_id):
        return False
        
    spam_patterns = [
        r'(.)\1{15,}',
        r'[!@#$%^&*]{8,}',
        r'(?i)(spam|test|bot|hack){5,}',
    ]
    return any(re.search(pattern, text) for pattern in spam_patterns)

async def process_advanced_natural_language(user_input: str, user_id: str, username: str = None) -> dict:
    """고급 자연어 요청 분석 및 처리 (응답 반복 문제 완전 해결)"""
    user_input_lower = user_input.lower()
    
    # 🔧 시스템 관리 기능
    if any(keyword in user_input_lower for keyword in ["모델 바꿔", "모델 변경", "프로로 바꿔", "플래시로 바꿔"]):
        if "프로" in user_input_lower or "pro" in user_input_lower:
            user_manager.user_preferences[user_id] = user_manager.user_preferences.get(user_id, {})
            user_manager.user_preferences[user_id]["model"] = "pro"
            return {"action": "model_change", "model": "pro", "message": "🧠 Gemini Pro 모델로 변경했습니다! 더 깊이 있는 분석을 제공할게요.", "stop_processing": True}
        elif "플래시" in user_input_lower or "flash" in user_input_lower:
            user_manager.user_preferences[user_id] = user_manager.user_preferences.get(user_id, {})
            user_manager.user_preferences[user_id]["model"] = "flash"
            return {"action": "model_change", "model": "flash", "message": "⚡ Gemini Flash 모델로 변경했습니다! 빠른 응답을 제공할게요.", "stop_processing": True}
    
    if any(keyword in user_input_lower for keyword in ["대화 초기화", "대화 리셋", "새로 시작", "처음부터", "대화 삭제"]):
        if user_id in user_manager.user_chats:
            del user_manager.user_chats[user_id]
        return {"action": "reset_chat", "message": "🔄 대화를 초기화했습니다! 새로운 대화를 시작해보세요.", "stop_processing": True}
    
    # 📊 정보 조회 기능
    if any(keyword in user_input_lower for keyword in ["설정 보여줘", "내 설정", "현재 설정", "내 정보"]):
        user_data = await user_manager.get_user_from_db(user_id, username)
        prefs = user_manager.user_preferences.get(user_id, {})
        model = prefs.get("model", user_data.get("model_preference", "flash"))
        
        info = f"""📋 **{username or '사용자'}님의 설정 정보**
        
🤖 **현재 모델**: {model.upper()}
📈 **총 메시지**: {user_data.get('total_messages', 0)}개
⏰ **마지막 사용**: 방금 전
🔑 **권한**: {'최고관리자 🔥' if user_manager.is_super_admin(user_id) else '일반 사용자'}"""
        
        return {"action": "show_settings", "message": info, "stop_processing": True}
    
    # 🤖 실시간 학습 기반 스마트 응답 (최우선)
    if REALTIME_LEARNING_AVAILABLE:
        # GPT-5, Claude 3.5, Gemini 2.0 등 최신 AI 모델 관련 질문
        ai_model_keywords = ['gpt-5', 'gpt5', 'claude 3.5', 'claude3.5', 'gemini 2.0', 'gemini2.0', 
                           'llama 3', 'llama3', 'o1-preview', 'o1-mini', '나왔어', '출시', '발표']
        
        if any(keyword in user_input_lower for keyword in ai_model_keywords):
            try:
                smart_answer = await get_smart_answer(user_input)
                return {"action": "smart_learning", "message": smart_answer, "stop_processing": True}
            except Exception as e:
                logger.error(f"스마트 답변 생성 오류: {e}")
        
        # 최신 기술 트렌드 질문
        tech_trend_keywords = ['최신', '트렌드', '동향', '뉴스', '발전', '혁신', '기술', 'ai', '인공지능']
        if any(keyword in user_input_lower for keyword in tech_trend_keywords):
            try:
                smart_answer = await get_smart_answer(user_input)
                if "찾을 수 없어요" not in smart_answer:  # 유효한 답변이 있을 때만
                    return {"action": "smart_trend", "message": smart_answer, "stop_processing": True}
            except Exception as e:
                logger.error(f"스마트 트렌드 답변 오류: {e}")
    
    # 🔥 최신 기술 뉴스 및 정보 검색
    if any(keyword in user_input_lower for keyword in ["최신 뉴스", "기술 뉴스", "ai 뉴스", "gpt 뉴스", "기술 동향", "최신 정보"]):
        if KNOWLEDGE_SYSTEM_AVAILABLE:
            try:
                # 카테고리 감지
                category = None
                if "ai" in user_input_lower or "인공지능" in user_input_lower:
                    category = "ai"
                elif "gpt" in user_input_lower:
                    category = "gpt"
                elif "기술" in user_input_lower or "tech" in user_input_lower:
                    category = "tech"
                
                news_summary = await get_tech_news_summary(category)
                return {"action": "tech_news", "message": news_summary, "stop_processing": True}
            except Exception as e:
                logger.error(f"기술 뉴스 요청 오류: {e}")
                return {"action": "tech_news", "message": "기술 뉴스를 가져오는 중 오류가 발생했습니다.", "stop_processing": True}
        else:
            return {"action": "tech_news", "message": "🔧 고급 지식 시스템이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.", "stop_processing": True}
    
    # 🔍 고급 웹 검색 기능
    if any(keyword in user_input_lower for keyword in ["검색해줘", "찾아줘", "위키피디아", "웹 검색", "구글 검색", "빙 검색"]):
        # 검색어 추출
        search_terms = ["검색해줘", "찾아줘", "위키피디아", "웹 검색", "구글 검색", "빙 검색", "검색"]
        query = user_input
        for term in search_terms:
            query = query.replace(term, "").strip()
        
        if query:
            # 검색 엔진 지정 확인
            engines = None
            if "위키피디아" in user_input_lower:
                engines = ["wikipedia"]
            elif "구글" in user_input_lower or "google" in user_input_lower:
                engines = ["google", "wikipedia"]
            elif "빙" in user_input_lower or "bing" in user_input_lower:
                engines = ["bing", "wikipedia"]
            elif "깃허브" in user_input_lower or "github" in user_input_lower:
                engines = ["github", "google"]
            elif "레딧" in user_input_lower or "reddit" in user_input_lower:
                engines = ["reddit", "google"]
            
            # 기술 관련 검색어인지 확인
            tech_keywords = ["ai", "gpt", "chatgpt", "머신러닝", "딥러닝", "프로그래밍", "개발", "기술", "스타트업", "코딩", "알고리즘"]
            is_tech_query = any(keyword in query.lower() for keyword in tech_keywords)
            
            # 벡터 검색 우선 시도 (기술 관련 쿼리)
            if is_tech_query and KNOWLEDGE_SYSTEM_AVAILABLE:
                try:
                    search_results = await search_knowledge_base(query, top_k=3)
                    
                    if search_results:
                        result_text = f"🔍 **'{query}' 벡터 검색 결과:**\n\n"
                        for i, result in enumerate(search_results, 1):
                            title = result.get('title', '')
                            source = result.get('source', '')
                            summary = result.get('summary', '')[:150] + "..." if len(result.get('summary', '')) > 150 else result.get('summary', '')
                            score = result.get('similarity_score', 0)
                            
                            result_text += f"{i}. **{title}**\n"
                            result_text += f"   📰 {source} (관련도: {score:.2f})\n"
                            result_text += f"   {summary}\n\n"
                        
                        result_text += "💡 *벡터 검색으로 최신 기술 정보를 제공했습니다.*\n\n"
                        
                        # 추가로 지능형 웹 검색도 수행
                        try:
                            web_result = await web_searcher.intelligent_web_search(query)
                            result_text += f"🌐 **추가 웹 검색 결과:**\n{web_result}"
                        except Exception as e:
                            logger.error(f"지능형 웹 검색 오류: {e}")
                        
                        return {"action": "hybrid_search", "message": result_text, "stop_processing": True}
                        
                except Exception as e:
                    logger.error(f"벡터 검색 오류: {e}")
            
            # 지능형 웹 검색 수행
            try:
                result = await web_searcher.intelligent_web_search(query)
                
                # 검색 결과로부터 기술 키워드 업데이트
                try:
                    from dynamic_response_system import update_keywords_from_search
                    # 검색 결과에서 키워드 추출을 위해 결과 전달
                    if hasattr(web_searcher, 'intelligent_searcher') and web_searcher.intelligent_searcher:
                        # 최근 검색 결과 가져오기 (간단한 방법)
                        asyncio.create_task(update_keywords_from_search({'results': [{'title': query, 'snippet': result}]}))
                except Exception as keyword_error:
                    logger.error(f"키워드 업데이트 오류: {keyword_error}")
                
                return {"action": "intelligent_web_search", "message": result, "stop_processing": True}
                
            except Exception as e:
                    logger.error(f"고급 웹 검색 오류: {e}")
                    # 폴백: 기본 검색
                    if "위키피디아" in user_input_lower:
                        result = await WebSearcher.search_wikipedia(query)
                    else:
                        result = await WebSearcher.web_search(query)
                    return {"action": "fallback_search", "message": result, "stop_processing": True}
            else:
                # 기본 검색 시스템 사용
                if "위키피디아" in user_input_lower:
                    result = await WebSearcher.search_wikipedia(query)
                else:
                    result = await WebSearcher.web_search(query)
                return {"action": "basic_search", "message": result, "stop_processing": True}
        else:
            search_status = "🔥 활성화됨" if WEB_SEARCH_AVAILABLE else "⏳ 초기화 중"
            return {"action": "search_help", "message": f"""🔍 **고급 웹 검색 도우미** (상태: {search_status})

💡 **사용법:**
• "검색해줘 [검색어]" - 종합 웹 검색
• "구글 검색해줘 [검색어]" - 구글 중심 검색
• "위키피디아 검색해줘 [검색어]" - 위키피디아 검색
• "깃허브 검색해줘 [검색어]" - 깃허브 프로젝트 검색
• "요약 검색해줘 [검색어]" - 간단 요약 검색

🎯 **특화 기능:**
• 기술/AI 관련 검색 시 벡터 검색 + 웹 검색 결합
• 다중 검색 엔진 동시 검색
• 결과 중복 제거 및 관련성 순 정렬
• 실시간 캐싱으로 빠른 응답

검색어를 알려주세요!""", "stop_processing": True}
    
    # 📊 검색 통계 조회
    if any(keyword in user_input_lower for keyword in ["검색 통계", "검색 현황", "검색 상태", "웹 검색 통계"]):
        try:
            stats_result = await web_searcher.get_search_stats()
            return {"action": "search_stats", "message": stats_result, "stop_processing": True}
        except Exception as e:
            logger.error(f"검색 통계 조회 오류: {e}")
            return {"action": "search_stats", "message": f"❌ 검색 통계 조회 중 오류가 발생했습니다: {str(e)}", "stop_processing": True}
    
    # ===== 초고급 기능들 =====
    
    # 🎵 음성 처리 기능
    if any(keyword in user_input_lower for keyword in ["음성으로 말해", "tts", "읽어줘", "소리내서 읽어"]):
        text_to_speak = user_input
        for keyword in ["음성으로 말해", "tts", "읽어줘", "소리내서 읽어"]:
            text_to_speak = text_to_speak.replace(keyword, "").strip()
        
        if not text_to_speak:
            text_to_speak = "안녕하세요! 루시아 봇입니다."
        
        try:
            audio_buffer = await voice_processor.text_to_speech(text_to_speak)
            if audio_buffer:
                return {"action": "tts", "audio_data": audio_buffer, "message": f"🎵 음성으로 변환했습니다: '{text_to_speak}'", "stop_processing": True}
            else:
                return {"action": "tts_error", "message": "음성 변환 중 오류가 발생했습니다.", "stop_processing": True}
        except Exception as e:
            logger.error(f"TTS 처리 오류: {e}")
            return {"action": "tts_error", "message": f"음성 변환 오류: {str(e)}", "stop_processing": True}
    
    # 🌍 번역 기능
    if any(keyword in user_input_lower for keyword in ["번역해", "translate", "영어로", "한국어로", "일본어로", "중국어로"]):
        # 번역할 텍스트와 목표 언어 추출
        text_to_translate = user_input
        target_lang = 'en'  # 기본값
        
        if "영어로" in user_input_lower or "english" in user_input_lower:
            target_lang = 'en'
            text_to_translate = text_to_translate.replace("영어로", "").replace("english", "")
        elif "한국어로" in user_input_lower or "korean" in user_input_lower:
            target_lang = 'ko'
            text_to_translate = text_to_translate.replace("한국어로", "").replace("korean", "")
        elif "일본어로" in user_input_lower or "japanese" in user_input_lower:
            target_lang = 'ja'
            text_to_translate = text_to_translate.replace("일본어로", "").replace("japanese", "")
        elif "중국어로" in user_input_lower or "chinese" in user_input_lower:
            target_lang = 'zh'
            text_to_translate = text_to_translate.replace("중국어로", "").replace("chinese", "")
        
        # 번역 키워드 제거
        for keyword in ["번역해", "translate"]:
            text_to_translate = text_to_translate.replace(keyword, "").strip()
        
        if text_to_translate:
            try:
                translation_result = await translation_system.translate_text(text_to_translate, target_lang)
                
                if 'error' not in translation_result:
                    result_message = f"""🌍 **번역 결과**
                    
**원문** ({translation_result['source_language']}): {translation_result['original_text']}
**번역** ({translation_result['target_language']}): {translation_result['translated_text']}
**신뢰도**: {translation_result['confidence']:.2%}"""
                    
                    return {"action": "translate", "message": result_message, "stop_processing": True}
                else:
                    return {"action": "translate_error", "message": f"번역 오류: {translation_result['error']}", "stop_processing": True}
            except Exception as e:
                logger.error(f"번역 처리 오류: {e}")
                return {"action": "translate_error", "message": f"번역 처리 중 오류가 발생했습니다: {str(e)}", "stop_processing": True}
        else:
            return {"action": "translate_help", "message": """🌍 **번역 도우미**
            
**사용법:**
• "번역해 Hello World" - 자동 언어 감지 후 번역
• "영어로 번역해 안녕하세요" - 한국어를 영어로
• "한국어로 번역해 Hello" - 영어를 한국어로
• "일본어로 번역해 안녕" - 일본어로 번역
• "중국어로 번역해 Hello" - 중국어로 번역

**지원 언어:** 한국어, 영어, 일본어, 중국어, 스페인어, 프랑스어, 독일어, 러시아어 등""", "stop_processing": True}
    
    # 🎮 게임 기능
    if any(keyword in user_input_lower for keyword in ["체스", "chess", "체스 게임"]):
        if "시작" in user_input_lower or "start" in user_input_lower:
            try:
                game_result = await game_system.start_chess_game(user_id)
                if 'error' not in game_result:
                    return {
                        "action": "chess_start", 
                        "message": game_result['message'],
                        "image_data": game_result['board_image'],
                        "game_id": game_result['game_id'],
                        "stop_processing": True
                    }
                else:
                    return {"action": "chess_error", "message": f"체스 게임 시작 오류: {game_result['error']}", "stop_processing": True}
            except Exception as e:
                logger.error(f"체스 게임 시작 오류: {e}")
                return {"action": "chess_error", "message": f"체스 게임 시작 중 오류가 발생했습니다: {str(e)}", "stop_processing": True}
        else:
            return {"action": "chess_help", "message": """♟️ **체스 게임**
            
**사용법:**
• "체스 시작" - 새 체스 게임 시작
• "체스 수 e2e4" - 체스 수 두기 (UCI 표기법)
• "체스 상태" - 현재 게임 상태 확인

**UCI 표기법 예시:**
• e2e4 (폰을 e2에서 e4로)
• g1f3 (나이트를 g1에서 f3로)
• e1g1 (킹사이드 캐슬링)""", "stop_processing": True}
    
    if any(keyword in user_input_lower for keyword in ["퀴즈", "quiz", "문제"]):
        try:
            category = 'general'
            if "과학" in user_input_lower or "science" in user_input_lower:
                category = 'science'
            
            quiz_result = await game_system.play_trivia_game(category)
            if 'error' not in quiz_result:
                options_text = "\n".join([f"{i+1}. {option}" for i, option in enumerate(quiz_result['options'])])
                quiz_message = f"""🧠 **퀴즈 게임** ({quiz_result['category']})

**문제:** {quiz_result['question']}

**선택지:**
{options_text}

정답을 번호로 답해주세요! (예: 1, 2, 3, 4)"""
                
                return {"action": "quiz", "message": quiz_message, "quiz_data": quiz_result, "stop_processing": True}
            else:
                return {"action": "quiz_error", "message": f"퀴즈 생성 오류: {quiz_result['error']}", "stop_processing": True}
        except Exception as e:
            logger.error(f"퀴즈 게임 오류: {e}")
            return {"action": "quiz_error", "message": f"퀴즈 게임 중 오류가 발생했습니다: {str(e)}", "stop_processing": True}
    
    if any(keyword in user_input_lower for keyword in ["끝말잇기", "word chain", "단어 게임"]):
        # 끝말잇기 단어 추출
        word = user_input
        for keyword in ["끝말잇기", "word chain", "단어 게임"]:
            word = word.replace(keyword, "").strip()
        
        if word:
            try:
                game_result = await game_system.word_chain_game(user_id, word)
                if 'error' not in game_result:
                    return {"action": "word_chain", "message": game_result['message'], "game_data": game_result, "stop_processing": True}
                else:
                    return {"action": "word_chain_error", "message": f"끝말잇기 오류: {game_result['error']}", "stop_processing": True}
            except Exception as e:
                logger.error(f"끝말잇기 게임 오류: {e}")
                return {"action": "word_chain_error", "message": f"끝말잇기 게임 중 오류가 발생했습니다: {str(e)}", "stop_processing": True}
        else:
            return {"action": "word_chain_help", "message": """🎯 **끝말잇기 게임**
            
**사용법:**
• "끝말잇기 사과" - '사과'로 끝말잇기 시작
• 이후 마지막 글자로 시작하는 단어를 말하세요!

**규칙:**
• 이미 사용한 단어는 사용할 수 없습니다
• 한글 단어만 가능합니다
• AI가 자동으로 응답합니다

단어를 말해서 게임을 시작해보세요!""", "stop_processing": True}
    
    # 💻 코드 분석 기능
    if any(keyword in user_input_lower for keyword in ["코드 분석", "code analysis", "코드 검사", "코드 리뷰"]):
        return {"action": "code_analysis_help", "message": """💻 **코드 분석 시스템**
        
**사용법:**
• 코드 블록을 ```python 또는 ```javascript 등으로 감싸서 보내주세요
• "코드 분석해줘" + 코드 블록
• "파이썬 코드 검사해줘" + 코드 블록

**분석 기능:**
• 구문 검사 (Syntax Check)
• 코드 복잡도 분석
• 보안 이슈 검사
• 개선 제안
• 성능 최적화 팁

**지원 언어:** Python, JavaScript, Java, C++, C, HTML, CSS, SQL

코드를 보내주시면 자세히 분석해드릴게요!""", "stop_processing": True}
    
    # 📊 데이터 분석 기능
    if any(keyword in user_input_lower for keyword in ["데이터 분석", "data analysis", "통계", "차트", "그래프"]):
        return {"action": "data_analysis_help", "message": """📊 **데이터 분석 시스템**
        
**사용법:**
• "데이터 분석해줘 [JSON 데이터]"
• "차트 만들어줘 [데이터]"
• "통계 내줘 [숫자 데이터]"
• "워드클라우드 만들어줘 [텍스트]"

**분석 기능:**
• 기본 통계 (평균, 중앙값, 표준편차 등)
• 데이터 타입 분석
• 결측값 검사
• 상관관계 분석
• 다양한 차트 생성 (막대, 선, 산점도, 히스토그램 등)

**시각화 옵션:**
• bar, line, scatter, histogram, box, pie

예시: "차트 만들어줘 {'A': 10, 'B': 20, 'C': 15}" """, "stop_processing": True}
    
    # ⏰ 스케줄링 기능
    if any(keyword in user_input_lower for keyword in ["알림 설정", "reminder", "스케줄", "schedule"]):
        return {"action": "scheduler_help", "message": """⏰ **스마트 스케줄러**
        
**사용법:**
• "알림 설정해줘 2024-12-25 09:00 크리스마스!"
• "매일 알림 설정해줘 0 9 * * * 아침 운동"
• "내 알림 목록"
• "알림 취소 [알림ID]"

**시간 형식:**
• 날짜: YYYY-MM-DD HH:MM
• 크론 표현식: 분 시 일 월 요일

**반복 알림 예시:**
• "0 9 * * *" - 매일 오전 9시
• "0 9 * * 1" - 매주 월요일 오전 9시
• "0 9 1 * *" - 매월 1일 오전 9시

**기능:**
• 일회성 알림
• 반복 알림
• 알림 목록 조회
• 알림 취소

알림을 설정해보세요!""", "stop_processing": True}
    
    # 🔒 보안 기능
    if any(keyword in user_input_lower for keyword in ["내 경고", "warning", "보안 상태"]):
        try:
            warning_info = await security_moderator.get_user_warnings(user_id)
            warning_message = f"""🔒 **보안 상태**
            
**사용자:** <@{user_id}>
**경고 횟수:** {warning_info['warning_count']}회
**상태:** {warning_info['status']}

**보안 등급:**
{'🟢 안전' if warning_info['warning_count'] == 0 else '🟡 주의' if warning_info['warning_count'] < 3 else '🔴 위험'}

**참고사항:**
• 경고 3회 이상: 임시 제재
• 경고 5회 이상: 영구 제재
• 스팸, 욕설, 악성 링크 등은 자동 감지됩니다"""
            
            return {"action": "security_status", "message": warning_message, "stop_processing": True}
        except Exception as e:
            logger.error(f"보안 상태 조회 오류: {e}")
            return {"action": "security_error", "message": f"보안 상태 조회 중 오류가 발생했습니다: {str(e)}", "stop_processing": True}
    
    # 📊 차트 및 이미지 생성
    if any(keyword in user_input_lower for keyword in ["차트 만들어", "그래프 그려", "차트 그려", "시각화"]):
        # 간단한 예시 데이터로 차트 생성
        sample_data = {"월": 10, "화": 15, "수": 8, "목": 20, "금": 25}
        chart_buffer = await ImageGenerator.create_simple_chart(sample_data, "주간 활동 통계")
        
        if chart_buffer:
            return {"action": "create_chart", "message": "📊 차트를 생성했습니다!", "file": chart_buffer, "stop_processing": True}
        else:
            return {"action": "create_chart", "message": "차트 생성 중 오류가 발생했습니다.", "stop_processing": True}
    
    # 🖥️ 시스템 모니터링
    if any(keyword in user_input_lower for keyword in ["시스템 상태", "서버 상태", "성능 확인", "모니터링"]):
        system_info = SystemMonitor.get_system_info()
        
        status_msg = f"""🖥️ **시스템 상태 보고서**
        
💻 **CPU 사용률**: {system_info.get('cpu_percent', 0):.1f}%
🧠 **메모리 사용률**: {system_info.get('memory_percent', 0):.1f}%
💾 **디스크 사용률**: {system_info.get('disk_percent', 0):.1f}%
🔄 **실행 중인 프로세스**: {system_info.get('process_count', 0)}개
⏰ **시스템 부팅 시간**: {system_info.get('boot_time', 'Unknown')}"""
        
        # 상태 이미지도 생성
        stats_data = {
            'users': {'활성': len(user_manager.user_chats), '전체': len(user_manager.user_preferences)},
            'messages': {'오늘': 50, '어제': 30, '그제': 20}  # 예시 데이터
        }
        status_image = await ImageGenerator.create_status_image(stats_data)
        
        if status_image:
            return {"action": "system_status", "message": status_msg, "file": status_image, "stop_processing": True}
        else:
            return {"action": "system_status", "message": status_msg, "stop_processing": True}
    
    # 🔥 관리자 전용 기능
    if user_manager.is_super_admin(user_id):
        if any(keyword in user_input_lower for keyword in ["통계 보여줘", "봇 통계", "사용 통계", "전체 통계"]):
            # 데이터베이스에서 실제 통계 가져오기
            async with aiosqlite.connect('lucia_bot.db') as db:
                async with db.execute('SELECT COUNT(*) FROM users') as cursor:
                    total_users = (await cursor.fetchone())[0]
                
                async with db.execute('SELECT SUM(total_messages) FROM users') as cursor:
                    total_messages = (await cursor.fetchone())[0] or 0
                
                async with db.execute('SELECT COUNT(*) FROM conversations WHERE date(timestamp) = date("now")') as cursor:
                    today_messages = (await cursor.fetchone())[0]
            
            stats_info = f"""📊 **루시아 봇 전체 통계**
            
👥 **총 사용자**: {total_users}명
💬 **총 메시지**: {total_messages:,}개
📅 **오늘 메시지**: {today_messages}개
🚫 **차단된 사용자**: {len(user_manager.blocked_users)}명
🔄 **활성 세션**: {len(user_manager.user_chats)}개
🖥️ **시스템 가동률**: 99.9%"""
            
            return {"action": "admin_stats", "message": stats_info, "stop_processing": True}
        
        if any(keyword in user_input_lower for keyword in ["전체 초기화", "모든 데이터 초기화", "시스템 리셋"]):
            user_manager.user_chats.clear()
            user_manager.rate_limits.clear()
            user_manager.blocked_users.clear()
            user_manager.user_preferences.clear()
            return {"action": "full_reset", "message": "🔄 모든 시스템 데이터를 초기화했습니다!", "stop_processing": True}
    
    # 🎨 다양한 응답 시스템 적용
    # "뭐해?" 류의 질문들
    if any(keyword in user_input_lower for keyword in ["뭐해", "뭐하고", "뭐하는", "무엇을 하고", "지금 뭐"]):
        try:
            from dynamic_response_system import get_dynamic_response
            dynamic_answer = await get_dynamic_response(user_id, user_input, 'activity')
            return {"action": "dynamic_activity", "message": dynamic_answer, "stop_processing": True}
        except Exception as e:
            logger.error(f"다양한 응답 생성 오류: {e}")
    
    # 인사 관련 질문들
    if any(keyword in user_input_lower for keyword in ["안녕", "반가워", "처음", "hello", "hi"]):
        try:
            from dynamic_response_system import get_dynamic_response
            dynamic_answer = await get_dynamic_response(user_id, user_input, 'greeting')
            return {"action": "dynamic_greeting", "message": dynamic_answer, "stop_processing": True}
        except Exception as e:
            logger.error(f"다양한 인사 응답 생성 오류: {e}")
    
    # 일반 채팅으로 처리
    return {"action": "normal_chat", "message": None, "stop_processing": False}

@client.event
async def on_ready():
    """봇 시작 시 초기화"""
    await init_database()
    logger.info(f"🚀 루시아 봇이 준비되었습니다: {client.user}")
    print(f"✅ 루시아 고급 AI 어시스턴트 로그인 완료: {client.user}")
    
    # 🔥 고급 지식 시스템 초기화 (비활성화 - API 제한)
    logger.info("고급 지식 시스템 비활성화 (API 제한으로 인해)")
    # if KNOWLEDGE_SYSTEM_AVAILABLE:
    #     try:
    #         logger.info("고급 지식 시스템 초기화 시작...")
    #         asyncio.create_task(initialize_knowledge_system())
    #         logger.info("고급 지식 시스템 백그라운드 초기화 시작")
    #     except Exception as e:
    #         logger.error(f"고급 지식 시스템 초기화 오류: {e}")
    
    # 🌐 고급 웹 검색 시스템 초기화
    try:
        logger.info("지능형 웹 검색 시스템 초기화 시작...")
        serpapi_key = SERPAPI_KEY if SERPAPI_KEY else "YOUR_SERPAPI_KEY_HERE"
        asyncio.create_task(web_searcher.initialize_intelligent_search(serpapi_key))
        logger.info("지능형 웹 검색 시스템 백그라운드 초기화 시작")
    except Exception as e:
        logger.error(f"지능형 웹 검색 시스템 초기화 오류: {e}")
    
    # 🎨 다양한 응답 시스템 초기화
    try:
        logger.info("다양한 응답 시스템 초기화 시작...")
        from dynamic_response_system import initialize_dynamic_responses
        asyncio.create_task(initialize_dynamic_responses())
        logger.info("다양한 응답 시스템 백그라운드 초기화 시작")
    except Exception as e:
        logger.error(f"다양한 응답 시스템 초기화 오류: {e}")
    
    # 🤖 실시간 학습 시스템 초기화 (비활성화 - API 제한)
    logger.info("실시간 학습 시스템 비활성화 (API 제한으로 인해)")
    # if REALTIME_LEARNING_AVAILABLE:
    #     try:
    #         logger.info("실시간 학습 시스템 초기화 시작...")
    #         asyncio.create_task(initialize_learning_system())
    #         logger.info("실시간 학습 시스템 백그라운드 초기화 시작")
    #     except Exception as e:
    #         logger.error(f"실시간 학습 시스템 초기화 오류: {e}")
    
    # 백그라운드 작업 시작
    asyncio.create_task(advanced_cleanup_task())
    asyncio.create_task(performance_monitor())

@client.event
async def on_message(message):
    """메시지 처리 (응답 반복 문제 완전 해결)"""
    if message.author.bot:
        return
    
    # 중복 처리 방지 (강화)
    message_id = message.id
    async with message_lock:
        if message_id in processing_messages or message_id in responded_messages:
            return
        processing_messages.add(message_id)
    
    try:
        # 🔒 보안 모더레이션 (최고관리자 제외)
        user_id = str(message.author.id)
        if not user_manager.is_super_admin(user_id):
            try:
                moderation_result = await security_moderator.moderate_message(message)
                
                if moderation_result['action'] == 'delete':
                    try:
                        await message.delete()
                        warning_result = await security_moderator.warn_user(user_id, moderation_result['reason'])
                        
                        warning_message = f"""🚫 **메시지가 삭제되었습니다**
                        
**사유:** {moderation_result['reason']}
**경고 횟수:** {warning_result['warning_count']}회

{warning_result['message']}"""
                        
                        await message.channel.send(f"<@{user_id}> {warning_message}")
                        return
                    except discord.NotFound:
                        pass  # 메시지가 이미 삭제됨
                    except discord.Forbidden:
                        logger.warning(f"메시지 삭제 권한 없음: {message.id}")
                
                elif moderation_result['action'] == 'warn':
                    warning_result = await security_moderator.warn_user(user_id, moderation_result['reason'])
                    
                    warning_message = f"""⚠️ **경고**
                    
**사유:** {moderation_result['reason']}
**경고 횟수:** {warning_result['warning_count']}회

{warning_result['message']}"""
                    
                    await message.reply(warning_message)
                    
                    # 경고 누적에 따른 추가 조치
                    if warning_result['action'] == 'timeout':
                        try:
                            timeout_duration = timedelta(minutes=10)  # 10분 타임아웃
                            await message.author.timeout(timeout_duration, reason=f"경고 누적: {moderation_result['reason']}")
                            await message.channel.send(f"<@{user_id}> 경고 누적으로 10분간 타임아웃되었습니다.")
                        except discord.Forbidden:
                            logger.warning(f"타임아웃 권한 없음: {user_id}")
                    elif warning_result['action'] == 'ban':
                        try:
                            await message.author.ban(reason=f"경고 누적 5회: {moderation_result['reason']}")
                            await message.channel.send(f"<@{user_id}> 경고 누적으로 서버에서 차단되었습니다.")
                        except discord.Forbidden:
                            logger.warning(f"차단 권한 없음: {user_id}")
                    return  # 모더레이션 처리 후 종료
                        
            except Exception as moderation_error:
                logger.error(f"모더레이션 오류: {moderation_error}")
        
        # 자연어 대화 처리 (모더레이션 통과한 경우만)
        if client.user.mentioned_in(message) or message.content.lower().startswith("루시아"):
            await handle_advanced_natural_chat(message)
    
    finally:
        # 처리 완료 후 메시지 ID 제거 및 응답 완료 표시 (중요!)
        async with message_lock:
            processing_messages.discard(message_id)
            responded_messages.add(message_id)
            
            # 응답 완료 메시지 목록 크기 제한 (메모리 관리)
            if len(responded_messages) > 10000:
                # 가장 오래된 메시지들 제거
                old_messages = list(responded_messages)[:5000]
                for old_msg in old_messages:
                    responded_messages.discard(old_msg)

async def handle_advanced_natural_chat(message):
    """고급 자연어 기반 채팅 처리 (응답 반복 문제 완전 해결)"""
    start_time = time.time()
    user_id = str(message.author.id)
    username = str(message.author.display_name)
    message_id = message.id
    user_input = message.content.replace(f"<@{client.user.id}>", "").replace("루시아", "").strip()
    
    # 이미 응답한 메시지인지 다시 한번 확인
    async with message_lock:
        if message_id in responded_messages:
            return

    # 🔍 특수 기능 우선 처리 (다른 시스템 호출 방지)
    try:
        nl_result = await process_advanced_natural_language(user_input, user_id, username)
        
        # stop_processing가 True면 특수 기능 처리하고 즉시 종료
        if nl_result.get("stop_processing", False):
            # 다양한 응답 타입 처리
            if "audio_data" in nl_result:
                file_obj = discord.File(nl_result["audio_data"], filename="tts_output.wav")
                await message.reply(nl_result["message"], file=file_obj)
            elif "image_data" in nl_result:
                filename = "chess_board.png" if nl_result.get("action") == "chess_start" else "generated_image.png"
                file_obj = discord.File(nl_result["image_data"], filename=filename)
                await message.reply(nl_result["message"], file=file_obj)
            elif "file" in nl_result:
                file_obj = discord.File(nl_result["file"], filename="generated_content.png")
                await message.reply(nl_result["message"], file=file_obj)
            else:
                await message.reply(nl_result["message"])
            
            # 성능 메트릭 및 대화 기록
            processing_time = time.time() - start_time
            user_manager.update_performance_metrics("special_function_time", processing_time)
            await user_manager.save_conversation(user_id, user_input, nl_result["message"], "special")
            
            # 응답 완료 표시
            async with message_lock:
                responded_messages.add(message_id)
            
            logger.info(f"특수 기능 처리 완료: {username} - {nl_result.get('action', 'unknown')}")
            return  # 특수 기능 처리 완료 - 다른 시스템 호출하지 않음
            
    except Exception as special_error:
        logger.error(f"특수 기능 처리 오류: {special_error}")
        await message.reply("처리 중 문제가 발생했어요. 다시 시도해주세요.")
        
        # 응답 완료 표시
        async with message_lock:
            responded_messages.add(message_id)
        return

    if not user_input:
        knowledge_status = "🔥 활성화됨" if KNOWLEDGE_SYSTEM_AVAILABLE else "⏳ 초기화 중"
        web_search_status = "🔥 활성화됨" if WEB_SEARCH_AVAILABLE else "⏳ 초기화 중"
        
        await message.reply(f"""안녕하세요! 초고급 AI 어시스턴트 루시아입니다! 2025년 최신 정보와 함께 다양한 고급 기능들을 제공해드려요 ✨

**🔥 핵심 기능들:**
🧠 **2025년 최신 기술 정보** (시스템: {knowledge_status})
• "최신 AI 뉴스 알려줘" - GPT-5, Claude 3.5, Gemini 2.0 등 실시간 뉴스
• "기술 동향 보여줘" - 2025년 AI/ML 트렌드 분석

🌐 **고급 웹 검색** (시스템: {web_search_status})
• "검색해줘 [검색어]" - 다중 엔진 종합 검색
• "구글/위키피디아/깃허브 검색해줘" - 특화 검색

**🎵 음성 & 번역 기능**
• "음성으로 말해 [텍스트]" - TTS 음성 변환
• 음성 파일 업로드 시 자동 텍스트 변환
• "번역해 [텍스트]" / "영어로 번역해" - 다국어 번역

**🖼️ 이미지 & 미디어 분석**
• 이미지 업로드 시 자동 분석 (객체 감지, 얼굴 인식, OCR)
• "차트 만들어줘" - 데이터 시각화
• 색상 분석 및 품질 평가

**💻 코드 분석 & 개발 도구**
• 코드 파일 업로드 시 자동 분석
• ```python 코드 블록 분석
• 구문 검사, 복잡도 분석, 보안 이슈 검사

**🎮 게임 & 엔터테인먼트**
• "체스 시작" - 체스 게임
• "퀴즈" - 다양한 분야 퀴즈
• "끝말잇기 [단어]" - 끝말잇기 게임

**⏰ 스마트 기능**
• "알림 설정해줘" - 스케줄 관리
• "내 경고" - 보안 상태 확인
• "시스템 상태" - 봇 상태 모니터링

**🔒 보안 & 모더레이션**
• 자동 스팸/욕설 감지
• 악성 링크 차단
• 사용자별 경고 시스템

🎯 **특화 분야:** AI/GPT 동향, 프로그래밍, 이미지/음성 처리, 게임, 번역

💫 **초고급 특징:** 멀티모달 처리, 실시간 학습, 보안 강화, 게임 시스템

궁금한 것이 있으시면 언제든 말씀해주세요! 모든 기능을 자연어로 편하게 사용하실 수 있어요! 🚀""")
        
        # 응답 완료 표시
        async with message_lock:
            responded_messages.add(message_id)
        return
    
    try:
        # 📎 첨부파일 처리 (이미지, 음성 등)
        if message.attachments:
            for attachment in message.attachments:
                try:
                    # 이미지 파일 처리
                    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']):
                        # 이미지 다운로드
                        image_data = await attachment.read()
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                            temp_file.write(image_data)
                            temp_image_path = temp_file.name
                        
                        # 이미지 분석
                        analysis_result = await image_analyzer.analyze_image(temp_image_path)
                        
                        if 'error' not in analysis_result:
                            # 분석 결과 포맷팅
                            basic_info = analysis_result.get('basic_info', {})
                            faces = analysis_result.get('faces', {})
                            colors = analysis_result.get('colors', {})
                            objects = analysis_result.get('objects', [])
                            classification = analysis_result.get('classification', [])
                            quality = analysis_result.get('quality', {})
                            
                            analysis_message = f"""🖼️ **이미지 분석 결과**
                            
**기본 정보:**
• 크기: {basic_info.get('size', 'N/A')}
• 종횡비: {basic_info.get('aspect_ratio', 'N/A')}
• 채널: {basic_info.get('channels', 'N/A')}

**얼굴 감지:**
• 감지된 얼굴: {faces.get('count', 0)}개

**주요 색상:**"""
                            
                            dominant_colors = colors.get('dominant_colors', [])[:3]
                            for i, color in enumerate(dominant_colors, 1):
                                analysis_message += f"\n• {i}. {color.get('hex', 'N/A')} ({color.get('percentage', 0):.1f}%)"
                            
                            if objects:
                                analysis_message += f"\n\n**감지된 객체:**"
                                for obj in objects[:5]:  # 상위 5개만
                                    analysis_message += f"\n• {obj.get('label', 'N/A')} (신뢰도: {obj.get('confidence', 0):.2f})"
                            
                            if classification:
                                analysis_message += f"\n\n**이미지 분류:**"
                                for cls in classification:
                                    analysis_message += f"\n• {cls.get('label', 'N/A')} (신뢰도: {cls.get('confidence', 0):.2f})"
                            
                            analysis_message += f"\n\n**품질 평가:**"
                            analysis_message += f"\n• 선명도: {quality.get('blur_level', 'N/A')}"
                            analysis_message += f"\n• 밝기: {quality.get('brightness', 'N/A')}"
                            analysis_message += f"\n• 전체 품질: {quality.get('overall_quality', 'N/A')}"
                            
                            # OCR 텍스트가 있다면 추가
                            extracted_text = analysis_result.get('text', '').strip()
                            if extracted_text and extracted_text != "텍스트 추출 실패":
                                analysis_message += f"\n\n**추출된 텍스트:**\n{extracted_text}"
                            
                            await message.reply(analysis_message)
                            
                            # 응답 완료 표시
                            async with message_lock:
                                responded_messages.add(message_id)
                        else:
                            await message.reply(f"이미지 분석 중 오류가 발생했습니다: {analysis_result['error']}")
                            
                            # 응답 완료 표시
                            async with message_lock:
                                responded_messages.add(message_id)
                        
                        # 임시 파일 삭제
                        os.unlink(temp_image_path)
                        return
                    
                    # 음성 파일 처리
                    elif any(attachment.filename.lower().endswith(ext) for ext in ['.wav', '.mp3', '.ogg', '.m4a']):
                        # 음성 파일 다운로드
                        audio_data = await attachment.read()
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                            temp_file.write(audio_data)
                            temp_audio_path = temp_file.name
                        
                        # 음성을 텍스트로 변환
                        transcribed_text = await voice_processor.speech_to_text(temp_audio_path)
                        
                        if transcribed_text and "인식할 수 없습니다" not in transcribed_text:
                            response_message = f"""🎵 **음성 인식 결과**
                            
**인식된 텍스트:** {transcribed_text}

이제 이 내용에 대해 답변드릴게요!"""
                            
                            await message.reply(response_message)
                            
                            # 응답 완료 표시
                            async with message_lock:
                                responded_messages.add(message_id)
                            
                            # 인식된 텍스트로 AI 응답 생성
                            user_input = transcribed_text
                        else:
                            await message.reply(f"🎵 음성 인식 결과: {transcribed_text}")
                            
                            # 응답 완료 표시
                            async with message_lock:
                                responded_messages.add(message_id)
                            return
                        
                        # 임시 파일 삭제
                        os.unlink(temp_audio_path)
                    
                    # 코드 파일 처리
                    elif any(attachment.filename.lower().endswith(ext) for ext in ['.py', '.js', '.java', '.cpp', '.c', '.html', '.css', '.sql']):
                        # 코드 파일 다운로드
                        code_data = await attachment.read()
                        code_content = code_data.decode('utf-8', errors='ignore')
                        
                        # 파일 확장자로 언어 감지
                        file_ext = attachment.filename.lower().split('.')[-1]
                        language_map = {
                            'py': 'python', 'js': 'javascript', 'java': 'java',
                            'cpp': 'cpp', 'c': 'c', 'html': 'html', 'css': 'css', 'sql': 'sql'
                        }
                        language = language_map.get(file_ext, 'python')
                        
                        # 코드 분석
                        analysis_result = await code_analyzer.analyze_code(code_content, language)
                        
                        if 'error' not in analysis_result:
                            syntax_check = analysis_result.get('syntax_check', {})
                            complexity = analysis_result.get('complexity', {})
                            suggestions = analysis_result.get('suggestions', [])
                            security_issues = analysis_result.get('security_issues', [])
                            
                            analysis_message = f"""💻 **코드 분석 결과** ({language})
                            
**파일:** {attachment.filename}
**라인 수:** {analysis_result.get('line_count', 0)}
**문자 수:** {analysis_result.get('character_count', 0)}

**구문 검사:**
{'✅ 구문이 올바릅니다' if syntax_check.get('valid', False) else f'❌ 구문 오류: {syntax_check.get("error", "알 수 없는 오류")}'}"""
                            
                            if complexity:
                                analysis_message += f"\n\n**복잡도 분석:**"
                                analysis_message += f"\n• 코드 라인: {complexity.get('lines_of_code', 0)}"
                                analysis_message += f"\n• 제어 구조: {complexity.get('control_structures', 0)}"
                                analysis_message += f"\n• 함수: {complexity.get('functions', 0)}"
                                analysis_message += f"\n• 클래스: {complexity.get('classes', 0)}"
                                analysis_message += f"\n• 복잡도 등급: {complexity.get('complexity_level', 'N/A')}"
                            
                            if suggestions:
                                analysis_message += f"\n\n**개선 제안:**"
                                for suggestion in suggestions[:5]:  # 상위 5개만
                                    analysis_message += f"\n• {suggestion}"
                            
                            if security_issues:
                                analysis_message += f"\n\n**보안 이슈:**"
                                for issue in security_issues:
                                    analysis_message += f"\n⚠️ {issue}"
                            
                            await message.reply(analysis_message)
                            
                            # 응답 완료 표시
                            async with message_lock:
                                responded_messages.add(message_id)
                        else:
                            await message.reply(f"코드 분석 중 오류가 발생했습니다: {analysis_result['error']}")
                            
                            # 응답 완료 표시
                            async with message_lock:
                                responded_messages.add(message_id)
                        
                        return
                        
                except Exception as attachment_error:
                    logger.error(f"첨부파일 처리 오류: {attachment_error}")
                    await message.reply(f"첨부파일 처리 중 오류가 발생했습니다: {str(attachment_error)}")
                    
                    # 응답 완료 표시
                    async with message_lock:
                        responded_messages.add(message_id)
                    return
        
        # 📝 코드 블록 처리
        if "```" in user_input:
            code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', user_input, re.DOTALL)
            if code_blocks:
                for language, code in code_blocks:
                    if not language:
                        language = 'python'  # 기본값
                    
                    # 코드 분석
                    analysis_result = await code_analyzer.analyze_code(code.strip(), language.lower())
                    
                    if 'error' not in analysis_result:
                        syntax_check = analysis_result.get('syntax_check', {})
                        complexity = analysis_result.get('complexity', {})
                        suggestions = analysis_result.get('suggestions', [])
                        security_issues = analysis_result.get('security_issues', [])
                        
                        analysis_message = f"""💻 **코드 분석 결과** ({language})
                        
**라인 수:** {analysis_result.get('line_count', 0)}

**구문 검사:**
{'✅ 구문이 올바릅니다' if syntax_check.get('valid', False) else f'❌ 구문 오류: {syntax_check.get("error", "알 수 없는 오류")}'}"""
                        
                        if complexity and complexity.get('complexity_level'):
                            analysis_message += f"\n\n**복잡도:** {complexity.get('complexity_level', 'N/A')}"
                        
                        if suggestions:
                            analysis_message += f"\n\n**개선 제안:**"
                            for suggestion in suggestions[:3]:  # 상위 3개만
                                analysis_message += f"\n• {suggestion}"
                        
                        if security_issues:
                            analysis_message += f"\n\n**보안 이슈:**"
                            for issue in security_issues:
                                analysis_message += f"\n⚠️ {issue}"
                        
                        await message.reply(analysis_message)
                        
                        # 응답 완료 표시
                        async with message_lock:
                            responded_messages.add(message_id)
                        return
                    else:
                        await message.reply(f"코드 분석 중 오류가 발생했습니다: {analysis_result['error']}")
                        
                        # 응답 완료 표시
                        async with message_lock:
                            responded_messages.add(message_id)
                        return
        
        # 특수 기능은 이미 위에서 처리됨 - 여기서는 일반 AI 대화만 진행
        
        # 🛡️ 보안 검사 (관리자는 제한 없음)
        if not user_manager.is_super_admin(user_id):
            rate_check, rate_msg = user_manager.advanced_rate_limit(user_id)
            if not rate_check:
                await message.reply(f"⏰ 잠시만요! {rate_msg}. 조금 천천히 대화해주세요")
                
                # 응답 완료 표시
                async with message_lock:
                    responded_messages.add(message_id)
                return
            
            if len(user_input) > 3000:
                await message.reply("📝 메시지가 너무 길어요. 3000자 이하로 줄여주시면 감사하겠어요")
                
                # 응답 완료 표시
                async with message_lock:
                    responded_messages.add(message_id)
                return
            
            if is_spam_or_abuse(user_input, user_id):
                user_manager.blocked_users.add(user_id)
                await message.reply("🚫 부적절한 메시지가 감지되었어요. 정상적으로 대화해주세요")
                logger.warning(f"사용자 {user_id} 스팸 감지로 차단")
                
                # 응답 완료 표시
                async with message_lock:
                    responded_messages.add(message_id)
                return

        # 💬 일반 AI 대화 처리
        async with message.channel.typing():
            # 사용자 정보 업데이트
            await user_manager.update_user_stats(user_id, username)
            
            # 응답 캐시 확인 (중복 방지)
            cache_key = f"{user_id}:{hashlib.md5(user_input.encode()).hexdigest()}"
            async with cache_lock:
                if cache_key in response_cache:
                    cached_time = response_cache[cache_key]['timestamp']
                    if time.time() - cached_time < 30:  # 30초 내 같은 질문은 캐시 사용
                        logger.info(f"캐시된 응답 사용: {username}")
                        await message.reply(response_cache[cache_key]['response'])
                        
                        # 응답 완료 표시
                        async with message_lock:
                            responded_messages.add(message_id)
                        return
            
            # 채팅 세션 관리
            if user_id not in user_manager.user_chats:
                model = user_manager.get_user_model(user_id)
                user_manager.user_chats[user_id] = model.start_chat()
                logger.info(f"새 채팅 세션 시작: {username} ({user_id})")
            
            # AI 응답 생성 (중복 방지)
            chat_session = user_manager.user_chats[user_id]
            
            try:
                response = chat_session.send_message(user_input)
                # 기본 응답 텍스트
                response_text = response.text.strip()
                    
            except Exception as api_error:
                logger.error(f"Gemini API 오류: {api_error}")
                await message.reply("죄송해요, 일시적인 문제가 발생했어요. 잠시 후 다시 시도해주세요.")
                
                # 응답 완료 표시
                async with message_lock:
                    responded_messages.add(message_id)
                return
            
            # 응답 강화 시스템 비활성화 (중복 응답 방지)
            
            # 대화 이력 관리 (메모리 최적화)
            if len(chat_session.history) > 40:  # 더 긴 대화 허용
                model = user_manager.get_user_model(user_id)
                user_manager.user_chats[user_id] = model.start_chat()
                await message.channel.send("💭 대화가 길어져서 새로운 세션으로 전환했어요. 계속 대화해요!")
            
            # 응답을 캐시에 저장 (중복 방지)
            async with cache_lock:
                response_cache[cache_key] = {
                    'response': response_text,
                    'timestamp': time.time()
                }
                # 캐시 크기 제한 (메모리 관리)
                if len(response_cache) > 1000:
                    oldest_key = min(response_cache.keys(), key=lambda k: response_cache[k]['timestamp'])
                    del response_cache[oldest_key]
            
            # 응답 전송 (스마트 분할)
            if len(response_text) > 2000:
                # 자연스러운 위치에서 분할
                chunks = []
                current_chunk = ""
                
                for paragraph in response_text.split('\n\n'):
                    if len(current_chunk) + len(paragraph) > 1900:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = paragraph
                    else:
                        current_chunk += '\n\n' + paragraph if current_chunk else paragraph
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 첫 번째 청크는 reply로, 나머지는 일반 메시지로
                if chunks:
                    await message.reply(chunks[0])
                    for chunk in chunks[1:]:
                        await message.channel.send(chunk)
                        await asyncio.sleep(0.5)  # 스팸 방지
            else:
                await message.reply(response_text)
            
            # 대화 저장
            model_used = user_manager.user_preferences.get(user_id, {}).get("model", "flash")
            await user_manager.save_conversation(user_id, user_input, response_text, model_used)
            
            # 응답 완료 표시
            async with message_lock:
                responded_messages.add(message_id)
            
            # 성능 메트릭 업데이트
            processing_time = time.time() - start_time
            user_manager.update_performance_metrics("chat_response_time", processing_time)
            
            logger.info(f"대화 처리 완료: {username} ({processing_time:.2f}초)")
                
    except Exception as e:
        logger.error(f"고급 채팅 처리 오류 (사용자: {username}): {e}")
        await message.reply("😅 처리 중 문제가 발생했어요. 잠시 후 다시 시도해주세요")
        
        # 응답 완료 표시
        async with message_lock:
            responded_messages.add(message_id)
        
        # 오류 발생 시 세션 초기화
        if user_id in user_manager.user_chats:
            del user_manager.user_chats[user_id]
    
    finally:
        # 처리 완료 후 메시지 추적에서 제거
        async with message_lock:
            processing_messages.discard(message_id)

# 고급 백그라운드 작업들
async def advanced_cleanup_task():
    """고급 정리 작업"""
    while True:
        try:
            current_time = time.time()
            
            # 2시간 이상 비활성 세션 정리
            inactive_users = []
            for user_id, rate_deque in user_manager.rate_limits.items():
                if rate_deque and current_time - rate_deque[-1] > 7200:  # 2시간
                    inactive_users.append(user_id)
            
            for user_id in inactive_users:
                if user_id in user_manager.user_chats:
                    del user_manager.user_chats[user_id]
                if user_id in user_manager.rate_limits:
                    del user_manager.rate_limits[user_id]
                if user_id in user_manager.user_contexts:
                    del user_manager.user_contexts[user_id]
            
            # 데이터베이스 정리 (30일 이상 된 대화 기록)
            async with aiosqlite.connect('lucia_bot.db') as db:
                await db.execute('''
                    DELETE FROM conversations 
                    WHERE timestamp < datetime('now', '-30 days')
                ''')
                await db.commit()
            
            # 성능 메트릭 정리
            for metric_name in list(user_manager.performance_metrics.keys()):
                metrics = user_manager.performance_metrics[metric_name]
                # 24시간 이상 된 메트릭 제거
                user_manager.performance_metrics[metric_name] = [
                    m for m in metrics if current_time - m['timestamp'] < 86400
                ]
            
            # 응답 캐시 정리 (중복 방지 시스템)
            async with cache_lock:
                expired_keys = []
                for cache_key, cache_data in response_cache.items():
                    if current_time - cache_data['timestamp'] > 300:  # 5분 이상 된 캐시 제거
                        expired_keys.append(cache_key)
                
                for key in expired_keys:
                    del response_cache[key]
                
                if expired_keys:
                    logger.info(f"🗑️ 응답 캐시 정리: {len(expired_keys)}개 항목 제거")
            
            # 응답 완료 메시지 목록 정리 (메모리 관리)
            async with message_lock:
                if len(responded_messages) > 50000:  # 5만개 초과시 절반으로 줄임
                    old_messages = list(responded_messages)[:25000]
                    for old_msg in old_messages:
                        responded_messages.discard(old_msg)
                    logger.info(f"🗑️ 응답 완료 메시지 목록 정리: {len(old_messages)}개 항목 제거")
            
            if inactive_users:
                logger.info(f"🧹 정리 완료: 비활성 세션 {len(inactive_users)}개, 메모리 최적화")
                
        except Exception as e:
            logger.error(f"정리 작업 오류: {e}")
        
        await asyncio.sleep(7200)  # 2시간마다 실행

async def performance_monitor():
    """성능 모니터링"""
    while True:
        try:
            # 시스템 리소스 모니터링
            system_info = SystemMonitor.get_system_info()
            
            # 메트릭 저장
            async with aiosqlite.connect('lucia_bot.db') as db:
                await db.execute('''
                    INSERT INTO system_stats (metric_name, metric_value)
                    VALUES (?, ?)
                ''', ('cpu_usage', str(system_info.get('cpu_percent', 0))))
                
                await db.execute('''
                    INSERT INTO system_stats (metric_name, metric_value)
                    VALUES (?, ?)
                ''', ('memory_usage', str(system_info.get('memory_percent', 0))))
                
                await db.execute('''
                    INSERT INTO system_stats (metric_name, metric_value)
                    VALUES (?, ?)
                ''', ('active_sessions', str(len(user_manager.user_chats))))
                
                await db.commit()
            
            # 경고 임계값 체크
            if system_info.get('cpu_percent', 0) > 80:
                logger.warning(f"⚠️ 높은 CPU 사용률: {system_info['cpu_percent']:.1f}%")
            
            if system_info.get('memory_percent', 0) > 85:
                logger.warning(f"⚠️ 높은 메모리 사용률: {system_info['memory_percent']:.1f}%")
            
            # 성능 통계 로깅
            active_sessions = len(user_manager.user_chats)
            total_users = len(user_manager.user_preferences)
            
            logger.info(f"📊 성능 현황: CPU {system_info.get('cpu_percent', 0):.1f}%, "
                       f"메모리 {system_info.get('memory_percent', 0):.1f}%, "
                       f"활성 세션 {active_sessions}개, 총 사용자 {total_users}명")
                
        except Exception as e:
            logger.error(f"성능 모니터링 오류: {e}")
        
        await asyncio.sleep(300)  # 5분마다 실행

# 초고급 시스템 초기화 함수
async def initialize_advanced_systems():
    """모든 초고급 시스템들을 초기화합니다"""
    logger.info("🚀 초고급 시스템 초기화 시작...")
    
    try:
        # 음성 처리 시스템 초기화
        logger.info("🎵 음성 처리 시스템 초기화 중...")
        await voice_processor.initialize()
        
        # 번역 시스템 초기화
        logger.info("🌍 번역 시스템 초기화 중...")
        await translation_system.initialize()
        
        # 게임 시스템 초기화
        logger.info("🎮 게임 시스템 초기화 중...")
        await game_system.initialize()
        
        # 코드 분석기 초기화
        logger.info("💻 코드 분석 시스템 초기화 중...")
        await code_analyzer.initialize()
        
        # 이미지 분석기 초기화
        logger.info("🖼️ 이미지 분석 시스템 초기화 중...")
        await image_analyzer.initialize()
        
        # 스케줄러 초기화
        logger.info("⏰ 스케줄러 시스템 초기화 중...")
        smart_scheduler.scheduler.start()
        await smart_scheduler.initialize()
        
        # 보안 모더레이터 초기화
        logger.info("🔒 보안 모더레이션 시스템 초기화 중...")
        await security_moderator.initialize()
        
        logger.info("✅ 모든 초고급 시스템 초기화 완료!")
        
    except Exception as e:
        logger.error(f"초고급 시스템 초기화 오류: {e}")
        print(f"⚠️ 일부 고급 기능이 제한될 수 있습니다: {e}")

# 봇 실행
if __name__ == "__main__":
    try:
        print("🤖 루시아 봇 시작 중...")
        print("📦 초고급 기능들을 로딩하고 있습니다...")
        
        # 이벤트 루프 생성 및 초고급 시스템 초기화
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 초고급 시스템들 초기화
        loop.run_until_complete(initialize_advanced_systems())
        
        print("🚀 모든 시스템이 준비되었습니다!")
        print("💫 루시아 봇이 초고급 모드로 실행됩니다!")
        
        # 봇 실행
        client.run(DISCORD_TOKEN)
        
    except KeyboardInterrupt:
        print("\n👋 봇이 안전하게 종료됩니다...")
        logger.info("사용자에 의해 봇 종료")
    except Exception as e:
        logger.error(f"봇 실행 오류: {e}")
        print(f"❌ 봇 실행 실패: {e}")
        print("🔧 설정을 확인하고 다시 시도해주세요.")