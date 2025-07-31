"""
다양한 응답 생성 시스템
사용자 질문에 대해 매번 다른 스타일과 내용으로 응답하는 시스템
"""

import random
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import aiosqlite
import logging

logger = logging.getLogger(__name__)

class DynamicResponseSystem:
    """다양한 응답 생성 시스템"""
    
    def __init__(self, db_path: str = "lucia_bot.db"):
        self.db_path = db_path
        self.recent_responses = {}  # 사용자별 최근 응답 기록
        self.tech_keywords = []  # 최신 기술 키워드들
        self.last_keyword_update = None
        
        # 응답 스타일 패턴들 (누나 같은 말투)
        self.response_styles = {
            'casual': ['네', '거든요', '~해요', '😊'],
            'playful': ['ㅎㅎ', '그치?', '어때요?', '재밌죠?'],
            'professional': ['습니다', '네요', '거예요', '알겠어요?'],
            'cute': ['~네요', '어떻게 생각해요?', '괜찮죠?', '😌']
        }
        
        # 활동 상태 응답 패턴들 (누나 같은 말투)
        self.activity_responses = {
            'monitoring': [
                "지금 최신 뉴스 보고 있어요",
                "2025년 기술 소식들 체크하는 중이에요",
                "새로운 정보 찾아보고 있거든요",
                "데이터 정리하고 있어요, 꽤 흥미로워요",
                "실시간으로 정보 수집 중이네요"
            ],
            'learning': [
                "새로운 기술 공부하고 있어요",
                "2025년 트렌드 파악하는 중이거든요",
                "지식 업데이트하고 있어요, 재밌네요",
                "최신 AI 동향 분석하고 있어요",
                "흥미로운 기술들 배우는 중이에요"
            ],
            'helping': [
                "도움 드릴 준비 되어있어요",
                "언제든 질문해주세요",
                "궁금한 거 있으면 말해봐요",
                "뭐든 도와드릴게요",
                "대기 중이에요, 편하게 말해주세요"
            ],
            'relaxing': [
                "잠깐 쉬고 있어요",
                "여유롭게 있는 중이에요",
                "평화로운 시간이네요",
                "조용히 생각하고 있어요",
                "마음의 여유를 갖고 있어요"
            ]
        }
        
        # 반복 질문 감지 응답들 (누나 같은 말투)
        self.repeat_responses = [
            "또 궁금하시네요, 이번엔 다르게 말해볼게요",
            "같은 질문이네요? 다른 방식으로 설명해드릴게요",
            "또 물어봐주셨네요, 새로운 답변 드릴게요",
            "반복 질문이군요, 다른 관점에서 말해볼게요",
            "또 궁금한가봐요? 좀 더 자세히 설명해드릴게요"
        ]
    
    async def initialize(self):
        """시스템 초기화"""
        await self.init_response_database()
        await self.load_tech_keywords()
        logger.info("다양한 응답 시스템 초기화 완료! 🎨")
    
    async def init_response_database(self):
        """응답 기록 데이터베이스 초기화"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS response_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    question_hash TEXT,
                    response_style TEXT,
                    response_content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tech_keywords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword TEXT UNIQUE,
                    category TEXT,
                    relevance_score REAL DEFAULT 1.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.commit()
    
    async def load_tech_keywords(self):
        """최신 기술 키워드 로드"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT keyword, category FROM tech_keywords 
                    WHERE last_updated > datetime('now', '-7 days')
                    ORDER BY relevance_score DESC LIMIT 50
                """) as cursor:
                    rows = await cursor.fetchall()
                    self.tech_keywords = [{'keyword': row[0], 'category': row[1]} for row in rows]
            
            # 기본 키워드가 없으면 추가
            if not self.tech_keywords:
                await self.add_default_keywords()
            
            self.last_keyword_update = datetime.now()
            logger.info(f"기술 키워드 {len(self.tech_keywords)}개 로드 완료")
            
        except Exception as e:
            logger.error(f"기술 키워드 로드 오류: {e}")
            await self.add_default_keywords()
    
    async def add_default_keywords(self):
        """기본 기술 키워드 추가"""
        default_keywords = [
            ('GPT-5', 'AI'), ('Claude 3', 'AI'), ('Gemini Pro', 'AI'),
            ('ChatGPT', 'AI'), ('OpenAI', 'Company'), ('Google AI', 'Company'),
            ('Python 3.12', 'Programming'), ('JavaScript', 'Programming'),
            ('React 19', 'Web'), ('Next.js', 'Web'), ('TypeScript', 'Programming'),
            ('Docker', 'DevOps'), ('Kubernetes', 'DevOps'), ('AWS', 'Cloud'),
            ('블록체인', 'Blockchain'), ('NFT', 'Blockchain'), ('메타버스', 'VR/AR'),
            ('양자컴퓨터', 'Quantum'), ('5G', 'Network'), ('IoT', 'Hardware')
        ]
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for keyword, category in default_keywords:
                    await db.execute("""
                        INSERT OR IGNORE INTO tech_keywords (keyword, category, relevance_score)
                        VALUES (?, ?, ?)
                    """, (keyword, category, random.uniform(0.7, 1.0)))
                await db.commit()
                
            await self.load_tech_keywords()
            
        except Exception as e:
            logger.error(f"기본 키워드 추가 오류: {e}")
    
    def _get_question_hash(self, question: str) -> str:
        """질문 해시 생성"""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()
    
    async def check_recent_responses(self, user_id: str, question: str) -> Optional[Dict]:
        """최근 응답 기록 확인"""
        question_hash = self._get_question_hash(question)
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT response_style, response_content, timestamp 
                    FROM response_history 
                    WHERE user_id = ? AND question_hash = ? 
                    AND timestamp > datetime('now', '-1 hour')
                    ORDER BY timestamp DESC LIMIT 3
                """, (user_id, question_hash)) as cursor:
                    rows = await cursor.fetchall()
                    
                    if rows:
                        return {
                            'is_repeat': True,
                            'previous_responses': [
                                {'style': row[0], 'content': row[1], 'time': row[2]} 
                                for row in rows
                            ]
                        }
            
            return {'is_repeat': False, 'previous_responses': []}
            
        except Exception as e:
            logger.error(f"응답 기록 확인 오류: {e}")
            return {'is_repeat': False, 'previous_responses': []}
    
    def _choose_response_style(self, previous_styles: List[str]) -> str:
        """이전과 다른 응답 스타일 선택"""
        available_styles = ['casual', 'playful', 'professional', 'cute']
        
        # 이전에 사용한 스타일 제외
        for style in previous_styles:
            if style in available_styles:
                available_styles.remove(style)
        
        # 사용 가능한 스타일이 없으면 모든 스타일 사용
        if not available_styles:
            available_styles = ['casual', 'playful', 'professional', 'cute']
        
        return random.choice(available_styles)
    
    def _get_random_tech_info(self) -> Optional[str]:
        """랜덤 기술 정보 생성"""
        if not self.tech_keywords:
            return None
        
        keyword_info = random.choice(self.tech_keywords)
        keyword = keyword_info['keyword']
        category = keyword_info['category']
        
        tech_phrases = [
            f"요즘 {keyword} 소식이 흥미로워요",
            f"오늘은 {keyword} 관련 뉴스를 봤어요",
            f"{keyword}에 대한 새로운 정보가 나왔더라고요",
            f"{category} 분야에서 {keyword}가 화제네요",
            f"{keyword} 업데이트 소식을 확인하고 있어요"
        ]
        
        return random.choice(tech_phrases)
    
    def _apply_style_decorations(self, text: str, style: str) -> str:
        """스타일에 맞는 장식 추가"""
        decorations = self.response_styles.get(style, [''])
        decoration = random.choice(decorations)
        
        if style == 'casual':
            return f"{text} {decoration}"
        elif style == 'playful':
            return f"{text} {decoration}"
        elif style == 'professional':
            return f"{text}{decoration}"
        elif style == 'cute':
            return f"{text} {decoration}"
        else:
            return text
    
    async def generate_activity_response(self, user_id: str, question: str) -> str:
        """활동 상태 질문에 대한 다양한 응답 생성"""
        try:
            # 최근 응답 기록 확인
            recent_check = await self.check_recent_responses(user_id, question)
            
            # 반복 질문인지 확인
            if recent_check['is_repeat'] and recent_check['previous_responses']:
                repeat_intro = random.choice(self.repeat_responses)
                previous_styles = [resp['style'] for resp in recent_check['previous_responses']]
            else:
                repeat_intro = None
                previous_styles = []
            
            # 새로운 스타일 선택
            response_style = self._choose_response_style(previous_styles)
            
            # 활동 카테고리 선택
            activity_category = random.choice(list(self.activity_responses.keys()))
            base_response = random.choice(self.activity_responses[activity_category])
            
            # 기술 정보 추가 (10% 확률로 낮춤)
            tech_info = None
            if random.random() < 0.1:
                tech_info = self._get_random_tech_info()
            
            # 응답 조합
            response_parts = []
            
            if repeat_intro:
                response_parts.append(repeat_intro)
            
            response_parts.append(base_response)
            
            if tech_info:
                response_parts.append(tech_info)
            
            # 최종 응답 생성
            final_response = " ".join(response_parts)
            final_response = self._apply_style_decorations(final_response, response_style)
            
            # 응답 기록 저장
            await self.save_response_history(user_id, question, response_style, final_response)
            
            return final_response
            
        except Exception as e:
            logger.error(f"활동 응답 생성 오류: {e}")
            return "지금은 여러 가지 일들을 하고 있어요! ✨"
    
    async def generate_greeting_response(self, user_id: str, question: str) -> str:
        """인사 질문에 대한 다양한 응답 생성"""
        greeting_patterns = [
            "안녕하세요, 반가워요",
            "안녕! 오늘 어떻게 지내세요?",
            "안녕하세요, 만나서 기뻐요",
            "좋은 하루네요, 뭐 도와드릴까요?",
            "안녕하세요, 오늘 기분 어때요?",
            "반가워요, 어떤 이야기 나눠볼까요?",
            "안녕하세요, 편하게 말해주세요"
        ]
        
        try:
            recent_check = await self.check_recent_responses(user_id, question)
            previous_styles = [resp['style'] for resp in recent_check['previous_responses']] if recent_check['is_repeat'] else []
            
            response_style = self._choose_response_style(previous_styles)
            base_response = random.choice(greeting_patterns)
            
            # 기술 정보 추가 (5% 확률로 낮춤)
            if random.random() < 0.05:
                tech_info = self._get_random_tech_info()
                if tech_info:
                    base_response += f" {tech_info}!"
            
            final_response = self._apply_style_decorations(base_response, response_style)
            await self.save_response_history(user_id, question, response_style, final_response)
            
            return final_response
            
        except Exception as e:
            logger.error(f"인사 응답 생성 오류: {e}")
            return "안녕하세요! 반가워요~ ✨"
    
    async def save_response_history(self, user_id: str, question: str, style: str, response: str):
        """응답 기록 저장"""
        try:
            question_hash = self._get_question_hash(question)
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO response_history (user_id, question_hash, response_style, response_content)
                    VALUES (?, ?, ?, ?)
                """, (user_id, question_hash, style, response))
                await db.commit()
                
        except Exception as e:
            logger.error(f"응답 기록 저장 오류: {e}")
    
    async def update_tech_keywords_from_search(self, search_results: Dict):
        """검색 결과에서 기술 키워드 업데이트"""
        try:
            if not search_results.get('results'):
                return
            
            # 검색 결과에서 키워드 추출
            new_keywords = []
            for result in search_results['results']:
                title = result.get('title', '').lower()
                snippet = result.get('snippet', '').lower()
                
                # 기술 관련 키워드 패턴 찾기
                tech_patterns = [
                    r'gpt-?\d+', r'claude\s*\d*', r'gemini', r'chatgpt',
                    r'python\s*\d*\.\d*', r'javascript', r'typescript', r'react\s*\d*',
                    r'ai', r'머신러닝', r'딥러닝', r'블록체인', r'nft', r'메타버스'
                ]
                
                import re
                for pattern in tech_patterns:
                    matches = re.findall(pattern, title + ' ' + snippet)
                    for match in matches:
                        if len(match) > 2:  # 너무 짧은 키워드 제외
                            new_keywords.append(match.strip())
            
            # 새로운 키워드 저장
            if new_keywords:
                async with aiosqlite.connect(self.db_path) as db:
                    for keyword in set(new_keywords):  # 중복 제거
                        await db.execute("""
                            INSERT OR REPLACE INTO tech_keywords (keyword, category, relevance_score, last_updated)
                            VALUES (?, ?, ?, ?)
                        """, (keyword, 'Auto-detected', 0.8, datetime.now().isoformat()))
                    await db.commit()
                
                # 키워드 목록 새로고침
                await self.load_tech_keywords()
                
        except Exception as e:
            logger.error(f"기술 키워드 업데이트 오류: {e}")

# 전역 인스턴스
dynamic_response_system = None

async def initialize_dynamic_responses():
    """다양한 응답 시스템 초기화"""
    global dynamic_response_system
    dynamic_response_system = DynamicResponseSystem()
    await dynamic_response_system.initialize()
    return dynamic_response_system

async def get_dynamic_response(user_id: str, question: str, question_type: str = 'activity') -> str:
    """다양한 응답 생성"""
    if dynamic_response_system is None:
        return "시스템 초기화 중이에요... 잠시만 기다려주세요! 🔄"
    
    if question_type == 'activity':
        return await dynamic_response_system.generate_activity_response(user_id, question)
    elif question_type == 'greeting':
        return await dynamic_response_system.generate_greeting_response(user_id, question)
    else:
        return await dynamic_response_system.generate_activity_response(user_id, question)

async def update_keywords_from_search(search_results: Dict):
    """검색 결과로부터 키워드 업데이트"""
    if dynamic_response_system is not None:
        await dynamic_response_system.update_tech_keywords_from_search(search_results)