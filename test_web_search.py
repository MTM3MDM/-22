"""
지능형 웹 검색 시스템 테스트 스크립트
"""

import asyncio
import os
from dotenv import load_dotenv
from intelligent_web_search import initialize_web_search, search_and_remember, get_search_statistics

# 환경변수 로드
load_dotenv()

async def test_web_search():
    """웹 검색 시스템 테스트"""
    print("🔍 지능형 웹 검색 시스템 테스트 시작...")
    
    # SERPAPI 키 설정 (테스트용)
    serpapi_key = os.getenv("SERPAPI_KEY", "YOUR_SERPAPI_KEY_HERE")
    
    if serpapi_key == "YOUR_SERPAPI_KEY_HERE":
        print("⚠️  SERPAPI_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 SERPAPI_KEY=your_actual_key 를 추가하세요.")
        print("   SerpAPI 키는 https://serpapi.com 에서 무료로 받을 수 있습니다.")
        return
    
    try:
        # 웹 검색 시스템 초기화
        web_searcher = await initialize_web_search(serpapi_key)
        print("✅ 웹 검색 시스템 초기화 완료")
        
        # 테스트 검색어들
        test_queries = [
            "GPT-5 언제 나와?",
            "파이썬 3.12 새로운 기능",
            "디스코드 봇 만들기",
            "GPT-5는 어떤 모델이야?"  # 관련 질문
        ]
        
        print("\n" + "="*50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 테스트 {i}: {query}")
            print("-" * 30)
            
            result = await search_and_remember(query)
            
            if result.get('type') == 'error':
                print(f"❌ 오류: {result.get('error')}")
            elif result.get('type') == 'memory_based':
                print(f"🧠 기억에서 찾은 답변 (유사도: {result.get('similarity_score', 0):.2f})")
                print(f"원본 질문: {result.get('original_query')}")
                print(f"답변: {result.get('answer')[:200]}...")
            elif result.get('type') == 'new_search':
                print("🔍 새로운 검색 수행")
                print(f"답변: {result.get('answer')[:200]}...")
                
                related = result.get('related_memories', [])
                if related:
                    print(f"🔗 관련 기억 {len(related)}개 발견")
            
            # 잠시 대기 (API 제한 고려)
            await asyncio.sleep(2)
        
        # 통계 조회
        print("\n" + "="*50)
        print("📊 검색 통계")
        print("-" * 30)
        
        stats = await get_search_statistics()
        if stats:
            print(f"총 검색 수: {stats.get('total_searches', 0)}")
            print(f"최근 24시간: {stats.get('recent_searches', 0)}")
            print(f"평균 응답 시간: {stats.get('avg_response_time', 0)}초")
            print(f"기억 저장소: {stats.get('memory_size', 0)}개")
            print(f"캐시 크기: {stats.get('cache_size', 0)}개")
        
        print("\n✅ 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(test_web_search())