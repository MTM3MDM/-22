"""
다양한 응답 시스템 테스트 스크립트
"""

import asyncio
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

async def test_dynamic_responses():
    """다양한 응답 시스템 테스트"""
    print("🎨 다양한 응답 시스템 테스트 시작...")
    
    try:
        # 시스템 초기화
        from dynamic_response_system import initialize_dynamic_responses, get_dynamic_response
        
        print("✅ 시스템 초기화 중...")
        await initialize_dynamic_responses()
        print("✅ 초기화 완료!")
        
        # 테스트 사용자 ID
        test_user_id = "test_user_123"
        
        # 테스트 질문들
        test_cases = [
            ("뭐해?", "activity"),
            ("뭐하고 있어?", "activity"),
            ("지금 뭐하는 중이야?", "activity"),
            ("뭐해?", "activity"),  # 반복 질문
            ("뭐하고 있어?", "activity"),  # 반복 질문
            ("안녕하세요", "greeting"),
            ("안녕!", "greeting"),
            ("반가워요", "greeting"),
            ("안녕하세요", "greeting"),  # 반복 질문
        ]
        
        print("\n" + "="*60)
        print("🧪 응답 다양성 테스트")
        print("="*60)
        
        for i, (question, q_type) in enumerate(test_cases, 1):
            print(f"\n🔍 테스트 {i}: {question} ({q_type})")
            print("-" * 40)
            
            try:
                response = await get_dynamic_response(test_user_id, question, q_type)
                print(f"🤖 응답: {response}")
                
                # 잠시 대기 (실제 사용 시뮬레이션)
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"❌ 오류: {e}")
        
        print("\n" + "="*60)
        print("✅ 테스트 완료!")
        print("="*60)
        
        # 응답 패턴 분석
        print("\n📊 응답 패턴 분석:")
        print("- 같은 질문에 대해 다른 응답이 생성되는지 확인")
        print("- 반복 질문 감지 기능 작동 확인")
        print("- 다양한 스타일 적용 확인")
        print("- 기술 키워드 포함 여부 확인")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

async def test_keyword_system():
    """기술 키워드 시스템 테스트"""
    print("\n🔧 기술 키워드 시스템 테스트...")
    
    try:
        from dynamic_response_system import dynamic_response_system, update_keywords_from_search
        
        if dynamic_response_system is None:
            print("❌ 시스템이 초기화되지 않았습니다.")
            return
        
        # 가짜 검색 결과로 키워드 업데이트 테스트
        fake_search_results = {
            'results': [
                {
                    'title': 'GPT-5 출시 소식과 새로운 AI 기능들',
                    'snippet': 'OpenAI에서 발표한 GPT-5는 이전 모델보다 향상된 성능을 보여줍니다. Python 3.12와 함께 사용하면 더욱 효과적입니다.'
                },
                {
                    'title': 'React 19 업데이트와 TypeScript 지원',
                    'snippet': '최신 React 19 버전에서는 TypeScript 지원이 크게 개선되었으며, Next.js와의 호환성도 향상되었습니다.'
                }
            ]
        }
        
        print("📝 가짜 검색 결과로 키워드 업데이트 테스트...")
        await update_keywords_from_search(fake_search_results)
        print("✅ 키워드 업데이트 완료!")
        
        # 업데이트된 키워드로 응답 생성 테스트
        print("\n🎯 업데이트된 키워드를 사용한 응답 생성 테스트...")
        for i in range(3):
            response = await dynamic_response_system.generate_activity_response("test_user", "뭐해?")
            print(f"응답 {i+1}: {response}")
            await asyncio.sleep(0.5)
        
    except Exception as e:
        print(f"❌ 키워드 시스템 테스트 오류: {e}")

if __name__ == "__main__":
    asyncio.run(test_dynamic_responses())
    asyncio.run(test_keyword_system())