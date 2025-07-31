"""
Replit용 메인 실행 파일
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Replit 환경 감지 및 최적화
if "REPL_ID" in os.environ:
    print("🌐 Replit 환경에서 실행 중...")
    
    # 메모리 최적화
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    
    # 임시 디렉토리 사용 (Replit의 제한된 저장공간 고려)
    os.environ["DATABASE_PATH"] = "/tmp/lucia_bot.db"
    os.environ["SEARCH_DB_PATH"] = "/tmp/search_memory.db"
    
    # 필요한 디렉토리 생성
    Path("/tmp/knowledge_base").mkdir(exist_ok=True)
    Path("/tmp/knowledge_base/chroma_db").mkdir(exist_ok=True)

def check_environment():
    """환경 변수 확인"""
    required_vars = {
        "GEMINI_API_KEY": "Gemini API 키가 필요합니다. Google AI Studio에서 발급받으세요.",
        "DISCORD_TOKEN": "Discord 봇 토큰이 필요합니다. Discord Developer Portal에서 발급받으세요."
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"❌ {var}: {description}")
        else:
            print(f"✅ {var}: 설정됨")
    
    if missing_vars:
        print("\n🚨 필수 환경변수가 설정되지 않았습니다:")
        for var in missing_vars:
            print(var)
        print("\nReplit Secrets에서 환경변수를 설정한 후 다시 실행해주세요.")
        return False
    
    # 선택적 환경변수 확인
    if os.getenv("SERPAPI_KEY"):
        print("✅ SERPAPI_KEY: 설정됨 (웹 검색 기능 활성화)")
    else:
        print("⚠️  SERPAPI_KEY: 미설정 (웹 검색 기능 제한됨)")
        print("   SerpAPI 키를 설정하면 고급 웹 검색 기능을 사용할 수 있습니다.")
    
    return True

def main():
    """메인 실행 함수"""
    print("🚀 루시아 디스코드 봇 시작")
    print("=" * 50)
    
    # 환경 확인
    if not check_environment():
        sys.exit(1)
    
    print("\n🤖 봇 초기화 중...")
    
    try:
        # 메인 봇 실행
        from gemini_discord_bot import client, DISCORD_TOKEN
        
        print("✅ 봇 모듈 로드 완료")
        print("🔗 Discord에 연결 중...")
        
        # 봇 실행
        client.run(DISCORD_TOKEN)
        
    except ImportError as e:
        print(f"❌ 모듈 로드 실패: {e}")
        print("필요한 패키지가 설치되지 않았을 수 있습니다.")
        print("requirements.txt의 패키지들을 설치해주세요.")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ 봇 실행 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()