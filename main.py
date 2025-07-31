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
    print("🌐 Replit 환경에서 실행 중... 최적화를 적용합니다.")
    
    # 메모리 최적화
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    
    # 임시 디렉토리 사용 (Replit의 제한된 저장공간 고려)
    # 데이터베이스 경로를 환경변수로 설정 (필요 시)
    # os.environ["DATABASE_PATH"] = "/tmp/lucia_bot.db"
    
    # 필요한 디렉토리 생성
    # Path("/tmp/knowledge_base").mkdir(exist_ok=True)

def check_environment():
    """필수 환경 변수를 확인하고 설정 상태를 안내합니다."""
    print("\n--- 환경 변수 확인 시작 ---")
    required_vars = {
        "GEMINI_API_KEY": "Gemini API 키가 필요합니다. Google AI Studio에서 발급받으세요.",
        "DISCORD_TOKEN": "Discord 봇 토큰이 필요합니다. Discord Developer Portal에서 발급받으세요."
    }
    
    all_set = True
    for var, description in required_vars.items():
        if not os.getenv(var):
            print(f"❌ [필수] {var}: 설정되지 않음. ({description})")
            all_set = False
        else:
            print(f"✅ [필수] {var}: 설정됨")
    
    if not all_set:
        print("\n🚨 필수 환경변수가 설정되지 않았습니다. Replit의 'Secrets' 또는 .env 파일에서 환경변수를 설정한 후 다시 실행해주세요.")
        return False
    
    # 선택적 환경변수 확인
    if os.getenv("SERPAPI_KEY"):
        print("✅ [선택] SERPAPI_KEY: 설정됨 (고급 웹 검색 기능 활성화)")
    else:
        print("⚠️  [선택] SERPAPI_KEY: 미설정 (웹 검색 기능이 제한됩니다.)")
        print("   (SerpAPI 키를 설정하면 더 정확한 웹 검색이 가능합니다.)")
    
    print("--- 환경 변수 확인 완료 ---\n")
    return True

def main():
    """메인 실행 함수"""
    print("="*50)
    print("🚀 루시아 AI 어시스턴트 봇을 시작합니다.")
    print("="*50)
    
    # 환경 확인
    if not check_environment():
        sys.exit(1)
    
    print("🤖 봇 모듈을 로드하고 초기화합니다...")
    
    try:
        # 메인 봇 실행
        import discord
        from gemini_discord_bot import client, DISCORD_TOKEN
        
        print("✅ 봇 모듈 로드 완료.")
        print("🔗 Discord 서버에 연결을 시도합니다...")
        
        # 봇 실행
        client.run(DISCORD_TOKEN)
        
    except ImportError as e:
        print(f"❌ 모듈 로드 실패: {e}")
        print("   필요한 패키지가 설치되지 않았을 수 있습니다.")
        print("   'pip install -r requirements.txt' 명령어로 패키지를 설치해주세요.")
        sys.exit(1)
    except discord.errors.LoginFailure:
        print("❌ Discord 로그인 실패: 잘못된 토큰입니다.")
        print("   DISCORD_TOKEN이 올바른지 확인해주세요.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 봇 실행 중 예기치 않은 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()