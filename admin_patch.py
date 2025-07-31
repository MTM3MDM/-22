#!/usr/bin/env python3
# 최고관리자 전용 기능 패치 스크립트

import re

def patch_admin_features():
    """기존 봇에 최고관리자 전용 기능 추가"""
    
    # 파일 읽기
    with open('gemini_discord_bot.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 최고관리자 설정 추가
    if 'SUPER_ADMINS' not in content:
        # DISCORD_TOKEN 설정 뒤에 추가
        token_pattern = r'(DISCORD_TOKEN = os\.getenv\("DISCORD_TOKEN"\).*?\n)'
        admin_config = '''
# 최고관리자 설정 (절대 변경 불가)
SUPER_ADMINS = ["1295232354205569075"]  # 최고관리자 ID 목록

'''
        content = re.sub(token_pattern, r'\1' + admin_config, content, flags=re.DOTALL)
    
    # 2. 사용자 활동 감시 함수 추가
    if 'update_user_activity' not in content:
        activity_monitor = '''
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

'''
        # init_database 함수 뒤에 추가
        init_pattern = r'(logger\.info\("데이터베이스 초기화 완료"\)\n)'
        content = re.sub(init_pattern, r'\1' + activity_monitor, content, flags=re.DOTALL)
    
    # 3. is_super_admin 함수 추가 (UserManager에)
    if 'def is_super_admin' not in content:
        admin_check = '''
    def is_super_admin(self, user_id: str) -> bool:
        """최고관리자 확인"""
        return str(user_id) in SUPER_ADMINS
    
'''
        # AdvancedUserManager 클래스 내부에 추가
        manager_pattern = r'(class AdvancedUserManager:.*?def __init__\(self\):.*?\n)'
        content = re.sub(manager_pattern, r'\1' + admin_check, content, flags=re.DOTALL)
    
    # 4. on_message 함수 수정 - 관리자 권한 체크 추가
    if '# 최고관리자 권한 확인' not in content:
        # on_message 함수 찾기
        message_pattern = r'(@client\.event\nasync def on_message\(message\):.*?if message\.author == client\.user:\s+return.*?\n)'
        
        admin_check_code = '''
    user_id = message.author.id
    username = str(message.author.display_name)
    channel_name = message.channel.name if hasattr(message.channel, 'name') else 'DM'
    server_name = message.guild.name if message.guild else 'DM'
    
    # 모든 메시지 감시 및 로깅
    try:
        log_message = f"📊 [감시] {server_name}#{channel_name} | {username} ({user_id}): {message.content[:100]}{'...' if len(message.content) > 100 else ''}"
        logger.info(log_message)
        
        # 중요 키워드 감지 시 관리자 알림
        alert_keywords = ['관리자', 'admin', '해킹', 'hack', '봇', 'bot', '루시아', '문제', '오류']
        if any(keyword in message.content.lower() for keyword in alert_keywords):
            for admin_id in SUPER_ADMINS:
                try:
                    admin_user = await client.fetch_user(int(admin_id))
                    alert_msg = f"🚨 **감시 알림**\\n서버: {server_name}\\n채널: #{channel_name}\\n사용자: {username} ({user_id})\\n내용: {message.content[:200]}"
                    await admin_user.send(alert_msg)
                except:
                    pass
        
        # 사용자 활동 기록
        await update_user_activity(str(user_id), username, server_name, channel_name)
        
    except Exception as e:
        logger.error(f"감시 시스템 오류: {e}")
    
'''
        content = re.sub(message_pattern, r'\1' + admin_check_code, content, flags=re.DOTALL)
    
    # 5. 관리자 권한 체크 추가
    if '# 최고관리자 권한 확인' not in content:
        # 멘션 체크 부분 뒤에 권한 체크 추가
        mention_pattern = r'(# 멘션 또는 DM인지 확인.*?return\n)'
        
        permission_check = '''
    # 최고관리자 권한 확인
    if not user_manager.is_super_admin(str(user_id)):
        # 일반 사용자에게는 제한된 응답
        restricted_response = f"""🔒 **접근 제한**

안녕하세요 {username}님! 현재 루시아는 **관리자 전용 모드**로 운영 중입니다.

📊 **현재 상태:**
• 일반 사용자 채팅 기능: ❌ 비활성화
• 감시 시스템: ✅ 활성화
• 관리자 전용 기능: ✅ 활성화

🔍 **알림:** 모든 활동이 관리자에게 보고됩니다.

관리자 권한이 필요한 경우 시스템 관리자에게 문의하세요."""
        
        await message.reply(restricted_response)
        
        # 관리자에게 접근 시도 알림
        for admin_id in SUPER_ADMINS:
            try:
                admin_user = await client.fetch_user(int(admin_id))
                access_attempt = f"🚨 **접근 시도 알림**\\n사용자: {username} ({user_id})\\n서버: {server_name}\\n채널: #{channel_name}\\n시도한 명령: {user_input[:100]}\\n시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                await admin_user.send(access_attempt)
            except:
                pass
        
        return

'''
        content = re.sub(mention_pattern, r'\1' + permission_check, content, flags=re.DOTALL)
    
    # 파일 저장
    with open('gemini_discord_bot.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 최고관리자 전용 기능 패치 완료!")
    print("🔒 일반 사용자는 이제 봇 사용이 제한됩니다.")
    print("📊 모든 사용자 활동이 감시되고 기록됩니다.")

if __name__ == "__main__":
    patch_admin_features()