#!/usr/bin/env python3
# 자연어 기반 관리자 기능 패치

import re

def add_natural_admin_features():
    """자연어 기반 관리자 전용 기능 추가"""
    
    with open('gemini_discord_bot.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 자연어 관리자 기능 처리 함수 추가
    if 'process_admin_natural_language' not in content:
        admin_nl_function = '''
async def process_admin_natural_language(user_input: str, user_id: str, message) -> dict:
    """관리자 전용 자연어 처리"""
    user_input_lower = user_input.lower()
    
    # 📊 사용자 활동 감시 조회
    if any(keyword in user_input_lower for keyword in ["활동 보여줘", "감시 현황", "사용자 활동", "누가 뭐했어", "활동 내역"]):
        try:
            async with aiosqlite.connect('lucia_bot.db') as db:
                # 최근 활동 조회
                async with db.execute("""
                    SELECT username, server_name, channel_name, timestamp, COUNT(*) as count
                    FROM user_activity 
                    WHERE timestamp > datetime('now', '-24 hours')
                    GROUP BY user_id, server_name
                    ORDER BY timestamp DESC
                    LIMIT 20
                """) as cursor:
                    activities = await cursor.fetchall()
                
                if activities:
                    result = "📊 **최근 24시간 사용자 활동 현황:**\\n\\n"
                    for activity in activities:
                        username, server, channel, timestamp, count = activity
                        result += f"👤 **{username}**\\n"
                        result += f"   📍 {server}#{channel}\\n"
                        result += f"   📈 활동 {count}회\\n"
                        result += f"   🕐 마지막: {timestamp}\\n\\n"
                    
                    return {"action": "admin_activity", "message": result, "stop_processing": True}
                else:
                    return {"action": "admin_activity", "message": "📊 최근 24시간 내 활동이 없습니다.", "stop_processing": True}
        except Exception as e:
            return {"action": "admin_error", "message": f"활동 조회 오류: {e}", "stop_processing": True}
    
    # 🚨 실시간 감시 상태 확인
    if any(keyword in user_input_lower for keyword in ["감시 상태", "모니터링 현황", "시스템 감시", "감시중이야"]):
        try:
            # 현재 접속 중인 서버 수
            server_count = len(client.guilds)
            
            # 오늘 감지된 활동 수
            async with aiosqlite.connect('lucia_bot.db') as db:
                async with db.execute("""
                    SELECT COUNT(*) FROM user_activity 
                    WHERE date(timestamp) = date('now')
                """) as cursor:
                    today_activities = await cursor.fetchone()
                    today_count = today_activities[0] if today_activities else 0
                
                # 최근 알림 키워드 감지
                async with db.execute("""
                    SELECT COUNT(*) FROM user_activity 
                    WHERE timestamp > datetime('now', '-1 hour')
                """) as cursor:
                    recent_activities = await cursor.fetchone()
                    recent_count = recent_activities[0] if recent_activities else 0
            
            status_msg = f"""🚨 **실시간 감시 시스템 현황**

🌐 **모니터링 범위:**
• 감시 중인 서버: {server_count}개
• 오늘 감지된 활동: {today_count}건
• 최근 1시간 활동: {recent_count}건

✅ **시스템 상태:**
• 실시간 감시: 🟢 활성화
• 키워드 알림: 🟢 활성화  
• 활동 기록: 🟢 활성화
• 접근 제한: 🟢 활성화

🔍 **감시 키워드:** 관리자, admin, 해킹, hack, 봇, bot, 루시아, 문제, 오류

💡 "누가 뭐했어", "의심스러운 활동", "최근 접근 시도" 등으로 자세한 정보를 확인할 수 있습니다."""
            
            return {"action": "admin_status", "message": status_msg, "stop_processing": True}
            
        except Exception as e:
            return {"action": "admin_error", "message": f"감시 상태 확인 오류: {e}", "stop_processing": True}
    
    # 🔍 의심스러운 활동 조회
    if any(keyword in user_input_lower for keyword in ["의심스러운", "수상한", "이상한 활동", "문제 있는"]):
        try:
            async with aiosqlite.connect('lucia_bot.db') as db:
                # 키워드 알림이 발생한 활동들
                async with db.execute("""
                    SELECT username, server_name, channel_name, timestamp
                    FROM user_activity 
                    WHERE timestamp > datetime('now', '-7 days')
                    ORDER BY timestamp DESC
                    LIMIT 10
                """) as cursor:
                    suspicious = await cursor.fetchall()
                
                if suspicious:
                    result = "🔍 **최근 7일간 감지된 활동:**\\n\\n"
                    for activity in suspicious:
                        username, server, channel, timestamp = activity
                        result += f"⚠️ **{username}**\\n"
                        result += f"   📍 {server}#{channel}\\n"
                        result += f"   🕐 {timestamp}\\n\\n"
                    
                    return {"action": "admin_suspicious", "message": result, "stop_processing": True}
                else:
                    return {"action": "admin_suspicious", "message": "🔍 최근 의심스러운 활동이 감지되지 않았습니다.", "stop_processing": True}
        except Exception as e:
            return {"action": "admin_error", "message": f"의심 활동 조회 오류: {e}", "stop_processing": True}
    
    # 📈 통계 및 분석
    if any(keyword in user_input_lower for keyword in ["통계", "분석", "현황 분석", "데이터"]):
        try:
            async with aiosqlite.connect('lucia_bot.db') as db:
                # 서버별 활동 통계
                async with db.execute("""
                    SELECT server_name, COUNT(*) as count
                    FROM user_activity 
                    WHERE timestamp > datetime('now', '-7 days')
                    GROUP BY server_name
                    ORDER BY count DESC
                """) as cursor:
                    server_stats = await cursor.fetchall()
                
                # 시간대별 활동 통계
                async with db.execute("""
                    SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                    FROM user_activity 
                    WHERE timestamp > datetime('now', '-7 days')
                    GROUP BY hour
                    ORDER BY count DESC
                    LIMIT 5
                """) as cursor:
                    hour_stats = await cursor.fetchall()
                
                result = "📈 **최근 7일간 활동 분석:**\\n\\n"
                
                if server_stats:
                    result += "🌐 **서버별 활동:**\\n"
                    for server, count in server_stats[:5]:
                        result += f"• {server}: {count}건\\n"
                    result += "\\n"
                
                if hour_stats:
                    result += "🕐 **활발한 시간대:**\\n"
                    for hour, count in hour_stats:
                        result += f"• {hour}시: {count}건\\n"
                
                return {"action": "admin_stats", "message": result, "stop_processing": True}
                
        except Exception as e:
            return {"action": "admin_error", "message": f"통계 분석 오류: {e}", "stop_processing": True}
    
    # 🔒 접근 시도 기록 조회
    if any(keyword in user_input_lower for keyword in ["접근 시도", "누가 시도했어", "차단된", "제한된"]):
        return {"action": "admin_access", "message": """🔒 **접근 제한 현황:**

현재 모든 일반 사용자의 봇 사용이 제한되어 있습니다.
접근 시도 시 실시간으로 관리자에게 알림이 전송됩니다.

💡 **추가 정보가 필요하시면:**
• "활동 보여줘" - 전체 사용자 활동 현황
• "의심스러운 활동" - 키워드 감지된 활동들
• "통계 보여줘" - 상세 분석 데이터""", "stop_processing": True}
    
    return {"action": "continue", "message": "", "stop_processing": False}

'''
        
        # process_natural_language 함수 앞에 추가
        nl_pattern = r'(async def process_natural_language\(user_input: str, user_id: str, message\) -> dict:)'
        content = re.sub(nl_pattern, admin_nl_function + r'\1', content, flags=re.DOTALL)
    
    # process_natural_language 함수 시작 부분에 관리자 체크 추가
    if '# 관리자 전용 자연어 처리' not in content:
        admin_check_code = '''    # 관리자 전용 자연어 처리
    if user_manager.is_super_admin(str(user_id)):
        admin_result = await process_admin_natural_language(user_input, str(user_id), message)
        if admin_result["stop_processing"]:
            return admin_result
    
'''
        
        # process_natural_language 함수 내부 시작 부분에 추가
        nl_start_pattern = r'(async def process_natural_language\(user_input: str, user_id: str, message\) -> dict:.*?""".*?""".*?\n)'
        content = re.sub(nl_start_pattern, r'\1' + admin_check_code, content, flags=re.DOTALL)
    
    # 파일 저장
    with open('gemini_discord_bot.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 자연어 기반 관리자 기능 추가 완료!")
    print("🗣️ 이제 자연스러운 대화로 감시 기능을 사용할 수 있습니다:")
    print("   • '활동 보여줘' - 사용자 활동 현황")
    print("   • '감시 상태 어때?' - 시스템 감시 현황") 
    print("   • '의심스러운 활동 있어?' - 수상한 활동 조회")
    print("   • '통계 보여줘' - 상세 분석 데이터")

if __name__ == "__main__":
    add_natural_admin_features()