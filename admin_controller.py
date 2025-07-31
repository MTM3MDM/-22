"""
admin_controller.py

최고관리자의 자연어 지시를 처리하는 컨트롤러.
"""
import discord
from config import get_config
from intent_router import parse_intent, FALLBACK_INTENT
from server_actions import execute_action
import logging

logger = logging.getLogger(__name__)

async def initialize_admin_system():
    """
    관리자 시스템을 초기화합니다.
    현재는 관리 기능이 이벤트 기반으로 동작하므로 플레이스홀더 역할을 합니다.
    """
    logger.info("관리 시스템이 초기화되었습니다.")
    pass

async def handle_admin_command(message: discord.Message) -> bool:
    """
    최고관리자의 메시지를 기반으로 서버 관리자 명령을 처리합니다.
    성공적으로 처리 시 embed 응답을 보내고 True를 반환합니다.
    처리되지 않은 경우 False를 반환합니다.
    """
    config = get_config()
    # super-admin ID check using string comparison
    super_id = config.get('super_admin', {}).get('id')
    if not super_id or str(message.author.id) != super_id:
        return False

    # 서버(길드) 컨텍스트에서만 관리 명령을 허용
    if not message.guild:
        # 이 메시지는 사용자에게 직접 보내지 않고, 로그만 남겨서 조용히 처리합니다.
        logger.warning(f"Admin command '{message.content}' attempted in DM by {message.author.id}. Ignored.")
        return False # DM에서는 관리 명령을 처리하지 않음

    content = message.content.strip()
    # Intent 파싱
    intent, params = parse_intent(content)
    # Only handle known admin intents
    if intent == FALLBACK_INTENT:
        return False

    try:
        # 서버 액션 실행
        result = await execute_action(intent, params, message.guild)
        if result and result.get('message'):
            embed = discord.Embed(title="✅ 서버 관리자", description=result.get('message'), color=0x00ff00)
            await message.channel.send(embed=embed)
        # 액션이 실행되었으므로 True 반환
        return True
    except Exception as e:
        logger.error(f"Admin command error: {e}", exc_info=True)
        embed = discord.Embed(title="❌ 오류", description="명령 처리 중 오류가 발생했습니다.", color=0xff0000)
        await message.channel.send(embed=embed)
        # 오류가 발생해도 명령 처리를 시도했으므로 True 반환하여 중복 처리 방지
        return True

    # 처리할 의도가 없는 경우
    return False
