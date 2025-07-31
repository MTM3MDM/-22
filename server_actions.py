"""
server_actions.py

서버의 채널, 역할, 권한, 시스템 메시지를 관리하는 액션들을 정의합니다.
"""
import discord
from discord import Guild, CategoryChannel, TextChannel, Role, PermissionOverwrite
from typing import Dict, Any, List
import json

async def execute_action(intent: str, params: Dict[str, Any], guild: Guild) -> Dict[str, Any]:
    """
    intent에 따라 적절한 서버 액션을 실행하고 결과 메시지를 반환합니다.
    """
    action_map = {
        "configure_server": configure_server,
        "modify_system_message": modify_system_message,
        "lock_channels": lock_channels,
        "unlock_channels": unlock_channels,
        "show_available_actions": show_available_actions,
        "reorder_channel": reorder_channel,
        "delete_channel": delete_channel,
        "change_design_mode": change_design_mode,
        "set_bot_name": set_bot_name,
        "set_channel_topic": set_channel_topic,
    }

    action_func = action_map.get(intent)
    
    if action_func:
        # params에 guild 객체를 추가하여 전달
        return await action_func(guild=guild, **params)
    else:
        return {"message": f"알 수 없는 관리자 명령입니다: {intent}"}

async def configure_server(guild: Guild, **kwargs) -> Dict[str, Any]:
    # 예시: 기본 카테고리 및 채널 생성
    try:
        # 기존에 관리 카테고리가 있는지 확인
        existing_category = discord.utils.get(guild.categories, name="🛠️ 관리 카테고리")
        if existing_category:
            return {"message": "이미 '관리 카테고리'가 존재합니다."}

        overwrites = {guild.default_role: PermissionOverwrite(send_messages=False)}
        category = await guild.create_category("🛠️ 관리 카테고리", overwrites=overwrites)
        await guild.create_text_channel("관리-로그", category=category)
        await guild.create_text_channel("공지사항", category=category)
        return {"message": "✅ 서버 기본 구성(관리 카테고리, 채널)을 완료했습니다."}
    except discord.Forbidden:
        return {"message": "❌ 권한 부족: 채널을 생성할 수 없습니다."}
    except Exception as e:
        return {"message": f"❌ 서버 구성 중 오류 발생: {e}"}

async def modify_system_message(guild: Guild, target: str = None, **kwargs) -> Dict[str, Any]:
    # 이 함수는 '봇 소개 변경' 등 더 구체적인 기능으로 대체될 수 있습니다.
    # 현재는 사용되지 않으므로 플레이스홀더로 둡니다.
    return {"message": "이 기능은 현재 지원되지 않습니다. '봇 이름 변경' 등을 사용해보세요."}

async def lock_channels(guild: Guild, **kwargs) -> Dict[str, Any]:
    try:
        locked_count = 0
        for channel in guild.text_channels:
            # 봇이 권한을 수정할 수 있는 채널만 잠금
            if channel.permissions_for(guild.me).manage_permissions:
                await channel.set_permissions(guild.default_role, send_messages=False)
                locked_count += 1
        return {"message": f"✅ {locked_count}개의 텍스트 채널을 잠갔습니다."}
    except discord.Forbidden:
        return {"message": "❌ 권한 부족: 채널 권한을 수정할 수 없습니다."}
    except Exception as e:
        return {"message": f"❌ 채널 잠금 중 오류 발생: {e}"}

async def unlock_channels(guild: Guild, **kwargs) -> Dict[str, Any]:
    try:
        unlocked_count = 0
        for channel in guild.text_channels:
            if channel.permissions_for(guild.me).manage_permissions:
                await channel.set_permissions(guild.default_role, send_messages=True)
                unlocked_count += 1
        return {"message": f"✅ {unlocked_count}개의 텍스트 채널을 열었습니다."}
    except discord.Forbidden:
        return {"message": "❌ 권한 부족: 채널 권한을 수정할 수 없습니다."}
    except Exception as e:
        return {"message": f"❌ 채널 잠금 해제 중 오류 발생: {e}"}

async def reorder_channel(guild: Guild, target: str = None, position: int = 0, **kwargs) -> Dict[str, Any]:
    if not target:
        return {"message": "ℹ️ 채널 이름과 위치를 함께 입력해주세요. 예: `채널 순서 변경 일반 1`"}
    
    channel = discord.utils.get(guild.channels, name=target)
    if not channel:
        return {"message": f"❌ '{target}' 채널을 찾을 수 없습니다."}
        
    try:
        await channel.edit(position=position)
        return {"message": f"✅ '{target}' 채널을 순서 {position}(으)로 이동했습니다."}
    except discord.Forbidden:
        return {"message": f"❌ 권한 부족: 채널 순서를 변경할 수 없습니다."}
    except Exception as e:
        return {"message": f"❌ 채널 순서 변경 중 오류: {e}"}

async def delete_channel(guild: Guild, target: str = None, **kwargs) -> Dict[str, Any]:
    if not target:
        return {"message": "ℹ️ 삭제할 채널 이름을 입력해주세요. 예: `채널 삭제 일반`"}

    channel = discord.utils.get(guild.channels, name=target)
    if not channel:
        return {"message": f"❌ '{target}' 채널을 찾을 수 없습니다."}
        
    try:
        await channel.delete(reason="봇 관리자 명령에 의해 삭제됨")
        return {"message": f"✅ '{target}' 채널을 삭제했습니다."}
    except discord.Forbidden:
        return {"message": f"❌ 권한 부족: 채널을 삭제할 수 없습니다."}
    except Exception as e:
        return {"message": f"❌ 채널 삭제 중 오류: {e}"}

async def change_design_mode(guild: Guild, target: str = None, **kwargs) -> Dict[str, Any]:
    if not target:
        return {"message": "ℹ️ 변경할 모드 이름을 입력해주세요. 예: `디자인 모드 프로페셔널`"}
        
    mode = target.lower()
    new_nick = guild.me.nick # 현재 닉네임 유지
    
    if "프로페셔널" in mode or "professional" in mode:
        new_nick = "루시아 (Professional)"
        message = "✅ 봇 모드를 '프로페셔널'로 변경했습니다."
    elif "디자인" in mode or "design" in mode:
        new_nick = "루시아 (Design Mode)"
        message = "✅ 봇 모드를 '디자인'으로 변경했습니다."
    elif "기본" in mode or "default" in mode:
        new_nick = "루시아" # 기본 이름
        message = "✅ 봇 모드를 '기본'으로 변경했습니다."
    else:
        return {"message": f"❌ 알 수 없는 모드입니다: {target}. (프로페셔널, 디자인, 기본)"}

    try:
        await guild.me.edit(nick=new_nick)
        return {"message": message}
    except discord.Forbidden:
        return {"message": "❌ 권한 부족: 봇의 닉네임을 변경할 수 없습니다."}
    except Exception as e:
        return {"message": f"❌ 모드 변경 중 오류: {e}"}

async def set_bot_name(guild: Guild, target: str = None, **kwargs) -> Dict[str, Any]:
    if not target:
        return {"message": "ℹ️ 변경할 봇의 새 이름을 입력해주세요."}
    try:
        await guild.me.edit(nick=target)
        return {"message": f"✅ 봇의 이름을 '{target}'(으)로 변경했습니다."}
    except discord.Forbidden:
        return {"message": "❌ 권한 부족: 봇의 이름을 변경할 수 없습니다."}
    except Exception as e:
        return {"message": f"❌ 이름 변경 중 오류: {e}"}

async def set_channel_topic(guild: Guild, target: str = None, **kwargs) -> Dict[str, Any]:
    # 이 기능은 특정 채널을 대상으로 해야 하므로, 파라미터 분석이 더 필요합니다.
    # 예: "채널 주제 변경 일반 새로운 주제입니다"
    # 현재 구현에서는 단순화하여, 명령이 실행된 채널의 주제를 변경합니다.
    # 이 기능을 사용하려면 `handle_admin_command`에서 `message` 객체를 넘겨받아야 합니다.
    # 지금은 플레이스홀더로 남겨둡니다.
    return {"message": "이 기능은 현재 개발 중입니다. 특정 채널의 주제를 변경하려면 더 복잡한 명령 분석이 필요합니다."}


# 관리자 명령 목록 제공
async def show_available_actions(**kwargs) -> Dict[str, Any]:
    actions = [
        "`서버 구성`",
        "`봇 이름 변경 [새 이름]`",
        "`전체 채널 잠금` / `전체 채널 해제`",
        "`채널 순서 변경 [채널명] [순서]`",
        "`채널 삭제 [채널명]`",
        "`디자인 모드 [모드명]` (프로페셔널, 디자인, 기본)",
        "`명령어 목록` (현재 목록)"
    ]
    message = "**사용 가능한 관리자 명령:**\n" + "\n".join(f"- {a}" for a in actions)
    return {"message": message}
