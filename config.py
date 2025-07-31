# 루시아 봇 설정 파일 - 자연어 전용 (명령어 시스템 제거됨)
# 이 설정은 자연어 기반 봇을 위한 것이며, 명령어 관련 설정은 절대 추가할 수 없습니다.

import os
from typing import Dict, Any, List

# 기본 설정 (자연어 전용)
DEFAULT_CONFIG = {
    # 모델 설정 - 자연어 처리 최적화
    "models": {
        "flash": {
            "name": "gemini-1.5-flash",
            "max_tokens": 8192,
            "temperature": 0.7,
            "natural_language_only": True  # 자연어 전용 모드
        },
        "pro": {
            "name": "gemini-1.5-pro", 
            "max_tokens": 32768,
            "temperature": 0.8,
            "natural_language_only": True  # 자연어 전용 모드
        }
    },
    
    # 보안 설정 - 강화됨
    "security": {
        "rate_limit_per_minute": 10,
        "max_message_length": 2000,
        "max_history_length": 30,
        "spam_threshold": 3,
        "auto_block_duration": 3600,
        "super_admin_unlimited": True  # 최고관리자 무제한
    },
    
    # 기능 설정 - 자연어 전용
    "features": {
        "natural_language_only": True,  # 자연어만 사용 (절대 변경 불가)
        "command_system_disabled": True,  # 명령어 시스템 비활성화 (절대 변경 불가)
        "slash_commands": False,  # 슬래시 커맨드 비활성화 (절대 변경 불가)
        "button_interactions": False,  # 버튼 상호작용 비활성화 (절대 변경 불가)
        "auto_cleanup": True,
        "logging": True,
        "statistics": True,
        "natural_admin_functions": True  # 자연어 관리 기능
    },
    
    # 자연어 키워드 설정
    "natural_keywords": {
        "model_change": ["모델 바꿔", "모델 변경", "프로로 바꿔", "플래시로 바꿔"],
        "reset_chat": ["대화 초기화", "대화 리셋", "새로 시작", "처음부터"],
        "show_settings": ["설정 보여줘", "내 설정", "현재 설정"],
        "show_stats": ["통계 보여줘", "봇 통계", "사용 통계"],
        "full_reset": ["전체 초기화", "모든 데이터 초기화"],
        "block_user": ["사용자 차단", "차단해줘"],
        "unblock_user": ["차단 해제", "차단해제"],
        "configure_server": ["서버 구성", "이 서버 좀 꾸며줘", "채널 구조 맘에 안 들어", "관리 채널 하나만 따로 만들어줘"],
        "modify_system_message": ["봇 소개 넣어줘", "인사말 좀 바꿔줘", "테마 바꿔봐"],
        "lock_channels": ["전체 채널 잠궈줘", "전체 채널 잠궈", "채널 잠가줘"],
        "unlock_channels": ["전체 채널 열어줘", "다시 열어줘"],
        "show_available_actions": ["명령어 목록 보여줘", "내가 할 수 있는 게 뭐야?", "무엇을 할 수 있지?"],
        "reorder_channel": ["채널 순서 바꿔줘", "채널 순서 변경해줘"],
        "delete_channel": ["이 채널 없애줘", "채널 삭제해줘"],
        "change_design_mode": ["디자인 모드로 바꿔", "AI답게 말해봐", "프로페셔널 모드로 전환해줘"],
        "set_welcome_message": ["환영 메시지 설정", "웰컴 메시지 설정"],
        "set_farewell_message": ["작별 메시지 설정", "굿바이 메시지 설정"],
        "set_bot_status": ["봇 상태 설정", "봇 활성화", "봇 비활성화"],
        "set_language": ["언어 설정", "언어 변경"],
        "set_timezone": ["시간대 설정", "시간대 변경"],
        "backup_data": ["데이터 백업", "백업 생성"],
        "restore_data": ["데이터 복원", "복원하기"],
        "set_custom_command": ["커스텀 명령어 설정", "자주 사용하는 명령어 추가"],
        "remove_custom_command": ["커스텀 명령어 제거", "자주 사용하는 명령어 삭제"],
        "list_custom_commands": ["커스텀 명령어 목록", "자주 사용하는 명령어 보기"],
        "set_auto_responder": ["자동 응답 설정", "자동 응답기 설정"],
        "remove_auto_responder": ["자동 응답 제거", "자동 응답기 제거"],
        "list_auto_responders": ["자동 응답 목록", "자동 응답기 목록 보기"],
        "set_channel_permissions": ["채널 권한 설정", "채널 접근 설정"],
        "set_role_permissions": ["역할 권한 설정", "역할 접근 설정"],
        "set_bot_reaction": ["봇 반응 설정", "봇 이모지 설정"],
        "set_command_prefix": ["명령어 접두사 설정", "프리픽스 설정"],
        "set_help_command": ["도움말 명령어 설정", "헬프 명령어 설정"],
        "set_alias": ["별칭 설정", "알리아스 설정"],
        "set_invite_link": ["초대 링크 설정", "인비테이션 링크 설정"],
        "set_server_rules": ["서버 규칙 설정", "규칙 설정"],
        "set_welcome_channel": ["환영 채널 설정", "웰컴 채널 설정"],
        "set_farewell_channel": ["작별 채널 설정", "굿바이 채널 설정"],
        "set_bot_avatar": ["봇 아바타 설정", "봇 프로필 사진 설정"],
        "set_bot_name": ["봇 이름 설정", "봇 닉네임 설정"],
        "set_status_message": ["상태 메시지 설정", "상태 업데이트"],
        "set_activity": ["활동 설정", "현재 활동 설정"],
        "set_streaming": ["스트리밍 설정", "현재 스트리밍 설정"],
        "set_game": ["게임 설정", "현재 게임 설정"],
        "set_website": ["웹사이트 설정", "사이트 설정"],
        "set_support_channel": ["지원 채널 설정", "서포트 채널 설정"],
        "set_feedback_channel": ["피드백 채널 설정", "의견 채널 설정"],
        "set_suggestion_channel": ["제안 채널 설정", "아이디어 채널 설정"],
        "set_poll_channel": ["투표 채널 설정", "설문 채널 설정"],
        "set_event_channel": ["이벤트 채널 설정", "행사 채널 설정"],
        "set_announcement_channel": ["공지 채널 설정", "알림 채널 설정"],
        "set_private_channel": ["비공개 채널 설정", "프라이빗 채널 설정"],
        "set_public_channel": ["공개 채널 설정", "퍼블릭 채널 설정"],
        "set_category": ["카테고리 설정", "카테고리 추가"],
        "set_channel_topic": ["채널 주제 설정", "채널 설명 설정"],
        "set_channel_emoji": ["채널 이모지 설정", "채널 심볼 설정"],
        "set_channel_bitrate": ["채널 비트레이트 설정", "음질 설정"],
        "set_channel_user_limit": ["채널 사용자 제한 설정", "채널 인원 제한 설정"],
        "set_channel_nsfw": ["채널 NSFW 설정", "채널 성인 설정"],
        "set_channel_slowmode": ["채널 슬로우모드 설정", "채널 지연 모드 설정"],
        "set_channel_pinned_messages": ["채널 고정 메시지 설정", "채널 핀 메시지 설정"],
        "set_channel_webhook": ["채널 웹훅 설정", "채널 후크 설정"],
        "set_channel_bot_permissions": ["채널 봇 권한 설정", "봇 접근 설정"],
        "set_channel_role_permissions": ["채널 역할 권한 설정", "역할 접근 설정"],
        "set_channel_overwrite_permissions": ["채널 권한 덮어쓰기 설정", "오버라이트 설정"],
        "set_channel_view_channel": ["채널 보기 설정", "채널 보기 권한"],
        "set_channel_send_messages": ["채널 메시지 전송 설정", "글쓰기 권한"],
        "set_channel_read_messages": ["채널 메시지 읽기 설정", "읽기 권한"],
        "set_channel_connect": ["채널 연결 설정", "음성 채널 입장"],
        "set_channel_speak": ["채널 말하기 설정", "음성 채널 발언"],
        "set_channel_mute": ["채널 음소거 설정", "멤버 음소거"],
        "set_channel_deafen": ["채널 청각 차단 설정", "멤버 소리 차단"],
        "set_channel_move": ["채널 이동 설정", "멤버 이동"],
        "set_channel_priority_speaker": ["채널 우선 발언자 설정", "우선 발언권"],
        "set_channel_video_quality": ["채널 비디오 품질 설정", "화질 설정"],
        "set_channel_rtc_region": ["채널 RTC 지역 설정", "서버 지역 설정"],
        "set_channel_flags": ["채널 플래그 설정", "채널 속성"],
        "set_channel_application_command_permissions": ["채널 앱 명령어 권한 설정", "앱 명령어 권한"],
        "set_channel_message_tts": ["채널 TTS 설정", "TTS 메시지"],
        "set_channel_message_pins": ["채널 메시지 핀 설정", "메시지 고정"],
        "set_channel_message_reactions": ["채널 반응 설정", "이모지 반응"],
        "set_channel_message_mentions": ["채널 멘션 설정", "언급 허용"],
        "set_channel_message_stickers": ["채널 스티커 설정", "스티커 사용"],
        "set_channel_message_embeds": ["채널 임베드 설정", "링크 미리보기"],
        "set_channel_message_attachments": ["채널 첨부파일 설정", "파일 전송"],
        "set_channel_message_suppress": ["채널 메시지 억제 설정", "메시지 숨기기"],
        "set_channel_message_flags": ["채널 메시지 플래그 설정", "메시지 속성"],
        "set_channel_message_type": ["채널 메시지 유형 설정", "메시지 포맷"],
        "set_channel_message_content": ["채널 메시지 내용 설정", "메시지 본문"],
        "set_channel_message_components": ["채널 메시지 컴포넌트 설정", "메시지 버튼"],
        "set_channel_message_replies": ["채널 답글 설정", "답장 허용"],
        "set_channel_message_thread": ["채널 스레드 설정", "스레드 생성"],
        "set_channel_message_activity": ["채널 활동 설정", "메시지 활동"],
        "set_channel_message_application": ["채널 앱 설정", "메시지 앱"],
        "set_channel_message_reference": ["채널 참조 설정", "메시지 참조"],
        "set_channel_message_sticker": ["채널 스티커 아이콘 설정", "스티커 아이콘"],
        "set_channel_message_embed": ["채널 임베드 미리보기 설정", "임베드 미리보기"],
        "set_channel_message_attachment": ["채널 첨부파일 파일 설정", "첨부 파일"],
        "set_channel_message_suppress_embeds": ["채널 임베드 억제 설정", "링크 미리보기 숨기기"],
        "set_channel_message_flags_all": ["채널 모든 메시지 플래그 설정", "모든 메시지 속성"],
        "set_channel_message_type_all": ["채널 모든 메시지 유형 설정", "모든 메시지 포맷"],
        "set_channel_message_content_all": ["채널 모든 메시지 내용 설정", "모든 메시지 본문"],
        "set_channel_message_components_all": ["채널 모든 메시지 컴포넌트 설정", "모든 메시지 요소"],
        "set_channel_message_replies_all": ["채널 모든 메시지 답글 설정", "모든 메시지 회신"],
        "set_channel_message_thread_all": ["채널 모든 메시지 스레드 설정", "모든 메시지 토론"],
        "set_channel_message_activity_all": ["채널 모든 메시지 활동 설정", "모든 메시지 액티비티"],
        "set_channel_message_application_all": ["채널 모든 메시지 앱 설정", "모든 메시지 앱"],
        "set_channel_message_reference_all": ["채널 모든 메시지 참조 설정", "모든 메시지 링크"],
        "set_channel_message_sticker_all": ["채널 모든 메시지 스티커 설정", "모든 메시지 아이콘"],
        "set_channel_message_embed_all": ["채널 모든 메시지 임베드 설정", "모든 메시지 미리보기"],
        "set_channel_message_attachment_all": ["채널 모든 메시지 첨부파일 설정", "모든 메시지 파일"],
        "set_channel_message_suppress_embeds_all": ["채널 모든 메시지 임베드 억제 설정", "모든 링크 숨기기"]
    },
    
    # 메시지 설정 - 자연어 친화적
    "messages": {
        "welcome": "안녕하세요! 루시아입니다. 무엇을 도와드릴까요? 😊",
        "rate_limit": "잠깐! 너무 빨라요. 조금 천천히 말해주세요! ⏰",
        "spam_detected": "부적절한 메시지입니다. 정상적으로 대화해주세요. 🚫",
        "error": "죄송해요, 처리 중 문제가 발생했습니다. 다시 시도해주세요! 😅",
        "reset_chat": "대화가 길어져서 새로 시작할게요! 🔄",
        "model_changed_pro": "Gemini Pro 모델로 변경했어요! 🧠",
        "model_changed_flash": "Gemini Flash 모델로 변경했어요! ⚡",
        "chat_reset": "대화를 초기화했어요! 새로 시작할게요! 🔄",
        "full_reset_complete": "모든 데이터를 초기화했어요! 🔄"
    },
    
    # 최고관리자 설정 (절대 변경 불가)
    "super_admin": {
        "id": "1295232354205569075",  # 절대 변경할 수 없음
        "unlimited_access": True,
        "bypass_all_limits": True,
        "natural_admin_commands": True
    }
}

def get_config() -> Dict[str, Any]:
    """설정 반환 (자연어 전용)"""
    return DEFAULT_CONFIG.copy()

def update_config(key: str, value: Any) -> bool:
    """설정 업데이트 (명령어 관련 설정은 변경 불가)"""
    # 명령어 시스템 관련 설정은 절대 변경할 수 없음
    forbidden_keys = [
        "features.natural_language_only",
        "features.command_system_disabled", 
        "features.slash_commands",
        "features.button_interactions",
        "super_admin.id"
    ]
    
    if key in forbidden_keys:
        return False  # 변경 거부
    
    keys = key.split('.')
    config = DEFAULT_CONFIG
    
    for k in keys[:-1]:
        if k not in config:
            config[k] = {}
        config = config[k]
    
    config[keys[-1]] = value
    return True

def is_natural_language_only() -> bool:
    """자연어 전용 모드 확인 (항상 True)"""
    return True

def get_natural_keywords() -> Dict[str, List[str]]:
    """자연어 키워드 반환"""
    return DEFAULT_CONFIG["natural_keywords"].copy()

# 관리자 기능 키워드를 natural_keywords에 통합
DEFAULT_CONFIG["natural_keywords"].update({
    "set_welcome_message": ["환영 메시지 설정", "웰컴 메시지 설정"],
    "set_farewell_message": ["작별 메시지 설정", "굿바이 메시지 설정"],
    "set_bot_status": ["봇 상태 설정", "봇 활성화", "봇 비활성화"],
    "set_language": ["언어 설정", "언어 변경"],
    "set_timezone": ["시간대 설정", "시간대 변경"],
    "backup_data": ["데이터 백업", "백업 생성"],
    "restore_data": ["데이터 복원", "복원하기"],
    "set_custom_command": ["커스텀 명령어 설정", "자주 사용하는 명령어 추가"],
    "remove_custom_command": ["커스텀 명령어 제거", "자주 사용하는 명령어 삭제"],
    "list_custom_commands": ["커스텀 명령어 목록", "자주 사용하는 명령어 보기"],
    "set_auto_responder": ["자동 응답 설정", "자동 응답기 설정"],
    "remove_auto_responder": ["자동 응답 제거", "자동 응답기 제거"],
    "list_auto_responders": ["자동 응답 목록", "자동 응답기 목록 보기"],
    "set_channel_permissions": ["채널 권한 설정", "채널 접근 설정"],
    "set_role_permissions": ["역할 권한 설정", "역할 접근 설정"],
    "set_bot_reaction": ["봇 반응 설정", "봇 이모지 설정"],
    "set_command_prefix": ["명령어 접두사 설정", "프리픽스 설정"],
    "set_help_command": ["도움말 명령어 설정", "헬프 명령어 설정"],
    "set_alias": ["별칭 설정", "알리아스 설정"],
    "set_invite_link": ["초대 링크 설정", "인비테이션 링크 설정"],
    "set_server_rules": ["서버 규칙 설정", "규칙 설정"],
    "set_welcome_channel": ["환영 채널 설정", "웰컴 채널 설정"],
    "set_farewell_channel": ["작별 채널 설정", "굿바이 채널 설정"],
    "set_bot_avatar": ["봇 아바타 설정", "봇 프로필 사진 설정"],
    "set_bot_name": ["봇 이름 설정", "봇 닉네임 설정"],
    "set_status_message": ["상태 메시지 설정", "상태 업데이트"],
    "set_activity": ["활동 설정", "현재 활동 설정"],
    "set_streaming": ["스트리밍 설정", "현재 스트리밍 설정"],
    "set_game": ["게임 설정", "현재 게임 설정"],
    "set_website": ["웹사이트 설정", "사이트 설정"],
    "set_support_channel": ["지원 채널 설정", "서포트 채널 설정"],
    "set_feedback_channel": ["피드백 채널 설정", "의견 채널 설정"],
    "set_suggestion_channel": ["제안 채널 설정", "아이디어 채널 설정"],
    "set_poll_channel": ["투표 채널 설정", "설문 채널 설정"],
    "set_event_channel": ["이벤트 채널 설정", "행사 채널 설정"],
    "set_announcement_channel": ["공지 채널 설정", "알림 채널 설정"],
    "set_private_channel": ["비공개 채널 설정", "프라이빗 채널 설정"],
    "set_public_channel": ["공개 채널 설정", "퍼블릭 채널 설정"],
    "set_category": ["카테고리 설정", "카테고리 추가"],
    "set_channel_topic": ["채널 주제 설정", "채널 설명 설정"],
    "set_channel_emoji": ["채널 이모지 설정", "채널 심볼 설정"],
    "set_channel_bitrate": ["채널 비트레이트 설정", "음질 설정"],
    "set_channel_user_limit": ["채널 사용자 제한 설정", "채널 인원 제한 설정"],
    "set_channel_nsfw": ["채널 NSFW 설정", "채널 성인 설정"],
    "set_channel_slowmode": ["채널 슬로우모드 설정", "채널 지연 모드 설정"],
    "set_channel_pinned_messages": ["채널 고정 메시지 설정", "채널 핀 메시지 설정"],
    "set_channel_webhook": ["채널 웹훅 설정", "채널 후크 설정"],
    "set_channel_bot_permissions": ["채널 봇 권한 설정", "봇 접근 설정"],
    "set_channel_role_permissions": ["채널 역할 권한 설정", "역할 접근 설정"],
    "set_channel_overwrite_permissions": ["채널 권한 덮어쓰기 설정", "오버라이트 설정"],
    "set_channel_view_channel": ["채널 보기 설정", "채널 보기 권한"],
    "set_channel_send_messages": ["채널 메시지 전송 설정", "글쓰기 권한"],
    "set_channel_read_messages": ["채널 메시지 읽기 설정", "읽기 권한"],
    "set_channel_connect": ["채널 연결 설정", "음성 채널 입장"],
    "set_channel_speak": ["채널 말하기 설정", "음성 채널 발언"],
    "set_channel_mute": ["채널 음소거 설정", "멤버 음소거"],
    "set_channel_deafen": ["채널 청각 차단 설정", "멤버 소리 차단"],
    "set_channel_move": ["채널 이동 설정", "멤버 이동"],
    "set_channel_priority_speaker": ["채널 우선 발언자 설정", "우선 발언권"],
    "set_channel_video_quality": ["채널 비디오 품질 설정", "화질 설정"],
    "set_channel_rtc_region": ["채널 RTC 지역 설정", "서버 지역 설정"],
    "set_channel_flags": ["채널 플래그 설정", "채널 속성"],
    "set_channel_application_command_permissions": ["채널 앱 명령어 권한 설정", "앱 명령어 권한"],
    "set_channel_message_tts": ["채널 TTS 설정", "TTS 메시지"],
    "set_channel_message_pins": ["채널 메시지 핀 설정", "메시지 고정"],
    "set_channel_message_reactions": ["채널 반응 설정", "이모지 반응"],
    "set_channel_message_mentions": ["채널 멘션 설정", "언급 허용"],
    "set_channel_message_stickers": ["채널 스티커 설정", "스티커 사용"],
    "set_channel_message_embeds": ["채널 임베드 설정", "링크 미리보기"],
    "set_channel_message_attachments": ["채널 첨부파일 설정", "파일 전송"],
    "set_channel_message_suppress": ["채널 메시지 억제 설정", "메시지 숨기기"],
    "set_channel_message_flags": ["채널 메시지 플래그 설정", "메시지 속성"],
    "set_channel_message_type": ["채널 메시지 유형 설정", "메시지 포맷"],
    "set_channel_message_content": ["채널 메시지 내용 설정", "메시지 본문"],
    "set_channel_message_components": ["채널 메시지 컴포넌트 설정", "메시지 버튼"],
    "set_channel_message_replies": ["채널 답글 설정", "답장 허용"],
    "set_channel_message_thread": ["채널 스레드 설정", "스레드 생성"],
    "set_channel_message_activity": ["채널 활동 설정", "메시지 활동"],
    "set_channel_message_application": ["채널 앱 설정", "메시지 앱"],
    "set_channel_message_reference": ["채널 참조 설정", "메시지 참조"],
    "set_channel_message_sticker": ["채널 스티커 아이콘 설정", "스티커 아이콘"],
    "set_channel_message_embed": ["채널 임베드 미리보기 설정", "임베드 미리보기"],
    "set_channel_message_attachment": ["채널 첨부파일 파일 설정", "첨부 파일"],
    "set_channel_message_suppress_embeds": ["채널 임베드 억제 설정", "링크 미리보기 숨기기"],
    "set_channel_message_flags_all": ["채널 모든 메시지 플래그 설정", "모든 메시지 속성"],
    "set_channel_message_type_all": ["채널 모든 메시지 유형 설정", "모든 메시지 포맷"],
    "set_channel_message_content_all": ["채널 모든 메시지 내용 설정", "모든 메시지 본문"],
    "set_channel_message_components_all": ["채널 모든 메시지 컴포넌트 설정", "모든 메시지 요소"],
    "set_channel_message_replies_all": ["채널 모든 메시지 답글 설정", "모든 메시지 회신"],
    "set_channel_message_thread_all": ["채널 모든 메시지 스레드 설정", "모든 메시지 토론"],
    "set_channel_message_activity_all": ["채널 모든 메시지 활동 설정", "모든 메시지 액티비티"],
    "set_channel_message_application_all": ["채널 모든 메시지 앱 설정", "모든 메시지 앱"],
    "set_channel_message_reference_all": ["채널 모든 메시지 참조 설정", "모든 메시지 링크"],
    "set_channel_message_sticker_all": ["채널 모든 메시지 스티커 설정", "모든 메시지 아이콘"],
    "set_channel_message_embed_all": ["채널 모든 메시지 임베드 설정", "모든 메시지 미리보기"],
    "set_channel_message_attachment_all": ["채널 모든 메시지 첨부파일 설정", "모든 메시지 파일"],
    "set_channel_message_suppress_embeds_all": ["채널 모든 메시지 임베드 억제 설정", "모든 링크 숨기기"]
})
