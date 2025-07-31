# 루시아 봇 설정 파일 - 자연어 전용 (명령어 시스템 제거됨)
# 이 설정은 자연어 기반 봇을 위한 것이며, 명령어 관련 설정은 절대 추가할 수 없습니다.

import os
from typing import Dict, Any

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
        "unblock_user": ["차단 해제", "차단해제"]
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