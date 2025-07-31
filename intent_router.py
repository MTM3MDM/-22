"""
intent_router.py

자연어 메시지에서 intent와 parameters를 추출하는 모듈
"""
from typing import Tuple, Dict, Any
from config import get_natural_keywords

# intent-키워드 매핑
INTENT_KEYWORDS = get_natural_keywords()

# fallback intent for unknown commands
FALLBACK_INTENT = "unknown"


def parse_intent(message: str) -> Tuple[str, Dict[str, Any]]:
    """
    메시지에서 intent와 파라미터를 추출합니다.
    intent에 해당하는 키워드가 없으면 'unknown'을 반환합니다.
    """
    text = message.lower().strip()
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                # 파라미터 추출 로직 (개선 필요)
                params = extract_params(text, kw)
                return intent, params
    
    # Gemini를 이용한 fallback은 비용 및 안정성 문제로 제거
    return FALLBACK_INTENT, {}

def extract_params(text: str, keyword: str) -> Dict[str, Any]:
    """간단한 파라미터 추출 로직"""
    params: Dict[str, Any] = {}
    # 키워드 이후의 텍스트를 파라미터로 간주
    # 예: "채널 삭제해줘 일반" -> params['target'] = '일반'
    try:
        # 키워드를 기준으로 텍스트를 분리하고, 뒷부분을 파라미터로 사용
        parts = text.split(keyword)
        if len(parts) > 1 and parts[1].strip():
            # 기본적인 'target' 파라미터로 저장
            params['target'] = parts[1].strip()
            
            # 추가적인 파라미터 분석 (예: 숫자, 특정 단어)
            # "채널 순서 변경 공지 1" -> target: '공지', position: 1
            words = params['target'].split()
            if len(words) > 1 and words[-1].isdigit():
                params['position'] = int(words[-1])
                params['target'] = ' '.join(words[:-1]) # 숫자 제외한 나머지
                
    except Exception as e:
        # 파라미터 추출 중 오류 발생 시, 빈 파라미터 반환
        print(f"Error extracting params: {e}") # 로깅으로 대체 권장
        
    return params
