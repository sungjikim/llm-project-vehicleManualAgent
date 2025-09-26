"""
응급 상황 감지 및 처리 유틸리티
"""

import re
from typing import Dict, List, Tuple, Any


class EmergencyDetector:
    """응급 상황 질문 감지 및 분류 클래스"""
    
    def __init__(self):
        # 응급 상황 키워드 (가중치 포함)
        self.emergency_keywords = {
            # 최고 위험도 (10점)
            "critical": {
                "keywords": ["화재", "불", "타는냄새", "타는", "냄새", "연기", "폭발", "사고", "충돌", "부딪", "뒤집", "전복"],
                "weight": 10,
                "priority": "CRITICAL"
            },
            # 높은 위험도 (8점)  
            "high": {
                "keywords": ["브레이크", "제동", "핸들", "조향", "스티어링", "엔진정지", "멈춤", "꺼짐", "시동꺼짐"],
                "weight": 8,
                "priority": "HIGH"
            },
            # 중간 위험도 (6점)
            "medium": {
                "keywords": ["과열", "온도", "빨간등", "경고등", "펑크", "타이어터짐", "바퀴", "와이퍼", "시야"],
                "weight": 6,
                "priority": "MEDIUM"
            },
            # 낮은 위험도 (4점)
            "low": {
                "keywords": ["배터리", "방전", "시동안걸림", "전기안들어옴", "연료부족", "기름부족"],
                "weight": 4,
                "priority": "LOW"
            }
        }
        
        # 긴급성 표현 (추가 가중치)
        self.urgency_expressions = {
            "immediate": {
                "keywords": ["즉시", "지금", "바로", "당장", "급히", "빨리", "응급"],
                "weight": 3
            },
            "urgent": {
                "keywords": ["갑자기", "突然", "어떻게해야", "어떻게해", "도와줘", "도움"],
                "weight": 2
            },
            "concern": {
                "keywords": ["걱정", "불안", "무서워", "위험", "문제"],
                "weight": 1
            }
        }
        
        # 응급 상황별 최적 검색 전략
        self.emergency_search_strategies = {
            "CRITICAL": {
                "search_method": "hybrid_keyword",  # 키워드 우선 (3:7)
                "compression_method": "rerank_only",  # 빠른 재순위화만
                "timeout": 5,  # 5초 제한
                "priority_keywords": ["안전", "대처", "조치", "방법", "즉시"]
            },
            "HIGH": {
                "search_method": "hybrid_keyword",
                "compression_method": "rerank_compress_troubleshooting",
                "timeout": 8,
                "priority_keywords": ["응급", "대처", "해결", "방법"]
            },
            "MEDIUM": {
                "search_method": "hybrid_balanced", 
                "compression_method": "rerank_compress_troubleshooting",
                "timeout": 10,
                "priority_keywords": ["확인", "점검", "조치"]
            },
            "LOW": {
                "search_method": "hybrid_semantic",
                "compression_method": "rerank_compress_general", 
                "timeout": 12,
                "priority_keywords": ["방법", "해결"]
            }
        }
    
    def detect_emergency(self, query: str) -> Dict[str, Any]:
        """응급 상황 감지 및 분석"""
        query_lower = query.lower()
        
        # 응급도 점수 계산
        emergency_score = 0
        detected_categories = []
        priority_level = "NORMAL"
        
        # 1. 응급 키워드 검사
        for category, data in self.emergency_keywords.items():
            for keyword in data["keywords"]:
                if keyword in query_lower:
                    emergency_score += data["weight"]
                    detected_categories.append({
                        "category": category,
                        "keyword": keyword,
                        "weight": data["weight"],
                        "priority": data["priority"]
                    })
                    
                    # 최고 우선순위 업데이트
                    if data["priority"] == "CRITICAL":
                        priority_level = "CRITICAL"
                    elif data["priority"] == "HIGH" and priority_level != "CRITICAL":
                        priority_level = "HIGH"
                    elif data["priority"] == "MEDIUM" and priority_level not in ["CRITICAL", "HIGH"]:
                        priority_level = "MEDIUM"
                    elif data["priority"] == "LOW" and priority_level == "NORMAL":
                        priority_level = "LOW"
        
        # 2. 긴급성 표현 검사
        urgency_score = 0
        urgency_expressions_found = []
        
        for category, data in self.urgency_expressions.items():
            for keyword in data["keywords"]:
                if keyword in query_lower:
                    urgency_score += data["weight"]
                    urgency_expressions_found.append({
                        "category": category,
                        "keyword": keyword,
                        "weight": data["weight"]
                    })
        
        # 총 점수 계산
        total_score = emergency_score + urgency_score
        
        # 응급 상황 판정 (임계값: 6점 이상)
        is_emergency = total_score >= 6 or priority_level in ["CRITICAL", "HIGH"]
        
        return {
            "is_emergency": is_emergency,
            "emergency_score": emergency_score,
            "urgency_score": urgency_score,
            "total_score": total_score,
            "priority_level": priority_level,
            "detected_categories": detected_categories,
            "urgency_expressions": urgency_expressions_found,
            "search_strategy": self.emergency_search_strategies.get(
                priority_level, 
                self.emergency_search_strategies["LOW"]
            ) if is_emergency else None
        }
    
    def get_emergency_prompt_enhancement(self, priority_level: str) -> str:
        """응급 상황별 프롬프트 강화 텍스트"""
        enhancements = {
            "CRITICAL": """
⚠️ **CRITICAL EMERGENCY** ⚠️
이것은 생명과 직결된 매우 위험한 상황입니다.
답변 시 다음을 최우선으로 해주세요:
1. 즉시 실행 가능한 안전 조치를 첫 번째로 제시
2. 추가 위험 요소 경고
3. 전문가/응급 서비스 연락 강력 권고
4. 단계별 명확한 행동 지침
""",
            "HIGH": """
🚨 **HIGH PRIORITY EMERGENCY** 🚨  
이것은 즉시 조치가 필요한 위험 상황입니다.
답변 시 다음을 우선해주세요:
1. 안전 확보를 위한 즉시 조치
2. 상황 악화 방지 방법
3. 전문가 상담 권고
4. 명확하고 구체적인 단계별 안내
""",
            "MEDIUM": """
⚠️ **MEDIUM EMERGENCY** ⚠️
이것은 신속한 대응이 필요한 상황입니다.
답변 시 다음을 포함해주세요:
1. 상황 확인 및 안전 점검 방법
2. 응급 대처 방법
3. 추가 점검 사항
4. 전문가 상담 시점 안내
""",
            "LOW": """
🔍 **LOW EMERGENCY** 🔍
이것은 주의가 필요한 상황입니다.
답변 시 다음을 포함해주세요:
1. 현재 상황 진단 방법
2. 안전한 임시 대처 방법  
3. 근본적 해결 방안
4. 예방 조치
"""
        }
        
        return enhancements.get(priority_level, "")
    
    def get_emergency_examples(self) -> List[Dict[str, str]]:
        """응급 상황 Few-shot 예시"""
        return [
            {
                "query": "브레이크를 밟아도 차가 안 멈춰요",
                "context": "브레이크 시스템 완전 실패 시에는 즉시 다음 조치를 취해야 합니다. 엔진 브레이크를 사용하고, 사이드 브레이크를 점진적으로 사용하세요. [페이지 195-196]",
                "answer": "**🚨 즉시 조치 (생명 위험)**:\n\n1. **엔진 브레이크 사용**: 기어를 낮은 단으로 변경\n2. **사이드 브레이크**: 점진적으로 천천히 당기기 (급하게 당기면 스핀 위험)\n3. **비상등 점등**: 다른 차량에 위험 알림\n4. **안전 지대 이동**: 가능한 한 도로 밖 안전한 곳으로\n\n⚠️ **절대 금지**: 급격한 조향, 급작스런 사이드 브레이크 사용\n\n**즉시 연락**: 119 응급상황 신고, 견인 서비스\n\n📚 참고 페이지: 195-196"
            },
            {
                "query": "엔진에서 연기가 나고 있어요",
                "context": "엔진 과열로 인한 연기 발생 시에는 즉시 엔진을 정지하고 안전한 장소로 이동해야 합니다. 후드를 열지 말고 냉각을 기다려야 합니다. [페이지 201-203]",
                "answer": "**🔥 즉시 조치 (화재 위험)**:\n\n1. **즉시 정차**: 안전한 장소에 차량 정지\n2. **엔진 즉시 정지**: 시동 끄고 키 제거\n3. **차량 대피**: 모든 승객 차량에서 하차\n4. **후드 절대 금지**: 뜨거운 증기로 화상 위험\n\n**대기 및 확인**:\n- 최소 30분 냉각 대기\n- 냉각수 레벨 확인 (식은 후)\n- 호스나 연결부 누수 점검\n\n**즉시 연락**: 견인 서비스, 심각한 경우 119\n\n📚 참고 페이지: 201-203"
            }
        ]
