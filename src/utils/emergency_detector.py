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
                "keywords": ["화재", "불", "타는냄새", "타는", "냄새", "연기", "폭발", "사고", "충돌", "부딪", "뒤집", "전복", 
                            "전자장비", "모든", "꺼졌어요", "꺼져서", "작동안", "응답없", "반응없"],
                "weight": 10,
                "priority": "CRITICAL"
            },
            # 높은 위험도 (8점)  
            "high": {
                "keywords": ["브레이크", "제동", "핸들", "조향", "스티어링", "엔진정지", "멈춤", "꺼짐", "시동꺼짐",
                            "가속", "속도", "안올라", "페달", "밟아도", "안", "위험해", "쏠려"],
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
                "keywords": ["즉시", "지금", "바로", "당장", "급히", "빨리", "응급", "어떻게해", "어떻게해야"],
                "weight": 4  # 가중치 증가
            },
            "urgent": {
                "keywords": ["갑자기", "도와줘", "도움"],
                "weight": 2
            },
            "concern": {
                "keywords": ["걱정", "불안", "무서워", "위험", "문제"],
                "weight": 1
            }
        }
        
        # 기술 문의 식별을 위한 추가 키워드
        self.technology_indicators = [
            "작동", "원리", "구조", "설계", "기능", "특징", "장점", "단점",
            "비교", "차이점", "종류", "타입", "모델", "버전", "업그레이드",
            "설정", "조정", "튜닝", "최적화", "성능", "효율", "소비량",
            "궁금해요", "알려주세요", "설명해주세요", "무엇인가요", "뭔가요"
        ]
        
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
        """응급 상황 감지 및 분석 - 최적화된 빠른 버전"""
        query_lower = query.lower()
        
        # 빠른 응급 키워드 체크 (CRITICAL/HIGH 우선)
        critical_high_keywords = []
        for category, data in self.emergency_keywords.items():
            if data["priority"] in ["CRITICAL", "HIGH"]:
                critical_high_keywords.extend(data["keywords"])
        
        # 정비/교체 관련 키워드 (응급도 감소)
        maintenance_keywords = [
            "교체", "갈아", "바꿔", "주기", "언제", "시기", "정비", "점검", 
            "관리", "유지", "보수", "서비스", "수리", "방법", "알려줘", "알려주세요"
        ]
        
        # 기술/시스템 문의 키워드 (응급도 감소)
        technology_keywords = [
            "시스템", "기능", "설정", "사용법", "작동", "뭔가요", "무엇", 
            "설명", "궁금해요", "차이점", "원리", "방법", "알려줘", "알려주세요"
        ]
        
        # 정비 관련 질문인지 확인
        is_maintenance_question = any(keyword in query_lower for keyword in maintenance_keywords)
        
        # 기술 문의인지 확인
        is_technology_question = (
            any(keyword in query_lower for keyword in technology_keywords) or
            any(keyword in query_lower for keyword in self.technology_indicators)
        )
        
        # CRITICAL/HIGH 키워드가 있으면 즉시 응급 상황으로 판정 (단, 정비/기술 문의가 아닌 경우만)
        for keyword in critical_high_keywords:
            if keyword in query_lower:
                # 해당 카테고리 찾기
                for category, data in self.emergency_keywords.items():
                    if keyword in data["keywords"] and data["priority"] in ["CRITICAL", "HIGH"]:
                        # 정비/기술 문의인 경우 응급 상황이 아님
                        if is_maintenance_question or is_technology_question:
                            # 일반 질문으로 처리
                            break
                        
                        return {
                            "is_emergency": True,
                            "emergency_score": data["weight"],
                            "urgency_score": 0,
                            "total_score": data["weight"],
                            "priority_level": data["priority"],
                            "detected_categories": [{
                                "category": category,
                                "keyword": keyword,
                                "weight": data["weight"],
                                "original_weight": data["weight"],
                                "priority": data["priority"],
                                "maintenance_adjusted": False
                            }],
                            "urgency_expressions": [],
                            "search_strategy": self.emergency_search_strategies.get(data["priority"])
                        }
        
        
        # 응급도 점수 계산 (간소화)
        emergency_score = 0
        detected_categories = []
        priority_level = "NORMAL"
        has_emergency_keywords = False
        
        # 1. 응급 키워드 검사 (CRITICAL/HIGH 제외)
        for category, data in self.emergency_keywords.items():
            if data["priority"] not in ["CRITICAL", "HIGH"]:  # 이미 위에서 처리됨
                for keyword in data["keywords"]:
                    if keyword in query_lower:
                        has_emergency_keywords = True
                        weight = data["weight"]
                        
                        # 정비/기술 문의인 경우 가중치 조정
                        if is_maintenance_question or is_technology_question:
                            weight = max(1, weight // 2)
                        
                        emergency_score += weight
                        detected_categories.append({
                            "category": category,
                            "keyword": keyword,
                            "weight": weight,
                            "original_weight": data["weight"],
                            "priority": data["priority"],
                            "maintenance_adjusted": is_maintenance_question or is_technology_question
                        })
                        
                        # 우선순위 업데이트
                        if data["priority"] == "MEDIUM" and priority_level == "NORMAL":
                            priority_level = "MEDIUM"
                        elif data["priority"] == "LOW" and priority_level == "NORMAL":
                            priority_level = "LOW"
        
        # 2. 긴급성 표현 검사 (간소화)
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
        
        # 응급 상황 판정 (간소화)
        if has_emergency_keywords:
            # 기술 문의인 경우 특별 처리
            if is_technology_question and priority_level not in ["CRITICAL", "HIGH"]:
                is_emergency = False
                priority_level = "NORMAL"
            else:
                # 점수 기반 판정
                threshold = 4 if (is_maintenance_question or is_technology_question) else 3
                is_emergency = total_score >= threshold
        else:
            # 응급 키워드가 없으면 일반 질문
            is_emergency = False
            priority_level = "NORMAL"
        
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
