"""
LLM 기반 응급 상황 감지 및 주행 상황 감지 시스템
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


class EmergencyAnalysis(BaseModel):
    """응급 상황 분석 결과"""
    is_emergency: bool = Field(description="응급 상황 여부")
    priority_level: str = Field(description="응급 수준: CRITICAL, HIGH, MEDIUM, LOW, NORMAL")
    confidence: float = Field(description="판단 신뢰도 (0-1)")
    reasoning: str = Field(description="판단 근거")
    emergency_indicators: List[str] = Field(description="응급 상황을 나타내는 지표들")
    context_type: str = Field(description="질문 유형: emergency, maintenance, technical, general")


class DrivingAnalysis(BaseModel):
    """주행 상황 분석 결과"""
    is_driving: bool = Field(description="주행 중 여부")
    confidence: float = Field(description="판단 신뢰도 (0-1)")
    urgency_level: str = Field(description="긴급도: immediate, urgent, normal")
    reasoning: str = Field(description="판단 근거")
    driving_indicators: List[str] = Field(description="주행 중임을 나타내는 지표들")
    compression_needed: bool = Field(description="답변 압축 필요 여부")


class LLMEmergencyDetector:
    """LLM 기반 응급 상황 감지기"""
    
    def __init__(self, llm_model: str = "gpt-4o-mini", temperature: float = 0):
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        
        # 구조화된 출력을 위한 LLM 설정
        self.emergency_analyzer = self.llm.with_structured_output(EmergencyAnalysis)
        self.driving_analyzer = self.llm.with_structured_output(DrivingAnalysis)
        
        # 응급 상황 감지 프롬프트
        self.emergency_detection_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 차량 관련 질문에서 응급 상황을 감지하는 전문가입니다.

응급 상황 분류 기준:

**CRITICAL (생명 위험)**:
- 화재, 폭발, 연기, 타는 냄새
- 사고, 충돌, 전복
- 전자장비 완전 고장 (모든 시스템 꺼짐)
- 즉시 생명에 위험한 상황

**HIGH (즉시 조치 필요)**:
- 브레이크 고장, 제동 불가
- 핸들 고장, 조향 불가
- 엔진 정지, 시동 꺼짐
- 가속 불가, 페달 고장
- 즉시 안전 조치가 필요한 상황

**MEDIUM (신속 대응 필요)**:
- 과열, 온도 경고
- 경고등 점등
- 펑크, 타이어 문제
- 시야 문제 (와이퍼 고장 등)

**LOW (주의 필요)**:
- 배터리 방전
- 연료 부족
- 시동 문제

**NORMAL (일반 질문)**:
- 정비, 교체, 관리 방법 문의
- 기술적 원리, 작동 방식 문의
- 일반적인 정보 요청
- "방법", "알려줘", "궁금해요" 등이 포함된 질문

판단 시 고려사항:
1. 질문의 맥락과 의도를 종합적으로 분석
2. 긴급성 표현 ("지금", "즉시", "당장" 등) 고려
3. 정비/기술 문의는 일반 질문으로 분류
4. 실제 위험 상황과 정보 요청을 구분
5. 한국어 표현의 뉘앙스 이해"""),
            ("human", "사용자 질문: {query}")
        ])
        
        # 주행 상황 감지 프롬프트
        self.driving_detection_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 사용자의 발화에서 현재 주행 중인지 판단하는 전문가입니다.

주행 중 상황을 나타내는 지표들:

**명시적 주행 표현** (가중치: 높음):
- "운전 중", "주행 중", "차 안에서", "도로에서"
- "고속도로에서", "시내 주행", "교통체증 중"
- "출근 중", "퇴근 중", "이동 중", "가는 길"

**시간적 긴급성** (가중치: 중간):
- "지금", "현재", "바로", "즉시", "당장"
- "갑자기", "방금", "막", "급하게"

**상황적 맥락** (가중치: 낮음):
- "~하고 있는데", "~중인데", "~하면서"
- "~하다가", "~하던 중", "~진행 중"

**위치/이동 관련** (가중치: 중간):
- "길에서", "도로 위", "터널 안", "다리 위"
- "톨게이트", "휴게소", "주유소"

**차량 상태/동작** (가중치: 중간):
- "시동 걸고", "기어 넣고", "브레이크 밟고"
- "주차 중", "정차 중", "후진 중", "회전 중"

**음성/핸즈프리 단서** (가중치: 높음):
- "음성으로", "말로", "핸즈프리", "블루투스"

긴급도 수준:
- immediate: 즉시 대응 필요 (안전 위험, 생명 위험)
- urgent: 빠른 대응 필요 (기능 문제, 주행 방해)
- normal: 일반적 문의 (정보 요청)

판단 기준:
- 명확한 주행 중 표현 + 음성 단서: 95% 이상 신뢰도
- 명확한 주행 중 표현: 85-95% 신뢰도
- 강한 시간적 긴급성 + 위치 정보: 70-85% 신뢰도
- 차량 상태/동작 표현: 60-80% 신뢰도
- 상황적 맥락만: 40-60% 신뢰도
- 일반적 질문: 30% 미만 신뢰도"""),
            ("human", "사용자 발화: {query}")
        ])
        
        # 체인 구성
        self.emergency_chain = self.emergency_detection_prompt | self.emergency_analyzer
        self.driving_chain = self.driving_detection_prompt | self.driving_analyzer
    
    def detect_emergency(self, query: str) -> Dict[str, Any]:
        """LLM 기반 응급 상황 감지"""
        try:
            analysis = self.emergency_chain.invoke({"query": query})
            
            return {
                "is_emergency": analysis.is_emergency,
                "emergency_score": self._convert_priority_to_score(analysis.priority_level),
                "urgency_score": 0,  # LLM에서는 별도 계산하지 않음
                "total_score": self._convert_priority_to_score(analysis.priority_level),
                "priority_level": analysis.priority_level,
                "detected_categories": [{
                    "category": "llm_analysis",
                    "keyword": "llm_based",
                    "weight": self._convert_priority_to_score(analysis.priority_level),
                    "original_weight": self._convert_priority_to_score(analysis.priority_level),
                    "priority": analysis.priority_level,
                    "maintenance_adjusted": False
                }],
                "urgency_expressions": [],
                "search_strategy": self._get_search_strategy(analysis.priority_level),
                "reasoning": analysis.reasoning,
                "emergency_indicators": analysis.emergency_indicators,
                "context_type": analysis.context_type,
                "confidence": analysis.confidence
            }
            
        except Exception as e:
            print(f"LLM 응급 상황 감지 오류: {str(e)}")
            # 오류 시 안전하게 일반 질문으로 처리
            return {
                "is_emergency": False,
                "emergency_score": 0,
                "urgency_score": 0,
                "total_score": 0,
                "priority_level": "NORMAL",
                "detected_categories": [],
                "urgency_expressions": [],
                "search_strategy": None,
                "reasoning": f"LLM 분석 오류: {str(e)}",
                "emergency_indicators": [],
                "context_type": "general",
                "confidence": 0.5
            }
    
    def detect_driving_context(self, query: str) -> Dict[str, Any]:
        """LLM 기반 주행 상황 감지"""
        try:
            analysis = self.driving_chain.invoke({"query": query})
            
            return {
                "is_driving": analysis.is_driving,
                "confidence": analysis.confidence,
                "driving_indicators": analysis.driving_indicators,
                "urgency_level": analysis.urgency_level,
                "compression_needed": analysis.compression_needed,
                "reasoning": analysis.reasoning
            }
            
        except Exception as e:
            print(f"LLM 주행 상황 감지 오류: {str(e)}")
            # 오류 시 안전하게 주행 중이 아닌 것으로 처리
            return {
                "is_driving": False,
                "confidence": 0.5,
                "driving_indicators": [],
                "urgency_level": "normal",
                "compression_needed": False,
                "reasoning": f"LLM 분석 오류: {str(e)}"
            }
    
    def _convert_priority_to_score(self, priority_level: str) -> float:
        """우선순위 레벨을 점수로 변환"""
        score_map = {
            "CRITICAL": 10.0,
            "HIGH": 8.0,
            "MEDIUM": 6.0,
            "LOW": 4.0,
            "NORMAL": 0.0
        }
        return score_map.get(priority_level, 0.0)
    
    def _get_search_strategy(self, priority_level: str) -> Dict[str, Any]:
        """우선순위에 따른 검색 전략 반환"""
        strategies = {
            "CRITICAL": {
                "search_method": "hybrid_keyword",
                "compression_method": "rerank_only",
                "timeout": 5
            },
            "HIGH": {
                "search_method": "hybrid_keyword",
                "compression_method": "rerank_compress_troubleshooting",
                "timeout": 8
            },
            "MEDIUM": {
                "search_method": "hybrid_balanced",
                "compression_method": "rerank_compress_troubleshooting",
                "timeout": 10
            },
            "LOW": {
                "search_method": "hybrid_semantic",
                "compression_method": "rerank_compress_general",
                "timeout": 12
            },
            "NORMAL": None
        }
        return strategies.get(priority_level)
    
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
