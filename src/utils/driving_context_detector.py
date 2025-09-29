"""
주행 중 상황 감지 및 답변 압축 시스템
"""

import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class DrivingContextAnalysis(BaseModel):
    """주행 상황 분석 결과"""
    is_driving: bool = Field(description="주행 중 여부")
    confidence: float = Field(description="판단 신뢰도 (0-1)")
    driving_indicators: List[str] = Field(description="주행 중임을 나타내는 지표들")
    urgency_level: str = Field(description="긴급도 수준: immediate, urgent, normal")
    compression_needed: bool = Field(description="답변 압축 필요 여부")


class CompressedAnswer(BaseModel):
    """압축된 답변 구조"""
    key_action: str = Field(description="핵심 행동 지침")
    safety_warning: Optional[str] = Field(description="안전 경고 (필요시)")
    quick_steps: List[str] = Field(description="간단한 단계별 지침 (최대 3단계)")
    follow_up: Optional[str] = Field(description="주행 후 상세 확인 사항")


class DrivingContextDetector:
    """주행 중 상황 감지 및 답변 압축기"""
    
    def __init__(self, llm_model: str = "gpt-4o-mini", temperature: float = 0):
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        
        # 구조화된 출력을 위한 LLM 설정
        self.structured_analyzer = self.llm.with_structured_output(DrivingContextAnalysis)
        self.structured_compressor = self.llm.with_structured_output(CompressedAnswer)
        
        # 주행 중 상황 감지 프롬프트
        self.driving_detection_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 사용자의 발화에서 현재 주행 중인지 판단하는 전문가입니다.

주행 중 상황을 나타내는 지표들:

1. **명시적 주행 표현** (가중치: 높음)
   - 직접적: "운전 중", "주행 중", "차 안에서", "도로에서", "핸들 잡고"
   - 이동 관련: "출근 중", "퇴근 중", "이동 중", "가는 길", "오는 길"
   - 도로 상황: "고속도로에서", "시내 주행", "교통체증 중", "신호 대기"

2. **시간적 긴급성** (가중치: 중간)
   - 즉시성: "지금", "현재", "바로", "즉시", "당장", "빨리"
   - 급박함: "갑자기", "방금", "막", "급하게", "서둘러", "긴급하게"
   - 실시간: "실시간", "라이브", "곧바로", "신속히"

3. **상황적 맥락** (가중치: 낮음)
   - 진행 상황: "~하고 있는데", "~중인데", "~하면서", "~하는 동안"
   - 현재 상태: "~하다가", "~하던 중", "~진행 중"

4. **위치/이동 관련** (가중치: 중간)
   - 도로: "길에서", "도로 위", "차선", "터널 안", "다리 위"
   - 시설: "톨게이트", "휴게소", "주유소", "정비소 가는"
   - 목적지: "회사 가는", "집 가는", "목적지 향해"

5. **차량 상태/동작** (가중치: 중간)
   - 조작: "시동 걸고", "기어 넣고", "브레이크 밟고", "액셀 밟고"
   - 상태: "주차 중", "정차 중", "후진 중", "회전 중", "추월 중"

6. **음성/핸즈프리 단서** (가중치: 높음)
   - 음성 입력: "음성으로", "말로", "핸즈프리", "블루투스"
   - 소리: "소리 내서", "큰 소리로", "음성 명령", "대화 중"

판단 기준:
- 명확한 주행 중 표현 + 음성 단서: 95% 이상 신뢰도
- 명확한 주행 중 표현: 85-95% 신뢰도
- 강한 시간적 긴급성 + 위치 정보: 70-85% 신뢰도
- 차량 상태/동작 표현: 60-80% 신뢰도
- 상황적 맥락만: 40-60% 신뢰도
- 일반적 질문: 30% 미만 신뢰도

긴급도 수준:
- immediate: 즉시 대응 필요 (안전 위험, 생명 위험)
- urgent: 빠른 대응 필요 (기능 문제, 주행 방해)
- normal: 일반적 문의 (정보 요청)"""),
            ("human", "사용자 발화: {query}")
        ])
        
        # 답변 압축 프롬프트
        self.answer_compression_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 주행 중인 운전자를 위해 답변을 압축하는 전문가입니다.

압축 원칙:
1. 안전 최우선: 주행에 방해되지 않도록
2. 핵심만 전달: 가장 중요한 1-2가지 행동만
3. 간결한 표현: 한 문장으로 핵심 전달
4. 단계별 최소화: 최대 3단계까지만
5. 시각적 주의 최소화: 긴 텍스트 금지

압축 기준:
- 즉시 대응 필요: 핵심 행동 1개 + 안전 경고
- 빠른 대응 필요: 핵심 행동 + 간단 단계 2개
- 일반 상황: 핵심 + 단계 3개 + 후속 조치

금지 사항:
- 긴 설명문
- 복잡한 절차
- 페이지 번호나 상세 참조
- 신뢰도 정보 (주의 분산)"""),
            ("human", """원본 답변: {original_answer}
사용자 질문: {query}
긴급도: {urgency_level}

위 답변을 주행 중인 운전자에게 적합하도록 압축해주세요.""")
        ])
        
        # 체인 구성
        self.detection_chain = self.driving_detection_prompt | self.structured_analyzer
        self.compression_chain = self.answer_compression_prompt | self.structured_compressor
        
        # 주행 중 키워드 패턴
        self.driving_keywords = {
            "explicit": [  # 명시적 주행 표현
                r"운전\s*중", r"주행\s*중", r"차\s*안에서", r"도로에서",
                r"운전하면서", r"주행하면서", r"차에서", r"운전석에서",
                r"핸들\s*잡고", r"드라이브\s*중", r"출근\s*중", r"퇴근\s*중",
                r"이동\s*중", r"가는\s*길", r"오는\s*길", r"드라이빙\s*중",
                r"고속도로에서", r"시내\s*주행", r"교통체증\s*중", r"신호\s*대기",
                r"주차장\s*찾는", r"길\s*찾는", r"내비\s*켜고", r"GPS\s*보면서"
            ],
            "temporal": [  # 시간적 긴급성
                r"지금", r"현재", r"바로", r"즉시", r"당장", r"빨리",
                r"갑자기", r"방금", r"막", r"금방", r"급하게", r"서둘러",
                r"긴급하게", r"응급", r"위급", r"급히", r"얼른", r"신속히",
                r"곧바로", r"직접", r"실시간", r"라이브"
            ],
            "situational": [  # 상황적 맥락
                r"~하고\s*있는데", r"~중인데", r"~하면서", r"~하는\s*동안",
                r"~할\s*때", r"~하려고\s*하는데", r"~하고\s*있어서",
                r"~하는\s*상황", r"~하는\s*와중", r"~하다가", r"~하던\s*중",
                r"~하고\s*계신", r"~하시는\s*중", r"~진행\s*중"
            ],
            "location": [  # 위치/이동 관련
                r"길에서", r"도로\s*위", r"차선", r"터널\s*안", r"다리\s*위",
                r"고가도로", r"언더패스", r"램프", r"톨게이트", r"휴게소",
                r"주유소", r"세차장", r"정비소\s*가는", r"서비스센터\s*가는",
                r"목적지\s*향해", r"회사\s*가는", r"집\s*가는", r"마트\s*가는"
            ],
            "vehicle_state": [  # 차량 상태/동작 관련
                r"시동\s*걸고", r"기어\s*넣고", r"액셀\s*밟고", r"브레이크\s*밟고",
                r"클러치\s*밟고", r"핸드브레이크", r"사이드브레이크", r"후진\s*중",
                r"주차\s*중", r"유턴\s*중", r"회전\s*중", r"추월\s*중",
                r"정차\s*중", r"신호\s*기다리는", r"대기\s*중", r"멈춰\s*있는"
            ],
            "emergency_malfunction": [  # 응급 고장 상황 (높은 가중치)
                r"브레이크.*?안.*?(들어|동작|작동|멈춤)", r"브레이크.*?(고장|이상|문제)",
                r"브레이크.*?(딱딱|무거워|소리|진동|끌려)", r"브레이크.*?페달",
                r"엔진.*?안.*?(돌아|켜져|시동)", r"엔진.*?(고장|이상|문제|꺼짐)",
                r"핸들.*?안.*?(돌아|움직)", r"핸들.*?(고장|이상|무거워)",
                r"가속.*?안.*?(돼|되)", r"액셀.*?안.*?(밟혀|눌러)",
                r"동작.*?안.*?해", r"작동.*?안.*?해", r"안.*?들어", r"안.*?멈춰",
                r"안.*?돌아", r"안.*?켜져", r"안.*?움직", r"고장.*?나",
                r"이상.*?(해|소리|진동)", r"문제.*?생겨", r"멈추지.*?않아",
                r"응답.*?없어", r"반응.*?없어", r"먹통", r"죽어버려",
                r"필요.*?한데.*?안.*?돼", r"밟아도.*?안", r"눌러도.*?안",
                r"에서.*?소리", r"에서.*?진동", r"에서.*?이상"
            ],
            "audio_cues": [  # 음성/소리 관련 (핸즈프리 환경)
                r"음성\s*으로", r"말로", r"소리\s*내서", r"큰\s*소리로",
                r"핸즈프리", r"블루투스", r"스피커", r"마이크",
                r"음성\s*인식", r"음성\s*명령", r"말하기", r"대화\s*중",
                r"연결해서", r"물어보는데", r"질문하는데"
            ]
        }
    
    def detect_driving_context(self, query: str) -> Dict[str, Any]:
        """주행 중 상황 감지"""
        try:
            # 키워드 기반 사전 분석
            keyword_score = self._calculate_keyword_score(query)
            
            # LLM 기반 상세 분석 (키워드 점수가 일정 이상일 때만)
            if keyword_score > 0.15:  # 임계값을 낮춰서 더 많은 경우에 LLM 분석 수행
                analysis = self.detection_chain.invoke({"query": query})
                
                # 키워드 점수와 LLM 분석 결과 결합 (키워드에 더 높은 가중치)
                final_confidence = (keyword_score * 0.6 + analysis.confidence * 0.4)
                
                # 키워드 점수가 높으면 LLM이 반대해도 주행으로 판단
                is_driving_decision = (
                    (keyword_score > 0.4) or  # 키워드 점수가 높으면 무조건 주행
                    (analysis.is_driving and final_confidence > 0.45)  # 임계값 조정
                )
                
                return {
                    "is_driving": is_driving_decision,
                    "confidence": final_confidence,
                    "driving_indicators": analysis.driving_indicators,
                    "urgency_level": analysis.urgency_level,
                    "compression_needed": analysis.compression_needed,
                    "keyword_score": keyword_score,
                    "analysis": analysis
                }
            else:
                # 키워드 점수가 낮으면 주행 중이 아닌 것으로 판단
                return {
                    "is_driving": False,
                    "confidence": 1.0 - keyword_score,
                    "driving_indicators": [],
                    "urgency_level": "normal",
                    "compression_needed": False,
                    "keyword_score": keyword_score,
                    "analysis": None
                }
                
        except Exception as e:
            print(f"주행 상황 감지 오류: {str(e)}")
            # 오류 시 안전하게 주행 중이 아닌 것으로 처리
            return {
                "is_driving": False,
                "confidence": 0.5,
                "driving_indicators": [],
                "urgency_level": "normal",
                "compression_needed": False,
                "keyword_score": 0.0,
                "analysis": None
            }
    
    def compress_answer(self, original_answer: str, query: str, urgency_level: str) -> Dict[str, Any]:
        """주행 중 상황에 맞게 답변 압축"""
        try:
            # 원본 답변에서 불필요한 정보 제거
            cleaned_answer = self._clean_answer_for_driving(original_answer)
            
            # LLM을 통한 지능적 압축
            compressed = self.compression_chain.invoke({
                "original_answer": cleaned_answer,
                "query": query,
                "urgency_level": urgency_level
            })
            
            # 최종 압축 답변 생성
            final_answer = self._format_compressed_answer(compressed, urgency_level)
            
            return {
                "compressed_answer": final_answer,
                "key_action": compressed.key_action,
                "safety_warning": compressed.safety_warning,
                "quick_steps": compressed.quick_steps,
                "follow_up": compressed.follow_up,
                "compression_ratio": len(final_answer) / len(original_answer)
            }
            
        except Exception as e:
            print(f"답변 압축 오류: {str(e)}")
            # 오류 시 간단한 압축 적용
            return {
                "compressed_answer": self._simple_compression(original_answer),
                "key_action": "원본 답변 참조",
                "safety_warning": "⚠️ 안전한 곳에서 상세 확인 필요",
                "quick_steps": [],
                "follow_up": "주행 후 매뉴얼 확인",
                "compression_ratio": 0.3
            }
    
    def _calculate_keyword_score(self, query: str) -> float:
        """키워드 기반 주행 상황 점수 계산"""
        score = 0.0
        query_lower = query.lower()
        
        # 명시적 주행 표현 (가중치 가장 높음)
        for pattern in self.driving_keywords["explicit"]:
            if re.search(pattern, query_lower):
                score += 0.4
        
        # 시간적 긴급성 표현
        for pattern in self.driving_keywords["temporal"]:
            if re.search(pattern, query_lower):
                score += 0.2
        
        # 상황적 맥락 표현
        for pattern in self.driving_keywords["situational"]:
            if re.search(pattern, query_lower):
                score += 0.1
        
        # 위치/이동 관련 표현
        for pattern in self.driving_keywords["location"]:
            if re.search(pattern, query_lower):
                score += 0.15
        
        # 차량 상태/동작 관련 표현
        for pattern in self.driving_keywords["vehicle_state"]:
            if re.search(pattern, query_lower):
                score += 0.2
        
        # 음성/핸즈프리 관련 표현
        for pattern in self.driving_keywords["audio_cues"]:
            if re.search(pattern, query_lower):
                score += 0.25
        
        # 응급 고장 상황 (가장 높은 가중치)
        for pattern in self.driving_keywords["emergency_malfunction"]:
            if re.search(pattern, query_lower):
                score += 0.5  # 응급 고장은 매우 높은 점수
        
        return min(score, 1.0)  # 최대 1.0으로 제한
    
    def _clean_answer_for_driving(self, answer: str) -> str:
        """주행 중 압축을 위해 답변에서 불필요한 정보 제거"""
        # 제거할 패턴들
        patterns_to_remove = [
            r'📚\s*참고\s*페이지[:\s]*[\d\-,\s]+',  # 페이지 참조
            r'🔍\s*\*\*답변\s*신뢰도\*\*[^🔍]*',    # 신뢰도 정보
            r'✅\s*높은\s*신뢰도[^⚠️❌]*',          # 신뢰도 메시지
            r'⚠️\s*추가\s*확인[^❌]*',              # 추가 확인 안내
            r'❌\s*전문가\s*상담[^📝]*',            # 전문가 상담 안내
            r'📝\s*\*\*일반\s*질문\*\*\s*\n\n',    # 일반 질문 헤더
        ]
        
        cleaned = answer
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # 연속된 공백과 줄바꿈 정리
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _format_compressed_answer(self, compressed: CompressedAnswer, urgency_level: str) -> str:
        """압축된 답변을 최종 형태로 포맷팅"""
        if urgency_level == "immediate":
            # 즉시 대응 - 최소한의 정보만
            answer = f"🚨 {compressed.key_action}"
            if compressed.safety_warning:
                answer += f"\n⚠️ {compressed.safety_warning}"
            return answer
            
        elif urgency_level == "urgent":
            # 빠른 대응 - 핵심 + 간단 단계
            answer = f"⚡ {compressed.key_action}"
            if compressed.quick_steps and len(compressed.quick_steps) <= 2:
                answer += "\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(compressed.quick_steps[:2])])
            return answer
            
        else:
            # 일반 상황 - 상대적으로 상세
            answer = f"📍 {compressed.key_action}"
            if compressed.quick_steps:
                answer += "\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(compressed.quick_steps[:3])])
            if compressed.follow_up:
                answer += f"\n📋 주행 후: {compressed.follow_up}"
            return answer
    
    def _simple_compression(self, answer: str) -> str:
        """간단한 압축 (LLM 실패 시 백업)"""
        # 첫 번째 문장만 추출
        sentences = answer.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 100:
                return first_sentence[:100] + "..."
            return first_sentence + "."
        return "⚠️ 안전한 곳에서 상세 확인 필요"
