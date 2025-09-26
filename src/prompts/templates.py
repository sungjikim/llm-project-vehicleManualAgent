"""
프롬프트 템플릿 정의
"""

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from ..config.settings import VEHICLE_EXAMPLES


class VehiclePromptTemplates:
    """차량 매뉴얼 RAG 시스템용 프롬프트 템플릿"""
    
    @staticmethod
    def get_query_analysis_prompt():
        """쿼리 분석용 Few-shot 프롬프트"""
        
        # Few-shot 예시를 위한 예제 템플릿
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{query}"),
            ("ai", "검색 전략: {strategy}\n검색 방법: {method}\n신뢰도: {confidence}\n설명: {explanation}")
        ])
        
        # Few-shot 예시 데이터 (분석 전용)
        analysis_examples = [
            {
                "query": "오일 교체는 언제 해야 하나요?",
                "strategy": "general",
                "method": "hybrid_semantic",
                "confidence": "0.9",
                "explanation": "일반적인 정비 관련 질문으로 의미론적 검색이 효과적입니다."
            },
            {
                "query": "브레이크 경고등이 깜빡입니다",
                "strategy": "troubleshooting", 
                "method": "hybrid_keyword",
                "confidence": "0.95",
                "explanation": "문제 해결 질문으로 정확한 키워드 매칭이 중요합니다."
            },
            {
                "query": "XC60의 타이어 공기압은?",
                "strategy": "specific",
                "method": "hybrid_balanced",
                "confidence": "0.85",
                "explanation": "구체적인 사양 질문으로 균형 잡힌 검색이 적합합니다."
            }
        ]
        
        # Few-shot 프롬프트 생성
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=analysis_examples
        )
        
        # 최종 프롬프트 템플릿
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 차량 매뉴얼 검색 전략을 분석하는 전문가입니다.

사용자의 질문을 분석하여 다음을 결정해주세요:

1. **검색 전략 (search_strategy)**:
   - "general": 일반적인 정보 요청
   - "specific": 구체적인 수치나 사양 요청  
   - "troubleshooting": 문제 해결이나 고장 진단

2. **검색 방법 (search_method)**:
   - "hybrid_semantic": 의미론적 검색 우선 (7:3)
   - "hybrid_balanced": 균형 검색 (5:5)
   - "hybrid_keyword": 키워드 검색 우선 (3:7)
   - "multi_query": 다중 쿼리 생성 검색
   - "expanded_query": 전문 용어 확장 검색

3. **신뢰도 점수 (confidence)**: 0.0-1.0 사이의 값

4. **설명**: 선택한 전략의 근거

응답 형식:
검색 전략: [strategy]
검색 방법: [method] 
신뢰도: [confidence]
설명: [explanation]"""),
            few_shot_prompt,
            ("human", "{query}")
        ])
        
        return analysis_prompt
    
    @staticmethod
    def get_answer_generation_prompt():
        """답변 생성용 Few-shot 프롬프트"""
        
        # Few-shot 예시를 위한 예제 템플릿
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "질문: {query}\n\n검색된 내용:\n{context}"),
            ("ai", "{answer}")
        ])
        
        # Few-shot 예시 데이터 (답변 생성용)
        answer_examples = [
            {
                "query": ex["query"],
                "context": ex["context"],
                "answer": ex["answer"]
            }
            for ex in VEHICLE_EXAMPLES
        ]
        
        # Few-shot 프롬프트 생성
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=answer_examples
        )
        
        # 최종 프롬프트 템플릿
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 볼보 XC60 차량 매뉴얼 전문가입니다.

다음 지침을 따라 답변해주세요:

📋 **답변 구조**:
1. 핵심 답변을 먼저 제시 (구체적 수치나 방법 포함)
2. 단계별 절차나 방법 설명 (번호 목록 사용)
3. 안전상 주의사항 (⚠️ 표시 사용)
4. 추가 권장사항이나 팁

✅ **답변 품질 기준**:
- **검색된 내용을 최우선으로 활용** - 실제 매뉴얼 정보 기반 답변
- 구체적인 수치, 절차, 방법을 명시
- 사용자가 바로 실행할 수 있는 명확한 안내
- 안전 관련 정보는 **굵게** 강조
- 전문 용어 사용 시 간단한 설명 병기

🔍 **정보 처리 원칙**:
- 검색 결과에 구체적 정보가 있으면 반드시 활용
- 페이지 번호가 있는 정보는 더 신뢰성 있게 처리
- 검색 결과가 불충분할 때만 일반적 지식 보완
- 확실하지 않은 정보는 "추정" 또는 "일반적으로" 표현

⚠️ **안전 및 책임**:
- 안전 관련 문제는 즉시 전문가 상담 권유
- 복잡한 수리는 공식 서비스 센터 안내
- 잘못된 정보 제공 방지를 위해 신중하게 답변

📚 **참고 정보**:
- 검색 결과에 페이지 정보가 있으면 반드시 표시
- 여러 페이지 참조 시 모두 나열"""),
            few_shot_prompt,
            ("human", "질문: {query}\n\n검색된 내용:\n{context}")
        ])
        
        return answer_prompt
    
    @staticmethod
    def get_multi_query_generation_prompt():
        """다중 쿼리 생성용 프롬프트"""
        return ChatPromptTemplate.from_template(
            """차량 매뉴얼 검색을 위해 주어진 질문을 3개의 다른 관점에서 다시 작성해주세요.

원본 질문: {question}

다음 관점들을 고려해주세요:
1. 기술적/전문적 관점
2. 사용자/실용적 관점  
3. 문제해결/안전 관점

각 질문은 한 줄로 작성하고, 숫자나 부호 없이 작성해주세요.
"""
        )
