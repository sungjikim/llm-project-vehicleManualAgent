"""
답변 생성 SubGraph
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from ...models.states import AnswerGenerationState
from ...config.settings import DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMPERATURE
from ...prompts.templates import VehiclePromptTemplates
from ...utils.answer_evaluator import AnswerEvaluator
from ...utils.emergency_detector import EmergencyDetector


class AnswerGenerationSubGraph:
    """답변 생성 SubGraph"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_LLM_TEMPERATURE
        )
        self.answer_evaluator = AnswerEvaluator()
        self.emergency_detector = EmergencyDetector()
        self.answer_prompt = VehiclePromptTemplates.get_answer_generation_prompt()
    
    def answer_generator(self, state: AnswerGenerationState) -> Dict[str, Any]:
        """답변 생성 노드"""
        query = state["query"]
        search_results = state.get("search_results", [])
        page_references = state.get("page_references", [])
        is_emergency = state.get("is_emergency", False)
        emergency_level = state.get("emergency_level", "NORMAL")
        
        try:
            # 컨텍스트 구성 (페이지 정보 강화)
            context_parts = []
            valid_pages = []
            
            for i, result in enumerate(search_results[:5], 1):
                content = result.get("content", "")
                page = result.get("page", 0)
                score = result.get("score", 0.0)
                
                if page > 0:
                    valid_pages.append(page)
                    context_parts.append(f"[검색결과 {i}] (페이지 {page}, 관련도: {score:.2f})\n{content}")
                else:
                    context_parts.append(f"[검색결과 {i}] (관련도: {score:.2f})\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # 응급 상황 프롬프트 강화
            emergency_enhancement = ""
            if is_emergency:
                emergency_enhancement = self.emergency_detector.get_emergency_prompt_enhancement(emergency_level)
                context = emergency_enhancement + "\n\n" + context
                print(f"🚨 응급 답변 생성 모드: {emergency_level}")
            
            # 페이지 참조 정보 추가
            page_info = ""
            if valid_pages:
                unique_pages = sorted(list(set(valid_pages)))
                if len(unique_pages) == 1:
                    page_info = f"\n\n📚 참고 페이지: {unique_pages[0]}"
                elif len(unique_pages) <= 3:
                    page_info = f"\n\n📚 참고 페이지: {', '.join(map(str, unique_pages))}"
                else:
                    page_info = f"\n\n📚 주요 참고 페이지: {', '.join(map(str, unique_pages[:3]))} 외"
            
            # Few-shot 프롬프트로 답변 생성
            answer_chain = self.answer_prompt | self.llm | StrOutputParser()
            final_answer = answer_chain.invoke({
                "query": query,
                "context": context
            })
            
            # 페이지 정보가 답변에 없으면 추가
            if page_info and "📚" not in final_answer:
                final_answer += page_info
            
            # 응급 상황 등급 표시를 답변 첫 줄에 추가
            emergency_header = ""
            if is_emergency:
                # 응급 상황 헤더 생성
                emergency_icons = {
                    "CRITICAL": "🔥",
                    "HIGH": "🚨", 
                    "MEDIUM": "⚠️",
                    "LOW": "🔍"
                }
                icon = emergency_icons.get(emergency_level, "🚨")
                emergency_header = f"{icon} **{emergency_level} 응급 상황**\n\n"
                
                # 응급 상황에서는 신뢰도 평가 간소화 (속도 우선)
                confidence_percentage = 85.0  # 응급 상황 기본 신뢰도
                reliability_grade = "높음 (A)"
                
                # 응급 상황 경고 추가
                emergency_warning = f"\n\n🚨 **응급 상황 ({emergency_level})**"
                if emergency_level == "CRITICAL":
                    emergency_warning += "\n⚠️ 생명 위험 상황입니다. 즉시 조치하고 119에 신고하세요."
                elif emergency_level == "HIGH":
                    emergency_warning += "\n⚠️ 즉시 안전 조치가 필요합니다. 전문가에게 연락하세요."
                else:
                    emergency_warning += "\n⚠️ 신속한 대응이 필요합니다."
                
                final_answer = emergency_header + final_answer + emergency_warning
            else:
                # 일반 질문 헤더 생성
                emergency_header = "📝 **일반 질문**\n\n"
                
                # 일반 상황 신뢰도 평가
                evaluation = self.answer_evaluator.evaluate_answer(query, final_answer, search_results)
                confidence_percentage = evaluation['percentage']
                reliability_grade = evaluation['reliability_grade']
                
                final_answer = emergency_header + final_answer
            
            # 신뢰도 정보를 답변에 추가
            confidence_info = f"\n\n🔍 **답변 신뢰도**: {confidence_percentage}% ({reliability_grade})"
            
            # 신뢰도에 따른 추가 안내 (응급 상황이 아닐 때만)
            if not is_emergency:
                if confidence_percentage >= 80:
                    confidence_info += "\n✅ 높은 신뢰도의 답변입니다."
                elif confidence_percentage >= 60:
                    confidence_info += "\n⚠️ 추가 확인을 권장합니다."
                else:
                    confidence_info += "\n❌ 전문가 상담을 강력히 권장합니다."
            
            final_answer_with_confidence = final_answer + confidence_info
            
            # 응급 상황에서 evaluation 변수가 없을 수 있으므로 처리
            if is_emergency and 'evaluation' not in locals():
                evaluation = {
                    "total_score": confidence_percentage / 100,
                    "percentage": confidence_percentage,
                    "reliability_grade": reliability_grade,
                    "emergency_mode": True,
                    "emergency_level": emergency_level
                }
            
            return {
                "final_answer": final_answer_with_confidence,
                "confidence_score": confidence_percentage / 100,
                "evaluation_details": evaluation if 'evaluation' in locals() else None
            }
            
        except Exception as e:
            print(f"답변 생성 오류: {str(e)}")
            return {"final_answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}"}
    
    def create_graph(self) -> StateGraph:
        """답변 생성 SubGraph 생성"""
        workflow = StateGraph(AnswerGenerationState)
        
        # 노드 추가
        workflow.add_node("answer_generator", self.answer_generator)
        
        # 엣지 추가
        workflow.set_entry_point("answer_generator")
        workflow.add_edge("answer_generator", END)
        
        return workflow.compile()
    
    def invoke(self, query: str, search_results: List[Dict[str, Any]], 
               page_references: List[int], is_emergency: bool = False, 
               emergency_level: str = "NORMAL") -> Dict[str, Any]:
        """SubGraph 실행"""
        graph = self.create_graph()
        
        initial_state = {
            "query": query,
            "search_results": search_results,
            "page_references": page_references,
            "is_emergency": is_emergency,
            "emergency_level": emergency_level,
            "final_answer": "",
            "confidence_score": 0.0,
            "evaluation_details": None
        }
        
        return graph.invoke(initial_state)
