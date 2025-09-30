"""
주행 상황 처리 SubGraph
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from ...models.subgraph_states import DrivingContextState
from ...utils.driving_context_detector import DrivingContextDetector


class DrivingContextSubGraph:
    """주행 상황 처리 SubGraph"""
    
    def __init__(self):
        self.driving_detector = DrivingContextDetector()
    
    def driving_context_processor(self, state: DrivingContextState) -> Dict[str, Any]:
        """주행 중 상황 감지 및 답변 압축 노드"""
        query = state["query"]
        original_answer = state.get("original_answer", "")
        
        # Emergency 정보 참조 (정보 공유 최적화)
        is_emergency = state.get("is_emergency", False)
        emergency_level = state.get("emergency_level", "NORMAL")
        
        try:
            print("🚗 주행 상황 분석 중...")
            
            # 1. 주행 중 상황 감지 (Emergency 정보 활용)
            if is_emergency and emergency_level in ["CRITICAL", "HIGH"]:
                # 응급 상황이면 주행 중으로 가정하고 간소화된 분석
                print("⚡ 응급 상황 감지됨 - 주행 중으로 가정")
                is_driving = True
                confidence = 0.95
                urgency_level = "immediate" if emergency_level == "CRITICAL" else "urgent"
                driving_analysis = {
                    "is_driving": True,
                    "confidence": confidence,
                    "urgency_level": urgency_level,
                    "compression_needed": True,
                    "driving_indicators": [f"응급상황({emergency_level})"]
                }
            else:
                # 일반 상황에서만 상세 주행 분석 수행
                driving_analysis = self.driving_detector.detect_driving_context(query)
                is_driving = driving_analysis["is_driving"]
                confidence = driving_analysis["confidence"]
                urgency_level = driving_analysis["urgency_level"]
            
            print(f"🚗 주행 상황 분석 결과:")
            print(f"   • 주행 중 여부: {is_driving} (신뢰도: {confidence:.2f})")
            print(f"   • 긴급도: {urgency_level}")
            
            if driving_analysis["driving_indicators"]:
                indicators = ", ".join(driving_analysis["driving_indicators"])
                print(f"   • 감지된 지표: {indicators}")
            
            # 2. 주행 중이면 답변 압축
            if is_driving and driving_analysis["compression_needed"]:
                print("📱 주행 중 모드 - 답변 압축 중...")
                
                compression_result = self.driving_detector.compress_answer(
                    original_answer, query, urgency_level
                )
                
                compressed_answer = compression_result["compressed_answer"]
                compression_ratio = compression_result["compression_ratio"]
                
                print(f"✅ 답변 압축 완료 (압축률: {compression_ratio:.1%})")
                
                # 주행 중 안전 메시지 추가
                if urgency_level == "immediate":
                    safety_message = "\n\n🛑 **즉시 안전한 곳에 정차하세요**"
                elif urgency_level == "urgent":
                    safety_message = "\n\n⚠️ **가능한 빨리 안전한 곳에서 확인하세요**"
                else:
                    safety_message = "\n\n📋 **주행 후 상세 내용을 확인하세요**"
                
                final_compressed = compressed_answer + safety_message
                
                return {
                    "is_driving": True,
                    "driving_confidence": confidence,
                    "driving_indicators": driving_analysis["driving_indicators"],
                    "driving_urgency": urgency_level,
                    "compression_needed": True,
                    "compressed_answer": final_compressed,
                    "final_answer": final_compressed  # 최종 답변을 압축된 버전으로 대체
                }
            
            else:
                # 주행 중이 아니거나 압축이 필요하지 않은 경우
                print("🏠 일반 모드 - 원본 답변 유지")
                
                return {
                    "is_driving": False,
                    "driving_confidence": confidence,
                    "driving_indicators": driving_analysis.get("driving_indicators", []),
                    "driving_urgency": "normal",
                    "compression_needed": False,
                    "compressed_answer": "",
                    "final_answer": original_answer  # 원본 답변 유지
                }
                
        except Exception as e:
            print(f"주행 상황 처리 오류: {str(e)}")
            # 오류 시 원본 답변 유지
            return {
                "is_driving": False,
                "driving_confidence": 0.0,
                "driving_indicators": [],
                "driving_urgency": "normal",
                "compression_needed": False,
                "compressed_answer": "",
                "final_answer": original_answer
            }
    
    def create_graph(self) -> StateGraph:
        """주행 상황 처리 SubGraph 생성"""
        workflow = StateGraph(DrivingContextState)
        
        # 노드 추가
        workflow.add_node("driving_context_processor", self.driving_context_processor)
        
        # 엣지 추가
        workflow.set_entry_point("driving_context_processor")
        workflow.add_edge("driving_context_processor", END)
        
        return workflow.compile()
    
    def invoke(self, query: str, original_answer: str, is_emergency: bool = False, 
               emergency_level: str = "NORMAL") -> Dict[str, Any]:
        """SubGraph 실행"""
        graph = self.create_graph()
        
        initial_state = {
            "query": query,
            "original_answer": original_answer,
            "is_emergency": is_emergency,
            "emergency_level": emergency_level,
            "is_driving": False,
            "driving_confidence": 0.0,
            "driving_indicators": [],
            "driving_urgency": "normal",
            "compression_needed": False,
            "compressed_answer": "",
            "final_answer": ""
        }
        
        return graph.invoke(initial_state)
