"""
응급 상황 감지 SubGraph
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from ...models.states import EmergencyDetectionState
from ...utils.emergency_detector import EmergencyDetector


class EmergencyDetectionSubGraph:
    """응급 상황 감지 SubGraph"""
    
    def __init__(self):
        self.emergency_detector = EmergencyDetector()
    
    def emergency_classifier(self, state: EmergencyDetectionState) -> Dict[str, Any]:
        """응급 상황 분류 노드"""
        query = state["query"]
        
        try:
            # 응급 상황 감지
            emergency_analysis = self.emergency_detector.detect_emergency(query)
            
            print(f"🚨 응급 상황 분석: {emergency_analysis['priority_level']} "
                  f"(점수: {emergency_analysis['total_score']})")
            
            if emergency_analysis["is_emergency"]:
                # 응급 상황 처리 모드
                search_strategy = emergency_analysis["search_strategy"]
                
                return {
                    "is_emergency": True,
                    "emergency_level": emergency_analysis["priority_level"],
                    "emergency_score": emergency_analysis["total_score"],
                    "search_strategy": "troubleshooting",  # 강제로 문제해결 전략
                    "search_method": search_strategy["search_method"],
                    "compression_method": search_strategy["compression_method"],
                    "emergency_analysis": emergency_analysis
                }
            else:
                # 일반 질문 처리 모드  
                return {
                    "is_emergency": False,
                    "emergency_level": "NORMAL",
                    "emergency_score": emergency_analysis["total_score"],
                    "emergency_analysis": emergency_analysis
                }
                
        except Exception as e:
            print(f"응급 상황 분류 오류: {str(e)}")
            # 오류 시 안전하게 일반 모드로 처리
            return {
                "is_emergency": False,
                "emergency_level": "NORMAL", 
                "emergency_score": 0,
                "emergency_analysis": None
            }
    
    def create_graph(self) -> StateGraph:
        """응급 상황 감지 SubGraph 생성"""
        workflow = StateGraph(EmergencyDetectionState)
        
        # 노드 추가
        workflow.add_node("emergency_classifier", self.emergency_classifier)
        
        # 엣지 추가
        workflow.set_entry_point("emergency_classifier")
        workflow.add_edge("emergency_classifier", END)
        
        return workflow.compile()
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """SubGraph 실행"""
        graph = self.create_graph()
        
        initial_state = {
            "query": query,
            "is_emergency": False,
            "emergency_level": "NORMAL",
            "emergency_score": 0.0,
            "emergency_analysis": {},
            "search_strategy": "",
            "search_method": "",
            "compression_method": ""
        }
        
        return graph.invoke(initial_state)
