"""
검색 파이프라인 SubGraph
"""

import re
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from ...models.states import SearchPipelineState
from ...config.settings import DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMPERATURE, DEFAULT_TOP_K
from ...prompts.templates import VehiclePromptTemplates
from ...tools.search_tools import (
    vector_store, bm25_retriever, multi_query_retriever,
    cross_encoder_retriever, compression_retriever
)


class SearchPipelineSubGraph:
    """검색 파이프라인 SubGraph"""
    
    def __init__(self, search_options: Dict[str, Any], rerank_compression_options: Dict[str, Any]):
        self.llm = ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_LLM_TEMPERATURE
        )
        self.search_options = search_options
        self.rerank_compression_options = rerank_compression_options
        self.analysis_prompt = VehiclePromptTemplates.get_query_analysis_prompt()
    
    def query_analyzer(self, state: SearchPipelineState) -> Dict[str, Any]:
        """쿼리 분석 노드"""
        query = state["query"]
        is_emergency = state.get("is_emergency", False)
        
        try:
            # 응급 상황이면 이미 설정된 전략 사용
            if is_emergency:
                search_strategy = state.get("search_strategy", "troubleshooting")
                search_method = state.get("search_method", "hybrid_keyword")
                compression_method = state.get("compression_method", "rerank_compress_troubleshooting")
                confidence_score = 0.95  # 응급 상황은 높은 신뢰도로 처리
                
                print(f"🚨 응급 모드: {search_strategy} -> {search_method}")
                
                return {
                    "search_strategy": search_strategy,
                    "search_method": search_method,
                    "confidence_score": confidence_score,
                    "compression_method": compression_method
                }
            
            # 일반 질문 처리
            analysis_chain = self.analysis_prompt | self.llm | StrOutputParser()
            analysis_result = analysis_chain.invoke({"query": query})
            
            # 결과 파싱
            search_strategy = self._extract_field(analysis_result, "검색 전략")
            search_method = self._extract_field(analysis_result, "검색 방법")
            confidence_str = self._extract_field(analysis_result, "신뢰도")
            
            # 신뢰도 점수 변환
            try:
                confidence_score = float(confidence_str)
            except:
                confidence_score = 0.8
            
            # 쿼리 확장 및 다중 쿼리 로직
            original_search_method = search_method
            if any(keyword in query.lower() for keyword in ['교체', '문제', '고장', '이상']):
                search_method = "expanded_query"
                print(f"🔧 키워드 기반 검색 방법 변경: {original_search_method} → {search_method}")
            elif len(query) > 20 and '?' in query:
                search_method = "multi_query"
                print(f"🔧 복잡한 질문 감지, 검색 방법 변경: {original_search_method} → {search_method}")
            
            # 재순위화/압축 방법 선택
            compression_method = self._select_compression_method(search_strategy)
            
            # 검색 전략 로그 출력
            print(f"📋 쿼리 분석 결과:")
            print(f"   • 검색 전략: {search_strategy}")
            print(f"   • 검색 방법: {search_method}")
            print(f"   • 신뢰도: {confidence_score}")
            print(f"   • 압축 방법: {compression_method}")
            
            return {
                "search_strategy": search_strategy,
                "search_method": search_method,
                "confidence_score": confidence_score,
                "compression_method": compression_method
            }
            
        except Exception as e:
            print(f"쿼리 분석 오류: {str(e)}")
            return {
                "search_strategy": "general",
                "search_method": "hybrid_semantic",
                "confidence_score": 0.5,
                "compression_method": "rerank_compress_general"
            }
    
    def search_executor(self, state: SearchPipelineState) -> Dict[str, Any]:
        """검색 실행 노드"""
        query = state["query"]
        search_method = state.get("search_method", "hybrid_semantic")
        compression_method = state.get("compression_method", "rerank_compress_general")
        
        print(f"🔍 검색 실행 시작:")
        print(f"   • 선택된 검색 방법: {search_method}")
        print(f"   • 압축/재순위화 방법: {compression_method}")
        
        try:
            # 1차 검색 수행
            if search_method == "expanded_query":
                print("🔎 확장 쿼리 검색 실행 중...")
                from ...tools.search_tools import expanded_query_search
                search_results = expanded_query_search.invoke({"query": query, "top_k": DEFAULT_TOP_K})
            elif search_method == "multi_query":
                print("🔎 다중 쿼리 검색 실행 중...")
                from ...tools.search_tools import multi_query_search
                search_results = multi_query_search.invoke({"query": query, "top_k": DEFAULT_TOP_K})
            else:
                # 기본 검색 수행
                print(f"🔎 기본 검색 실행 중... (방법: {search_method})")
                retriever = self.search_options.get(search_method)
                if retriever:
                    docs = retriever.invoke(query)
                    search_results = [
                        {
                            "content": doc.page_content,
                            "page": doc.metadata.get("page", 0),
                            "source": doc.metadata.get("source", ""),
                            "score": 1.0
                        }
                        for doc in docs[:DEFAULT_TOP_K]
                    ]
                else:
                    print(f"❌ 검색 방법 '{search_method}'을 찾을 수 없습니다.")
                    search_results = [{"content": "검색 방법을 찾을 수 없습니다.", "page": 0, "score": 0.0}]
            
            print(f"📊 1차 검색 결과: {len(search_results)}개 문서 발견")
            
            # 2차 재순위화/압축 적용
            if compression_method and compression_method != "none":
                print(f"🔄 재순위화/압축 적용 중... (방법: {compression_method})")
                original_count = len(search_results)
                search_results = self._apply_compression(query, search_results, compression_method)
                print(f"✅ 재순위화 완료: {original_count}개 → {len(search_results)}개 문서")
            else:
                print("⏭️  재순위화/압축 건너뜀")
            
            # 페이지 참조 추출
            page_references = list(set([
                result.get("page", 0) for result in search_results if result.get("page", 0) > 0
            ]))
            
            print(f"📄 최종 검색 결과: {len(search_results)}개 문서, {len(page_references)}개 페이지 참조")
            
            return {
                "search_results": search_results,
                "page_references": page_references
            }
            
        except Exception as e:
            print(f"검색 실행 오류: {str(e)}")
            return {
                "search_results": [{"content": f"검색 중 오류 발생: {str(e)}", "page": 0, "score": 0.0}],
                "page_references": []
            }
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """텍스트에서 특정 필드 값 추출"""
        pattern = rf"{field_name}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return "general" if "전략" in field_name else "hybrid_semantic"
    
    def _select_compression_method(self, search_strategy: str) -> str:
        """검색 전략에 따른 압축 방법 선택"""
        compression_map = {
            "troubleshooting": "rerank_compress_troubleshooting",
            "specific": "rerank_compress_specific",
            "general": "rerank_compress_general"
        }
        return compression_map.get(search_strategy, "rerank_compress_general")
    
    def _apply_compression(self, query: str, search_results: List[Dict], compression_method: str) -> List[Dict]:
        """압축/재순위화 적용"""
        try:
            retriever = self.rerank_compression_options.get(compression_method)
            if retriever:
                # 압축 검색 수행
                if compression_method.startswith("rerank_compress"):
                    from ...tools.search_tools import cross_encoder_rerank_search
                    return cross_encoder_rerank_search.invoke({"query": query, "top_k": DEFAULT_TOP_K})
                else:
                    from ...tools.search_tools import contextual_compression_search
                    return contextual_compression_search.invoke({"query": query, "top_k": DEFAULT_TOP_K})
            
        except Exception as e:
            print(f"압축 적용 오류: {str(e)}")
        
        return search_results
    
    def create_graph(self) -> StateGraph:
        """검색 파이프라인 SubGraph 생성"""
        workflow = StateGraph(SearchPipelineState)
        
        # 노드 추가
        workflow.add_node("query_analyzer", self.query_analyzer)
        workflow.add_node("search_executor", self.search_executor)
        
        # 엣지 추가
        workflow.set_entry_point("query_analyzer")
        workflow.add_edge("query_analyzer", "search_executor")
        workflow.add_edge("search_executor", END)
        
        return workflow.compile()
    
    def invoke(self, query: str, is_emergency: bool = False, emergency_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """SubGraph 실행"""
        graph = self.create_graph()
        
        initial_state = {
            "query": query,
            "search_strategy": "",
            "search_method": "",
            "compression_method": "",
            "confidence_score": 0.0,
            "search_results": [],
            "page_references": []
        }
        
        # 응급 상황 데이터가 있으면 추가
        if is_emergency and emergency_data:
            initial_state.update({
                "is_emergency": True,
                "search_strategy": emergency_data.get("search_strategy", "troubleshooting"),
                "search_method": emergency_data.get("search_method", "hybrid_keyword"),
                "compression_method": emergency_data.get("compression_method", "rerank_compress_troubleshooting")
            })
        
        return graph.invoke(initial_state)
