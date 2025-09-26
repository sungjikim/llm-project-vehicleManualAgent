"""
차량 매뉴얼 RAG 에이전트
"""

import re
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from ..models.state import AgentState
from ..config.settings import (
    DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMPERATURE, DEFAULT_TOP_K, WEIGHT_CONFIGS
)
from ..retrievers.vector_retriever import VectorStoreManager
from ..retrievers.hybrid_retriever import HybridRetrieverManager
from ..retrievers.compression_retriever import CompressionRetrieverManager
from ..prompts.templates import VehiclePromptTemplates
from ..utils.document_loader import DocumentLoader
from ..tools.search_tools import (
    vector_store, bm25_retriever, hybrid_retriever, multi_query_retriever,
    cross_encoder_retriever, compression_retriever
)


class VehicleManualAgent:
    """차량 매뉴얼 RAG 에이전트"""
    
    def __init__(self, pdf_path: str):
        # LLM 및 임베딩 모델 초기화
        self.llm = ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_LLM_TEMPERATURE
        )
        self.embeddings = OpenAIEmbeddings()
        
        # 문서 로더 초기화
        self.document_loader = DocumentLoader()
        
        # 프롬프트 템플릿 초기화
        self.analysis_prompt = VehiclePromptTemplates.get_query_analysis_prompt()
        self.answer_prompt = VehiclePromptTemplates.get_answer_generation_prompt()
        
        # 검색기 관리자들 초기화
        self.vector_manager = VectorStoreManager(pdf_path)
        self.hybrid_manager = None
        self.compression_manager = None
        
        # 검색 옵션 설정
        self.search_options = {}
        self.rerank_compression_options = {}
        
        # 시스템 초기화
        self._initialize_system()
    
    def _initialize_system(self):
        """시스템 전체 초기화"""
        print("🚀 차량 매뉴얼 RAG 시스템 초기화 중...")
        
        # 1. 벡터 저장소 초기화
        vector_store_instance = self.vector_manager.initialize_vector_store()
        if vector_store_instance is None:
            raise Exception("벡터 저장소 초기화에 실패했습니다.")
        
        # 전역 변수 설정
        global vector_store
        vector_store = vector_store_instance
        
        # 2. 문서 로드 (하이브리드 검색기용)
        documents = self.document_loader.load_and_split_pdf(self.vector_manager.pdf_path)
        if not documents:
            raise Exception("문서 로딩에 실패했습니다.")
        
        # 3. 하이브리드 검색기 초기화
        self.hybrid_manager = HybridRetrieverManager(vector_store_instance, self.llm)
        self.hybrid_manager.initialize_bm25_retriever(documents)
        self.hybrid_manager.initialize_multi_query_retriever()
        
        # 전역 변수 설정
        global bm25_retriever, multi_query_retriever
        bm25_retriever = self.hybrid_manager.get_bm25_retriever()
        multi_query_retriever = self.hybrid_manager.get_multi_query_retriever()
        
        # 4. 압축/재순위화 검색기 초기화
        self.compression_manager = CompressionRetrieverManager(
            vector_store_instance, self.embeddings, self.llm
        )
        self.compression_manager.initialize_cross_encoder_retriever()
        self.compression_manager.initialize_contextual_compression()
        
        # 전역 변수 설정
        global cross_encoder_retriever, compression_retriever
        cross_encoder_retriever = self.compression_manager.get_cross_encoder_retriever()
        compression_retriever = self.compression_manager.get_compression_retriever()
        
        # 5. 검색 옵션 설정
        self._setup_search_options()
        
        print("✅ 시스템 초기화 완료!")
    
    def _setup_search_options(self):
        """검색 옵션 설정"""
        semantic_retriever = vector_store.as_retriever(search_kwargs={"k": DEFAULT_TOP_K})
        
        self.search_options = {
            "vector_only": semantic_retriever,
            "bm25_only": bm25_retriever,
            "hybrid_semantic": EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[WEIGHT_CONFIGS["hybrid_semantic"], 1 - WEIGHT_CONFIGS["hybrid_semantic"]]
            ),
            "hybrid_balanced": EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[WEIGHT_CONFIGS["hybrid_balanced"], 1 - WEIGHT_CONFIGS["hybrid_balanced"]]
            ),
            "hybrid_keyword": EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[WEIGHT_CONFIGS["hybrid_keyword"], 1 - WEIGHT_CONFIGS["hybrid_keyword"]]
            ),
            "multi_query": multi_query_retriever,
            "expanded_query": self.search_options.get("hybrid_semantic")  # 기본값으로 설정
        }
        
        # 재순위화 및 압축 옵션
        self.rerank_compression_options = {
            "rerank_only": cross_encoder_retriever,
            "compress_only": compression_retriever,
            "rerank_compress_general": cross_encoder_retriever,
            "rerank_compress_specific": compression_retriever,
            "rerank_compress_troubleshooting": cross_encoder_retriever
        }
    
    def query_analyzer(self, state: AgentState) -> Dict[str, Any]:
        """쿼리 분석 노드"""
        query = state["query"]
        
        try:
            # Few-shot 프롬프트로 쿼리 분석
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
            if any(keyword in query.lower() for keyword in ['교체', '문제', '고장', '이상']):
                search_method = "expanded_query"
            elif len(query) > 20 and '?' in query:
                search_method = "multi_query"
            
            # 재순위화/압축 방법 선택
            compression_method = self._select_compression_method(search_strategy)
            
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
    
    def search_executor(self, state: AgentState) -> Dict[str, Any]:
        """검색 실행 노드"""
        query = state["query"]
        search_method = state.get("search_method", "hybrid_semantic")
        compression_method = state.get("compression_method", "rerank_compress_general")
        
        try:
            # 1차 검색 수행
            if search_method == "expanded_query":
                from ..tools.search_tools import expanded_query_search
                search_results = expanded_query_search.invoke({"query": query, "top_k": DEFAULT_TOP_K})
            elif search_method == "multi_query":
                from ..tools.search_tools import multi_query_search
                search_results = multi_query_search.invoke({"query": query, "top_k": DEFAULT_TOP_K})
            else:
                # 기본 검색 수행
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
                    search_results = [{"content": "검색 방법을 찾을 수 없습니다.", "page": 0, "score": 0.0}]
            
            # 2차 재순위화/압축 적용
            if compression_method and compression_method != "none":
                search_results = self._apply_compression(query, search_results, compression_method)
            
            # 페이지 참조 추출
            page_references = list(set([
                result.get("page", 0) for result in search_results if result.get("page", 0) > 0
            ]))
            
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
    
    def answer_generator(self, state: AgentState) -> Dict[str, Any]:
        """답변 생성 노드"""
        query = state["query"]
        search_results = state.get("search_results", [])
        page_references = state.get("page_references", [])
        
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
            
            return {"final_answer": final_answer}
            
        except Exception as e:
            print(f"답변 생성 오류: {str(e)}")
            return {"final_answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}"}
    
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
                    from ..tools.search_tools import cross_encoder_rerank_search
                    return cross_encoder_rerank_search.invoke({"query": query, "top_k": DEFAULT_TOP_K})
                else:
                    from ..tools.search_tools import contextual_compression_search
                    return contextual_compression_search.invoke({"query": query, "top_k": DEFAULT_TOP_K})
            
        except Exception as e:
            print(f"압축 적용 오류: {str(e)}")
        
        return search_results
    
    def create_graph(self) -> StateGraph:
        """LangGraph 워크플로우 생성"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("query_analyzer", self.query_analyzer)
        workflow.add_node("search_executor", self.search_executor)
        workflow.add_node("answer_generator", self.answer_generator)
        
        # 엣지 추가
        workflow.set_entry_point("query_analyzer")
        workflow.add_edge("query_analyzer", "search_executor")
        workflow.add_edge("search_executor", "answer_generator")
        workflow.add_edge("answer_generator", END)
        
        return workflow.compile()
    
    def query(self, user_query: str) -> str:
        """사용자 쿼리 처리"""
        try:
            graph = self.create_graph()
            
            # 초기 상태 설정
            initial_state = {
                "messages": [],
                "query": user_query,
                "search_results": [],
                "context": "",
                "final_answer": "",
                "search_strategy": "",
                "search_method": "",
                "confidence_score": 0.0,
                "page_references": [],
                "need_clarification": False
            }
            
            # 그래프 실행
            result = graph.invoke(initial_state)
            
            return result.get("final_answer", "답변을 생성할 수 없습니다.")
            
        except Exception as e:
            return f"쿼리 처리 중 오류가 발생했습니다: {str(e)}"
