"""
차량 매뉴얼 RAG 에이전트 - SubGraph 아키텍처
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langgraph.graph import StateGraph, END

from ..models.subgraph_states import MainAgentState
from ..config.settings import (
    DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMPERATURE, DEFAULT_TOP_K, WEIGHT_CONFIGS
)
from ..retrievers.vector_retriever import VectorStoreManager
from ..retrievers.hybrid_retriever import HybridRetrieverManager
from ..retrievers.compression_retriever import CompressionRetrieverManager
from ..utils.document_loader import DocumentLoader
from ..tools.search_tools import (
    vector_store, bm25_retriever, hybrid_retriever, multi_query_retriever,
    cross_encoder_retriever, compression_retriever
)
from .subgraphs import (
    EmergencyDetectionSubGraph,
    SearchPipelineSubGraph,
    AnswerGenerationSubGraph,
    DrivingContextSubGraph
)


class VehicleManualAgentSubGraph:
    """차량 매뉴얼 RAG 에이전트 - SubGraph 아키텍처"""
    
    def __init__(self, pdf_path: str):
        # LLM 및 임베딩 모델 초기화
        self.llm = ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_LLM_TEMPERATURE
        )
        self.embeddings = OpenAIEmbeddings()
        
        # 문서 로더 초기화
        self.document_loader = DocumentLoader()
        
        # 검색기 관리자들 초기화
        self.vector_manager = VectorStoreManager(pdf_path)
        self.hybrid_manager = None
        self.compression_manager = None
        
        # 검색 옵션 설정
        self.search_options = {}
        self.rerank_compression_options = {}
        
        # SubGraph 인스턴스들
        self.emergency_subgraph = None
        self.search_subgraph = None
        self.answer_subgraph = None
        self.driving_subgraph = None
        
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
        
        # search_tools 모듈의 전역 변수 업데이트
        import src.tools.search_tools as search_tools
        search_tools.vector_store = vector_store
        search_tools.bm25_retriever = bm25_retriever
        search_tools.hybrid_retriever = None  # 하이브리드 검색기는 EnsembleRetriever로 동적 생성됨
        search_tools.multi_query_retriever = multi_query_retriever
        search_tools.cross_encoder_retriever = self.compression_manager.get_cross_encoder_retriever()
        search_tools.compression_retriever = self.compression_manager.get_compression_retriever()
        
        # 5. 검색 옵션 설정
        self._setup_search_options()
        
        # 6. SubGraph 인스턴스 초기화
        self._initialize_subgraphs()
        
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
            "expanded_query": EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[WEIGHT_CONFIGS["hybrid_semantic"], 1 - WEIGHT_CONFIGS["hybrid_semantic"]]
            )
        }
        
        # 재순위화 및 압축 옵션
        cross_encoder_ret = self.compression_manager.get_cross_encoder_retriever()
        compression_ret = self.compression_manager.get_compression_retriever()
        
        self.rerank_compression_options = {
            "rerank_only": cross_encoder_ret,
            "compress_only": compression_ret,
            "rerank_compress_general": cross_encoder_ret,
            "rerank_compress_specific": compression_ret,
            "rerank_compress_troubleshooting": cross_encoder_ret
        }
    
    def _initialize_subgraphs(self):
        """SubGraph 인스턴스들 초기화"""
        print("🔧 SubGraph 인스턴스 초기화 중...")
        
        # Emergency Detection SubGraph
        self.emergency_subgraph = EmergencyDetectionSubGraph()
        
        # Search Pipeline SubGraph
        self.search_subgraph = SearchPipelineSubGraph(
            self.search_options, 
            self.rerank_compression_options
        )
        
        # Answer Generation SubGraph
        self.answer_subgraph = AnswerGenerationSubGraph()
        
        # Driving Context SubGraph
        self.driving_subgraph = DrivingContextSubGraph()
        
        print("✅ SubGraph 인스턴스 초기화 완료!")
    
    def emergency_detection_wrapper(self, state: MainAgentState) -> Dict[str, Any]:
        """응급 상황 감지 래퍼 노드"""
        query = state["query"]
        
        print("🚨 응급 상황 감지 SubGraph 실행 중...")
        
        try:
            # Emergency Detection SubGraph 실행
            emergency_result = self.emergency_subgraph.invoke(query)
            
            print(f"✅ 응급 상황 감지 완료: {emergency_result['is_emergency']}")
            
            return {
                "is_emergency": emergency_result["is_emergency"],
                "emergency_level": emergency_result["emergency_level"],
                "emergency_score": emergency_result["emergency_score"],
                "emergency_analysis": emergency_result["emergency_analysis"],
                "search_strategy": emergency_result.get("search_strategy", ""),
                "search_method": emergency_result.get("search_method", ""),
                "compression_method": emergency_result.get("compression_method", "")
            }
            
        except Exception as e:
            print(f"❌ 응급 상황 감지 오류: {str(e)}")
            return {
                "is_emergency": False,
                "emergency_level": "NORMAL",
                "emergency_score": 0.0,
                "emergency_analysis": {},
                "search_strategy": "",
                "search_method": "",
                "compression_method": ""
            }
    
    def search_pipeline_wrapper(self, state: MainAgentState) -> Dict[str, Any]:
        """검색 파이프라인 래퍼 노드"""
        query = state["query"]
        is_emergency = state.get("is_emergency", False)
        
        print("🔍 검색 파이프라인 SubGraph 실행 중...")
        
        try:
            # 응급 상황 데이터 준비
            emergency_data = None
            if is_emergency:
                emergency_data = {
                    "search_strategy": state.get("search_strategy", "troubleshooting"),
                    "search_method": state.get("search_method", "hybrid_keyword"),
                    "compression_method": state.get("compression_method", "rerank_compress_troubleshooting")
                }
            
            # Search Pipeline SubGraph 실행
            search_result = self.search_subgraph.invoke(
                query, 
                is_emergency=is_emergency, 
                emergency_data=emergency_data
            )
            
            print(f"✅ 검색 파이프라인 완료: {len(search_result['search_results'])}개 문서")
            
            return {
                "search_strategy": search_result["search_strategy"],
                "search_method": search_result["search_method"],
                "compression_method": search_result["compression_method"],
                "confidence_score": search_result["confidence_score"],
                "search_results": search_result["search_results"],
                "page_references": search_result["page_references"]
            }
            
        except Exception as e:
            print(f"❌ 검색 파이프라인 오류: {str(e)}")
            return {
                "search_strategy": "general",
                "search_method": "hybrid_semantic",
                "compression_method": "rerank_compress_general",
                "confidence_score": 0.5,
                "search_results": [{"content": f"검색 중 오류 발생: {str(e)}", "page": 0, "score": 0.0}],
                "page_references": []
            }
    
    def answer_generation_wrapper(self, state: MainAgentState) -> Dict[str, Any]:
        """답변 생성 래퍼 노드"""
        query = state["query"]
        search_results = state.get("search_results", [])
        page_references = state.get("page_references", [])
        is_emergency = state.get("is_emergency", False)
        emergency_level = state.get("emergency_level", "NORMAL")
        
        print("📝 답변 생성 SubGraph 실행 중...")
        
        try:
            # Answer Generation SubGraph 실행
            answer_result = self.answer_subgraph.invoke(
                query=query,
                search_results=search_results,
                page_references=page_references,
                is_emergency=is_emergency,
                emergency_level=emergency_level
            )
            
            print(f"✅ 답변 생성 완료: {len(answer_result['final_answer'])}자")
            
            return {
                "final_answer": answer_result["final_answer"],
                "confidence_score": answer_result["confidence_score"],
                "evaluation_details": answer_result["evaluation_details"]
            }
            
        except Exception as e:
            print(f"❌ 답변 생성 오류: {str(e)}")
            return {
                "final_answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                "confidence_score": 0.0,
                "evaluation_details": None
            }
    
    def driving_context_wrapper(self, state: MainAgentState) -> Dict[str, Any]:
        """주행 상황 처리 래퍼 노드"""
        query = state["query"]
        original_answer = state.get("final_answer", "")
        is_emergency = state.get("is_emergency", False)
        emergency_level = state.get("emergency_level", "NORMAL")
        
        print("🚗 주행 상황 처리 SubGraph 실행 중...")
        
        try:
            # Driving Context SubGraph 실행
            driving_result = self.driving_subgraph.invoke(
                query=query,
                original_answer=original_answer,
                is_emergency=is_emergency,
                emergency_level=emergency_level
            )
            
            print(f"✅ 주행 상황 처리 완료: {driving_result['is_driving']}")
            
            return {
                "is_driving": driving_result["is_driving"],
                "driving_confidence": driving_result["driving_confidence"],
                "driving_indicators": driving_result["driving_indicators"],
                "driving_urgency": driving_result["driving_urgency"],
                "compression_needed": driving_result["compression_needed"],
                "compressed_answer": driving_result["compressed_answer"],
                "final_answer": driving_result["final_answer"]
            }
            
        except Exception as e:
            print(f"❌ 주행 상황 처리 오류: {str(e)}")
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
        """LangGraph 워크플로우 생성 - SubGraph 아키텍처"""
        workflow = StateGraph(MainAgentState)
        
        # 노드 추가 (SubGraph 래퍼들)
        workflow.add_node("emergency_detection", self.emergency_detection_wrapper)
        workflow.add_node("search_pipeline", self.search_pipeline_wrapper)
        workflow.add_node("answer_generation", self.answer_generation_wrapper)
        workflow.add_node("driving_context", self.driving_context_wrapper)
        
        # 엣지 추가 (순차적 실행)
        workflow.set_entry_point("emergency_detection")
        workflow.add_edge("emergency_detection", "search_pipeline")
        workflow.add_edge("search_pipeline", "answer_generation")
        workflow.add_edge("answer_generation", "driving_context")
        workflow.add_edge("driving_context", END)
        
        return workflow.compile()
    
    def query(self, user_query: str, callbacks=None) -> str:
        """사용자 쿼리 처리 - SubGraph 아키텍처"""
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
                "need_clarification": False,
                # 응급 상황 관련 초기값
                "is_emergency": False,
                "emergency_level": "NORMAL",
                "emergency_score": 0.0,
                "emergency_analysis": {},
                "compression_method": "",
                # 주행 상황 관련 초기값
                "is_driving": False,
                "driving_confidence": 0.0,
                "driving_indicators": [],
                "driving_urgency": "normal",
                "compression_needed": False,
                "compressed_answer": "",
                # 평가 관련
                "evaluation_details": None
            }
            
            # 콜백이 있으면 설정에 포함
            config = {}
            if callbacks:
                config["callbacks"] = callbacks
            
            # 그래프 실행
            result = graph.invoke(initial_state, config=config)
            
            return result.get("final_answer", "답변을 생성할 수 없습니다.")
            
        except Exception as e:
            return f"쿼리 처리 중 오류가 발생했습니다: {str(e)}"
