"""
차량 매뉴얼 RAG 에이전트 - SubGraph 아키텍처
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langgraph.graph import StateGraph, END

from ..models.states import MainAgentState
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
    DrivingContextSubGraph,
    SpeechRecognitionSubGraph
)


class VehicleManualAgent:
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
        self.speech_subgraph = None
        
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
        
        # Speech Recognition SubGraph
        self.speech_subgraph = SpeechRecognitionSubGraph()
        
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
    
    def speech_recognition_wrapper(self, state: MainAgentState) -> Dict[str, Any]:
        """음성 인식 래퍼 노드"""
        audio_data = state.get("audio_data")
        audio_file_path = state.get("audio_file_path")
        existing_query = state.get("query", "")
        
        # 이미 텍스트 쿼리가 있으면 음성 인식 건너뛰기
        if existing_query and existing_query.strip():
            print("📝 텍스트 쿼리 감지 - 음성 인식 건너뛰기")
            return {
                "recognized_text": "",
                "speech_confidence": 0.0,
                "speech_error": None,
                "query": existing_query  # 기존 쿼리 유지
            }
        
        print("🎤 음성 인식 SubGraph 실행 중...")
        
        try:
            # Speech Recognition SubGraph 실행
            speech_result = self.speech_subgraph.invoke(
                audio_data=audio_data,
                audio_file_path=audio_file_path
            )
            
            print(f"✅ 음성 인식 완료: '{speech_result['final_text']}'")
            
            return {
                "recognized_text": speech_result["final_text"],
                "speech_confidence": speech_result["confidence"],
                "speech_error": speech_result["error"],
                "query": speech_result["final_text"]  # 인식된 텍스트를 쿼리로 설정
            }
            
        except Exception as e:
            print(f"❌ 음성 인식 오류: {str(e)}")
            return {
                "recognized_text": "",
                "speech_confidence": 0.0,
                "speech_error": f"음성 인식 중 오류가 발생했습니다: {str(e)}",
                "query": ""
            }
    
    def create_graph(self) -> StateGraph:
        """LangGraph 워크플로우 생성 - SubGraph 아키텍처"""
        workflow = StateGraph(MainAgentState)
        
        # 노드 추가 (SubGraph 래퍼들)
        workflow.add_node("speech_recognition", self.speech_recognition_wrapper)
        workflow.add_node("emergency_detection", self.emergency_detection_wrapper)
        workflow.add_node("search_pipeline", self.search_pipeline_wrapper)
        workflow.add_node("answer_generation", self.answer_generation_wrapper)
        workflow.add_node("driving_context", self.driving_context_wrapper)
        
        # 엣지 추가 (음성 인식 → 기존 워크플로우)
        workflow.set_entry_point("speech_recognition")
        workflow.add_edge("speech_recognition", "emergency_detection")
        workflow.add_edge("emergency_detection", "search_pipeline")
        workflow.add_edge("search_pipeline", "answer_generation")
        workflow.add_edge("answer_generation", "driving_context")
        workflow.add_edge("driving_context", END)
        
        return workflow.compile()
    
    def create_emergency_fast_path(self) -> StateGraph:
        """응급 상황 전용 빠른 경로 - 최소한의 처리로 빠른 응답"""
        workflow = StateGraph(MainAgentState)
        
        # 응급 상황에서는 최소한의 노드만 실행
        workflow.add_node("speech_recognition", self.speech_recognition_wrapper)
        workflow.add_node("emergency_detection", self.emergency_detection_wrapper)
        workflow.add_node("emergency_search", self.emergency_search_wrapper)
        workflow.add_node("emergency_answer", self.emergency_answer_wrapper)
        
        # 빠른 경로: 음성인식 → 응급감지 → 간소검색 → 간소답변
        workflow.set_entry_point("speech_recognition")
        workflow.add_edge("speech_recognition", "emergency_detection")
        workflow.add_edge("emergency_detection", "emergency_search")
        workflow.add_edge("emergency_search", "emergency_answer")
        workflow.add_edge("emergency_answer", END)
        
        return workflow.compile()
    
    def emergency_search_wrapper(self, state: MainAgentState) -> Dict[str, Any]:
        """응급 상황 전용 간소화된 검색 - 속도 우선"""
        query = state["query"]
        
        print("🚨 응급 상황 검색 - 빠른 키워드 검색 실행")
        
        try:
            # 응급 상황에서는 가장 빠른 BM25 키워드 검색만 사용
            retriever = self.search_options["bm25_only"]
            docs = retriever.invoke(query)
            
            # 최대 3개 문서만 사용 (속도 우선)
            search_results = [
                {
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", 0),
                    "source": doc.metadata.get("source", ""),
                    "score": 1.0  # 응급 상황에서는 점수 계산 생략
                }
                for doc in docs[:3]  # 3개로 제한
            ]
            
            # 페이지 참조 추출
            page_references = list(set([
                result.get("page", 0) for result in search_results if result.get("page", 0) > 0
            ]))
            
            print(f"⚡ 응급 검색 완료: {len(search_results)}개 문서, {len(page_references)}개 페이지")
            
            return {
                "search_strategy": "emergency_fast",
                "search_method": "bm25_only",
                "compression_method": "none",  # 압축 생략
                "confidence_score": 0.9,  # 고정 신뢰도
                "search_results": search_results,
                "page_references": page_references
            }
            
        except Exception as e:
            print(f"❌ 응급 검색 오류: {str(e)}")
            return {
                "search_strategy": "emergency_fast",
                "search_method": "bm25_only",
                "compression_method": "none",
                "confidence_score": 0.5,
                "search_results": [{"content": f"응급 검색 중 오류: {str(e)}", "page": 0, "score": 0.0}],
                "page_references": []
            }
    
    def emergency_answer_wrapper(self, state: MainAgentState) -> Dict[str, Any]:
        """응급 상황 전용 간소화된 답변 생성 - 속도 우선"""
        query = state["query"]
        search_results = state.get("search_results", [])
        page_references = state.get("page_references", [])
        emergency_level = state.get("emergency_level", "HIGH")
        
        print(f"🚨 응급 답변 생성 - {emergency_level} 수준")
        
        try:
            # 간소화된 컨텍스트 구성 (최대 2개 문서, 각 200자만)
            context_parts = []
            for i, result in enumerate(search_results[:2], 1):
                content = result.get("content", "")
                page = result.get("page", 0)
                
                # 200자로 제한
                if len(content) > 200:
                    content = content[:200] + "..."
                
                if page > 0:
                    context_parts.append(f"[참고 {i}] (페이지 {page})\n{content}")
                else:
                    context_parts.append(f"[참고 {i}]\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # 간소화된 응급 프롬프트 (ChatPromptTemplate 사용)
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            emergency_prompt = ChatPromptTemplate.from_messages([
                ("system", """응급 상황입니다. 다음 정보를 바탕으로 즉시 실행 가능한 안전 조치만 간단히 제시하세요.
        
답변 형식:
🚨 즉시 조치: [핵심 행동 1-2개]
⚠️ 안전 경고: [중요한 주의사항]
📞 연락처: [필요시 응급 서비스]

응급 수준별 대응:
- CRITICAL: 생명 위험, 즉시 119 신고
- HIGH: 즉시 안전 조치 필요
- MEDIUM: 신속한 대응 필요
- LOW: 주의 필요"""),
                ("human", """질문: {query}
참고 정보: {context}
응급 수준: {emergency_level}

답변:""")
            ])
            
            # LLM 호출 (간소화된 프롬프트)
            answer_chain = emergency_prompt | self.llm | StrOutputParser()
            final_answer = answer_chain.invoke({
                "query": query, 
                "context": context,
                "emergency_level": emergency_level
            })
            
            # 응급 상황 헤더 추가
            emergency_icons = {
                "CRITICAL": "🔥",
                "HIGH": "🚨", 
                "MEDIUM": "⚠️",
                "LOW": "🔍"
            }
            icon = emergency_icons.get(emergency_level, "🚨")
            emergency_header = f"{icon} **{emergency_level} 응급 상황**\n\n"
            
            # 응급 상황 경고 추가
            if emergency_level == "CRITICAL":
                emergency_warning = "\n\n🚨 **생명 위험 상황입니다. 즉시 119에 신고하세요.**"
            elif emergency_level == "HIGH":
                emergency_warning = "\n\n⚠️ **즉시 안전 조치가 필요합니다. 전문가에게 연락하세요.**"
            else:
                emergency_warning = "\n\n⚠️ **신속한 대응이 필요합니다.**"
            
            final_answer_with_header = emergency_header + final_answer + emergency_warning
            
            print(f"✅ 응급 답변 생성 완료: {len(final_answer_with_header)}자")
            
            return {
                "final_answer": final_answer_with_header,
                "confidence_score": 0.85,  # 고정 신뢰도 (평가 생략)
                "evaluation_details": None  # 평가 생략
            }
            
        except Exception as e:
            print(f"❌ 응급 답변 생성 오류: {str(e)}")
            return {
                "final_answer": f"🚨 응급 상황 답변 생성 중 오류가 발생했습니다: {str(e)}\n\n⚠️ 즉시 안전한 곳에 정차하고 전문가에게 연락하세요.",
                "confidence_score": 0.5,
                "evaluation_details": None
            }
    
    def query(self, user_query: str = None, audio_data: bytes = None, 
              audio_file_path: str = None, callbacks=None) -> str:
        """사용자 쿼리 처리 - 응급 상황 감지 후 적절한 워크플로우 선택"""
        try:
            # 1. 먼저 빠른 응급 상황 감지 (텍스트 쿼리가 있는 경우만)
            if user_query and user_query.strip():
                # LLM 기반 응급 상황 감지 사용
                from ..utils.llm_emergency_detector import LLMEmergencyDetector
                llm_emergency_detector = LLMEmergencyDetector()
                emergency_result = llm_emergency_detector.detect_emergency(user_query)
                
                # CRITICAL 또는 HIGH 응급 상황이면 빠른 경로 사용
                if emergency_result["is_emergency"] and emergency_result["priority_level"] in ["CRITICAL", "HIGH"]:
                    print(f"🚨 응급 상황 감지 ({emergency_result['priority_level']}) - 빠른 경로 실행")
                    graph = self.create_emergency_fast_path()
                    use_emergency_path = True
                else:
                    print("📝 일반 질문 - 전체 워크플로우 실행")
                    graph = self.create_graph()
                    use_emergency_path = False
            else:
                # 음성 인식만 있는 경우는 전체 워크플로우 사용
                print("🎤 음성 입력 - 전체 워크플로우 실행")
                graph = self.create_graph()
                use_emergency_path = False
            
            # 초기 상태 설정
            initial_state = {
                "messages": [],
                "query": user_query or "",  # 텍스트 쿼리 (음성 인식 시 덮어씌워짐)
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
                # 음성 인식 관련 초기값
                "audio_data": audio_data,
                "audio_file_path": audio_file_path,
                "recognized_text": "",
                "speech_confidence": 0.0,
                "speech_error": None,
                # 평가 관련
                "evaluation_details": None
            }
            
            # 응급 경로 사용 시 응급 상황 정보 미리 설정
            if use_emergency_path and user_query:
                initial_state.update({
                    "is_emergency": True,
                    "emergency_level": emergency_result["priority_level"],
                    "emergency_score": emergency_result["total_score"],
                    "emergency_analysis": emergency_result
                })
            
            # 콜백이 있으면 설정에 포함
            config = {}
            if callbacks:
                config["callbacks"] = callbacks
            
            # 그래프 실행
            result = graph.invoke(initial_state, config=config)
            
            return result.get("final_answer", "답변을 생성할 수 없습니다.")
            
        except Exception as e:
            return f"쿼리 처리 중 오류가 발생했습니다: {str(e)}"
