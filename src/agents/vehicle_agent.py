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
from ..utils.answer_evaluator import AnswerEvaluator
from ..utils.emergency_detector import EmergencyDetector
from ..utils.driving_context_detector import DrivingContextDetector
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
        
        # 답변 평가기 초기화
        self.answer_evaluator = AnswerEvaluator()
        
        # 응급 상황 감지기 초기화
        self.emergency_detector = EmergencyDetector()
        
        # 주행 상황 감지기 초기화
        self.driving_detector = DrivingContextDetector()
        
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
    
    def emergency_classifier(self, state: AgentState) -> Dict[str, Any]:
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
    
    def query_analyzer(self, state: AgentState) -> Dict[str, Any]:
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
    
    def search_executor(self, state: AgentState) -> Dict[str, Any]:
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
                from ..tools.search_tools import expanded_query_search
                search_results = expanded_query_search.invoke({"query": query, "top_k": DEFAULT_TOP_K})
            elif search_method == "multi_query":
                print("🔎 다중 쿼리 검색 실행 중...")
                from ..tools.search_tools import multi_query_search
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
    
    def answer_generator(self, state: AgentState) -> Dict[str, Any]:
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
            
            return {
                "final_answer": final_answer_with_confidence,
                "confidence_score": confidence_percentage / 100,
                "evaluation_details": evaluation if not is_emergency else None
            }
            
        except Exception as e:
            print(f"답변 생성 오류: {str(e)}")
            return {"final_answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}"}
    
    def driving_context_processor(self, state: AgentState) -> Dict[str, Any]:
        """주행 중 상황 감지 및 답변 압축 노드"""
        query = state["query"]
        original_answer = state.get("final_answer", "")
        
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
        
        # 노드 추가 (응급 분류 노드 추가)
        workflow.add_node("emergency_classifier", self.emergency_classifier)
        workflow.add_node("query_analyzer", self.query_analyzer)
        workflow.add_node("search_executor", self.search_executor)
        workflow.add_node("answer_generator", self.answer_generator)
        workflow.add_node("driving_context_processor", self.driving_context_processor)
        
        # 엣지 추가 (응급 분류 -> 쿼리 분석 -> 검색 -> 답변 -> 주행 상황 처리)
        workflow.set_entry_point("emergency_classifier")
        workflow.add_edge("emergency_classifier", "query_analyzer")
        workflow.add_edge("query_analyzer", "search_executor")
        workflow.add_edge("search_executor", "answer_generator")
        workflow.add_edge("answer_generator", "driving_context_processor")
        workflow.add_edge("driving_context_processor", END)
        
        return workflow.compile()
    
    def query(self, user_query: str, callbacks=None) -> str:
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
                "compressed_answer": ""
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
