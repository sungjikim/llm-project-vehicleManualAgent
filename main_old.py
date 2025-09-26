from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import (
    CrossEncoderReranker,
    LLMChainExtractor,
    DocumentCompressorPipeline,
    EmbeddingsFilter
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from textwrap import dedent
import operator
import os
from pathlib import Path
from dotenv import load_dotenv
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False
    print("⚠️ Kiwi 토크나이저가 설치되지 않았습니다. 기본 토크나이저를 사용합니다.")

# 환경 변수 로드
load_dotenv()

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    search_results: List[Document]
    context: str
    final_answer: str
    search_strategy: str  # "general", "specific", "troubleshooting"
    search_method: str   # "hybrid_semantic", "hybrid_balanced", "hybrid_keyword"
    confidence_score: float
    page_references: List[int]
    need_clarification: bool

# 전역 변수로 검색기들 관리
vector_store = None
bm25_retriever = None
hybrid_retriever = None
kiwi_model = None
multi_query_retriever = None
cross_encoder_retriever = None
compression_retriever = None

# 도구들 정의
@tool
def vector_search(query: str, top_k: int = 5) -> List[Dict]:
    """벡터 유사도 검색으로 관련 문서 찾기"""
    global vector_store
    if vector_store is None:
        return [{"content": "벡터 저장소가 초기화되지 않았습니다.", "page": 0, "score": 0.0}]
    
    try:
        # 벡터 검색 수행
        results = vector_store.similarity_search_with_score(query, k=top_k)
        
        search_results = []
        for doc, score in results:
            search_results.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", ""),
                "score": float(score)
            })
        
        return search_results
    except Exception as e:
        return [{"content": f"검색 중 오류 발생: {str(e)}", "page": 0, "score": 0.0}]

@tool
def bm25_search(query: str, top_k: int = 5) -> List[Dict]:
    """키워드 기반 BM25 검색"""
    global bm25_retriever
    if bm25_retriever is None:
        return [{"content": "BM25 검색기가 초기화되지 않았습니다.", "page": 0, "score": 0.0}]
    
    try:
        # BM25 검색 수행
        results = bm25_retriever.invoke(query)
        
        search_results = []
        for doc in results[:top_k]:
            search_results.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", ""),
                "score": 1.0  # BM25는 점수를 직접 반환하지 않음
            })
        
        return search_results
    except Exception as e:
        return [{"content": f"BM25 검색 중 오류 발생: {str(e)}", "page": 0, "score": 0.0}]

@tool
def hybrid_search(query: str, top_k: int = 5, semantic_weight: float = 0.5) -> List[Dict]:
    """하이브리드 검색 (벡터 + BM25)"""
    global hybrid_retriever, vector_store, bm25_retriever
    
    # 하이브리드 검색기가 없거나 가중치가 다른 경우 재생성
    if (hybrid_retriever is None or 
        getattr(hybrid_retriever, '_weights', [0.5, 0.5])[0] != semantic_weight):
        
        if vector_store is None or bm25_retriever is None:
            return [{"content": "벡터 저장소 또는 BM25 검색기가 초기화되지 않았습니다.", "page": 0, "score": 0.0}]
        
        try:
            # 하이브리드 검색기 생성
            semantic_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
            keyword_weight = 1.0 - semantic_weight
            
            hybrid_retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[semantic_weight, keyword_weight]
            )
            hybrid_retriever._weights = [semantic_weight, keyword_weight]  # 가중치 저장
            
        except Exception as e:
            return [{"content": f"하이브리드 검색기 생성 오류: {str(e)}", "page": 0, "score": 0.0}]
    
    try:
        # 하이브리드 검색 수행
        results = hybrid_retriever.invoke(query)
        
        search_results = []
        for doc in results[:top_k]:
            search_results.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", ""),
                "score": 1.0  # 하이브리드는 점수를 직접 반환하지 않음
            })
        
        return search_results
    except Exception as e:
        return [{"content": f"하이브리드 검색 중 오류 발생: {str(e)}", "page": 0, "score": 0.0}]

@tool
def multi_query_search(query: str, top_k: int = 5) -> List[Dict]:
    """다중 쿼리 생성 후 검색"""
    global multi_query_retriever
    
    if multi_query_retriever is None:
        return [{"content": "다중 쿼리 검색기가 초기화되지 않았습니다.", "page": 0, "score": 0.0}]
    
    try:
        # 다중 쿼리 검색 수행
        results = multi_query_retriever.invoke(query)
        
        search_results = []
        for doc in results[:top_k]:
            search_results.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", ""),
                "score": 1.0
            })
        
        return search_results
    except Exception as e:
        return [{"content": f"다중 쿼리 검색 중 오류 발생: {str(e)}", "page": 0, "score": 0.0}]

@tool
def expanded_query_search(query: str, top_k: int = 5) -> List[Dict]:
    """차량 전문 용어로 쿼리 확장 후 검색"""
    global hybrid_retriever
    
    if hybrid_retriever is None:
        return [{"content": "하이브리드 검색기가 초기화되지 않았습니다.", "page": 0, "score": 0.0}]
    
    try:
        # 차량 전문 용어 매핑
        vehicle_terms = {
            '타이어': ['타이어', '타이어 압력', '공기압', '압력', 'PSI', 'bar'],
            '브레이크': ['브레이크', '제동', '제동장치', '브레이크 패드', '제동 패드'],
            '엔진': ['엔진', '모터', '동력장치', '엔진오일', '오일'],
            '경고등': ['경고등', '표시등', '인디케이터', '알림등', '경고 신호'],
            '문제': ['문제', '고장', '오류', '이상', '결함', '오작동'],
            '교체': ['교체', '교체방법', '바꿀기', '수리', '정비', '유지보수']
        }
        
        # 쿼리 확장
        expanded_terms = []
        query_lower = query.lower()
        
        for key, synonyms in vehicle_terms.items():
            if key in query_lower:
                expanded_terms.extend(synonyms)
        
        # 기본 쿼리에 확장 용어 추가
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms)}"
        else:
            expanded_query = query
        
        # 확장된 쿼리로 검색
        results = hybrid_retriever.invoke(expanded_query)
        
        search_results = []
        for doc in results[:top_k]:
            search_results.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", ""),
                "score": 1.0
            })
        
        return search_results
    except Exception as e:
        return [{"content": f"확장 쿼리 검색 중 오류 발생: {str(e)}", "page": 0, "score": 0.0}]

@tool
def cross_encoder_rerank_search(query: str, top_k: int = 5) -> List[Dict]:
    """Cross-Encoder 모델을 사용한 재순위화 검색"""
    global cross_encoder_retriever
    
    if cross_encoder_retriever is None:
        return [{"content": "Cross-Encoder 재순위화 검색기가 초기화되지 않았습니다.", "page": 0, "score": 0.0}]
    
    try:
        # Cross-Encoder 재순위화 검색 수행
        results = cross_encoder_retriever.invoke(query)
        
        search_results = []
        for doc in results[:top_k]:
            search_results.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", ""),
                "score": getattr(doc, 'score', 1.0)  # Cross-Encoder 점수
            })
        
        return search_results
    except Exception as e:
        return [{"content": f"Cross-Encoder 재순위화 검색 중 오류 발생: {str(e)}", "page": 0, "score": 0.0}]

@tool
def contextual_compression_search(query: str, top_k: int = 5) -> List[Dict]:
    """맥락 압축을 사용한 검색 (관련 정보만 추출)"""
    global compression_retriever
    
    if compression_retriever is None:
        return [{"content": "맥락 압축 검색기가 초기화되지 않았습니다.", "page": 0, "score": 0.0}]
    
    try:
        # 맥락 압축 검색 수행
        results = compression_retriever.invoke(query)
        
        search_results = []
        for doc in results[:top_k]:
            search_results.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", ""),
                "score": 1.0  # 압축된 문서는 모두 관련성 높음
            })
        
        return search_results
    except Exception as e:
        return [{"content": f"맥락 압축 검색 중 오류 발생: {str(e)}", "page": 0, "score": 0.0}]

@tool  
def keyword_search(query: str, keywords: List[str] = None) -> List[Dict]:
    """키워드 기반 정확한 검색"""
    global vector_store
    if vector_store is None:
        return [{"content": "벡터 저장소가 초기화되지 않았습니다.", "page": 0}]
    
    try:
        # 키워드가 제공되지 않으면 쿼리에서 추출
        if keywords is None:
            keywords = query.split()
        
        # 키워드를 포함한 확장 쿼리로 검색
        expanded_query = f"{query} {' '.join(keywords)}"
        results = vector_store.similarity_search(expanded_query, k=3)
        
        search_results = []
        for doc in results:
            search_results.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", "")
            })
        
        return search_results
    except Exception as e:
        return [{"content": f"키워드 검색 중 오류 발생: {str(e)}", "page": 0}]

@tool
def page_context_search(page_numbers: List[int]) -> List[Dict]:
    """특정 페이지 주변 컨텍스트 수집"""
    global vector_store
    if vector_store is None:
        return [{"content": "벡터 저장소가 초기화되지 않았습니다.", "page": 0}]
    
    try:
        results = []
        for page_num in page_numbers:
            # 특정 페이지와 인접 페이지 검색
            for offset in [-1, 0, 1]:  # 이전, 현재, 다음 페이지
                target_page = page_num + offset
                if target_page > 0:
                    # 페이지 번호로 필터링하여 검색
                    page_docs = vector_store.get(where={"page": target_page})
                    if page_docs and 'documents' in page_docs:
                        for i, content in enumerate(page_docs['documents']):
                            metadata = page_docs.get('metadatas', [{}])[i] if page_docs.get('metadatas') else {}
                            results.append({
                                "content": content,
                                "page": target_page,
                                "source": metadata.get("source", "")
                            })
        
        return results if results else [{"content": "해당 페이지를 찾을 수 없습니다.", "page": 0}]
    except Exception as e:
        return [{"content": f"페이지 검색 중 오류 발생: {str(e)}", "page": 0}]

# 노드들 정의
class VehicleManualAgent:
    def __init__(self, pdf_path: str = None):
        self.llm = ChatOpenAI(temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.pdf_path = pdf_path or "/Users/1112931/llm-3/project/data/backup/kr_ko-KR_xc60_2026.pdf"
        self.vector_db = None
        self.tools = [vector_search, keyword_search, page_context_search, bm25_search, hybrid_search, multi_query_search, expanded_query_search, cross_encoder_rerank_search, contextual_compression_search]
        
        # Few-shot 예시 데이터 초기화
        self._initialize_few_shot_examples()
        
        # 검색 전략 옵션
        self.search_options = {
            "vector_only": "벡터 검색만 사용",
            "bm25_only": "BM25 키워드 검색만 사용", 
            "hybrid_balanced": "하이브리드 (5:5)",
            "hybrid_semantic": "하이브리드 (7:3 - 의미론적 우선)",
            "hybrid_keyword": "하이브리드 (3:7 - 키워드 우선)",
            "multi_query": "다중 쿼리 생성 검색",
            "expanded_query": "전문 용어 확장 검색",
            "cross_encoder_rerank": "Cross-Encoder 재순위화 검색",
            "contextual_compression": "맥락 압축 검색"
        }
        
        # PDF 로드 및 벡터 저장소 초기화
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """벡터 저장소 초기화 및 PDF 문서 로드"""
        global vector_store
        
        try:
            # PDF 파일 존재 확인
            if not os.path.exists(self.pdf_path):
                print(f"경고: PDF 파일을 찾을 수 없습니다: {self.pdf_path}")
                return
            
            print(f"PDF 파일 로딩 중: {self.pdf_path}")
            
            # PDF 로더로 문서 로드
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            print(f"로드된 페이지 수: {len(documents)}")
            
            # 텍스트 분할 (토큰 제한을 고려하여 더 작은 청크로 분할)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,  # 더더 작은 청크 크기
                chunk_overlap=30,  # 격청 간격 축소
                length_function=len,
            )
            
            split_docs = text_splitter.split_documents(documents)
            print(f"분할된 문서 조각 수: {len(split_docs)}")
            
            # 벡터 저장소 생성
            persist_directory = "/Users/1112931/llm-3/project/chroma_db"
            
            # 기존 데이터베이스가 있는지 확인
            if os.path.exists(persist_directory) and os.listdir(persist_directory):
                print("기존 벡터 데이터베이스 로드 중...")
                vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                print("새로운 벡터 데이터베이스 생성 중...")
                
                # 배치 처리로 문서 추가 (토큰 제한 회피)
                batch_size = 50  # 배치 크기 설정
                vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                
                print(f"총 {len(split_docs)}개 문서를 {batch_size}개씩 배치 처리...")
                
                for i in range(0, len(split_docs), batch_size):
                    batch = split_docs[i:i+batch_size]
                    try:
                        vector_store.add_documents(batch)
                        print(f"배치 {i//batch_size + 1}/{(len(split_docs)-1)//batch_size + 1} 완료")
                    except Exception as batch_error:
                        print(f"배치 {i//batch_size + 1} 오류: {str(batch_error)}")
                        continue
                
                # BM25 검색기 초기화
                self._initialize_bm25_retriever(split_docs)
            
            self.vector_db = vector_store
            print("벡터 저장소 초기화 완료!")
            
            # MultiQueryRetriever 초기화
            self._initialize_multi_query_retriever()
            
            # Cross-Encoder 재순위화 초기화
            self._initialize_cross_encoder_retriever()
            
            # 맥락 압축 초기화
            self._initialize_contextual_compression()
            
            # BM25 검색기 초기화 (기존 데이터베이스 로드시)
            if os.path.exists(persist_directory) and os.listdir(persist_directory):
                # 저장된 문서들을 다시 로드하여 BM25 검색기 생성
                try:
                    all_docs = vector_store.get()
                    if all_docs and 'documents' in all_docs and all_docs['documents']:
                        # 문서를 Document 객체로 변환
                        documents_for_bm25 = []
                        for i, content in enumerate(all_docs['documents']):
                            metadata = all_docs.get('metadatas', [{}])[i] if all_docs.get('metadatas') else {}
                            documents_for_bm25.append(Document(page_content=content, metadata=metadata))
                        
                        self._initialize_bm25_retriever(documents_for_bm25)
                except Exception as e:
                    print(f"BM25 검색기 초기화 중 오류 (기존 DB): {str(e)}")
            
        except Exception as e:
            print(f"벡터 저장소 초기화 오류: {str(e)}")
            vector_store = None
    
    def _setup_korean_tokenizer(self):
        """한국어 토크나이저 설정"""
        global kiwi_model
        
        if not KIWI_AVAILABLE:
            return None
            
        try:
            kiwi_model = Kiwi()
            # 사용자 정의 단어 추가
            custom_words = [
                ('볼보', 'NNP'),
                ('XC60', 'NNP'), 
                ('타이어', 'NNG'),
                ('브레이크', 'NNG'),
                ('엔진', 'NNG'),
                ('경고등', 'NNG')
            ]
            
            for word, pos in custom_words:
                kiwi_model.add_user_word(word, pos)
                
            print(f"✅ 한국어 토크나이저 설정 완료: {len(custom_words)}개 사용자 단어 추가")
            return kiwi_model
            
        except Exception as e:
            print(f"한국어 토크나이저 설정 오류: {str(e)}")
            return None
    
    def _korean_tokenizer(self, text: str) -> List[str]:
        """한국어 텍스트 토크나이징"""
        if kiwi_model is None:
            # Kiwi가 없으면 기본 공백 분리
            return text.split()
        
        try:
            # Kiwi로 토크나이징
            tokens = []
            for token, pos, _, _ in kiwi_model.analyze(text):
                # 명사, 동사, 형용사, 고유명사만 추출
                if pos.startswith(('NN', 'VV', 'VA', 'NP')):
                    tokens.append(token)
            return tokens if tokens else text.split()
        except:
            return text.split()
    
    def _initialize_bm25_retriever(self, documents: List[Document]):
        """BM25 검색기 초기화"""
        global bm25_retriever
        
        try:
            # 한국어 토크나이저 설정
            self._setup_korean_tokenizer()
            
            # BM25 검색기 생성
            if KIWI_AVAILABLE and kiwi_model is not None:
                # 한국어 토크나이저 사용
                bm25_retriever = BM25Retriever.from_documents(
                    documents=documents,
                    preprocess_func=self._korean_tokenizer,
                    k=5
                )
                print("✅ BM25 검색기 생성 완료 (한국어 토크나이저 적용)")
            else:
                # 기본 토크나이저 사용
                bm25_retriever = BM25Retriever.from_documents(
                    documents=documents,
                    k=5
                )
                print("✅ BM25 검색기 생성 완료 (기본 토크나이저)")
                
        except Exception as e:
            print(f"BM25 검색기 초기화 오류: {str(e)}")
            bm25_retriever = None
    
    def _initialize_multi_query_retriever(self):
        """MultiQueryRetriever 초기화"""
        global multi_query_retriever, vector_store
        
        if vector_store is None:
            print("⚠️ 벡터 저장소가 없어 MultiQueryRetriever를 초기화할 수 없습니다.")
            return
        
        try:
            # 차량 매뉴얼 전용 다중 쿼리 생성 프롬프트
            vehicle_prompt = ChatPromptTemplate.from_template(
                """차량 매뉴얼 검색을 위해 주어진 질문을 3개의 다른 관점에서 다시 작성해주세요.
                
                원본 질문: {question}
                
                다음 관점들을 고려해주세요:
                1. 기술적/전문적 관점
                2. 사용자/실용적 관점  
                3. 문제해결/안전 관점
                
                각 질문은 한 줄로 작성하고, 숫자나 부호 없이 작성해주세요.
                """
            )
            
            # 커스텀 출력 파서
            class LineListOutputParser(BaseOutputParser[List[str]]):
                def parse(self, text: str) -> List[str]:
                    lines = text.strip().split('\n')
                    return [line.strip() for line in lines if line.strip()]
            
            # 다중 쿼리 체인 생성
            multi_query_chain = vehicle_prompt | self.llm | LineListOutputParser()
            
            # MultiQueryRetriever 생성
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            multi_query_retriever = MultiQueryRetriever(
                retriever=base_retriever,
                llm_chain=multi_query_chain,
                parser_key="lines"
            )
            
            print("✅ MultiQueryRetriever 초기화 완료")
            
        except Exception as e:
            print(f"MultiQueryRetriever 초기화 오류: {str(e)}")
            multi_query_retriever = None
    
    def _initialize_cross_encoder_retriever(self):
        """Cross-Encoder 모델을 사용한 재순위화 검색기 초기화"""
        global cross_encoder_retriever, vector_store
        
        if vector_store is None:
            print("⚠️ 벡터 저장소가 없어 Cross-Encoder 재순위화기를 초기화할 수 없습니다.")
            return
        
        try:
            print("🔄 Cross-Encoder 재순위화 시스템 초기화 중...")
            
            # Cross-Encoder 모델 로드 (경량화된 모델 사용)
            cross_encoder_model = HuggingFaceCrossEncoder(
                model_name="BAAI/bge-reranker-v2-m3"  # 다국어 지원 모델
            )
            
            # 재순위화 컴프레서 생성
            reranker = CrossEncoderReranker(
                model=cross_encoder_model,
                top_n=5  # 상위 5개 문서만 반환
            )
            
            # 베이스 리트리버 생성 (더 많은 후보 문서 검색)
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})  # 20개 후보 검색
            
            # 컨텍스츄얼 컴프레션 리트리버 생성
            cross_encoder_retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=base_retriever
            )
            
            print("✅ Cross-Encoder 재순위화 시스템 초기화 완료")
            print(f"   모델: BAAI/bge-reranker-v2-m3")
            print(f"   후보 문서: 20개 -> 상위 5개 선별")
            
        except Exception as e:
            print(f"Cross-Encoder 재순위화 초기화 오류: {str(e)}")
            cross_encoder_retriever = None
    
    def _initialize_contextual_compression(self):
        """맥락 압축 검색기 초기화"""
        global compression_retriever, vector_store
        
        if vector_store is None:
            print("⚠️ 벡터 저장소가 없어 맥락 압축기를 초기화할 수 없습니다.")
            return
        
        try:
            print("📝 맥락 압축 시스템 초기화 중...")
            
            # 1. 임베딩 기반 필터링 (유사도 임계값 설정)
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.6  # 60% 이상 유사한 문서만 통과
            )
            
            # 2. 중복 제거 필터
            redundant_filter = EmbeddingsRedundantFilter(
                embeddings=self.embeddings,
                similarity_threshold=0.9  # 90% 이상 유사한 문서 제거
            )
            
            # 3. LLM 기반 정보 추출기
            llm_extractor = LLMChainExtractor.from_llm(llm=self.llm)
            
            # 다단계 압축 파이프라인 구성
            pipeline_compressor = DocumentCompressorPipeline(
                transformers=[
                    embeddings_filter,      # 1단계: 유사도 필터링
                    redundant_filter,       # 2단계: 중복 제거
                    llm_extractor          # 3단계: 핵심 정보 추출
                ]
            )
            
            # 베이스 리트리버 생성
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})  # 15개 후보 검색
            
            # 컨텍스츄얼 컴프레션 리트리버 생성
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=pipeline_compressor,
                base_retriever=base_retriever
            )
            
            print("✅ 맥락 압축 시스템 초기화 완료")
            print(f"   1단계: 임베딩 필터링 (유사도 > 60%)")
            print(f"   2단계: 중복 제거 (유사도 > 90%)")
            print(f"   3단계: LLM 기반 핵심 정보 추출")
            
        except Exception as e:
            print(f"맥락 압축 초기화 오류: {str(e)}")
            compression_retriever = None
    
    def _initialize_few_shot_examples(self):
        """차량 매뉴얼 관련 Few-shot 예시 데이터 초기화"""
        
        # 차량 매뉴얼 질문-답변 예시 데이터
        self.vehicle_examples = [
            {
                "query": "오일 교체는 언제 해야 하나요?",
                "context": "엔진 오일은 주행 거리에 따라 교체 주기가 달라집니다. 일반적으로 10,000km 또는 12개월마다 교체하는 것이 권장됩니다. 오일 교체는 볼보 공식 서비스 센터에서 실시하는 것이 좋습니다. [페이지 326-327]",
                "answer": "엔진 오일은 주행 거리 10,000km 또는 12개월마다 교체하는 것이 권장됩니다. 정확한 교체 시기는 주행 환경과 운전 습관에 따라 달라질 수 있으니, 볼보 공식 서비스 센터에서 점검받으시기 바랍니다. 안전상 주의사항으로 오일 레벨을 정기적으로 확인하시기 바랍니다.\n\n📚 참고 페이지: 326-327"
            },
            {
                "query": "브레이크가 제대로 작동하지 않는 것 같아요.",
                "context": "브레이크 시스템에 문제가 생겼을 때는 즉시 주행을 중단하고 안전한 장소에 주차해야 합니다. 브레이크 패드의 마모나 브레이크 액체 부족이 원인일 수 있습니다. 브레이크 경고등이 켜져 있는지 확인하세요. [페이지 195-196]",
                "answer": "브레이크 시스템에 이상이 있을 때는 즉시 안전한 장소에 주차하고 주행을 중단하세요. 다음 사항들을 확인해주세요:\n\n1. 브레이크 경고등 점등 여부\n2. 브레이크 페달의 느낌 변화\n3. 브레이크 액체 레벨\n\n이러한 증상들은 브레이크 패드 마모나 브레이크 액체 부족을 나타낼 수 있습니다. 안전상 주의사항으로 즉시 볼보 공식 서비스 센터에 연락하여 점검을 받으시기 바랍니다.\n\n📚 참고 페이지: 195-196"
            },
            {
                "query": "타이어 소음이 이상해요.",
                "context": "타이어에서 비정상적인 소음이 나는 경우 여러 원인이 있을 수 있습니다. 타이어 공기압 부족, 타이어 마모, 이물질 끼임 등이 주요 원인입니다. 타이어 공기압을 정기적으로 확인하고, 타이어 표면에 손상이 없는지 검사하세요. [페이지 316-321]",
                "answer": "타이어에서 비정상적인 소음이 나는 경우 다음 사항들을 확인해주세요:\n\n1. **타이어 공기압 확인**: 권장 압력에 맞게 조정\n2. **타이어 표면 검사**: 균열, 손상, 이물질 끼임 여부 확인\n3. **타이어 마모도 점검**: 트레드 깊이 확인\n\n이러한 문제들은 주행 안전에 직접적인 영향을 미칠 수 있습니다. 소음이 지속되거나 악화될 경우 즉시 전문가의 점검을 받으시기 바랍니다.\n\n📚 참고 페이지: 316-321"
            }
        ]
        
        print("✅ Few-shot 예시 데이터 초기화 완료")
    
    def _create_few_shot_prompt(self, system_message: str, user_query: str) -> ChatPromptTemplate:
        """차량 매뉴얼 전용 Few-shot 프롬프트 생성"""
        
        # 예시 템플릿 정의
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{query}"),
            ("assistant", "{answer}")
        ])
        
        # Few-shot 프롬프트 생성
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.vehicle_examples
        )
        
        # 최종 프롬프트 구성
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            few_shot_prompt,
            ("human", user_query)
        ])
        
        return final_prompt
    
    def query_analyzer(self, state: AgentState) -> AgentState:
        """사용자 질문 분석 및 검색 전략 결정"""
        query = state["query"]
        
        # Few-shot 방식으로 질문 분석 프롬프트 생성
        analysis_system_message = """당신은 차량 매뉴얼 전문가입니다. 사용자의 질문을 분석하여 최적의 검색 전략을 결정해주세요.
        
        분류 기준:
        - general: 일반적인 차량 정보 질문
        - specific: 구체적인 부품/절차 질문
        - troubleshooting: 문제 해결 질문
        
        각 전략에 따른 검색 방법도 함께 결정해주세요."""
        
        analysis_query = f"질문: {query}\n\n이 질문의 전략과 검색 방법을 결정해주세요."
        
        # Few-shot 프롬프트 사용 (예시 데이터는 질문 분석용으로 사용)
        analysis_prompt = self._create_few_shot_prompt(analysis_system_message, analysis_query)
        
        # Few-shot 프롬프트 체인 실행
        try:
            analysis_chain = analysis_prompt | self.llm | StrOutputParser()
            response = analysis_chain.invoke({"query": query})
        except Exception as e:
            print(f"질문 분석 오류: {str(e)}")
            response = None
        
        # 실제로는 JSON 파싱 필요 - 여기서는 간단한 로직으로 대체
        
        # 키워드 기반 전략 결정 (Re-rank 및 Compression 고려)
        query_lower = query.lower()
        
        # 복잡한 질문 판단 (여러 요소 포함)
        complexity_indicators = ['그리고', '또한', '만약', '어떤', '언제', '왜']
        is_complex = any(indicator in query_lower for indicator in complexity_indicators) or len(query.split()) > 8
        
        # 중요도/안전성 판단
        high_importance = any(word in query_lower for word in ['안전', '위험', '경고', '비상', '응급'])
        
        if any(word in query_lower for word in ['문제', '오류', '경고', '안됨', '이상', '고장']):
            search_strategy = "troubleshooting"
            # 안전 관련 문제는 고품질 검색 필요
            search_method = "cross_encoder_rerank" if high_importance else "expanded_query"
        elif any(word in query_lower for word in ['방법', '어떻게', '교체', '수리', '설치']):
            search_strategy = "specific" 
            # 복잡한 절차는 압축된 정보 필요
            search_method = "contextual_compression" if is_complex else "multi_query"
        else:
            search_strategy = "general"
            # 일반적 질문은 전문용어 확장 또는 재순위화 사용
            if any(term in query_lower for term in ['타이어', '브레이크', '엔진']):
                search_method = "cross_encoder_rerank"  # 전문 용어는 정확한 재순위화 필요
            else:
                search_method = "hybrid_semantic"
        
        state["search_strategy"] = search_strategy
        state["search_method"] = search_method  # 새로운 필드 추가
        state["messages"].append(AIMessage(content=f"질문을 분석했습니다. 전략: {search_strategy}, 방법: {search_method}"))
        
        return state
    
    def search_executor(self, state: AgentState) -> AgentState:
        """검색 전략에 따른 정보 수집"""
        query = state["query"]
        strategy = state["search_strategy"]
        search_method = state.get("search_method", "hybrid_balanced")
        
        # 하이브리드 검색 가중치 설정
        weight_configs = {
            "hybrid_semantic": 0.7,    # 의미론적 검색 우선
            "hybrid_balanced": 0.5,    # 균형 검색
            "hybrid_keyword": 0.3      # 키워드 검색 우선
        }
        
        semantic_weight = weight_configs.get(search_method, 0.5)
        
        if strategy == "general":
            if search_method == "cross_encoder_rerank":
                # Cross-Encoder 재순위화 검색
                results = cross_encoder_rerank_search.invoke({"query": query, "top_k": 5})
            elif search_method == "expanded_query":
                # 전문 용어 확장 검색
                results = expanded_query_search.invoke({"query": query, "top_k": 5})
            else:
                # 의미론적 검색 우선 하이브리드
                results = hybrid_search.invoke({
                    "query": query, 
                    "top_k": 5, 
                    "semantic_weight": semantic_weight
                })
            
        elif strategy == "specific":
            if search_method == "contextual_compression":
                # 맥락 압축 검색 (복잡한 절차에 적합)
                results = contextual_compression_search.invoke({"query": query, "top_k": 5})
            elif search_method == "multi_query":
                # 다중 쿼리 검색 (복잡한 질문에 적합)
                results = multi_query_search.invoke({"query": query, "top_k": 5})
            else:
                # 균형 하이브리드 검색
                results = hybrid_search.invoke({
                    "query": query, 
                    "top_k": 5, 
                    "semantic_weight": semantic_weight
                })
            
            # 추가 컨텍스트 검색
            if results and any(r.get("page", 0) > 0 for r in results):
                pages = [r.get("page", 0) for r in results if r.get("page", 0) > 0]
                context_results = page_context_search.invoke({"page_numbers": pages[:3]})
                results.extend(context_results)
            
        elif strategy == "troubleshooting":
            if search_method == "cross_encoder_rerank":
                # Cross-Encoder 재순위화 (안전 관련 문제에 고품질 검색)
                results = cross_encoder_rerank_search.invoke({"query": query, "top_k": 5})
            elif search_method == "expanded_query":
                # 전문 용어 확장 검색 (경고등 등 전문 용어 포함)
                results = expanded_query_search.invoke({"query": query, "top_k": 5})
            else:
                # 키워드 검색 우선 하이브리드
                # 문제해결 관련 쿼리 확장
                expanded_query = f"문제해결 오류 해결 {query}"
                results = hybrid_search.invoke({
                    "query": expanded_query, 
                    "top_k": 5, 
                    "semantic_weight": semantic_weight
                })
            
            # 관련 페이지의 컨텍스트도 수집
            if results and any(r.get("page", 0) > 0 for r in results):
                pages = [r.get("page", 0) for r in results if r.get("page", 0) > 0]
                context_results = page_context_search.invoke({"page_numbers": pages[:3]})
                results.extend(context_results)
        
        else:
            # 기본적으로 균형 하이브리드 검색
            results = hybrid_search.invoke({
                "query": query, 
                "top_k": 5, 
                "semantic_weight": 0.5
            })
        
        # Document 객체로 변환
        documents = []
        for r in results:
            if isinstance(r, dict):
                documents.append(Document(
                    page_content=r.get("content", ""), 
                    metadata={
                        "page": r.get("page", 0),
                        "source": r.get("source", ""),
                        "score": r.get("score", 0.0)
                    }
                ))
            else:
                # 이미 Document 객체인 경우
                documents.append(r)
        
        state["search_results"] = documents
        state["messages"].append(AIMessage(content=f"{len(documents)}개의 관련 문서를 찾았습니다. (검색 방법: {search_method})"))
        
        return state
    
    def context_evaluator(self, state: AgentState) -> AgentState:
        """검색 결과의 품질 평가 및 추가 검색 필요성 판단"""
        search_results = state["search_results"]
        query = state["query"]
        
        if not search_results:
            state["need_clarification"] = True
            state["confidence_score"] = 0.0
            return state
        
        # 컨텍스트 품질 평가
        evaluation_prompt = f"""
        질문: {query}
        
        검색 결과: {[doc.page_content[:200] for doc in search_results[:3]]}
        
        이 검색 결과가 질문에 충분히 답할 수 있는지 0-1 점수로 평가하세요.
        0.7 이상이면 충분, 0.7 미만이면 추가 정보 필요
        
        점수만 반환: 0.85
        """
        
        # 실제로는 LLM 호출 및 점수 파싱
        confidence = 0.8  # 예시
        
        state["confidence_score"] = confidence
        state["need_clarification"] = confidence < 0.7
        
        # 페이지 번호 수집
        pages = []
        for doc in search_results:
            if "page" in doc.metadata:
                pages.append(doc.metadata["page"])
        state["page_references"] = list(set(pages))
        
        return state
    
    def answer_generator(self, state: AgentState) -> AgentState:
        """최종 답변 생성"""
        query = state["query"]
        search_results = state["search_results"]
        confidence = state["confidence_score"]
        pages = state["page_references"]
        
        # 컨텍스트 구성
        context_text = "\n\n".join([
            f"[페이지 {doc.metadata.get('page', '?')}] {doc.page_content}" 
            for doc in search_results[:5]
        ])
        
        # Few-shot 방식으로 답변 생성 프롬프트 생성
        answer_system_message = f"""당신은 볼보 XC60 차량 매뉴얼 전문가입니다. 사용자의 질문에 대해 정확하고 도움이 되는 답변을 제공해주세요.
        
        답변 지침:
        1. 정확하고 실용적인 정보 제공
        2. 관련 페이지 번호 명시
        3. 안전상 주의사항 반드시 포함
        4. 전문가 상담 권유 (필요시)
        5. 친절하고 이해하기 쉬운 언어 사용
        
        참고 자료: {context_text}
        확신도: {confidence}"""
        
        answer_query = f"질문: {query}\n\n위 참고 자료를 바탕으로 정확하고 도움이 되는 답변을 해주세요."
        
        # Few-shot 프롬프트 사용
        answer_prompt = self._create_few_shot_prompt(answer_system_message, answer_query)
        
        # Few-shot 프롬프트 체인 실행
        try:
            answer_chain = answer_prompt | self.llm | StrOutputParser()
            final_answer = answer_chain.invoke({"query": query, "context_text": context_text, "confidence": confidence})
        except Exception as e:
            print(f"답변 생성 오류: {str(e)}")
            # 폴백으로 zero-shot 방식 사용
            fallback_prompt = f"""
            차량 매뉴얼을 참고하여 사용자 질문에 답변하세요.
            
            질문: {query}
            참고 자료: {context_text}
            
            정확하고 도움이 되는 답변을 제공해주세요.
            """
            response = self.llm.invoke(fallback_prompt)
            final_answer = response.content if hasattr(response, 'content') else str(response)
        
        # 페이지 참조 정보 추가
        if pages:
            final_answer += f"\n\n📚 참고 페이지: {', '.join(map(str, sorted(pages)))}"
        
        state["final_answer"] = final_answer
        state["context"] = context_text
        state["messages"].append(AIMessage(content=final_answer))
        
        return state
    
    def clarification_handler(self, state: AgentState) -> AgentState:
        """추가 정보가 필요한 경우 처리"""
        query = state["query"]
        
        clarification = f"""
        죄송합니다. '{query}'에 대한 정확한 정보를 메뉴얼에서 찾기 어렵습니다.
        
        더 구체적으로 질문해 주시거나 다음을 확인해 보세요:
        - 차량 모델과 연식
        - 구체적인 상황이나 증상
        - 관련 부품명
        
        또는 다른 방식으로 질문해 보시겠어요?
        """
        
        state["final_answer"] = clarification
        state["messages"].append(AIMessage(content=clarification))
        
        return state
    
    def should_clarify(self, state: AgentState) -> str:
        """추가 설명이 필요한지 판단"""
        return "clarify" if state["need_clarification"] else "generate_answer"
    
    def build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("analyze_query", self.query_analyzer)
        workflow.add_node("search", self.search_executor)  
        workflow.add_node("evaluate", self.context_evaluator)
        workflow.add_node("generate_answer", self.answer_generator)
        workflow.add_node("clarify", self.clarification_handler)
        
        # 엣지 연결
        workflow.set_entry_point("analyze_query")
        
        workflow.add_edge("analyze_query", "search")
        workflow.add_edge("search", "evaluate")
        
        workflow.add_conditional_edges(
            "evaluate",
            self.should_clarify,
            {
                "generate_answer": "generate_answer",
                "clarify": "clarify"
            }
        )
        
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("clarify", END)
        
        return workflow.compile()

# 사용 예시
def main():
    # PDF 파일 경로 지정
    pdf_path = "/Users/1112931/llm-3/project/data/backup/kr_ko-KR_xc60_2026.pdf"
    agent = VehicleManualAgent(pdf_path=pdf_path)
    app = agent.build_graph()
    
    # Re-rank 및 Contextual Compression 효과를 테스트할 수 있는 질문들
    test_queries = [
        "타이어 공기압 측정 및 조정 방법을 알려주세요",  # Cross-Encoder 재순위화 테스트
        "브레이크 패드 교체 절차와 주의사항 그리고 필요한 도구들을 자세히 알려주세요",  # Contextual Compression 테스트
        "엔진 경고등 비상 상황 대처 방법",  # Cross-Encoder (안전 관련)
        "차량 시동이 안 걸릴 때 체크할 항목들과 단계별 해결 방법",  # Contextual Compression
        "경고등 점등 의미와 대응 방법"  # Cross-Encoder 재순위화
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"테스트 {i}: {query}")
        print(f"{'='*50}")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "search_results": [],
            "context": "",
            "final_answer": "",
        "search_strategy": "",
        "search_method": "",
        "confidence_score": 0.0,
        "page_references": [],
        "need_clarification": False
        }
        
        try:
            result = app.invoke(initial_state)
            print("\n📝 최종 답변:")
            print(result["final_answer"])
            print(f"\n🎯 신뢰도: {result['confidence_score']:.2f}")
            if result["page_references"]:
                print(f"📚 참고 페이지: {result['page_references']}")
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
        
        if i < len(test_queries):
            print("\n다음 테스트를 3초 후 시작합니다...")
            import time
            time.sleep(3)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {str(e)}")
        import traceback
        traceback.print_exc()