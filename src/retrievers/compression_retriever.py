"""
압축 및 재순위화 리트리버
"""

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    CrossEncoderReranker,
    LLMChainExtractor,
    DocumentCompressorPipeline,
    EmbeddingsFilter
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_transformers import EmbeddingsRedundantFilter

from ..config.settings import (
    CROSS_ENCODER_MODEL, CANDIDATE_DOCS_COUNT, 
    SIMILARITY_THRESHOLD, REDUNDANCY_THRESHOLD
)


class CompressionRetrieverManager:
    """압축 및 재순위화 리트리버 관리 클래스"""
    
    def __init__(self, vector_store, embeddings, llm):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.llm = llm
        self.cross_encoder_retriever = None
        self.compression_retriever = None
    
    def initialize_cross_encoder_retriever(self):
        """Cross-Encoder 모델을 사용한 재순위화 검색기 초기화"""
        if self.vector_store is None:
            print("⚠️ 벡터 저장소가 없어 Cross-Encoder 재순위화기를 초기화할 수 없습니다.")
            return
        
        try:
            print("🔄 Cross-Encoder 재순위화 시스템 초기화 중...")
            
            # Cross-Encoder 모델 로드 (경량화된 모델 사용)
            cross_encoder_model = HuggingFaceCrossEncoder(
                model_name=CROSS_ENCODER_MODEL  # 다국어 지원 모델
            )
            
            # 재순위화 컴프레서 생성
            reranker = CrossEncoderReranker(
                model=cross_encoder_model,
                top_n=5  # 상위 5개 문서만 반환
            )
            
            # 베이스 리트리버 생성 (더 많은 후보 문서 검색)
            base_retriever = self.vector_store.as_retriever(search_kwargs={"k": CANDIDATE_DOCS_COUNT})
            
            # 컨텍스츄얼 컴프레션 리트리버 생성
            self.cross_encoder_retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=base_retriever
            )
            
            print("✅ Cross-Encoder 재순위화 시스템 초기화 완료")
            print(f"   모델: {CROSS_ENCODER_MODEL}")
            print(f"   후보 문서: {CANDIDATE_DOCS_COUNT}개 -> 상위 5개 선별")
            
        except Exception as e:
            print(f"Cross-Encoder 재순위화 초기화 오류: {str(e)}")
            self.cross_encoder_retriever = None
    
    def initialize_contextual_compression(self):
        """맥락 압축 검색기 초기화"""
        if self.vector_store is None:
            print("⚠️ 벡터 저장소가 없어 맥락 압축기를 초기화할 수 없습니다.")
            return
        
        try:
            print("📝 맥락 압축 시스템 초기화 중...")
            
            # 1. 임베딩 기반 필터링 (유사도 임계값 설정)
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=SIMILARITY_THRESHOLD  # 60% 이상 유사한 문서만 통과
            )
            
            # 2. 중복 제거 필터
            redundant_filter = EmbeddingsRedundantFilter(
                embeddings=self.embeddings,
                similarity_threshold=REDUNDANCY_THRESHOLD  # 90% 이상 유사한 문서 제거
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
            base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})  # 15개 후보 검색
            
            # 컨텍스츄얼 컴프레션 리트리버 생성
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=pipeline_compressor,
                base_retriever=base_retriever
            )
            
            print("✅ 맥락 압축 시스템 초기화 완료")
            print(f"   1단계: 임베딩 필터링 (유사도 > {SIMILARITY_THRESHOLD*100}%)")
            print(f"   2단계: 중복 제거 (유사도 > {REDUNDANCY_THRESHOLD*100}%)")
            print(f"   3단계: LLM 기반 핵심 정보 추출")
            
        except Exception as e:
            print(f"맥락 압축 초기화 오류: {str(e)}")
            self.compression_retriever = None
    
    def get_cross_encoder_retriever(self):
        """Cross-Encoder 재순위화 검색기 반환"""
        return self.cross_encoder_retriever
    
    def get_compression_retriever(self):
        """맥락 압축 검색기 반환"""
        return self.compression_retriever
