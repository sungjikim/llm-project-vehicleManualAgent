"""
하이브리드 검색 리트리버 (벡터 + BM25)
"""

from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.documents import Document

from ..config.settings import CUSTOM_WORDS

try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False
    print("⚠️ Kiwi 토크나이저가 설치되지 않았습니다. 기본 토크나이저를 사용합니다.")


class HybridRetrieverManager:
    """하이브리드 검색 리트리버 관리 클래스"""
    
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.kiwi_model = None
        self.bm25_retriever = None
        self.multi_query_retriever = None
        
    def _setup_korean_tokenizer(self):
        """한국어 토크나이저 설정"""
        if not KIWI_AVAILABLE:
            return None
            
        try:
            self.kiwi_model = Kiwi()
            # 사용자 정의 단어 추가
            for word, pos in CUSTOM_WORDS:
                self.kiwi_model.add_user_word(word, pos)
                
            print(f"✅ 한국어 토크나이저 설정 완료: {len(CUSTOM_WORDS)}개 사용자 단어 추가")
            return self.kiwi_model
            
        except Exception as e:
            print(f"한국어 토크나이저 설정 오류: {str(e)}")
            return None
    
    def _korean_tokenizer(self, text: str) -> List[str]:
        """한국어 텍스트 토크나이징"""
        if self.kiwi_model is None:
            # Kiwi가 없으면 기본 공백 분리
            return text.split()
        
        try:
            # Kiwi로 토크나이징
            tokens = []
            for token, pos, _, _ in self.kiwi_model.analyze(text):
                # 명사, 동사, 형용사, 고유명사만 추출
                if pos.startswith(('NN', 'VV', 'VA', 'NP')):
                    tokens.append(token)
            return tokens if tokens else text.split()
        except:
            return text.split()
    
    def initialize_bm25_retriever(self, documents: List[Document]):
        """BM25 검색기 초기화"""
        try:
            # 한국어 토크나이저 설정
            self._setup_korean_tokenizer()
            
            # BM25 검색기 생성
            if KIWI_AVAILABLE and self.kiwi_model is not None:
                # 한국어 토크나이저 사용
                self.bm25_retriever = BM25Retriever.from_documents(
                    documents=documents,
                    preprocess_func=self._korean_tokenizer,
                    k=5
                )
                print("✅ BM25 검색기 생성 완료 (한국어 토크나이저 적용)")
            else:
                # 기본 토크나이저 사용
                self.bm25_retriever = BM25Retriever.from_documents(
                    documents=documents,
                    k=5
                )
                print("✅ BM25 검색기 생성 완료 (기본 토크나이저)")
                
        except Exception as e:
            print(f"BM25 검색기 초기화 오류: {str(e)}")
            self.bm25_retriever = None
    
    def initialize_multi_query_retriever(self):
        """MultiQueryRetriever 초기화"""
        if self.vector_store is None:
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
            base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            self.multi_query_retriever = MultiQueryRetriever(
                retriever=base_retriever,
                llm_chain=multi_query_chain,
                parser_key="lines"
            )
            
            print("✅ MultiQueryRetriever 초기화 완료")
            
        except Exception as e:
            print(f"MultiQueryRetriever 초기화 오류: {str(e)}")
            self.multi_query_retriever = None
    
    def get_bm25_retriever(self):
        """BM25 검색기 반환"""
        return self.bm25_retriever
    
    def get_multi_query_retriever(self):
        """다중 쿼리 검색기 반환"""
        return self.multi_query_retriever
