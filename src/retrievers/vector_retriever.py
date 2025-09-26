"""
벡터 검색 리트리버
"""

import os
from typing import List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config.settings import (
    DEFAULT_PDF_PATH, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE
)


class VectorStoreManager:
    """벡터 저장소 관리 클래스"""
    
    def __init__(self, pdf_path: str = None):
        self.pdf_path = pdf_path or str(DEFAULT_PDF_PATH)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        
    def initialize_vector_store(self):
        """벡터 저장소 초기화 및 PDF 문서 로드"""
        try:
            # PDF 파일 존재 확인
            if not os.path.exists(self.pdf_path):
                print(f"경고: PDF 파일을 찾을 수 없습니다: {self.pdf_path}")
                return None
            
            print(f"PDF 파일 로딩 중: {self.pdf_path}")
            
            # PDF 로더로 문서 로드
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            print(f"로드된 페이지 수: {len(documents)}")
            
            # 텍스트 분할 (토큰 제한을 고려하여 더 작은 청크로 분할)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
            )
            
            split_docs = text_splitter.split_documents(documents)
            print(f"분할된 문서 조각 수: {len(split_docs)}")
            
            # 벡터 저장소 생성
            persist_directory = str(CHROMA_DB_DIR)
            
            # 기존 데이터베이스가 있는지 확인
            if os.path.exists(persist_directory) and os.listdir(persist_directory):
                print("기존 벡터 데이터베이스 로드 중...")
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                print("새로운 벡터 데이터베이스 생성 중...")
                
                # 배치 처리로 문서 추가 (토큰 제한 회피)
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                
                print(f"총 {len(split_docs)}개 문서를 {BATCH_SIZE}개씩 배치 처리...")
                
                for i in range(0, len(split_docs), BATCH_SIZE):
                    batch = split_docs[i:i+BATCH_SIZE]
                    try:
                        self.vector_store.add_documents(batch)
                        print(f"배치 {i//BATCH_SIZE + 1}/{(len(split_docs)-1)//BATCH_SIZE + 1} 완료")
                    except Exception as batch_error:
                        print(f"배치 {i//BATCH_SIZE + 1} 오류: {str(batch_error)}")
                        continue
            
            print("벡터 저장소 초기화 완료!")
            return self.vector_store
            
        except Exception as e:
            print(f"벡터 저장소 초기화 오류: {str(e)}")
            return None
    
    def get_vector_store(self):
        """벡터 저장소 반환"""
        return self.vector_store
