"""
문서 로딩 유틸리티
"""

import os
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config.settings import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentLoader:
    """PDF 문서 로딩 및 전처리 유틸리티"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """PDF 파일을 로드하고 문서로 변환"""
        try:
            # 파일 존재 확인
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            
            print(f"📄 PDF 파일 로딩 중: {Path(pdf_path).name}")
            
            # PDF 로더로 문서 로드
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            print(f"   📖 로드된 페이지 수: {len(documents)}")
            
            # 메타데이터 보강
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source': pdf_path,
                    'page': i + 1,
                    'total_pages': len(documents)
                })
            
            return documents
            
        except Exception as e:
            print(f"❌ PDF 로딩 오류: {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서를 청크 단위로 분할"""
        try:
            print(f"✂️  문서 분할 중 (청크 크기: {self.chunk_size}, 겹침: {self.chunk_overlap})")
            
            split_docs = self.text_splitter.split_documents(documents)
            
            print(f"   📑 분할된 문서 조각 수: {len(split_docs)}")
            
            # 분할된 문서의 메타데이터 보강
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(doc.page_content)
                })
            
            return split_docs
            
        except Exception as e:
            print(f"❌ 문서 분할 오류: {str(e)}")
            return []
    
    def load_and_split_pdf(self, pdf_path: str) -> List[Document]:
        """PDF 로드 및 분할을 한번에 수행"""
        documents = self.load_pdf(pdf_path)
        if not documents:
            return []
        
        return self.split_documents(documents)
    
    def get_document_stats(self, documents: List[Document]) -> dict:
        """문서 통계 정보 반환"""
        if not documents:
            return {"total_docs": 0, "total_chars": 0, "avg_chunk_size": 0}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chunk_size = total_chars / len(documents)
        
        pages = set()
        for doc in documents:
            if 'page' in doc.metadata:
                pages.add(doc.metadata['page'])
        
        return {
            "total_docs": len(documents),
            "total_chars": total_chars,
            "avg_chunk_size": round(avg_chunk_size, 2),
            "unique_pages": len(pages),
            "min_chunk_size": min(len(doc.page_content) for doc in documents),
            "max_chunk_size": max(len(doc.page_content) for doc in documents)
        }
