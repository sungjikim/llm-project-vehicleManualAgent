"""
검색 관련 도구들
"""

from typing import List, Dict
from langchain_core.tools import tool
from ..config.settings import VEHICLE_TERMS

# 전역 변수로 검색기들 관리
vector_store = None
bm25_retriever = None
hybrid_retriever = None
multi_query_retriever = None
cross_encoder_retriever = None
compression_retriever = None


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
            from langchain.retrievers import EnsembleRetriever
            
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
        # 쿼리 확장
        expanded_terms = []
        query_lower = query.lower()
        
        for key, synonyms in VEHICLE_TERMS.items():
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
