"""
에이전트 상태 정의
"""

from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
import operator


class AgentState(TypedDict):
    """
    차량 매뉴얼 RAG 에이전트의 상태 정의
    """
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    search_results: List[Document]
    context: str
    final_answer: str
    search_strategy: str  # "general", "specific", "troubleshooting"
    search_method: str   # "hybrid_semantic", "hybrid_balanced", "hybrid_keyword", etc.
    confidence_score: float
    page_references: List[int]
    need_clarification: bool
    
    # 응급 상황 관련 필드
    is_emergency: bool
    emergency_level: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"
    emergency_score: float
    emergency_analysis: dict
    compression_method: str
    
    # 주행 중 상황 관련 필드
    is_driving: bool
    driving_confidence: float
    driving_indicators: List[str]
    driving_urgency: str  # "immediate", "urgent", "normal"
    compression_needed: bool
    compressed_answer: str