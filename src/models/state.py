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
