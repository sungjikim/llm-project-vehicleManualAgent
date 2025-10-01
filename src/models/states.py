"""
SubGraph용 상태 정의
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage


class EmergencyDetectionState(TypedDict):
    """응급 상황 감지 SubGraph 상태"""
    query: str
    is_emergency: bool
    emergency_level: str
    emergency_score: float
    emergency_analysis: Dict[str, Any]
    search_strategy: str
    search_method: str
    compression_method: str


class SearchPipelineState(TypedDict):
    """검색 파이프라인 SubGraph 상태"""
    query: str
    search_strategy: str
    search_method: str
    compression_method: str
    confidence_score: float
    search_results: List[Dict[str, Any]]
    page_references: List[int]


class AnswerGenerationState(TypedDict):
    """답변 생성 SubGraph 상태"""
    query: str
    search_results: List[Dict[str, Any]]
    page_references: List[int]
    is_emergency: bool
    emergency_level: str
    final_answer: str
    confidence_score: float
    evaluation_details: Optional[Dict[str, Any]]


class DrivingContextState(TypedDict):
    """주행 상황 처리 SubGraph 상태"""
    query: str
    original_answer: str
    is_emergency: bool
    emergency_level: str
    is_driving: bool
    driving_confidence: float
    driving_indicators: List[str]
    driving_urgency: str
    compression_needed: bool
    compressed_answer: str
    final_answer: str


class SpeechRecognitionState(TypedDict):
    """음성 인식 SubGraph 상태"""
    audio_data: Optional[bytes]
    audio_file_path: Optional[str]
    recognized_text: str
    confidence: float
    error: Optional[str]
    processing_method: str
    is_valid: bool
    validation_error: Optional[str]
    final_text: str


class MainAgentState(TypedDict):
    """메인 에이전트 상태 (SubGraph 통합용)"""
    messages: List[BaseMessage]
    query: str
    search_results: List[Dict[str, Any]]
    context: str
    final_answer: str
    search_strategy: str
    search_method: str
    confidence_score: float
    page_references: List[int]
    need_clarification: bool
    
    # 응급 상황 관련
    is_emergency: bool
    emergency_level: str
    emergency_score: float
    emergency_analysis: Dict[str, Any]
    compression_method: str
    
    # 주행 상황 관련
    is_driving: bool
    driving_confidence: float
    driving_indicators: List[str]
    driving_urgency: str
    compression_needed: bool
    compressed_answer: str
    
    # 음성 인식 관련
    audio_data: Optional[bytes]
    audio_file_path: Optional[str]
    recognized_text: str
    speech_confidence: float
    speech_error: Optional[str]
    
    # 평가 관련
    evaluation_details: Optional[Dict[str, Any]]
