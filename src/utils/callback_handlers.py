"""
콜백 핸들러 - 성능 모니터링 및 실시간 알림
"""

import time
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class PerformanceMonitoringHandler(BaseCallbackHandler):
    """성능 모니터링 콜백 핸들러"""
    
    def __init__(self, enable_detailed_logging: bool = True):
        super().__init__()
        self.enable_detailed_logging = enable_detailed_logging
        self.session_start_time = None
        self.query_start_time = None
        self.llm_calls = []
        self.retriever_calls = []
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # 성능 통계
        self.performance_stats = {
            "total_queries": 0,
            "total_llm_calls": 0,
            "total_retriever_calls": 0,
            "total_tokens_used": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "session_duration": 0.0
        }
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """체인 시작 시 호출"""
        if self.session_start_time is None:
            self.session_start_time = time.time()
        
        self.query_start_time = time.time()
        self.performance_stats["total_queries"] += 1
        
        if self.enable_detailed_logging:
            print(f"🔍 [{datetime.now().strftime('%H:%M:%S')}] 쿼리 처리 시작")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """체인 종료 시 호출"""
        if self.query_start_time:
            query_duration = time.time() - self.query_start_time
            
            # 평균 응답 시간 업데이트
            total_queries = self.performance_stats["total_queries"]
            current_avg = self.performance_stats["average_response_time"]
            self.performance_stats["average_response_time"] = (
                (current_avg * (total_queries - 1) + query_duration) / total_queries
            )
            
            if self.enable_detailed_logging:
                print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] 쿼리 처리 완료 ({query_duration:.2f}초)")
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """LLM 호출 시작 시 호출"""
        self.performance_stats["total_llm_calls"] += 1
        # serialized가 None일 수 있으므로 안전하게 처리
        if serialized and serialized.get("id"):
            model_name = serialized["id"][-1] if serialized["id"] else "unknown"
        else:
            model_name = "unknown"
            
        llm_call = {
            "start_time": time.time(),
            "prompts": prompts if self.enable_detailed_logging else len(prompts),
            "model": model_name
        }
        self.llm_calls.append(llm_call)
        
        if self.enable_detailed_logging:
            print(f"🤖 [{datetime.now().strftime('%H:%M:%S')}] LLM 호출 시작 (모델: {llm_call['model']})")
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM 호출 종료 시 호출"""
        if self.llm_calls:
            current_call = self.llm_calls[-1]
            current_call["end_time"] = time.time()
            current_call["duration"] = current_call["end_time"] - current_call["start_time"]
            
            # 토큰 사용량 계산
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                tokens_used = token_usage.get('total_tokens', 0)
                self.total_tokens += tokens_used
                self.performance_stats["total_tokens_used"] += tokens_used
                
                # 비용 추정 (GPT-4o-mini 기준: $0.15/1M input tokens, $0.6/1M output tokens)
                input_tokens = token_usage.get('prompt_tokens', 0)
                output_tokens = token_usage.get('completion_tokens', 0)
                cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.6 / 1_000_000)
                self.total_cost += cost
                self.performance_stats["total_cost"] += cost
                
                current_call["tokens"] = tokens_used
                current_call["cost"] = cost
                
                if self.enable_detailed_logging:
                    print(f"💰 [{datetime.now().strftime('%H:%M:%S')}] LLM 완료 "
                          f"({tokens_used} 토큰, ${cost:.4f}, {current_call['duration']:.2f}초)")
    
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs) -> None:
        """리트리버 호출 시작 시 호출"""
        self.performance_stats["total_retriever_calls"] += 1
        
        # serialized가 None일 수 있으므로 안전하게 처리
        if serialized and serialized.get("id"):
            retriever_type = serialized["id"][-1] if serialized["id"] else "unknown"
        else:
            retriever_type = "unknown"
            
        retriever_call = {
            "start_time": time.time(),
            "query": query[:100] + "..." if len(query) > 100 else query,
            "type": retriever_type
        }
        self.retriever_calls.append(retriever_call)
        
        if self.enable_detailed_logging:
            print(f"🔍 [{datetime.now().strftime('%H:%M:%S')}] 검색 시작 (타입: {retriever_call['type']})")
    
    def on_retriever_end(self, documents: List[Document], **kwargs) -> None:
        """리트리버 호출 종료 시 호출"""
        if self.retriever_calls:
            current_call = self.retriever_calls[-1]
            current_call["end_time"] = time.time()
            current_call["duration"] = current_call["end_time"] - current_call["start_time"]
            current_call["documents_found"] = len(documents)
            
            if self.enable_detailed_logging:
                print(f"📄 [{datetime.now().strftime('%H:%M:%S')}] 검색 완료 "
                      f"({len(documents)}개 문서, {current_call['duration']:.2f}초)")
    
    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """체인 오류 시 호출"""
        print(f"❌ [{datetime.now().strftime('%H:%M:%S')}] 오류 발생: {str(error)}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        if self.session_start_time:
            self.performance_stats["session_duration"] = time.time() - self.session_start_time
        
        return {
            **self.performance_stats,
            "llm_calls_detail": self.llm_calls[-5:] if self.enable_detailed_logging else [],
            "retriever_calls_detail": self.retriever_calls[-5:] if self.enable_detailed_logging else []
        }
    
    def print_performance_report(self):
        """성능 리포트 출력"""
        stats = self.get_performance_summary()
        
        print("\n" + "=" * 60)
        print("📊 성능 모니터링 리포트")
        print("=" * 60)
        print(f"🕐 세션 시간: {stats['session_duration']:.1f}초")
        print(f"❓ 총 쿼리 수: {stats['total_queries']}개")
        print(f"🤖 LLM 호출: {stats['total_llm_calls']}회")
        print(f"🔍 검색 호출: {stats['total_retriever_calls']}회")
        print(f"🎯 평균 응답 시간: {stats['average_response_time']:.2f}초")
        print(f"💎 총 토큰 사용: {stats['total_tokens_used']:,}개")
        print(f"💰 총 예상 비용: ${stats['total_cost']:.4f}")
        print("=" * 60)


class RealTimeNotificationHandler(BaseCallbackHandler):
    """실시간 알림 콜백 핸들러"""
    
    def __init__(self, enable_progress_bar: bool = True, enable_notifications: bool = True):
        super().__init__()
        self.enable_progress_bar = enable_progress_bar
        self.enable_notifications = enable_notifications
        self.current_step = 0
        self.total_steps = 0
        self.step_descriptions = []
        
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """체인 시작 알림"""
        if self.enable_notifications:
            query = ""
            if isinstance(inputs, dict):
                query = inputs.get("query", inputs.get("input", ""))
            elif hasattr(inputs, 'content'):
                query = str(inputs.content)
            elif isinstance(inputs, str):
                query = inputs
            else:
                query = str(inputs)
            
            if len(query) > 50:
                query = query[:50] + "..."
            
            print(f"\n🚀 처리 시작: {query}")
            
        # 진행 단계 설정
        self.current_step = 0
        self.total_steps = 4  # 분석 -> 검색 -> 재순위화 -> 답변생성
        self.step_descriptions = [
            "쿼리 분석 중...",
            "문서 검색 중...",
            "결과 최적화 중...",
            "답변 생성 중..."
        ]
        
        if self.enable_progress_bar:
            self._print_progress_bar()
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """LLM 시작 알림"""
        self.current_step += 1
        if self.enable_progress_bar and self.current_step <= len(self.step_descriptions):
            self._print_progress_bar()
    
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs) -> None:
        """검색 시작 알림"""
        if self.enable_notifications:
            # serialized가 None일 수 있으므로 안전하게 처리
            if serialized and serialized.get("id"):
                retriever_type = serialized["id"][-1] if serialized["id"] else "unknown"
            else:
                retriever_type = "unknown"
            print(f"🔍 {retriever_type} 검색 실행 중...")
    
    def on_retriever_end(self, documents: List[Document], **kwargs) -> None:
        """검색 완료 알림"""
        if self.enable_notifications:
            print(f"📄 {len(documents)}개 관련 문서 발견")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """체인 완료 알림"""
        if self.enable_progress_bar:
            self.current_step = self.total_steps
            self._print_progress_bar()
            print()  # 줄바꿈
        
        if self.enable_notifications:
            print("✅ 답변 생성 완료!")
    
    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """오류 알림"""
        print(f"\n🚨 처리 중 오류 발생!")
        print(f"💥 오류 내용: {str(error)}")
        
        # 일반적인 해결 방법 제안
        if "API" in str(error):
            print("💡 해결 방법: OpenAI API 키를 확인해주세요.")
        elif "token" in str(error).lower():
            print("💡 해결 방법: 질문을 더 간단하게 만들어보세요.")
        elif "network" in str(error).lower() or "connection" in str(error).lower():
            print("💡 해결 방법: 인터넷 연결을 확인해주세요.")
    
    def _print_progress_bar(self):
        """진행률 바 출력"""
        if not self.enable_progress_bar:
            return
            
        progress = min(self.current_step / self.total_steps, 1.0)
        bar_length = 30
        filled_length = int(bar_length * progress)
        
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        percentage = int(progress * 100)
        
        current_desc = ""
        if self.current_step <= len(self.step_descriptions):
            current_desc = self.step_descriptions[self.current_step - 1] if self.current_step > 0 else self.step_descriptions[0]
        
        print(f"\r🔄 [{bar}] {percentage}% - {current_desc}", end="", flush=True)


class AlertHandler(BaseCallbackHandler):
    """경고 및 알림 핸들러"""
    
    def __init__(self, token_limit: int = 100000, cost_limit: float = 1.0):
        super().__init__()
        self.token_limit = token_limit
        self.cost_limit = cost_limit
        self.session_tokens = 0
        self.session_cost = 0.0
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM 완료 후 사용량 체크"""
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            tokens_used = token_usage.get('total_tokens', 0)
            self.session_tokens += tokens_used
            
            # 비용 계산
            input_tokens = token_usage.get('prompt_tokens', 0)
            output_tokens = token_usage.get('completion_tokens', 0)
            cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.6 / 1_000_000)
            self.session_cost += cost
            
            # 토큰 한도 체크
            if self.session_tokens > self.token_limit * 0.8:  # 80% 도달 시 경고
                print(f"\n⚠️ 토큰 사용량 경고: {self.session_tokens:,}/{self.token_limit:,} "
                      f"({self.session_tokens/self.token_limit*100:.1f}%)")
            
            # 비용 한도 체크
            if self.session_cost > self.cost_limit * 0.8:  # 80% 도달 시 경고
                print(f"\n💸 비용 경고: ${self.session_cost:.4f}/${self.cost_limit:.2f} "
                      f"({self.session_cost/self.cost_limit*100:.1f}%)")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """사용량 요약 반환"""
        return {
            "tokens_used": self.session_tokens,
            "token_limit": self.token_limit,
            "token_usage_percentage": (self.session_tokens / self.token_limit) * 100,
            "cost_incurred": self.session_cost,
            "cost_limit": self.cost_limit,
            "cost_usage_percentage": (self.session_cost / self.cost_limit) * 100
        }
