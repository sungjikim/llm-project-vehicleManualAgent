"""
차량 매뉴얼 RAG 시스템 - 메인 진입점

모듈화된 LangChain/LangGraph 기반 차량 매뉴얼 RAG 에이전트
터미널 및 Gradio 웹 인터페이스 지원
"""

import time
import signal
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from src.agents.vehicle_agent import VehicleManualAgent
from src.config.settings import DEFAULT_PDF_PATH
from src.utils.callback_handlers import (
    PerformanceMonitoringHandler,
    RealTimeNotificationHandler,
    AlertHandler
)

# Gradio는 선택적 임포트
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


class GradioVehicleChatbot:
    """Gradio 기반 차량 매뉴얼 RAG 챗봇"""
    
    def __init__(self, agent, callbacks):
        self.agent = agent
        self.callbacks = callbacks
        self.chat_history = []
        self.performance_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "emergency_detected": 0,
            "driving_context_detected": 0
        }
    
    def chat_with_agent(self, message: str, history: List[List[str]]) -> List[List[str]]:
        """에이전트와 채팅하는 메인 함수"""
        if not message.strip():
            return history
        
        try:
            # 새로운 쿼리를 위해 콜백 핸들러 세션 초기화
            for callback in self.callbacks:
                if hasattr(callback, 'reset_session'):
                    callback.reset_session()
            
            # 에이전트에게 질문 전달
            start_time = time.time()
            response = self.agent.query(message, callbacks=self.callbacks)
            end_time = time.time()
            
            # 통계 업데이트
            self.performance_stats["total_queries"] += 1
            self.performance_stats["successful_queries"] += 1
            
            # 응급 상황 및 주행 중 감지 확인
            if any(indicator in response for indicator in ["🚨", "CRITICAL", "HIGH", "응급", "즉시", "위험"]):
                self.performance_stats["emergency_detected"] += 1
            
            if any(indicator in response for indicator in ["🚗", "주행 중", "운전 중", "압축"]):
                self.performance_stats["driving_context_detected"] += 1
            
            # 응답 시간 추가
            response_time = end_time - start_time
            response_with_time = f"{response}\n\n⏱️ 응답 시간: {response_time:.1f}초"
            
            # 마지막 메시지의 답변 부분 업데이트
            if history and history[-1][1] is None:
                history[-1][1] = response_with_time
            else:
                history.append([message, response_with_time])
            
            return history
            
        except Exception as e:
            error_message = f"❌ 오류가 발생했습니다: {str(e)}"
            # 마지막 메시지의 답변 부분 업데이트
            if history and history[-1][1] is None:
                history[-1][1] = error_message
            else:
                history.append([message, error_message])
            self.performance_stats["total_queries"] += 1
            return history
    
    def clear_chat(self) -> Tuple[str, List]:
        """채팅 히스토리 초기화"""
        self.chat_history = []
        return "", []
    
    def get_performance_stats(self) -> str:
        """성능 통계 반환"""
        if self.performance_stats["total_queries"] == 0:
            return "아직 질문이 없습니다."
        
        success_rate = (self.performance_stats["successful_queries"] / 
                       self.performance_stats["total_queries"]) * 100
        
        stats_text = f"""
📊 **세션 통계**
- 총 질문 수: {self.performance_stats['total_queries']}개
- 성공률: {success_rate:.1f}%
- 응급 상황 감지: {self.performance_stats['emergency_detected']}개
- 주행 중 감지: {self.performance_stats['driving_context_detected']}개
        """
        
        return stats_text.strip()
    
    def create_interface(self) -> gr.Blocks:
        """Gradio 인터페이스 생성"""
        
        with gr.Blocks(
            title="🚗 지능형 차량 어시스턴트",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            .chat-message {
                font-size: 14px;
                line-height: 1.6;
            }
            .emergency-message {
                background-color: #ffebee;
                border-left: 4px solid #f44336;
                padding: 10px;
                margin: 10px 0;
            }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🚗 지능형 차량 어시스턴트
            
            **SubGraph 아키텍처 기반 실시간 차량 매뉴얼 AI 도우미**
            
            - 🚨 **응급 상황 자동 감지** 및 즉시 대응
            - 🔍 **하이브리드 검색**, 쿼리 확장, 재순위화, 맥락 압축 지원
            - 🤖 **Few-shot 프롬프팅**으로 향상된 답변 품질
            - 📊 **실시간 성능 모니터링** 및 알림 지원
            - 🔧 **SubGraph 아키텍처**로 모듈화 및 재사용성 향상
            
            ---
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    # 채팅 인터페이스
                    chatbot = gr.Chatbot(
                        label="💬 차량 어시스턴트와 대화하기",
                        height=500,
                        show_label=True,
                        container=True,
                        bubble_full_width=False,
                        elem_classes=["chat-message"]
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="차량에 관한 질문을 입력하세요... (예: 브레이크가 안 밟혀요, 엔진 오일 교체 주기는?)",
                            label="질문 입력",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("전송", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ 대화 초기화", variant="secondary")
                        stats_btn = gr.Button("📊 통계 보기", variant="secondary")
                
                with gr.Column(scale=1):
                    # 사이드바 정보
                    gr.Markdown("""
                    ### 💡 사용법
                    
                    1. **질문 입력**: 텍스트 박스에 질문을 입력하세요
                    2. **전송**: Enter 키 또는 전송 버튼 클릭
                    3. **대화 초기화**: 새로운 대화를 시작하려면 초기화 버튼 클릭
                    
                    ### 🚨 응급 상황 예시
                    - "갑자기 브레이크가 안 밟혀요"
                    - "엔진에서 연기가 나고 있어요"
                    - "고속도로에서 타이어가 터졌어요"
                    
                    ### 🔧 일반 질문 예시
                    - "엔진 오일 교체 주기는?"
                    - "겨울철 차량 관리 방법은?"
                    - "블루투스 연결 방법을 알려주세요"
                    """)
                    
                    # 통계 표시 영역
                    stats_display = gr.Markdown(
                        value="아직 질문이 없습니다.",
                        label="📊 세션 통계"
                    )
            
            # 이벤트 핸들러
            def add_message(message, history):
                if not message.strip():
                    return "", history
                # 사용자 메시지를 히스토리에 추가하고 입력창 비우기
                history.append([message, None])
                return "", history
            
            def bot_response(history):
                if not history or not history[-1][0]:
                    return history
                # 마지막 사용자 메시지로 봇 응답 생성
                user_message = history[-1][0]
                return self.chat_with_agent(user_message, history)
            
            # 이벤트 연결
            msg.submit(
                add_message, 
                [msg, chatbot], 
                [msg, chatbot], 
                queue=False
            ).then(
                bot_response, 
                [chatbot], 
                [chatbot]
            )
            
            send_btn.click(
                add_message, 
                [msg, chatbot], 
                [msg, chatbot], 
                queue=False
            ).then(
                bot_response, 
                [chatbot], 
                [chatbot]
            )
            
            clear_btn.click(
                self.clear_chat,
                outputs=[msg, chatbot]
            )
            
            stats_btn.click(
                self.get_performance_stats,
                outputs=[stats_display]
            )
            
            # 예시 질문 버튼들
            gr.Markdown("### 🎯 빠른 질문 예시")
            with gr.Row():
                example_btn1 = gr.Button("🚨 브레이크가 안 밟혀요", size="sm")
                example_btn2 = gr.Button("🔧 엔진 오일 교체 주기", size="sm")
                example_btn3 = gr.Button("❄️ 겨울철 차량 관리", size="sm")
            
            # 예시 질문 이벤트
            example_btn1.click(
                lambda: "갑자기 브레이크가 안 밟혀요",
                outputs=[msg]
            )
            example_btn2.click(
                lambda: "엔진 오일 교체 주기가 언제인가요?",
                outputs=[msg]
            )
            example_btn3.click(
                lambda: "겨울철 차량 관리 방법을 알려주세요",
                outputs=[msg]
            )
        
        return interface


def run_gradio_interface(agent, callbacks, port=7860):
    """Gradio 웹 인터페이스 실행"""
    if not GRADIO_AVAILABLE:
        print("❌ Gradio가 설치되지 않았습니다.")
        print("📦 설치 방법: pip install gradio")
        return
    
    print("🚀 Gradio 웹 인터페이스 시작 중...")
    
    try:
        # 챗봇 인스턴스 생성
        chatbot = GradioVehicleChatbot(agent, callbacks)
        
        # Gradio 인터페이스 생성
        interface = chatbot.create_interface()
        
        # 서버 실행
        print("✅ 웹 서버 준비 완료!")
        print(f"🌐 웹 브라우저에서 http://localhost:{port} 으로 접속하세요")
        print("🛑 서버를 중지하려면 Ctrl+C를 누르세요")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Gradio 서버 실행 실패: {str(e)}")
        raise e


def run_terminal_interface(agent, callbacks):
    """터미널 기반 인터페이스 실행"""
    print("\n" + "=" * 60)
    print("💬 지능형 차량 어시스턴트 - SubGraph 대화형 모드")
    print("=" * 60)
    print("🚗 차량에 관한 모든 질문을 자유롭게 해주세요!")
    print("📝 일반 질문부터 🚨 응급 상황까지 즉시 대응합니다.")
    print("🔧 SubGraph 아키텍처로 더욱 안정적이고 확장 가능합니다.")
    print("")
    print("💡 사용법:")
    print("   • 질문 입력 후 Enter")
    print("   • 'stats' - 성능 통계 확인")
    print("   • 'quit' 또는 'exit' - 종료")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n❓ 질문: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("👋 시스템을 종료합니다.")
                break
            
            if user_input.lower() == 'stats':
                performance_handler = callbacks[0]
                alert_handler = callbacks[2]
                performance_handler.print_performance_report()
                usage = alert_handler.get_usage_summary()
                print(f"\n💰 현재 세션 사용량:")
                print(f"   토큰: {usage['tokens_used']:,}개 / {usage['token_limit']:,}개 ({usage['token_usage_percentage']:.1f}%)")
                print(f"   비용: ${usage['cost_incurred']:.4f} / ${usage['cost_limit']:.2f} ({usage['cost_usage_percentage']:.1f}%)")
                continue
            
            if not user_input:
                print("질문을 입력해주세요.")
                continue
            
            print("-" * 50)
            
            # 새로운 쿼리를 위해 콜백 핸들러 세션 초기화
            for callback in callbacks:
                if hasattr(callback, 'reset_session'):
                    callback.reset_session()
            
            # 콜백과 함께 쿼리 실행 (SubGraph 아키텍처)
            answer = agent.query(user_input, callbacks=callbacks)
            
            print(f"\n💡 답변:\n{answer}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            continue


def main():
    """메인 실행 함수 - SubGraph 아키텍처"""
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(
        description="🚗 지능형 차량 어시스턴트 - SubGraph 아키텍처",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                    # 터미널 인터페이스 (기본)
  python main.py --gradio           # Gradio 웹 인터페이스
  python main.py --help             # 도움말 표시
        """
    )
    
    parser.add_argument(
        '--gradio', 
        action='store_true', 
        help='Gradio 웹 인터페이스로 실행 (기본: 터미널 인터페이스)'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=7860, 
        help='Gradio 서버 포트 (기본: 7860)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚗 지능형 차량 어시스턴트 - SubGraph 아키텍처")
    print("=" * 60)
    print("👨‍💼 운전자를 위한 실시간 차량 매뉴얼 AI 도우미")
    print("🚨 응급 상황 자동 감지 및 즉시 대응 시스템")
    print("🔍 하이브리드 검색, 쿼리 확장, 재순위화, 맥락 압축 지원")
    print("🤖 Few-shot 프롬프팅으로 향상된 답변 품질")
    print("📊 실시간 성능 모니터링 및 알림 지원")
    print("🔧 SubGraph 아키텍처로 모듈화 및 재사용성 향상")
    print("=" * 60)
    
    # 콜백 핸들러 초기화
    performance_handler = PerformanceMonitoringHandler(enable_detailed_logging=False)
    notification_handler = RealTimeNotificationHandler(enable_progress_bar=False, enable_notifications=False)
    alert_handler = AlertHandler(token_limit=50000, cost_limit=5.0)  # 토큰 50K, 비용 $5 제한
    
    callbacks = [performance_handler, notification_handler, alert_handler]
    
    def signal_handler(sig, frame):
        """Ctrl+C 처리를 위한 시그널 핸들러"""
        print("\n\n🛑 시스템 종료 중...")
        performance_handler.print_performance_report()
        usage = alert_handler.get_usage_summary()
        print(f"\n📊 세션 사용량:")
        print(f"   토큰: {usage['tokens_used']:,}개 ({usage['token_usage_percentage']:.1f}%)")
        print(f"   비용: ${usage['cost_incurred']:.4f} ({usage['cost_usage_percentage']:.1f}%)")
        print("👋 안전하게 종료되었습니다.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # PDF 파일 경로 설정
        pdf_path = str(DEFAULT_PDF_PATH)
        print(f"📄 PDF 파일: {Path(pdf_path).name}")
        
        # 에이전트 초기화 (SubGraph 아키텍처)
        print("\n🔧 SubGraph 시스템 초기화 중...")
        agent = VehicleManualAgent(pdf_path)
        print("✅ SubGraph 시스템 준비 완료!")
        print("📊 성능 모니터링 활성화")
        print("🔔 실시간 알림 활성화")
        print("🔧 SubGraph 모듈화 완료")
        
        # 인터페이스 선택
        if args.gradio:
            print(f"\n🌐 Gradio 웹 인터페이스 모드 (포트: {args.port})")
            run_gradio_interface(agent, callbacks, args.port)
        else:
            print(f"\n💻 터미널 인터페이스 모드")
            run_terminal_interface(agent, callbacks)
    
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {str(e)}")
        print("📋 해결 방법:")
        print("1. .env 파일에 OPENAI_API_KEY가 설정되어 있는지 확인")
        print("2. PDF 파일이 올바른 경로에 있는지 확인")
        print("3. 필요한 Python 패키지들이 설치되어 있는지 확인")
        print("4. SubGraph 모듈들이 올바르게 초기화되었는지 확인")
        if args.gradio:
            print("5. Gradio가 설치되어 있는지 확인: pip install gradio")


if __name__ == "__main__":
    main()
