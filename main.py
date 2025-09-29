"""
차량 매뉴얼 RAG 시스템 - 메인 진입점

모듈화된 LangChain/LangGraph 기반 차량 매뉴얼 RAG 에이전트
"""

import time
import signal
import sys
from pathlib import Path

from src.agents.vehicle_agent import VehicleManualAgent
from src.config.settings import DEFAULT_PDF_PATH
from src.utils.callback_handlers import (
    PerformanceMonitoringHandler,
    RealTimeNotificationHandler,
    AlertHandler
)


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🚗 지능형 차량 어시스턴트")
    print("=" * 60)
    print("👨‍💼 운전자를 위한 실시간 차량 매뉴얼 AI 도우미")
    print("🚨 응급 상황 자동 감지 및 즉시 대응 시스템")
    print("🔍 하이브리드 검색, 쿼리 확장, 재순위화, 맥락 압축 지원")
    print("🤖 Few-shot 프롬프팅으로 향상된 답변 품질")
    print("📊 실시간 성능 모니터링 및 알림 지원")
    print("=" * 60)
    
    # 콜백 핸들러 초기화
    performance_handler = PerformanceMonitoringHandler(enable_detailed_logging=True)
    notification_handler = RealTimeNotificationHandler(enable_progress_bar=True, enable_notifications=True)
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
        
        # 에이전트 초기화
        print("\n🔧 시스템 초기화 중...")
        agent = VehicleManualAgent(pdf_path)
        print("✅ 시스템 준비 완료!")
        print("📊 성능 모니터링 활성화")
        print("🔔 실시간 알림 활성화")
        
        
        # 대화형 모드 안내
        print("\n" + "=" * 60)
        print("💬 지능형 차량 어시스턴트 - 대화형 모드")
        print("=" * 60)
        print("🚗 차량에 관한 모든 질문을 자유롭게 해주세요!")
        print("📝 일반 질문부터 🚨 응급 상황까지 즉시 대응합니다.")
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
                
                # 콜백과 함께 쿼리 실행
                answer = agent.query(user_input, callbacks=callbacks)
                
                print(f"\n💡 답변:\n{answer}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")
                continue
    
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {str(e)}")
        print("📋 해결 방법:")
        print("1. .env 파일에 OPENAI_API_KEY가 설정되어 있는지 확인")
        print("2. PDF 파일이 올바른 경로에 있는지 확인")
        print("3. 필요한 Python 패키지들이 설치되어 있는지 확인")


if __name__ == "__main__":
    main()
