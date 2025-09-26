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
    print("🚗 차량 매뉴얼 RAG 시스템")
    print("=" * 60)
    print("📚 LangChain + LangGraph 기반 모듈화된 RAG 에이전트")
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
        
        # 테스트 쿼리들
        test_queries = [
            "타이어 공기압은 얼마로 맞춰야 하나요?",
            "브레이크 경고등이 켜졌어요. 어떻게 해야 하나요?",
            "엔진 오일 교체 주기는 언제인가요?",
            "XC60의 연료 탱크 용량은?",
            "겨울철 타이어 관리 방법을 알려주세요"
        ]
        
        print(f"\n🧪 {len(test_queries)}개 테스트 쿼리 실행 중...")
        print("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[테스트 {i}/{len(test_queries)}]")
            print(f"❓ 질문: {query}")
            print("-" * 50)
            
            try:
                # 시작 시간 기록
                start_time = time.time()
                
                # 콜백과 함께 쿼리 실행
                answer = agent.query(query, callbacks=callbacks)
                
                # 소요 시간 계산
                elapsed_time = time.time() - start_time
                
                # 결과 출력
                print(f"\n💡 답변:\n{answer}")
                print(f"\n⏱️  소요 시간: {elapsed_time:.2f}초")
                
                # 현재 세션 통계 출력
                stats = performance_handler.get_performance_summary()
                print(f"📊 누적 통계: {stats['total_queries']}개 쿼리, "
                      f"{stats['total_tokens_used']:,} 토큰, "
                      f"${stats['total_cost']:.4f}")
                
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")
            
            print("=" * 60)
            
            # 다음 테스트 전 잠시 대기
            if i < len(test_queries):
                print("다음 테스트 준비 중...")
                time.sleep(2)
        
        print("\n🎉 모든 테스트 완료!")
        
        # 테스트 완료 후 성능 리포트 출력
        performance_handler.print_performance_report()
        
        # 대화형 모드 안내
        print("\n" + "=" * 60)
        print("💬 대화형 모드")
        print("=" * 60)
        print("직접 질문을 입력하세요 (종료: 'quit' 또는 'exit')")
        print("💡 팁: 'stats'를 입력하면 현재 성능 통계를 확인할 수 있습니다.")
        
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
                start_time = time.time()
                
                # 콜백과 함께 쿼리 실행
                answer = agent.query(user_input, callbacks=callbacks)
                
                elapsed_time = time.time() - start_time
                
                print(f"\n💡 답변:\n{answer}")
                print(f"\n⏱️  소요 시간: {elapsed_time:.2f}초")
                
                # 간단한 통계 출력
                stats = performance_handler.get_performance_summary()
                print(f"📊 세션 통계: {stats['total_queries']}개 쿼리, "
                      f"평균 {stats['average_response_time']:.2f}초")
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
