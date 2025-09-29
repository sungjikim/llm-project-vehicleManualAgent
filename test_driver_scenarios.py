"""
운전자 실제 상황 테스트 스크립트

차량 내에서 운전자가 실제로 궁금해할 수 있는 상황들을 기반으로 한 
자동 테스트를 실행합니다.
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


def run_driver_scenario_tests():
    """운전자 실제 상황 테스트 실행"""
    print("=" * 60)
    print("🚗 운전자 실제 상황 자동 테스트")
    print("=" * 60)
    print("👨‍💼 운전자가 실제로 궁금해할 수 있는 상황들을 테스트합니다")
    print("📝 일반 질문 5개 + 🚨 응급 상황 5개")
    print("=" * 60)
    
    # 콜백 핸들러 초기화
    performance_handler = PerformanceMonitoringHandler(enable_detailed_logging=True)
    notification_handler = RealTimeNotificationHandler(enable_progress_bar=True, enable_notifications=True)
    alert_handler = AlertHandler(token_limit=50000, cost_limit=5.0)
    
    callbacks = [performance_handler, notification_handler, alert_handler]
    
    def signal_handler(sig, frame):
        """Ctrl+C 처리를 위한 시그널 핸들러"""
        print("\n\n🛑 테스트 중단됨...")
        performance_handler.print_performance_report()
        usage = alert_handler.get_usage_summary()
        print(f"\n📊 현재까지 사용량:")
        print(f"   토큰: {usage['tokens_used']:,}개")
        print(f"   비용: ${usage['cost_incurred']:.4f}")
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
        
        # 차량 내 운전자가 궁금해할 수 있는 실제 상황들
        test_queries = [
            # === 일반 질문 (5개) ===
            "운전석 시트를 내 체형에 맞게 조정하는 방법이 궁금해요",
            "겨울철 히터를 효율적으로 사용하는 방법을 알려주세요",
            "후방 카메라 화면이 흐릿한데 청소는 어떻게 하나요?",
            "연비를 좋게 하려면 어떤 운전 습관을 가져야 할까요?",
            "블루투스로 스마트폰 음악을 들으려면 어떻게 연결하나요?",
            
            # === 응급 상황 (5개) ===
            "주행 중 갑자기 엔진 경고등이 빨갛게 켜졌어요! 어떻게 해야 하나요?",
            "고속도로에서 타이어가 터진 것 같은데 안전하게 대처하는 방법은?",
            "브레이크 페달을 밟는데 바닥까지 들어가요! 급한 상황인가요?",
            "운전 중 시동이 꺼지면서 파워 스티어링이 무거워졌어요!",
            "차 안에 가스 냄새가 나는데 즉시 해야 할 조치가 뭔가요?"
        ]
        
        print(f"\n🧪 운전자 실제 상황 테스트 ({len(test_queries)}개 질문)")
        print("📝 일반 질문 5개 + 🚨 응급 상황 5개")
        print("=" * 60)
        
        total_start_time = time.time()
        
        for i, query in enumerate(test_queries, 1):
            # 질문 유형 구분
            if i <= 5:
                question_type = "📝 일반 질문"
                question_icon = "📝"
            else:
                question_type = "🚨 응급 상황"
                question_icon = "🚨"
            
            print(f"\n[테스트 {i}/{len(test_queries)}] {question_type}")
            print(f"{question_icon} 질문: {query}")
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
        
        # 전체 테스트 시간 계산
        total_elapsed = time.time() - total_start_time
        
        print("\n🎉 모든 테스트 완료!")
        print(f"⏱️  전체 소요 시간: {total_elapsed:.2f}초")
        
        # 테스트 완료 후 성능 리포트 출력
        performance_handler.print_performance_report()
        
        # 최종 사용량 리포트
        usage = alert_handler.get_usage_summary()
        print(f"\n💰 최종 사용량:")
        print(f"   토큰: {usage['tokens_used']:,}개 ({usage['token_usage_percentage']:.1f}%)")
        print(f"   비용: ${usage['cost_incurred']:.4f} ({usage['cost_usage_percentage']:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실행 실패: {str(e)}")
        print("📋 해결 방법:")
        print("1. .env 파일에 OPENAI_API_KEY가 설정되어 있는지 확인")
        print("2. PDF 파일이 올바른 경로에 있는지 확인")
        print("3. 필요한 Python 패키지들이 설치되어 있는지 확인")
        return False


def run_quick_driver_test():
    """빠른 운전자 상황 테스트 (3개 질문만)"""
    print("🚗 빠른 운전자 상황 테스트")
    print("=" * 40)
    
    try:
        agent = VehicleManualAgent(str(DEFAULT_PDF_PATH))
        
        # 선별된 테스트 케이스 (각 유형별 1-2개)
        quick_tests = [
            ("📝 일반", "운전석 시트를 내 체형에 맞게 조정하는 방법이 궁금해요"),
            ("🚨 응급", "브레이크 페달을 밟는데 바닥까지 들어가요! 급한 상황인가요?"),
            ("📝 일반", "블루투스로 스마트폰 음악을 들으려면 어떻게 연결하나요?")
        ]
        
        total_start_time = time.time()
        
        for i, (q_type, question) in enumerate(quick_tests, 1):
            print(f"\n[테스트 {i}] {q_type}")
            print(f"질문: {question}")
            print("-" * 40)
            
            start_time = time.time()
            answer = agent.query(question)
            elapsed_time = time.time() - start_time
            
            # 답변 미리보기 (첫 150자)
            preview = answer[:150] + "..." if len(answer) > 150 else answer
            print(f"답변: {preview}")
            print(f"⏱️  소요시간: {elapsed_time:.2f}초")
            
            # 헤더 확인
            first_line = answer.split('\n')[0] if answer else ''
            if '응급 상황' in first_line:
                print("✅ 응급 상황으로 감지됨")
            elif '일반 질문' in first_line:
                print("✅ 일반 질문으로 분류됨")
        
        total_elapsed = time.time() - total_start_time
        print(f"\n⏱️  전체 소요 시간: {total_elapsed:.2f}초")
        print("🎉 빠른 테스트 완료!")
        
        return True
        
    except Exception as e:
        print(f"❌ 빠른 테스트 실패: {str(e)}")
        return False


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="운전자 실제 상황 테스트")
    parser.add_argument(
        "--mode", 
        choices=["full", "quick"],
        default="full",
        help="테스트 모드: full(전체 10개) 또는 quick(빠른 3개)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        success = run_quick_driver_test()
    else:
        success = run_driver_scenario_tests()
    
    if success:
        print("\n✅ 테스트 성공적으로 완료!")
    else:
        print("\n❌ 테스트 실패!")
        sys.exit(1)


if __name__ == "__main__":
    main()
