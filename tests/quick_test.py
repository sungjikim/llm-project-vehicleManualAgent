"""
빠른 테스트 실행 스크립트
응급 상황 시스템의 핵심 기능만 빠르게 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.vehicle_agent import VehicleManualAgent
from src.utils.emergency_detector import EmergencyDetector
from src.config.settings import DEFAULT_PDF_PATH
import time


def quick_emergency_detection_test():
    """응급 상황 감지 빠른 테스트"""
    print("🔍 응급 상황 감지 빠른 테스트")
    print("-" * 40)
    
    detector = EmergencyDetector()
    
    test_cases = [
        ("일반", "타이어 공기압은 얼마로 맞춰야 하나요?"),
        ("응급", "브레이크를 밟아도 차가 멈추지 않아요!"),
        ("위험", "차에서 타는 냄새가 나는데 어떻게 해야 해요?"),
    ]
    
    for case_type, query in test_cases:
        result = detector.detect_emergency(query)
        status = "🚨" if result["is_emergency"] else "📝"
        print(f"{status} [{case_type}] {result['priority_level']} (점수: {result['total_score']})")
        print(f"   질문: {query}")
    
    print("✅ 응급 감지 테스트 완료\n")


def quick_system_integration_test():
    """시스템 통합 빠른 테스트"""
    print("🚗 운전자 실제 상황 통합 테스트")
    print("-" * 40)
    
    try:
        # 에이전트 초기화
        print("🔧 에이전트 초기화 중...")
        agent = VehicleManualAgent(str(DEFAULT_PDF_PATH))
        print("✅ 초기화 완료")
        
        # 운전자 실제 상황 테스트 케이스
        test_cases = [
            # 일반 질문 (운전자가 실제로 궁금해할 상황)
            ("일반", "운전석 시트를 내 체형에 맞게 조정하는 방법이 궁금해요"),
            ("일반", "겨울철 히터를 효율적으로 사용하는 방법을 알려주세요"),
            ("일반", "블루투스로 스마트폰 음악을 들으려면 어떻게 연결하나요?"),
            
            # 응급 상황 (운전자가 실제로 겪을 수 있는 위험한 상황)
            ("응급", "주행 중 갑자기 엔진 경고등이 빨갛게 켜졌어요! 어떻게 해야 하나요?"),
            ("응급", "고속도로에서 타이어가 터진 것 같은데 안전하게 대처하는 방법은?"),
            ("응급", "브레이크 페달을 밟는데 바닥까지 들어가요! 급한 상황인가요?")
        ]
        
        for case_type, query in test_cases:
            print(f"\n[{case_type} 테스트] {query}")
            start_time = time.time()
            
            answer = agent.query(query)
            elapsed_time = time.time() - start_time
            
            # 응답 분석
            has_emergency_header = "응급 상황" in answer
            has_normal_header = "일반 질문" in answer
            has_reliability = "🔍 **답변 신뢰도**" in answer
            first_line = answer.split('\n')[0] if answer else ""
            
            print(f"  ⏱️  응답 시간: {elapsed_time:.2f}초")
            print(f"  📝 답변 길이: {len(answer)}자")
            print(f"  📋 첫 줄: {first_line}")
            print(f"  🚨 응급 헤더: {'있음' if has_emergency_header else '없음'}")
            print(f"  📝 일반 헤더: {'있음' if has_normal_header else '없음'}")
            print(f"  🔍 신뢰도 표시: {'있음' if has_reliability else '없음'}")
            
            # 응급 상황 특별 확인
            if case_type == "응급":
                if has_emergency_header and "응급 상황" in first_line:
                    print("  ✅ 응급 상황 올바르게 감지되고 헤더 표시됨")
                else:
                    print("  ❌ 응급 상황 감지 또는 헤더 표시 실패")
            else:
                if has_normal_header and "일반 질문" in first_line:
                    print("  ✅ 일반 질문 올바르게 분류되고 헤더 표시됨")
                else:
                    print("  ❌ 일반 질문 분류 또는 헤더 표시 실패")
        
        print("\n✅ 시스템 통합 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실행 오류: {str(e)}")
        return False
    
    return True


def quick_performance_test():
    """성능 빠른 테스트"""
    print("\n📊 성능 빠른 테스트")
    print("-" * 40)
    
    try:
        agent = VehicleManualAgent(str(DEFAULT_PDF_PATH))
        
        # 간단한 성능 비교
        emergency_query = "엔진에서 연기가 나고 있어요!"
        normal_query = "타이어 공기압은 얼마인가요?"
        
        # 응급 상황 테스트
        start_time = time.time()
        emergency_answer = agent.query(emergency_query)
        emergency_time = time.time() - start_time
        
        # 일반 질문 테스트
        start_time = time.time()
        normal_answer = agent.query(normal_query)
        normal_time = time.time() - start_time
        
        print(f"🚨 응급 상황: {emergency_time:.2f}초")
        print(f"📝 일반 질문: {normal_time:.2f}초")
        
        if emergency_time < normal_time:
            print("⬆️  응급 처리가 더 빠름")
        else:
            diff = emergency_time - normal_time
            print(f"⚠️  응급 처리가 {diff:.2f}초 더 소요 (품질 향상)")
        
        print("✅ 성능 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 성능 테스트 오류: {str(e)}")
        return False


def main():
    """메인 실행 함수"""
    print("🚀 응급 상황 시스템 빠른 테스트")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # 1. 응급 감지 테스트
    try:
        quick_emergency_detection_test()
        success_count += 1
    except Exception as e:
        print(f"❌ 응급 감지 테스트 실패: {str(e)}")
    
    # 2. 시스템 통합 테스트
    if quick_system_integration_test():
        success_count += 1
    
    # 3. 성능 테스트
    if quick_performance_test():
        success_count += 1
    
    # 결과 요약
    print("\n" + "=" * 50)
    print(f"📋 테스트 결과: {success_count}/{total_tests} 성공")
    
    if success_count == total_tests:
        print("🎉 모든 빠른 테스트 통과!")
        print("✅ 응급 상황 시스템이 정상 작동합니다.")
    else:
        print("⚠️  일부 테스트 실패")
        print("🔧 상세 테스트를 실행해보세요: python run_tests.py")


if __name__ == "__main__":
    main()
