"""
응급 상황 시스템 테스트 실행 스크립트
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.test_emergency_system import run_emergency_tests
from tests.test_performance_benchmark import run_performance_tests


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="응급 상황 시스템 테스트 실행")
    parser.add_argument(
        "--test-type", 
        choices=["emergency", "performance", "all"],
        default="all",
        help="실행할 테스트 타입 (기본값: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 출력 모드"
    )
    
    args = parser.parse_args()
    
    print("🚗 차량 매뉴얼 RAG 시스템 테스트 실행")
    print("=" * 60)
    print(f"📋 테스트 타입: {args.test_type}")
    print(f"📊 상세 모드: {'ON' if args.verbose else 'OFF'}")
    print("=" * 60)
    
    success = True
    
    if args.test_type in ["emergency", "all"]:
        print("\n🚨 응급 상황 시스템 테스트 시작")
        print("-" * 40)
        try:
            result = run_emergency_tests()
            if not result.wasSuccessful():
                success = False
                print(f"❌ 응급 시스템 테스트 실패: {len(result.failures)} 실패, {len(result.errors)} 오류")
        except Exception as e:
            success = False
            print(f"❌ 응급 시스템 테스트 실행 오류: {str(e)}")
    
    if args.test_type in ["performance", "all"]:
        print("\n📊 성능 벤치마크 테스트 시작")
        print("-" * 40)
        try:
            report = run_performance_tests()
            if report is None:
                success = False
                print("❌ 성능 벤치마크 테스트 실패")
            else:
                print("✅ 성능 벤치마크 테스트 완료")
        except Exception as e:
            success = False
            print(f"❌ 성능 벤치마크 테스트 실행 오류: {str(e)}")
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 모든 테스트 완료!")
        print("✅ 응급 상황 최적화 시스템이 정상적으로 작동합니다.")
    else:
        print("❌ 일부 테스트 실패")
        print("🔧 시스템 점검이 필요합니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
