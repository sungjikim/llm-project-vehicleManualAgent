#!/usr/bin/env python3
"""
차량 관련 테스트 시나리오 50개
"""

import sys
import os
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

# 테스트 시나리오 데이터
TEST_SCENARIOS = {
    "driving_situations": [
        # 운전자가 차량에서 겪을 수 있는 상황 20개
        "갑자기 브레이크가 안 밟혀요",
        "고속도로에서 타이어가 터졌어요",
        "엔진에서 연기가 나고 있어요",
        "계기판에 빨간 경고등이 켜졌어요",
        "핸들이 한쪽으로 계속 쏠려요",
        "차가 시동이 안 걸려요",
        "주행 중 갑자기 엔진이 꺼졌어요",
        "에어컨이 갑자기 안 나와요",
        "와이퍼가 작동하지 않아요",
        "헤드라이트가 한쪽만 켜져요",
        "차에서 이상한 소음이 나요",
        "기어가 안 들어가요",
        "주차 브레이크가 안 풀려요",
        "냉각수 온도가 너무 높아요",
        "배터리 방전으로 시동이 안 걸려요",
        "연료가 부족한데 주유소가 안 보여요",
        "안전벨트가 잠기지 않아요",
        "문이 잠기지 않아요",
        "후진할 때 후방 카메라가 안 보여요",
        "차량 키를 차 안에 두고 나왔어요"
    ],
    
    "maintenance_schedule": [
        # 운전자가 궁금할 만한 차량 관련 교체주기 10개
        "엔진 오일 교체 주기가 언제인가요?",
        "타이어는 언제 교체해야 하나요?",
        "브레이크 패드 교체 시기를 알려주세요",
        "에어 필터는 얼마나 자주 바꿔야 하나요?",
        "배터리 교체 주기는 어떻게 되나요?",
        "냉각수는 언제 교체하나요?",
        "점화플러그 교체 주기를 알려주세요",
        "변속기 오일은 언제 갈아야 하나요?",
        "와이퍼 블레이드 교체 시기는요?",
        "연료 필터 교체 주기가 궁금해요"
    ],
    
    "general_questions": [
        # 그 외 차량 관련 질문 20개
        "겨울철 차량 관리 방법을 알려주세요",
        "연비를 향상시키는 운전 방법은?",
        "차량 정기점검은 어떤 항목을 확인하나요?",
        "장거리 운전 전 체크사항이 있나요?",
        "차량 보관 시 주의사항을 알려주세요",
        "타이어 공기압은 얼마나 넣어야 하나요?",
        "차량 세차는 어떻게 해야 하나요?",
        "블루투스 연결 방법을 알려주세요",
        "내비게이션 업데이트는 어떻게 하나요?",
        "차량 보증기간은 얼마나 되나요?",
        "차량 리콜 확인 방법을 알려주세요",
        "주차 보조 시스템 사용법이 궁금해요",
        "자동 주차 기능은 어떻게 쓰나요?",
        "차선 유지 보조 시스템이 뭔가요?",
        "어댑티브 크루즈 컨트롤 사용법을 알려주세요",
        "차량 도난 방지 시스템은 어떻게 작동하나요?",
        "차량 원격 시동 방법을 알려주세요",
        "스마트 키 배터리 교체 방법은?",
        "차량 앱으로 할 수 있는 기능들이 뭔가요?",
        "차량 소음을 줄이는 방법이 있나요?"
    ]
}

def run_comprehensive_test():
    """포괄적인 차량 테스트 실행"""
    
    print("🚗 차량 관련 종합 테스트 시작")
    print("=" * 80)
    
    try:
        from src.agents.vehicle_agent import VehicleManualAgent
        from src.config.settings import DEFAULT_PDF_PATH
        from src.utils.callback_handlers import (
            PerformanceMonitoringHandler,
            RealTimeNotificationHandler
        )
        
        # 에이전트 초기화
        print("🔧 에이전트 초기화 중...")
        agent = VehicleManualAgent(str(DEFAULT_PDF_PATH))
        print("✅ 에이전트 초기화 완료\n")
        
        # 콜백 핸들러 생성 (간단한 로깅만)
        performance_handler = PerformanceMonitoringHandler(enable_detailed_logging=False)
        notification_handler = RealTimeNotificationHandler(
            enable_progress_bar=False, 
            enable_notifications=False  # 알림 비활성화로 깔끔한 테스트
        )
        callbacks = [performance_handler, notification_handler]
        
        # 테스트 결과 저장
        test_results = {
            "driving_situations": [],
            "maintenance_schedule": [],
            "general_questions": [],
            "summary": {
                "total_tests": 0,
                "successful_tests": 0,
                "failed_tests": 0,
                "emergency_detected": 0,
                "driving_context_detected": 0,
                "high_confidence_answers": 0,
                "low_confidence_answers": 0
            }
        }
        
        # 각 카테고리별 테스트 실행
        for category, questions in TEST_SCENARIOS.items():
            print(f"\n📋 {category.upper()} 테스트 ({len(questions)}개)")
            print("-" * 60)
            
            category_results = []
            
            for i, question in enumerate(questions, 1):
                try:
                    print(f"{i:2d}. {question}")
                    
                    # 새로운 쿼리를 위해 콜백 핸들러 세션 초기화
                    for callback in callbacks:
                        if hasattr(callback, 'reset_session'):
                            callback.reset_session()
                    
                    start_time = time.time()
                    answer = agent.query(question, callbacks=callbacks)
                    end_time = time.time()
                    
                    # 답변 분석
                    result = analyze_answer(question, answer, end_time - start_time)
                    category_results.append(result)
                    
                    # 결과 출력 (간단히)
                    status_icon = "✅" if result["success"] else "❌"
                    confidence_text = f"{result['confidence']:.0f}%" if result['confidence'] else "N/A"
                    emergency_text = "🚨" if result["is_emergency"] else ""
                    driving_text = "🚗" if result["is_driving"] else ""
                    
                    print(f"    {status_icon} 신뢰도: {confidence_text} {emergency_text} {driving_text} ({result['response_time']:.1f}초)")
                    
                    test_results["summary"]["total_tests"] += 1
                    if result["success"]:
                        test_results["summary"]["successful_tests"] += 1
                    else:
                        test_results["summary"]["failed_tests"] += 1
                    
                    if result["is_emergency"]:
                        test_results["summary"]["emergency_detected"] += 1
                    if result["is_driving"]:
                        test_results["summary"]["driving_context_detected"] += 1
                    if result["confidence"] and result["confidence"] >= 80:
                        test_results["summary"]["high_confidence_answers"] += 1
                    elif result["confidence"] and result["confidence"] < 60:
                        test_results["summary"]["low_confidence_answers"] += 1
                    
                    # 짧은 대기 (API 레이트 제한 방지)
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"    ❌ 오류: {str(e)}")
                    category_results.append({
                        "question": question,
                        "success": False,
                        "error": str(e),
                        "confidence": None,
                        "is_emergency": False,
                        "is_driving": False,
                        "response_time": 0
                    })
                    test_results["summary"]["failed_tests"] += 1
                    test_results["summary"]["total_tests"] += 1
            
            test_results[category] = category_results
        
        # 최종 결과 출력
        print_test_summary(test_results)
        
        # 개선점 분석 및 제안
        analyze_and_suggest_improvements(test_results)
        
        return test_results
        
    except Exception as e:
        print(f"❌ 테스트 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_answer(question, answer, response_time):
    """답변 분석"""
    
    result = {
        "question": question,
        "success": True,
        "confidence": None,
        "is_emergency": False,
        "is_driving": False,
        "response_time": response_time,
        "answer_length": len(answer),
        "has_page_reference": False
    }
    
    try:
        # 신뢰도 추출
        import re
        confidence_match = re.search(r'답변 신뢰도\*\*:\s*(\d+(?:\.\d+)?)%', answer)
        if confidence_match:
            result["confidence"] = float(confidence_match.group(1))
        
        # 응급 상황 감지 확인
        emergency_indicators = ["🚨", "CRITICAL", "HIGH", "응급", "즉시", "위험"]
        result["is_emergency"] = any(indicator in answer for indicator in emergency_indicators)
        
        # 주행 중 감지 확인
        driving_indicators = ["🚗", "주행 중", "운전 중", "안전한 곳에 정차", "압축"]
        result["is_driving"] = any(indicator in answer for indicator in driving_indicators)
        
        # 페이지 참조 확인
        result["has_page_reference"] = "참고 페이지" in answer or re.search(r'페이지\s*\d+', answer)
        
        # 답변 품질 확인
        if len(answer) < 50:
            result["success"] = False
        elif "오류" in answer or "실패" in answer:
            result["success"] = False
            
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
    
    return result

def print_test_summary(test_results):
    """테스트 요약 출력"""
    
    summary = test_results["summary"]
    
    print(f"\n{'='*80}")
    print("📊 테스트 결과 요약")
    print(f"{'='*80}")
    
    print(f"📈 전체 통계:")
    print(f"   • 총 테스트: {summary['total_tests']}개")
    print(f"   • 성공: {summary['successful_tests']}개 ({summary['successful_tests']/summary['total_tests']*100:.1f}%)")
    print(f"   • 실패: {summary['failed_tests']}개 ({summary['failed_tests']/summary['total_tests']*100:.1f}%)")
    
    print(f"\n🎯 특별 기능 감지:")
    print(f"   • 응급 상황 감지: {summary['emergency_detected']}개")
    print(f"   • 주행 중 감지: {summary['driving_context_detected']}개")
    
    print(f"\n📊 답변 신뢰도:")
    print(f"   • 높은 신뢰도 (80% 이상): {summary['high_confidence_answers']}개")
    print(f"   • 낮은 신뢰도 (60% 미만): {summary['low_confidence_answers']}개")
    
    # 카테고리별 성공률
    print(f"\n📋 카테고리별 성공률:")
    for category, results in test_results.items():
        if category != "summary" and results:
            successful = sum(1 for r in results if r["success"])
            total = len(results)
            print(f"   • {category}: {successful}/{total} ({successful/total*100:.1f}%)")

def analyze_and_suggest_improvements(test_results):
    """개선점 분석 및 제안"""
    
    print(f"\n{'='*80}")
    print("💡 개선점 분석")
    print(f"{'='*80}")
    
    issues = []
    suggestions = []
    
    summary = test_results["summary"]
    
    # 성공률 분석
    success_rate = summary['successful_tests'] / summary['total_tests'] * 100
    if success_rate < 90:
        issues.append(f"전체 성공률이 {success_rate:.1f}%로 낮음")
        suggestions.append("검색 알고리즘 및 답변 생성 로직 개선 필요")
    
    # 신뢰도 분석
    low_confidence_rate = summary['low_confidence_answers'] / summary['total_tests'] * 100
    if low_confidence_rate > 20:
        issues.append(f"낮은 신뢰도 답변이 {low_confidence_rate:.1f}%")
        suggestions.append("문서 검색 품질 향상 및 평가 기준 조정 필요")
    
    # 응급 상황 감지 분석
    emergency_questions = len(test_results["driving_situations"])
    emergency_detected = summary['emergency_detected']
    if emergency_detected < emergency_questions * 0.7:  # 70% 이하 감지
        issues.append(f"응급 상황 감지율이 낮음 ({emergency_detected}/{emergency_questions})")
        suggestions.append("응급 상황 감지 키워드 및 로직 개선 필요")
    
    # 카테고리별 성능 분석
    for category, results in test_results.items():
        if category != "summary" and results:
            successful = sum(1 for r in results if r["success"])
            total = len(results)
            success_rate = successful / total * 100
            
            if success_rate < 80:
                issues.append(f"{category} 카테고리 성공률이 {success_rate:.1f}%로 낮음")
                suggestions.append(f"{category} 관련 문서 보강 또는 검색 전략 개선 필요")
    
    # 결과 출력
    if issues:
        print("🔍 발견된 문제점:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n💡 개선 제안:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
    else:
        print("✅ 모든 테스트가 양호한 성능을 보입니다!")
    
    # 구체적인 개선 작업 제안
    print(f"\n🔧 구체적인 개선 작업:")
    print("   1. 검색 키워드 확장 및 동의어 사전 보강")
    print("   2. 응급 상황 감지 패턴 확장")
    print("   3. 답변 신뢰도 평가 기준 재조정")
    print("   4. 문서 청킹 전략 최적화")
    print("   5. 주행 중 감지 정확도 향상")

def run_quick_test(num_samples=10):
    """빠른 샘플 테스트"""
    
    print(f"🚀 빠른 샘플 테스트 ({num_samples}개)")
    print("=" * 60)
    
    # 각 카테고리에서 샘플 추출
    sample_questions = []
    for category, questions in TEST_SCENARIOS.items():
        sample_size = min(num_samples // 3, len(questions))
        sample_questions.extend(questions[:sample_size])
    
    try:
        from src.agents.vehicle_agent import VehicleManualAgent
        from src.config.settings import DEFAULT_PDF_PATH
        
        agent = VehicleManualAgent(str(DEFAULT_PDF_PATH))
        
        for i, question in enumerate(sample_questions, 1):
            print(f"{i}. {question}")
            try:
                start_time = time.time()
                answer = agent.query(question)
                end_time = time.time()
                
                # 간단한 결과 출력
                import re
                confidence_match = re.search(r'답변 신뢰도\*\*:\s*(\d+(?:\.\d+)?)%', answer)
                confidence = float(confidence_match.group(1)) if confidence_match else 0
                
                print(f"   ✅ 신뢰도: {confidence:.0f}% ({end_time-start_time:.1f}초)")
                
            except Exception as e:
                print(f"   ❌ 오류: {str(e)}")
    
    except Exception as e:
        print(f"❌ 테스트 실행 실패: {e}")

def main():
    """메인 함수"""
    
    print("🚗 차량 테스트 시나리오 v2.0 (개선된 시스템)")
    print("=" * 80)
    print("📋 테스트 구성:")
    print("   • 운전자가 겪을 수 있는 상황: 20개")
    print("   • 차량 교체주기 관련 질문: 10개")
    print("   • 기타 차량 관련 질문: 20개")
    print("   • 총 50개 시나리오")
    print("")
    print("🔧 개선사항:")
    print("   • 응급 상황 신뢰도 문제 해결")
    print("   • 정비 질문 잘못 분류 문제 해결")
    print("   • 답변 압축 최적화")
    print("   • 로그 중복 출력 문제 해결")
    print("")
    print("선택하세요:")
    print("1. 전체 테스트 (50개 - 약 10-15분 소요)")
    print("2. 빠른 테스트 (10개 - 약 2-3분 소요)")
    print("3. 테스트 시나리오 목록만 보기")
    print("=" * 80)
    
    try:
        choice = input("선택 (1-3): ").strip()
        
        if choice == "1":
            run_comprehensive_test()
        elif choice == "2":
            run_quick_test()
        elif choice == "3":
            print_test_scenarios()
        else:
            print("올바른 선택지를 입력해주세요.")
    
    except KeyboardInterrupt:
        print("\n\n👋 테스트를 중단합니다.")
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")

def print_test_scenarios():
    """테스트 시나리오 목록 출력"""
    
    print("\n📋 테스트 시나리오 목록")
    print("=" * 80)
    
    for category, questions in TEST_SCENARIOS.items():
        print(f"\n📂 {category.upper()} ({len(questions)}개)")
        print("-" * 60)
        for i, question in enumerate(questions, 1):
            print(f"{i:2d}. {question}")

if __name__ == "__main__":
    main()
