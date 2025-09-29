#!/usr/bin/env python3
"""
확장된 차량 관련 테스트 시나리오 100개
"""

import sys
import os
import time
import random
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

# 확장된 테스트 시나리오 데이터 (기존 50개와 다른 새로운 100개)
EXTENDED_TEST_SCENARIOS = {
    "emergency_driving_situations": [
        # 응급 주행 상황 25개
        "주행 중 갑자기 엔진에서 이상한 소리가 나요",
        "고속도로에서 스티어링 휠이 떨려요",
        "차가 한쪽으로 계속 쏠려서 위험해요",
        "브레이크 페달이 바닥까지 들어가요",
        "가속 페달을 밟아도 속도가 안 올라가요",
        "차에서 타는 냄새가 심하게 나요",
        "계기판이 모두 꺼졌어요",
        "주행 중 갑자기 파워 스티어링이 무거워졌어요",
        "터널 안에서 헤드라이트가 갑자기 꺼졌어요",
        "비 오는 날 와이퍼가 멈춰서 앞이 안 보여요",
        "고속 주행 중 타이어에서 펑 소리가 났어요",
        "차량이 갑자기 흔들리면서 진동이 심해요",
        "엔진 온도 경고등이 빨갛게 켜졌어요",
        "주행 중 기어가 빠져서 중립으로 들어가요",
        "ABS 경고등이 켜지면서 브레이크가 이상해요",
        "에어백 경고등이 켜졌는데 계속 운전해도 되나요",
        "주행 중 갑자기 엔진이 꺼져서 시동이 안 걸려요",
        "차량 하부에서 기름이 새고 있어요",
        "타이어가 심하게 마모되어 철사가 보여요",
        "주차 브레이크가 풀리지 않은 채로 운전했어요",
        "차량 뒤에서 하얀 연기가 계속 나와요",
        "운전석 창문이 갑자기 내려가서 올라오지 않아요",
        "차량 경보음이 계속 울려서 끌 수가 없어요",
        "엔진룸에서 쇳소리가 계속 나요",
        "주행 중 갑자기 모든 전자장비가 꺼졌어요"
    ],
    
    "maintenance_and_service": [
        # 정비 및 서비스 25개
        "트랜스미션 오일은 언제 교체해야 하나요",
        "스파크 플러그 교체 주기를 알려주세요",
        "에어컨 필터는 얼마나 자주 바꿔야 하나요",
        "연료 필터 교체 시기가 궁금해요",
        "파워 스티어링 오일 점검 방법을 알려주세요",
        "브레이크 액 교체는 언제 해야 하나요",
        "부동액 교체 주기는 어떻게 되나요",
        "차량 정기점검은 몇 개월마다 받아야 하나요",
        "타이밍 벨트는 언제 교체하나요",
        "차량 배터리 수명은 보통 얼마나 되나요",
        "에어 서스펜션 점검은 어떻게 하나요",
        "디퍼렌셜 오일 교체 주기를 알려주세요",
        "클러치 액 점검 방법이 궁금해요",
        "차량 하부 점검은 어떻게 하나요",
        "벨트류 점검 및 교체 시기는 언제인가요",
        "에어백 시스템 점검은 필요한가요",
        "서스펜션 부시 교체 시기를 알려주세요",
        "라디에이터 호스 교체는 언제 하나요",
        "차량 알루미늄 휠 관리 방법을 알려주세요",
        "엔진 마운트 점검은 어떻게 하나요",
        "차량 언더코팅은 언제 해야 하나요",
        "머플러 교체 시기를 알려주세요",
        "차량 도장 관리 방법이 궁금해요",
        "타이어 로테이션은 언제 해야 하나요",
        "차량 실내 청소 및 관리 방법을 알려주세요"
    ],
    
    "vehicle_features_and_technology": [
        # 차량 기능 및 기술 25개
        "어댑티브 크루즈 컨트롤 설정 방법을 알려주세요",
        "차선 이탈 경고 시스템은 어떻게 작동하나요",
        "자동 긴급 제동 시스템이 뭔가요",
        "사각지대 모니터링 시스템 사용법을 알려주세요",
        "파크 어시스트 기능은 어떻게 사용하나요",
        "원격 시동 기능 설정 방법이 궁금해요",
        "스마트 키 배터리 교체는 어떻게 하나요",
        "차량 앱으로 할 수 있는 기능들이 뭐가 있나요",
        "음성 인식 시스템 설정 방법을 알려주세요",
        "무선 충전 패드 사용법이 궁금해요",
        "헤드업 디스플레이 조정 방법을 알려주세요",
        "360도 카메라 시스템은 어떻게 보나요",
        "트레일러 견인 모드 설정 방법을 알려주세요",
        "오프로드 주행 모드는 언제 사용하나요",
        "에코 모드와 스포츠 모드의 차이점은 뭐예요",
        "차량 보안 시스템 설정 방법을 알려주세요",
        "자동 하이빔 기능은 어떻게 작동하나요",
        "레인 센싱 와이퍼 설정 방법이 궁금해요",
        "시트 메모리 기능 사용법을 알려주세요",
        "클라이밋 컨트롤 자동 설정 방법을 알려주세요",
        "차량 내 와이파이 연결 방법이 궁금해요",
        "USB 포트로 기기 연결하는 방법을 알려주세요",
        "차량 소프트웨어 업데이트는 어떻게 하나요",
        "타이어 공기압 모니터링 시스템 리셋 방법을 알려주세요",
        "전자식 주차 브레이크 사용법이 궁금해요"
    ],
    
    "seasonal_and_environmental": [
        # 계절별 및 환경별 25개
        "겨울철 엔진 예열은 얼마나 해야 하나요",
        "눈길 운전 시 주의사항을 알려주세요",
        "여름철 에어컨 사용 시 연비 절약 방법은 뭐예요",
        "장마철 차량 관리 방법을 알려주세요",
        "해안가 운전 후 차량 관리는 어떻게 하나요",
        "산길 운전 시 주의사항이 뭐가 있나요",
        "터널 운전 시 안전 수칙을 알려주세요",
        "안개 낀 날 운전 요령을 알려주세요",
        "강풍 시 운전 주의사항이 궁금해요",
        "폭우 시 차량 운행은 어떻게 해야 하나요",
        "결빙 구간 운전 방법을 알려주세요",
        "사막 지역 운전 시 준비사항이 뭐예요",
        "고산 지대 운전 시 주의점을 알려주세요",
        "도심 정체 구간에서 연비 절약 방법은 뭐예요",
        "지하주차장 환기 시 주의사항이 있나요",
        "차량 보관 시 습기 방지 방법을 알려주세요",
        "염해 지역 운전 후 관리 방법이 궁금해요",
        "황사 발생 시 차량 관리는 어떻게 하나요",
        "미세먼지 심한 날 차량 환기 방법을 알려주세요",
        "극한 추위에서 차량 시동 방법을 알려주세요",
        "폭염 시 차량 실내 온도 관리 방법은 뭐예요",
        "우박 피해 방지 방법을 알려주세요",
        "태풍 경보 시 차량 보관 방법이 궁금해요",
        "스모그 지역 운전 시 주의사항을 알려주세요",
        "일교차 큰 지역에서 차량 관리 방법을 알려주세요"
    ]
}

def run_extended_comprehensive_test():
    """확장된 포괄적인 차량 테스트 실행"""
    
    print("🚗 확장된 차량 관련 종합 테스트 시작 (100개)")
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
            "emergency_driving_situations": [],
            "maintenance_and_service": [],
            "vehicle_features_and_technology": [],
            "seasonal_and_environmental": [],
            "summary": {
                "total_tests": 0,
                "successful_tests": 0,
                "failed_tests": 0,
                "emergency_detected": 0,
                "driving_context_detected": 0,
                "high_confidence_answers": 0,
                "low_confidence_answers": 0,
                "average_response_time": 0.0,
                "category_performance": {}
            }
        }
        
        total_response_time = 0.0
        
        # 각 카테고리별 테스트 실행
        for category, questions in EXTENDED_TEST_SCENARIOS.items():
            print(f"\n📋 {category.upper()} 테스트 ({len(questions)}개)")
            print("-" * 60)
            
            category_results = []
            category_response_time = 0.0
            
            for i, question in enumerate(questions, 1):
                try:
                    print(f"{i:2d}. {question[:60]}{'...' if len(question) > 60 else ''}")
                    
                    # 새로운 쿼리를 위해 콜백 핸들러 세션 초기화
                    for callback in callbacks:
                        if hasattr(callback, 'reset_session'):
                            callback.reset_session()
                    
                    start_time = time.time()
                    answer = agent.query(question, callbacks=callbacks)
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    category_response_time += response_time
                    total_response_time += response_time
                    
                    # 답변 분석
                    result = analyze_answer(question, answer, response_time)
                    category_results.append(result)
                    
                    # 결과 출력 (간단히)
                    status_icon = "✅" if result["success"] else "❌"
                    confidence_text = f"{result['confidence']:.0f}%" if result['confidence'] else "N/A"
                    emergency_text = "🚨" if result["is_emergency"] else ""
                    driving_text = "🚗" if result["is_driving"] else ""
                    
                    print(f"    {status_icon} 신뢰도: {confidence_text} {emergency_text} {driving_text} ({result['response_time']:.1f}초)")
                    
                    # 통계 업데이트
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
                    time.sleep(0.3)
                    
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
            
            # 카테고리별 성능 저장
            category_success_rate = sum(1 for r in category_results if r["success"]) / len(category_results) * 100
            category_avg_time = category_response_time / len(category_results)
            test_results["summary"]["category_performance"][category] = {
                "success_rate": category_success_rate,
                "average_time": category_avg_time,
                "total_questions": len(category_results)
            }
            
            test_results[category] = category_results
        
        # 전체 평균 응답 시간 계산
        test_results["summary"]["average_response_time"] = total_response_time / test_results["summary"]["total_tests"]
        
        # 최종 결과 출력
        print_extended_test_summary(test_results)
        
        # 개선점 분석 및 제안
        analyze_and_suggest_extended_improvements(test_results)
        
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
        "has_page_reference": False,
        "has_safety_warning": False,
        "answer_type": "general"
    }
    
    try:
        # 신뢰도 추출
        import re
        confidence_match = re.search(r'답변 신뢰도\*\*:\s*(\d+(?:\.\d+)?)%', answer)
        if confidence_match:
            result["confidence"] = float(confidence_match.group(1))
        
        # 응급 상황 감지 확인
        emergency_indicators = ["🚨", "🔥", "CRITICAL", "HIGH", "응급", "즉시", "위험", "생명"]
        result["is_emergency"] = any(indicator in answer for indicator in emergency_indicators)
        
        # 주행 중 감지 확인
        driving_indicators = ["🚗", "주행 중", "운전 중", "안전한 곳에 정차", "압축", "주행"]
        result["is_driving"] = any(indicator in answer for indicator in driving_indicators)
        
        # 페이지 참조 확인
        result["has_page_reference"] = "참고 페이지" in answer or re.search(r'페이지\s*\d+', answer)
        
        # 안전 경고 확인
        safety_indicators = ["⚠️", "🛑", "주의", "경고", "안전", "위험"]
        result["has_safety_warning"] = any(indicator in answer for indicator in safety_indicators)
        
        # 답변 타입 분류
        if result["is_emergency"]:
            result["answer_type"] = "emergency"
        elif "교체" in question or "주기" in question or "정비" in question:
            result["answer_type"] = "maintenance"
        elif any(tech in question for tech in ["시스템", "기능", "설정", "앱", "기술"]):
            result["answer_type"] = "technology"
        elif any(season in question for season in ["겨울", "여름", "눈", "비", "폭우", "안개"]):
            result["answer_type"] = "seasonal"
        
        # 답변 품질 확인
        if len(answer) < 30:
            result["success"] = False
        elif "오류" in answer or "실패" in answer:
            result["success"] = False
        elif result["confidence"] and result["confidence"] < 30:
            result["success"] = False
            
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
    
    return result

def print_extended_test_summary(test_results):
    """확장된 테스트 요약 출력"""
    
    summary = test_results["summary"]
    
    print(f"\n{'='*80}")
    print("📊 확장 테스트 결과 요약 (100개 시나리오)")
    print(f"{'='*80}")
    
    print(f"📈 전체 통계:")
    print(f"   • 총 테스트: {summary['total_tests']}개")
    print(f"   • 성공: {summary['successful_tests']}개 ({summary['successful_tests']/summary['total_tests']*100:.1f}%)")
    print(f"   • 실패: {summary['failed_tests']}개 ({summary['failed_tests']/summary['total_tests']*100:.1f}%)")
    print(f"   • 평균 응답 시간: {summary['average_response_time']:.2f}초")
    
    print(f"\n🎯 특별 기능 감지:")
    print(f"   • 응급 상황 감지: {summary['emergency_detected']}개 ({summary['emergency_detected']/summary['total_tests']*100:.1f}%)")
    print(f"   • 주행 중 감지: {summary['driving_context_detected']}개 ({summary['driving_context_detected']/summary['total_tests']*100:.1f}%)")
    
    print(f"\n📊 답변 신뢰도:")
    print(f"   • 높은 신뢰도 (80% 이상): {summary['high_confidence_answers']}개 ({summary['high_confidence_answers']/summary['total_tests']*100:.1f}%)")
    print(f"   • 낮은 신뢰도 (60% 미만): {summary['low_confidence_answers']}개 ({summary['low_confidence_answers']/summary['total_tests']*100:.1f}%)")
    
    # 카테고리별 성능
    print(f"\n📋 카테고리별 성능:")
    for category, performance in summary['category_performance'].items():
        category_name = {
            'emergency_driving_situations': '응급 주행 상황',
            'maintenance_and_service': '정비 및 서비스',
            'vehicle_features_and_technology': '차량 기능 및 기술',
            'seasonal_and_environmental': '계절별 및 환경별'
        }.get(category, category)
        
        print(f"   • {category_name}: {performance['success_rate']:.1f}% "
              f"(평균 {performance['average_time']:.1f}초, {performance['total_questions']}개)")

def analyze_and_suggest_extended_improvements(test_results):
    """확장된 개선점 분석 및 제안"""
    
    print(f"\n{'='*80}")
    print("💡 확장 테스트 개선점 분석")
    print(f"{'='*80}")
    
    issues = []
    suggestions = []
    priority_improvements = []
    
    summary = test_results["summary"]
    
    # 전체 성공률 분석
    success_rate = summary['successful_tests'] / summary['total_tests'] * 100
    if success_rate < 85:
        issues.append(f"전체 성공률이 {success_rate:.1f}%로 목표치(85%) 미달")
        suggestions.append("검색 알고리즘 및 답변 생성 로직 전반적 개선 필요")
        priority_improvements.append("HIGH")
    
    # 응답 시간 분석
    avg_time = summary['average_response_time']
    if avg_time > 15:
        issues.append(f"평균 응답 시간이 {avg_time:.1f}초로 너무 긴 편")
        suggestions.append("검색 및 생성 프로세스 최적화 필요")
        priority_improvements.append("MEDIUM")
    
    # 신뢰도 분석
    low_confidence_rate = summary['low_confidence_answers'] / summary['total_tests'] * 100
    if low_confidence_rate > 15:
        issues.append(f"낮은 신뢰도 답변이 {low_confidence_rate:.1f}%로 높음")
        suggestions.append("문서 검색 품질 향상 및 평가 기준 재조정 필요")
        priority_improvements.append("HIGH")
    
    # 카테고리별 성능 분석
    problem_categories = []
    for category, performance in summary['category_performance'].items():
        if performance['success_rate'] < 75:
            problem_categories.append((category, performance['success_rate']))
    
    if problem_categories:
        for category, rate in problem_categories:
            category_name = {
                'emergency_driving_situations': '응급 주행 상황',
                'maintenance_and_service': '정비 및 서비스', 
                'vehicle_features_and_technology': '차량 기능 및 기술',
                'seasonal_and_environmental': '계절별 및 환경별'
            }.get(category, category)
            
            issues.append(f"{category_name} 카테고리 성공률이 {rate:.1f}%로 낮음")
            
            if category == 'emergency_driving_situations':
                suggestions.append("응급 상황 키워드 확장 및 감지 로직 개선")
                priority_improvements.append("CRITICAL")
            elif category == 'maintenance_and_service':
                suggestions.append("정비 관련 문서 보강 및 검색 전략 개선")
                priority_improvements.append("HIGH")
            elif category == 'vehicle_features_and_technology':
                suggestions.append("최신 차량 기술 관련 문서 업데이트 필요")
                priority_improvements.append("MEDIUM")
            elif category == 'seasonal_and_environmental':
                suggestions.append("계절별/환경별 운전 가이드 문서 보강")
                priority_improvements.append("MEDIUM")
    
    # 응급 상황 감지 분석
    emergency_rate = summary['emergency_detected'] / summary['total_tests'] * 100
    expected_emergency_rate = 25  # 응급 상황 카테고리가 25개이므로 25% 기대
    if emergency_rate < expected_emergency_rate * 0.7:  # 70% 이하 감지
        issues.append(f"응급 상황 감지율이 {emergency_rate:.1f}%로 낮음 (기대: ~{expected_emergency_rate}%)")
        suggestions.append("응급 상황 감지 키워드 패턴 확장 및 임계값 조정")
        priority_improvements.append("CRITICAL")
    
    # 결과 출력
    if issues:
        print("🔍 발견된 주요 문제점:")
        for i, (issue, priority) in enumerate(zip(issues, priority_improvements), 1):
            priority_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(priority, "⚪")
            print(f"   {i}. {priority_icon} {issue}")
        
        print(f"\n💡 개선 제안:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
    else:
        print("✅ 모든 테스트가 양호한 성능을 보입니다!")
    
    # 구체적인 개선 작업 제안
    print(f"\n🔧 우선순위별 개선 작업:")
    
    critical_items = [s for i, s in enumerate(suggestions) if priority_improvements[i] == "CRITICAL"]
    high_items = [s for i, s in enumerate(suggestions) if priority_improvements[i] == "HIGH"]
    medium_items = [s for i, s in enumerate(suggestions) if priority_improvements[i] == "MEDIUM"]
    
    if critical_items:
        print("   🔴 CRITICAL (즉시 수정 필요):")
        for item in critical_items:
            print(f"     • {item}")
    
    if high_items:
        print("   🟠 HIGH (우선 수정):")
        for item in high_items:
            print(f"     • {item}")
    
    if medium_items:
        print("   🟡 MEDIUM (계획적 수정):")
        for item in medium_items:
            print(f"     • {item}")

def run_sample_test(num_samples=20):
    """샘플 테스트 (각 카테고리에서 5개씩)"""
    
    print(f"🚀 확장 시나리오 샘플 테스트 ({num_samples}개)")
    print("=" * 60)
    
    # 각 카테고리에서 5개씩 랜덤 샘플 추출
    sample_questions = []
    for category, questions in EXTENDED_TEST_SCENARIOS.items():
        samples = random.sample(questions, min(5, len(questions)))
        sample_questions.extend(samples)
    
    # 총 20개로 제한
    sample_questions = sample_questions[:num_samples]
    
    try:
        from src.agents.vehicle_agent import VehicleManualAgent
        from src.config.settings import DEFAULT_PDF_PATH
        
        agent = VehicleManualAgent(str(DEFAULT_PDF_PATH))
        
        success_count = 0
        total_time = 0
        
        for i, question in enumerate(sample_questions, 1):
            print(f"{i:2d}. {question}")
            try:
                start_time = time.time()
                answer = agent.query(question)
                end_time = time.time()
                
                response_time = end_time - start_time
                total_time += response_time
                
                # 간단한 결과 분석
                import re
                confidence_match = re.search(r'답변 신뢰도\*\*:\s*(\d+(?:\.\d+)?)%', answer)
                confidence = float(confidence_match.group(1)) if confidence_match else 0
                
                if confidence > 0 and len(answer) > 50:
                    success_count += 1
                    status = "✅"
                else:
                    status = "❌"
                
                print(f"   {status} 신뢰도: {confidence:.0f}% ({response_time:.1f}초)")
                
            except Exception as e:
                print(f"   ❌ 오류: {str(e)}")
        
        print(f"\n📊 샘플 테스트 결과:")
        print(f"   • 성공률: {success_count}/{num_samples} ({success_count/num_samples*100:.1f}%)")
        print(f"   • 평균 응답 시간: {total_time/num_samples:.1f}초")
    
    except Exception as e:
        print(f"❌ 테스트 실행 실패: {e}")

def print_extended_test_scenarios():
    """확장된 테스트 시나리오 목록 출력"""
    
    print("\n📋 확장된 테스트 시나리오 목록 (100개)")
    print("=" * 80)
    
    for category, questions in EXTENDED_TEST_SCENARIOS.items():
        category_name = {
            'emergency_driving_situations': '🚨 응급 주행 상황',
            'maintenance_and_service': '🔧 정비 및 서비스',
            'vehicle_features_and_technology': '💻 차량 기능 및 기술',
            'seasonal_and_environmental': '🌦️ 계절별 및 환경별'
        }.get(category, category)
        
        print(f"\n📂 {category_name} ({len(questions)}개)")
        print("-" * 60)
        for i, question in enumerate(questions, 1):
            print(f"{i:2d}. {question}")

def main():
    """메인 함수"""
    
    print("🚗 확장된 차량 테스트 시나리오 v2.0 (개선 완료)")
    print("=" * 80)
    print("📋 확장 테스트 구성:")
    print("   • 응급 주행 상황: 25개")
    print("   • 정비 및 서비스: 25개")
    print("   • 차량 기능 및 기술: 25개")
    print("   • 계절별 및 환경별: 25개")
    print("   • 총 100개 시나리오 (기존 50개와 중복 없음)")
    print("")
    print("🎯 목적:")
    print("   • 기존 테스트에서 다루지 않은 새로운 시나리오 검증")
    print("   • 시스템의 다양한 상황 대응 능력 평가")
    print("   • 추가적인 개선점 도출 및 적용")
    print("")
    print("🔧 v2.0 개선사항:")
    print("   • 기술 문의 오분류 문제 해결 (자동 긴급 제동 시스템 등)")
    print("   • 심각한 응급 상황 감지 개선 (전자장비 고장, 가속 불량 등)")
    print("   • 응급 키워드 확장 및 정비/기술 질문 필터링 강화")
    print("   • 응급 상황 신뢰도 정보 제공 개선")
    print("")
    print("선택하세요:")
    print("1. 전체 테스트 (100개 - 약 20-25분 소요)")
    print("2. 샘플 테스트 (20개 - 약 4-5분 소요)")
    print("3. 테스트 시나리오 목록만 보기")
    print("=" * 80)
    
    try:
        choice = input("선택 (1-3): ").strip()
        
        if choice == "1":
            run_extended_comprehensive_test()
        elif choice == "2":
            run_sample_test()
        elif choice == "3":
            print_extended_test_scenarios()
        else:
            print("올바른 선택지를 입력해주세요.")
    
    except KeyboardInterrupt:
        print("\n\n👋 테스트를 중단합니다.")
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")

if __name__ == "__main__":
    main()
