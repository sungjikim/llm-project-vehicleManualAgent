#!/usr/bin/env python3
"""
통합된 차량 관련 테스트 시나리오

기존 3개 테스트 파일의 기능을 통합:
- vehicle_test_scenarios.py: 기본 테스트 시나리오 50개
- extended_test_scenarios.py: 확장 테스트 시나리오 100개  
- test_driver_scenarios.py: 운전자 실제 상황 테스트

SubGraph 아키텍처 기반으로 통일
"""

import sys
import os
import time
import random
import signal
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# 통합된 테스트 시나리오 데이터
INTEGRATED_TEST_SCENARIOS = {
    "emergency_situations": [
        # 응급 상황 30개 (기존 + 확장)
        "갑자기 브레이크가 안 밟혀요",
        "고속도로에서 타이어가 터졌어요",
        "엔진에서 연기가 나고 있어요",
        "계기판에 빨간 경고등이 켜졌어요",
        "핸들이 한쪽으로 계속 쏠려요",
        "차가 시동이 안 걸려요",
        "주행 중 갑자기 엔진이 꺼졌어요",
        "차에서 이상한 소음이 나요",
        "기어가 안 들어가요",
        "주차 브레이크가 안 풀려요",
        "냉각수 온도가 너무 높아요",
        "배터리 방전으로 시동이 안 걸려요",
        "연료가 부족한데 주유소가 안 보여요",
        "안전벨트가 잠기지 않아요",
        "문이 잠기지 않아요",
        "후진할 때 후방 카메라가 안 보여요",
        "차량 키를 차 안에 두고 나왔어요",
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
        "엔진 온도 경고등이 빨갛게 켜졌어요"
    ],
    
    "maintenance_questions": [
        # 정비 및 교체 주기 25개
        "엔진 오일 교체 주기가 언제인가요?",
        "타이어는 언제 교체해야 하나요?",
        "브레이크 패드 교체 시기를 알려주세요",
        "에어 필터는 얼마나 자주 바꿔야 하나요?",
        "배터리 교체 주기는 어떻게 되나요?",
        "냉각수는 언제 교체하나요?",
        "점화플러그 교체 주기를 알려주세요",
        "변속기 오일은 언제 갈아야 하나요?",
        "와이퍼 블레이드 교체 시기는요?",
        "연료 필터 교체 주기가 궁금해요",
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
        "벨트류 점검 및 교체 시기는 언제인가요"
    ],
    
    "general_questions": [
        # 일반 질문 25개
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
        "차량 소음을 줄이는 방법이 있나요?",
        "어댑티브 크루즈 컨트롤 설정 방법을 알려주세요",
        "차선 이탈 경고 시스템은 어떻게 작동하나요",
        "자동 긴급 제동 시스템이 뭔가요",
        "사각지대 모니터링 시스템 사용법을 알려주세요",
        "파크 어시스트 기능은 어떻게 사용하나요"
    ],
    
    "driver_real_scenarios": [
        # 운전자 실제 상황 10개
        "운전석 시트를 내 체형에 맞게 조정하는 방법이 궁금해요",
        "겨울철 히터를 효율적으로 사용하는 방법을 알려주세요",
        "후방 카메라 화면이 흐릿한데 청소는 어떻게 하나요?",
        "연비를 좋게 하려면 어떤 운전 습관을 가져야 할까요?",
        "블루투스로 스마트폰 음악을 들으려면 어떻게 연결하나요?",
        "주행 중 갑자기 엔진 경고등이 빨갛게 켜졌어요! 어떻게 해야 하나요?",
        "고속도로에서 타이어가 터진 것 같은데 안전하게 대처하는 방법은?",
        "브레이크 페달을 밟는데 바닥까지 들어가요! 급한 상황인가요?",
        "운전 중 시동이 꺼지면서 파워 스티어링이 무거워졌어요!",
        "차 안에 가스 냄새가 나는데 즉시 해야 할 조치가 뭔가요?"
    ]
}


class IntegratedTestRunner:
    """통합 테스트 실행기"""
    
    def __init__(self):
        self.agent = None
        self.callbacks = []
        self.results = {
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "emergency_detected": 0,
            "driving_context_detected": 0,
            "total_time": 0,
            "average_response_time": 0
        }
    
    def initialize_agent(self):
        """SubGraph 에이전트 초기화"""
        try:
            from src.agents.vehicle_agent import VehicleManualAgent
            from src.config.settings import DEFAULT_PDF_PATH
            from src.utils.callback_handlers import (
                PerformanceMonitoringHandler,
                RealTimeNotificationHandler,
                AlertHandler
            )
            
            print("🔧 SubGraph 에이전트 초기화 중...")
            self.agent = VehicleManualAgent(str(DEFAULT_PDF_PATH))
            
            # 콜백 핸들러 초기화
            performance_handler = PerformanceMonitoringHandler(enable_detailed_logging=False)
            notification_handler = RealTimeNotificationHandler(enable_progress_bar=False, enable_notifications=False)
            alert_handler = AlertHandler(token_limit=50000, cost_limit=5.0)
            
            self.callbacks = [performance_handler, notification_handler, alert_handler]
            print("✅ SubGraph 에이전트 초기화 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 에이전트 초기화 실패: {str(e)}")
            return False
    
    def run_test_category(self, category_name: str, questions: List[str], max_questions: int = None) -> Dict[str, Any]:
        """특정 카테고리 테스트 실행"""
        print(f"\n{'='*60}")
        print(f"🧪 {category_name} 테스트 시작")
        print(f"{'='*60}")
        
        if max_questions:
            questions = questions[:max_questions]
        
        category_results = {
            "category": category_name,
            "total": len(questions),
            "successful": 0,
            "failed": 0,
            "emergency_detected": 0,
            "driving_context_detected": 0,
            "total_time": 0,
            "questions": []
        }
        
        for i, question in enumerate(questions, 1):
            print(f"\n[테스트 {i}/{len(questions)}] {category_name}")
            print(f"❓ 질문: {question}")
            print("-" * 50)
            
            try:
                start_time = time.time()
                
                # 콜백 세션 초기화
                for callback in self.callbacks:
                    if hasattr(callback, 'reset_session'):
                        callback.reset_session()
                
                # 질문 실행
                response = self.agent.query(question, callbacks=self.callbacks)
                end_time = time.time()
                
                response_time = end_time - start_time
                category_results["total_time"] += response_time
                category_results["successful"] += 1
                self.results["successful_tests"] += 1
                
                # 응급 상황 및 주행 중 감지 확인
                if any(indicator in response for indicator in ["🚨", "CRITICAL", "HIGH", "응급", "즉시", "위험"]):
                    category_results["emergency_detected"] += 1
                    self.results["emergency_detected"] += 1
                
                if any(indicator in response for indicator in ["🚗", "주행 중", "운전 중", "압축"]):
                    category_results["driving_context_detected"] += 1
                    self.results["driving_context_detected"] += 1
                
                # 결과 저장
                question_result = {
                    "question": question,
                    "response": response[:200] + "..." if len(response) > 200 else response,
                    "response_time": response_time,
                    "success": True
                }
                category_results["questions"].append(question_result)
                
                print(f"✅ 성공 (응답시간: {response_time:.1f}초)")
                print(f"📝 응답 미리보기: {response[:100]}...")
                
            except Exception as e:
                category_results["failed"] += 1
                self.results["failed_tests"] += 1
                print(f"❌ 실패: {str(e)}")
                
                question_result = {
                    "question": question,
                    "response": f"오류: {str(e)}",
                    "response_time": 0,
                    "success": False
                }
                category_results["questions"].append(question_result)
            
            # 잠시 대기 (API 제한 방지)
            time.sleep(1)
        
        return category_results
    
    def run_quick_test(self, max_per_category: int = 3):
        """빠른 테스트 실행 (각 카테고리별 3개씩)"""
        print("🚀 빠른 통합 테스트 시작")
        print("=" * 60)
        print(f"📝 각 카테고리별 {max_per_category}개 질문씩 테스트")
        print("=" * 60)
        
        if not self.initialize_agent():
            return False
        
        start_time = time.time()
        
        # 각 카테고리별 테스트 실행
        for category_name, questions in INTEGRATED_TEST_SCENARIOS.items():
            category_results = self.run_test_category(
                category_name, 
                questions, 
                max_questions=max_per_category
            )
            self.results["total_tests"] += category_results["total"]
        
        end_time = time.time()
        self.results["total_time"] = end_time - start_time
        self.results["average_response_time"] = self.results["total_time"] / self.results["total_tests"] if self.results["total_tests"] > 0 else 0
        
        self.print_summary()
        return True
    
    def run_full_test(self):
        """전체 테스트 실행 (모든 질문)"""
        print("🚀 전체 통합 테스트 시작")
        print("=" * 60)
        print("📝 모든 카테고리의 모든 질문을 테스트합니다")
        print("=" * 60)
        
        if not self.initialize_agent():
            return False
        
        start_time = time.time()
        
        # 각 카테고리별 테스트 실행
        for category_name, questions in INTEGRATED_TEST_SCENARIOS.items():
            category_results = self.run_test_category(category_name, questions)
            self.results["total_tests"] += category_results["total"]
        
        end_time = time.time()
        self.results["total_time"] = end_time - start_time
        self.results["average_response_time"] = self.results["total_time"] / self.results["total_tests"] if self.results["total_tests"] > 0 else 0
        
        self.print_summary()
        return True
    
    def run_emergency_focus_test(self):
        """응급 상황 집중 테스트"""
        print("🚨 응급 상황 집중 테스트 시작")
        print("=" * 60)
        print("📝 응급 상황과 운전자 실제 상황을 집중 테스트합니다")
        print("=" * 60)
        
        if not self.initialize_agent():
            return False
        
        start_time = time.time()
        
        # 응급 상황 테스트
        emergency_results = self.run_test_category(
            "응급 상황", 
            INTEGRATED_TEST_SCENARIOS["emergency_situations"][:10]  # 10개만
        )
        self.results["total_tests"] += emergency_results["total"]
        
        # 운전자 실제 상황 테스트
        driver_results = self.run_test_category(
            "운전자 실제 상황", 
            INTEGRATED_TEST_SCENARIOS["driver_real_scenarios"]
        )
        self.results["total_tests"] += driver_results["total"]
        
        end_time = time.time()
        self.results["total_time"] = end_time - start_time
        self.results["average_response_time"] = self.results["total_time"] / self.results["total_tests"] if self.results["total_tests"] > 0 else 0
        
        self.print_summary()
        return True
    
    def print_summary(self):
        """테스트 결과 요약 출력"""
        print("\n" + "=" * 60)
        print("📊 통합 테스트 결과 요약")
        print("=" * 60)
        
        success_rate = (self.results["successful_tests"] / self.results["total_tests"] * 100) if self.results["total_tests"] > 0 else 0
        
        print(f"📝 총 테스트 수: {self.results['total_tests']}개")
        print(f"✅ 성공: {self.results['successful_tests']}개")
        print(f"❌ 실패: {self.results['failed_tests']}개")
        print(f"📈 성공률: {success_rate:.1f}%")
        print(f"🚨 응급 상황 감지: {self.results['emergency_detected']}개")
        print(f"🚗 주행 중 감지: {self.results['driving_context_detected']}개")
        print(f"⏱️ 총 소요 시간: {self.results['total_time']:.1f}초")
        print(f"⚡ 평균 응답 시간: {self.results['average_response_time']:.1f}초")
        
        if self.results["successful_tests"] > 0:
            print("\n🎉 SubGraph 아키텍처 기반 통합 테스트가 성공적으로 완료되었습니다!")
        else:
            print("\n❌ 테스트 실행에 문제가 있었습니다.")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="통합 차량 테스트 시나리오 실행")
    parser.add_argument(
        "--mode", 
        choices=["quick", "full", "emergency"],
        default="quick",
        help="테스트 모드 선택 (기본값: quick)"
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=3,
        help="빠른 테스트에서 카테고리별 최대 질문 수 (기본값: 3)"
    )
    
    args = parser.parse_args()
    
    # 시그널 핸들러 설정
    def signal_handler(sig, frame):
        print("\n\n🛑 테스트 중단됨...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 테스트 실행기 생성
    runner = IntegratedTestRunner()
    
    try:
        if args.mode == "quick":
            success = runner.run_quick_test(args.max_per_category)
        elif args.mode == "full":
            success = runner.run_full_test()
        elif args.mode == "emergency":
            success = runner.run_emergency_focus_test()
        
        if success:
            print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
        else:
            print("\n❌ 테스트 실행에 실패했습니다.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 사용자에 의해 테스트가 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
