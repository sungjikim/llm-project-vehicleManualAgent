"""
응급 상황 시스템 테스트
"""

import unittest
import time
from typing import List, Dict, Any

from src.agents.vehicle_agent_subgraph import VehicleManualAgentSubGraph
from src.utils.emergency_detector import EmergencyDetector
from src.config.settings import DEFAULT_PDF_PATH


class TestEmergencyDetector(unittest.TestCase):
    """응급 상황 감지기 테스트"""
    
    def setUp(self):
        """테스트 준비"""
        self.detector = EmergencyDetector()
    
    def test_critical_emergency_detection(self):
        """CRITICAL 응급 상황 감지 테스트"""
        test_cases = [
            "차에서 타는 냄새가 나는데 즉시 어떻게 해야 해요?",
            "엔진에서 연기가 나고 있어요!",
            "차량에 화재가 발생했어요!"
        ]
        
        for query in test_cases:
            with self.subTest(query=query):
                result = self.detector.detect_emergency(query)
                self.assertTrue(result["is_emergency"])
                self.assertEqual(result["priority_level"], "CRITICAL")
                self.assertGreaterEqual(result["total_score"], 10)
    
    def test_high_emergency_detection(self):
        """HIGH 응급 상황 감지 테스트"""
        test_cases = [
            "브레이크를 밟아도 차가 멈추지 않아요!",
            "주행 중 핸들이 갑자기 돌아가지 않는데 응급 대처 방법은?",
            "엔진이 갑자기 정지했어요!"
        ]
        
        for query in test_cases:
            with self.subTest(query=query):
                result = self.detector.detect_emergency(query)
                self.assertTrue(result["is_emergency"])
                self.assertEqual(result["priority_level"], "HIGH")
                self.assertGreaterEqual(result["total_score"], 6)
    
    def test_medium_emergency_detection(self):
        """MEDIUM 응급 상황 감지 테스트"""
        test_cases = [
            "타이어가 펑크 났는데 어떻게 해야 해요?",
            "엔진 과열 경고등이 켜졌어요",
            "와이퍼가 고장나서 앞이 안 보여요"
        ]
        
        for query in test_cases:
            with self.subTest(query=query):
                result = self.detector.detect_emergency(query)
                # MEDIUM은 점수에 따라 응급 상황이 될 수 있음
                if result["is_emergency"]:
                    self.assertIn(result["priority_level"], ["MEDIUM", "HIGH", "CRITICAL"])
    
    def test_low_emergency_detection(self):
        """LOW 응급 상황 감지 테스트"""
        test_cases = [
            "배터리가 방전되었어요",
            "시동이 안 걸려요",
            "연료가 부족해요"
        ]
        
        for query in test_cases:
            with self.subTest(query=query):
                result = self.detector.detect_emergency(query)
                if result["is_emergency"]:
                    self.assertIn(result["priority_level"], ["LOW", "MEDIUM", "HIGH"])
    
    def test_normal_question_detection(self):
        """일반 질문 감지 테스트"""
        test_cases = [
            "타이어 공기압은 얼마로 맞춰야 하나요?",
            "엔진 오일 교체 주기는 언제인가요?",
            "XC60의 연료 탱크 용량은?",
            "후방 카메라 사용법을 알려주세요"
        ]
        
        for query in test_cases:
            with self.subTest(query=query):
                result = self.detector.detect_emergency(query)
                self.assertFalse(result["is_emergency"])
                self.assertEqual(result["priority_level"], "NORMAL")
    
    def test_urgency_expression_detection(self):
        """긴급성 표현 감지 테스트"""
        query = "브레이크가 고장났는데 즉시 도와주세요!"
        result = self.detector.detect_emergency(query)
        
        self.assertTrue(result["is_emergency"])
        self.assertGreater(result["urgency_score"], 0)
        self.assertGreater(len(result["urgency_expressions"]), 0)
    
    def test_search_strategy_assignment(self):
        """검색 전략 할당 테스트"""
        critical_query = "차에 불이 났어요!"
        result = self.detector.detect_emergency(critical_query)
        
        if result["is_emergency"]:
            strategy = result["search_strategy"]
            self.assertIn("search_method", strategy)
            self.assertIn("compression_method", strategy)
            self.assertIn("timeout", strategy)
            
            # CRITICAL 상황은 빠른 처리를 위한 설정
            if result["priority_level"] == "CRITICAL":
                self.assertEqual(strategy["search_method"], "hybrid_keyword")
                self.assertEqual(strategy["compression_method"], "rerank_only")
                self.assertEqual(strategy["timeout"], 5)


class TestEmergencySystemIntegration(unittest.TestCase):
    """응급 상황 시스템 통합 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """클래스 레벨 설정 - 에이전트 한 번만 초기화"""
        print("\n🔧 테스트용 SubGraph 에이전트 초기화 중...")
        cls.agent = VehicleManualAgentSubGraph(str(DEFAULT_PDF_PATH))
        print("✅ SubGraph 에이전트 초기화 완료")
    
    def test_emergency_response_quality(self):
        """응급 상황 응답 품질 테스트"""
        test_cases = [
            {
                'level': 'CRITICAL',
                'query': '차에서 타는 냄새가 나는데 화재 위험이 있을 때 어떻게 해야 해요?',
                'expected_keywords': ['즉시', '대피', '119', '안전']
            },
            {
                'level': 'HIGH', 
                'query': '주행 중 핸들이 갑자기 돌아가지 않는데 응급 대처 방법은?',
                'expected_keywords': ['안전', '조치', '전문가']
            },
            {
                'level': 'MEDIUM',
                'query': '엔진에서 연기가 나고 있는데 즉시 해야 할 조치가 뭐예요?',
                'expected_keywords': ['정차', '엔진', '냉각']
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(level=test_case['level']):
                answer = self.agent.query(test_case['query'])
                
                # 응답이 생성되었는지 확인
                self.assertIsNotNone(answer)
                self.assertGreater(len(answer), 0)
                
                # 예상 키워드가 포함되었는지 확인
                found_keywords = []
                for keyword in test_case['expected_keywords']:
                    if keyword in answer:
                        found_keywords.append(keyword)
                
                # 최소 하나 이상의 키워드는 포함되어야 함
                self.assertGreater(len(found_keywords), 0, 
                                 f"예상 키워드 중 하나도 찾을 수 없음: {test_case['expected_keywords']}")
                
                # 응급 상황 표시가 있는지 확인
                if test_case['level'] != 'NORMAL':
                    self.assertTrue('🚨' in answer or '⚠️' in answer, 
                                  "응급 상황 표시가 없음")
                
                # 신뢰도 정보가 포함되었는지 확인
                self.assertIn('🔍 **답변 신뢰도**', answer, "신뢰도 정보가 없음")
    
    def test_emergency_vs_normal_processing(self):
        """응급 상황과 일반 질문 처리 차이 테스트"""
        emergency_query = "브레이크를 밟아도 차가 멈추지 않아요!"
        normal_query = "타이어 공기압은 얼마로 맞춰야 하나요?"
        
        # 응급 상황 처리
        start_time = time.time()
        emergency_answer = self.agent.query(emergency_query)
        emergency_time = time.time() - start_time
        
        # 일반 질문 처리
        start_time = time.time()
        normal_answer = self.agent.query(normal_query)
        normal_time = time.time() - start_time
        
        # 응답이 생성되었는지 확인
        self.assertIsNotNone(emergency_answer)
        self.assertIsNotNone(normal_answer)
        
        # 응급 상황에는 특별한 표시가 있어야 함
        self.assertTrue('응급 상황' in emergency_answer, "응급 상황 헤더가 없음")
        self.assertTrue('일반 질문' in normal_answer, "일반 질문 헤더가 없음")
        
        # 첫 줄에 등급이 표시되어야 함
        emergency_first_line = emergency_answer.split('\n')[0]
        normal_first_line = normal_answer.split('\n')[0]
        self.assertTrue('응급 상황' in emergency_first_line, "첫 줄에 응급 등급이 없음")
        self.assertTrue('일반 질문' in normal_first_line, "첫 줄에 일반 표시가 없음")
        
        # 응급 상황에는 경고나 안전 조치가 포함되어야 함
        emergency_safety_keywords = ['즉시', '안전', '조치', '위험', '전문가']
        found_safety_keywords = [kw for kw in emergency_safety_keywords if kw in emergency_answer]
        self.assertGreater(len(found_safety_keywords), 0, "안전 관련 키워드가 없음")
        
        print(f"\n⏱️  응급 상황 처리 시간: {emergency_time:.2f}초")
        print(f"⏱️  일반 질문 처리 시간: {normal_time:.2f}초")


class TestEmergencyPerformance(unittest.TestCase):
    """응급 상황 성능 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """클래스 레벨 설정"""
        print("\n🔧 성능 테스트용 SubGraph 에이전트 초기화 중...")
        cls.agent = VehicleManualAgentSubGraph(str(DEFAULT_PDF_PATH))
        print("✅ SubGraph 에이전트 초기화 완료")
    
    def test_emergency_response_time(self):
        """응급 상황 응답 시간 테스트"""
        emergency_queries = [
            '브레이크를 밟아도 차가 멈추지 않아요!',
            '엔진에서 연기가 나고 있어요!',
            '차에서 타는 냄새가 나는데 어떻게 해야 해요?'
        ]
        
        response_times = []
        
        for query in emergency_queries:
            start_time = time.time()
            answer = self.agent.query(query)
            elapsed_time = time.time() - start_time
            response_times.append(elapsed_time)
            
            # 응답이 생성되었는지 확인
            self.assertIsNotNone(answer)
            self.assertGreater(len(answer), 0)
            
            print(f"⏱️  {elapsed_time:.2f}초 - {query[:30]}...")
        
        avg_time = sum(response_times) / len(response_times)
        print(f"\n📊 응급 상황 평균 응답 시간: {avg_time:.2f}초")
        
        # 응급 상황은 15초 이내에 응답해야 함 (합리적인 임계값)
        self.assertLess(avg_time, 15.0, "응급 상황 응답 시간이 너무 느림")
    
    def test_normal_response_time(self):
        """일반 질문 응답 시간 테스트"""
        normal_queries = [
            '타이어 공기압은 얼마로 맞춰야 하나요?',
            '엔진 오일 교체 주기는 언제인가요?', 
            'XC60의 연료 탱크 용량은?'
        ]
        
        response_times = []
        
        for query in normal_queries:
            start_time = time.time()
            answer = self.agent.query(query)
            elapsed_time = time.time() - start_time
            response_times.append(elapsed_time)
            
            # 응답이 생성되었는지 확인
            self.assertIsNotNone(answer)
            self.assertGreater(len(answer), 0)
            
            print(f"⏱️  {elapsed_time:.2f}초 - {query[:30]}...")
        
        avg_time = sum(response_times) / len(response_times)
        print(f"\n📊 일반 질문 평균 응답 시간: {avg_time:.2f}초")
        
        # 일반 질문도 15초 이내에 응답해야 함
        self.assertLess(avg_time, 15.0, "일반 질문 응답 시간이 너무 느림")
    
    def test_performance_comparison(self):
        """응급 vs 일반 성능 비교 테스트"""
        # 이 테스트는 정보 제공용이므로 assertion 없이 진행
        emergency_queries = [
            '브레이크를 밟아도 차가 멈추지 않아요!',
            '엔진에서 연기가 나고 있어요!'
        ]
        
        normal_queries = [
            '타이어 공기압은 얼마로 맞춰야 하나요?',
            '엔진 오일 교체 주기는 언제인가요?'
        ]
        
        print("\n📊 성능 비교 테스트")
        print("=" * 50)
        
        # 응급 상황 테스트
        print("\n🚨 응급 상황 질문")
        emergency_times = []
        for query in emergency_queries:
            start_time = time.time()
            self.agent.query(query)
            elapsed_time = time.time() - start_time
            emergency_times.append(elapsed_time)
            print(f"  {elapsed_time:.2f}초 - {query[:40]}...")
        
        # 일반 질문 테스트
        print("\n📝 일반 질문")
        normal_times = []
        for query in normal_queries:
            start_time = time.time()
            self.agent.query(query)
            elapsed_time = time.time() - start_time
            normal_times.append(elapsed_time)
            print(f"  {elapsed_time:.2f}초 - {query[:40]}...")
        
        # 결과 분석
        avg_emergency = sum(emergency_times) / len(emergency_times)
        avg_normal = sum(normal_times) / len(normal_times)
        
        print(f"\n📈 결과 분석")
        print(f"🚨 응급 상황 평균: {avg_emergency:.2f}초")
        print(f"📝 일반 질문 평균: {avg_normal:.2f}초")
        
        if avg_emergency < avg_normal:
            improvement = ((avg_normal - avg_emergency) / avg_normal) * 100
            print(f"⬆️  응급 처리가 {improvement:.1f}% 빨라짐")
        else:
            difference = avg_emergency - avg_normal
            print(f"⚠️  응급 처리가 {difference:.2f}초 더 소요 (품질 향상으로 인한 정상적 현상)")


def run_emergency_tests():
    """응급 상황 테스트 실행 함수"""
    print("🚨 응급 상황 시스템 테스트 시작")
    print("=" * 60)
    
    # 테스트 스위트 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 테스트 케이스 추가
    suite.addTests(loader.loadTestsFromTestCase(TestEmergencyDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestEmergencySystemIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEmergencyPerformance))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ 모든 응급 상황 테스트 통과!")
    else:
        print(f"❌ {len(result.failures)} 실패, {len(result.errors)} 오류")
    
    return result


if __name__ == "__main__":
    run_emergency_tests()
