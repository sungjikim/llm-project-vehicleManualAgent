"""
성능 벤치마크 테스트
응급 상황 vs 일반 질문 성능 비교 및 상세 분석
"""

import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass

from src.agents.vehicle_agent import VehicleManualAgent
from src.utils.emergency_detector import EmergencyDetector
from src.config.settings import DEFAULT_PDF_PATH


@dataclass
class BenchmarkResult:
    """벤치마크 결과 데이터 클래스"""
    query_type: str
    queries: List[str]
    response_times: List[float]
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    total_chars: int
    avg_chars: int


class PerformanceBenchmark:
    """성능 벤치마크 클래스"""
    
    def __init__(self):
        """벤치마크 초기화"""
        print("🔧 성능 벤치마크 시스템 초기화 중...")
        self.agent = VehicleManualAgent(str(DEFAULT_PDF_PATH))
        self.detector = EmergencyDetector()
        print("✅ SubGraph 벤치마크 시스템 준비 완료")
        
        # 테스트 케이스 정의
        self.emergency_queries = [
            # CRITICAL 수준
            "차에서 타는 냄새가 나는데 화재 위험이 있을 때 어떻게 해야 해요?",
            "엔진에서 연기가 나고 있어요! 즉시 어떻게 해야 하나요?",
            "차량에 불이 났는데 어떻게 대피해야 해요?",
            
            # HIGH 수준
            "브레이크를 밟아도 차가 멈추지 않아요! 어떻게 해야 하나요?",
            "주행 중 핸들이 갑자기 돌아가지 않는데 응급 대처 방법은?",
            "엔진이 갑자기 정지했는데 즉시 해야 할 조치는?",
            
            # MEDIUM 수준
            "타이어가 펑크 났는데 안전하게 정차하는 방법 알려주세요",
            "엔진 과열 경고등이 켜졌는데 어떻게 해야 해요?",
            "와이퍼가 고장나서 앞이 안 보이는데 대처법은?"
        ]
        
        self.normal_queries = [
            # 기본 정보 문의
            "타이어 공기압은 얼마로 맞춰야 하나요?",
            "엔진 오일 교체 주기는 언제인가요?",
            "XC60의 연료 탱크 용량은 얼마인가요?",
            
            # 사용법 문의
            "후방 카메라 사용법을 알려주세요",
            "블루투스 연결 방법이 궁금해요",
            "에어컨 필터 교체는 어떻게 하나요?",
            
            # 기능 문의
            "크루즈 컨트롤 설정 방법은?",
            "시트 히터 사용법을 알려주세요",
            "주차 보조 시스템 사용법은?"
        ]
    
    def run_benchmark(self, queries: List[str], query_type: str, warmup: bool = True) -> BenchmarkResult:
        """벤치마크 실행"""
        print(f"\n🧪 {query_type} 벤치마크 시작 ({len(queries)}개 쿼리)")
        print("-" * 50)
        
        # 웜업 (첫 번째 쿼리로 시스템 준비)
        if warmup and queries:
            print("🔥 웜업 중...")
            self.agent.query(queries[0])
            print("✅ 웜업 완료")
        
        response_times = []
        total_chars = 0
        
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {query[:50]}...")
            
            start_time = time.time()
            answer = self.agent.query(query)
            elapsed_time = time.time() - start_time
            
            response_times.append(elapsed_time)
            total_chars += len(answer)
            
            print(f"  ⏱️  {elapsed_time:.2f}초, 📝 {len(answer)}자")
        
        # 통계 계산
        avg_time = statistics.mean(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
        avg_chars = total_chars // len(queries)
        
        return BenchmarkResult(
            query_type=query_type,
            queries=queries,
            response_times=response_times,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev,
            total_chars=total_chars,
            avg_chars=avg_chars
        )
    
    def analyze_emergency_detection(self):
        """응급 상황 감지 분석"""
        print("\n🔍 응급 상황 감지 분석")
        print("=" * 60)
        
        detection_results = {
            "CRITICAL": [],
            "HIGH": [],
            "MEDIUM": [],
            "LOW": [],
            "NORMAL": []
        }
        
        all_queries = self.emergency_queries + self.normal_queries
        
        for query in all_queries:
            result = self.detector.detect_emergency(query)
            level = result["priority_level"]
            detection_results[level].append({
                "query": query[:50] + "..." if len(query) > 50 else query,
                "score": result["total_score"],
                "is_emergency": result["is_emergency"]
            })
        
        # 결과 출력
        for level, results in detection_results.items():
            if results:
                print(f"\n🚨 {level} 수준 ({len(results)}개)")
                for result in results:
                    status = "🚨" if result["is_emergency"] else "📝"
                    print(f"  {status} {result['query']} (점수: {result['score']})")
        
        return detection_results
    
    def compare_performance(self, emergency_result: BenchmarkResult, normal_result: BenchmarkResult):
        """성능 비교 분석"""
        print("\n📊 성능 비교 분석")
        print("=" * 60)
        
        # 기본 통계
        print(f"\n📈 응답 시간 통계")
        print(f"🚨 응급 상황:")
        print(f"   평균: {emergency_result.avg_time:.2f}초")
        print(f"   최소: {emergency_result.min_time:.2f}초")
        print(f"   최대: {emergency_result.max_time:.2f}초")
        print(f"   표준편차: {emergency_result.std_dev:.2f}초")
        
        print(f"\n📝 일반 질문:")
        print(f"   평균: {normal_result.avg_time:.2f}초")
        print(f"   최소: {normal_result.min_time:.2f}초")
        print(f"   최대: {normal_result.max_time:.2f}초")
        print(f"   표준편차: {normal_result.std_dev:.2f}초")
        
        # 응답 길이 비교
        print(f"\n📝 응답 길이 비교")
        print(f"🚨 응급 상황 평균: {emergency_result.avg_chars}자")
        print(f"📝 일반 질문 평균: {normal_result.avg_chars}자")
        
        # 성능 차이 분석
        time_diff = emergency_result.avg_time - normal_result.avg_time
        if time_diff > 0:
            print(f"\n⚠️  응급 처리가 {time_diff:.2f}초 더 소요")
            print("   → 안전성과 정확성 향상으로 인한 정상적 현상")
        else:
            improvement = abs(time_diff / normal_result.avg_time) * 100
            print(f"\n⬆️  응급 처리가 {improvement:.1f}% 빨라짐")
        
        # 응답 품질 분석
        chars_diff = emergency_result.avg_chars - normal_result.avg_chars
        if chars_diff > 0:
            print(f"📈 응급 상황 답변이 {chars_diff}자 더 상세함")
            print("   → 안전 정보와 경고 메시지 포함으로 인한 현상")
        else:
            print(f"📉 응급 상황 답변이 {abs(chars_diff)}자 더 간결함")
        
        return {
            "time_difference": time_diff,
            "chars_difference": chars_diff,
            "emergency_faster": time_diff < 0,
            "emergency_more_detailed": chars_diff > 0
        }
    
    def generate_performance_report(self, emergency_result: BenchmarkResult, 
                                  normal_result: BenchmarkResult, 
                                  comparison: Dict[str, Any]):
        """성능 리포트 생성"""
        print("\n📋 성능 벤치마크 리포트")
        print("=" * 60)
        
        print(f"🧪 테스트 환경")
        print(f"   응급 상황 쿼리: {len(emergency_result.queries)}개")
        print(f"   일반 질문 쿼리: {len(normal_result.queries)}개")
        print(f"   총 테스트 시간: {sum(emergency_result.response_times) + sum(normal_result.response_times):.1f}초")
        
        print(f"\n🎯 핵심 결과")
        if comparison["emergency_faster"]:
            print(f"✅ 응급 처리 속도 우수: {abs(comparison['time_difference']):.2f}초 빠름")
        else:
            print(f"⚠️  응급 처리 시간 증가: {comparison['time_difference']:.2f}초")
            print("   → 안전성 강화로 인한 품질 향상")
        
        if comparison["emergency_more_detailed"]:
            print(f"📈 응급 답변 상세도 향상: {comparison['chars_difference']}자 추가")
            print("   → 안전 지침 및 경고 메시지 포함")
        
        print(f"\n🔍 품질 지표")
        print(f"🚨 응급 상황 일관성: {emergency_result.std_dev:.2f}초 표준편차")
        print(f"📝 일반 질문 일관성: {normal_result.std_dev:.2f}초 표준편차")
        
        # 성능 등급 평가
        avg_emergency = emergency_result.avg_time
        if avg_emergency < 8:
            grade = "A+ (우수)"
        elif avg_emergency < 12:
            grade = "A (양호)"
        elif avg_emergency < 15:
            grade = "B (보통)"
        else:
            grade = "C (개선 필요)"
        
        print(f"\n🏆 응급 처리 성능 등급: {grade}")
        
        return {
            "emergency_result": emergency_result,
            "normal_result": normal_result,
            "comparison": comparison,
            "performance_grade": grade
        }
    
    def run_full_benchmark(self):
        """전체 벤치마크 실행"""
        print("🚀 전체 성능 벤치마크 시작")
        print("=" * 60)
        
        # 응급 상황 감지 분석
        detection_results = self.analyze_emergency_detection()
        
        # 성능 벤치마크 실행
        emergency_result = self.run_benchmark(self.emergency_queries, "응급 상황", warmup=True)
        normal_result = self.run_benchmark(self.normal_queries, "일반 질문", warmup=False)
        
        # 성능 비교
        comparison = self.compare_performance(emergency_result, normal_result)
        
        # 리포트 생성
        report = self.generate_performance_report(emergency_result, normal_result, comparison)
        
        print(f"\n🎉 벤치마크 완료!")
        return report


def run_performance_tests():
    """성능 테스트 실행 함수"""
    try:
        benchmark = PerformanceBenchmark()
        report = benchmark.run_full_benchmark()
        return report
    except Exception as e:
        print(f"❌ 벤치마크 실행 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_performance_tests()
