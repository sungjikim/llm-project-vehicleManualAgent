"""
답변 품질 평가 유틸리티
"""

import re
from typing import Dict, List, Any


class AnswerEvaluator:
    """답변 품질 및 신뢰성 평가 클래스"""
    
    def __init__(self):
        # 품질 평가 기준
        self.quality_criteria = {
            'page_reference': {
                'weight': 0.25,
                'description': '페이지 참조 정보 포함 여부'
            },
            'specific_info': {
                'weight': 0.30,
                'description': '구체적 정보 (수치, 절차) 포함 여부'
            },
            'answer_length': {
                'weight': 0.20,
                'description': '적절한 답변 길이'
            },
            'safety_info': {
                'weight': 0.15,
                'description': '안전 관련 정보 포함 여부'
            },
            'error_free': {
                'weight': 0.10,
                'description': '오류 메시지 없음'
            }
        }
        
        # 신뢰성 키워드
        self.specific_keywords = [
            '압력', '주기', '방법', '절차', '단계', '온도', '용량', 
            '거리', '시간', '속도', 'km', 'psi', 'bar', '리터', '도'
        ]
        
        self.safety_keywords = [
            '안전', '주의', '경고', '위험', '즉시', '전문가', '서비스센터',
            '점검', '확인', '조치', '대처'
        ]
    
    def evaluate_answer(self, question: str, answer: str, search_results: List[Dict] = None) -> Dict[str, Any]:
        """답변 종합 평가"""
        scores = {}
        
        # 1. 페이지 참조 평가
        scores['page_reference'] = self._evaluate_page_reference(answer)
        
        # 2. 구체적 정보 평가
        scores['specific_info'] = self._evaluate_specific_info(answer)
        
        # 3. 답변 길이 평가
        scores['answer_length'] = self._evaluate_answer_length(answer)
        
        # 4. 안전 정보 평가
        scores['safety_info'] = self._evaluate_safety_info(answer)
        
        # 5. 오류 없음 평가
        scores['error_free'] = self._evaluate_error_free(answer)
        
        # 6. 검색 결과 품질 평가 (새로 추가)
        if search_results:
            scores['search_quality'] = self._evaluate_search_quality(search_results)
            # 가중치 재조정 (검색 품질 포함)
            self.quality_criteria['search_quality'] = {
                'weight': 0.15,
                'description': '검색 결과 품질 및 관련성'
            }
            # 기존 가중치들을 약간 조정
            self.quality_criteria['page_reference']['weight'] = 0.20
            self.quality_criteria['specific_info']['weight'] = 0.25
            self.quality_criteria['answer_length']['weight'] = 0.15
            self.quality_criteria['safety_info']['weight'] = 0.15
            self.quality_criteria['error_free']['weight'] = 0.10
        
        # 가중 평균 계산
        total_score = sum(
            scores[criterion] * self.quality_criteria[criterion]['weight']
            for criterion in scores
        )
        
        # 신뢰도 등급 결정
        reliability_grade = self._get_reliability_grade(total_score)
        
        return {
            'total_score': round(total_score, 2),
            'max_score': 1.0,
            'percentage': round(total_score * 100, 1),
            'reliability_grade': reliability_grade,
            'detailed_scores': scores,
            'evaluation_summary': self._generate_summary(scores, total_score)
        }
    
    def _evaluate_page_reference(self, answer: str) -> float:
        """페이지 참조 정보 평가"""
        if '📚' in answer and '페이지' in answer:
            # 구체적 페이지 번호가 있는지 확인
            page_pattern = r'페이지[:\s]*(\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)'
            if re.search(page_pattern, answer):
                return 1.0
            else:
                return 0.5  # 페이지 언급은 있지만 구체적 번호 없음
        return 0.0
    
    def _evaluate_specific_info(self, answer: str) -> float:
        """구체적 정보 포함 여부 평가"""
        score = 0.0
        answer_lower = answer.lower()
        
        # 구체적 키워드 확인
        keyword_count = sum(1 for keyword in self.specific_keywords if keyword in answer_lower)
        if keyword_count >= 3:
            score += 0.6
        elif keyword_count >= 1:
            score += 0.3
        
        # 숫자 정보 확인
        number_pattern = r'\d+(?:\.\d+)?(?:\s*[a-zA-Z%]+)?'
        numbers = re.findall(number_pattern, answer)
        if len(numbers) >= 2:
            score += 0.4
        elif len(numbers) >= 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_answer_length(self, answer: str) -> float:
        """답변 길이 적절성 평가"""
        length = len(answer)
        if 200 <= length <= 800:
            return 1.0
        elif 100 <= length < 200 or 800 < length <= 1200:
            return 0.7
        elif 50 <= length < 100 or 1200 < length <= 1500:
            return 0.4
        else:
            return 0.1
    
    def _evaluate_safety_info(self, answer: str) -> float:
        """안전 관련 정보 평가"""
        answer_lower = answer.lower()
        safety_count = sum(1 for keyword in self.safety_keywords if keyword in answer_lower)
        
        if safety_count >= 3:
            return 1.0
        elif safety_count >= 2:
            return 0.7
        elif safety_count >= 1:
            return 0.4
        else:
            return 0.0
    
    def _evaluate_error_free(self, answer: str) -> float:
        """오류 메시지 없음 평가"""
        error_indicators = ['오류', '실패', '찾을 수 없습니다', 'error', '없습니다']
        answer_lower = answer.lower()
        
        for indicator in error_indicators:
            if indicator in answer_lower:
                return 0.0
        return 1.0
    
    def _evaluate_search_quality(self, search_results: List[Dict]) -> float:
        """검색 결과 품질 평가"""
        if not search_results:
            return 0.0
        
        score = 0.0
        
        # 검색 결과 개수 평가 (적절한 개수)
        result_count = len(search_results)
        if 3 <= result_count <= 5:
            score += 0.3
        elif 1 <= result_count <= 7:
            score += 0.2
        
        # 페이지 정보 포함 비율
        page_info_count = sum(1 for result in search_results if result.get('page', 0) > 0)
        if page_info_count > 0:
            page_ratio = page_info_count / result_count
            score += 0.3 * page_ratio
        
        # 검색 점수 평가 (있는 경우)
        scores = [result.get('score', 0) for result in search_results if result.get('score', 0) > 0]
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score > 0.7:
                score += 0.2
            elif avg_score > 0.5:
                score += 0.1
        
        # 내용 길이 평가 (너무 짧거나 길지 않은지)
        content_lengths = [len(result.get('content', '')) for result in search_results]
        if content_lengths:
            avg_length = sum(content_lengths) / len(content_lengths)
            if 100 <= avg_length <= 500:
                score += 0.2
            elif 50 <= avg_length <= 800:
                score += 0.1
        
        return min(score, 1.0)
    
    def _get_reliability_grade(self, score: float) -> str:
        """신뢰도 등급 결정"""
        if score >= 0.9:
            return "매우 높음 (A+)"
        elif score >= 0.8:
            return "높음 (A)"
        elif score >= 0.7:
            return "양호 (B)"
        elif score >= 0.6:
            return "보통 (C)"
        elif score >= 0.5:
            return "낮음 (D)"
        else:
            return "매우 낮음 (F)"
    
    def _generate_summary(self, scores: Dict[str, float], total_score: float) -> str:
        """평가 요약 생성"""
        strong_points = []
        weak_points = []
        
        for criterion, score in scores.items():
            description = self.quality_criteria[criterion]['description']
            if score >= 0.8:
                strong_points.append(description)
            elif score <= 0.3:
                weak_points.append(description)
        
        summary = f"전체 점수: {total_score:.2f}/1.0 ({total_score*100:.1f}%)\n"
        
        if strong_points:
            summary += f"✅ 강점: {', '.join(strong_points)}\n"
        
        if weak_points:
            summary += f"⚠️ 개선점: {', '.join(weak_points)}"
        
        return summary
    
    def batch_evaluate(self, qa_pairs: List[Dict]) -> Dict[str, Any]:
        """여러 답변 일괄 평가"""
        evaluations = []
        
        for qa_pair in qa_pairs:
            question = qa_pair.get('question', '')
            answer = qa_pair.get('answer', '')
            search_results = qa_pair.get('search_results', [])
            
            evaluation = self.evaluate_answer(question, answer, search_results)
            evaluation['question'] = question
            evaluations.append(evaluation)
        
        # 전체 통계 계산
        total_scores = [eval['total_score'] for eval in evaluations]
        avg_score = sum(total_scores) / len(total_scores) if total_scores else 0
        
        grade_distribution = {}
        for evaluation in evaluations:
            grade = evaluation['reliability_grade']
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
        
        return {
            'evaluations': evaluations,
            'summary': {
                'total_questions': len(evaluations),
                'average_score': round(avg_score, 2),
                'average_percentage': round(avg_score * 100, 1),
                'grade_distribution': grade_distribution,
                'high_quality_count': len([e for e in evaluations if e['total_score'] >= 0.8])
            }
        }
