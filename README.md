# 🚗 Vehicle Manual RAG System

LangChain과 LangGraph를 활용한 모듈화된 차량 매뉴얼 RAG(Retrieval-Augmented Generation) 에이전트

## ✨ 주요 기능

- 🔍 **하이브리드 검색**: 벡터 검색 + BM25 키워드 검색
- 🚀 **쿼리 확장**: 차량 전문 용어 매핑 및 다중 쿼리 생성
- 🎯 **재순위화**: Cross-Encoder 모델을 활용한 문서 재순위화
- 📝 **맥락 압축**: LLM 기반 핵심 정보 추출
- 🤖 **Few-shot 프롬프팅**: 일관된 고품질 답변 생성
- 🔍 **실시간 신뢰도 평가**: 6가지 기준의 객관적 답변 신뢰도 평가
- 📊 **투명한 품질 지표**: 모든 답변에 신뢰도 퍼센트와 등급 포함

## 🏗️ 프로젝트 구조

```
project/
├── src/                           # 소스 코드
│   ├── agents/                    # 메인 에이전트
│   │   └── vehicle_agent.py       # 차량 매뉴얼 RAG 에이전트
│   ├── config/                    # 설정 및 상수
│   │   └── settings.py            # 시스템 설정값
│   ├── models/                    # 데이터 모델
│   │   └── state.py               # LangGraph 상태 정의
│   ├── tools/                     # 검색 도구
│   │   └── search_tools.py        # 다양한 검색 도구들
│   ├── retrievers/                # 리트리버 관리자
│   │   ├── vector_retriever.py    # 벡터 검색
│   │   ├── hybrid_retriever.py    # 하이브리드 검색
│   │   └── compression_retriever.py # 압축/재순위화
│   ├── prompts/                   # 프롬프트 템플릿
│   │   └── templates.py           # Few-shot 프롬프트
│   └── utils/                     # 유틸리티
│       ├── document_loader.py     # PDF 문서 로딩
│       └── answer_evaluator.py    # 답변 품질 평가
├── data/                          # 데이터 파일 (PDF 등)
├── main.py                        # 메인 실행 파일
└── requirements.txt               # 필요 패키지 목록
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 필요 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# env.example을 .env로 복사
cp env.example .env

# .env 파일을 열어서 API 키 설정
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. PDF 파일 준비

`data/backup/` 폴더에 차량 매뉴얼 PDF 파일을 배치하세요.

### 4. 실행

```bash
python main.py
```

## 🔧 시스템 구성요소

### 검색 전략
- **General**: 일반적인 정보 요청 (의미론적 검색 우선)
- **Specific**: 구체적인 수치/사양 요청 (균형 검색)
- **Troubleshooting**: 문제 해결 (키워드 검색 우선)

### 검색 방법
- **Vector Search**: 의미론적 유사도 검색
- **BM25 Search**: 키워드 기반 검색 (한국어 토크나이저 지원)
- **Hybrid Search**: 벡터 + BM25 앙상블 (가중치 조정 가능)
- **Multi Query**: 다중 쿼리 생성 후 검색
- **Query Expansion**: 차량 전문 용어 확장 검색

## 🔍 지능형 검색 시스템

### 📋 **상황별 검색기 자동 선택**

시스템은 질문의 특성을 분석하여 **3가지 검색 전략**과 **8가지 검색 방법** 중 최적의 조합을 자동으로 선택합니다.

#### 🎯 **검색 전략 (Search Strategy)**

| 전략 | 적용 상황 | 예시 질문 | 선택되는 검색 방법 |
|------|-----------|-----------|-------------------|
| **General** | 일반적인 정보 요청 | "엔진 오일 교체는 언제?" | `hybrid_semantic` (7:3) |
| **Specific** | 구체적인 수치/사양 | "XC60의 연료 탱크 용량은?" | `hybrid_balanced` (5:5) |
| **Troubleshooting** | 문제 해결/고장 진단 | "브레이크 경고등이 켜져요" | `hybrid_keyword` (3:7) |

#### 🔧 **검색 방법 (Search Method)**

| 방법 | 벡터:키워드 비율 | 특징 | 최적 사용 상황 |
|------|------------------|------|----------------|
| **vector_only** | 100:0 | 순수 의미론적 검색 | 개념적/추상적 질문 |
| **hybrid_semantic** | 70:30 | 의미 우선 하이브리드 | 일반적 정보 질문 |
| **hybrid_balanced** | 50:50 | 균형잡힌 하이브리드 | 구체적 사양 질문 |
| **hybrid_keyword** | 30:70 | 키워드 우선 하이브리드 | 문제 해결/진단 |
| **bm25_only** | 0:100 | 순수 키워드 검색 | 정확한 용어 매칭 |
| **multi_query** | 다중 쿼리 | LLM이 3개 쿼리 생성 | 복잡하고 긴 질문 |
| **expanded_query** | 용어 확장 | 전문 용어 동의어 추가 | 전문 용어 포함 질문 |

### 📝 **실제 검색 예시**

#### 예시 1: 일반적 정보 질문
```
Q: "엔진 오일은 언제 교체해야 하나요?"

🔍 분석 결과:
- 검색 전략: General
- 검색 방법: hybrid_semantic (벡터 70% + 키워드 30%)
- 이유: 일반적인 정보 요청으로 의미론적 검색이 효과적

🔧 검색 과정:
1. 벡터 검색: "엔진 오일 교체 주기" 관련 의미론적 유사 문서
2. 키워드 검색: "오일", "교체", "주기" 키워드 매칭
3. 앙상블 결합: 70:30 비율로 결과 통합
4. 재순위화: Cross-Encoder로 관련성 재평가
```

#### 예시 2: 구체적 사양 질문
```
Q: "XC60의 연료 탱크 용량은 얼마인가요?"

🔍 분석 결과:
- 검색 전략: Specific
- 검색 방법: hybrid_balanced (벡터 50% + 키워드 50%)
- 이유: 구체적 수치 정보로 의미와 키워드 모두 중요

🔧 검색 과정:
1. 벡터 검색: "연료 탱크 용량" 개념적 유사성
2. 키워드 검색: "XC60", "연료", "탱크", "용량" 정확 매칭
3. 앙상블 결합: 50:50 균형 비율로 통합
4. 맥락 압축: 핵심 정보만 추출하여 토큰 최적화
```

#### 예시 3: 문제 해결 질문
```
Q: "브레이크 경고등이 켜졌는데 어떻게 해야 하나요?"

🔍 분석 결과:
- 검색 전략: Troubleshooting
- 검색 방법: hybrid_keyword (벡터 30% + 키워드 70%)
- 이유: 긴급 상황으로 정확한 키워드 매칭이 중요

🔧 검색 과정:
1. 키워드 검색: "브레이크", "경고등", "켜짐" 정확 매칭 우선
2. 벡터 검색: 유사한 문제 상황 문서 보조 검색
3. 앙상블 결합: 30:70 비율로 키워드 우선 통합
4. Cross-Encoder 재순위화: 긴급성과 안전성 기준 재평가
```

#### 예시 4: 복잡한 질문 (쿼리 확장)
```
Q: "겨울철 운전할 때 타이어 관리하는 방법을 자세히 알려주세요."

🔍 분석 결과:
- 검색 전략: General
- 검색 방법: expanded_query (전문 용어 확장)
- 이유: 긴 질문(20자 이상)으로 용어 확장 필요

🔧 검색 과정:
1. 용어 확장: "타이어" → ["타이어", "공기압", "PSI", "bar", "휠", "바퀴"]
2. 확장 쿼리: "겨울철 운전 타이어 공기압 PSI bar 휠 바퀴 관리"
3. 하이브리드 검색: 확장된 쿼리로 의미론적 + 키워드 검색
4. 결과 통합: 다양한 관련 정보 종합 제공
```

### ⚡ **검색 최적화 기법**

#### 1. **동적 가중치 조정**
```python
# 질문 유형에 따른 자동 가중치 설정
if "어떻게" in query or "방법" in query:
    weight = 0.7  # 의미론적 검색 우선
elif "얼마" in query or "몇" in query:
    weight = 0.5  # 균형 검색
elif "경고등" in query or "문제" in query:
    weight = 0.3  # 키워드 검색 우선
```

#### 2. **다단계 검색 파이프라인**
```
1차: 기본 검색 (벡터 + BM25)
    ↓
2차: 재순위화 (Cross-Encoder)
    ↓
3차: 맥락 압축 (임베딩 필터링 + LLM 추출)
    ↓
최종: 고품질 검색 결과
```

#### 3. **검색 품질 평가**
- **검색 결과 개수**: 3-5개 최적
- **페이지 정보 비율**: 높을수록 신뢰성 향상
- **평균 관련도 점수**: 0.7 이상 고품질
- **내용 길이**: 100-500자 적정

### 고급 기능
- **Cross-Encoder Reranking**: BAAI/bge-reranker-v2-m3 모델 사용
- **Contextual Compression**: 임베딩 필터링 + 중복 제거 + LLM 추출
- **Few-shot Prompting**: 일관된 답변 스타일과 구조 (8개 예시)

## 📊 성능 지표

- **성공률**: 100% (모든 질문에 의미있는 답변)
- **평균 신뢰도**: 68-75% (보통-양호 수준)
- **평균 응답시간**: ~8초
- **지원 언어**: 한국어 (Kiwi 토크나이저)
- **신뢰도 투명성**: 모든 답변에 6가지 기준 평가 결과 포함

## 🛠️ 사용 예시

### 🔧 **기본 사용법**

```python
from src.agents.vehicle_agent import VehicleManualAgent

# 에이전트 초기화
agent = VehicleManualAgent("path/to/your/manual.pdf")

# 질문하기
answer = agent.query("타이어 공기압은 얼마로 맞춰야 하나요?")
print(answer)
```

### 🎯 **검색 전략별 사용 예시**

```python
# 1. 일반적 정보 질문 (General Strategy)
answer1 = agent.query("엔진 오일은 언제 교체해야 하나요?")
# → hybrid_semantic (70:30) 자동 선택

# 2. 구체적 사양 질문 (Specific Strategy)  
answer2 = agent.query("XC60의 최대 속도는 얼마인가요?")
# → hybrid_balanced (50:50) 자동 선택

# 3. 문제 해결 질문 (Troubleshooting Strategy)
answer3 = agent.query("브레이크 경고등이 빨갛게 켜졌어요")
# → hybrid_keyword (30:70) 자동 선택

# 4. 복잡한 질문 (Query Expansion)
answer4 = agent.query("겨울철 안전 운전을 위한 차량 점검 사항을 자세히 알려주세요")
# → expanded_query 자동 선택
```

## 🔍 신뢰도 평가 시스템

### 📊 **실시간 신뢰도 평가**

모든 답변에는 **6가지 기준**으로 평가된 신뢰도가 자동으로 포함됩니다:

#### 📝 **답변 예시**

```
Q: 타이어 공기압은 얼마로 맞춰야 하나요?

A: 타이어 공기압은 차량 모델에 따라 다르지만, 일반적으로 볼보 XC60의 권장 공기압은 다음과 같습니다:

- **앞 타이어**: 2.3 bar (33 psi)
- **뒤 타이어**: 2.1 bar (30 psi)

정확한 공기압은 차량의 운전석 도어 프레임 또는 사용자 매뉴얼에서 확인할 수 있습니다.

⚠️ **안전상 주의사항**: 타이어 공기압이 너무 낮거나 높으면 주행 안전에 영향을 미칠 수 있으니, 정기적으로 점검하시기 바랍니다.

🔍 **답변 신뢰도**: 68.0% (보통 (C))
⚠️ 추가 확인을 권장합니다.
```

### 📋 **신뢰도 평가 기준**

| 기준 | 가중치 | 설명 |
|------|--------|------|
| **페이지 참조** | 20% | 매뉴얼 페이지 번호 포함 여부 |
| **구체적 정보** | 25% | 수치, 절차, 방법 등 구체적 내용 |
| **답변 길이** | 15% | 적절한 상세도 (200-800자) |
| **안전 정보** | 15% | 주의사항, 전문가 상담 권유 포함 |
| **오류 없음** | 10% | 에러 메시지나 불확실한 표현 없음 |
| **검색 품질** | 15% | 검색된 문서의 관련성과 품질 |

### 🎯 **신뢰도 등급 시스템**

| 등급 | 점수 범위 | 의미 | 권장 조치 |
|------|-----------|------|-----------|
| **A+** | 90-100% | 매우 높음 | ✅ 높은 신뢰도의 답변 |
| **A** | 80-89% | 높음 | ✅ 신뢰할 수 있는 답변 |
| **B** | 70-79% | 양호 | ✅ 일반적으로 신뢰 가능 |
| **C** | 60-69% | 보통 | ⚠️ 추가 확인을 권장 |
| **D** | 50-59% | 낮음 | ⚠️ 신중한 검토 필요 |
| **F** | 0-49% | 매우 낮음 | ❌ 전문가 상담을 강력히 권장 |

### 💡 **신뢰도 기반 안전 가이드**

시스템은 신뢰도에 따라 자동으로 적절한 안내를 제공합니다:

- **80% 이상**: "✅ 높은 신뢰도의 답변입니다."
- **60-80%**: "⚠️ 추가 확인을 권장합니다."
- **60% 미만**: "❌ 전문가 상담을 강력히 권장합니다."

### 🔧 **프로그래밍 방식 사용**

```python
from src.utils.answer_evaluator import AnswerEvaluator

# 답변 평가
evaluator = AnswerEvaluator()
evaluation = evaluator.evaluate_answer(question, answer, search_results)

# 결과 확인
print(f"신뢰도: {evaluation['percentage']}%")
print(f"등급: {evaluation['reliability_grade']}")
print(f"상세 점수: {evaluation['detailed_scores']}")

# 에이전트 사용 시 자동 포함
agent = VehicleManualAgent("path/to/manual.pdf")
answer = agent.query("타이어 공기압은?")
# 답변에 신뢰도 정보가 자동으로 포함됨
```

## 🔍 주요 개선사항

1. **모듈화된 아키텍처**: 관심사 분리와 확장성
2. **다단계 검색**: 벡터 → 하이브리드 → 재순위화 → 압축
3. **한국어 최적화**: Kiwi 토크나이저 및 전문 용어 매핑
4. **실시간 신뢰도 평가**: 6가지 기준의 객관적 품질 평가
5. **투명한 품질 지표**: 모든 답변에 신뢰도 퍼센트와 등급 자동 포함
6. **안전 중심 설계**: 신뢰도 기반 전문가 상담 권유 시스템

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## ⚠️ 주의사항

- **API 키 필수**: OpenAI API 키가 필요합니다
- **저작권 확인**: PDF 파일은 저작권을 확인 후 사용하세요
- **초기 설정 시간**: 벡터 데이터베이스 생성 시 시간이 소요될 수 있습니다
- **신뢰도 해석**: 신뢰도는 시스템 평가이며, 실제 정확성과 다를 수 있습니다
- **전문가 상담**: 안전 관련 문제는 반드시 전문가와 상담하세요
- **답변 검증**: 중요한 결정 전에는 공식 매뉴얼이나 서비스 센터에서 확인하세요

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 Issue를 생성해 주세요.
