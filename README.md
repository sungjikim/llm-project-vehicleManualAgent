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

### 고급 기능
- **Cross-Encoder Reranking**: BAAI/bge-reranker-v2-m3 모델 사용
- **Contextual Compression**: 임베딩 필터링 + 중복 제거 + LLM 추출
- **Few-shot Prompting**: 일관된 답변 스타일과 구조

## 📊 성능 지표

- **성공률**: 100% (모든 질문에 의미있는 답변)
- **평균 신뢰도**: 68-75% (보통-양호 수준)
- **평균 응답시간**: ~8초
- **지원 언어**: 한국어 (Kiwi 토크나이저)
- **신뢰도 투명성**: 모든 답변에 6가지 기준 평가 결과 포함

## 🛠️ 사용 예시

```python
from src.agents.vehicle_agent import VehicleManualAgent

# 에이전트 초기화
agent = VehicleManualAgent("path/to/your/manual.pdf")

# 질문하기
answer = agent.query("타이어 공기압은 얼마로 맞춰야 하나요?")
print(answer)
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
