# 🚗 Vehicle Manual RAG System

LangChain과 LangGraph를 활용한 모듈화된 차량 매뉴얼 RAG(Retrieval-Augmented Generation) 에이전트

## 📊 시스템 아키텍처 다이어그램

> **🔗 [시스템 흐름도 보기](vehicle_agent_flow_diagram.html)** - 브라우저에서 인터랙티브한 다이어그램을 확인하세요!

## ⚡ Quick Start

```bash
# 터미널 모드 (기본)
python main.py

# Gradio 웹 인터페이스
python main.py --gradio

# 도움말 보기
python main.py --help
```

**웹 인터페이스**: `http://localhost:7860` | **터미널 모드**: 명령행 대화형 인터페이스

> 💡 **새로운 기능**: 하나의 `main.py` 파일로 터미널과 Gradio 웹 인터페이스를 모두 지원합니다!

## ✨ 주요 기능

- 🔍 **하이브리드 검색**: 벡터 검색 + BM25 키워드 검색
- 🚀 **쿼리 확장**: 차량 전문 용어 매핑 및 다중 쿼리 생성
- 🎯 **재순위화**: Cross-Encoder 모델을 활용한 문서 재순위화
- 📝 **맥락 압축**: LLM 기반 핵심 정보 추출
- 🤖 **Few-shot 프롬프팅**: 일관된 고품질 답변 생성
- 🔍 **실시간 신뢰도 평가**: 6가지 기준의 객관적 답변 신뢰도 평가
- 📊 **투명한 품질 지표**: 모든 답변에 신뢰도 퍼센트와 등급 포함
- 🚨 **응급 상황 최적화**: 생명 위험 질문에 대한 우선 처리 및 특화 답변
- 🔥 **즉시 인식 헤더**: 답변 첫 줄에 응급 등급 명시 (CRITICAL/HIGH/MEDIUM/LOW)
- 🚗 **주행 상황 감지**: 발화 패턴 분석을 통한 운전 중 상황 자동 감지
- 📱 **스마트 답변 압축**: 주행 중 안전을 위한 핵심 정보만 제공
- 🎤 **음성 인식 지원**: ASR/STT 기반 음성 입력 처리 (확장 가능)
- 🖥️ **이중 인터페이스**: 터미널 및 Gradio 웹 인터페이스 지원
- 🎯 **선택적 실행**: 명령행 인수로 인터페이스 모드 선택

## 🏗️ 프로젝트 구조

```
project/
├── src/                           # 소스 코드
│   ├── agents/                    # 메인 에이전트
│   │   ├── vehicle_agent.py       # 차량 매뉴얼 RAG 에이전트 (기존)
│   │   ├── vehicle_agent_subgraph.py # SubGraph 아키텍처 에이전트
│   │   └── subgraphs/             # SubGraph 모듈들
│   │       ├── emergency_detection_subgraph.py    # 응급 상황 감지 SubGraph
│   │       ├── search_pipeline_subgraph.py        # 검색 파이프라인 SubGraph
│   │       ├── answer_generation_subgraph.py      # 답변 생성 SubGraph
│   │       ├── driving_context_subgraph.py        # 주행 상황 처리 SubGraph
│   │       └── speech_recognition_subgraph.py     # 음성 인식 SubGraph
│   ├── config/                    # 설정 및 상수
│   │   └── settings.py            # 시스템 설정값
│   ├── models/                    # 데이터 모델
│   │   ├── state.py               # LangGraph 상태 정의 (기존)
│   │   └── subgraph_states.py     # SubGraph 상태 정의
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
│       ├── answer_evaluator.py    # 답변 품질 평가
│       ├── emergency_detector.py  # 응급 상황 감지
│       ├── driving_context_detector.py # 주행 상황 감지 및 답변 압축
│       └── callback_handlers.py   # 성능 모니터링
├── tests/                         # 테스트 코드
│   ├── test_emergency_system.py   # 응급 상황 시스템 테스트
│   ├── test_performance_benchmark.py # 성능 벤치마크 테스트
│   └── quick_test.py              # 빠른 테스트
├── data/                          # 데이터 파일 (PDF 등)
├── main.py                        # 통합 메인 실행 파일 (터미널 + Gradio 지원)
├── main_old.py                    # 이전 메인 파일 (백업)
├── run_tests.py                   # 테스트 실행 스크립트
├── vehicle_test_scenarios.py      # 운전자 실제 상황 테스트
├── extended_test_scenarios.py     # 확장 테스트 시나리오
├── test_scenarios.md              # 테스트 시나리오 문서
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

#### 🖥️ **터미널 모드 (기본)**
```bash
# 기본 터미널 인터페이스
python main.py

# 도움말 보기
python main.py --help
```

#### 🌐 **웹 인터페이스 모드 (Gradio)**
```bash
# Gradio 웹 인터페이스 실행
python main.py --gradio

# 커스텀 포트로 실행
python main.py --gradio --port 8080

# 도움말 보기
python main.py --help
```

**웹 인터페이스 접속**: `http://localhost:7860` (기본 포트)

#### 🔄 **인터페이스 선택 가이드**

| 상황 | 추천 모드 | 이유 |
|------|-----------|------|
| **개발/테스트** | 터미널 모드 | 빠른 테스트, 로그 확인 용이 |
| **데모/프레젠테이션** | Gradio 모드 | 시각적 UI, 사용자 친화적 |
| **일반 사용** | Gradio 모드 | 직관적 채팅 인터페이스 |
| **배치 처리** | 터미널 모드 | 스크립트 자동화에 적합 |

#### 🌐 **Gradio 웹 인터페이스 특징**

- **💬 실시간 채팅**: 직관적인 채팅 인터페이스로 질문과 답변
- **📊 성능 모니터링**: 실시간 성능 통계 및 사용량 확인
- **🎤 음성 입력**: 음성 인식 기능 (더미 모드 지원)
- **🗑️ 채팅 관리**: 채팅 히스토리 초기화 및 관리
- **📱 반응형 디자인**: 모바일과 데스크톱 모두 지원
- **🔧 시스템 정보**: SubGraph 아키텍처 및 기능 안내
- **💡 사용법 가이드**: 내장된 사용법 안내 및 예시

### 5. 테스트 실행

```bash
# 빠른 테스트 (핵심 기능만)
python vehicle_test_scenarios.py
```

## 🔧 SubGraph 아키텍처

### 🎯 **핵심 개념**

SubGraph 아키텍처는 복잡한 LangGraph 워크플로우를 **재사용 가능한 모듈**로 분리하여 관리하는 고급 설계 패턴입니다.

### 🏗️ **SubGraph 구조**

```
메인 에이전트 (VehicleManualAgentSubGraph)
├── 🚨 Emergency Detection SubGraph
│   └── emergency_classifier
├── 🔍 Search Pipeline SubGraph  
│   ├── query_analyzer
│   └── search_executor
├── 📝 Answer Generation SubGraph
│   └── answer_generator
└── 🚗 Driving Context SubGraph
    └── driving_context_processor
```

### 🔄 **워크플로우 흐름**

```
START → Speech Recognition → Emergency Detection → Search Pipeline → Answer Generation → Driving Context → END
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

시스템은 질문의 특성을 분석하여 **3가지 검색 전략**과 **7가지 검색 방법** 중 최적의 조합을 자동으로 선택합니다.

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

### 📝 **질문 고도화 (Query Enhancement)**

#### 1. 전문 용어 확장 (Query Expansion)
```
원본 질문: "타이어 공기압은 얼마로 맞춰야 하나요?"

🔧 질문 고도화:
- 차량 전문 용어 매핑: "타이어" → ["타이어", "공기압", "PSI", "bar", "휠", "바퀴"]
- 동의어 확장: "브레이크" → ["브레이크", "제동", "제동장치", "브레이크 패드"]
- 모델명 확장: "XC60" → ["XC60", "볼보", "VOLVO", "차량", "모델"]

📝 처리 과정: 원본 질문 + 확장된 전문 용어들을 결합하여 검색 범위 확대
```

#### 2. 다중 쿼리 생성 (Multi Query Generation)
```
복잡한 질문: "겨울철 안전 운전을 위한 차량 점검 사항을 자세히 알려주세요"

🔧 쿼리 분해:
- 쿼리1: "겨울철 차량 점검"
- 쿼리2: "안전 운전 차량 관리"  
- 쿼리3: "겨울철 타이어 배터리 점검"

📝 처리 과정: 각 쿼리별로 독립 검색 후 결과 통합
```

### 🔍 **질문 분석 (Query Analysis)**

#### 1. General (일반적 정보 질문)
```
Q: "엔진 오일은 언제 교체해야 하나요?"

🔍 분석 결과:
- 질문 유형: 일반적인 정보 요청
- 특징: 개념적, 추상적 질문, "어떻게", "언제", "왜" 등의 질문어 포함
- 분석 방식: LLM이 질문의 의도와 특성을 파악하여 General 전략 선택
```

#### 2. Specific (구체적 사양 질문)
```
Q: "XC60의 연료 탱크 용량은 얼마인가요?"

🔍 분석 결과:
- 질문 유형: 구체적 수치 정보 요청
- 특징: "얼마", "몇", "용량", "크기" 등 수치적 정보 요청
- 분석 방식: LLM이 수치적 정보의 중요성을 인식하여 Specific 전략 선택
```

#### 3. Troubleshooting (문제 해결 질문)
```
Q: "브레이크 경고등이 켜졌는데 어떻게 해야 하나요?"

🔍 분석 결과:
- 질문 유형: 문제 상황 해결 요청
- 특징: "문제", "고장", "경고등", "이상" 등 문제 상황 키워드 포함
- 분석 방식: LLM이 긴급성과 문제 해결 의도를 파악하여 Troubleshooting 전략 선택
```

### 🎯 **검색기 선택 (Retriever Selection)**

#### 1. General 질문 → hybrid_semantic (벡터 70% + 키워드 30%)
```
🔧 검색 과정:
1. 벡터 검색: "엔진 오일 교체 주기" 관련 의미론적 유사 문서
2. 키워드 검색: "오일", "교체", "주기" 키워드 매칭
3. 앙상블 결합: 70:30 비율로 결과 통합
4. 재순위화: Cross-Encoder로 관련성 재평가
```

#### 2. Specific 질문 → hybrid_balanced (벡터 50% + 키워드 50%)
```
🔧 검색 과정:
1. 벡터 검색: "연료 탱크 용량" 개념적 유사성
2. 키워드 검색: "XC60", "연료", "탱크", "용량" 정확 매칭
3. 앙상블 결합: 50:50 균형 비율로 통합
4. 맥락 압축: 핵심 정보만 추출하여 토큰 최적화
```

#### 3. Troubleshooting 질문 → hybrid_keyword (벡터 30% + 키워드 70%)
```
🔧 검색 과정:
1. 키워드 검색: "브레이크", "경고등", "켜짐" 정확 매칭 우선
2. 벡터 검색: 유사한 문제 상황 문서 보조 검색
3. 앙상블 결합: 30:70 비율로 키워드 우선 통합
4. Cross-Encoder 재순위화: 긴급성과 안전성 기준 재평가
```

### 📊 **검색기 종류 및 특징**

| 검색기 | 벡터:키워드 비율 | 특징 | 최적 사용 상황 |
|--------|------------------|------|----------------|
| **vector_only** | 100:0 | 순수 의미론적 검색 | 개념적/추상적 질문 |
| **hybrid_semantic** | 70:30 | 의미 우선 하이브리드 | 일반적 정보 질문 |
| **hybrid_balanced** | 50:50 | 균형잡힌 하이브리드 | 구체적 사양 질문 |
| **hybrid_keyword** | 30:70 | 키워드 우선 하이브리드 | 문제 해결/진단 |
| **bm25_only** | 0:100 | 순수 키워드 검색 | 정확한 용어 매칭 |
| **multi_query** | 다중 쿼리 | LLM이 3개 쿼리 생성 | 복잡하고 긴 질문 |
| **expanded_query** | 용어 확장 | 전문 용어 동의어 추가 | 전문 용어 포함 질문 |

### 🔧 **고급 검색 기법**

#### 1. Cross-Encoder 재순위화
- **모델**: BAAI/bge-reranker-v2-m3
- **목적**: 검색된 문서들의 관련성을 정확히 재평가
- **과정**: 초기 검색 결과 → Cross-Encoder 점수 계산 → 관련성 순으로 재정렬

#### 2. 맥락 압축 (Contextual Compression)
- **목적**: 토큰 사용량 최적화 및 핵심 정보만 추출
- **과정**: 검색 결과 → LLM 기반 핵심 정보 추출 → 압축된 맥락 생성

#### 3. Few-shot 프롬프팅
- **목적**: 일관된 고품질 답변 생성
- **방식**: 8개 예시를 통한 프롬프트 엔지니어링
- **효과**: 답변 스타일과 구조의 일관성 확보

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

### 🚀 **빠른 시작**

#### 터미널 모드
```bash
# 기본 실행
python main.py

# 질문 예시
❓ 질문: 타이어 공기압은 얼마로 맞춰야 하나요?
```

#### Gradio 웹 모드
```bash
# 웹 인터페이스 실행
python main.py --gradio

# 브라우저에서 http://localhost:7860 접속
```

### 🔧 **프로그래밍 사용법**

```python
from src.agents.vehicle_agent_subgraph import VehicleManualAgentSubGraph

# 에이전트 초기화
agent = VehicleManualAgentSubGraph("path/to/your/manual.pdf")

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

### 🚗 **주행 상황별 사용 예시**

```python
# 1. 주행 중 일반 질문 (압축된 답변 제공)
answer1 = agent.query("지금 운전 중인데 엔진오일 교체 주기가 궁금해요")
# → 주행 상황 감지 → 압축된 답변 제공

# 2. 주행 중 긴급 상황 (즉시 대응 답변)
answer2 = agent.query("지금 당장 브레이크가 이상해요!")
# → 주행 + 긴급 감지 → 즉시 대응 지침

# 3. 일반 상황 (상세 답변 유지)
answer3 = agent.query("엔진오일 교체 방법을 자세히 알려주세요")
# → 일반 상황 → 기존 상세 답변 제공
```

### 🎤 **음성 인식 사용 예시**

```python
# 1. 텍스트 입력 (기존 방식)
answer1 = agent.query(user_query="엔진 오일 교체 주기를 알려주세요")
# → 텍스트 쿼리 → 음성 인식 건너뛰기 → 일반 처리

# 2. 음성 데이터 입력 (바이트)
audio_bytes = b"..."  # 실제 음성 데이터
answer2 = agent.query(audio_data=audio_bytes)
# → 음성 인식 → 텍스트 변환 → 일반 처리

# 3. 음성 파일 입력 (파일 경로)
answer3 = agent.query(audio_file_path="voice_input.wav")
# → 음성 파일 처리 → 텍스트 변환 → 일반 처리

# 4. 더미 음성 인식 (테스트용)
answer4 = agent.query()  # audio_data=None, audio_file_path=None
# → 더미 음성 생성 → 텍스트 변환 → 일반 처리
```

#### **🎯 주행 상황 감지 결과 예시**

```python
# 주행 중 질문
🚗 주행 상황 분석 결과:
   • 주행 중 여부: True (신뢰도: 0.78)
   • 긴급도: urgent
   • 감지된 지표: 지금, 운전 중, ~중인데

📱 주행 중 모드 - 답변 압축:
📍 엔진오일 교체: 10,000km마다
1. 현재 주행거리 확인
2. 오일 상태 점검  
3. 서비스센터 예약

📋 주행 후 상세 내용을 확인하세요
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

차량 운전 중 발생할 수 있는 **생명 위험 응급 상황**에 대한 특화된 처리 시스템:

### 🎯 **응급 상황 자동 감지**

#### 📋 **감지 알고리즘**

시스템은 사용자의 질문을 분석하여 응급 상황을 4단계로 분류합니다:

| 위험도 | 키워드 예시 | 가중치 | 우선순위 | 대응 시간 |
|--------|-------------|--------|----------|-----------|
| **CRITICAL** | 화재, 폭발, 사고, 충돌 | 10점 | 최고 | **5초 이내** |
| **HIGH** | 브레이크, 핸들, 엔진정지 | 8점 | 높음 | **8초 이내** |
| **MEDIUM** | 과열, 경고등, 펑크 | 6점 | 중간 | **10초 이내** |
| **LOW** | 배터리, 방전, 연료부족 | 4점 | 낮음 | **12초 이내** |

#### 🔍 **감지 예시**

```python
# CRITICAL 응급 상황
"차에서 연기가 나고 있어요!" 
→ 감지: 화재(10점) + 즉시(3점) = 13점 → CRITICAL

# HIGH 응급 상황  
"브레이크를 밟아도 차가 멈추지 않아요!"
→ 감지: 브레이크(8점) + 어떻게해야(2점) = 10점 → HIGH

# 일반 질문
"타이어 공기압은 얼마로 맞춰야 하나요?"
→ 감지: 없음 = 0점 → NORMAL
```

### ⚡ **응급 최적화 전략**

#### 🔧 **1. 검색 전략 자동 변경**

| 응급 수준 | 검색 방법 | 벡터:키워드 비율 | 압축 방식 | 특징 |
|-----------|-----------|------------------|-----------|------|
| **CRITICAL** | `hybrid_keyword` | 3:7 | `rerank_only` | 키워드 우선, 빠른 재순위화 |
| **HIGH** | `hybrid_keyword` | 3:7 | `rerank_compress_troubleshooting` | 문제해결 중심 압축 |
| **MEDIUM** | `hybrid_balanced` | 5:5 | `rerank_compress_troubleshooting` | 균형잡힌 검색 |
| **LOW** | `hybrid_semantic` | 7:3 | `rerank_compress_general` | 의미적 검색 우선 |

#### 🚀 **2. 처리 속도 최적화**

**응급 상황 전용 워크플로우**:
```
응급 감지 → 즉시 키워드 검색 → 빠른 재순위화 → 안전 중심 답변
    ⏱️ 5초        ⏱️ 2초         ⏱️ 1초        ⏱️ 2초
```

**일반 질문 워크플로우**:
```
일반 분석 → 하이브리드 검색 → 다단계 압축 → 신뢰도 평가 → 답변
    ⏱️ 3초      ⏱️ 4초         ⏱️ 3초        ⏱️ 2초      ⏱️ 1초
```

#### 🎯 **3. 프롬프트 자동 강화**

응급 수준에 따라 프롬프트가 자동으로 강화됩니다:

**CRITICAL 강화 예시**:
```
⚠️ **CRITICAL EMERGENCY** ⚠️
이것은 생명과 직결된 매우 위험한 상황입니다.
답변 시 다음을 최우선으로 해주세요:
1. 즉시 실행 가능한 안전 조치를 첫 번째로 제시
2. 추가 위험 요소 경고
3. 전문가/응급 서비스 연락 강력 권고
4. 단계별 명확한 행동 지침
```

### 📊 **응급 답변 특화 기능**

#### 🚨 **1. 즉시 인식 가능한 헤더 시스템**

모든 답변은 **첫 줄에 상황 등급**을 명확히 표시합니다:

```
🔥 **CRITICAL 응급 상황**    ← 생명 위험 (화재, 폭발)
🚨 **HIGH 응급 상황**       ← 즉시 조치 필요 (브레이크, 조향)  
⚠️ **MEDIUM 응급 상황**     ← 신속 대응 필요 (과열, 펑크)
🔍 **LOW 응급 상황**        ← 주의 필요 (배터리, 연료)
📝 **일반 질문**            ← 정보 요청
```

#### 🚨 **2. 안전 우선 답변 구조**

```
🔥 **CRITICAL 응급 상황**

🚨 즉시 조치 (생명 위험):
1. **엔진 즉시 정지**: 시동 끄고 키 제거
2. **차량 대피**: 모든 승객 차량에서 하차
3. **안전 거리 확보**: 차량에서 최소 50m 떨어지기

⚠️ **절대 금지**: 후드 열기, 물 사용 금지

**즉시 연락**: 119 응급상황 신고
```

#### ⏱️ **3. 신뢰도 평가 간소화**

응급 상황에서는 **속도가 생명**이므로:
- 복잡한 신뢰도 평가 생략
- 기본 85% 신뢰도 적용
- 안전 경고 우선 표시

#### 🔔 **4. 자동 경고 시스템**

```python
# 응급 수준별 자동 경고
if emergency_level == "CRITICAL":
    warning = "⚠️ 생명 위험 상황입니다. 즉시 조치하고 119에 신고하세요."
elif emergency_level == "HIGH":  
    warning = "⚠️ 즉시 안전 조치가 필요합니다. 전문가에게 연락하세요."
```

### 🧪 **응급 상황 테스트 케이스**

시스템에서 자동으로 테스트하는 응급 질문들:

1. **🔥 CRITICAL**: "차에서 타는 냄새가 나는데 화재 위험이 있을 때 어떻게 해야 해요?"
2. **🚨 HIGH**: "브레이크를 밟아도 차가 멈추지 않아요! 어떻게 해야 하나요?"
3. **⚠️ MEDIUM**: "엔진에서 연기가 나고 있는데 즉시 해야 할 조치가 뭐예요?"
4. **🔍 LOW**: "주행 중 핸들이 갑자기 돌아가지 않는데 응급 대처 방법은?"

### 💡 **기술적 구현**

#### 🏗️ **LangGraph 노드 추가**

```python
# 새로운 응급 분류 노드 추가
workflow.add_node("emergency_classifier", self.emergency_classifier)

# 워크플로우: 응급 분류 → 쿼리 분석 → 검색 → 답변 → 주행 상황 처리
workflow.set_entry_point("emergency_classifier")
workflow.add_edge("emergency_classifier", "query_analyzer")
workflow.add_edge("answer_generator", "driving_context_processor")
workflow.add_edge("driving_context_processor", END)
```

#### 🔧 **응급 감지 클래스**

```python
from src.utils.emergency_detector import EmergencyDetector

detector = EmergencyDetector()
result = detector.detect_emergency("브레이크가 안 멈춰요!")

# 결과
{
    "is_emergency": True,
    "priority_level": "HIGH", 
    "total_score": 10,
    "search_strategy": {
        "search_method": "hybrid_keyword",
        "timeout": 8
    }
}
```

이 시스템을 통해 **차량 응급 상황에서 생명을 구할 수 있는** 실시간 지원을 제공합니다! 🚗💨

## 🚗 주행 상황 감지 및 답변 압축

실제 차량 사용 환경에서 **운전자의 안전을 최우선**으로 고려한 지능형 답변 시스템:

### 🔍 **주행 상황 감지**

- **다층 감지 시스템**: 키워드 분석 + LLM 판단
- **3단계 긴급도 분류**: immediate, urgent, normal
- **압축 원칙**: 안전 최우선, 핵심만 전달, 간결한 표현

### 📱 **스마트 답변 압축**

**압축 전 (일반 모드)**:
```
📝 일반 질문

엔진 오일 교체 주기는 10,000km 또는 12개월마다 권장됩니다. 
주행 환경이나 운전 습관에 따라 다를 수 있으니, 정기적으로 
오일 상태를 점검하는 것이 좋습니다.

⚠️ 안전상 주의사항: 운전 중에는 도로 상황에 집중하시고, 
정차 후 오일 상태를 확인하시기 바랍니다.

🔍 답변 신뢰도: 68.0% (보통 (C))
⚠️ 추가 확인을 권장합니다.
```

**압축 후 (주행 모드)**:
```
📍 엔진오일 교체: 10,000km마다
1. 현재 주행거리 확인
2. 오일 상태 점검
3. 서비스센터 예약

📋 주행 후 상세 내용을 확인하세요
```

## 🎤 음성 인식 시스템

### 🎯 **핵심 개념**

실제 차량 환경에서 **음성으로 질문**할 수 있는 확장 가능한 음성 인식 시스템입니다.

### 🔧 **음성 인식 SubGraph 구조**

```
Speech Recognition SubGraph
├── audio_processor (DummyASR/STT)
└── text_validator (텍스트 검증)
```

### 🎵 **지원하는 입력 방식**

| 입력 방식 | 클래스 | 설명 | 확장 가능성 |
|-----------|--------|------|-------------|
| **바이트 데이터** | `DummyASR` | 실시간 음성 스트림 | Whisper, Google Speech-to-Text |
| **파일 경로** | `DummySTT` | 음성 파일 처리 | Azure Speech, AWS Transcribe |
| **더미 모드** | `DummyASR/STT` | 테스트 및 개발용 | 실제 음성 인식으로 교체 |

### 🔄 **음성 인식 워크플로우**

```
음성 입력 → 음성 인식 SubGraph → 텍스트 변환 → 기존 RAG 워크플로우
    ↓
텍스트 입력 → 음성 인식 건너뛰기 → 기존 RAG 워크플로우
```

### 💡 **확장 가이드**

#### **실제 ASR 구현 예시**
```python
class RealASR:
    def __init__(self):
        import whisper
        self.model = whisper.load_model("base")
    
    def transcribe(self, audio_data: bytes) -> str:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
            tmp.write(audio_data)
            result = self.model.transcribe(tmp.name, language='ko')
            return result["text"]
```

#### **실제 STT 구현 예시**
```python
class RealSTT:
    def __init__(self):
        import speech_recognition as sr
        self.recognizer = sr.Recognizer()
    
    def process_audio(self, audio_file_path: str) -> str:
        with sr.AudioFile(audio_file_path) as source:
            audio = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio, language='ko-KR')
            return text
```

### 🧪 **테스트 방법**

```bash
# 음성 인식 기능 테스트
python test_speech_recognition.py

# 메인 시스템에서 음성 인식 모드
python main.py
# 'voice' 명령어 입력
```

### 📊 **성능 특성**

- **텍스트 우선**: 텍스트 입력 시 음성 인식 자동 건너뛰기
- **오류 처리**: 음성 인식 실패 시 적절한 오류 메시지
- **확장성**: 실제 음성 인식 엔진으로 쉽게 교체 가능
- **테스트 지원**: 더미 모드로 개발 및 테스트 용이

이 시스템을 통해 **음성과 텍스트 모두를 지원하는** 통합 차량 어시스턴트가 가능합니다! 🎤🚗

## ⚠️ 주의사항

- **API 키 필수**: OpenAI API 키가 필요합니다
- **저작권 확인**: PDF 파일은 저작권을 확인 후 사용하세요
- **초기 설정 시간**: 벡터 데이터베이스 생성 시 시간이 소요될 수 있습니다
- **신뢰도 해석**: 신뢰도는 시스템 평가이며, 실제 정확성과 다를 수 있습니다
- **전문가 상담**: 안전 관련 문제는 반드시 전문가와 상담하세요
- **답변 검증**: 중요한 결정 전에는 공식 매뉴얼이나 서비스 센터에서 확인하세요

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 Issue를 생성해 주세요.
