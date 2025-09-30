# 차량 매뉴얼 RAG 시스템 - Node 구성 및 흐름도

## 전체 시스템 아키텍처

```mermaid
graph TD
    A[사용자 입력] --> B{인터페이스 선택}
    B -->|터미널| C[Terminal Interface]
    B -->|웹| D[Gradio Interface]
    
    C --> E[VehicleManualAgentSubGraph]
    D --> E
    
    E --> F[LangGraph Workflow]
    
    F --> G[Speech Recognition Node]
    G --> H[Emergency Detection Node]
    H --> I[Search Pipeline Node]
    I --> J[Answer Generation Node]
    J --> K[Driving Context Node]
    K --> L[최종 답변]
    
    L --> M{인터페이스 타입}
    M -->|터미널| N[콘솔 출력]
    M -->|웹| O[웹 UI 표시]
```

## SubGraph 상세 구조

```mermaid
graph TD
    subgraph "Main Agent (VehicleManualAgentSubGraph)"
        A[사용자 쿼리] --> B[Speech Recognition Wrapper]
        B --> C[Emergency Detection Wrapper]
        C --> D[Search Pipeline Wrapper]
        D --> E[Answer Generation Wrapper]
        E --> F[Driving Context Wrapper]
        F --> G[최종 답변]
    end
    
    subgraph "Speech Recognition SubGraph"
        B1[Audio Data Processing]
        B2[DummyASR Transcription]
        B3[Text Validation]
        B1 --> B2 --> B3
    end
    
    subgraph "Emergency Detection SubGraph"
        C1[Emergency Classifier]
        C2[Priority Level Analysis]
        C3[Search Strategy Selection]
        C1 --> C2 --> C3
    end
    
    subgraph "Search Pipeline SubGraph"
        D1[Query Analyzer]
        D2[Search Method Selector]
        D3[Document Retrieval]
        D4[Re-ranking/Compression]
        D1 --> D2 --> D3 --> D4
    end
    
    subgraph "Answer Generation SubGraph"
        E1[Context Builder]
        E2[Answer Generator]
        E3[Answer Evaluator]
        E1 --> E2 --> E3
    end
    
    subgraph "Driving Context SubGraph"
        F1[Driving Context Detector]
        F2[Answer Compressor]
        F3[Final Answer Formatter]
        F1 --> F2 --> F3
    end
    
    B --> B1
    C --> C1
    D --> D1
    E --> E1
    F --> F1
```

## 검색 파이프라인 상세 흐름

```mermaid
graph TD
    A[사용자 쿼리] --> B{응급 상황?}
    
    B -->|Yes| C[응급 모드]
    B -->|No| D[일반 모드]
    
    C --> E[Hybrid Keyword Search]
    D --> F[Query Analysis]
    
    F --> G{검색 전략 선택}
    G -->|General| H[Hybrid Semantic]
    G -->|Specific| I[Vector Only]
    G -->|Keyword| J[BM25 Only]
    G -->|Multi-query| K[Multi Query]
    
    E --> L[Document Retrieval]
    H --> L
    I --> L
    J --> L
    K --> L
    
    L --> M{재순위화/압축?}
    M -->|Yes| N[Cross-Encoder Reranking]
    M -->|No| O[Raw Results]
    
    N --> P[Contextual Compression]
    O --> Q[Search Results]
    P --> Q
```

## 콜백 시스템

```mermaid
graph TD
    A[사용자 쿼리] --> B[Performance Monitoring Handler]
    A --> C[Real-time Notification Handler]
    A --> D[Alert Handler]
    
    B --> E[실행 시간 측정]
    B --> F[토큰 사용량 추적]
    B --> G[성능 통계 수집]
    
    C --> H[진행 상황 표시]
    C --> I[실시간 알림]
    
    D --> J[토큰 제한 모니터링]
    D --> K[비용 제한 모니터링]
    D --> L[알림 발송]
    
    E --> M[성능 리포트]
    F --> M
    G --> M
    
    H --> N[사용자 피드백]
    I --> N
    
    J --> O[사용량 경고]
    K --> O
    L --> O
```

## 인터페이스 구조

```mermaid
graph TD
    subgraph "Terminal Interface"
        A1[run_terminal_interface]
        A2[사용자 입력 대기]
        A3[에이전트 쿼리 실행]
        A4[결과 출력]
        A5[통계 표시]
        A1 --> A2 --> A3 --> A4 --> A5
    end
    
    subgraph "Gradio Interface"
        B1[GradioVehicleChatbot]
        B2[웹 UI 생성]
        B3[채팅 인터페이스]
        B4[사이드바 정보]
        B5[예시 질문 버튼]
        B6[통계 표시]
        B1 --> B2
        B2 --> B3
        B2 --> B4
        B2 --> B5
        B2 --> B6
    end
    
    A3 --> C[VehicleManualAgentSubGraph]
    B3 --> C
```

## 데이터 흐름

```mermaid
graph LR
    A[PDF 문서] --> B[Document Loader]
    B --> C[Vector Store]
    B --> D[BM25 Index]
    
    C --> E[Semantic Search]
    D --> F[Keyword Search]
    
    E --> G[Hybrid Retriever]
    F --> G
    
    G --> H[Search Results]
    H --> I[Re-ranking]
    I --> J[Context Compression]
    J --> K[Answer Generation]
    K --> L[Driving Context Processing]
    L --> M[Final Answer]
```

## 주요 특징

1. **SubGraph 아키텍처**: 각 기능을 독립적인 SubGraph로 모듈화
2. **응급 상황 감지**: 자동으로 응급 상황을 감지하고 우선순위 처리
3. **하이브리드 검색**: 벡터 검색과 키워드 검색을 결합
4. **주행 중 최적화**: 운전 중 상황을 감지하여 답변을 압축
5. **음성 인식 지원**: 오디오 입력을 텍스트로 변환
6. **실시간 모니터링**: 성능 및 사용량을 실시간으로 추적
7. **다중 인터페이스**: 터미널과 웹 인터페이스 모두 지원
