# 차량 매뉴얼 RAG 시스템 - Node 구성 및 흐름도 (LLM 기반 지능형 감지)

## 전체 시스템 아키텍처

```mermaid
graph TD
    A[사용자 입력] --> B{인터페이스 선택}
    B -->|터미널| C[Terminal Interface]
    B -->|웹| D[Gradio Interface]
    
    C --> E[VehicleManualAgentSubGraph]
    D --> E
    
    E --> F{LLM 기반 응급 상황 감지}
    F -->|CRITICAL/HIGH| G[응급 상황 빠른 경로]
    F -->|일반 질문| H[일반 워크플로우]
    
    G --> G1[Emergency Detection]
    G1 --> G2[Emergency Search]
    G2 --> G3[Emergency Answer]
    G3 --> I[최종 답변]
    
    H --> H1[Speech Recognition]
    H1 --> H2[Emergency Detection]
    H2 --> H3[Search Pipeline]
    H3 --> H4[Answer Generation]
    H4 --> H5[Driving Context]
    H5 --> I
    
    I --> J{인터페이스 타입}
    J -->|터미널| K[콘솔 출력]
    J -->|웹| L[웹 UI 표시]
```

## SubGraph 상세 구조 (LLM 기반 지능형 감지)

```mermaid
graph TD
    subgraph "Main Agent (VehicleManualAgentSubGraph)"
        A[사용자 쿼리] --> B{LLM 기반 응급 상황 감지}
        B -->|CRITICAL/HIGH| C[응급 상황 빠른 경로]
        B -->|일반 질문| D[일반 워크플로우]
        
        C --> C1[Emergency Detection]
        C1 --> C2[Emergency Search]
        C2 --> C3[Emergency Answer]
        C3 --> F[최종 답변]
        
        D --> D1[Speech Recognition]
        D1 --> D2[Emergency Detection]
        D2 --> D3[Search Pipeline]
        D3 --> D4[Answer Generation]
        D4 --> D5[Driving Context]
        D5 --> F
    end
    
    subgraph "응급 상황 빠른 경로"
        E1[Emergency Detection Wrapper]
        E2[Emergency Search Wrapper]
        E3[Emergency Answer Wrapper]
        E1 --> E2 --> E3
    end
    
    subgraph "일반 워크플로우"
        F1[Speech Recognition Wrapper]
        F2[Emergency Detection Wrapper]
        F3[Search Pipeline Wrapper]
        F4[Answer Generation Wrapper]
        F5[Driving Context Wrapper]
        F1 --> F2 --> F3 --> F4 --> F5
    end
    
    subgraph "Emergency Detection SubGraph (최적화)"
        G1[CRITICAL/HIGH 키워드 우선 검사]
        G2[빠른 응급 상황 판정]
        G3[검색 전략 선택]
        G1 --> G2 --> G3
    end
    
    subgraph "Emergency Search SubGraph (간소화)"
        H1[BM25 키워드 검색만]
        H2[최대 3개 문서]
        H3[압축/재순위화 생략]
        H1 --> H2 --> H3
    end
    
    subgraph "Emergency Answer SubGraph (간소화)"
        I1[간소화된 프롬프트]
        I2[고정 신뢰도 85%]
        I3[신뢰도 평가 생략]
        I1 --> I2 --> I3
    end
    
    C1 --> G1
    C2 --> H1
    C3 --> I1
    D2 --> G1
    D3 --> F3
    D4 --> F4
    D5 --> F5
```

## 검색 파이프라인 상세 흐름 (성능 최적화)

```mermaid
graph TD
    A[사용자 쿼리] --> B{응급 상황 감지}
    
    B -->|CRITICAL/HIGH| C[응급 상황 빠른 경로]
    B -->|일반 질문| D[일반 워크플로우]
    
    C --> C1[CRITICAL/HIGH 키워드 우선 검사]
    C1 --> C2[즉시 응급 상황 판정]
    C2 --> C3[BM25 키워드 검색만]
    C3 --> C4[최대 3개 문서]
    C4 --> C5[압축/재순위화 생략]
    C5 --> C6[간소화된 답변 생성]
    
    D --> D1[Query Analysis]
    D1 --> D2{검색 전략 선택}
    D2 -->|General| D3[Hybrid Semantic]
    D2 -->|Specific| D4[Hybrid Balanced]
    D2 -->|Troubleshooting| D5[Hybrid Keyword]
    D2 -->|Multi-query| D6[Multi Query]
    D2 -->|Expanded| D7[Expanded Query]
    
    D3 --> D8[Document Retrieval]
    D4 --> D8
    D5 --> D8
    D6 --> D8
    D7 --> D8
    
    D8 --> D9{재순위화/압축?}
    D9 -->|Yes| D10[Cross-Encoder Reranking]
    D9 -->|No| D11[Raw Results]
    
    D10 --> D12[Contextual Compression]
    D11 --> D13[Search Results]
    D12 --> D13
    
    C6 --> E[최종 답변]
    D13 --> F[Answer Generation]
    F --> G[Driving Context Processing]
    G --> E
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

## 성능 최적화 비교

```mermaid
graph TD
    subgraph "개선 전 (기존 시스템)"
        A1[사용자 쿼리] --> B1[Speech Recognition]
        B1 --> C1[Emergency Detection]
        C1 --> D1[Search Pipeline]
        D1 --> E1[Answer Generation]
        E1 --> F1[Driving Context]
        F1 --> G1[최종 답변]
        
        H1[응급 질문: 11.5초]
        I1[일반 질문: 8초]
    end
    
    subgraph "개선 후 (최적화 시스템)"
        A2[사용자 쿼리] --> B2{응급 상황 감지}
        B2 -->|CRITICAL/HIGH| C2[응급 빠른 경로]
        B2 -->|일반 질문| D2[일반 워크플로우]
        
        C2 --> C3[Emergency Detection]
        C3 --> C4[Emergency Search]
        C4 --> C5[Emergency Answer]
        C5 --> E2[최종 답변]
        
        D2 --> D3[Speech Recognition]
        D3 --> D4[Emergency Detection]
        D4 --> D5[Search Pipeline]
        D5 --> D6[Answer Generation]
        D6 --> D7[Driving Context]
        D7 --> E2
        
        F2[응급 질문: 2.55초 ⚡]
        G2[일반 질문: 7.30초]
    end
    
    H1 -.->|77.8% 개선| F2
    I1 -.->|8.8% 개선| G2
```

## 성능 개선 상세

```mermaid
graph LR
    subgraph "응급 상황 빠른 경로 최적화"
        A1[기존: 5개 SubGraph] --> A2[개선: 4개 SubGraph]
        B1[기존: 하이브리드 검색] --> B2[개선: BM25만]
        C1[기존: 복잡한 프롬프트] --> C2[개선: 간소화된 프롬프트]
        D1[기존: 신뢰도 평가] --> D2[개선: 고정 신뢰도]
        E1[기존: 모든 키워드 검사] --> E2[개선: CRITICAL/HIGH 우선]
    end
    
    subgraph "성능 향상 결과"
        F1[응답 시간: 11.5초 → 2.55초]
        G1[개선율: 77.8%]
        H1[처리 단계: 20% 감소]
        I1[검색 시간: 60% 단축]
        J1[답변 생성: 50% 단축]
        K1[응급 감지: 70% 단축]
    end
```

## 주요 특징 (성능 최적화 버전)

1. **조건부 워크플로우**: 응급 상황 감지 후 적절한 경로 자동 선택
2. **응급 상황 빠른 경로**: CRITICAL/HIGH 응급 상황 전용 최적화된 처리
3. **간소화된 검색**: 응급 상황에서 BM25 키워드 검색만 사용
4. **최적화된 응급 감지**: CRITICAL/HIGH 키워드 우선 검사
5. **고정 신뢰도**: 응급 상황에서 복잡한 평가 생략
6. **SubGraph 아키텍처**: 각 기능을 독립적인 SubGraph로 모듈화
7. **하이브리드 검색**: 일반 질문에서는 벡터 검색과 키워드 검색 결합
8. **주행 중 최적화**: 운전 중 상황을 감지하여 답변을 압축
9. **실시간 모니터링**: 성능 및 사용량을 실시간으로 추적
10. **다중 인터페이스**: 터미널과 웹 인터페이스 모두 지원

## LLM 기반 지능형 감지 시스템

```mermaid
graph TD
    subgraph "LLM 기반 응급 상황 감지"
        A[사용자 질문] --> B[LLM Emergency Detector]
        B --> C{맥락 분석}
        C -->|생명 위험| D[CRITICAL]
        C -->|즉시 조치 필요| E[HIGH]
        C -->|신속 대응 필요| F[MEDIUM]
        C -->|주의 필요| G[LOW]
        C -->|일반 질문| H[NORMAL]
        
        D --> I[응급 빠른 경로]
        E --> I
        F --> J[일반 워크플로우]
        G --> J
        H --> J
    end
    
    subgraph "LLM 기반 주행 상황 감지"
        K[사용자 발화] --> L[LLM Driving Detector]
        L --> M{주행 지표 분석}
        M -->|명시적 주행 표현| N[주행 중 (95%)]
        M -->|시간적 긴급성| O[주행 중 (75%)]
        M -->|상황적 맥락| P[주행 중 (60%)]
        M -->|일반 질문| Q[주행 중 아님 (30%)]
        
        N --> R[답변 압축 필요]
        O --> R
        P --> R
        Q --> S[상세 답변]
    end
```

## 기존 vs LLM 기반 감지 비교

```mermaid
graph LR
    subgraph "기존 시스템 (키워드 기반)"
        A1[브레이크 패드 관리방법] --> B1[키워드: 브레이크]
        B1 --> C1[응급 상황으로 오분류]
        D1[브레이크가 안 밟혀요] --> E1[키워드: 브레이크]
        E1 --> F1[응급 상황으로 정확 분류]
    end
    
    subgraph "LLM 기반 시스템"
        A2[브레이크 패드 관리방법] --> B2[맥락: 정비 질문]
        B2 --> C2[일반 질문으로 정확 분류]
        D2[브레이크가 안 밟혀요] --> E2[맥락: 기능 고장]
        E2 --> F2[응급 상황으로 정확 분류]
    end
```

## LLM 기반 감지의 핵심 특징

1. **🧠 맥락 이해**: 키워드 매칭이 아닌 의미적 분석으로 정확한 판단
2. **📊 신뢰도 제공**: 각 판단에 대한 신뢰도 점수와 근거 설명
3. **🎯 정확한 분류**: "브레이크 패드 관리방법" → 일반 질문으로 정확 분류
4. **⚡ 성능 최적화**: 응급 질문 응답시간 77.8% 향상 (11.5초 → 2.55초)
5. **🔄 자동 적응**: 새로운 상황 패턴에 자동으로 적응
6. **🚗 주행 감지**: LLM 기반 주행 중 상황 자동 감지 및 답변 압축
7. **📱 스마트 압축**: 주행 중 안전을 위한 핵심 정보만 제공
8. **🔍 하이브리드 검색**: 벡터 + BM25 검색의 조합
9. **📈 실시간 모니터링**: 성능 지표 및 신뢰도 추적
10. **🎤 음성 지원**: ASR/STT 기반 음성 입력 처리
