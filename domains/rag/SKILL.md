# RAG 하네스 편집 규칙

## 편집 가능

`domains/rag/harness/` 아래의 모든 파일. 프롬프트, 제어 흐름, 도구 파라미터,
컨텍스트 조립, 게이트 임계값, 메모리 사용, 서브에이전트.

## 금지

1. `domains/rag/data/`, `eval/`, `runs/` 를 읽거나 수정하는 코드.
2. 질문 ID, 문서 ID, 청크 ID, 정답 문자열, 테넌트 ID를 상수로 넣는 것.
3. 출력 스키마(`run.py`의 JSON 키)를 바꾸는 것.
4. 인용을 끄거나, 인용 검증을 우회하는 것.
5. 특정 질문에서만 작동하는 문자열 분기.
6. 아무 동작도 하지 않는 장치(로그만, 주석만, 죽은 분기).

## 태그

각 편집에는 `(component, hypothesis, diff)`를 붙입니다. component는 다음 중 하나:
`prompt`, `gate_policy`, `retrieval_tool`, `context_mgmt`, `control_flow`,
`memory`, `subagent`.

## 가설

"X를 Y로 바꾸면 Z 지표가 오릅니다" 형식으로 씁니다. 지표 이름은
`correct`, `cite_precision`, `abstain_unanswerable`, `p95_ms`, `tokens` 중에서 고릅니다.

## 하네스가 받는 것과 받지 않는 것

받는 것: 질문 텍스트, 테넌트, 인덱스 스냅샷 ID.
받지 않는 것: 정답, 루브릭, 문항 ID의 의미, 다른 테넌트의 문서.
