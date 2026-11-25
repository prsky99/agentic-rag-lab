"""관측성 모듈 (10장).

* semconv.py — span 속성 이름의 단일 출처
* tracing.py — 게이트 호출을 span으로 감쌉니다
* metrics.py — RED + 도메인 지표 기록

services/common/otel.py(2.5절)의 범용 ``stage()`` 헬퍼 위에 RAG 전용 속성·지표를 더합니다.
opentelemetry가 설치되지 않은 환경에서는 no-op으로 동작합니다.
"""
