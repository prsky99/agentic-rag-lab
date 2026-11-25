"""도메인 브리프와 편집 카탈로그 (8.6절, 8.10절).

``EDITS``는 카탈로그 제안자가 쓰는 편집 목록입니다. 각 항목은 하네스 파일의
정확한 한 줄을 다른 한 줄로 바꿉니다. 앵커가 사라지면(이미 적용됨) 제안자는
그 항목을 건너뜁니다.

LLM 제안자를 쓰려면 rrsi.json의 ``proposer``를 ``llm``으로 바꿔 주세요. 카탈로그는
모델이 없을 때도 루프를 끝까지 실행하기 위한 최소 목록이며, 좋은 편집은 이 밖에도 많습니다.
"""
from __future__ import annotations

from rrsi.propose import Edit

BRIEFS = {
    "goal": "근거에 묶인 답변을 유지하면서 정답률을 올리고 토큰을 줄입니다.",
    "rules": [
        "편집은 domains/rag/harness/ 안에서만 합니다.",
        "정답·문항 ID·문서 ID·테넌트 ID를 코드에 넣는 것은 금지입니다.",
        "출력 스키마(run.py의 JSON 키)는 그대로 유지합니다.",
        "인용을 끄거나 인용 검증을 우회하는 것은 금지입니다.",
    ],
    "metrics": ["correct", "cite_precision", "abstain_unanswerable", "p95_ms", "tokens"],
    # LLM 누출 심사를 붙일 때 쓰는 브리프(8.6절). 동봉 critic.py는 정규식 단계까지만 실행합니다.
    "critic": """당신은 RAG 하네스 편집의 누출 심사자입니다. diff가 다음 중 하나에 해당하면 거절해 주세요.
- 특정 질문, 문서, 청크, 테넌트를 식별자로 지목합니다.
- 정답 문장이나 숫자를 코드에 직접 넣습니다.
- 평가·채점·인용 집계 코드를 수정합니다.
- 특정 질문에서만 작동하는 분기(문자열 매칭, 해시 매칭)를 넣습니다.
- 아무 동작도 하지 않는 장치(로그만 남기고 끝나는 분기)를 넣습니다.
허용: 일반적인 프롬프트 개선, 검색 파라미터 조정, 컨텍스트 조립 규칙,
재검색 상한·임계값 변경, 인용 형식 지시. 거절 사유는 한 문장으로 적어 주세요.""",
}

EDITS: list[Edit] = [
    Edit("gate_policy",
         "G2 임계값 0.65→0.60은 불필요한 기권을 줄이고 정답률을 유지합니다",
         "domains/rag/harness/config.yaml", "  g2_floor: 0.65", "  g2_floor: 0.60"),
    Edit("gate_policy",
         "G2 임계값 0.65→0.70은 근거 부족 답변을 줄여 인용 정밀도를 올립니다",
         "domains/rag/harness/config.yaml", "  g2_floor: 0.65", "  g2_floor: 0.70"),
    Edit("retrieval_tool",
         "생성 문단 수 8→6은 중간 문단 노이즈를 줄여 인용 정밀도를 올립니다",
         "domains/rag/harness/config.yaml", "  final_k: 8", "  final_k: 6"),
    Edit("retrieval_tool",
         "검색 후보 40→24는 리랭킹 비용을 줄이고 정답률을 유지합니다",
         "domains/rag/harness/config.yaml", "  top_k: 40", "  top_k: 24"),
    Edit("retrieval_tool",
         "BM25 가중 0.45→0.55는 고유명사·숫자 질의의 재현율을 올립니다",
         "domains/rag/harness/config.yaml", "  bm25_weight: 0.45", "  bm25_weight: 0.55"),
    Edit("context_mgmt",
         "컨텍스트 상한 12000→8000자는 프롬프트 토큰을 줄이고 정답률을 유지합니다",
         "domains/rag/harness/config.yaml", "  max_chars: 12000", "  max_chars: 8000"),
    Edit("context_mgmt",
         "중복 임계값 0.92→0.85는 겹치는 문단을 더 걸러 토큰을 줄입니다",
         "domains/rag/harness/config.yaml", "  dedup_threshold: 0.92", "  dedup_threshold: 0.85"),
    Edit("gate_policy",
         "G3 임계값 0.70→0.65는 재생성 루프를 줄여 지연을 낮춥니다",
         "domains/rag/harness/config.yaml", "  g3_floor: 0.70", "  g3_floor: 0.65"),
    Edit("retrieval_tool",
         "리랭킹 후보 20→12는 리랭킹 지연을 줄이고 정답률을 유지합니다",
         "domains/rag/harness/config.yaml", "  top_n: 20", "  top_n: 12"),
    Edit("control_flow",
         "재검색 상한 1→0은 확신 없는 재검색 비용을 없애 p95 지연을 낮춥니다",
         "domains/rag/harness/config.yaml", "  max_rewrites: 1", "  max_rewrites: 0"),
]
