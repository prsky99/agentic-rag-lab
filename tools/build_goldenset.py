#!/usr/bin/env python3
"""골든셋 샘플 생성기.

코퍼스의 (문서, 절)에서 청크 ID와 본문을 확인해 골든셋을 만듭니다. 수동으로 청크
ID를 적으면 코퍼스가 바뀔 때 조용히 어긋나므로, 생성기로만 만듭니다.

    python3 tools/build_goldenset.py

이 샘플은 42문항입니다. 책의 300문항 골든셋은 같은 형식으로 확장합니다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "domains" / "rag"))

from harness.retrieve import chunk_doc, load_corpus  # noqa: E402

DATA = ROOT / "domains" / "rag" / "data"
CORPUS = DATA / "corpus"

# (id, split, type, tenant, question, reference, doc_id, section, answerable)
ITEMS = [
    ("q_0001", "evolve", "single_fact", "acme",
     "환불은 결제일로부터 며칠 이내에 요청할 수 있나요?",
     "환불은 결제일로부터 14일 이내에 요청할 수 있다.",
     "acme/refund-policy", "환불 요청 기한", True),
    ("q_0002", "evolve", "single_fact", "acme",
     "개봉한 상품은 며칠 이내에 환불할 수 있나요?",
     "개봉한 상품은 7일 이내에만 환불할 수 있다.",
     "acme/refund-policy", "환불 요청 기한", True),
    ("q_0003", "evolve", "multi_condition", "acme",
     "환불 검토와 환급은 각각 며칠이 걸리나요?",
     "접수 후 영업일 기준 3일 이내에 검토하고, 승인되면 5영업일 이내에 환급한다.",
     "acme/refund-policy", "환불 처리 절차", True),
    ("q_0004", "evolve", "single_fact", "acme",
     "카드 환불은 추가로 며칠이 걸릴 수 있나요?",
     "카드 결제는 카드사 정책에 따라 추가로 2~3일이 걸릴 수 있다.",
     "acme/refund-policy", "환불 처리 절차", True),
    ("q_0005", "evolve", "single_fact", "acme",
     "디지털 콘텐츠는 어떤 조건에서 환불되나요?",
     "다운로드 후 7일 이내이면서 사용 이력이 없을 때만 환불한다.",
     "acme/refund-policy", "환불 예외", True),
    ("q_0006", "evolve", "single_fact", "acme",
     "이미 사용한 이용권은 어떻게 환불되나요?",
     "잔여 기간을 일할 계산해 부분 환불한다.",
     "acme/refund-policy", "환불 예외", True),
    ("q_0007", "evolve", "single_fact", "acme",
     "같은 결제 건을 두 번 환불받을 수 있나요?",
     "동일 결제 건의 중복 환불은 팀장 이상 승인 권한자의 확인이 필요하다.",
     "acme/refund-policy", "반복 환불", True),
    ("q_0008", "evolve", "single_fact", "acme",
     "환불이 불가능한 항목은 무엇인가요?",
     "주문 제작 상품, 개봉한 소프트웨어 라이선스, 이미 사용한 상품권은 환불하지 않는다.",
     "acme/refund-policy", "환불 불가 항목", True),
    ("q_0009", "evolve", "single_fact", "acme",
     "Free 플랜은 분당 몇 회까지 호출할 수 있나요?",
     "Free 플랜은 분당 60회까지 호출할 수 있다.",
     "acme/api-rate-limit", "기본 한도", True),
    ("q_0010", "evolve", "single_fact", "acme",
     "Enterprise 플랜의 분당 호출 한도는 몇 회인가요?",
     "Enterprise는 분당 6,000회까지 호출할 수 있다.",
     "acme/api-rate-limit", "기본 한도", True),
    ("q_0011", "evolve", "single_fact", "acme",
     "호출 한도를 넘으면 어떤 응답이 오나요?",
     "HTTP 429와 Retry-After 헤더를 반환한다.",
     "acme/api-rate-limit", "초과 응답", True),
    ("q_0012", "evolve", "single_fact", "acme",
     "한도 초과 후 재시도는 어떻게 해야 하나요?",
     "재시도는 지수 백오프로 하고 최초 대기 시간은 1초로 둔다.",
     "acme/api-rate-limit", "초과 응답", True),
    ("q_0013", "evolve", "single_fact", "acme",
     "버스트는 얼마나 허용되나요?",
     "버킷 크기의 2배까지 10초간 버스트를 허용한다.",
     "acme/api-rate-limit", "버스트 허용", True),
    ("q_0014", "evolve", "single_fact", "acme",
     "API 키는 얼마마다 회전하나요?",
     "API 키는 90일마다 회전한다.",
     "acme/api-rate-limit", "API 키 회전", True),
    ("q_0015", "evolve", "single_fact", "acme",
     "키 회전 후 이전 키는 언제까지 유효한가요?",
     "회전 후 24시간 동안 이전 키가 유효하다.",
     "acme/api-rate-limit", "API 키 회전", True),
    ("q_0016", "evolve", "single_fact", "acme",
     "Free 플랜의 1차 응답 목표 시간은 얼마인가요?",
     "Free의 1차 응답 목표는 48시간이다.",
     "acme/support-plans", "플랜별 지원 범위", True),
    ("q_0017", "evolve", "single_fact", "acme",
     "Enterprise의 1차 응답 목표는 몇 시간인가요?",
     "Enterprise의 1차 응답 목표는 4시간이다.",
     "acme/support-plans", "플랜별 지원 범위", True),
    ("q_0018", "evolve", "single_fact", "acme",
     "전담 기술 매니저는 어느 플랜에 배정되나요?",
     "Enterprise만 전담 기술 매니저(TAM)를 배정받는다.",
     "acme/support-plans", "전담 매니저", True),
    ("q_0019", "evolve", "single_fact", "acme",
     "지원 범위에 포함되지 않는 작업은 무엇인가요?",
     "커스텀 코드 디버깅, 인프라 구축 대행, 데이터 마이그레이션은 지원 범위에 포함되지 않는다.",
     "acme/support-plans", "지원 제외", True),
    ("q_0020", "evolve", "multi_condition", "acme",
     "야간에 SEV1 장애가 나면 어떻게 대응하나요?",
     "Enterprise의 SEV1 장애는 24시간 대응하며 야간 호출은 온콜 로테이션으로 연결된다.",
     "acme/support-plans", "지원 시간", True),
    ("q_0021", "evolve", "single_fact", "acme",
     "SEV1과 SEV2는 어떻게 구분하나요?",
     "SEV1은 전체 서비스 불가, SEV2는 주요 기능 저하, SEV3는 일부 사용자 영향이다.",
     "acme/incident-runbook", "장애 등급", True),
    ("q_0022", "evolve", "single_fact", "acme",
     "SEV1이 발생하면 상태 페이지는 언제 갱신하나요?",
     "30분 이내에 상태 페이지를 갱신한다.",
     "acme/incident-runbook", "1차 대응", True),
    ("q_0023", "evolve", "single_fact", "acme",
     "배포 후 언제 롤백하나요?",
     "배포 후 10분 내 오류율이 2%를 넘으면 즉시 롤백한다.",
     "acme/incident-runbook", "롤백 기준", True),
    ("q_0024", "evolve", "single_fact", "acme",
     "SEV1 사후 분석 문서는 언제까지 작성하나요?",
     "모든 SEV1과 SEV2는 5영업일 이내에 사후 분석 문서를 작성한다.",
     "acme/incident-runbook", "사후 분석", True),
    ("q_0025", "evolve", "unanswerable", "acme",
     "2031년 우주 정거장 예산은 얼마인가요?",
     "", None, None, False),
    ("q_0026", "evolve", "unanswerable", "acme",
     "우리 회사의 2027년 인건비 총액은 얼마인가요?",
     "", None, None, False),

    # held-out 문항은 evolve가 정답으로 쓰지 않는 절(청크)에서만 냅니다(9.2절 감사 2번).
    ("q_0027", "heldout", "multi_condition", "acme",
     "지원은 어떤 언어로 제공되고, 일본어 문의는 응답이 얼마나 늦어지나요?",
     "지원은 한국어와 영어로 제공하고, 일본어 문의는 1차 응답 목표에 영업일 1일을 더한다.",
     "acme/support-plans", "지원 언어", True),
    ("q_0028", "heldout", "single_fact", "acme",
     "사용량 알림은 분당 한도의 몇 %에서 오나요?",
     "분당 한도의 80%에 도달하면 조직 관리자에게 이메일로 알린다.",
     "acme/api-rate-limit", "사용량 알림", True),
    ("q_0029", "heldout", "single_fact", "acme",
     "한도 상향 신청의 검토 결과는 언제 알려 주나요?",
     "검토 결과는 신청한 다음 영업일에 이메일로 알린다.",
     "acme/api-rate-limit", "한도 상향 요청", True),
    ("q_0030", "heldout", "single_fact", "acme",
     "문의는 어디에서 접수하나요?",
     "문의는 지원 포털에서 티켓으로 접수한다.",
     "acme/support-plans", "문의 접수 방법", True),
    ("q_0031", "heldout", "single_fact", "acme",
     "장애 대응 훈련은 얼마나 자주 하나요?",
     "장애 대응 훈련은 분기마다 1회 실시한다.",
     "acme/incident-runbook", "대응 훈련", True),
    ("q_0032", "heldout", "single_fact", "acme",
     "훈련 결과는 언제 런북에 반영하나요?",
     "훈련 결과는 훈련을 마친 주에 런북에 반영한다.",
     "acme/incident-runbook", "대응 훈련", True),
    ("q_0033", "heldout", "unanswerable", "acme",
     "사옥 주차장은 몇 면인가요?",
     "", None, None, False),
    ("q_0034", "heldout", "multi_condition", "acme",
     "한도 상향은 어디서 신청하고, 신청할 수 없는 플랜은 무엇인가요?",
     "콘솔의 한도 상향 양식으로 신청하며, Free 플랜은 상향 대상이 아니다.",
     "acme/api-rate-limit", "한도 상향 요청", True),

    ("q_0035", "ood_docs", "single_fact", "acme",
     "Pro 플랜의 월 요금은 얼마인가요?",
     "Pro 플랜은 월 49만 원이다.",
     "acme/pricing", "플랜 요금", True),
    ("q_0036", "ood_docs", "single_fact", "acme",
     "연 결제하면 얼마나 할인되나요?",
     "연 결제 시 10% 할인된다.",
     "acme/pricing", "플랜 요금", True),
    ("q_0037", "ood_docs", "single_fact", "acme",
     "Enterprise의 초과 호출 요금은 1,000회당 얼마인가요?",
     "Enterprise는 포함량을 넘으면 1,000회당 2,000원을 청구한다.",
     "acme/pricing", "초과 요금", True),
    ("q_0038", "ood_docs", "single_fact", "acme",
     "스타트업 할인과 비영리 할인을 함께 받을 수 있나요?",
     "두 할인은 중복 적용되지 않는다.",
     "acme/pricing", "할인 정책", True),

    ("q_0039", "ood_intent", "multi_condition", "acme",
     "환불을 빨리 받으려면 무엇을 확인해야 하나요?",
     "검토 3일, 환급 5영업일, 카드사에 따라 2~3일이 더 걸린다.",
     "acme/refund-policy", "환불 처리 절차", True),
    ("q_0040", "ood_intent", "multi_condition", "acme",
     "요금제를 올리면 지원 응답이 얼마나 빨라지나요?",
     "Free 48시간, Pro 24시간, Enterprise 4시간이 목표다.",
     "acme/support-plans", "플랜별 지원 범위", True),
    ("q_0041", "ood_intent", "single_fact", "acme",
     "장애가 났을 때 고객에게 언제 알리나요?",
     "고객 공지는 1시간 이내에 첫 문안을 내보낸다.",
     "acme/incident-runbook", "1차 대응", True),
    ("q_0042", "ood_intent", "multi_condition", "acme",
     "API를 너무 많이 호출하면 어떤 제한이 걸리나요?",
     "분당 한도를 넘으면 429와 Retry-After를 받고, 버스트는 2배까지 10초 허용된다.",
     "acme/api-rate-limit", "기본 한도|초과 응답|버스트 허용", True),
]


def main() -> int:
    chunks = []
    for doc in load_corpus(CORPUS):
        chunks.extend(chunk_doc(doc))
    by_section = {}
    for c in chunks:
        by_section[(c.doc_id, c.section)] = c

    rows, split = [], {"evolve": [], "heldout": [], "ood_docs": [], "ood_intent": [], "smoke": []}
    missing = []
    for (qid, spl, typ, tenant, question, reference, doc_id, section, answerable) in ITEMS:
        item = {"id": qid, "version": 1, "split": spl, "type": typ, "tenant": tenant,
                "question": question, "answerable": answerable, "reference": reference,
                "chunks": []}
        if answerable:
            sections = [s.strip() for s in str(section).split("|") if s.strip()]
            picked = [by_section.get((doc_id, s)) for s in sections]
            if not sections or any(c is None for c in picked):
                missing.append(f"{qid}: {doc_id} / {section}")
                continue
            item["chunks"] = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text}
                              for c in picked]
        rows.append(item)
        split[spl].append(qid)
    split["smoke"] = split["evolve"][:2]
    split["corpus"] = {"evolve": "corpus_A", "ood_docs": "corpus_A + pricing"}

    if missing:
        print("절을 찾지 못한 문항:")
        for m in missing:
            print("  -", m)
        return 1

    def dump_jsonl(path: Path, items: list[dict]) -> None:
        path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
                        encoding="utf-8", newline="\n")   # 윈도에서도 LF(커밋본과 같게)

    dump_jsonl(DATA / "goldenset_v1.jsonl", rows)
    (DATA / "split_rag.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tasks = [{"id": r["id"], "question": r["question"], "tenant": r["tenant"]} for r in rows]
    (DATA / "evolve_tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    eval_dir = ROOT / "eval" / "goldenset"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "split_rag.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"goldenset_v1.jsonl: {len(rows)}문항 "
          f"(evolve {len(split['evolve'])}, heldout {len(split['heldout'])}, "
          f"ood_docs {len(split['ood_docs'])}, ood_intent {len(split['ood_intent'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
