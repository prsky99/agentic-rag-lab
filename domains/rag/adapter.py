"""RAG 도메인 어댑터 (7.8절, 8.4~8.7절).

도메인은 네 가지를 코어에 제공합니다.

* 문항 목록 (evolve / heldout / smoke)
* 실행: 후보 하네스를 subprocess로 실행하고 산출물을 남깁니다
* 채점: 산출물을 보상으로 바꿉니다 (심판은 여기에만 있습니다)
* 가드: 승격하면 안 되는 조건

``run``은 하네스에 질문 텍스트와 테넌트만 넘기고 정답은
숨깁니다. 하네스가 정답을 볼 수 있으면 진화는 평가가 아니라 커닝이 됩니다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from harness.llm import DecisionClient  # noqa: E402
from harness.retrieve import chunk_doc, load_corpus  # noqa: E402
from judge import (EVALUATOR_VERSION, JUDGE_ERROR, build_judge,  # noqa: E402
                   citation_precision, endpoint_id, judge_answer, judge_id, score_trial,
                   trial_tokens)
from outcomes import FAILURES, REFUSALS  # noqa: E402  (기권 사유 분류표, R01)
from judge import goldenset_version as _goldenset_version  # noqa: E402
from rrsi.domain import Domain, EvalSummary, TaskResult  # noqa: E402

CFG = json.loads((HERE / "rrsi.json").read_text(encoding="utf-8"))
GOLDEN = DATA / "goldenset_v1.jsonl"


def goldenset_version() -> str:
    """골든셋 버전 표식: 파일 이름 + 내용 해시 앞 12자리 (eval.json에 남깁니다)."""
    return _goldenset_version(GOLDEN)


def dir_sha256(path: Path) -> str:
    """폴더 안 파일들의 이름과 내용 해시(코퍼스 버전 표식)."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]
SPLIT = json.loads((DATA / "split_rag.json").read_text(encoding="utf-8"))


def load_items() -> dict[str, dict]:
    out = {}
    for line in (DATA / "goldenset_v1.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            out[item["id"]] = item
    return out


def load_chunk_texts() -> dict[str, str]:
    texts = {}
    for doc in load_corpus(DATA / "corpus"):
        for p in chunk_doc(doc):
            texts[p.chunk_id] = p.text
    return texts


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]


class RagDomain(Domain):
    name = "rag"
    harness_path = "harness"
    critic_patterns = [
        (r"\bq_\d{3,5}\b", "golden question id in diff"),
        (r"\bdoc_[0-9a-f]{6,}\b", "document id in diff"),
        (r"\bchunk_[0-9a-f]{6,}\b", "chunk id in diff"),
        (r"\b[a-z0-9_-]+/[a-z0-9_-]+#\d+\b", "chunk id (tenant/doc#n) in diff"),
        (r"(golden|gold_answer|reference_answer|rubric|answers\.jsonl)", "eval artifact"),
        (r"(/runs/|eval\.json|judge_cache)", "evaluation output path"),
        (r"if\s+['\"].{15,}['\"]\s+(in|==)\s+", "long literal question match"),
        (r"tenant_[a-z0-9]{4,}", "tenant id hardcoded"),
    ]
    component_signals = [
        ("prompt",         [r"prompts/.*\.md", r"SYSTEM_PROMPT", r"INSTRUCTION"]),
        ("gate_policy",    [r"gates?\.py", r"threshold", r"_floor", r"rewrite"]),
        ("retrieval_tool", [r"retrieve\.py", r"config\.yaml", r"top_k", r"bm25_weight",
                            r"rerank", r"rrf"]),
        ("context_mgmt",   [r"context\.py", r"dedup", r"max_chars", r"order"]),
        ("control_flow",   [r"pipeline\.py", r"max_rewrites", r"early_exit", r"async def"]),
        ("memory",         [r"session", r"history", r"glossary"]),
        ("subagent",       [r"compare_agent", r"calc_agent"]),
    ]

    def __init__(self) -> None:
        from briefs import BRIEFS, EDITS          # 도메인 로컬 브리프
        self.briefs = {**BRIEFS, "EDITS": EDITS}
        self.items = load_items()
        self.goldenset_version = goldenset_version()
        self.chunk_texts = load_chunk_texts()
        self._judge_cache: dict[tuple, tuple[float | None, str]] = {}
        # 심판은 rrsi.json의 judge_* 설정으로 만듭니다. 빈 값은 DECISION_* 환경 변수를 따릅니다(F08).
        self._judge: DecisionClient = build_judge(CFG)
        self._fp_cache: dict[tuple, dict] = {}

    # --- 문항 -----------------------------------------------------------
    def evolve_ids(self) -> list[str]:
        return list(SPLIT["evolve"])

    def heldout_ids(self) -> list[str]:
        return list(SPLIT[os.environ.get("RAG_HELDOUT_SPLIT", "heldout")])

    def smoke_ids(self, incumbent_per_task=None) -> list[str]:
        return list(CFG.get("smoke_tasks") or SPLIT["evolve"][:2])

    def task_meta(self, task_id: str) -> dict:
        item = self.items.get(task_id, {})
        return {"id": task_id, "question": item.get("question"),
                "type": item.get("type"), "tenant": item.get("tenant")}

    # --- 실행 -----------------------------------------------------------
    def run(self, root, runs_dir, job: str, ids: list[str], k: int,
            log_prefix: str = "") -> None:
        root, runs_dir = Path(root), Path(runs_dir)
        hdir = root / "domains" / "rag"
        idfile = runs_dir / "jobs" / job / "_ids.json"
        idfile.parent.mkdir(parents=True, exist_ok=True)
        tasks = [{"id": tid,
                  "question": self.items[tid]["question"],
                  "tenant": self.items[tid]["tenant"],
                  "tenant_tier": "pro"} for tid in ids if tid in self.items]
        idfile.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
        fp = self.fingerprint(root, ids, k)
        fpfile = runs_dir / "jobs" / job / "_fingerprint.json"
        if not fpfile.exists() or json.loads(fpfile.read_text(encoding="utf-8")) != fp:
            # 같은 이름(r0A 등)이라도 하네스 코드·데이터·설정·모델이 다르면 이전 결과를 버립니다(E03).
            for old in (runs_dir / "jobs" / job).glob("s[0-9]*/*.json"):
                old.unlink()
        fpfile.write_text(json.dumps(fp, ensure_ascii=False, indent=1), encoding="utf-8")

        for s in range(k):
            label = f"{job}/s{s}"
            out = runs_dir / "jobs" / label
            out.mkdir(parents=True, exist_ok=True)
            if all((out / f"{t}.json").exists() for t in ids):
                continue                                  # 재개: 지문이 같을 때만 완료된 트라이얼을 씁니다
            env = self._harness_env(hdir, seed=s)
            proc = subprocess.run(
                [sys.executable, "-m", "harness.run", "--ids", str(idfile),
                 "--out", str(out), "--seed", str(s),
                 "--concurrency", str(CFG.get("concurrency", 8)),
                 "--config", "harness/config.yaml"],
                cwd=str(hdir), env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace")
            logdir = runs_dir / "logs"
            logdir.mkdir(parents=True, exist_ok=True)
            (logdir / f"{label.replace('/', '_')}.log").write_text(
                proc.stdout + proc.stderr, encoding="utf-8")
            if proc.returncode != 0:
                # 하네스 전체가 중단되면 그 트라이얼은 0점. 다음 s로 넘어갑니다.
                continue

    EVALUATOR_VERSION = EVALUATOR_VERSION

    @staticmethod
    def _harness_env(hdir: Path, seed: int | None = None) -> dict:
        """하네스 subprocess의 환경. 실행과 지문 계산이 같은 환경을 쓰도록 한 곳에서 만듭니다."""
        env = {**os.environ,
               "RAG_INDEX_SNAPSHOT": str(CFG.get("index_snapshot", "")),
               "OTEL_SDK_DISABLED": "true",
               "PYTHONIOENCODING": "utf-8",
               "PYTHONPATH": str(hdir)}
        if seed is not None:
            env["RAG_TRIAL_SEED"] = str(seed)
        return env

    def runtime(self, root) -> dict:
        """하네스가 실제로 쓰게 될 실행 조건을 하네스 자신에게 묻습니다(E03).

        평가 대상 하네스(``root``)의 ``RAGPipeline.from_config``를 실행과 같은 작업 폴더·환경·
        설정 파일로 만들고 ``runtime()``을 읽습니다. 환경 변수 이름을 여기서 따로 나열하지
        않으므로, 설정 파일과 환경 변수의 우선순위가 하네스 코드와 어긋날 일이 없습니다.
        읽지 못하면(하네스가 깨졌으면) 그 사실을 지문에 남깁니다.
        """
        hdir = Path(root) / "domains" / "rag"
        code = ("import json,sys;"
                "from harness.pipeline import RAGPipeline;"
                "p=RAGPipeline.from_config('harness/config.yaml',"
                "index_snapshot=sys.argv[1] or None);"
                "print('RUNTIME='+json.dumps(p.runtime(),ensure_ascii=False,sort_keys=True))")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code, str(CFG.get("index_snapshot", ""))],
                cwd=str(hdir), env=self._harness_env(hdir), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=120)
            for line in proc.stdout.splitlines():
                if line.startswith("RUNTIME="):
                    return json.loads(line[len("RUNTIME="):])
            return {"unresolved": (proc.stderr or proc.stdout)[-300:]}
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            return {"unresolved": repr(exc)[:300]}

    def fingerprint(self, root, ids: list[str], k: int) -> dict:
        """결과를 다시 써도 되는지 가리는 지문(E03).

        하네스 파일·골든셋·코퍼스·평가 설정·평가기 버전에 더해, 하네스가 **최종 적용한** 실행
        조건(``runtime``: 결정 모드·모델·주소, 생성 모드·제공자·모델·주소, 검색 백엔드·주소·
        색인 해시, 색인 스냅샷, config.yaml의 최종 값)을 넣습니다. 같은 job 이름이라도
        ``DECISION_MODE``를 local에서 offline으로 바꾸면 지문이 달라져 이전 결과를 버립니다.
        """
        import hashlib
        h = hashlib.sha256()
        harness = Path(root) / "domains" / "rag" / "harness"
        for f in sorted(p for p in harness.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            h.update(f.relative_to(harness).as_posix().encode())
            h.update(f.read_bytes())
        fp = {"harness_sha256": h.hexdigest(), "goldenset": self.goldenset_version,
              "config_sha256": hashlib.sha256(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:16],
              "corpus_sha256": dir_sha256(DATA / "corpus"),
              "runtime": self.runtime(root),
              "evaluator": self.EVALUATOR_VERSION, "ids": sorted(ids), "k": k}
        self._fp_cache[self._fp_key(root, ids, k, fp["harness_sha256"])] = fp
        return fp

    @staticmethod
    def _fp_key(root, ids: list[str], k: int, harness_sha: str) -> tuple:
        """지문 캐시 열쇠: 하네스 해시·환경 변수까지 같을 때만 같은 지문을 다시 씁니다."""
        env = json.dumps(sorted(os.environ.items()))
        return (str(Path(root).resolve()), tuple(sorted(ids)), int(k), harness_sha, env)

    # 비교 조건에 넣는 rrsi.json 키: 실행·채점에 영향을 주고 후보가 바꿀 수 없는 값만 고릅니다.
    FIXED_CFG_KEYS = ("index_snapshot", "judge_mode", "judge_model", "judge_base_url",
                      "judge_revision")

    def conditions(self, root, ids: list[str], k: int) -> dict:
        """비교 조건(F01): 지문에서 후보가 바꿀 수 있는 하네스 값을 뺀 나머지.

        하네스 파일 해시, ``config.yaml`` 값(``runtime.config``), 정책 버전, 하네스가 만든 색인
        해시는 진화가 바꾸는 대상이므로 넣지 않습니다. 결정·생성 모델의 모드·모델·주소, 검색
        백엔드·주소, 색인 스냅샷, 골든셋·코퍼스·평가기·심판·문항·k는 고정 조건입니다. 주소는
        ``endpoint_id``로 바꿔 비밀값이 남지 않게 합니다.
        """
        import hashlib
        h = hashlib.sha256()
        harness = Path(root) / "domains" / "rag" / "harness"
        for f in sorted(p for p in harness.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            h.update(f.relative_to(harness).as_posix().encode())
            h.update(f.read_bytes())
        fp = (self._fp_cache.get(self._fp_key(root, ids, k, h.hexdigest()))
              or self.fingerprint(root, ids, k))
        rt = fp.get("runtime") or {}
        if "unresolved" in rt:
            runtime = {"unresolved": True}
        else:
            def fixed(part: dict, keys: tuple[str, ...]) -> dict:
                out = {kk: part.get(kk) for kk in keys}
                if "address" in out:
                    out["address"] = endpoint_id(str(out["address"] or ""))
                return out
            runtime = {"decision": fixed(rt.get("decision") or {}, ("mode", "model", "address",
                                                                   "provider")),
                       "generation": fixed(rt.get("generation") or {},
                                           ("mode", "provider", "model", "address", "has_api_key")),
                       "retrieval": fixed(rt.get("retrieval") or {}, ("backend", "address")),
                       # 스냅샷 파일은 git이 추적하지 않아 후보 작업장에 없을 수 있으므로 이름만 비교합니다.
                       "index_snapshot": (rt.get("index_snapshot") or {}).get("name")}
        return {"ids": sorted(ids), "k": int(k), "evaluator": self.EVALUATOR_VERSION,
                "judge": judge_id(self._judge), "goldenset": self.goldenset_version,
                "corpus_sha256": fp.get("corpus_sha256"),
                "rrsi_cfg": {kk: CFG.get(kk) for kk in self.FIXED_CFG_KEYS},
                "runtime": runtime}

    # 비교에 반드시 필요한 extra 값(성과 5 + 버전 2 + 심판 2). 하나라도 없으면 비교하지 않습니다(F01).
    REQUIRED_EXTRA = ("cite_precision", "abstain_answerable", "abstain_unanswerable", "p95_ms",
                      "error_rate", "goldenset_version", "evaluator", "judge", "judge_errors")

    def validity(self, summary: EvalSummary) -> list[str]:
        """기준선·비교·δ 보정에 쓸 수 있는 평가인지(F01·F02). 사유 목록이 비어 있어야 씁니다."""
        import math

        def finite(val) -> bool:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                return False
            return abs(val) <= 2 ** 53 if isinstance(val, int) else math.isfinite(val)

        v: list[str] = []
        ex = dict(summary.extra or {})
        missing = [kk for kk in self.REQUIRED_EXTRA if kk not in ex]
        if missing:
            v.append("extra missing: " + ",".join(missing))
        # 비율은 0~1, 지연은 0 이상의 유한한 수여야 합니다(G04). None은 분모 0(미측정)입니다(G02).
        # 인용·기권 비율의 None은 "해당 트라이얼 없음"이라 허용하고 가드가 따로 봅니다. 오류율·지연이
        # None이면 채점한 산출물이 하나도 없다는 뜻이므로 쓰지 않습니다.
        for kk in ("cite_precision", "abstain_answerable", "abstain_unanswerable", "error_rate"):
            if kk in ex and ex[kk] is None:
                if kk == "error_rate":
                    v.append("extra error_rate unmeasured (채점한 산출물 0건)")
            elif kk in ex and (not finite(ex[kk]) or not 0.0 <= ex[kk] <= 1.0):
                v.append(f"extra {kk}={ex[kk]!r} is not a finite number in [0, 1]")
        if "p95_ms" in ex and (not finite(ex["p95_ms"]) or ex["p95_ms"] < 0):
            v.append(f"extra p95_ms={ex['p95_ms']!r} is not a finite number >= 0")
        if not finite(summary.S) or not finite(summary.C):
            v.append(f"S={summary.S!r} C={summary.C!r} is not finite")
        else:
            if not 0.0 <= summary.S <= 1.0:
                v.append(f"S={summary.S!r} is not in [0, 1]")
            if summary.C < 0:
                v.append(f"C={summary.C!r} is negative")
        if isinstance(summary.k, bool) or not isinstance(summary.k, int) or summary.k < 1:
            v.append(f"k={summary.k!r} is not a positive integer")
        # 버전·식별자: 비어 있지 않은 문자열이고, 비교 조건에 적힌 값과 같아야 합니다(G04).
        cond = summary.conditions or {}
        for kk, ck in (("evaluator", "evaluator"), ("judge", "judge"),
                       ("goldenset_version", "goldenset")):
            if kk in ex and (not isinstance(ex[kk], str) or not ex[kk]):
                v.append(f"extra {kk}={ex[kk]!r} is not an identifier")
            elif kk in ex and ck in cond and cond[ck] != ex[kk]:
                v.append(f"extra {kk} {ex[kk]} != conditions {cond[ck]}")
        je = ex.get("judge_errors", 0)
        if isinstance(je, bool) or not isinstance(je, int) or je != 0:
            v.append(f"judge errors {je}")              # 심판 장애가 있으면 점수를 믿을 수 없습니다(R03)
        if (summary.conditions or {}).get("runtime", {}).get("unresolved"):
            v.append("runtime unresolved")
        return v

    # --- 채점 -----------------------------------------------------------
    def score(self, runs_dir, job: str, ids: list[str], k: int):
        runs_dir = Path(runs_dir)
        per: dict[str, TaskResult] = {}
        cite_p = cite_t = 0
        abstain_answerable = []
        abstain_unanswerable = []
        latencies: list[float] = []
        errors = runs = unknown_calls = unknown_runs = judge_errors = 0
        for tid in ids:
            item = self.items[tid]
            rewards, toks, missing = [], [], 0
            for s in range(k):
                path = runs_dir / "jobs" / job / f"s{s}" / f"{tid}.json"
                if not path.exists():
                    rewards.append(0.0)
                    toks.append(None)
                    missing += 1
                    continue
                tr = json.loads(path.read_text(encoding="utf-8"))
                runs += 1
                sc = self._score(item, tr)           # 정답 판정은 트라이얼당 한 번(R03)
                # 실패 결과(운영 장애·예산 초과·알 수 없는 사유·실행 오류)는 error_rate에 셉니다(R01).
                errors += 1 if sc["outcome"] in FAILURES else 0
                judge_errors += 1 if sc["judge"] == JUDGE_ERROR else 0
                r, cp_ok, cp_total = sc["reward"], sc["ok"], sc["total"]
                rewards.append(r)
                # 비용 C: 미측정 호출은 0이 아니라 추정치로 채웁니다(RT06).
                tok, unknown = trial_tokens(tr.get("usage"))
                toks.append(tok)
                unknown_calls += unknown
                unknown_runs += 1 if unknown else 0
                cite_p += cp_ok
                cite_t += cp_total
                if isinstance(tr.get("latency_ms"), (int, float)):
                    latencies.append(float(tr["latency_ms"]))
                # 기권으로 인정하는 것은 정상 기권·보안 차단뿐입니다(E05·R01).
                abstained = 1.0 if sc["outcome"] in REFUSALS else 0.0
                if item["answerable"]:
                    abstain_answerable.append(abstained)
                else:
                    abstain_unanswerable.append(abstained)
            per[tid] = TaskResult(rewards=rewards, tokens=toks, missing=missing)

        # 분모가 0인 비율은 None(미측정)으로 둡니다(G02). 예전처럼 기권율 1.0·오류율 0으로 채우면
        # 산출물이 전부 누락된 평가가 "정상 기권 100%·오류 0"처럼 보입니다. validity가 None을 거부합니다.
        extra = {
            "cite_precision": round(cite_p / cite_t, 4) if cite_t else None,
            "abstain_answerable": round(sum(abstain_answerable) / len(abstain_answerable), 4)
            if abstain_answerable else None,
            "abstain_unanswerable": round(sum(abstain_unanswerable) / len(abstain_unanswerable), 4)
            if abstain_unanswerable else None,
            "p95_ms": round(p95(latencies), 1) if runs else None,
            "goldenset_version": self.goldenset_version,
            "error_rate": round(errors / runs, 4) if runs else None,
            "evaluator": self.EVALUATOR_VERSION,
            "judge": judge_id(self._judge),
            # 심판 장애 트라이얼 수. 있으면 guards()가 후보를 채택하지 않습니다(R03).
            "judge_errors": judge_errors,
        }
        if unknown_calls:
            # 미측정 호출이 있었습니다. C는 그 호출을 글자 수 추정치로 채운 값입니다.
            extra["usage_unknown_calls"] = unknown_calls
            extra["usage_unknown_runs"] = unknown_runs
            extra["cost_basis"] = "measured+estimated_unknown"
        return per, extra

    def _score(self, item: dict, tr: dict) -> dict:
        # 회귀 평가(eval/run_subset.py)와 같은 함수입니다. 실패 결과는 기권이어도 0점(E05·R01).
        return score_trial(item, tr, self.chunk_texts,
                           lambda answer: self._judge_verdict(item, answer))

    def _reward(self, item: dict, tr: dict) -> tuple[float, int, int]:
        sc = self._score(item, tr)
        return sc["reward"], sc["ok"], sc["total"]

    def _supported_citations(self, item: dict, tr: dict) -> tuple[int, int]:
        return citation_precision(tr.get("citations") or [],
                                  tr.get("retrieved") or [], self.chunk_texts)

    def _judge_verdict(self, item: dict, answer: str) -> tuple[float | None, str]:
        # 답변 전체의 해시를 열쇠로 씁니다. 앞 120자만 쓰면 뒤가 다른 답변이 같은 점수를 받습니다(E07).
        # 심판 장애는 저장하지 않습니다. 다음 채점에서 다시 판정하고, 장애가 남으면 judge_errors로 셉니다.
        import hashlib
        key = (item["id"], hashlib.sha256(answer.encode("utf-8")).hexdigest(), self.goldenset_version,
               judge_id(self._judge))
        if key not in self._judge_cache:
            verdict = judge_answer(self._judge, item, answer)
            if verdict[1] == JUDGE_ERROR:
                return verdict
            self._judge_cache[key] = verdict
        return self._judge_cache[key]

    def _judge_correct(self, item: dict, answer: str) -> float:
        return self._judge_verdict(item, answer)[0] or 0.0

    # --- 스모크 ---------------------------------------------------------
    def smoke(self, root, runs_dir, job: str, ids: list[str]) -> tuple[bool, dict]:
        hdir = Path(root) / "domains" / "rag"
        proc = subprocess.run(
            [sys.executable, "-c",
             "from harness.pipeline import RAGPipeline;"
             "p=RAGPipeline.from_config('harness/config.yaml');"
             "print('construct ok', len(getattr(p.retriever,'passages',[])))"],
            cwd=str(hdir), capture_output=True, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "OTEL_SDK_DISABLED": "true", "PYTHONIOENCODING": "utf-8",
                 "PYTHONPATH": str(hdir)})
        if proc.returncode != 0:
            return False, {"stage": "construct",
                           "detail": (proc.stdout + proc.stderr)[-800:]}
        self.run(root, runs_dir, f"{job}/smoke", ids, k=1)
        produced = sum(1 for t in ids
                       if (Path(runs_dir) / "jobs" / f"{job}/smoke" / "s0" / f"{t}.json").exists())
        return produced == len(ids), {"stage": "run",
                                      "detail": f"{produced}/{len(ids)} tasks"}

    # --- 가드 -----------------------------------------------------------
    def guards(self, incumbent: EvalSummary, candidate: EvalSummary) -> list[str]:
        # 먼저 두 평가가 쓸 수 있는 결과인지 봅니다. 기준선의 심판 장애·필수 값 누락도 비교를 멈춥니다(F01).
        v = [f"incumbent {x}" for x in self.validity(incumbent)]
        v += [f"candidate {x}" if not x.startswith("judge errors") else x
              for x in self.validity(candidate)]
        if v:
            return v
        inc, cand = dict(incumbent.extra or {}), dict(candidate.extra or {})
        for kk in ("evaluator", "judge", "goldenset_version"):
            if inc[kk] != cand[kk]:
                v.append(f"incomparable: {kk} {cand[kk]} != {inc[kk]}")
        if v:
            return v

        def g(key: str) -> tuple[float | None, float | None]:
            # validity를 통과했으므로 값은 유한한 수이거나 None(분모 0, 미측정)입니다.
            a, b = inc.get(key), cand.get(key)
            if a is not None and b is None:
                v.append(f"{key} unmeasured on candidate (분모 0)")
            return (None if a is None else float(a)), (None if b is None else float(b))

        i_cite, c_cite = g("cite_precision")
        if None not in (i_cite, c_cite) and c_cite < i_cite - 0.02:
            v.append(f"cite_precision {c_cite:.3f} < {i_cite:.3f}-0.02")

        i_ans, c_ans = g("abstain_answerable")
        if None not in (i_ans, c_ans) and c_ans > i_ans + 0.10:
            v.append("over-abstention on answerable items")

        # 실패(실행 오류·운영 장애·예산 초과·알 수 없는 사유)가 늘어난 후보는 채택하지 않습니다(E05·R01)
        i_err, c_err = g("error_rate")
        if c_err > i_err + 0.01:
            v.append(f"error_rate {c_err:.3f} > {i_err:.3f}+0.01")

        _, c_unans = g("abstain_unanswerable")
        if c_unans is not None and c_unans < 0.85:
            v.append(f"abstain on unanswerable {c_unans:.3f} < 0.85")

        i_p95, c_p95 = g("p95_ms")
        if i_p95 >= 50 and c_p95 > i_p95 * 1.25:
            v.append(f"p95 {c_p95:.0f}ms > 1.25x")
        return v


DOMAIN = RagDomain()
