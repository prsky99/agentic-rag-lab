"""한 라운드 = 분석 → 예산 → 제안 → 심사 → 평가 → 선택 → 승격 (7.3절).

라운드는 재개할 수 있습니다. 후보 작업장은 라운드마다 새로 만들고, 승격 기록은
``runs/<domain>/history.jsonl``에 한 줄씩 남습니다. 중간에 중단되어도 frontier.json과
이력만으로 이어서 실행할 수 있습니다.

재개하기 전과 승격하기 전에 비교 조건을 확인합니다(F01). frontier의 비교 조건(문항·k·평가기·
심판·골든셋·고정 실행 조건)이나 현행 하네스 해시가 지금과 다르면, 이전 점수와 S*는 다른 조건의
값이므로 버리고 기준선을 새로 잽니다. 기준선 평가에 심판 장애·필수 값 누락이 있으면 기준선으로
쓰지 않고 멈춥니다.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from . import critic
from ._log import append_jsonl, log, read_json, write_json
from .domain import Domain, EvalSummary
from .evaluate import evaluate, p95_latency
from .gitops import Workspace
from .history import History
from .propose import Proposal, build_proposer, commit_edits
from .schedule import edit_budget
from .selection import Decision, decide, pick_best


class BaselineInvalid(RuntimeError):
    """기준선으로 쓸 수 없는 평가(심판 장애·필수 값 누락·조건 미확인). 비교와 승격을 멈춥니다."""


def load_config(root: Path, domain: Domain) -> dict:
    cfg = read_json(Path(root) / "domains" / domain.name / "rrsi.json", {}) or {}
    cfg.setdefault("T", 15)
    cfg.setdefault("k", 2)
    cfg.setdefault("m", 2)
    cfg.setdefault("b_min", 1)
    cfg.setdefault("b_max", 3)
    cfg.setdefault("delta_prior", 0.011)
    cfg.setdefault("delta_z", 2.0)
    cfg.setdefault("proposer", "catalog")
    cfg.setdefault("invalid_missing_frac", 0.1)
    return cfg


def calibration_stale(calib: dict, conditions: dict | None, harness_tree: str | None) -> list[str]:
    """보정값을 지금 조건에 다시 써도 되는지(F02). 비어 있으면 써도 됩니다.

    δ를 잰 비교 조건(문항·k·평가기·심판·고정 실행 조건)과 기준선 하네스가 지금과 같아야 합니다.
    지문이 없는 이전 판의 calibration.json은 쓰지 않습니다.
    """
    fp = (calib or {}).get("fingerprint") or {}
    if not fp:
        return ["fingerprint missing"]
    out = []
    if conditions is not None and fp.get("conditions") != conditions:
        out.append("conditions differ")
    if harness_tree is not None and fp.get("harness_tree") != harness_tree:
        out.append("harness differs")
    return out


def effective_delta(runs_dir: Path, cfg: dict, conditions: dict | None = None,
                    harness_tree: str | None = None) -> tuple[float, str]:
    calib = read_json(Path(runs_dir) / "calibration.json", {}) or {}
    delta = calib.get("delta")
    if isinstance(delta, (int, float)) and not isinstance(delta, bool) and delta > 0:
        stale = calibration_stale(calib, conditions, harness_tree)
        if not stale:
            return float(delta), "calibrated"
        # 조건이 바뀐 보정값은 다시 쓰지 않습니다. 사전값으로 돌아가고 재측정을 요청합니다.
        return (float(cfg.get("delta_prior", 0.011)),
                f"prior; calibration stale ({', '.join(stale)}) → calibrate 다시 실행")
    return float(cfg.get("delta_prior", 0.011)), "prior"


def _summary_from_frontier(frontier: dict, cfg: dict) -> EvalSummary:
    inc = (frontier or {}).get("incumbent") or {}
    return EvalSummary(S=float(inc.get("S", 0.0)), C=float(inc.get("C", 0.0)),
                       k=int(inc.get("k", cfg.get("k", 2))), extra=dict(inc.get("extra") or {}),
                       conditions=dict(inc.get("conditions") or {}),
                       harness_tree=str(inc.get("harness_tree") or ""))


def _record(history: History, t: int, proposal: Proposal, summary: EvalSummary | None,
            accepted: bool, reason: str, commit: str = "",
            delta_S: float | None = None, delta_C: float | None = None,
            run_id: str = "") -> None:
    history.append({
        "t": t,
        "run": run_id,
        "label": proposal.label,
        "components": sorted(proposal.components),
        "hypothesis": "; ".join(e.hypothesis for e in proposal.edits),
        "edits": [e.key for e in proposal.edits],
        "S": round(summary.S, 6) if summary else None,
        "C": round(summary.C, 2) if summary else None,
        "delta_S": round(delta_S, 6) if delta_S is not None else None,
        "delta_C": round(delta_C, 6) if delta_C is not None else None,
        "accepted": accepted,
        "reason": reason,
        "commit": commit,
        "note": proposal.note,
    })


def baseline(domain: Domain, root: Path, runs_dir: Path, cfg: dict, job: str) -> dict:
    ids = domain.evolve_ids()
    summary = evaluate(domain, root, runs_dir, job, ids, int(cfg["k"]))
    # 기준선으로 쓸 수 있는 평가인지 먼저 봅니다. 심판 장애·필수 값 누락이면 frontier를 쓰지 않습니다(F01).
    problems = list(domain.validity(summary))
    if not summary.conditions:
        problems.append("conditions missing")
    if problems:
        log(domain.name, f"baseline {job} 거부: {'; '.join(problems)[:200]}")
        raise BaselineInvalid("; ".join(problems))
    ws = Workspace(root, runs_dir, f"domains/{domain.name}/harness")
    delta, src = effective_delta(runs_dir, cfg, summary.conditions, summary.harness_tree)
    old = read_json(runs_dir / "frontier.json", {}) or {}
    rounds_done = int(old.get("rounds_done", 0) or 0)   # 라운드 번호는 기준선을 새로 잡아도 이어 갑니다(F05)
    commit = ws.head_commit()
    frontier = {
        "domain": domain.name,
        "incumbent": {
            "t": rounds_done,
            "commit": commit,
            "harness_tree": summary.harness_tree,
            "job": job,
            "S": round(summary.S, 6),
            "C": round(summary.C, 2),
            "k": summary.k,
            "extra": summary.extra,
            "conditions": summary.conditions,
        },
        "S_star": round(summary.S, 6),
        "baseline_harness_tree": summary.harness_tree,
        "rounds_done": rounds_done,
        "trajectory": [{"t": rounds_done, "S": round(summary.S, 6), "C": round(summary.C, 2),
                        "commit": commit, "job": job}],
        "config": {kk: cfg.get(kk) for kk in
                   ("T", "k", "m", "delta", "delta_prior", "beta0", "beta1", "w_s", "w_c", "w_n")},
    }
    frontier["config"]["delta"] = delta
    frontier["config"]["delta_source"] = src
    write_json(runs_dir / "frontier.json", frontier)
    log(domain.name, f"baseline written runs/{domain.name}/frontier.json "
                     f"(delta {delta:.4f} {src})")
    return frontier


def resume_problems(domain: Domain, root: Path, ws: Workspace, frontier: dict, cfg: dict,
                    ids: list[str]) -> list[str]:
    """저장된 frontier를 지금 이어 써도 되는지(F01). 비어 있으면 이어 씁니다."""
    inc = (frontier or {}).get("incumbent") or {}
    out = []
    if not inc.get("conditions"):
        out.append("frontier에 비교 조건이 없습니다(이전 판)")
    else:
        cur = domain.conditions(root, ids, int(cfg["k"]))
        diff = sorted(k for k in set(cur) | set(inc["conditions"])
                      if cur.get(k) != inc["conditions"].get(k))
        if diff:
            out.append("비교 조건이 바뀌었습니다: " + ",".join(diff))
    if inc.get("harness_tree") != ws.tree_hash():
        out.append("현행 하네스가 frontier의 harness_tree와 다릅니다")
    out += [f"기준선 {x}" for x in domain.validity(_summary_from_frontier(frontier, cfg))]
    return out


def run_rounds(domain: Domain, root: Path, runs_dir: Path, cfg: dict, T: int) -> dict:
    cfg_full = load_config(root, domain)
    ws = Workspace(root, runs_dir, f"domains/{domain.name}/harness")
    ids = domain.evolve_ids()
    frontier = read_json(runs_dir / "frontier.json", {}) or {}
    if not frontier:
        log(domain.name, "frontier 없음 → baseline 먼저 실행")
        frontier = baseline(domain, root, runs_dir, cfg, "base")
    else:
        stale = resume_problems(domain, root, ws, frontier, cfg_full, ids)
        if stale:
            # 다른 조건에서 잰 현행 점수·S*와는 비교하지 않습니다. 이전 frontier는 보관하고 새로 잽니다.
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
            write_json(runs_dir / f"frontier.stale-{stamp}.json", frontier)
            log(domain.name, f"frontier 재설정: {'; '.join(stale)[:200]} → baseline 다시 측정")
            frontier = baseline(domain, root, runs_dir, cfg, "base")

    delta, src = effective_delta(runs_dir, cfg_full, frontier["incumbent"].get("conditions"),
                                 frontier.get("baseline_harness_tree"))
    history = History(runs_dir / "history.jsonl")
    ws.ensure_baseline()
    proposer = build_proposer(cfg_full, list(domain.briefs.get("EDITS", [])), history)
    # 라운드 식별자(F05): 실행 ID + 영속 라운드 번호. 다시 실행해도 t가 0부터 겹치지 않습니다.
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    start = int(frontier.get("rounds_done", 0) or 0)

    log(domain.name, f"delta={delta:.4f} ({src}) T={T} k={cfg_full['k']} tasks={len(ids)} "
                     f"workspace={ws.mode}")

    for i in range(T):
        t = start + i
        incumbent = _summary_from_frontier(frontier, cfg_full)
        b_t = edit_budget(i, T, int(cfg_full["b_min"]), int(cfg_full["b_max"]))
        log(domain.name, f"round {t}: b_t={b_t} delta={delta:.4f}")

        results: list[tuple[str, Any, Proposal, EvalSummary | None]] = []
        used: set[str] = set()
        for label in [chr(ord("A") + j) for j in range(int(cfg_full["m"]))]:
            name = f"r{t}{label}"
            cand_root = ws.candidate(name)
            try:
                proposal = proposer.propose(cand_root, name, b_t, t, exclude=used)
                used |= {e.key for e in proposal.edits}
                if not proposal.edits:
                    log(domain.name, f"{name}: 편집 없음 → 건너뜀")
                    results.append((name, None, proposal, None))
                    continue

                screening = critic.screen(proposal.diff, domain.critic_patterns,
                                          harness_prefix=f"domains/{domain.name}/harness/")
                if not screening.ok:
                    log(domain.name, f"{name}: critic reject → {'; '.join(screening.reasons)[:120]}")
                    _record(history, t, proposal, None, False,
                            "critic: " + "; ".join(screening.reasons), run_id=run_id)
                    results.append((name, None, proposal, None))
                    continue

                commit_edits(proposal.writes)           # 비평가를 통과한 편집만 씁니다
                smoke_ok, detail = domain.smoke(cand_root, runs_dir, name, domain.smoke_ids())
                if not smoke_ok:
                    log(domain.name, f"{name}: smoke fail ({detail.get('stage')})")
                    _record(history, t, proposal, None, False, f"smoke: {detail.get('stage')}",
                            run_id=run_id)
                    results.append((name, None, proposal, None))
                    continue
                log(domain.name, f"{name}: draft -> critic ok -> smoke ok")

                summary = evaluate(domain, cand_root, runs_dir, name, ids, int(cfg_full["k"]))
                summary.extra.setdefault("p95_ms", p95_latency(runs_dir, name))
                results.append((name, cand_root, proposal, summary))
            except Exception as exc:                    # noqa: BLE001
                import traceback
                log(domain.name, f"{name}: 실행 실패 {exc!r}")
                log(domain.name, traceback.format_exc().strip().splitlines()[-1][:200])
                results.append((name, cand_root, Proposal(name, note=repr(exc)), None))

        decisions = []
        for name, cand_root, proposal, summary in results:
            if summary is None:
                continue
            # decide가 비교 조건(comparability)을 먼저 보고, 그다음 도메인 가드를 봅니다(F01).
            guards = domain.guards(incumbent, summary)
            d = decide(summary, incumbent, delta, cfg_full, guards, proposal.novelty,
                       s_star=float(frontier.get("S_star", incumbent.S)))
            log(domain.name, f"{name}: S={summary.S:.4f} dS={d.delta_S:+.4f} "
                             f"dC={d.delta_C:+.4f} {d.reason[:90]}")
            decisions.append((name, d))

        winner = pick_best(decisions)
        if winner and ws.tree_hash() != frontier["incumbent"].get("harness_tree"):
            # 라운드 도중 현행 하네스가 바뀌었습니다. 후보는 이전 현행과 비교한 것이므로 승격하지 않습니다.
            log(domain.name, f"r{t}: 현행 하네스가 라운드 중에 바뀌어 승격을 멈춥니다")
            winner = None
            decisions = [(n, Decision(False, "aborted: incumbent harness changed",
                                      d.delta_S, d.delta_C)) for n, d in decisions]
        accepted_label = winner[0] if winner else ""
        for name, cand_root, proposal, summary in results:
            d = dict(decisions).get(name)
            accepted = bool(winner and winner[0] == name)
            commit = ""
            if accepted and cand_root is not None:
                commit = ws.promote(cand_root, f"rrsI {domain.name} t={t}: "
                                               f"{proposal.note}; {proposal.components}")
                frontier["incumbent"] = {
                    "t": t + 1,
                    "commit": commit,
                    "harness_tree": ws.tree_hash(),
                    "job": name,
                    "S": round(summary.S, 6),
                    "C": round(summary.C, 2),
                    "k": summary.k,
                    "extra": summary.extra,
                    "conditions": summary.conditions,
                }
                frontier["S_star"] = round(max(float(frontier.get("S_star", 0.0)),
                                               summary.S), 6)
                frontier.setdefault("trajectory", []).append(
                    {"t": t + 1, "S": round(summary.S, 6), "C": round(summary.C, 2),
                     "commit": commit, "job": name})
                frontier["config"] = {**frontier.get("config", {}), "delta": delta}
                _record(history, t, proposal, summary, True, d.reason if d else "", commit,
                        d.delta_S if d else None, d.delta_C if d else None, run_id=run_id)
                append_jsonl(runs_dir / "attribution.jsonl", {
                    "t": t, "run": run_id, "label": name, "components": sorted(proposal.components),
                    "hypothesis": "; ".join(e.hypothesis for e in proposal.edits),
                    "S": round(summary.S, 6), "accepted": True,
                })
            else:
                reason = (d.reason if d else "no_eval")
                _record(history, t, proposal, summary, accepted, reason, commit,
                        d.delta_S if d else None, d.delta_C if d else None, run_id=run_id)
            if cand_root is not None:
                ws.cleanup(cand_root)

        if winner:
            log(domain.name, f"r{t}: winner {accepted_label}; fast-forward "
                             f"{ws.baseline_ref()} -> {frontier['incumbent']['commit']}")
        else:
            log(domain.name, f"r{t}: 승격 없음 (정체 {history.stagnation()})")

        frontier["rounds_done"] = t + 1
        write_json(runs_dir / "frontier.json", frontier)

    return frontier


def heldout(domain: Domain, root: Path, runs_dir: Path, cfg: dict, label: str) -> dict:
    ids = domain.heldout_ids()
    job = f"heldout_{label}"
    summary = evaluate(domain, root, runs_dir, job, ids, int(cfg["k"]))
    commit = Workspace(root, runs_dir, f"domains/{domain.name}/harness").head_commit()
    out = {"label": label, "commit": commit, "S": round(summary.S, 6),
           "C": round(summary.C, 2), "n": summary.n, "extra": summary.extra,
           "split": domain.heldout_ids()[:0] or label}
    write_json(runs_dir / "heldout" / f"{label}.json", out)
    log(domain.name, f"heldout {label} @ {commit}: S={summary.S:.4f} C={summary.C:.1f}")
    return out


def status(domain: Domain, runs_dir: Path) -> None:
    frontier = read_json(runs_dir / "frontier.json", {}) or {}
    if not frontier:
        log(domain.name, "frontier 없음. `baseline`을 먼저 실행해 주세요")
        return
    inc = frontier.get("incumbent", {})
    log(domain.name, f"frontier t={inc.get('t')} S={inc.get('S')} C={inc.get('C')} "
                     f"commit={inc.get('commit')} job={inc.get('job')}")
    log(domain.name, f"S_star={frontier.get('S_star')} delta={frontier.get('config', {}).get('delta')}")
    for rec in (read_json(runs_dir / "frontier.json", {}) or {}).get("trajectory", [])[-5:]:
        log(domain.name, f"  t={rec.get('t')} S={rec.get('S')} C={rec.get('C')} {rec.get('job')}")
    hist = History(runs_dir / "history.jsonl")
    for rec in hist.records[-5:]:
        verdict = "승격" if rec.get("accepted") else "탈락"
        log(domain.name, f"  history t={rec.get('t')} {rec.get('label')} "
                         f"[{','.join(rec.get('components') or [])}] {verdict} "
                         f"{str(rec.get('reason'))[:70]}")
