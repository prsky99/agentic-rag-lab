"""rrsi CLI (7.2절).

명령은 여섯 가지입니다: smoke / baseline / run / status / heldout / calibrate.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from . import calibrate as calibrate_mod
from . import loop
from ._log import log, read_json, write_json
from .domain import EvalSummary


def load_domain(root: Path, name: str):
    sys.path.insert(0, str(root))
    module = importlib.import_module(f"domains.{name}.adapter")
    domain = getattr(module, "DOMAIN", None)
    if domain is None:
        raise SystemExit(f"domains/{name}/adapter.py에 DOMAIN이 없습니다")
    return domain


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rrsi", description="RRSI 루프 CLI")
    p.add_argument("--domain", required=True)
    p.add_argument("--root", default=".", help="저장소 루트 (기본: 현재 디렉터리)")
    p.add_argument("--T", type=int, default=None, help="라운드 수")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke"); s.add_argument("--job", default="smoke")
    s = sub.add_parser("baseline"); s.add_argument("--job", default="base")
    s = sub.add_parser("run"); s.add_argument("--T", dest="T_run", type=int, default=None)
    sub.add_parser("status")
    s = sub.add_parser("heldout"); s.add_argument("--label", default="id_heldout")
    s = sub.add_parser("calibrate"); s.add_argument("--jobs", required=True,
                                                    help="쉼표로 구분한 baseline job 이름")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    domain = load_domain(root, args.domain)
    cfg = loop.load_config(root, domain)
    runs_dir = root / "runs" / domain.name

    if args.cmd == "smoke":
        ok, detail = domain.smoke(root, runs_dir, args.job, domain.smoke_ids())
        log(domain.name, f"smoke: {'ok' if ok else 'FAIL'} {detail.get('stage')} "
                         f"{(detail.get('detail') or '')[:200]}")
        return 0 if ok else 1

    if args.cmd == "baseline":
        try:
            loop.baseline(domain, root, runs_dir, cfg, args.job)
        except loop.BaselineInvalid:             # 심판 장애·필수 값 누락: frontier를 쓰지 않습니다(F01)
            return 1
        return 0

    if args.cmd == "run":
        T = getattr(args, "T_run", None) or args.T or int(cfg.get("T", 1))
        try:
            loop.run_rounds(domain, root, runs_dir, cfg, T)
        except loop.BaselineInvalid:
            log(domain.name, "run: 기준선을 쓸 수 없어 비교·승격을 멈춥니다")
            return 1
        return 0

    if args.cmd == "status":
        loop.status(domain, runs_dir)
        return 0

    if args.cmd == "heldout":
        loop.heldout(domain, root, runs_dir, cfg, args.label)
        return 0

    if args.cmd == "calibrate":
        summaries = {}
        for job in [j.strip() for j in args.jobs.split(",") if j.strip()]:
            try:
                summaries[job] = read_json(runs_dir / "jobs" / job / "eval.json", {}) or {}
            except ValueError:                       # 깨진 eval.json은 제외 사유로 남깁니다
                summaries[job] = {}

        def validity(s: dict) -> list[str]:          # 도메인 검사: 심판 장애·필수 값 누락(F02)
            return domain.validity(EvalSummary(
                S=s.get("S"), C=s.get("C", 0.0), k=s.get("k"), extra=dict(s.get("extra") or {}),
                conditions=dict(s.get("conditions") or {}), harness_tree=str(s.get("harness_tree") or "")))

        out = calibrate_mod.from_summaries(summaries, z=float(cfg.get("delta_z", 2.0)),
                                           validity=validity)
        for job, why in out["excluded"].items():
            log(domain.name, f"calibrate: {job} 제외 — {'; '.join(why)[:160]}")
        if out["delta"] is None:
            log(domain.name, f"calibrate: 실패 — {out.get('note')} (유효 job {len(out['scores'])}개, "
                             f"같은 조건의 baseline {calibrate_mod.MIN_RUNS}회 이상 필요)")
            return 1
        write_json(runs_dir / "calibration.json", out)
        if out["delta"] in (0, 0.0):
            log(domain.name, "calibrate: sd=0 (결정적 로컬 모드). 실제 모델을 연결해 다시 측정해 보세요")
        log(domain.name, f"calibrated delta={out['delta']} "
                         f"(sd_null {out['sd_null']}, z={out['z']}, n={out['n']}) "
                         f"→ runs/{domain.name}/calibration.json")
        return 0

    return 2
