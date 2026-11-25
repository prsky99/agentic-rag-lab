"""제안자 (7.3절).

두 모드를 지원합니다.

* ``catalog`` — 도메인의 ``briefs.EDITS``에 적힌 편집 카탈로그에서 고릅니다.
  외부 모델 없이도 RRSI 루프 전체를 실행해 볼 수 있게 하는 기본 모드입니다.
* ``llm`` — 같은 편집을 LLM에게 제안받습니다. 출력은 JSON 한 개로 제한하고,
  파일 경로가 편집면 안인지 반드시 검사합니다.

두 모드 모두 결과는 "파일 + 교체할 문자열 + 새 문자열"입니다. diff를 직접
쓰게 하지 않는 이유는 적용 실패를 결정적으로 만들기 위해서입니다.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .components import novelty
from .history import History


@dataclass
class Edit:
    component: str
    hypothesis: str
    file: str
    old: str
    new: str

    @property
    def key(self) -> str:
        return f"{self.file}::{self.old.strip()}->{self.new.strip()}"


@dataclass
class Proposal:
    label: str
    edits: list[Edit] = field(default_factory=list)
    diff: str = ""
    components: set[str] = field(default_factory=set)
    novelty: float = 0.0
    note: str = ""
    writes: dict = field(default_factory=dict)   # 비평가 통과 뒤에 쓸 내용(경로 → 새 글)


class EditError(RuntimeError):
    pass


def _applicable(root: Path, edit: Edit) -> bool:
    path = Path(root) / edit.file
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return edit.old in text or bool(edit.new and edit.new in text)


HARNESS_PREFIX = "domains/rag/harness/"


def _safe_target(root: Path, file: str, prefix: str = HARNESS_PREFIX) -> Path:
    """편집 대상 경로를 정규화해 편집면(prefix) 안에 있는지 확인합니다(E01).

    절대 경로, 상위 폴더(..), 편집면 밖으로 이어지는 심볼릭 링크는 거절합니다.
    문자열이 prefix로 시작하는지만 보면 'domains/rag/harness/../../x'가 통과합니다.
    """
    if not file or Path(file).is_absolute() or "\\" in file or ".." in Path(file).parts:
        raise EditError(f"unsafe path: {file!r}")
    base = (Path(root) / prefix).resolve()
    target = (Path(root) / file).resolve()          # 링크도 실제 위치로 풉니다
    if target != base and base not in target.parents:
        raise EditError(f"path outside edit surface: {file!r}")
    return target


def plan_edits(root: Path, edits: list[Edit], prefix: str = HARNESS_PREFIX) -> tuple[str, dict[Path, str]]:
    """편집을 검증만 하고 (실제 unified diff, 쓸 내용)을 돌려줍니다. 파일은 바꾸지 않습니다.

    같은 교체가 이미 적용되어 있으면(앞 라운드의 승격) 건너뜁니다. 그래서 라운드를
    재개해도 결과가 같습니다. 바꿀 문자열이 비었거나 파일 안에서 두 번 이상 나오면
    어디를 바꿀지 모호하므로 거절합니다.
    """
    import difflib
    root = Path(root)
    texts: dict[Path, str] = {}
    originals: dict[Path, str] = {}
    names: dict[Path, str] = {}
    for e in edits:
        path = _safe_target(root, e.file, prefix)
        if not path.exists():
            raise EditError(f"missing file: {e.file}")
        if path not in texts:
            originals[path] = texts[path] = path.read_text(encoding="utf-8")
            names[path] = e.file
        text = texts[path]
        if e.new and e.new in text and e.old not in text:
            continue                                   # 이미 적용됨
        if not e.old:
            raise EditError(f"empty anchor in {e.file}")
        count = text.count(e.old)
        if count == 0:
            raise EditError(f"anchor not found in {e.file}: {e.old!r}")
        if count > 1:
            raise EditError(f"ambiguous anchor ({count} matches) in {e.file}: {e.old!r}")
        texts[path] = text.replace(e.old, e.new, 1)
    chunks = []
    writes: dict[Path, str] = {}
    for path, new_text in texts.items():
        if new_text == originals[path]:
            continue
        writes[path] = new_text
        chunks.extend(difflib.unified_diff(originals[path].splitlines(keepends=True),
                                           new_text.splitlines(keepends=True),
                                           fromfile="a/" + names[path], tofile="b/" + names[path]))
    return "".join(c if c.endswith("\n") else c + "\n" for c in chunks), writes


def commit_edits(writes: dict[Path, str]) -> None:
    """검증(비평가 통과)이 끝난 편집만 실제 파일에 씁니다(E01·E02)."""
    for path, text in writes.items():
        path.write_text(text, encoding="utf-8")


def apply_edits(root: Path, edits: list[Edit]) -> str:
    """검증한 뒤 바로 적용합니다(시험·도구용). 루프는 plan_edits → 비평가 → commit_edits 순서를 씁니다."""
    diff, writes = plan_edits(root, edits)
    commit_edits(writes)
    return diff


class CatalogProposer:
    """결정적 제안자. 같은 이력이면 같은 후보를 냅니다."""

    def __init__(self, catalog: list[Edit], history: History) -> None:
        self.catalog = list(catalog)
        self.history = history

    def propose(self, root: Path, label: str, budget: int, t: int,
                exclude: set[str] | None = None) -> Proposal:
        exclude = exclude or set()
        recent = self.history.touched(window=3)
        pool = [e for e in self.catalog if e.key not in exclude and _applicable(root, e)]
        if not pool:
            pool = [e for e in self.catalog if _applicable(root, e)] or list(self.catalog)

        # 미탐색 컴포넌트를 먼저, 그다음 카탈로그 순서(결정적)
        order = {id(e): i for i, e in enumerate(self.catalog)}
        pool.sort(key=lambda e: (e.component in recent, order[id(e)]))

        chosen: list[Edit] = []
        used_components: set[str] = set()
        for e in pool:
            if len(chosen) >= budget:
                break
            if e.component in used_components and len(pool) > budget:
                continue
            chosen.append(e)
            used_components.add(e.component)

        applied, writes = plan_edits(Path(root), chosen)
        comps = {e.component for e in chosen}
        return Proposal(label=label, edits=chosen, diff=applied, components=comps, writes=writes,
                        novelty=novelty(comps, self.history.records),
                        note=f"catalog x{len(chosen)}")


PROMPT = """당신은 RAG 하네스의 편집 제안자입니다. 아래 규칙과 이력을 지켜 주세요.
- 편집은 domains/rag/harness/ 안의 파일에서만 합니다.
- 출력은 JSON 하나: {"file": "...", "old": "기존 줄", "new": "새 줄",
  "component": "prompt|gate_policy|retrieval_tool|context_mgmt|control_flow|memory|subagent",
  "hypothesis": "X를 Y로 바꾸면 Z 지표가 오릅니다"}
- 특정 질문·문서·테넌트를 지목하는 편집과 정답 문자열을 넣는 편집은 금지입니다.
- 이미 시도한 가설 대신 새 가설을 제안합니다.
"""


DOMAIN_DIR = "domains/rag"
DOMAIN_DOCS = ("SKILL.md", "PATTERNS.md")       # 제안자에게 본문을 넘기는 문서(이 둘만)
DOC_LIMIT = 6000                                # 문서 하나당 글자 상한


def read_domain_doc(root: Path, name: str, limit: int = DOC_LIMIT) -> str:
    """도메인 폴더 바로 아래의 정해진 문서만 읽어 ``limit``자까지 돌려줍니다(N-E02).

    이름은 ``DOMAIN_DOCS``에 있는 것만 받고, 링크를 풀어도 도메인 폴더 안에 있어야 합니다.
    평가 데이터(``data/``)나 편집 경계 밖 파일은 이 함수로 읽을 수 없습니다.
    """
    if name not in DOMAIN_DOCS:
        raise EditError(f"not a proposer document: {name!r}")
    base = (Path(root) / DOMAIN_DIR).resolve()
    path = (base / name).resolve()
    if path.parent != base or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= limit else text[:limit] + "\n…(이하 생략)"


def proposer_context(root: Path, history: History, t: int, budget: int) -> dict:
    """LLM 제안자에게 보내는 입력. 규칙(SKILL.md)·패턴(PATTERNS.md) 본문, 이력 요약,
    최근 편집 원문, 현재 config.yaml을 담습니다. 모두 크기 상한이 있습니다."""
    cfg = _safe_target(root, HARNESS_PREFIX + "config.yaml")
    config = cfg.read_text(encoding="utf-8") if cfg.is_file() else ""
    return {
        "round": t, "budget": budget,
        "skill": read_domain_doc(root, "SKILL.md"),
        "patterns": read_domain_doc(root, "PATTERNS.md"),
        "history": history.to_prompt(),
        "recent_edits": history.recent_edits(),
        "config": config[:DOC_LIMIT],
    }


class LLMProposer:
    """LLM 제안자. 모델이 없거나 출력이 깨지면 카탈로그로 되돌아갑니다."""

    def __init__(self, catalog: list[Edit], history: History,
                 base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None) -> None:
        self.fallback = CatalogProposer(catalog, history)
        self.history = history
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("PROPOSER_MODEL", "gpt-6-luna")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def _ask(self, context: str) -> dict | None:
        if not (self.base_url and self.api_key):
            return None
        payload = {"model": self.model, "temperature": 0.3,
                   "messages": [{"role": "system", "content": PROMPT},
                                {"role": "user", "content": context}]}
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json",
                     "authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode())
            text = body["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", text, flags=re.S)
            return json.loads(match.group(0)) if match else None
        except Exception:                              # noqa: BLE001
            return None

    def propose(self, root: Path, label: str, budget: int, t: int,
                exclude: set[str] | None = None) -> Proposal:
        context = json.dumps(proposer_context(Path(root), self.history, t, budget),
                             ensure_ascii=False)
        obj = self._ask(context)
        if not obj:
            prop = self.fallback.propose(root, label, budget, t, exclude)
            prop.note = "catalog (llm unavailable)"
            return prop

        file = str(obj.get("file", ""))
        edit = Edit(component=str(obj.get("component", "control_flow")),
                    hypothesis=str(obj.get("hypothesis", "")),
                    file=file, old=str(obj.get("old", "")), new=str(obj.get("new", "")))
        try:
            applied, writes = plan_edits(Path(root), [edit])
        except EditError as exc:
            prop = self.fallback.propose(root, label, budget, t, exclude)
            prop.note = f"catalog (llm edit rejected: {str(exc)[:60]})"
            return prop
        return Proposal(label=label, edits=[edit], diff=applied, writes=writes,
                        components={edit.component},
                        novelty=novelty({edit.component}, self.history.records),
                        note="llm")


def build_proposer(cfg: dict, catalog: list[Edit], history: History):
    mode = str(cfg.get("proposer", "catalog")).lower()
    if mode == "llm":
        return LLMProposer(catalog, history)
    return CatalogProposer(catalog, history)
