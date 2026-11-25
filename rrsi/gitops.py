"""git 작업장 관리 (7.5절).

후보는 격리된 작업장에서 만듭니다. 같은 디렉터리에서 두 후보를 편집하면 서로
섞이고, 승격한 편집이 무엇인지 나중에 추적하기 어렵습니다.

git 저장소면 ``git worktree``를 쓰고, 아니면(압축을 풀어 바로 쓰는 경우)
디렉터리를 복사합니다. 두 경우 모두 "후보 → 승격"의 의미는 같습니다. 편집면 안의
파일만 현재 트리로 옮깁니다.

git 모드에서는 후보가 작업 트리가 아니라 ``evolve/rag`` 커밋에서 갈라져 나옵니다. 그래서 기준선을
재기 전에 작업 트리가 깨끗한지(커밋하지 않은 변경·새 파일이 없는지)와 ``evolve/rag``가 HEAD와
같은 커밋인지 확인합니다(G03). 둘 중 하나라도 어긋나면 기준선과 후보가 서로 다른 코드에서
출발하므로 ``WorkspaceError``로 멈춥니다. 복사 모드는 작업 트리 자체를 복사하므로 해당하지 않습니다.
"""
from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

# 복사 모드에서 후보 작업장에 옮기지 않는 것(F06): 실행 산출물, 가상환경, 캐시, 빌드 결과.
# 가상환경은 복사하지 않고 지금 실행 중인 인터프리터(sys.executable)를 후보도 함께 씁니다.
IGNORE = shutil.ignore_patterns("runs", ".git", "__pycache__", "*.pyc",
                                ".venv", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                                ".tox", ".nox", "node_modules", "build", "dist", "*.egg-info")


def copy_ignore(src: str, names: list[str]) -> set[str]:
    """``IGNORE`` + 이름과 무관하게 ``pyvenv.cfg``가 있는 폴더(가상환경)를 뺍니다(F06)."""
    out = set(IGNORE(src, names))
    for n in names:
        if n not in out and (Path(src) / n / "pyvenv.cfg").is_file():
            out.add(n)
    return out


class WorkspaceError(RuntimeError):
    """후보 출발 코드와 기준선 코드가 같다고 보장할 수 없는 상태(G03). 비교·승격을 멈춥니다."""


def _run(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=check)


def is_git_repo(root: Path) -> bool:
    try:
        return _run(["git", "rev-parse", "--is-inside-work-tree"], Path(root)).stdout.strip() == "true"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


@dataclass
class Workspace:
    root: Path
    runs_dir: Path
    harness_rel: str

    @property
    def mode(self) -> str:
        return "git" if is_git_repo(self.root) else "copy"

    def baseline_ref(self) -> str:
        return "evolve/rag"

    def dirty_paths(self) -> list[str]:
        """커밋하지 않은 변경(staged·unstaged·추적하지 않는 새 파일) 목록. ``runs/``는 뺍니다."""
        if self.mode != "git":
            return []
        out = _run(["git", "status", "--porcelain", "--untracked-files=all"], self.root,
                   check=False)
        if out.returncode != 0:
            return [f"git status 실패: {(out.stderr or out.stdout).strip()[:120]}"]
        paths = []
        for line in out.stdout.splitlines():
            rel = line[3:].strip().strip('"')
            if rel and not rel.startswith("runs/"):
                paths.append(f"{line[:2].strip() or '?'} {rel}")
        return paths

    def require_clean(self) -> None:
        """git 모드에서 커밋하지 않은 변경이 있으면 멈춥니다(G03). 복사 모드는 통과합니다."""
        dirty = self.dirty_paths()
        if dirty:
            raise WorkspaceError(
                "커밋하지 않은 변경이 있습니다: " + ", ".join(dirty[:5])
                + (f" 외 {len(dirty) - 5}개" if len(dirty) > 5 else "")
                + " → 후보는 커밋에서 갈라지므로 먼저 git add·git commit 후 다시 실행해 주세요")

    def _rev(self, ref: str) -> str | None:
        out = _run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], self.root, check=False)
        return out.stdout.strip() if out.returncode == 0 else None

    def ensure_baseline(self) -> str:
        """evolve/rag 브랜치를 HEAD에 맞춥니다. 맞출 수 없으면 ``WorkspaceError``로 멈춥니다(G03)."""
        if self.mode != "git":
            return "no-git"
        if self._rev(self.baseline_ref()) is None:
            _run(["git", "branch", self.baseline_ref()], self.root)
        else:
            # 그 뒤 HEAD에 커밋이 쌓였으면(직접 고친 하네스, git pull) evolve/rag를 HEAD로 옮깁니다.
            # 그대로 두면 후보가 예전 하네스에서 갈라져 나와 baseline과 다른 코드를 비교하게 됩니다.
            ff = _run(["git", "merge-base", "--is-ancestor", self.baseline_ref(), "HEAD"],
                      self.root, check=False)
            if ff.returncode == 0:
                _run(["git", "branch", "-f", self.baseline_ref(), "HEAD"], self.root, check=False)
        # 갈라졌거나(분기) HEAD가 evolve/rag보다 뒤에 있으면(역행) 옮기지 않고 멈춥니다.
        # 어느 쪽 코드를 기준선으로 삼을지는 사람이 정해야 합니다.
        head, base = self._rev("HEAD"), self._rev(self.baseline_ref())
        if head is None or head != base:
            raise WorkspaceError(
                f"{self.baseline_ref()}({(base or '없음')[:12]})가 HEAD({(head or '없음')[:12]})와 "
                f"다릅니다(분기 또는 역행) → 기준선으로 삼을 커밋을 정해 "
                f"git branch -f {self.baseline_ref()} <커밋> 후 다시 실행해 주세요")
        return self.head_commit()

    def head_commit(self) -> str:
        if self.mode != "git":
            return "no-git"
        try:
            return _run(["git", "rev-parse", "--short", "HEAD"], self.root).stdout.strip()
        except subprocess.CalledProcessError:
            return "no-head"

    def tree_hash(self, rel: str | None = None) -> str:
        """하네스 트리의 해시. 같은 하네스인지 비교하는 식별자입니다."""
        target = self.root / (rel or self.harness_rel)
        import hashlib
        h = hashlib.sha256()
        for p in sorted(target.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(target)).encode())
                h.update(p.read_bytes())
        return h.hexdigest()[:16]

    def candidate(self, label: str) -> Path:
        wt = self.runs_dir / "wt" / label.replace("/", "_")
        if wt.exists():
            shutil.rmtree(wt, ignore_errors=True)
        wt.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "git":
            self.ensure_baseline()
            # runs/를 지운 뒤에도 git에 남아 있는 작업장 등록을 정리합니다
            # (남아 있으면 worktree add가 'already registered'로 실패합니다).
            _run(["git", "worktree", "prune"], self.root, check=False)
            branch =f"cand/{label.replace('/', '_')}_{uuid.uuid4().hex[:6]}"
            _run(["git", "worktree", "add", "-b", branch, str(wt), self.baseline_ref()],
                 self.root)
        else:
            shutil.copytree(self.root, wt, ignore=copy_ignore)
        return wt

    def changed_files(self, candidate: Path, base: Path | None = None) -> list[str]:
        base = base or self.root
        files = []
        for p in (candidate / self.harness_rel).rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            rel = p.relative_to(candidate)
            other = base / rel
            if not other.exists() or other.read_bytes() != p.read_bytes():
                files.append(str(rel))
        return files

    def promote(self, candidate: Path, message: str) -> str:
        """승자의 편집면 파일만 현재 트리로 옮기고 커밋합니다.

        git 모드에서 ``git add``·``git commit``이 실패하면 옮긴 파일을 원래 내용으로 되돌리고
        ``WorkspaceError``를 냅니다. 실패한 승격을 성공으로 기록하지 않습니다(G03).
        """
        files = self.changed_files(candidate)
        before = {rel: ((self.root / rel).read_bytes() if (self.root / rel).exists() else None)
                  for rel in files}
        for rel in files:
            src, dst = candidate / rel, self.root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
        if self.mode != "git" or not files:
            return self.head_commit()
        old_head = self._rev("HEAD")
        add = _run(["git", "add", "--", *files], self.root, check=False)
        com = _run(["git", "-c", "user.name=rrsI", "-c", "user.email=rrsi@local",
                    "commit", "-m", message, "--", *files], self.root, check=False)             if add.returncode == 0 else add
        if com.returncode != 0 or self._rev("HEAD") == old_head:
            # 커밋되지 않았습니다. 작업 트리와 색인을 승격 전 상태로 되돌립니다.
            _run(["git", "reset", "-q", "--", *files], self.root, check=False)
            for rel, data in before.items():
                if data is None:
                    (self.root / rel).unlink(missing_ok=True)
                else:
                    (self.root / rel).write_bytes(data)
            raise WorkspaceError("승격 커밋 실패: " + (com.stderr or com.stdout).strip()[:200])
        # 다음 라운드 후보가 새 현행에서 분기하도록 evolve/rag를 승격 커밋으로 옮깁니다.
        # (현재 브랜치가 evolve/rag이면 이미 같은 커밋이므로 실패해도 무방합니다)
        _run(["git", "branch", "-f", self.baseline_ref(), "HEAD"], self.root, check=False)
        return self.head_commit()

    def cleanup(self, candidate: Path) -> None:
        if self.mode == "git":
            branch = self._branch_of(candidate)          # worktree를 지우기 전에 브랜치 이름을 읽습니다
            _run(["git", "worktree", "remove", str(candidate), "--force"], self.root, check=False)
            if branch:
                _run(["git", "branch", "-D", branch], self.root, check=False)
        shutil.rmtree(candidate, ignore_errors=True)

    def _branch_of(self, candidate: Path) -> str | None:
        try:
            out = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], candidate, check=False)
            name = out.stdout.strip()
            return name if name.startswith("cand/") else None
        except Exception:                           # noqa: BLE001
            return None
