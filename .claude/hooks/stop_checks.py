#!/usr/bin/env python3
"""Stop hook: a session working in a worktree may not stop until the repo's checks pass.

Runs only under .claude/worktrees/ and only when the branch differs from m1-integration (or has uncommitted
changes), so the coordinator and read-only sessions are never held. Checks: ruff format --check, ruff check,
pytest, and no ty diagnostic beyond .claude/ty-baseline.txt (keyed by file, rule and message, not line).
On failure prints {"decision": "block", "reason": ...} so the agent keeps working; Claude Code itself gives up
after repeated blocks with no progress.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASE = "m1-integration"


def run(cmd: list[str], cwd: Path, timeout: int = 900) -> tuple[int, str]:
    p = subprocess.run(cmd, check=False, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-3000:]


def ty_keys(text: str) -> Counter[str]:
    keys: Counter[str] = Counter()
    for line in text.splitlines():
        m = re.match(r"^(\S+?):\d+:\d+: (\w+)\[([\w-]+)\] (.*)$", line.strip())
        if m:
            keys[f"{m.group(1)}|{m.group(3)}|{m.group(4)}"] += 1
    return keys


def changed(root: Path) -> bool:
    dirty = run(["git", "status", "--porcelain", "--untracked-files=no"], root)[1].strip()
    if dirty:
        return True
    code, out = run(["git", "rev-list", "--count", f"{BASE}..HEAD"], root)
    return code != 0 or out.strip() != "0"


def main() -> None:
    data = json.load(sys.stdin)
    cwd = Path(str(data.get("cwd") or "."))
    if "/.claude/worktrees/" not in str(cwd.resolve()) + "/":
        return
    root = Path(run(["git", "rev-parse", "--show-toplevel"], cwd)[1].strip())
    if not changed(root):
        return
    failures: list[str] = []
    for name, cmd in (
        ("ruff format --check", ["uv", "run", "ruff", "format", "--check"]),
        ("ruff check", ["uv", "run", "ruff", "check"]),
        ("pytest", ["uv", "run", "pytest", "-q", "-x"]),
    ):
        code, out = run(cmd, root)
        if code != 0:
            failures.append(f"{name} failed:\n{out}")
    _, ty_out = run(["uv", "run", "ty", "check", "--output-format", "concise"], root)
    baseline_file = root / ".claude" / "ty-baseline.txt"
    baseline = Counter(baseline_file.read_text().splitlines()) if baseline_file.exists() else Counter()
    new = ty_keys(ty_out) - baseline
    if new:
        failures.append("new ty diagnostics (not in .claude/ty-baseline.txt):\n" + "\n".join(sorted(new)))
    if failures:
        reason = "Checks must pass before you finish (docs/milestones/M1-M2.md rule 1):\n\n" + "\n\n".join(
            failures
        )
        print(json.dumps({"decision": "block", "reason": reason[:8000]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"systemMessage": f"stop_checks.py error: {type(exc).__name__}: {exc}"}))
