#!/usr/bin/env python3
"""PreToolUse guard for autonomous sessions (see docs/milestones/M1-M2.md, "Rules").

Denies, in every permission mode:
  * git push to main/master, or any force push;
  * Modal spend (modal run/deploy/serve/shell, `s1 submit` of a Modal config, `s1 download`, the launcher's
    Modal entrypoints) unless the exact command matches an approved, unused ledger entry;
  * writing anything that looks like a Modal or Hugging Face token into a file or command;
  * modal token/setup commands (the user owns the credentials);
  * any write to the GPU ledger except through ledger.py, and ledger.py `add` from inside a worktree.

Stdlib only. Prints a deny decision as JSON on stdout; prints nothing to let the call through to the normal
permission flow. A crash prints a deny so a broken guard fails closed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger

SECRET = re.compile(
    r"\b(?:ak|as)-[A-Za-z0-9]{16,}\b"  # Modal token id / secret
    r"|\bhf_[A-Za-z0-9]{30,}\b"  # Hugging Face token
    r"|MODAL_TOKEN_SECRET\s*=\s*['\"]?[A-Za-z0-9-]{8,}"
)
PUSH = re.compile(r"\bgit\b[^;&|]*\bpush\b")
FORCE = re.compile(r"(?:^|\s)(?:--force(?:-with-lease)?|-f)(?:\s|$|=)|\s\+\S")
PROTECTED_BRANCH = re.compile(r"(?:^|[\s:/])(?:main|master)(?:\s|$)")
MODAL_SPEND = re.compile(r"\bmodal\s+(?:run|deploy|serve|shell|launch)\b")
MODAL_CREDS = re.compile(r"\bmodal\s+(?:token|setup|profile\s+activate)\b")
S1_SPEND = re.compile(r"\bs1\s+(?:submit|download)\b|\bs1\.launch\.modal_app\b")
LEDGER_WORD = "gpu-ledger"
MESSAGE_ARG = re.compile(r"""(\s(?:-m|--message|--body|--title|-t|-b))\s+("(?:\\.|[^"\\])*"|'[^']*')""")


def deny(reason: str) -> None:
    out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                  "permissionDecisionReason": f"[guard] {reason}"}}  # fmt: skip
    print(json.dumps(out))
    sys.exit(0)


def in_worktree(cwd: str) -> bool:
    return "/.claude/worktrees/" in cwd + "/"


def submit_is_modal(command: str, cwd: str) -> bool:
    """`s1 submit <cfg>` spends Modal credits when --on modal is given or the config's [launch] says modal."""
    if re.search(r"--on[=\s]+modal\b", command):
        return True
    if re.search(r"--on[=\s]+slurm\b", command):
        return False
    m = re.search(r"\bs1\s+submit\s+(\S+\.toml)", command)
    if not m:
        return True  # cannot tell: treat as spend
    cfg = Path(cwd, m.group(1))
    try:
        text = cfg.read_text()
    except OSError:
        return True
    return re.search(r"""^\s*orchestrator\s*=\s*["']modal["']""", text, re.M) is not None


def check_bash(command: str, cwd: str) -> None:
    if SECRET.search(command):
        deny(
            "command contains what looks like a Modal or HF token; credentials stay out of commands and files"
        )
    if MODAL_CREDS.search(command):
        deny("modal token/setup/profile commands are the user's to run")
    for part in re.split(r"&&|\|\||;|\n", command):
        if PUSH.search(part):
            if FORCE.search(part):
                deny("force push is not allowed")
            if PROTECTED_BRANCH.search(part):
                deny("pushing to main/master is not allowed; push your own branch")
    if LEDGER_WORD in command or "ledger.py" in command:
        if re.search(r"\bledger\.py\s+(?:add|cap)\b", command) and in_worktree(cwd):
            deny("only the coordinator (main checkout) may add ledger entries")
        writes = re.search(
            r">\s*\S*" + LEDGER_WORD + r"|\b(?:tee|mv|cp|rm|truncate|chmod|ln)\b[^;&|]*" + LEDGER_WORD
            + r"|sed\s+-i[^;&|]*" + LEDGER_WORD + r"|\b(?:python3?|perl|ruby|node)\b[^;&|]*" + LEDGER_WORD,
            command,
        )  # fmt: skip
        if writes and not re.search(r"\bledger\.py\b", command):
            deny("the GPU ledger is written only through .claude/hooks/ledger.py")
    # Message text (commit -m, gh --body/--title) can mention commands without running them.
    run_text = MESSAGE_ARG.sub(r"\1 ''", command)
    if re.search(r"\bmodal\s+deploy\b", run_text):
        deny("modal deploy is never allowed (it creates a persistent app)")
    spends = MODAL_SPEND.search(run_text) or (
        S1_SPEND.search(run_text)
        and (not re.search(r"\bs1\s+submit\b", run_text) or submit_is_modal(run_text, cwd))
    )
    if spends and not re.search(r"--dry-run\b", run_text):
        entry = ledger.consume(command)
        if entry is None:
            deny("Modal/GPU spend is not in the approved ledger. Stop and report the exact command and why; "
                 "the coordinator asks the user. Approved entries: " + ledger.summary())  # fmt: skip


def check_write(tool_input: dict[str, object]) -> None:
    path = str(tool_input.get("file_path", ""))
    if LEDGER_WORD in path:
        deny("the GPU ledger is written only through .claude/hooks/ledger.py")
    text = " ".join(str(tool_input.get(k, "")) for k in ("content", "new_string", "new_source"))
    for edit in tool_input.get("edits", []) or []:
        if isinstance(edit, dict):
            text += " " + str(edit.get("new_string", ""))
    if SECRET.search(text):
        deny(f"refusing to write what looks like a Modal or HF token into {path}")


def main() -> None:
    data = json.load(sys.stdin)
    tool = data.get("tool_name", "")
    tool_input = data.get("tool_input", {}) or {}
    cwd = str(data.get("cwd", ""))
    if tool == "Bash":
        check_bash(str(tool_input.get("command", "")), cwd)
    elif tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        check_write(tool_input)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        deny(f"guard error ({type(exc).__name__}: {exc}); fix .claude/hooks/guard.py")
