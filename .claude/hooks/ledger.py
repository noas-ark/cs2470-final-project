#!/usr/bin/env python3
"""The approved-GPU-spend ledger: which Modal commands may run, and how many times.

The ledger lives in the MAIN checkout (shared by every worktree) at .claude/gpu-ledger.json, gitignored.
Entries are added only by the coordinator, only for runs the user approved in chat:

    python3 .claude/hooks/ledger.py add <id> '<regex for the command>' <uses> <est_usd_per_use> '<approval>'
    python3 .claude/hooks/ledger.py cap <usd>     # refuse any run that would take estimated spend past this
    python3 .claude/hooks/ledger.py list

guard.py calls consume(): the first entry whose regex matches the command and has uses left is decremented
and the use is logged with a timestamp and its estimated cost. Estimates, not Modal's bill.
"""

from __future__ import annotations

import fcntl
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def path() -> Path:
    here = Path(__file__).resolve().parent
    try:
        common = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        return Path(common).parent / ".claude" / "gpu-ledger.json"
    except (OSError, subprocess.CalledProcessError):
        return here.parent / "gpu-ledger.json"


def _load(p: Path) -> dict[str, list[dict[str, object]]]:
    if not p.exists():
        return {"approved": [], "log": []}
    return json.loads(p.read_text())


def _update(fn):  # noqa: ANN001, ANN202
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        raw = fh.read()
        data = json.loads(raw) if raw.strip() else {"approved": [], "log": []}
        result = fn(data)
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(data, indent=2) + "\n")
        return result


def _spent(data: dict[str, object]) -> float:
    log = data.get("log", [])
    return sum(float(str(e.get("est_usd", 0))) for e in log if isinstance(e, dict))


def consume(command: str) -> dict[str, object] | None:
    """Use one approved run matching `command`, or None if none matches or it would pass the dollar cap."""

    def take(data: dict) -> dict[str, object] | None:
        cap = data.get("cap_usd")
        for entry in data["approved"]:
            if int(str(entry.get("uses", 0))) > 0 and re.search(str(entry["match"]), command):
                est = float(str(entry.get("est_usd", 0)))
                if cap is not None and _spent(data) + est > float(str(cap)):
                    return None
                entry["uses"] = int(str(entry["uses"])) - 1
                data["log"].append({"id": entry["id"], "at": datetime.now().isoformat(timespec="seconds"),
                                    "est_usd": est, "command": command[:500]})  # fmt: skip
                return entry
        return None

    return _update(take)


def summary() -> str:
    data = _load(path())
    live = [f"{e['id']} ({e['uses']} left, ~${e.get('est_usd', 0)}/use)" for e in data["approved"]
            if int(str(e.get("uses", 0))) > 0]  # fmt: skip
    cap = data.get("cap_usd")
    money = f"estimated spend ${_spent(data):.2f}" + (f" of ${cap} cap" if cap is not None else "")
    return (", ".join(live) or "none") + f"; {money}"


def main(argv: list[str]) -> None:
    if argv[:1] == ["list"]:
        print(json.dumps(_load(path()), indent=2))
        print(summary())
        return
    if len(argv) == 2 and argv[0] == "cap":
        cap = float(argv[1])

        def set_cap(data: dict) -> None:
            data["cap_usd"] = cap

        _update(set_cap)
        print(f"cap set to ${cap} in {path()}")
        return
    if len(argv) == 6 and argv[0] == "add":
        _, entry_id, match, uses, est_usd, note = argv
        re.compile(match)

        def add(data: dict) -> None:
            entry = {
                "id": entry_id,
                "match": match,
                "uses": int(uses),
                "est_usd": float(est_usd),
                "note": note,
            }
            entry["added"] = datetime.now().isoformat(timespec="seconds")
            data["approved"].append(entry)

        _update(add)
        print(f"added {entry_id} ({uses} uses, ~${est_usd}/use)")
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
