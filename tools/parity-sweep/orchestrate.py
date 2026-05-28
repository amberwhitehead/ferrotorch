#!/usr/bin/env python3
"""
Orchestrator for the ferrotorch ↔ PyTorch parity-sweep agent loop.

The actual three-agent dance (reader-corrector → discriminator → re-corrector)
is launched via `crosslink kickoff` so each agent runs in its own tmux session
with isolated context. This script prepares the per-op working directory and
prints the exact kickoff commands you (the human) run for each phase.

Why human-driven kickoff: each phase produces concrete artifacts (sweep logs,
discriminator findings, diffs) that must be inspected before the next phase
starts. The orchestrator does not auto-advance — that would defeat the
discriminator's adversarial role.

Subcommands
-----------

  orchestrate.py prepare <op>
      Render the three agent prompts into tools/parity-sweep/runs/<op>/prompts/
      with {{OP}} substituted. Print the kickoff commands.

  orchestrate.py status [<op>]
      Show the current audit state for one op or all ops.

  orchestrate.py record-sweep <op>
      Re-run the runner sweep, update parity_audit.json with fresh counts and
      a list of currently-failing samples (written to runs/<op>/divergences.json).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "parity_audit.json"
AGENT_DIR = ROOT / "agents"
RUNS_DIR = ROOT / "runs"


def load_audit() -> dict:
    return json.loads(AUDIT.read_text())


def save_audit(audit: dict) -> None:
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")


def render_prompt(template_path: Path, op: str) -> str:
    return template_path.read_text().replace("{{OP}}", op)


def prepare(op: str) -> int:
    audit = load_audit()
    if op not in audit["ops"]:
        # Pre-populate a "missing" entry. Source paths are null until the
        # reader-corrector reads PyTorch + ferrotorch and fills them in;
        # populating them here would be a guess that future tooling reads as truth.
        audit["ops"][op] = {
            "status": "missing",
            "last_sweep_at": None,
            "samples_attempted": 0, "samples_passed": 0,
            "samples_failed": 0, "samples_skipped": 0,
            "discriminator_rounds": 0,
            "pytorch_source": None,
            "ferrotorch_source": None,
            "known_divergences": [],
        }
        save_audit(audit)

    op_dir = RUNS_DIR / op
    prompts_dir = op_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    for phase in ("reader-corrector", "discriminator", "re-corrector"):
        src = AGENT_DIR / f"{phase}.md"
        dst = prompts_dir / f"{phase}.md"
        dst.write_text(render_prompt(src, op))

    print(f"\nPrepared prompts in {prompts_dir.relative_to(Path.cwd())}\n")
    print(f"Run each phase in order, inspecting artifacts between phases:\n")
    print(f"  # Phase 1 — read + correct")
    print(f"  crosslink kickoff run --prompt-file {prompts_dir}/reader-corrector.md \\")
    print(f"      --session parity-{op}-rc --branch parity/{op}")
    print()
    print(f"  # After phase 1 commits, re-run the sweep to update divergences:")
    print(f"  python3 {Path(__file__).relative_to(Path.cwd())} record-sweep {op}")
    print()
    print(f"  # Phase 2 — adversarial probing (does NOT edit code)")
    print(f"  crosslink kickoff run --prompt-file {prompts_dir}/discriminator.md \\")
    print(f"      --session parity-{op}-disc --branch parity/{op}-disc")
    print()
    print(f"  # Phase 3 — fix what the discriminator found")
    print(f"  crosslink kickoff run --prompt-file {prompts_dir}/re-corrector.md \\")
    print(f"      --session parity-{op}-rerc --branch parity/{op}")
    print()
    return 0


def status(op: str | None) -> int:
    audit = load_audit()
    ops = [op] if op else sorted(audit["ops"].keys())
    for o in ops:
        if o not in audit["ops"]:
            print(f"{o}: (not in audit — run `prepare {o}` first)")
            continue
        entry = audit["ops"][o]
        passed = entry.get("samples_passed", 0)
        failed = entry.get("samples_failed", 0)
        attempted = entry.get("samples_attempted", 0)
        ratio = f"{passed}/{attempted}" if attempted else "0/0"
        print(f"{o:30s} {entry['status']:25s} {ratio:>10s}  "
              f"disc_rounds={entry.get('discriminator_rounds', 0)}")
    return 0


def record_sweep(op: str) -> int:
    audit = load_audit()
    runner = ROOT.parent.parent / "target" / "release" / "parity-sweep"
    if not runner.is_file():
        print("building runner (release) …", file=sys.stderr)
        subprocess.run(
            ["cargo", "build", "--release", "-p", "parity-sweep-runner"],
            cwd=ROOT.parent.parent, check=True,
        )
    op_dir = RUNS_DIR / op
    op_dir.mkdir(parents=True, exist_ok=True)
    out = op_dir / "last_sweep.txt"
    print(f"sweeping {op} → {out.relative_to(Path.cwd())} …")
    proc = subprocess.run(
        [str(runner), "sweep", "--op", op, "--seeds", "8"],
        capture_output=True, text=True,
    )
    out.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    # Parse the human-readable report. The runner prints:
    #   [op] N/M passed (K skipped, F failed)
    #   FAIL: ...
    passed = attempted = failed = skipped = 0
    fails: list[str] = []
    for line in proc.stdout.splitlines():
        if line.startswith(f"[{op}]") and "passed" in line:
            # Runner prints: "[op] X/Y passed (Z skipped, F failed)"
            try:
                _, tail = line.split("] ", 1)
                left, _ = tail.split(" passed", 1)
                passed_s, attempted_s = left.split("/")
                passed = int(passed_s); attempted = int(attempted_s)
                paren = tail[tail.index("(")+1 : tail.rindex(")")]
                for piece in paren.split(", "):
                    n_s, label = piece.split(" ", 1)
                    n = int(n_s)
                    if label == "skipped": skipped = n
                    elif label == "failed": failed = n
            except (ValueError, IndexError) as e:
                # Runner output format changed — surface loudly rather than swallow.
                print(f"WARN: failed to parse runner summary {line!r}: {e}", file=sys.stderr)
        elif line.lstrip().startswith("FAIL:"):
            fails.append(line.lstrip()[5:].strip())

    entry = audit["ops"].setdefault(op, {
        "status": "missing", "pytorch_source": None, "ferrotorch_source": None,
        "known_divergences": [], "discriminator_rounds": 0,
    })
    from datetime import datetime, timezone
    entry["last_sweep_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry["samples_attempted"] = attempted
    entry["samples_passed"] = passed
    entry["samples_failed"] = failed
    entry["samples_skipped"] = skipped
    entry["status"] = "verified" if (failed == 0 and attempted > 0) else "diverges"
    (op_dir / "divergences.json").write_text(json.dumps(fails, indent=2) + "\n")
    save_audit(audit)
    print(f"  attempted={attempted} passed={passed} failed={failed} skipped={skipped}")
    print(f"  status → {entry['status']}")
    return 0 if failed == 0 else 1


def usage() -> int:
    print(__doc__)
    return 2


def main() -> int:
    args = sys.argv[1:]
    if not args:
        return usage()
    cmd = args[0]
    if cmd == "prepare" and len(args) == 2:
        return prepare(args[1])
    if cmd == "status":
        return status(args[1] if len(args) > 1 else None)
    if cmd == "record-sweep" and len(args) == 2:
        return record_sweep(args[1])
    return usage()


if __name__ == "__main__":
    sys.exit(main())
