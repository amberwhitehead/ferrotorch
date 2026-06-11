#!/usr/bin/env python3
"""Refresh the committed tracking-issue liveness snapshot for the
ferrotorch-core conformance surface-coverage gate (CORE-195 / #1889).

The exclusions file
``ferrotorch-core/tests/conformance/_surface_exclusions.toml`` marks every
entry as either ``kind = "permanent"`` (gate-limitation annotation; coverage
exists elsewhere) or ``kind = "deferred"`` (conformance coverage genuinely not
yet authored). Each *deferred* entry must reference an OPEN crosslink
tracking issue — but ``.crosslink/issues.db`` is sqlite and gitignored, so CI
cannot query issue state directly. This script bridges the gap: it queries
``crosslink issue show <n> -q --json`` for every tracking issue referenced by
a deferred entry and writes a committed snapshot,
``ferrotorch-core/tests/conformance/_tracking_issue_status.json``::

    {"generated_at": "YYYY-MM-DD", "issues": {"1913": "open", ...}}

The gate test ``exclusion_tracking_issues_are_live`` (in
``ferrotorch-core/tests/conformance_surface_coverage.rs``) consumes the
snapshot and fails when any deferred entry's issue is not "open", when the
snapshot is missing, or when ``generated_at`` is 45+ days old. Run this
script locally (where crosslink is available) and commit the result.

Exits non-zero if crosslink is unavailable, an issue cannot be queried, or a
deferred entry carries a malformed ``tracking_issue``.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUSIONS = (
    REPO_ROOT / "ferrotorch-core" / "tests" / "conformance" / "_surface_exclusions.toml"
)
SNAPSHOT = (
    REPO_ROOT
    / "ferrotorch-core"
    / "tests"
    / "conformance"
    / "_tracking_issue_status.json"
)


def main() -> int:
    with EXCLUSIONS.open("rb") as f:
        exclusions = tomllib.load(f)["exclusion"]

    numbers: set[int] = set()
    errors: list[str] = []
    for entry in exclusions:
        if entry.get("kind") != "deferred":
            continue
        ref = entry.get("tracking_issue", "")
        if not (ref.startswith("#") and ref[1:].isdigit()):
            errors.append(
                f"{entry.get('path', '<missing path>')}: deferred entry has "
                f"malformed tracking_issue {ref!r} (expected '#NNN')"
            )
            continue
        numbers.add(int(ref[1:]))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    statuses: dict[str, str] = {}
    for n in sorted(numbers):
        proc = subprocess.run(
            ["crosslink", "issue", "show", str(n), "-q", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            print(
                f"crosslink issue show {n} failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()}",
                file=sys.stderr,
            )
            return 1
        payload = json.loads(proc.stdout)
        status = payload.get("status")
        if not isinstance(status, str) or not status:
            print(f"issue #{n}: missing/invalid 'status' in JSON", file=sys.stderr)
            return 1
        statuses[str(n)] = status

    snapshot = {
        "generated_at": datetime.date.today().isoformat(),
        "issues": statuses,
    }
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    closed = sorted(k for k, v in statuses.items() if v != "open")
    print(f"wrote {SNAPSHOT.relative_to(REPO_ROOT)}: {len(statuses)} issue(s)")
    if closed:
        print(
            f"WARNING: non-open tracking issue(s) referenced by deferred "
            f"exclusions: {', '.join('#' + k for k in closed)} — the gate "
            f"will fail until they are re-pointed or the coverage is authored."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
