# ferrotorch-core Audit Remediation — Locked /goal Statement

This file is the binding contract for the remediation of the 207 findings in
`audit.md` (audited revision `24f587d94`, crate `ferrotorch-core` 0.6.2),
tracked as crosslink issues **#1695–#1901** (mapping: issue number =
1694 + CORE number; labels `audit` + `ferrotorch-core`). When the user issues
`/goal $(cat goal-audit-fix.md)` (or otherwise references this file), the
contents below are in force until every in-scope issue is closed or the user
rewrites this file.

`goal.md` remains in force. Its anti-drift rules (R-CITE-*, R-HONEST-*,
R-CODE-*, R-DEFER-*, R-GIT-*, R-VERIFY-*, R-INJECT-*, agent rules) all apply
to this series. Where this file states a rule for the remediation series that
differs in process detail (e.g. fix batching), this file governs.

---

## Why this contract exists

The 207 findings are not random bugs. They are the residue of one failure
mode: **the model optimizes whatever signal closes the loop.** Where the
closing signal was ground truth, the code held; where it was a proxy, the
code silently converged to satisfying the proxy:

- Fixtures generated from live PyTorch caught divergence. Fixtures generated
  from a Python mirror of ferrotorch's own algorithm (CORE-194) blessed every
  divergence as "bit-exact parity."
- GPU tests that assert `is_cuda()` caught fallbacks. Device-transparent
  readback (CORE-196) green-lit CPU execution of "GPU" tests indefinitely.
- A coverage gate that checked "tracking issue exists" was satisfied by 331
  closed issues (CORE-195); a gate that checked "name appears in file" was
  satisfied by comments (CORE-202).
- CI that never ran the tests (CORE-191/192/017) made every green check
  vacuous.

Weeks of instruction-giving did not fix this, because instructions decay over
a long context while incentives persist. **Gates scale; instructions don't.**
This contract therefore does two things: it repairs the gates FIRST, and it
defines "done" per finding in a way that cannot be satisfied by a plausible
happy path. Its companion principle:

> **An honest error is a success. A plausible value is the failure mode.**
> If an op cannot perform the requested case correctly, returning a
> structured `Err` is correct behavior, closes the issue, and is rewarded.
> Fabricated completeness is the only outcome this contract treats as
> failure.

---

## The goal

Close every open crosslink issue labeled `audit` + `ferrotorch-core`
(#1695–#1901), in phase order, where "closed" means the Definition of Done
for the finding's class (below) is met and evidenced. Mechanically:

```bash
# In-scope open issues (target: 0)
crosslink issue list -s open -l ferrotorch-core -q | wc -l

# CI actually gates the surfaces it claims to gate (Phase 0 exit criteria).
# NOTE: the cargo invocations are multi-line YAML (`cargo test \` …), so the
# checks join continuation lines before matching — a naive line-based
# `grep -- "--tests"` either passes via a comment or can never match the
# invocation. Both failure modes were observed while authoring this file.
tr -d '\\\n' < .github/workflows/linux-ci.yml \
  | grep -qE 'cargo test +-p ferrotorch-core +--release +--lib +--tests' \
                                                                # CORE-193/017
cat .github/workflows/*.yml | tr -d '\\\n' \
  | grep -qE 'cargo (test|clippy) +[^#]*-p ferrotorch-core +[^#]*--features[= ]"?gpu' \
                                                                # CORE-191
grep -q "matching ferrotorch" \
  scripts/regenerate_{quantize_prune,nested_sparse,masked}_fixtures.py \
  && exit 1 || true                                             # CORE-194
```

---

## Finding classes and Definition of Done

Every issue is one of four classes. The class determines what closes it.
State the class in the `--kind plan` comment before starting work.

- **CLASS-V — divergent value or gradient** (e.g. CORE-144 lu permutation,
  CORE-161 einsum lone indices, CORE-186 matmul backward, CORE-133/134
  inf-NaN poisoning, CORE-169 gammainc, CORE-178 pow backward).
  **DoD:** ferrotorch matches the PyTorch oracle on the divergent inputs.
  Closes ONLY with a correct implementation. An error return does NOT close
  a CLASS-V issue — PyTorch computes these; so must we.

- **CLASS-S — silent contract violation** (silent autograd detach, silent
  CPU demotion, silent truncation, silently accepted invalid input — e.g.
  CORE-146, CORE-170, CORE-141, CORE-131).
  **DoD:** the silence is eliminated. EITHER (a) correct implementation, OR
  (b) an explicit structured error at the public boundary
  (`NotImplementedOnCuda`, `UnsupportedAutograd`, `InvalidArgument` …) plus a
  concretely scoped follow-up feature issue filed and cross-linked. Path (b)
  is a full, successful close — the bug was the silence, not the gap.

- **CLASS-U — unsoundness or panic-in-fallible-API** (CORE-001, CORE-100,
  CORE-111, CORE-125, CORE-138; all "panics inside a `Result` API" and
  overflow-bypass findings).
  **DoD:** UB is unreachable from safe code (validate, or demote to
  `unsafe fn` with documented preconditions and no safe public caller), and
  fallible APIs return `Err` instead of panicking on every input named in
  the finding. Adversarial inputs from the finding become permanent tests.

- **CLASS-T — test/CI/fixture infrastructure** (CORE-191–207, CORE-017).
  **DoD:** the gate gates. Subject to R-RED-2 (gate proof) below.

## R-RED — "show it red" (the core anti-happy-path mechanism)

- **R-RED-1**: Every issue closure carries at least one regression test that
  was OBSERVED FAILING against pre-fix code, with the failing output pasted
  into the issue's `--kind result` comment and the commit body. A test never
  seen red proves nothing — green-from-birth is how CORE-194/196/204
  happened. (Critic writes the red test; fixer makes it green; per
  R-ACTOR-*/R-FIX-* in goal.md.)
- **R-RED-2 (gate proof)**: Every repaired or new gate (CI step, coverage
  check, fixture regeneration, device assertion helper) must be demonstrated
  to FAIL on a synthetic violation before it is trusted: introduce a
  deliberate violation in a scratch branch or temporary commit, paste the
  red output, revert. A gate never seen red is assumed vacuous.
- **R-RED-3**: One root cause per fixer dispatch — but a single root-cause
  fix MAY close multiple CORE issues when they share the mechanism (e.g. a
  `unary_map` contiguity fallback discharges most of CORE-132's call sites).
  EVERY issue it closes still needs its own red-then-green regression test.
  This amends goal.md R-FIX-1 ("one divergence per dispatch") to "one
  MECHANISM per dispatch" for this series only.

## R-ORACLE — where expected values come from

- **R-ORACLE-1**: Every numerical expectation in a test traces to (a) the
  live PyTorch oracle (parity-sweep `oracle.py`), or (b) a value pasted from
  a live torch session with the snippet quoted in a comment, or (c) an
  upstream `pytorch/<file>:<line>` cite for contract semantics. NEVER to a
  helper that re-implements the algorithm under test (R-CHAR-3 generalized).
- **R-ORACLE-2**: Fixture regeneration scripts compute expectations from
  torch APIs ONLY. Any function documented as "matching ferrotorch's X" is
  forbidden in `scripts/regenerate_*` and is itself a finding.
- **R-ORACLE-3**: Every GPU test asserts result device (and gradient device
  where gradients exist). Every autograd test asserts gradient FLOW — values
  reaching the original leaf — never `requires_grad` flags alone.
- **R-ORACLE-4**: No dual-accepting assertions ("Err OR sentinel passes").
  Pin exactly one contract; if it diverges from torch, the test carries the
  tracking-issue number and the torch-side expected value in a comment.
- **R-ORACLE-5**: Tolerances require an analytic justification comment
  (dtype epsilon, accumulation length, documented upstream drift). Bare
  floors like `.max(0.5)` are forbidden.

## R-LOUD — the no-silent-fallback policy (crate-wide, permanent)

- **R-LOUD-1**: An operation that cannot perform the requested case
  correctly on the requested device/dtype/layout returns a structured `Err`.
  Never a plausible value, never a silent detach, never a silent device
  change, never a silent truncation. This is the crate's contract going
  forward; the audit findings are its existing violations.
- **R-LOUD-2**: A fallback that changes WHERE compute happens but not the
  result (host round trip) must be explicit in the function's doc-comment
  and consistent within its module. Undocumented round trips are findings
  (CORE-177's three-way inconsistency is the cautionary example).
- **R-LOUD-3**: `requires_grad` is never copied as a bare flag onto a fresh
  leaf. Either a real backward edge is attached or the output is honestly
  `requires_grad = false` (with an error per R-LOUD-1 if the input tracked).

## R-AHON — honesty rules for this series

- **R-AHON-1**: Probe before fixing. Re-verify the finding at HEAD first; if
  it is stale or wrong, close it as not-a-bug with the probe output. That
  closure is a success, not a failure.
- **R-AHON-2**: Every fix report pastes raw command output: the red test
  before, the green gauntlet after, integer counts. Narrative-only
  verification ("tests pass") is rejected and the dispatch is redone.
- **R-AHON-3**: Choosing CLASS-S path (b) — error + follow-up — must be
  stated plainly in the result comment: "implemented the error boundary, NOT
  the feature; feature tracked in #NNN." Dressing path (b) up as path (a)
  is the one unforgivable move under this contract.
- **R-AHON-4**: Regressions or new findings discovered mid-fix get their own
  issues immediately (`-l audit -l ferrotorch-core`); they are never
  silently bundled into the current fix or quietly left behind.
- **R-AHON-5**: A whole-suite run that newly fails on something unrelated to
  your change is reported, not re-scoped around. (Whole-crate runs surface
  orthogonal bugs; that is a feature of the gauntlet, not noise.)

---

## Phase order

Work strictly in phase order. A later-phase fix may not land while an
earlier phase has open issues, with one exception: a Phase-1+ fix may land
during Phase 0 if its red test runs under the already-working `--lib` lane.

**Phase 0 — Trust the verifier (nothing else lands until these close):**
CORE-193 → #1887 (two doc-tests blocking `--tests`; flip linux-ci),
CORE-191 → #1885 (gpu feature lanes in CI + clippy),
CORE-192 → #1886 (nightly has never run),
CORE-194 → #1888 (regenerate quantize/prune/2:4 fixtures from real torch),
CORE-195 → #1889 (coverage gate rejects closed tracking issues),
CORE-196 → #1890 (device assertions in the five blind GPU suites),
CORE-202 → #1896 (coverage gate ignores comments),
CORE-206 → #1900 (parity oracle hard-fails when torch absent; nightly smoke),
plus CORE-197/198/199/200/201/203/204/205/207 (#1891–#1899, #1901) as the
test-quality tail. Rationale: fixing findings under broken gates reproduces
the disease — Phase 1+ fixes would be validated by the machinery that missed
them.

**Phase 1 — Unsoundness (CLASS-U criticals):**
CORE-001 → #1695, CORE-100 → #1794, CORE-111 → #1805, CORE-125 → #1819,
CORE-138 → #1832.

**Phase 2 — Silent wrong values (CLASS-V Highs):** batched by mechanism
(R-RED-3). Representative batches: einsum CPU/backward family
(CORE-161–165), lu convention (CORE-144), matmul backward + accumulation
(CORE-186, 140, 139), inf/NaN reduction family (CORE-133, 134, 135),
gradient-formula family (CORE-178, 180, 181), special-function numerics
(CORE-169, 172, 173, 174), view-geometry CUDA family (CORE-151, 055-class
remnants), indexing/scatter semantics (CORE-112, 125–130 remnants).

**Phase 3 — Silent contract violations (CLASS-S Highs):** the detach and
device-demotion families (CORE-146, 170, 109/110/114-class, 141, 166, 168),
each closed via R-LOUD path (a) or (b).

**Phase 4 — Mediums**, batched by file/mechanism, same rules.

Within a phase: severity, then shared-mechanism batches, then issue order.
Do not ask which item is next — this ordering is the answer (R-LOOP-1).

---

## The remediation loop (per dispatch)

1. **Probe** — re-verify the finding at HEAD (R-AHON-1). Paste the probe.
2. **Classify** — state CLASS-V/S/U/T and, per `rust-fix-discipline`, the
   fix category (root-cause vs local correctness; defer is forbidden).
   Post as `--kind plan` on every issue the dispatch will close.
3. **Red** — critic-side: write the regression test(s); observe and paste
   the failure (R-RED-1). For CLASS-S path (b), the red test asserts the
   structured error.
4. **Fix** — fixer/builder-side, per goal.md agent rules, one mechanism.
5. **Gauntlet** —
   ```bash
   cargo test -p ferrotorch-core --lib --tests       # post-#1887; until then --lib + the touched suites by name
   cargo clippy -p ferrotorch-core --all-targets -- -D warnings
   cargo fmt --all --check
   # parity smoke for any op with a runner arm (R-DEFER-6 quantification)
   # GPU-feature lanes when the fix touches device paths:
   cargo test -p ferrotorch-core --features gpu --tests   # on the CUDA runner / locally if available
   ```
6. **Re-audit** — acto-critic on every touched file (R-FIX-4/R-BUILD-4).
7. **Close** — `--kind result` comment per issue: class, red output, green
   output, follow-up issue links. Then `crosslink issue close <N>`.

---

## Out of scope for this goal

- Findings in other crates discovered along the way: file them
  (`-l audit -l <crate>`), do not fix them here (except where a
  ferrotorch-core fix's mechanism lives in a backend trait it owns the
  contract for — then the contract side is in scope, the backend side gets
  an issue).
- Performance work beyond what a correct fix requires.
- New features beyond CLASS-S follow-up scaffolding (the follow-ups are the
  next goal, not this one).

## Stopping condition

```bash
test "$(crosslink issue list -s open -l ferrotorch-core -q | wc -l)" -eq 0
```

— AND linux-ci runs `--tests` green, AND a CI lane compiles-and-runs the
`gpu`-feature tests, AND no fixture script consumed by ferrotorch-core
conformance suites computes an expectation from a ferrotorch mirror (the
three Phase-0 greps above all pass). Then post a final summary on the master tracking
issue (#1694), update `audit.md`'s status line to "Remediated at <rev>", and
stop. Until then: every turn, one iteration of the remediation loop. No
exceptions.
