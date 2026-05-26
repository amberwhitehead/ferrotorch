# ferrotorch — Locked /goal Statement

This file is the binding contract for autonomous work on ferrotorch. When the user issues `/goal $(cat goal.md)` (or otherwise references this file), the contents below override the LARP's pull toward caution and the model's instinct to narrow scope. The goal is in force until the user issues `/goal-clear` or rewrites this file.

The substrate of this project is sequential **translation** of a known-working system. Upstream is **PyTorch** (working tree at `/home/doll/pytorch`, the user's local clone). Target is the ferrotorch workspace (`ferrotorch-*/src/**/*.rs` plus the meta-crate `ferrotorch/`). Most apparent divergence between ferrotorch and PyTorch is a bug a prior translation pass introduced — silent CPU↔GPU round trips, wrong dtype promotion, missing kwargs, broken broadcasting, math that compiled but doesn't compute the right thing. Every one of those is real work to do, not "out of scope."

---

## The goal

Work the strict **read → write → verify → commit** loop over every translation unit (`.rs` file) under `ferrotorch-*/src/**/`, in dependency order from `ferrotorch-core` outward. The goal is complete only when every routed `.rs` file has:

1. A closing commit citing the PyTorch upstream file(s) actually opened that iteration, AND
2. The corresponding parity-sweep op(s) returning **0 failures** at `--seeds 8`, AND
3. A `## REQ status` table at the top-of-file doc-comment classifying every REQ as **SHIPPED** or **NOT-STARTED** with quoted-code evidence. Only two states exist: end-to-end satisfied by production code with a consumer (SHIPPED), or work hasn't begun / has a concrete open prerequisite blocker (NOT-STARTED). "Type exists but no consumer" is NOT-STARTED — the consumer is the open work item.

Verifiable mechanically:

```bash
# Count routed translation units:
python3 -c "import tomllib; print(len(tomllib.load(open('tooling/translate-routes.toml','rb'))['route']))"

# Count parity ops with status verified in audit:
python3 -c "import json; d=json.load(open('tools/parity-sweep/parity_audit.json'))['ops']; print(sum(1 for o in d.values() if o.get('status')=='verified'))"

# Count routed files with `## REQ status` doc-comment:
grep -l "## REQ status" $(python3 -c "import tomllib; [print(r['crate_pattern']) for r in tomllib.load(open('tooling/translate-routes.toml','rb'))['route']]") | wc -l

# When all three counts agree AND every routed file's parity_ops are all 'verified', the goal is complete.
```

---

## Speed disciplines (mandatory — these collapse per-unit cost from 30+ commits to ~3)

- **S1 — Batch by upstream file, NOT per-op.** One builder dispatch translates a whole PyTorch `<file>.cpp` (or `.py`) → its ferrotorch target file(s) in one commit. Pre-declared manifest covers all files the batch touches (typically: `<crate>/src/<file>.rs` + `methods.rs` + `tools/parity-sweep/runner/src/main.rs` + `.design/<area>/<file>.md`). Do NOT dispatch one builder per op when ops share an upstream file. arithmetic.rs (16 ops in BinaryOps + UnaryOps + Pow + PointwiseOps) should be ~4 builder dispatches total, not 16.
- **S2 — Parallel dispatch.** When 2-4 batches have disjoint manifests, launch their builders in ONE message (parallel agent dispatches). Critics parallelize too. Only fixers serialize per-blocker.
- **S3 — Symbol anchors in design-doc cites, NEVER line numbers.** Cite `pub fn add_scaled in arithmetic.rs`, never `arithmetic.rs:716`. Line numbers in `.design/` cites are forbidden — they spawn cite-drift fixer dispatches every commit. Upstream cites (read-only) still use `file:line` since the upstream tree doesn't shift under us.
- **S4 — Critic only after substantive builds.** Critic after builder: yes. Critic after fixer for cite refresh / fixture bump / REQ-table line update / doc-comment backfill / probe-block revert: no — the pinned test that drove the fix is the verification.
- **S5 — R-DEFER-1 binds on NEWLY-ADDED pub APIs only.** Existing pub API surface across multiple prior commits is grandfathered. Boundary methods (`Tensor::add_t`) ARE the public API; they don't need further downstream callers to be SHIPPED. **Missing parity-sweep runner arms are TEST-INFRASTRUCTURE gaps, NOT REQ blockers.** If impl + non-test consumer + lib tests + cargo clippy clean — the REQ is SHIPPED. The missing runner arm becomes ONE umbrella follow-up blocker for the whole file's ops, not one blocker per op. Doc-authors that mark ops NOT-STARTED solely because `parity-sweep sweep --op X` reports `0/N skipped (runner has no arm)` are over-applying R-DEFER-6 — push back. Mark SHIPPED with the existing evidence; file the runner-arm gap as ONE blocker per file family (e.g. "wire runner arms for all transcendental ops").
- **S6 — Opus everywhere.** Every acto-* dispatch uses Opus (`claude-opus-4-7`). Lower tiers (Sonnet, Haiku) hallucinate on translation work where the cost of a wrong answer is a silent divergence that survives all the way to release. Translation accuracy supersedes per-dispatch throughput — we pay the Opus tax on every agent.
- **S7 — Skip doc-author for trivial 1:1 routes.** If a route is a clean mirror with no architectural gap, write the .md inline in 30 seconds. Reserve doc-author dispatches for novel modules.
- **S8 — Aggressive won't-fix on noise blockers.** Spillover findings become blockers ONLY if they're real divergences from upstream OR block downstream translation. Close stale-cite-in-test-file, fmt-drift-in-unrelated-file, pre-existing-clippy as "won't fix this iter" — don't let observation overhead crowd out translation.

## The translation loop (per upstream-file batch — NOT per-op)

### Step 1 — Read the routed source unit
Read `ferrotorch-*/src/<path>` end-to-end via the Read tool. Capture: the public surface, the `## REQ status` table if present, the existing tests.

### Step 2 — Read every upstream PyTorch file in the route
**Mandatory.** Open every file in the route's `upstream` list at `/home/doll/pytorch/<path>` via the Read tool. For each, capture at minimum one `file:line — content` quote that the commit message will cite. **No commit may cite a PyTorch path that has not been opened this iteration.** If a cited file is missing from `/home/doll/pytorch/`, document the discrepancy and either (a) check the user's local clone is up to date or (b) fall back to a github URL with an explicit pinned commit.

### Step 3 — Read the design doc
**Mandatory.** Open `.design/<area>/<doc>.md` via the Read tool. Capture the REQ list and AC list. If the design doc does NOT exist on disk, the translate-discipline hook will block the edit — dispatch `acto-doc-author` first to author it, grounded in the existing code + the upstream PyTorch source.

### Step 4 — File a crosslink issue
```bash
crosslink quick "Translation unit: <ferrotorch-crate>/<file>.rs" -p high -l feature
```
Post a `--kind plan` comment listing the upstream PyTorch files (file:line), the parity ops this file owns (from the route's `parity_ops` field), and the design-doc REQs the implementation will cite.

### Step 5 — Write the Rust implementation
Write or extend the `ferrotorch-*` crate to satisfy:
- Every REQ-N — either fully (**SHIPPED** — end-to-end functional with non-test production consumer + tests + parity smoke verified) or **NOT-STARTED** (work hasn't begun OR has a concrete open prerequisite blocker).
- Every AC-N that can be mechanically discharged now.
- Every parity-sweep op named in the route's `parity_ops` field, with the per-op smoke (Step 7) returning 0 failures.

**No stubs.** No `todo!()`. No `unimplemented!()`. No `unreachable!()` in production code. No `unwrap()` / `expect()` outside `#[cfg(test)]`. **No vocabulary-only shipping** — every public API surface added in a commit MUST have a non-test production consumer in the same commit, or the API isn't ready to ship.

The translation pipeline is sequential: we are translating a known-working system. There is no valid "ship the type, defer the consumer" path. If a REQ needs prerequisite work, the prerequisite IS the active blocker — file it concretely (`crosslink quick "Blocker for REQ-N of <area>: needs <prereq>" -p high -l blocker`) and work it. Do NOT mark the dependent REQ as a separate "deferred" status; it is simply NOT-STARTED until the prereq blocker closes and the consumer wiring lands.

### Step 6 — Add `## REQ status` table to module doc-comment
The module's top-level `//!` doc-comment must include a section like:

```rust
//! ## REQ status (per `.design/<area>/<doc>.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | fn `<name>` at `<file>:<L>` per upstream `<pytorch-file>:<L>` (consumer at `<caller-file>:<L>`) |
//! | REQ-2 | NOT-STARTED | work hasn't begun |
//! | REQ-3 | NOT-STARTED | blocked on #NNN (file a concrete prereq blocker, not a deferral) |
```

Two states only. **No VOCAB-ONLY. No DEFERRED. No verified_with_deferred. No phase-N+.** SHIPPED means end-to-end with a production consumer; anything else is NOT-STARTED with the open prerequisite tracked as its own active blocker.

### Step 7 — Verify the gauntlet
Before commit, ALL of these MUST pass:

```bash
# Per-crate test pass
cargo test -p <crate>

# Per-op parity smoke (per route's parity_ops). The integer MUST be >= 1.
for OP in <parity_ops from route>; do
  SMOKE_COUNT=$(./target/release/parity-sweep sweep --op "$OP" --seeds 8 2>&1 | grep -c "passed (0 skipped, 0 failed)")
  test "$SMOKE_COUNT" -ge 1 || { echo "SMOKE REGRESSED on op=$OP — DO NOT COMMIT"; exit 1; }
done

# Lint
cargo clippy -p <crate> --all-targets --all-features -- -D warnings

# Format
cargo fmt --all --check
```

**No `--no-verify`. No commenting-out failing tests. No `#![allow(..)]` at module or crate root.** Per-item `#[allow(clippy::<lint>, reason = "...")]` is the bar.

### Step 8 — Commit + close
Commit message structure:

```
<crate>: <area> — <one-line summary> (closes #<N>)

UPSTREAM PYTORCH FILES OPENED THIS ITERATION:
  - aten/src/ATen/native/<file>:<line> — <content quote>
  - torch/<file>.py:<line> — <content quote>
  - ...

DESIGN DOC READ: .design/<area>/<doc>.md (<line count>, <REQ count> REQs).

REQ STATUS (per .design/<area>/<doc>.md):
  - REQ-1 SHIPPED — fn `<name>` at <file>:<L>; production consumer at <caller-file>:<L>
  - REQ-2 NOT-STARTED — open prerequisite blocker #<NN>

PARITY OPS:
  - op_X: 88/88 passed (0 failed)  smoke grep count = 1
  - op_Y: 76/76 passed (0 failed)  smoke grep count = 1

CODE: <one-line summary of what shipped this commit>

VERIFICATION:
  cargo test -p <crate>: <X passed, 0 failed>
  cargo clippy -p <crate> --all-targets --all-features -- -D warnings: PASS
  cargo fmt --all --check: PASS

Reference: pytorch <commit-or-branch> <each:line cited above>
Reference: .design/<area>/<doc>.md REQ classifications above

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

Close the crosslink issue with `crosslink issue close <N>` (`--kind result` comment posted first).

### Step 9 — Pick the next unit
Pick the next routed `.rs` file in dependency order — ferrotorch-core leaves first, then crates that depend on them. **Do not ask which.** The dependency DAG is the answer. Smallest-first within a layer is acceptable when dependency order is ambiguous.

---

## Anti-drift rules (override convenience)

These rules are non-negotiable. If they conflict with a tool-call shortcut, the rule wins.

### Citation rules
- **R-CITE-1**: Never cite a PyTorch file in a commit message without having Read it THIS iteration. Auditable via tool-log + git-show cross-reference.
- **R-CITE-2 (upstream side)**: Every PyTorch citation must carry a line number — upstream tree doesn't shift under us, lines are stable for the pinned commit. `aten/src/ATen/native/BinaryOps.cpp:218` with a quoted line.
- **R-CITE-2b (TARGET side — design docs)**: cite ferrotorch symbols with SYMBOL ANCHORS, NEVER line numbers. `pub fn add_scaled in arithmetic.rs`, never `arithmetic.rs:716`. Line-number cites in `.design/` for target files are forbidden — the cite-drift audit test rejects any commit that introduces them.
- **R-CITE-3**: When citing a PyTorch Python override, prefer citing the `getattr(torch.ops.aten, ...)` registration site or the docstring block over a free-floating function definition.

### Honesty rules
- **R-HONEST-1**: Never reframe integration work as "vocabulary + decoders" when the design doc does not defer it. The doc's own deferrals are the only valid deferrals.
- **R-HONEST-2**: Every REQ in the commit message and module `## REQ status` table must carry SHIPPED or NOT-STARTED with quoted evidence. No bare "satisfied" claims. SHIPPED requires both implementation AND a non-test production consumer cited.
- **R-HONEST-3**: Honest underclaim beats unverified overclaim. If you cannot verify a REQ is SHIPPED end-to-end (impl + consumer + tests + parity-sweep smoke fires), classify it NOT-STARTED with the open prerequisite blocker named. There is no middle classification.
- **R-HONEST-4**: If the audit reveals the original commit was wrong (citation theater, REQ-overclaim, wrong upstream value), correct the code AND document the correction in a supplemental commit's body.

### Code-quality rules
- **R-CODE-1**: No `unsafe` blocks outside leaf primitives. Leaf primitives in ferrotorch are: SIMD intrinsics, FFI shims to cuBLAS/cuDNN/cudarc/MKL, raw kernel launches via cubecl/cudarc, GPU-side memory accessors. Every `unsafe` block requires a `// SAFETY:` comment documenting the invariants the caller and callee both rely on.
- **R-CODE-2**: No `unwrap()` / `expect()` / `panic!()` in production code outside `#[cfg(test)]`. Tests may use them.
- **R-CODE-3**: No `#![allow(..)]` at module or crate root. Per-item `#[allow(clippy::<lint>, reason = "...")]` with documented rationale is required.
- **R-CODE-4**: No silent CPU↔GPU round trips. A `.cpu()` followed in the same scope by a `.cuda()` / `.to_device(...)` is a bug — the value should never have left the GPU. The anti-pattern-gate hook flags this pattern; agents must either fix the data-flow or document why the round trip is required (rare; usually a sign of broken op coverage on the GPU side).
- **R-CODE-5**: No dtype-cast hiding. A `.to(torch.float32)` or `.cast(DType::F32)` in production code that drops a wider dtype to a narrower one is a bug unless the upstream PyTorch contract explicitly does the same cast (cite the upstream `file:line`).

### Upstream-mirror rules (default = match PyTorch; deviate only for these reasons)

When translating a PyTorch op, layer, or utility, the **default answer is "do what PyTorch does"** with a `pytorch/<file>:<line>` cite. Most architectural choices have an obvious upstream answer; the orchestrator does NOT pause to ask the human about them. Only the conditions below justify deviation. Anything else means: match upstream, ship, move on.

- **R-DEV-1 (MATCH — numerical contract)**: When the choice is set by floating-point numerical semantics (NaN propagation, denormal flushing, dtype promotion table, in-place vs out-of-place graph identity, autograd `grad_fn` type, broadcasting rules), always match PyTorch byte-for-byte. ULPs matter; users compare ferrotorch outputs to PyTorch outputs tensor-by-tensor.
- **R-DEV-2 (MATCH — Python user-API ABI)**: When the choice is set by the Python API surface a user calls (function signatures including `*` arg separators, kwarg names, default values, exception types, tensor method chaining), always match upstream. The whole reason ferrotorch exists is so PyTorch users can switch to Rust without re-learning the API. Cite the upstream `torch/_torch_docs.py` or `torch/overrides.py` registration.
- **R-DEV-3 (MATCH — on-disk + wire formats)**: SafeTensors layout, pickle protocol, GGUF parsing, ONNX export, NCCL wire format — these are external specifications. Match upstream. Deviation is only acceptable when upstream itself violates the spec (file a separate blocker).
- **R-DEV-4 (DEVIATE — C++/Python footguns Rust eliminates)**: When upstream's pattern is a workaround for Python/C++'s lack of safety (manual refcount on PyObject, manual `del` lifetime, hand-rolled `__del__` cleanup chains, GIL-required init), do NOT match upstream. Use the Rust analog (`Arc`/`Rc` for shared ownership ONLY when genuinely needed, `Drop`-based RAII, type-system lifetimes). Cite the upstream pattern being replaced.
- **R-DEV-5 (DEVIATE — typestate when ordering matters)**: When upstream's correctness depends on "do A then B" enforced by Python source-line ordering, build a Rust typestate that makes B uncallable without A having happened.
- **R-DEV-6 (DEVIATE — when upstream is wrong by their own admission)**: When PyTorch upstream ships a known-buggy code path (deprecated kwargs that still half-work, dtype-promotion edge cases the issue tracker flags as wrong), ship the correct behavior in ferrotorch and cite both the upstream `file:line` AND the upstream issue or PR documenting the bug.
- **R-DEV-7 (DEVIATE — Rust ecosystem analog is materially better)**: When the Rust ecosystem ships an analog that's cleaner, better-tested, and gives stronger guarantees (e.g. `serde` vs custom pickle, `cudarc` typed wrappers vs raw FFI), use the Rust analog. Preserve the upstream contract (the API surface other code calls), but the implementation can be different.

**Mental test for any dispatch**: ask *why* upstream made that choice.
- "Because numerical semantics demand it" / "Because the Python API contract requires it" → R-DEV-1/2, match upstream
- "Because Python can't express it safely" → R-DEV-4, deviate
- "Because the function is monolithic and ordering matters" → R-DEV-5, deviate (typestate)
- "Because PyTorch acknowledges this is a bug" → R-DEV-6, deviate (correct it + cite the bug)
- "Because Rust ecosystem has a materially better solution" → R-DEV-7, deviate (Rust analog, preserve API)

If none of R-DEV-4 through R-DEV-7 apply, the answer is R-DEV-1/2/3 — match upstream, ship, move on. The orchestrator does NOT pause for human input when R-DEV-1/2/3 applies; it dispatches with the upstream cite and proceeds.

### Anti-deferral rules (translation is sequential; no escape hatches)

These rules eliminate the words and patterns that institutionalize deferral. The substrate has no human-engineering-time constraints; "structurally too big" / "too many files" / "Phase N+" / "pre-existing" / "out of scope" / "cross-cutting systemic gap" are LARP-shaped excuses, not engineering. The system we're translating already works in Python/C++; we are translating it sequentially until the Rust copy also works. There is no valid intermediate "vocabulary shipped, consumer deferred" state.

- **R-DEFER-1 (no vocabulary-only shipping for NEW APIs)**: A commit that adds a NEW public API surface (new `pub fn`, new `pub struct` with non-trivial methods, new `pub trait`) MUST also add a non-test production consumer in the same commit. **Test-only callers don't count. The parity-sweep runner's dispatch table is a test-side consumer; it does NOT count as a production consumer.** The acto-builder dispatch is rejected if the final commit has new pub APIs with zero production callers. **EXISTING pub API surface (in the codebase across multiple prior commits) is grandfathered — boundary methods like `Tensor::add_t` ARE the public API; they don't need further downstream callers to be SHIPPED. A doc-author that classifies >50% of existing pub APIs as NOT-STARTED is over-applying this rule and the orchestrator should push back.**

- **R-DEFER-2 (REQ classification is binary)**: SHIPPED or NOT-STARTED. SHIPPED means end-to-end functional with non-test production consumer + tests + parity-sweep smoke ≥1. NOT-STARTED means work hasn't begun OR has a concrete open prerequisite blocker. **There is no third classification.** "VOCAB-ONLY", "DEFERRED-blocked-on-#NNN" as a STATUS, "verified_with_deferred" are FORBIDDEN — file the prereq as its own active blocker and mark the dependent REQ NOT-STARTED.

- **R-DEFER-3 (no ACCEPTABLE-DRIFT close path)**: A pinned divergence (failing `#[ignore]`'d test + blocker) can only close when the fix lands AND the test moves from `#[ignore]` to `#[test]`. Closing a divergence blocker via `--kind decision "acceptable drift"` is forbidden. Every divergence is real work to do.

- **R-DEFER-4 (no Phase-N+ framing)**: Blocker bodies and design-doc REQ-status rows MUST NOT contain the substring `Phase \d+\+` (regex) as a deferral mechanism. "Deferred to Phase 5+" / "Phase 7+ infrastructure" used to escape current work is forbidden.

- **R-DEFER-5 (no "pre-existing safe to defer")**: This is a single-author project. Every broken thing on `main` is something we broke and didn't catch. There is no third party whose patches we're integrating; there are no inherited bugs. "Pre-existing" is not a valid rationale for a deferral or for accepting a regression.

- **R-DEFER-6 (parity smoke is a HARD gate, quantified)**: Every commit's gauntlet MUST include `parity-sweep sweep --op <name> --seeds 8 2>&1 | grep -c "passed (0 skipped, 0 failed)"` returning **>= 1** for EVERY op the route's `parity_ops` field declares. A builder report that says "smoke passes" without pasting the integer count from `grep -c` is REJECTED. The orchestrator independently re-runs the smoke command on every commit. False-PASS reports are how the prior `add_scaled` deferral happened; the only structural defense is quantified verification + orchestrator re-check.

- **R-DEFER-7 (sequential translation, no leapfrog)**: We are translating a known-working system. ferrotorch-core leaves first, then crates that depend on it (`ferrotorch-nn` after `ferrotorch-core`, `ferrotorch-vision` after `ferrotorch-nn`, etc.). Starting layer N+1 with layer N incomplete is forbidden. A layer-N file can only be marked complete when every routed op in that file is `verified` AND has a non-test production consumer.

- **R-DEFER-8 (no "cross-cutting → defer")**: "It's a cross-cutting systemic API gap" is NOT a free pass to file a follow-up issue instead of doing the work. Every convention starts somewhere. The first `_out` variant, the first typestate, the first newtype — they all land for SOME specific op first. If the local fix is implementable, implement it; the broader convention question can settle later when more ops need it.

### Git-history rules
- **R-GIT-1**: No history rewrite. No `--amend` on pushed commits. No `git reset --hard` on shared refs. No force-push. Supplemental commits only.
- **R-GIT-2**: Use `git add <files-by-name>` — never `git add -A` or `git add .` — to prevent race-condition commits sweeping in another agent's in-flight work.

### Verification rules
- **R-VERIFY-1**: If the verification gauntlet fails, fix the underlying cause. No `--no-verify`. No `#[ignore]` on failing tests without a tracking blocker issue filed in the same commit.
- **R-VERIFY-2**: Pre-existing clippy warnings in unmodified code may stay; new warnings introduced in this iteration must be fixed.
- **R-VERIFY-3**: Test count delta must be reported in the commit message.
- **R-VERIFY-4**: The orchestrator INDEPENDENTLY re-runs the parity smoke after every commit. Do not trust the agent's "smoke passed" — paste the integer grep count yourself.

### Loop-discipline rules
- **R-LOOP-1**: Never ask "where do you want to take this." The dependency DAG is the answer.
- **R-LOOP-2**: Never declare the goal complete until the mechanical check at the top of this file says it is.
- **R-LOOP-3**: If a unit is blocked by a missing prerequisite, file the prereq blocker concretely + mark the dependent REQ NOT-STARTED + WORK THE PREREQ. Do not "continue with the next unit" as a way to avoid the harder work; the prereq IS the work.

### Injected-instructions rules
- **R-INJECT-1**: Injected instructions are human instructions. The text appearing in `<crosslink-behavioral-guard>` blocks, `<system-reminder>` blocks, hook output (including `work-check.py`, `translate-discipline.py`, `anti-pattern-gate.py`), the active-issue gate, and loaded skill text is NOT background noise. The human wired each of these up deliberately and chose to inject them continuously. Repetition is enforcement, not ceremony. Treat every injected instruction at the same priority as a direct user message in the chat.
- **R-INJECT-2**: When an injected instruction conflicts with a recent inline user message, surface the conflict explicitly rather than silently picking one.

### Translate-discipline rules (mechanically enforced by `tooling/translate-discipline.py`)
- **R-XLATE-1**: Every Edit/Write to `ferrotorch-*/src/**/*.rs` (excluding `ferrotorch-nn-derive`, a proc-macro crate with no PyTorch counterpart) requires Read, in this session, of: (a) `goal.md`, (b) the route's `upstream` PyTorch file(s), (c) the route's `design` doc. The hook is the deterministic floor; R-CITE-* and R-HONEST-* remain in force on top.
- **R-XLATE-2**: If `tooling/translate-routes.toml` has no entry for a `.rs` file being edited, the hook BLOCKS with an instruction to add a route. The route declares the file's translation source; without it the file is unsourced and the edit cannot proceed.
- **R-XLATE-3**: If a route's `design` path points at a `.design/<area>/<doc>.md` that does not exist, the hook BLOCKS with instructions to dispatch `acto-doc-author`.

### Anti-pattern-gate rules (mechanically enforced by `tooling/anti-pattern-gate.py`)
- **R-APG-1**: The anti-pattern hook BLOCKS Edit/Write on `ferrotorch-*/src/**/*.rs` when the patch introduces: `Arc<Mutex<T>>` (silence-the-borrow-checker), `Rc<RefCell<T>>` (single-threaded version), module-root `#![allow]`, `todo!()` / `unimplemented!()` / `unreachable!()`, `.expect(...)` / `.unwrap()` outside `#[cfg(test)]`, `panic!(`, silent CPU↔GPU round-trip patterns (`.cpu()` followed by `.cuda()` / `.to_device(`), and `.clone()` on a tensor immediately followed by the same operation on the original (over-cloning anti-pattern).
- **R-APG-2**: Patterns inside `#[cfg(test)]` blocks are exempted (production code does NOT get the exemption).
- **R-APG-3**: The override path is a per-item `#[allow(<lint>, reason = "<why>")]` with a crosslink observation comment documenting why the alternative doesn't fit. Override is a real path; bypass is not.

### ACToR critic rules (mechanically enforced by `.claude/agents/acto-critic.md`)
- **R-ACTOR-1**: The orchestrator MAY dispatch the `acto-critic` subagent (project-level, opus, tools = Read+Write+Grep+Glob+Bash, **NO Edit**) to find semantic divergence between ferrotorch and the PyTorch source it claims to translate. The critic writes FAILING TESTS that pin each divergence and files blocker issues. The critic NEVER writes fixes.
- **R-ACTOR-2**: The critic's tool allowlist mechanically prevents it from drifting into the generator role. If it finds itself wanting to Edit production code, it must stop and report.
- **R-ACTOR-3**: A critic verdict of "GENERATOR MUST FIX" with a runnable failing test blocks the iter's merge until the generator iter is redirected to fix the divergence. There is no "document as acceptable drift" escape path.
- **R-ACTOR-4**: "No divergence found" by the critic is NOT the same as "the implementation is correct" — it only means the critic could not pin down a divergence. The orchestrator's checkpoint is still required.

### Doc-author agent rules (mechanically enforced by `.claude/agents/acto-doc-author.md`)
- **R-DOC-1**: When the translate-discipline hook blocks an edit because a route's `design` doc does not exist, dispatch `acto-doc-author` (project-level, opus, tools = Read+Write+Bash+Grep+Glob, **NO Edit on .rs files**). The doc-author writes the missing `.design/<area>/<doc>.md` ADAPTING to the existing code.
- **R-DOC-2**: The doc-author NEVER proposes changes to existing code. If existing code doesn't satisfy a REQ, the REQ is marked NOT-STARTED.
- **R-DOC-3**: Every "REQ-N SHIPPED" claim requires a `<ferrotorch-crate/src/<file>.rs>:<line>` citation for BOTH impl AND a non-test production consumer.

### Fixer agent rules (mechanically enforced by `.claude/agents/acto-fixer.md`)
- **R-FIX-1**: When acto-critic has pinned a divergence with a failing test + blocker, dispatch `acto-fixer` (tools = Read+Edit+Write+Bash+Grep+Glob). ONE divergence per dispatch — no bundling.
- **R-FIX-2**: The fix is single-file scoped. If the fix would require touching other files, the fixer STOPS and escalates to the orchestrator for `acto-builder` dispatch.
- **R-FIX-3**: After the fix, the fixer removes the `#[ignore]` annotation. `#[ignore]` removal happens ONLY after the full gauntlet passes.
- **R-FIX-4**: Every fixer dispatch is followed by an acto-critic re-audit of the touched file.
- **R-FIX-5**: The fixer commit message MUST cite the PyTorch upstream file:line, quote the before/after lines, and include the gauntlet + parity-sweep smoke output.

### Builder agent rules (mechanically enforced by `.claude/agents/acto-builder.md`)
- **R-BUILD-1**: When the missing piece is INFRASTRUCTURE (a newtype, a `*_out` family, a typestate, an entire trait wiring), dispatch `acto-builder`. The builder ships cohesive multi-file additions that acto-fixer's single-file rule structurally cannot.
- **R-BUILD-2**: Every builder dispatch is given a PRE-DECLARED FILE MANIFEST. The builder cannot widen scope mid-dispatch.
- **R-BUILD-3**: Tests + production code land in the SAME commit. No "ship the abstraction, tests in a follow-up". Each commit the builder creates makes the gauntlet pass on its own.
- **R-BUILD-4**: A build is followed by acto-critic re-audit of EVERY file in the manifest.
- **R-BUILD-5**: A build that spans more than ~10 files in one dispatch is a sign the abstraction is too big. Stop and escalate.
- **R-BUILD-6**: When a build moves a REQ from NOT-STARTED to SHIPPED, the design doc's REQ status table is updated in the SAME commit. Quote BOTH impl `file:line` AND non-test production consumer `file:line`.

### Characterization-test rules
- **R-CHAR-1**: For each REQ in a design doc that ships in iter-N, a test in `ferrotorch-*/tests/` (or the file's `#[cfg(test)] mod tests`) MUST exercise the upstream-PyTorch behavior the REQ mirrors. The test exists BEFORE the implementation lands and fails until the implementation is correct.
- **R-CHAR-2**: The acto-critic's failing tests count as characterization tests.
- **R-CHAR-3**: **No tautological tests.** Tests asserting a ferrotorch value equals an upstream value MUST construct the upstream value either by (a) live-calling PyTorch via the parity-sweep oracle, or (b) named typed bits / symbolic constants traceable to a PyTorch `file:line`. The self-referential pattern `const FERROTORCH_X = 0xABCD; const TORCH_X = 0xABCD; assert_eq!(FERROTORCH_X & TORCH_X, TORCH_X)` is tautologically true regardless of correctness and hides drift. acto-critic flags this pattern on sight as a divergence in its own right.

---

## Dispatch policy

For files whose REQs require more than ~800 LOC of new Rust to satisfy, dispatch via architect-mode subagents (one file per subagent, opus model, worktree isolation). Audit per Checkpoint 2 before merging. Smaller files can be done by the orchestrator directly.

When dispatching:
- Include the full 8-step loop verbatim in the subagent prompt.
- Set the subagent's evidence floor to the same per-REQ classification + per-upstream-file quoting requirement.
- The architect runs the gauntlet independently before merging the subagent's branch.

---

## Out of scope for this goal

- Adding new ops or layers that don't exist in PyTorch. (We are translating, not innovating.)
- Optimizing for performance ahead of correctness. (Correctness first; parity-sweep is the contract. Speed gains are bonus; speed regressions vs PyTorch are acceptable unless documented.)
- ferrotorch-nn-derive (a proc-macro crate with no PyTorch counterpart; routes are not required for it).
- The 28 high-level model crates (ferrotorch-llama, ferrotorch-bert, ferrotorch-whisper, ferrotorch-diffusion, etc.) — these COMPOSE ops; they're translated after their underlying ops are SHIPPED.

---

## Stopping condition

The goal halts only when ALL of these are true:

```bash
# 1. Every routed file has a closing commit
ROUTED_FILES=$(python3 -c "import tomllib; print(len(tomllib.load(open('tooling/translate-routes.toml','rb'))['route']))")
CLOSED_ISSUES=$(git log --oneline | grep -c "closes #")
test "$ROUTED_FILES" -le "$CLOSED_ISSUES"

# 2. Every routed file's parity_ops are all 'verified' in parity_audit.json
python3 -c "
import json, tomllib
routes = tomllib.load(open('tooling/translate-routes.toml','rb'))['route']
audit = json.load(open('tools/parity-sweep/parity_audit.json'))['ops']
all_ops = set(op for r in routes for op in r.get('parity_ops', []))
unverified = [op for op in all_ops if audit.get(op, {}).get('status') != 'verified']
print(f'unverified: {len(unverified)} of {len(all_ops)}')
"

# 3. Every routed file carries a `## REQ status` doc-comment
python3 -c "
import tomllib, subprocess
routes = tomllib.load(open('tooling/translate-routes.toml','rb'))['route']
missing = [r['crate_pattern'] for r in routes if subprocess.run(['grep','-l','## REQ status', r['crate_pattern']], capture_output=True).returncode != 0]
print(f'missing REQ status: {len(missing)}')
"
```

When all three are zero/equal, post a final summary commit + `crosslink issue close` on the master tracking issue, then stop.

Until then: every turn, one iteration of the eight-step loop. No exceptions.
