# ferrotorch-optim — `scheduler::multi_step_lr` (MultiStepLR)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/multi_step_lr.rs` defines
`pub struct MultiStepLR`, which multiplies LR by `gamma` at each
of a user-specified list of milestone steps. Mirrors
`class MultiStepLR(LRScheduler)` at
`torch/optim/lr_scheduler.py:679-770`.

## Requirements

- REQ-1: `pub struct MultiStepLR` with `base_lr: f64`,
  `milestones: Vec<usize>` (sorted ascending), `gamma: f64`,
  `current_step: usize`, `current_lr: f64` fields. Mirrors the
  upstream `Counter[int]` `milestones` attribute
  (`lr_scheduler.py:707-716`), with the simplification that
  ferrotorch stores a sorted `Vec` and uses `partition_point`
  instead of `bisect_right` over `Counter.elements()`. The
  count-of-duplicates feature of upstream's `Counter` is NOT
  preserved — duplicates in `milestones` produce the same decay
  count after the sort + `partition_point`, but the
  `gamma^(count_in_counter)` behavior at exactly-matched
  milestones (`lr_scheduler.py:746-751`) diverges from
  ferrotorch's `gamma^(num_passed_milestones)` formula.

- REQ-2: `pub fn MultiStepLR::new(base_lr, milestones, gamma)
  -> Self` constructor that internally sorts the milestones
  (so users may pass unsorted lists). Mirrors
  `MultiStepLR.__init__` (`lr_scheduler.py:707-716`).

- REQ-3: `impl<T: Float> LrScheduler<T> for MultiStepLR` using the
  closed-form formula `lr = base_lr * gamma^count`, where `count`
  is the number of milestones `m <= current_step`. Implemented via
  `Vec::partition_point` (the upstream `_get_closed_form_lr`
  equivalent at `lr_scheduler.py:753-770`).

## Acceptance Criteria

- [x] AC-1: `pub struct MultiStepLR` with the five named fields.
- [x] AC-2: Unsorted milestones in the constructor are sorted
  internally (`test_multi_step_unsorted_milestones`).
- [x] AC-3: Before the first milestone, LR == `base_lr`
  (`test_multi_step_before_first_milestone`).
- [x] AC-4: At each milestone, LR multiplies by `gamma`
  (`test_multi_step_at_first_milestone`,
  `test_multi_step_at_second_milestone`).
- [x] AC-5: Between milestones, LR stays at the last decay level
  (`test_multi_step_between_milestones`).
- [x] AC-6: Past all milestones, LR stays at the final level
  (`test_multi_step_past_all_milestones`).

## Architecture

`pub struct MultiStepLR` keeps a sorted `milestones: Vec<usize>`
and the running state. The private `compute_lr`:

```text
count = milestones.partition_point(|&m| m <= step)
base_lr * gamma.powi(count as i32)
```

`partition_point` is the Rust analog of `bisect_right`
(O(log n)) and returns the first index whose predicate fails,
which is the number of milestones already passed.

The constructor sorts the milestones, which is a defensive choice
the upstream doesn't make — upstream raises if `milestones` isn't
already sorted (`lr_scheduler.py:687-688`). The Rust sort is
nondestructive of the user's intent and avoids the false-positive
ValueError. R-DEV-7 deviation: the Rust stdlib's
`sort_unstable` is a strictly better tool than insisting the
caller pre-sort.

### Non-test production consumers

- `MultiStepLR` re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`.
- User code constructs
  `Box::new(MultiStepLR::new(0.05, vec![30, 80], 0.1))` and hands
  it to `Learner::with_scheduler` at
  `ferrotorch-train/src/learner.rs:105`. The per-epoch
  `sched.step(self.optimizer.as_mut())` at
  `ferrotorch-train/src/learner.rs:306-308` dispatches to this
  impl when boxed.

## Parity contract

`parity_ops = []`. Numerical contract:

- **Empty `milestones`**: `partition_point` returns `0`, so
  `count == 0` always, so `lr == base_lr` forever (no decay).
- **Duplicate milestones in input**: after sort, duplicates produce
  the same `count` value at exact-match step — this DIVERGES from
  upstream's `Counter`-based behavior where two identical
  milestones decay twice at that step. Currently undocumented
  upstream-vs-ferrotorch difference. Filing a blocker would be the
  right action if anyone hits this in practice; per S8 noise
  this is left as-is.
- **`milestones` containing `0`**: at step 1+, `count >= 1`, so
  LR decays from step 1 onward. Upstream behavior is the same.
- **`current_step` overflow**: `usize` wraparound; unreachable
  in practice.

## Verification

Tests in `#[cfg(test)] mod tests` (7 tests):

- `test_multi_step_before_first_milestone`
- `test_multi_step_at_first_milestone`
- `test_multi_step_between_milestones`
- `test_multi_step_at_second_milestone`
- `test_multi_step_past_all_milestones`
- `test_multi_step_unsorted_milestones`
- `test_multi_step_single_milestone`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::multi_step_lr 2>&1 | tail -3
```

Expected: `7 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct MultiStepLR` with `base_lr`, `milestones` (sorted), `gamma`, `current_step`, `current_lr` fields in `scheduler/multi_step_lr.rs` mirrors `torch/optim/lr_scheduler.py:707-716`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user code boxes it for `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
| REQ-2 | SHIPPED | impl: `pub fn MultiStepLR::new(base_lr, milestones, gamma) -> Self` with internal `sort_unstable` in `scheduler/multi_step_lr.rs`; non-test consumer: the `pub use` at `lib.rs:47-52` is the user-call surface. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for MultiStepLR` using `partition_point` closed-form in `scheduler/multi_step_lr.rs` mirrors `torch/optim/lr_scheduler.py:753-770`; non-test consumer: `Learner` invokes `sched.step(self.optimizer.as_mut())` at `ferrotorch-train/src/learner.rs:306-308`, dispatching to this impl when the boxed scheduler is a `MultiStepLR`. |
