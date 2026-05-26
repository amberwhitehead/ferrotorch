# ferrotorch-optim — `scheduler::cyclic_lr` (CyclicLR)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/cyclic_lr.rs` defines
`pub struct CyclicLR` and `pub enum CyclicMode`, the cyclical LR
schedule from Smith (2017). The LR cycles between `base_lr` and
`max_lr` using a triangular wave; the amplitude can be held
constant (`Triangular`), halved each cycle (`Triangular2`), or
scaled by `gamma^iteration` (`ExpRange`). Mirrors
`class CyclicLR(LRScheduler)` at
`torch/optim/lr_scheduler.py:1787-2102`.

## Requirements

- REQ-1: `pub enum CyclicMode { Triangular, Triangular2, ExpRange }`
  with `Debug, Clone, Copy, PartialEq, Eq` derived. Mirrors
  upstream's `mode: Literal["triangular", "triangular2",
  "exp_range"]` (`lr_scheduler.py:1893`).

- REQ-2: `pub struct CyclicLR` with `base_lr: f64`, `max_lr: f64`,
  `total_size: f64` (= `step_size_up + step_size_down`),
  `step_ratio: f64` (= `step_size_up / total_size`),
  `mode: CyclicMode`, `gamma: f64`, `current_step: usize`,
  `current_lr: f64` fields. Mirrors upstream's attributes at
  `lr_scheduler.py:1886-1965`.

- REQ-3: `pub fn CyclicLR::new(base_lr, max_lr, step_size_up,
  step_size_down, mode, gamma) -> Self` constructor where
  `step_size_down: Option<usize>` defaults to `step_size_up` when
  `None`. Mirrors `CyclicLR.__init__` (`lr_scheduler.py:1886-1965`).

- REQ-4: `impl<T: Float> LrScheduler<T> for CyclicLR` with the
  triangular-wave + amplitude-scaling formula:

  ```text
  cycle = floor(1 + step / total_size)
  x = 1 + step / total_size - cycle           # in [0, 1)
  scale = (x / step_ratio) if x <= step_ratio
          else (x - 1) / (step_ratio - 1)
  base_height = (max_lr - base_lr) * scale
  match mode:
      Triangular: lr = base_lr + base_height
      Triangular2: lr = base_lr + base_height / 2^(cycle - 1)
      ExpRange: lr = base_lr + base_height * gamma^step
  ```

  Mirrors upstream's `get_lr` at `lr_scheduler.py:1999-2098`.

- REQ-5: NOT-STARTED — `cycle_momentum`, `base_momentum`,
  `max_momentum` (the inverse-momentum-cycling feature from
  upstream `lr_scheduler.py:1840-1862, 1935-1963`) are NOT
  implemented. The `Optimizer<T>` trait doesn't expose a
  `set_momentum` method, so cycling momentum is impossible
  without infrastructure work. Tracked by blocker #1461.

- REQ-6: NOT-STARTED — `scale_fn` (user-provided custom scaling
  callback at `lr_scheduler.py:1830-1834`) is NOT implemented.
  Tracked by blocker #1462.

## Acceptance Criteria

- [x] AC-1: `pub enum CyclicMode` with the three variants.
- [x] AC-2: `pub struct CyclicLR` with the eight named fields.
- [x] AC-3: At `step_size_up`, LR == `max_lr` (peak)
  (`test_cyclic_triangular_peak`).
- [x] AC-4: At `total_size`, LR == `base_lr` (valley)
  (`test_cyclic_triangular_valley`).
- [x] AC-5: `Triangular2` halves amplitude each cycle
  (`test_cyclic_triangular2_halves_amplitude`).
- [x] AC-6: `ExpRange` causes monotonic peak decay
  (`test_cyclic_exp_range_decays`).
- [x] AC-7: Asymmetric `step_size_up` vs `step_size_down` works
  (`test_cyclic_asymmetric_cycle`).
- [ ] AC-8: `cycle_momentum` — blocker #1461.
- [ ] AC-9: User-provided `scale_fn` — blocker #1462.

## Architecture

The triangular-wave formula uses cycle-relative position `x ∈
[0, 1)`, with `step_ratio` being the fraction of the cycle spent
ramping up. The scale function maps `x` to a `[0, 1]` amplitude
factor: 0 at the cycle start/end, 1 at the peak (at `x ==
step_ratio`).

Three modes apply different per-cycle / per-iteration amplitude
scaling on top of the triangular wave:

- `Triangular`: no amplitude scaling.
- `Triangular2`: amplitude halves each cycle via
  `1 / 2^(cycle - 1)`.
- `ExpRange`: amplitude scales by `gamma^step` (per-iteration
  decay).

`impl LrScheduler<T> for CyclicLR` advances `current_step`,
recomputes `current_lr` via `compute_lr`, pushes to
`optimizer.set_lr`.

### Non-test production consumers

- `CyclicLR` and `CyclicMode` re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`.
- `Learner::with_scheduler` at
  `ferrotorch-train/src/learner.rs:105` accepts the boxed
  `CyclicLR`; per-epoch step at
  `ferrotorch-train/src/learner.rs:306-308`.

## Parity contract

`parity_ops = []`. Numerical contract:

- **`step_size_down == None`**: defaults to `step_size_up`
  (symmetric cycle).
- **`step_size_up == 0` and `step_size_down == 0`**:
  `total_size == 0` would cause div-zero in the formula.
  Upstream raises; ferrotorch panics on the division. No
  defensive guard.
- **`Triangular2` past many cycles**: `2^(cycle - 1)` overflows
  for `cycle > ~1024`; the formula gracefully degrades to LR ==
  `base_lr` (because the amplitude vanishes).
- **`ExpRange` with `gamma > 1.0`**: amplitude grows
  exponentially. Allowed but almost certainly a user error.
- **Momentum cycling**: NOT-STARTED. The Rust optimizer surface
  doesn't expose `set_momentum`; would require trait extension
  per blocker #1461.

## Verification

Tests in `#[cfg(test)] mod tests` (6 tests):

- `test_cyclic_triangular_peak`
- `test_cyclic_triangular_valley`
- `test_cyclic_triangular2_halves_amplitude`
- `test_cyclic_exp_range_decays`
- `test_cyclic_asymmetric_cycle`
- `test_cyclic_midpoint_ramp_up`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::cyclic_lr 2>&1 | tail -3
```

Expected: `6 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum CyclicMode { Triangular, Triangular2, ExpRange }` with `Debug, Clone, Copy, PartialEq, Eq` derived in `scheduler/cyclic_lr.rs` mirrors `torch/optim/lr_scheduler.py:1893`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52` (`CyclicMode` is in the `pub use` list); user code constructs `CyclicLR::new(_, _, _, _, CyclicMode::Triangular, _)`. |
| REQ-2 | SHIPPED | impl: `pub struct CyclicLR` with `base_lr`, `max_lr`, `total_size`, `step_ratio`, `mode`, `gamma`, `current_step`, `current_lr` fields in `scheduler/cyclic_lr.rs` mirrors `torch/optim/lr_scheduler.py:1886-1965`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user code boxes it for `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
| REQ-3 | SHIPPED | impl: `pub fn CyclicLR::new(base_lr, max_lr, step_size_up, step_size_down: Option<usize>, mode, gamma) -> Self` with default-to-symmetric in `scheduler/cyclic_lr.rs` mirrors `torch/optim/lr_scheduler.py:1886-1965`; non-test consumer: the `pub use` at `lib.rs:47-52` is the user-call surface. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for CyclicLR` with the triangular-wave + per-mode amplitude scaling in `scheduler/cyclic_lr.rs` mirrors `torch/optim/lr_scheduler.py:1999-2098`; non-test consumer: `Learner` invokes `sched.step(self.optimizer.as_mut())` at `ferrotorch-train/src/learner.rs:306-308`, dispatching to this impl when the boxed scheduler is a `CyclicLR`. |
| REQ-5 | NOT-STARTED | blocker #1461 — momentum cycling requires `Optimizer<T>` trait extension to expose `set_momentum`; ferrotorch optimizer surface currently has no such method. The `cycle_momentum`, `base_momentum`, `max_momentum` upstream features (`torch/optim/lr_scheduler.py:1840-1862, 1935-1963`) cannot be wired until the trait grows the missing accessor. |
| REQ-6 | NOT-STARTED | blocker #1462 — user-provided `scale_fn` (upstream `torch/optim/lr_scheduler.py:1830-1834`) is not exposed in the ferrotorch constructor. Mode selection is limited to the three built-in policies. Adding a closure field would require a structural change to `CyclicLR` that no current user code requests. |
