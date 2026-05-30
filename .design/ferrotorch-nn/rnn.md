# ferrotorch-nn — `rnn` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/rnn.py
  - aten/src/ATen/native/RNN.cpp
-->

## Summary

`ferrotorch-nn/src/rnn.rs` implements the recurrent neural-network
module family: `LSTM`, `GRU`, `RNN` (multi-layer sequence modules)
plus their single-step `LSTMCell`, `GRUCell`, `RNNCell` counterparts.
Mirrors `torch.nn.{LSTM, GRU, RNN, LSTMCell, GRUCell, RNNCell}` at
`torch/nn/modules/rnn.py:48-1860` for the API surface, parameter
layout (gates concatenated into a single weight matrix per layer),
and initialisation convention (`U(-k, k)` with `k = 1/sqrt(hidden)`).

The forward paths compose only differentiable primitives from
`ferrotorch_core::grad_fns` (`mm_differentiable`, `add`, `mul`, `sub`,
`sigmoid`, `tanh`, `relu`, `cat`, `reshape`), so autograd builds the
backward graph automatically — no custom `GradFn<T>` is required.

## Requirements

- REQ-1: `pub struct LSTM<T: Float>` — multi-layer LSTM with per-layer
  `weight_ih: [4 * hidden, input]`, `weight_hh: [4 * hidden, hidden]`,
  `bias_ih: [4 * hidden]`, `bias_hh: [4 * hidden]`. Mirrors upstream's
  flat parameter layout at `rnn.py:829-1203`.

- REQ-2: `LSTM::forward(input, h0, c0) -> (output, (h_n, c_n))` —
  computes the standard LSTM update
  `i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)`,
  `f = sigmoid(W_if x + b_if + W_hf h + b_hf)`,
  `g = tanh(W_ig x + b_ig + W_hg h + b_hg)`,
  `o = sigmoid(W_io x + b_io + W_ho h + b_ho)`,
  `c' = f * c + i * g`, `h' = o * tanh(c')`. Mirrors upstream's
  algebra at `rnn.py:829-870`.

- REQ-3: `pub struct GRU<T: Float>` — multi-layer GRU with the same
  `weight_ih`, `weight_hh`, `bias_ih`, `bias_hh` layout (3 gates ×
  `hidden_size` per matrix). Mirrors `rnn.py:1204-1481`.

- REQ-4: `GRU::forward(input, h0) -> (output, h_n)` — computes
  `r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)`,
  `z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)`,
  `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`,
  `h' = (1 - z) * n + z * h`. Mirrors upstream's algebra at
  `rnn.py:1204-1250`.

- REQ-5: `pub enum RNNNonlinearity { Tanh, ReLU }` — selects the
  per-step activation for `RNN<T>`. Mirrors upstream's `nonlinearity`
  string kwarg.

- REQ-6: `pub struct RNN<T: Float>` — multi-layer vanilla RNN with
  `weight_ih: [hidden, input]`, `weight_hh: [hidden, hidden]`,
  `bias_ih: [hidden]`, `bias_hh: [hidden]`. Mirrors `rnn.py:486-828`.

- REQ-7: `RNN::forward(input, h0) -> (output, h_n)` — computes
  `h' = activation(W_ih x + b_ih + W_hh h + b_hh)`. Mirrors
  upstream's algebra.

- REQ-8: `pub struct LSTMCell<T: Float>` /
  `pub struct GRUCell<T: Float>` /
  `pub struct RNNCell<T: Float>` — single-step cell counterparts.
  Same parameter layout as their multi-layer modules, but operate
  per-step rather than over a sequence. Mirror upstream's
  `LSTMCell` / `GRUCell` / `RNNCell` at
  `rnn.py:1540-1860`.

- REQ-9: Default weight initialisation — `U(-k, k)` with
  `k = 1 / sqrt(hidden_size)`; biases initialised to zero. Matches
  upstream's `reset_parameters` at `rnn.py:271-289`.

- REQ-10: `Module<T>` impl for every public struct — `forward` (the
  cell variants take `(input, state) -> state`; the multi-layer
  modules wrap that contract), `parameters`/`parameters_mut`,
  `named_parameters` (keys: `weight_ih_l<i>`, `weight_hh_l<i>`,
  `bias_ih_l<i>`, `bias_hh_l<i>` for multi-layer; `weight_ih`,
  `weight_hh`, `bias_ih`, `bias_hh` for cells), `train`/`eval`/
  `is_training`. Matches upstream's `state_dict` keys.

- REQ-11: Device-aware matmul — `mm_differentiable` is used (not the
  host-only `ops::linalg::mm`) so GPU-resident input/hidden tensors
  dispatch to the GPU matmul path. Required by issue #750.

- REQ-12: Parity op `nn.functional.lstm_cell` — single-step LSTM
  output matches upstream's `F.lstm_cell(input, (h, c), w_ih, w_hh,
  b_ih, b_hh)` to within float32 tolerance. NOT-STARTED until the
  parity-sweep runner has a dispatch arm (blocker #1456).

- REQ-13: Parity op `nn.functional.gru_cell` — single-step GRU output
  matches upstream. NOT-STARTED — blocker #1456.

- REQ-14: Parity op `nn.functional.rnn_relu_cell` — single-step
  vanilla RNN (ReLU activation) output matches upstream.
  NOT-STARTED — blocker #1456.

## Acceptance Criteria

- [x] AC-1: `LSTM::new(input_size=10, hidden_size=20, num_layers=2)`
  constructs with 8 parameter tensors (4 per layer).
- [x] AC-2: `LSTM::forward([B, T, 10], h0, c0)` returns
  `([B, T, 20], ([num_layers, B, 20], [num_layers, B, 20]))`.
- [x] AC-3: `GRU::new(10, 20)` constructs with 4 parameter tensors.
- [x] AC-4: `RNN::new(10, 20)` with `Tanh` / `ReLU` constructs
  correctly.
- [x] AC-5: `LSTMCell::forward(x, (h, c))` returns `(h', c')` of the
  expected shapes.
- [x] AC-6: `named_parameters` returns keys matching upstream's
  `weight_ih_l0`, `weight_hh_l0`, ... convention.
- [x] AC-7: GPU-resident inputs dispatch through `mm_differentiable`.
- [ ] AC-8: parity-sweep `nn.functional.lstm_cell` at status
  `verified` — blocker #1456.
- [ ] AC-9: parity-sweep `nn.functional.gru_cell` at status
  `verified` — blocker #1456.
- [ ] AC-10: parity-sweep `nn.functional.rnn_relu_cell` at status
  `verified` — blocker #1456.

## Architecture

### Per-layer parameter set (REQ-1, REQ-3, REQ-6)

`struct LSTMLayerParams<T>` / `struct GRULayerParams<T>` /
`struct RNNLayerParams<T>` (private) at the corresponding
`struct ...LayerParams in rnn.rs` items carry the four parameter
tensors per layer with the gate-concatenated shapes from upstream.
The multi-layer modules hold `layers: Vec<LayerParams<T>>`.

### LSTM (REQ-1, REQ-2)

`pub struct LSTM<T: Float>` at
`pub struct LSTM in rnn.rs` carries `input_size`, `hidden_size`,
`num_layers`, the layer params, and a `training` flag. `forward`
iterates over time steps and layers, applying the four-gate update
per step. Initialises `h0` and `c0` to zero tensors when callers
pass `None`.

### GRU (REQ-3, REQ-4)

`pub struct GRU<T: Float>` at
`pub struct GRU in rnn.rs`. Same multi-layer driver but with
three-gate update: reset, update, new-state. Uses `sub(one, z)` to
compute `(1 - z)` for the convex combination.

### RNN (REQ-5, REQ-6, REQ-7)

`pub enum RNNNonlinearity` at
`pub enum RNNNonlinearity in rnn.rs` and `pub struct RNN<T: Float>`
at `pub struct RNN in rnn.rs`. The forward dispatches on the enum
to choose `tanh` or `relu` per step.

### Cell counterparts (REQ-8)

`pub struct LSTMCell<T: Float>` /
`pub struct GRUCell<T: Float>` /
`pub struct RNNCell<T: Float>` at the corresponding
`pub struct ...Cell in rnn.rs` items expose the single-step
contract. Same parameter layout, but no multi-layer driver.

### Initialisation (REQ-9)

The constructors call `init::uniform_(param, -k, k)` from
`crate::init` to populate weights, then `init::zeros_(param)` for
biases. `k = 1.0 / (hidden_size as f64).sqrt()`.

### Module trait surface (REQ-10)

Every public struct has an `impl<T: Float> Module<T> for <Type><T>`
block; collectively at the `impl Module<T>` sites in `rnn.rs`. The
multi-layer modules expose `weight_ih_l<i>` / `weight_hh_l<i>` /
`bias_ih_l<i>` / `bias_hh_l<i>` keys; the cells expose
`weight_ih` / `weight_hh` / `bias_ih` / `bias_hh`. This matches
upstream's `state_dict` for portable checkpoints.

### Device routing (REQ-11)

The module-level `use ferrotorch_core::grad_fns::linalg::
mm_differentiable as mm` import at `use mm_differentiable as mm in
rnn.rs` is the entry to the device-aware, autograd-tracked matmul.
Required because the host-only `ops::linalg::mm` reads via
`.data()?` and errors on GPU storage after the Phase-2a
`try_as_slice` migration (#750).

### Sequence-forward performance (REQ-2, REQ-4, REQ-7)

The `LSTM`/`GRU`/`RNN` sequence forwards apply two reassociations to
the per-timestep loop, both pure (value- and gradient-identical) so
they preserve exact autograd parity:

1. **Hoisted recurrent-weight transpose (#1680).** Each layer
   transposes + materializes `weight_ih`/`weight_hh` once
   (`transpose_2d(...).contiguous()`) outside the timestep loop
   instead of per step.

2. **Batched input-to-hidden projection (#1690).** The input
   projection `x_t @ W_ih^T` has no time dependency, so the `seq_len`
   separate `[batch, in] @ [in, k*hs]` GEMMs are folded into ONE
   `[seq_len*batch, in] @ [in, k*hs]` GEMM via the private
   `fn batched_input_projection in rnn.rs` helper (stack the per-step
   inputs with `cat` dim 0, run one `mm`, slice back per step with
   `narrow(0, t*batch, batch)`). Only the recurrent `h @ W_hh^T`
   (genuine time dependency) stays inside the loop. This mirrors
   upstream `FullLayer::operator()` at
   `aten/src/ATen/native/RNN.cpp:863-869`, which projects the whole
   stacked sequence with `linear_ih` then consumes per timestep with
   `pre_compute_input=true`. The projection is kept bias-free so the
   GRU GPU fused-cell kernel (`fused_gru_cell_f32`) still receives the
   raw gate matrix plus separate biases; the GPU per-step slice is
   `.contiguous()`-materialized before its buffer handle reaches the
   offset-unaware kernel. The gradient to `weight_ih` accumulates
   through one matmul node — the concatenation of the per-step
   upstream grads — equal to the sum the per-step shape produced
   across `seq_len` matmul nodes (pinned vs LIVE torch in
   `divergence_rnn_hoist_autograd_reaudit.rs` and vs the per-step
   reference in `divergence_1690_rnn_batched_input_projection.rs`).

   On the CPU BLAS path the win is marginal (the per-call GEMM
   overhead this amortizes is small and the input `cat` adds a copy);
   the batching is a launch-overhead win on the GPU/cuBLAS path where
   many tiny GEMMs are dominated by per-launch cost. The residual CPU
   gap to cuDNN at `[128->256, seq=32, B=16]` is the `seq_len`
   per-step recurrent GEMMs plus composite gate ops, which carry a
   true time dependency and cannot be batched.

### Non-test production consumers

- `pub use rnn::{GRU, GRUCell, LSTM, LSTMCell, RNN, RNNCell,
  RNNNonlinearity}` at `ferrotorch-nn/src/lib.rs:245` —
  grandfathered public API surface.
- `ferrotorch/src/lib.rs:50` — `pub use ferrotorch_nn::{GRU, LSTM}`
  in the meta-crate prelude.
- `benchmarks/ferrotorch_bench.rs:178` — `GRU::new(128, 256)` for
  the benchmark suite (production-side host benchmark).

## Parity contract

### `nn.functional.lstm_cell`

- Upstream entry: `torch/nn/functional.py — lstm_cell` →
  `aten/src/ATen/native/RNN.cpp`.
- Edge cases preserved by `LSTMCell::forward`:
  - **Gate-bias additivity** — `b_ih` and `b_hh` are summed (not
    averaged); matches upstream's `b_ih + b_hh` convention.
  - **Output ordering** — `(i, f, g, o)` gate order from the
    concatenated `[4 * hidden, ...]` matrix. Matches upstream's
    `chunk` order.
- Parity-sweep audit status: `MISSING` (blocker #1456).

### `nn.functional.gru_cell`

- Upstream entry: `torch/nn/functional.py — gru_cell` →
  `aten/src/ATen/native/RNN.cpp`.
- Edge case: the reset gate `r` multiplies `(W_hn h + b_hn)`
  BEFORE the tanh, not `W_hn (r * h)`. Matches upstream's
  ordering at `rnn.py:1204-1250`.
- Parity-sweep audit status: `MISSING` (blocker #1456).

### `nn.functional.rnn_relu_cell`

- Upstream entry: `torch/nn/functional.py — rnn_relu_cell`.
- Edge case: ReLU output can blow up for poorly-initialised
  hidden state; ferrotorch matches upstream's no-clipping
  behaviour.
- Parity-sweep audit status: `MISSING` (blocker #1456).

## Verification

Tests in `mod tests in rnn.rs`. Highlights:

- Construction tests for each module.
- Shape contracts for `forward` returns.
- Parameter-count tests against upstream's gate-multiplier convention.
- Named-parameters tests verifying `state_dict` key layout.

Parity smoke command (blocker #1456 must close):

```bash
for OP in nn.functional.gru_cell \
          nn.functional.lstm_cell \
          nn.functional.rnn_relu_cell; do
  ./target/release/parity-sweep sweep --op "$OP" --seeds 8 2>&1 \
    | grep -c "passed (0 skipped, 0 failed)"
done
```

Expected (post-#1456): each line returns `>= 1`. Current: each
returns `0` (runner arm missing).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LSTM<T: Float>` plus the private `LSTMLayerParams<T>` in `rnn.rs`; non-test consumer: re-export at `ferrotorch-nn/src/lib.rs:245` + `ferrotorch/src/lib.rs:50`. |
| REQ-2 | SHIPPED | impl: `pub fn forward` on `LSTM` in `rnn.rs` with the four-gate update; non-test consumer: re-export at `lib.rs` + meta-crate prelude `ferrotorch/src/lib.rs` + benchmark consumer `benchmarks/ferrotorch_bench.rs`. |
| REQ-3 | SHIPPED | impl: `pub struct GRU<T: Float>` in `rnn.rs`; non-test consumer: re-export at `lib.rs` + meta-crate prelude `ferrotorch/src/lib.rs` + `benchmarks/ferrotorch_bench.rs`. |
| REQ-4 | SHIPPED | impl: `pub fn forward` on `GRU` in `rnn.rs` with the three-gate update; non-test consumer: as REQ-3. |
| REQ-5 | SHIPPED | impl: `pub enum RNNNonlinearity` in `rnn.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-6 | SHIPPED | impl: `pub struct RNN<T: Float>` in `rnn.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-7 | SHIPPED | impl: `pub fn forward` on `RNN` in `rnn.rs` dispatching on `RNNNonlinearity`; non-test consumer: re-export at `lib.rs`. |
| REQ-8 | SHIPPED | impl: `pub struct LSTMCell<T: Float>`, `pub struct GRUCell<T: Float>`, `pub struct RNNCell<T: Float>` in `rnn.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-9 | SHIPPED | impl: `init::uniform_` and `init::zeros_` calls in the constructors of `LSTM`/`GRU`/`RNN`/cells; non-test consumer: re-export at `lib.rs`. |
| REQ-10 | SHIPPED | impl: `impl<T: Float> Module<T> for ...` blocks for every public struct in `rnn.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-11 | SHIPPED | impl: `use ferrotorch_core::grad_fns::linalg::mm_differentiable as mm` import in `rnn.rs` plus its use in every forward path; non-test consumer: re-export at `lib.rs`. |
| REQ-12 | NOT-STARTED | parity-sweep runner arm for `nn.functional.lstm_cell` not wired — blocker #1456. |
| REQ-13 | NOT-STARTED | parity-sweep runner arm for `nn.functional.gru_cell` not wired — blocker #1456. |
| REQ-14 | NOT-STARTED | parity-sweep runner arm for `nn.functional.rnn_relu_cell` not wired — blocker #1456. |
