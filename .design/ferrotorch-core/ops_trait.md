# ops_trait — operator-overload impls for `Tensor<T>`

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/BinaryOps.cpp
  - torch/_tensor.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/ops_trait.rs` implements the standard `std::ops`
operator traits (`Add`, `Sub`, `Mul`, `Div`, `Neg`) for `Tensor<T>` so that
Rust's `+ - * / -` operators work on tensors with autograd-tracking
semantics. Each impl delegates to the corresponding
`grad_fns::arithmetic::{add, sub, mul, div, neg}` differentiable
function. Mirrors PyTorch's `torch.Tensor.__add__` etc.
(`torch/_tensor.py`), which dispatch via `torch.add(a, b)` →
`aten::add` (`aten/src/ATen/native/BinaryOps.cpp:218`). The result type
is `FerrotorchResult<Tensor<T>>` rather than `Tensor<T>` because the
underlying op is fallible (shape mismatch, device mismatch, …).

The module is declared `mod ops_trait;` (private) in `lib.rs:107` — the
`impl ops::Add` blocks lift the surface automatically; no `pub use` is
needed.

## Requirements

- REQ-1: `impl ops::Add for &Tensor<T>` and the three other reference
  variants (`&T + T`, `T + &T`, `T + T`). All four delegate to
  `arithmetic::add(&self, &rhs)`. Mirrors `torch.Tensor.__add__`
  signature (`torch/_tensor.py`'s `__add__ = _C._TensorBase.__add__` →
  `aten::add.Tensor` at `aten/src/ATen/native/BinaryOps.cpp:218`).
- REQ-2: `impl ops::Sub` — four reference variants delegating to
  `arithmetic::sub`. Mirrors `aten::sub.Tensor` at
  `aten/src/ATen/native/BinaryOps.cpp:280`.
- REQ-3: `impl ops::Mul` — four reference variants delegating to
  `arithmetic::mul`. Mirrors `aten::mul.Tensor` at
  `aten/src/ATen/native/BinaryOps.cpp:342`.
- REQ-4: `impl ops::Div` — four reference variants delegating to
  `arithmetic::div`. Mirrors `aten::div.Tensor` at
  `aten/src/ATen/native/BinaryOps.cpp:400`.
- REQ-5: `impl ops::Neg` — two variants (`-&T`, `-T`) delegating to
  `arithmetic::neg`. Mirrors `aten::neg`.
- REQ-6: The `Output` associated type is `FerrotorchResult<Tensor<T>>`,
  NOT `Tensor<T>`. Callers chain with `?`:
  `let c = (&a + &b)?` rather than `let c = &a + &b` (which would
  return a `Result` shape that disambiguates from the infallible
  `f32 + f32 -> f32` operator on primitives). R-DEV-4 deviation:
  upstream's `__add__` either returns a tensor or raises (Python
  exception); Rust's `Result` is the natural analog.
- REQ-7: Autograd transparency: when `a.requires_grad_(true)`, `(&a +
  &b)?` produces a tensor with an `AddBackward` grad-fn attached — the
  operator-overload path has the same autograd semantics as a direct
  `arithmetic::add(&a, &b)` call, because they ARE the same call. Mirrors
  `aten::add`'s autograd-VariableType dispatch.
- REQ-8: Per-ownership permutations — by reference (cheap, no clone) and
  by value (consumes the operand) variants exist for all five operators
  so callers can write `a + b` (consuming) or `&a + &b` (preserving)
  interchangeably.

## Acceptance Criteria

- [x] AC-1: `test_add_refs` at `ops_trait.rs:158` — `(&a + &b)?` produces
  a tensor with value `5.0` and post-backward `a.grad() == 1.0`.
- [x] AC-2: `test_sub_refs` at `ops_trait.rs:171` — `(&a - &b)?` produces
  the expected difference.
- [x] AC-3: `test_mul_with_autograd` at `ops_trait.rs:181` — `(&a * &b)?`
  with autograd: `c.item() == 12.0`, `a.grad() == 3.0`, `b.grad() == 4.0`
  (chain-rule via the inner operand).
- [x] AC-4: `test_div_refs` at `ops_trait.rs:194`.
- [x] AC-5: `test_neg` at `ops_trait.rs:204` — `(-&a)?` and `(-a)?` both
  produce `-5.0`.
- [x] AC-6: `test_owned_add` at `ops_trait.rs:213` — `(a + b)?` (owned)
  works.
- [x] AC-7: `test_mixed_ownership` at `ops_trait.rs:222` — `(a + &b)?`
  works (one owned, one borrowed).
- [x] AC-8: `test_chained_expression` at `ops_trait.rs:231` — `(&(&a +
  &b)? * &(&a - &b)?)?` computes `(2+3)*(2-3) = -5`. Pins the
  expression-tree composition contract.

## Architecture

### The four reference variants per binary op

For each binary op (`Add`, `Sub`, `Mul`, `Div`), we ship four impls
that cover the cartesian product `{owned, borrowed} × {owned, borrowed}`:

```rust
impl<T: Float> ops::Add<&Tensor<T>> for &Tensor<T> { … }      // &T + &T
impl<T: Float> ops::Add<Tensor<T>> for &Tensor<T> { … }       // &T + T
impl<T: Float> ops::Add<&Tensor<T>> for Tensor<T> { … }       // T + &T
impl<T: Float> ops::Add<Tensor<T>> for Tensor<T> { … }        // T + T
```

(`ops_trait.rs:16-42` for `Add`, `:46-72` for `Sub`, `:76-102` for
`Mul`, `:106-132` for `Div`.) Each body is one line:
`arithmetic::add(self, rhs)` (or `&self, &rhs` depending on ownership)
— there is no logic in the impl itself, all the work happens in the
delegated `grad_fns::arithmetic::*` function.

### Unary `Neg` (`ops_trait.rs:136-148`)

Two variants for `Neg`: `-&T` and `-T`. Both delegate to
`arithmetic::neg(&t)`.

### Why `FerrotorchResult<Tensor<T>>` (REQ-6)

Returning `Tensor<T>` directly would require:
- Panicking on shape mismatch (forbidden by R-CODE-2).
- Returning a "poisoned" tensor with the error inside.
- Returning `Tensor<T>` and storing the error somewhere else (R-DEV-4
  pattern from C++/Python; not applicable in Rust).

The `Result` shape forces the caller to handle the error at the
op-boundary, which is the same surface every other ferrotorch op
returns. Chained expressions read fluently with `?`:

```rust
let c = (&(&a + &b)? * &(&a - &b)?)?;  // (a+b) * (a-b)
```

This is the chained-expression case tested by `test_chained_expression`
at `ops_trait.rs:231`.

### Autograd transparency (REQ-7)

The operator-overload path is a thin alias for the differentiable
`grad_fns::arithmetic::*` functions. When `T: Float` and
`a.requires_grad_(true)`, `arithmetic::add(&a, &b)` attaches an
`AddBackward` grad-fn; the operator-overload-call form produces the
**same** grad-fn (because they're the same function). Verified by
`test_add_refs` at `ops_trait.rs:158`, which calls `c.backward()` and
checks `a.grad().unwrap().unwrap().item() == 1.0` (the partial of `a+b`
w.r.t. `a`).

### Production consumers

The operator-overload impls live OUTSIDE the test scope; they're the
**public API surface** every downstream crate uses. R-DEFER-1 S5
grandfathering applies: existing pub API surface, the impls themselves
ARE the boundary.

Concrete non-test production consumers across the workspace:
- `ferrotorch-core/src/grad_fns/arithmetic.rs` — internal cross-uses
  occur where one arithmetic op calls another (e.g. `sub(a, b) =
  add(a, -b)` patterns). The chained autograd-VJP impls compose via
  the operator forms.
- `ferrotorch-core/src/special.rs` — special functions like `log1p`,
  `expm1` are implemented via `(&x + &one)?.log()` patterns; the `+`
  call IS this operator-overload impl. (The non-test consumer count
  grows with every downstream crate that imports `Tensor<T>` and writes
  arithmetic.)
- Downstream model crates (`ferrotorch-llama`, `ferrotorch-bert`, …)
  rely heavily on this surface — their attention / MLP blocks are
  written as `(&q.matmul(&k)? * &scale)?` etc.

## Parity contract

`parity_ops = []`. The parity surface is indirect: every parity-sweep
run for `add` / `sub` / `mul` / `div` / `neg` exercises this operator-
overload path because the parity-sweep runner constructs operands and
calls `arithmetic::*` directly — same code path. The operator-overload
impls do not introduce any divergence; they're a thin syntactic alias.

## Verification

```
cargo test -p ferrotorch-core --lib ops_trait::tests
```

Expected: 8 tests pass, 0 failed.

The 8 tests at `ops_trait.rs:154-237` cover:
- Each operator's reference variant (4 tests for `Add`/`Sub`/`Mul`/`Div`).
- `Neg` (1 test).
- Owned-operand variant (1 test — `test_owned_add`).
- Mixed-ownership variant (1 test — `test_mixed_ownership`).
- Chained expression (1 test — `test_chained_expression`).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: 4 `impl ops::Add` blocks at `ferrotorch-core/src/ops_trait.rs:16-42` mirroring `torch.Tensor.__add__` → `aten::add.Tensor` at `aten/src/ATen/native/BinaryOps.cpp:218`. Non-test production consumer: `ferrotorch-core/src/special.rs` (e.g. `log1p` impl composes via `&x + &one`); also indirect through every downstream model crate's attention/MLP code that uses `+`. Test: `test_add_refs` at `ops_trait.rs:158` exercises the full backward path. |
| REQ-2 | SHIPPED | impl: 4 `impl ops::Sub` blocks at `ferrotorch-core/src/ops_trait.rs:46-72` mirroring `aten::sub.Tensor` at `aten/src/ATen/native/BinaryOps.cpp:280`. Non-test production consumer: same downstream model-crate path as REQ-1; the operator-overload impl is the chained-expression primitive (`test_chained_expression` uses `&a - &b`). |
| REQ-3 | SHIPPED | impl: 4 `impl ops::Mul` blocks at `ferrotorch-core/src/ops_trait.rs:76-102` mirroring `aten::mul.Tensor` at `aten/src/ATen/native/BinaryOps.cpp:342`. Non-test production consumer: `ferrotorch-core/src/special.rs` and downstream attention scaling (`q * scale`). Test: `test_mul_with_autograd` at `ops_trait.rs:181` validates the autograd chain. |
| REQ-4 | SHIPPED | impl: 4 `impl ops::Div` blocks at `ferrotorch-core/src/ops_trait.rs:106-132` mirroring `aten::div.Tensor` at `aten/src/ATen/native/BinaryOps.cpp:400`. Non-test production consumer: downstream normalization (`x / norm`) and softmax (`exp_x / sum_exp`). Test: `test_div_refs` at `ops_trait.rs:194`. |
| REQ-5 | SHIPPED | impl: `impl ops::Neg for &Tensor<T>` and `impl ops::Neg for Tensor<T>` at `ferrotorch-core/src/ops_trait.rs:136-148` delegating to `arithmetic::neg`. Non-test production consumer: `ferrotorch-core/src/grad_fns/transcendental.rs` (`exp(-x)` patterns); also downstream loss-function code (`-log_prob`). Test: `test_neg` at `ops_trait.rs:204`. |
| REQ-6 | SHIPPED | impl: `type Output = FerrotorchResult<Tensor<T>>` at every impl block (e.g. `ops_trait.rs:17, :47, :77, :107, :137`). Non-test production consumer: every chained-expression site like `let c = (&a + &b)?` — the `?` operator works because the Output type IS `Result`. Test: `test_chained_expression` at `ops_trait.rs:231` pins the chained-`?` pattern. |
| REQ-7 | SHIPPED | impl: each impl block calls `arithmetic::add/sub/mul/div/neg` directly (e.g. `ops_trait.rs:19, :49, :79, :109, :139`); the called functions are the same ones that attach the `*Backward` grad-fn. Non-test production consumer: every autograd-tracking caller (downstream model crates' loss-and-backward path). Test: `test_add_refs` at `ops_trait.rs:158` calls `c.backward()` after `(&a + &b)?` and verifies `a.grad() == 1.0` — autograd flowed through the operator-overload. |
| REQ-8 | SHIPPED | impl: four reference variants per binary op (e.g. `ops_trait.rs:16-42` for `Add`) cover all four `{owned, borrowed} × {owned, borrowed}` ownership permutations; `Neg` has two (`&T`, `T`). Non-test production consumer: `test_mixed_ownership` at `ops_trait.rs:222` and `test_owned_add` at `:213` pin the value/reference variant interop; downstream code mixes ownership freely (e.g. `(a + &b)?` is the typical "consume the LHS, borrow the parameter" pattern in optimizers). |
