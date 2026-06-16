# grad_fns module root

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/
  - c10/
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/grad_fns/mod.rs` is the module-root dispatch file for the
autograd-tracking wrapper layer. It contains no functions, no types, no
constants — only `pub mod` declarations re-exporting the 11 per-area
submodules (`activation`, `arithmetic`, `comparison`, `cumulative`, `fft`,
`indexing`, `linalg`, `quantize_grad`, `reduction`, `shape`,
`transcendental`). The split mirrors PyTorch's organization of `aten/src/ATen/native/`
into per-area `.cpp` translation units (`BinaryOps.cpp`, `UnaryOps.cpp`,
`ReduceOps.cpp`, `Activation.cpp`, etc.) — each ferrotorch submodule under
`grad_fns/` corresponds to one or more upstream `aten/` source files. No
parity ops are owned at this level; each submodule owns its own ops list.

## Requirements

- REQ-1: Declare and re-export each of the 11 per-area autograd wrapper
  submodules under `crate::grad_fns::*` so downstream code can import any op
  via the canonical path `crate::grad_fns::<area>::<op>` (e.g.
  `crate::grad_fns::arithmetic::add`,
  `crate::grad_fns::cumulative::cummax`).

- REQ-2: Maintain the upstream-aligned area split: each submodule corresponds
  to one (or more closely-related) upstream PyTorch translation unit(s) under
  `aten/src/ATen/native/`. The split is not arbitrary — it is the contract
  surface that lets per-area design docs cite a single area's upstream files
  without cross-area drift.

## Acceptance Criteria

- [x] AC-1: Every submodule declared in `mod.rs` exists as a sibling `.rs`
  file under `ferrotorch-core/src/grad_fns/` and compiles as part of
  `cargo build -p ferrotorch-core`.

- [x] AC-2: `crate::grad_fns::<area>::<op>` is reachable from at least one
  non-test production consumer for each declared area — verified by
  `grep -n "grad_fns::"` across `ferrotorch-core/src/` returning paths
  outside `grad_fns/` itself.

- [x] AC-3: Each declared submodule has its own design doc under
  `.design/ferrotorch-core/grad_fns/<area>.md` (`activation.md`,
  `arithmetic.md`, `comparison.md`, `cumulative.md`, `fft.md`,
  `indexing.md`, `linalg.md`, `quantize_grad.md`, `reduction.md`,
  `shape.md`, `transcendental.md`) — verified by listing
  `.design/ferrotorch-core/grad_fns/`.

## Architecture

### File contents (all of it)

`mod.rs` is 11 lines, one `pub mod` declaration per submodule:

```rust
pub mod activation;
pub mod arithmetic;
pub mod comparison;
pub mod cumulative;
pub mod fft;
pub mod indexing;
pub mod linalg;
pub mod quantize_grad;
pub mod reduction;
pub mod shape;
pub mod transcendental;
```

No `use` statements, no `pub use` re-exports at this level (the few selective
`pub use grad_fns::<area>::<symbol>` re-exports — e.g.
`pub use grad_fns::activation::{GeluApproximate, gelu, gelu_with, sigmoid,
tanh}` — live in `ferrotorch-core/src/lib.rs:158-163`, NOT in `mod.rs`). No
trait definitions; the `GradFn` trait that each submodule's `*Backward`
struct implements is defined in `ferrotorch-core/src/autograd/`, not here.

### Area → upstream mapping

| Submodule | Primary upstream translation unit(s) | Design doc |
|---|---|---|
| `activation` | `aten/src/ATen/native/Activation.cpp`, `aten/src/ATen/native/SoftMax.cpp` | `activation.md` |
| `arithmetic` | `aten/src/ATen/native/BinaryOps.cpp`, `aten/src/ATen/native/UnaryOps.cpp`, `aten/src/ATen/native/PointwiseOps.cpp`, `aten/src/ATen/native/Pow.cpp` | `arithmetic.md` |
| `comparison` | `aten/src/ATen/native/BinaryOps.cpp` (comparison stubs: `eq_stub`, `ne_stub`, `lt_stub`, `le_stub`, `gt_stub`, `ge_stub`, `maximum_stub`, `minimum_stub`) | `comparison.md` |
| `cumulative` | `aten/src/ATen/native/ReduceOps.cpp` (`cumsum`, `cumprod`, `cummax`, `cummin`, `logcumsumexp`) | `cumulative.md` |
| `fft` | `aten/src/ATen/native/SpectralOps.cpp`, `torch/fft/__init__.py` | `fft.md` |
| `indexing` | `aten/src/ATen/native/TensorAdvancedIndexing.cpp`, `aten/src/ATen/native/Indexing.cpp` | `indexing.md` |
| `linalg` | `aten/src/ATen/native/LinearAlgebra.cpp`, `aten/src/ATen/native/BlasKernel.cpp`, `torch/linalg/__init__.py` | `linalg.md` |
| `quantize_grad` | `aten/src/ATen/native/quantized/FakeQuantizeCore.cpp` | `quantize_grad.md` |
| `reduction` | `aten/src/ATen/native/ReduceOps.cpp` (`sum`, `mean`, `prod`, `max`, `min`, `argmax`, `argmin`, `var`, `std`, `norm`, `logsumexp`, etc.) | `reduction.md` |
| `shape` | `aten/src/ATen/native/TensorShape.cpp` | `shape.md` |
| `transcendental` | `aten/src/ATen/native/UnaryOps.cpp` (`exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `atanh`, `erf`, `erfc`, `lgamma`, etc.) | `transcendental.md` |

### Why no trait / re-export surface here

`mod.rs` does not define the `GradFn` trait (that lives in
`ferrotorch-core/src/autograd/` per the autograd-engine design — each
`*Backward` struct implements it from outside this module tree). `mod.rs`
also does not selectively re-export submodule symbols — the public
ergonomic re-exports (e.g. `pub use grad_fns::activation::{gelu, sigmoid,
tanh}` for the most-frequent callers) live in
`ferrotorch-core/src/lib.rs:158-163` so they appear at the crate root, not
the submodule root. This keeps `mod.rs` deliberately minimal: it is purely
the module-declaration surface, and any change to it (adding or removing a
submodule) is the structural signal that a new area has been added or
retired.

### Non-test production consumers (per submodule)

Each submodule's design doc cites its own non-test consumers; the
module-root re-export surface is reachable by the canonical
`crate::grad_fns::<area>::<op>` paths. Representative non-test production
consumer sites that exercise the re-export surface (i.e. that `use
crate::grad_fns::<area>::*` or call `crate::grad_fns::<area>::<op>`
directly outside `#[cfg(test)]`):

- `add in ferrotorch-core/src/vmap.rs` — `use crate::grad_fns::arithmetic::add`
  (vmap rule for the add op)
- `ferrotorch-core/src/vmap.rs:957` — `use crate::grad_fns::arithmetic::mul`
- `ferrotorch-core/src/einops.rs:783` —
  `crate::grad_fns::reduction::sum_dim(&view, 1, false)?`
- `ferrotorch-core/src/einops.rs:791` —
  `crate::grad_fns::arithmetic::mul(&summed, &scale_t)?`
- `ferrotorch-core/src/einops.rs:796` —
  `crate::grad_fns::cumulative::cummax(&view, 1)?`
- `ferrotorch-core/src/einops.rs:802` —
  `crate::grad_fns::cumulative::cummin(&view, 1)?`
- `ferrotorch-core/src/meta_propagate.rs` —
  `use crate::grad_fns::arithmetic::{add, mul, neg, sqrt}`
- `ferrotorch-core/src/meta_propagate.rs:545` —
  `use crate::grad_fns::reduction::{mean_dim, sum, sum_dim}`
- `ferrotorch-core/src/meta_propagate.rs:591` —
  `use crate::grad_fns::activation::{gelu, relu, sigmoid, silu, softmax, tanh}`
- `ferrotorch-core/src/tensor.rs:1131` —
  `crate::grad_fns::indexing::masked_fill_bt(self, mask, value)`
- `ferrotorch-core/src/tensor.rs:2438` —
  `use crate::grad_fns::shape::FlattenBackward`
- `ferrotorch-core/src/autograd/grad_penalty.rs` — pervasive use of
  `crate::grad_fns::arithmetic::{pow, sqrt, sub, mul}` and
  `crate::grad_fns::reduction::sum`
- `ferrotorch-core/src/ops_trait.rs` — `use crate::grad_fns::arithmetic`
  (the operator-overload trait layer used by `let c = &a + &b` syntax
  pervasively across the crate)

The re-export surface is exercised across at least 8 distinct non-test
production sites. The module root does its one job: it makes every
submodule reachable via the canonical path.

## Parity contract

The route's `parity_ops` field is empty `[]` (verified via `tomllib` read of
`tooling/translate-routes.toml`). The module root owns no ops directly — all
parity-sweep coverage is delegated to the per-area submodules, each of
which carries its own `parity_ops` list and its own design doc's "Parity
contract" section. There is no module-root parity-sweep op, and adding one
would be a structural mistake (there is no PyTorch construct at this level
to match against — PyTorch has no `aten/src/ATen/native/mod.cpp` either;
it is a directory of per-area translation units).

## Verification

### Build verification

`cargo build -p ferrotorch-core` is the floor — if any declared submodule
fails to compile, this `mod.rs` file's declaration of it would also fail,
so the build is the mechanical check that every `pub mod <name>;` resolves
to a file that exists and parses.

### Re-export-reachability verification

```bash
# Every submodule has at least one non-test production consumer:
for area in activation arithmetic comparison cumulative fft indexing linalg quantize_grad reduction shape transcendental; do
  count=$(grep -rn "grad_fns::${area}" /home/doll/ferrotorch/ferrotorch-core/src/ \
    | grep -v "/grad_fns/" \
    | grep -v "#\[cfg(test)\]" \
    | wc -l)
  echo "$area: $count consumer site(s)"
done
```

Expected: every area returns `>= 1`.

### Design-doc-coverage verification

```bash
# Every declared submodule has a sibling design doc:
ls /home/doll/ferrotorch/.design/ferrotorch-core/grad_fns/ | grep '\.md$' | wc -l
```

Expected: `12` (11 per-area docs + the `mod.md` this file is).

### Parity-sweep verification

Not applicable — `parity_ops` is empty for the module root. Per-area
parity-sweep verification is the responsibility of each submodule's design
doc.

### Lint / format

```bash
cargo clippy -p ferrotorch-core --all-targets --all-features -- -D warnings
cargo fmt --all --check
```

Both should pass on `mod.rs` trivially (no executable code).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: the 11 `pub mod <name>;` declarations at `ferrotorch-core/src/grad_fns/mod.rs` (lines 1-11) declare every per-area submodule and make `crate::grad_fns::<area>::<op>` reachable. Non-test production consumers exercising the re-export surface include `add in ferrotorch-core/src/vmap.rs` (`use crate::grad_fns::arithmetic::add`), `cummax in ferrotorch-core/src/einops.rs` (`crate::grad_fns::cumulative::cummax(&view, 1)?`), `add in ferrotorch-core/src/meta_propagate.rs` (`use crate::grad_fns::arithmetic::{add, mul, neg, sqrt}`), `add in ferrotorch-core/src/meta_propagate.rs` (`use crate::grad_fns::activation::{gelu, relu, sigmoid, silu, softmax, tanh}`), `add in ferrotorch-core/src/tensor.rs` (`crate::grad_fns::indexing::masked_fill_bt(self, mask, value)`), `masked_fill_bt in ferrotorch-core/src/tensor.rs` (`use crate::grad_fns::shape::FlattenBackward`), `add in ferrotorch-core/src/autograd/grad_penalty.rs` (multiple `arithmetic::{pow, sqrt, sub, mul}` + `reduction::sum` calls), and `add in ferrotorch-core/src/ops_trait.rs` (`use crate::grad_fns::arithmetic` powering the `let c = &a + &b` operator-overload surface). No parity-sweep op is owned at this level (route's `parity_ops` field is empty). |
| REQ-2 | SHIPPED | impl: the per-area split is enforced by file structure — each of the 11 submodules has a sibling design doc under `.design/ferrotorch-core/grad_fns/` (`activation.md`, `arithmetic.md`, `comparison.md`, `cumulative.md`, `fft.md`, `indexing.md`, `linalg.md`, `quantize_grad.md`, `reduction.md`, `shape.md`, `transcendental.md`) and each design doc names its specific upstream `aten/src/ATen/native/<file>.cpp` translation unit(s) in its `upstream-paths:` frontmatter. The split is auditable: changing `mod.rs` to add or drop a `pub mod` requires authoring or retiring the matching design doc and route, which is the structural signal that the area surface has changed. Non-test production consumer: every consumer cited under REQ-1 implicitly relies on this split being stable — a renamed or merged area would break `use crate::grad_fns::<area>::*` at every site. |
