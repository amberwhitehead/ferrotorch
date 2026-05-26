//! Module-root dispatch for the autograd-tracking wrapper layer.
//!
//! No functions, types, or constants live here — only `pub mod` declarations
//! re-exporting the 11 per-area submodules. Each submodule mirrors a
//! `aten/src/ATen/native/<file>.cpp` translation unit (or closely related
//! cluster). The split is the contract surface — adding or retiring a `pub
//! mod` declaration is the structural signal that an area surface has
//! changed. No `parity_ops` are owned at this level.
//!
//! ## REQ status (per `.design/ferrotorch-core/grad_fns/mod.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (declare 11 area submodules) | SHIPPED | the 11 `pub mod` lines below declare every per-area submodule and make `crate::grad_fns::<area>::<op>` reachable; non-test consumers include `crate::grad_fns::arithmetic::add` in `vmap.rs`, `crate::grad_fns::cumulative::cummax` in `einops.rs`, `crate::grad_fns::activation::{gelu, relu, sigmoid, silu, softmax, tanh}` in `meta_propagate.rs`, `crate::grad_fns::indexing::masked_fill_bt` in `tensor.rs`, `crate::grad_fns::shape::FlattenBackward` in `tensor.rs`, and `crate::grad_fns::arithmetic` in `ops_trait.rs` powering the operator-overload surface. |
//! | REQ-2 (upstream-aligned area split) | SHIPPED | each submodule has a sibling design doc under `.design/ferrotorch-core/grad_fns/` naming its upstream `aten/src/ATen/native/<file>.cpp` translation unit(s) in the `upstream-paths:` frontmatter; the split is auditable through file structure. |

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
