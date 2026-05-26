//! Kernel-layer op-module declarations. Mirrors `aten/src/ATen/native/`'s
//! directory-as-namespace convention. Each declared sub-module is the
//! forward-only (no autograd) op family for its area; the autograd
//! wrappers live in `ferrotorch-core/src/grad_fns/`.
//!
//! ## REQ status (per `.design/ferrotorch-core/ops/mod.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (9 sub-modules) | SHIPPED | 9 `pub mod` declarations at `ops/mod.rs:1-9`; consumer `grad_fns/cumulative.rs:32` (`use crate::ops::cumulative::{...}`), `grad_fns/transcendental.rs:15` (`use crate::ops::elementwise::{fast_cos, fast_sin, unary_map}`), `tensor.rs:1146` (`crate::ops::indexing::masked_select`) |
//! | REQ-2 (kernel/autograd split) | SHIPPED | the kernel-layer `ops::<family>` vs autograd-layer `grad_fns::<family>` split IS the organizational primitive; consumer `grad_fns/cumulative.rs:32-35` imports from `crate::ops::cumulative` then `pub fn cumsum` at `grad_fns/cumulative.rs:104` delegates the forward to `ops::cumulative::cumsum_forward(...)` — mirrors upstream `aten::cummax` (user) vs `_cummax_helper` (private) split 1:1 |
//! | REQ-3 (no module-level re-exports) | SHIPPED | this file has zero `pub use` (mechanical: 9 `pub mod` lines only); consumer `lib.rs:173-177` `pub use ops::indexing::{gather, masked_select, scatter, ...}` lifts specific symbols — the picking-by-symbol pattern requires the sub-modules NOT pre-re-export, which mod.rs preserves by being a pure-declaration file |

pub mod cumulative;
pub mod elementwise;
pub mod higher_order;
pub mod indexing;
pub mod linalg;
pub mod phase2c;
pub mod scatter;
pub mod search;
pub mod tensor_ops;
