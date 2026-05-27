//! Divergence-coverage tests for #1233 audit (commit `8572b0c38`).
//!
//! The commit "ferrotorch-core: cumulative — accept 0-D scalar inputs"
//! routes 0-D inputs through `cumulative_scalar_identity` /
//! `cumextreme_scalar_identity` and constructs an error string for invalid
//! `dim` values. The builder's claim:
//!
//! > "dim validation on 0-D: dim in {-1, 0} accepted, anything else
//! > returns FerrotorchError::InvalidArgument with the upstream error
//! > phrasing 'expected reduction dim -1 or 0 for scalar but got <N>'
//! > (live-verified torch 2.11.0 cummax message)."
//!
//! Two divergences:
//!
//! 1. **Wrong phrasing for cumsum/cumprod/logcumsumexp.** Upstream does NOT
//!    route these through `zero_numel_check_dims`; they go through
//!    `c10::maybe_wrap_dim` (see `aten/src/ATen/native/ReduceOps.cpp:506`
//!    via `impl_func_cum_ops`, and `c10/core/WrapDimMinimal.cpp:24-31`),
//!    which emits:
//!
//!        "Dimension out of range (expected to be in range of [<min>,
//!         <max>], but got <dim>)"
//!
//!    ferrotorch instead emits "expected reduction dim -1 or 0 for scalar
//!    but got <N>" (the cummax-style message) for all five ops.
//!
//! 2. **Wrong capitalisation even for cummax/cummin.** Upstream's
//!    `zero_numel_check_dims` at
//!    `aten/src/ATen/native/ReduceOpsUtils.h:280` emits
//!
//!        "<op>(): Expected reduction dim -1 or 0 for scalar but got <N>"
//!
//!    Capital 'E' on "Expected", and a `()` suffix on the op name.
//!    ferrotorch emits "cummax(): expected reduction dim -1 or 0 for
//!    scalar but got 1" (lowercase 'e').
//!
//! Live-verified torch 2.11.0+cu130 on 2026-05-25:
//!
//! ```text
//! >>> torch.cumsum(torch.tensor(5.0), 1)
//! IndexError: Dimension out of range (expected to be in range of [-1, 0],
//!             but got 1)
//! >>> torch.cumprod(torch.tensor(5.0), 100)
//! IndexError: Dimension out of range (expected to be in range of [-1, 0],
//!             but got 100)
//! >>> torch.logcumsumexp(torch.tensor(5.0), -2)
//! IndexError: Dimension out of range (expected to be in range of [-1, 0],
//!             but got -2)
//! >>> torch.cummax(torch.tensor(5.0), 1)
//! IndexError: cummax(): Expected reduction dim -1 or 0 for scalar but got 1
//! >>> torch.cummin(torch.tensor(5.0), -2)
//! IndexError: cummin(): Expected reduction dim -1 or 0 for scalar but got -2
//! ```
//!
//! These tests FAIL against `8572b0c38` because the message strings do
//! not contain the upstream key phrases.

use ferrotorch_core::{FerrotorchError, cummax, cummin, cumprod, cumsum, from_vec, logcumsumexp};

/// Named typed constants traceable to the upstream PyTorch source.
/// Per R-CHAR-3: the expected substring is sourced from
/// `aten/src/ATen/native/ReduceOpsUtils.h:280` and
/// `c10/core/WrapDimMinimal.cpp:24-31`, not copied from ferrotorch.
mod upstream_phrases {
    /// `c10::detail::maybe_wrap_dim_slow` at
    /// `c10/core/WrapDimMinimal.cpp:24-31`. Emitted for cumsum / cumprod
    /// / logcumsumexp on 0-D when `dim ∉ {-1, 0}`.
    pub const DIMENSION_OUT_OF_RANGE: &str = "Dimension out of range";

    /// `zero_numel_check_dims` at
    /// `aten/src/ATen/native/ReduceOpsUtils.h:280`. Emitted for cummax /
    /// cummin on 0-D when `dim ∉ {-1, 0}`. Capital 'E' on "Expected" and
    /// `()` suffix on the op name.
    pub const EXPECTED_REDUCTION_DIM_CAP_E: &str = "Expected reduction dim -1 or 0 for scalar";
    pub const CUMMAX_PAREN_PREFIX: &str = "cummax():";
    pub const CUMMIN_PAREN_PREFIX: &str = "cummin():";
}

fn scalar(v: f64) -> ferrotorch_core::Tensor<f64> {
    from_vec(vec![v], &[]).expect("0-D tensor construction")
}

fn err_message(e: FerrotorchError) -> String {
    e.to_string()
}

// ---------------------------------------------------------------------------
// Divergence 1: cumsum/cumprod/logcumsumexp use the WRONG family of error
// strings on bad 0-D dim. Upstream is `c10::maybe_wrap_dim`'s
// "Dimension out of range" phrasing; ferrotorch routes through a hand-rolled
// cummax-style message.
// ---------------------------------------------------------------------------

#[test]
fn divergence_cumsum_scalar_bad_dim_uses_wrap_dim_phrasing() {
    let x = scalar(5.0);
    let err = cumsum(&x, 100).expect_err("cumsum(tensor(5.0), 100) must error");
    let msg = err_message(err);
    assert!(
        msg.contains(upstream_phrases::DIMENSION_OUT_OF_RANGE),
        "cumsum 0-D bad-dim error must mirror c10::maybe_wrap_dim_slow at \
         c10/core/WrapDimMinimal.cpp:24-31 ('Dimension out of range'); \
         ferrotorch emitted: {:?}",
        msg
    );
}

#[test]
fn divergence_cumprod_scalar_bad_dim_uses_wrap_dim_phrasing() {
    let x = scalar(5.0);
    let err = cumprod(&x, 1).expect_err("cumprod(tensor(5.0), 1) must error");
    let msg = err_message(err);
    assert!(
        msg.contains(upstream_phrases::DIMENSION_OUT_OF_RANGE),
        "cumprod 0-D bad-dim error must mirror c10::maybe_wrap_dim_slow at \
         c10/core/WrapDimMinimal.cpp:24-31 ('Dimension out of range'); \
         ferrotorch emitted: {:?}",
        msg
    );
}

#[test]
fn divergence_logcumsumexp_scalar_bad_dim_uses_wrap_dim_phrasing() {
    let x = scalar(5.0);
    let err = logcumsumexp(&x, -2).expect_err("logcumsumexp(tensor(5.0), -2) must error");
    let msg = err_message(err);
    assert!(
        msg.contains(upstream_phrases::DIMENSION_OUT_OF_RANGE),
        "logcumsumexp 0-D bad-dim error must mirror c10::maybe_wrap_dim_slow \
         at c10/core/WrapDimMinimal.cpp:24-31 ('Dimension out of range'); \
         ferrotorch emitted: {:?}",
        msg
    );
}

// ---------------------------------------------------------------------------
// Divergence 2: cummax/cummin use lowercase 'e' instead of upstream's
// capital 'E'. Also the `()` suffix on op name is correct in ferrotorch.
// ---------------------------------------------------------------------------

#[test]
fn divergence_cummax_scalar_bad_dim_capital_e_phrasing() {
    let x = scalar(5.0);
    let err = cummax(&x, 1).expect_err("cummax(tensor(5.0), 1) must error");
    let msg = err_message(err);
    assert!(
        msg.contains(upstream_phrases::CUMMAX_PAREN_PREFIX),
        "cummax 0-D bad-dim error must start with 'cummax():' per upstream; \
         ferrotorch emitted: {:?}",
        msg
    );
    assert!(
        msg.contains(upstream_phrases::EXPECTED_REDUCTION_DIM_CAP_E),
        "cummax 0-D bad-dim error must contain 'Expected reduction dim -1 \
         or 0 for scalar' (capital E) per ReduceOpsUtils.h:280; \
         ferrotorch emitted: {:?}",
        msg
    );
}

#[test]
fn divergence_cummin_scalar_bad_dim_capital_e_phrasing() {
    let x = scalar(5.0);
    let err = cummin(&x, -2).expect_err("cummin(tensor(5.0), -2) must error");
    let msg = err_message(err);
    assert!(
        msg.contains(upstream_phrases::CUMMIN_PAREN_PREFIX),
        "cummin 0-D bad-dim error must start with 'cummin():' per upstream; \
         ferrotorch emitted: {:?}",
        msg
    );
    assert!(
        msg.contains(upstream_phrases::EXPECTED_REDUCTION_DIM_CAP_E),
        "cummin 0-D bad-dim error must contain 'Expected reduction dim -1 \
         or 0 for scalar' (capital E) per ReduceOpsUtils.h:280; \
         ferrotorch emitted: {:?}",
        msg
    );
}
