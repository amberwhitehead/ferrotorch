//! Critic audit for wave-I uncommitted builder work (#1542).
//!
//! Probes the closed-blocker claims in two uncommitted production files:
//!   - `ferrotorch-distributions/src/constraints.rs` (claims #1372:
//!     Simplex full-vector check + IntegerInterval + NonNegativeInteger)
//!   - `ferrotorch-distributions/src/dirichlet.rs` (claims #1412:
//!     N-D batched concentration + expand + arg_constraints + support)
//!
//! Each test pins one observable behaviour against the upstream PyTorch
//! contract. Tests that FAIL pin a real divergence; tests that PASS confirm
//! the claim is genuinely wired (not vocab-only).
//!
//! Expected values verified against live torch 2.11.0:
//!   Dirichlet([0.5,0.5,0.5]).mode      -> [1.0, 0.0, 0.0]   (one-hot, NOT NaN)
//!   Dirichlet([2,3,4]).mode            -> [1/6, 2/6, 3/6]
//!   Dirichlet([[..],[..]]).batch_shape -> [2], sample [2,3], log_prob [2]
//!   constraints.simplex.check([0.2,0.5,0.4]) -> False  (sum 1.1)
//!   constraints.simplex.check([-0.1,0.6,0.5]) -> False (negative element)

use ferrotorch_core::creation::{from_slice, tensor};
use ferrotorch_distributions::constraints::{
    self, Constraint, IntegerInterval, NonNegativeInteger, Simplex,
};
use ferrotorch_distributions::{Dirichlet, Distribution};

// ============================================================================
// #1372 constraints — IntegerInterval (GENUINELY WIRED expected)
// ============================================================================

/// IntegerInterval(0,5).check verifies integrality AND range.
/// Upstream `torch/distributions/constraints.py:367-369` (_IntegerInterval):
///   return (value % 1 == 0) & (lower_bound <= value) & (value <= upper_bound)
#[test]
fn audit_1372_integer_interval_check() {
    let c = IntegerInterval {
        lower_bound: 0.0f32,
        upper_bound: 5.0f32,
    };
    assert!(c.check(3.0f32), "3 in [0,5] integer must pass");
    assert!(!c.check(3.5f32), "3.5 non-integer must fail");
    assert!(!c.check(7.0f32), "7 out of [0,5] must fail");
    assert!(c.is_discrete(), "IntegerInterval must be discrete");
}

// ============================================================================
// #1372 constraints — NonNegativeInteger (GENUINELY WIRED expected)
// ============================================================================

/// NonNegativeInteger.check verifies value >= 0 AND integral.
/// Upstream `torch/distributions/constraints.py:411-412` (_IntegerGreaterThan,
/// constructed as `nonnegative_integer = _IntegerGreaterThan(0)` at :738):
///   return (value % 1 == 0) & (value >= lower_bound)
#[test]
fn audit_1372_nonnegative_integer_check() {
    let c = NonNegativeInteger;
    assert!(c.check(0.0f32), "0 must pass");
    assert!(c.check(5.0f32), "5 must pass");
    assert!(!c.check(-1.0f32), "-1 must fail");
    assert!(!c.check(2.5f32), "2.5 non-integer must fail");
    assert!(c.is_discrete(), "NonNegativeInteger must be discrete");
}

// ============================================================================
// #1372 constraints — Simplex FULL VECTOR check (DIVERGENCE expected)
// ============================================================================

/// Divergence: ferrotorch's `Simplex::check` diverges from
/// `pytorch torch/distributions/constraints.py:533` for a non-normalized
/// vector. Upstream `_Simplex.check`:
///   return torch.all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-6)
/// does a FULL vector check: all elements >= 0 AND sum ~= 1.
///
/// ferrotorch's `Constraint::check<T: Float>(&self, value: T)` takes a SCALAR,
/// so it can only verify the per-element non-negativity (`value >= 0`). It
/// cannot detect that [0.2, 0.5, 0.4] sums to 1.1 (not 1) — every element
/// passes the scalar check individually. Upstream returns False for this
/// vector; ferrotorch's surface returns True for each element.
///
/// This test demonstrates the missing full-vector check by asserting the
/// observable PyTorch contract: a vector that sums to 1.1 must be REJECTED.
/// We express "the constraint rejects this vector" as "at least one element
/// check returns false", which is the strongest statement the scalar surface
/// can make. It FAILS because all three elements are individually >= 0.
///
/// Probe input: [0.2, 0.5, 0.4] (sum = 1.1). Upstream: reject. ferrotorch: accept.
#[test]
fn audit_1372_simplex_rejects_non_normalized_vector() {
    let c = Simplex;
    let v = [0.2f32, 0.5, 0.4]; // sums to 1.1 -> upstream rejects this vector

    // Upstream `simplex.check([0.2,0.5,0.4])` == False (verified live torch).
    // The only way ferrotorch's scalar surface can reject the vector is for
    // some element check to fail. Since all elements are >= 0, the vector is
    // (incorrectly) accepted -> this assertion FAILS, pinning the divergence.
    let vector_rejected = v.iter().any(|&x| !c.check(x));
    assert!(
        vector_rejected,
        "Simplex must reject vector [0.2,0.5,0.4] (sum=1.1); upstream \
         constraints.py:533 checks (value.sum(-1)-1).abs() < 1e-6 but \
         ferrotorch's scalar Simplex::check only tests per-element >= 0"
    );
}

/// Companion probe: a properly normalized simplex vector must be ACCEPTED.
/// This one PASSES under both upstream and ferrotorch — it confirms the
/// divergence above is specifically the missing sum-to-one check, not a
/// blanket rejection.
/// Upstream: `simplex.check([0.2,0.5,0.3])` == True (verified live torch).
#[test]
fn audit_1372_simplex_accepts_normalized_vector() {
    let c = Simplex;
    let v = [0.2f32, 0.5, 0.3]; // sums to 1.0 -> accept
    let vector_accepted = v.iter().all(|&x| c.check(x));
    assert!(vector_accepted, "valid simplex vector must be accepted");
}

/// Divergence companion: a vector with a negative element AND that happens to
/// sum to 1.0 ([-0.1, 0.6, 0.5]) must be REJECTED by upstream because of the
/// negative element. The scalar surface CAN catch this one (the -0.1 element
/// fails `>= 0`), so this test PASSES — it documents that the per-element
/// non-negativity half is wired, isolating the sum-to-one half as the gap.
/// Upstream: `simplex.check([-0.1,0.6,0.5])` == False (verified live torch).
#[test]
fn audit_1372_simplex_rejects_negative_element_vector() {
    let c = Simplex;
    let v = [-0.1f32, 0.6, 0.5];
    let vector_rejected = v.iter().any(|&x| !c.check(x));
    assert!(
        vector_rejected,
        "simplex must reject vector with negative element"
    );
}

// ============================================================================
// #1412 dirichlet — N-D batched concentration (DIVERGENCE expected)
// ============================================================================

/// Divergence: ferrotorch's `Dirichlet::new` diverges from
/// `pytorch torch/distributions/dirichlet.py:65-72` for N-D concentration.
/// Upstream accepts `concentration.dim() >= 1`:
///   if concentration.dim() < 1: raise ValueError(...)
///   batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
/// A concentration of shape [B, K] yields batch_shape [B], event_shape [K],
/// sample shape [B, K], and log_prob shape [B].
///
/// ferrotorch's `Dirichlet::new` (dirichlet.rs:80-87) rejects `ndim != 1`,
/// so a batched concentration [2, 3] returns Err. Claim #1412 says N-D
/// batched concentration is shipped; it is not.
///
/// Probe: concentration shape [2, 3]. Upstream: sample shape [2, 3].
/// ferrotorch: `Dirichlet::new` returns Err.
#[test]
fn audit_1412_dirichlet_nd_batched_sample_shape() {
    let alpha = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let dist = Dirichlet::new(alpha)
        .expect("Dirichlet must accept N-D batched concentration [2,3] (dirichlet.py:65)");
    // Upstream: Dirichlet([[..],[..]]).sample().shape == [2, 3].
    let s = dist.sample(&[]).unwrap();
    assert_eq!(
        s.shape(),
        &[2, 3],
        "batched Dirichlet sample shape must be [2,3]"
    );
}

/// Divergence companion: batched log_prob shape is the batch_shape [B].
/// Upstream `dirichlet.py:90-97`: log_prob reduces the last (event) dim,
/// leaving batch_shape [2]. Probe: concentration [2,3], value [2,3].
/// Upstream: log_prob shape [2]. ferrotorch: `Dirichlet::new` Err.
#[test]
fn audit_1412_dirichlet_nd_batched_log_prob_shape() {
    let alpha = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let dist = Dirichlet::new(alpha)
        .expect("Dirichlet must accept N-D batched concentration [2,3]");
    let value = from_slice(
        &[0.2f32, 0.3, 0.5, 0.1, 0.4, 0.5],
        &[2, 3],
    )
    .unwrap();
    let lp = dist.log_prob(&value).unwrap();
    assert_eq!(lp.shape(), &[2], "batched Dirichlet log_prob shape must be [2]");
}

/// Divergence companion: batch_shape of a [B, K] Dirichlet is [B].
/// Upstream `dirichlet.py:72`: batch_shape = concentration.shape[:-1].
/// Probe: concentration [2,3] -> batch_shape [2]. ferrotorch: `new` Err.
#[test]
fn audit_1412_dirichlet_nd_batch_shape() {
    let alpha = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let dist = Dirichlet::new(alpha)
        .expect("Dirichlet must accept N-D batched concentration [2,3]");
    assert_eq!(dist.batch_shape(), vec![2], "batched Dirichlet batch_shape must be [2]");
}

// ============================================================================
// #1412 dirichlet — expand to a new batch dim (DIVERGENCE expected)
// ============================================================================

/// Divergence: ferrotorch's `Dirichlet::expand` diverges from
/// `pytorch torch/distributions/dirichlet.py:76-83`. Upstream:
///   new.concentration = self.concentration.expand(batch_shape + self.event_shape)
/// so expanding a base Dirichlet([2,3]) (K=2) to batch_shape [4] yields a
/// distribution with batch_shape [4], sample shape [4, 2].
///
/// ferrotorch's `expand` (dirichlet.rs:484-492) rejects any non-empty
/// batch_shape with InvalidArgument. Claim #1412 says expand is shipped.
///
/// Probe: base K=2, expand([4]). Upstream: sample shape [4, 2].
/// ferrotorch: `expand` returns Err.
#[test]
fn audit_1412_dirichlet_expand_to_new_batch() {
    let alpha = tensor(&[2.0f32, 3.0]).unwrap();
    let dist = Dirichlet::new(alpha).unwrap();
    let expanded = dist
        .expand(&[4])
        .expect("Dirichlet::expand([4]) must succeed (dirichlet.py:76)");
    let s = expanded.sample(&[]).unwrap();
    assert_eq!(
        s.shape(),
        &[4, 2],
        "expanded Dirichlet sample shape must be [4, 2]"
    );
}

// ============================================================================
// #1412 dirichlet — arg_constraints / support (GENUINELY WIRED expected)
// ============================================================================

/// arg_constraints returns {concentration: <positive-family>}.
/// Upstream `dirichlet.py:53-55`:
///   arg_constraints = {"concentration": constraints.independent(constraints.positive, 1)}
/// ferrotorch ships scalar `Positive` (Independent composite is #1372
/// follow-up); the key name + positivity semantics match.
#[test]
fn audit_1412_dirichlet_arg_constraints() {
    let alpha = tensor(&[2.0f32, 3.0]).unwrap();
    let dist = Dirichlet::new(alpha).unwrap();
    let ac = dist.arg_constraints();
    assert_eq!(ac.len(), 1, "arg_constraints must have one entry");
    let c = ac
        .get("concentration")
        .expect("arg_constraints must contain 'concentration'");
    // Positive-family constraint: rejects 0 and negatives at the boundary.
    assert!(
        c.name().contains("Positive") || c.name().contains("Independent"),
        "concentration constraint should be positive-family, got {}",
        c.name()
    );
}

/// support is the Simplex constraint with event_dim 1.
/// Upstream `dirichlet.py:56`: support = constraints.simplex.
#[test]
fn audit_1412_dirichlet_support_simplex() {
    let alpha = tensor(&[1.0f32, 1.0, 1.0]).unwrap();
    let dist = Dirichlet::new(alpha).unwrap();
    let sup = dist.support().expect("Dirichlet must declare support");
    assert_eq!(sup.name(), "Simplex");
    assert_eq!(sup.event_dim(), 1);
}

// ============================================================================
// #1412 dirichlet — mode all-alpha<1 (DIVERGENCE expected)
// ============================================================================

/// Divergence: ferrotorch's `Dirichlet::mode` diverges from
/// `pytorch torch/distributions/dirichlet.py:106-113` for all-alpha<1.
/// Upstream:
///   concentrationm1 = (concentration - 1).clamp(min=0.0)
///   mode = concentrationm1 / concentrationm1.sum(-1, True)
///   mask = (concentration < 1).all(dim=-1)
///   mode[mask] = one_hot(mode[mask].argmax(dim=-1), K).to(mode)
/// For alpha = [0.5, 0.5, 0.5] the clamped numerator is all-zero, argmax = 0,
/// so the mode is the ONE-HOT vector [1.0, 0.0, 0.0] (verified live torch 2.11).
///
/// ferrotorch's `mode` (dirichlet.rs:464-469) returns ALL-NaN for the
/// all-alpha<1 path instead of the upstream one-hot vector.
///
/// Probe: alpha = [0.5, 0.5, 0.5]. Upstream: [1.0, 0.0, 0.0].
/// ferrotorch: [NaN, NaN, NaN].
#[test]
fn audit_1412_dirichlet_mode_all_alpha_below_one_is_one_hot() {
    let alpha = tensor(&[0.5f32, 0.5, 0.5]).unwrap();
    let dist = Dirichlet::new(alpha).unwrap();
    let mode = dist.mode().unwrap();
    let data = mode.data().unwrap();
    // Upstream returns the one-hot [1.0, 0.0, 0.0], NOT NaN.
    assert_eq!(
        data[0], 1.0f32,
        "Dirichlet([0.5,0.5,0.5]).mode[0] must be 1.0 (one-hot, dirichlet.py:110-112), got {}",
        data[0]
    );
    assert_eq!(data[1], 0.0f32, "mode[1] must be 0.0, got {}", data[1]);
    assert_eq!(data[2], 0.0f32, "mode[2] must be 0.0, got {}", data[2]);
}

/// mode for alpha>1 (GENUINELY WIRED expected): (alpha-1)/sum(alpha-1).
/// Upstream `dirichlet.py:107-108`. alpha=[2,3,4] -> [1,2,3]/6.
#[test]
fn audit_1412_dirichlet_mode_alpha_gt_one() {
    let alpha = tensor(&[2.0f32, 3.0, 4.0]).unwrap();
    let dist = Dirichlet::new(alpha).unwrap();
    let mode = dist.mode().unwrap();
    let data = mode.data().unwrap();
    assert!((data[0] - 1.0 / 6.0).abs() < 1e-6);
    assert!((data[1] - 2.0 / 6.0).abs() < 1e-6);
    assert!((data[2] - 3.0 / 6.0).abs() < 1e-6);
}

// Reference the `constraints` module to keep the import meaningful even if
// the convenience constructors change.
#[test]
fn audit_1372_constraints_module_constructors_exist() {
    let _ = constraints::simplex();
    let _ = constraints::integer_interval(0.0f32, 5.0f32);
    let _ = constraints::nonnegative_integer();
}
