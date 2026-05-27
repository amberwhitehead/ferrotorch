//! Divergence audit of commit `a9c160511` (#1374, Geometric distribution +
//! Geometric-Geometric KL).
//!
//! The Geometric *distribution* methods (`log_prob`/`sample`/`mean`/`variance`/
//! `entropy`) were correctly written with right-aligned broadcast indexing
//! (`row_major_strides` + `broadcast_flat_index`), so they do NOT repeat the
//! #1569 batch-broadcast bug — their committed unit tests match live torch.
//!
//! BUT the new `kl_geometric_geometric` helper in `kl.rs` repeats the exact
//! #1569 bug class: it does
//! `p_probs.iter().zip(q_probs.iter().cycle())` and emits a tensor of shape
//! `p.probs().shape()`. This:
//!   1. truncates the output to `len(p_probs)` when `q` is the larger/longer
//!      operand (so `KL(scalar, batched)` silently drops elements), and
//!   2. never broadcasts disjoint batch dims (e.g. `p:[2,1]` vs `q:[1,3]`),
//!      whereas torch broadcasts to `[2,3]`.
//!
//! `torch.distributions.kl_divergence` broadcasts `p` and `q` element-wise via
//! the standard `broadcast_all` machinery inside each distribution's `entropy`/
//! `logits`/`probs` ops and the `log1p(-q.probs)/p.probs` arithmetic. See
//! `torch/distributions/kl.py:320-322` `_kl_geometric_geometric`:
//! `-p.entropy() - torch.log1p(-q.probs)/p.probs - q.logits`.
//!
//! Reference values/shapes from live `torch.distributions.kl_divergence` at
//! float64 (torch 2.11.0+cu130, this machine, 2026-05-27). Non-tautological per
//! R-CHAR-3: every expected value is a live-torch oracle output, never copied
//! from the ferrotorch side.
//!
//! Tracking: #1572 (blocker). The two divergence tests are `#[ignore]`d so the
//! whole-crate `cargo test --tests` stays green until the generator fixes the
//! helper; they become permanent regression coverage once the fix lands (the
//! binomial #1569 precedent in `divergence_binomial_3d1dd0881_batch_shape.rs`).

use ferrotorch_core::creation::{from_slice, scalar};
use ferrotorch_distributions::Geometric;
use ferrotorch_distributions::kl::kl_divergence;

/// Divergence: ferrotorch's `kl_geometric_geometric`
/// (`ferrotorch-distributions/src/kl.rs:2493-2542`) diverges from
/// `pytorch torch/distributions/kl.py:320-322` when `p` is scalar and `q` is
/// batched.
///
/// torch:  `kl_divergence(Geometric(0.3), Geometric([0.5, 0.2]))`
///   broadcasts -> shape `[2]` = `[0.27427626168350594, 0.09389185865094474]`.
/// ferrotorch: `p_probs` has length 1, so `zip(...cycle())` yields exactly one
///   pair and the output is shape `p.probs().shape()` = `[]` (scalar) holding
///   only the first element — `q.probs()[1]` is never visited.
/// Upstream returns `[2]`; ferrotorch returns `[]`.
/// Tracking: #1572
#[test]
fn divergence_kl_geometric_scalar_p_batched_q() {
    let p = Geometric::new(scalar(0.3f64).unwrap()).unwrap();
    let q = Geometric::new(from_slice(&[0.5f64, 0.2], &[2]).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap();

    // torch broadcasts the scalar p against the batched q -> shape [2].
    assert_eq!(
        kl.shape(),
        &[2],
        "torch broadcasts scalar p against batched q -> shape [2]"
    );
    let d = kl.data().unwrap();
    // Live torch 2.11 f64 oracle: [0.27427626168350594, 0.09389185865094474].
    assert!(
        (d[0] - 0.274_276_261_683_505_94).abs() < 1e-12,
        "KL(0.3 || 0.5) should be torch's 0.27427626168350594, got {}",
        d[0]
    );
    assert!(
        (d[1] - 0.093_891_858_650_944_74).abs() < 1e-12,
        "KL(0.3 || 0.2) should be torch's 0.09389185865094474, got {}",
        d[1]
    );
}

/// Divergence: ferrotorch's `kl_geometric_geometric` diverges from
/// `pytorch torch/distributions/kl.py:320-322` for disjoint broadcastable
/// batch shapes `p:[2,1]` vs `q:[1,3]`.
///
/// torch:  broadcasts to `[2,3]`.
/// ferrotorch: outputs `p.probs().shape()` = `[2,1]` and uses `cycle()` over
///   `q` element-wise against `p`, so both the shape AND the values disagree.
/// Upstream returns `[2,3]`; ferrotorch returns `[2,1]`.
/// Tracking: #1572
#[test]
fn divergence_kl_geometric_disjoint_2d_broadcast() {
    let p = Geometric::new(from_slice(&[0.3f64, 0.6], &[2, 1]).unwrap()).unwrap();
    let q = Geometric::new(from_slice(&[0.5f64, 0.2, 0.4], &[1, 3]).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap();

    // torch broadcasts [2,1] x [1,3] -> [2,3].
    assert_eq!(
        kl.shape(),
        &[2, 3],
        "torch broadcasts p:[2,1] q:[1,3] -> [2,3]"
    );
    let d = kl.data().unwrap();
    // Live torch 2.11 f64 oracle (row-major):
    // [[0.27427626168350594, 0.09389185865094474, 0.07200284714515492],
    //  [0.0335591892511482,  0.6365141682948129,  0.13515503603605483]]
    let expected = [
        0.274_276_261_683_505_94,
        0.093_891_858_650_944_74,
        0.072_002_847_145_154_92,
        0.033_559_189_251_148_2,
        0.636_514_168_294_812_9,
        0.135_155_036_036_054_83,
    ];
    for (i, e) in expected.iter().enumerate() {
        assert!(
            (d[i] - e).abs() < 1e-12,
            "KL[{i}] should be torch's {e}, got {}",
            d[i]
        );
    }
}

/// Regression guard (NOT a divergence): the well-formed elementwise `[2] x [2]`
/// KL path currently works and matches live torch. Kept un-ignored so a future
/// broadcast fix for #1572 does not regress the path that already works.
///
/// torch: `kl_divergence(Geometric([0.3,0.6]), Geometric([0.5,0.2]))`
///   == `[0.27427626168350594, 0.6365141682948129]`.
#[test]
fn kl_geometric_elementwise_2x2_matches_torch() {
    let p = Geometric::new(from_slice(&[0.3f64, 0.6], &[2]).unwrap()).unwrap();
    let q = Geometric::new(from_slice(&[0.5f64, 0.2], &[2]).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap();
    assert_eq!(kl.shape(), &[2]);
    let d = kl.data().unwrap();
    // Live torch: [0.27427626168350594, 0.6365141682948129].
    assert!(
        (d[0] - 0.274_276_261_683_505_94).abs() < 1e-12,
        "got {}",
        d[0]
    );
    assert!(
        (d[1] - 0.636_514_168_294_812_9).abs() < 1e-12,
        "got {}",
        d[1]
    );
}
