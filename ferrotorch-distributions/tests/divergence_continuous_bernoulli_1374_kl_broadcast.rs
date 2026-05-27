//! Critic re-audit of commit c5cb6529d (#1374 CB sub-part) — KL broadcast
//! semantics: incompatible batch shapes must ERROR (not silently cycle), and
//! multi-dim right-aligned broadcast `[2,1]` vs `[1,3]` -> `[2,3]` must match
//! torch elementwise.
//!
//! Expected values from live torch 2.11.0+cu130 (2026-05-27). R-CHAR-3.

use ferrotorch_core::creation::from_slice;
use ferrotorch_distributions::ContinuousBernoulli;
use ferrotorch_distributions::kl::kl_divergence;

#[test]
fn divergence_kl_cb_incompatible_shapes_error() {
    // torch raises RuntimeError for non-broadcastable [2] vs [3].
    let p = ContinuousBernoulli::new(from_slice(&[0.3f64, 0.5], &[2]).unwrap()).unwrap();
    let q = ContinuousBernoulli::new(from_slice(&[0.6f64, 0.4, 0.2], &[3]).unwrap()).unwrap();
    let res = kl_divergence(&p, &q);
    assert!(
        res.is_err(),
        "KL(CB[2],CB[3]) must error (torch RuntimeError), got Ok with shape {:?}",
        res.ok().map(|t| t.shape().to_vec())
    );
}

#[test]
fn divergence_kl_cb_2d_broadcast() {
    // torch: KL(CB([[0.3],[0.5]]), CB([[0.6,0.4,0.2]])) -> shape [2,3]:
    //   row0 = [0.06451926445321665, 0.00793458221874696, 0.011484897164652397]
    //   row1 = [0.006840721103852643, 0.006840721103852643, 0.07883084812988328]
    let p = ContinuousBernoulli::new(from_slice(&[0.3f64, 0.5], &[2, 1]).unwrap()).unwrap();
    let q = ContinuousBernoulli::new(from_slice(&[0.6f64, 0.4, 0.2], &[1, 3]).unwrap()).unwrap();
    let kl = kl_divergence(&p, &q).unwrap();
    assert_eq!(kl.shape(), &[2, 3], "broadcast shape");
    let d = kl.data().unwrap();
    let want = [
        0.064_519_264_453_216_65,
        0.007_934_582_218_746_96,
        0.011_484_897_164_652_397,
        0.006_840_721_103_852_643,
        0.006_840_721_103_852_643,
        0.078_830_848_129_883_28,
    ];
    for (i, w) in want.iter().enumerate() {
        assert!(
            (d[i] - w).abs() <= 1e-12,
            "KL 2d bcast [{i}]: ferrotorch={} torch={w}",
            d[i]
        );
    }
}
