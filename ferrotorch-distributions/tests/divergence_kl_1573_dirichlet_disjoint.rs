//! Critic re-audit of #1573 (commit e579271f6): a LEFT-ALONE KL pair that the
//! builder claimed was correct to leave unconverted, but which fails to
//! broadcast where torch broadcasts.
//!
//! The #1573 commit message states Dirichlet-Dirichlet was deliberately LEFT
//! ALONE ("bespoke per-row batch handling"). The builder's claim is that the
//! left-alone pairs "already produce torch-matching shapes for batched inputs".
//! That claim is FALSE for the DISJOINT batch case.
//!
//! `kl_dirichlet_dirichlet` (kl.rs:1599-1616) iterates `bi in 0..b` over p's
//! batch rows and only handles q being a single row (`qa.len() == k`, broadcast
//! to every p row) or q having exactly p's batch count. It emits
//! `p.batch_shape()`. It does NOT broadcast a disjoint p/q batch.
//!
//! torch's `_kl_dirichlet_dirichlet` (torch/distributions/kl.py:283-298)
//! operates on `broadcast_all`-aligned concentration tensors via
//! `sum_params_p`/`sum_params_q` reductions over the last (event) dim, so
//! `Dirichlet([2,1,K]) || Dirichlet([1,2,K])` broadcasts the batch dims to
//! `[2,2]` exactly like every other KL pair.
//!
//! R-CHAR-3: the expected constants are the live torch==2.11.0 float64 oracle:
//!   import torch; torch.set_default_dtype(torch.float64)
//!   from torch.distributions import Dirichlet, kl_divergence as kl
//!   pc = torch.tensor([[[1.,2.,3.]],[[2.,2.,2.]]])   # [2,1,3]
//!   qc = torch.tensor([[[2.,2.,2.],[1.,1.,1.]]])     # [1,2,3]
//!   kl(Dirichlet(pc), Dirichlet(qc))  ->  shape (2,2)
//!     [0.8068528194400547, 0.5511973816621554, 0.0, 0.24434456222210077]
//!
//! Divergence: ferrotorch's `kl_dirichlet_dirichlet` diverges from
//! `pytorch torch/distributions/kl.py:283-298` for
//! `Dirichlet([2,1,3]) || Dirichlet([1,2,3])`.
//! Upstream returns shape (2,2); ferrotorch returns shape [2,1] (truncated to
//! p's batch shape, only 2 of the 4 broadcast elements computed).
//! Tracking: #1574

use ferrotorch_core::creation::from_slice;
use ferrotorch_distributions::Dirichlet;
use ferrotorch_distributions::kl::kl_divergence;

#[test]
fn divergence_kl_1573_dirichlet_disjoint_batch_broadcast() {
    // p concentration [2,1,3], q concentration [1,2,3] -> batch broadcast [2,2].
    let p = Dirichlet::new(from_slice(&[1.0f64, 2.0, 3.0, 2.0, 2.0, 2.0], &[2, 1, 3]).unwrap())
        .unwrap();
    let q = Dirichlet::new(from_slice(&[2.0f64, 2.0, 2.0, 1.0, 1.0, 1.0], &[1, 2, 3]).unwrap())
        .unwrap();

    let out = kl_divergence(&p, &q).unwrap();
    assert_eq!(
        out.shape(),
        &[2, 2],
        "Dirichlet disjoint batch should broadcast [2,1] x [1,2] -> [2,2]; \
         ferrotorch emitted shape {:?}",
        out.shape()
    );
    let got: Vec<f64> = out.data_vec().unwrap();
    let exp: [f64; 4] = [
        0.8068528194400547,
        0.5511973816621554,
        0.0,
        0.24434456222210077,
    ];
    for (i, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-9,
            "dirichlet_disjoint[{i}]: ferrotorch={g} torch={e} delta={}",
            (g - e).abs()
        );
    }
}
