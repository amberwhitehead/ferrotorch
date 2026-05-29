//! RE-AUDIT regression guard for commit b70d757565 (#1654 fix) in
//! `ferrotorch-core/src/special.rs`.
//!
//! The fix changed the `special_gpu_binary` guard so broadcast / mixed-device
//! CUDA `zeta` pairs return `Err(NotImplementedOnCuda)` instead of leaking
//! `GpuTensorNotAccessible` via the dead CPU `binary_map` path. The guard
//! condition is:
//!
//! ```ignore
//! if x.is_cuda() != q.is_cuda() || x.shape() != q.shape() {
//!     return Err(FerrotorchError::NotImplementedOnCuda { op });
//! }
//! ```
//!
//! # Over-rejection risk (the critical regression the fix must NOT introduce)
//!
//! That guard fires on `x.shape() != q.shape()` REGARDLESS of device. If it
//! were reached for all-CPU inputs, every CPU broadcasting `zeta` call (e.g.
//! `zeta(CPU[2,3], CPU[3])`) would be wrongly rejected with
//! `NotImplementedOnCuda` instead of broadcasting on the host and matching
//! torch.
//!
//! It is NOT reached for all-CPU inputs: `special_gpu_binary` returns
//! `Ok(None)` at its FIRST guard (`special.rs:2587-2589`,
//! `if !x.is_cuda() && !q.is_cuda() { return Ok(None); }`) BEFORE the
//! `x.shape() != q.shape()` device-mixed/broadcast guard at `special.rs:2602`.
//! `zeta` (`special.rs:2305`) then falls through to
//! `binary_map(input, other, zeta_scalar)`, which broadcasts on the host
//! (`ops/elementwise.rs:889`). These tests PIN that all-CPU broadcast is NOT
//! over-rejected and still matches torch.
//!
//! Oracle values are LIVE torch 2.11.0+cu130 (R-CHAR-3), printed under
//! `LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH" python3`:
//!
//! ```text
//! import torch
//! x = torch.tensor([[2.,3.,4.],[1.5,2.5,5.]]); q = torch.tensor([1.,2.,1.])
//! torch.special.zeta(x, q).flatten().tolist() =
//!   [1.644934058189392, 0.2020568996667862, 1.0823231935501099,
//!    2.612375259399414, 0.34148725867271423, 1.0369277000427246]   shape [2,3]
//! ```

use ferrotorch_core::{Tensor, TensorStorage, special};

fn cpu(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Over-rejection guard (the key probe): all-CPU rank-2 broadcasting
/// `zeta(CPU[2,3], CPU[3])` must STILL broadcast on the host and match torch,
/// NOT hit the `NotImplementedOnCuda` guard. Confirms the all-CPU path
/// short-circuits at `special.rs:2588` (`Ok(None)`) before reaching the
/// device-mixed/broadcast reject at `special.rs:2602`.
///
/// Upstream: `torch.special.zeta([[2,3,4],[1.5,2.5,5]], [1,2,1])` -> shape
/// [2,3] = [1.644934058, 0.202056900, 1.082323194, 2.612375259, 0.341487259,
/// 1.036927700] (torch 2.11.0).
#[test]
fn cpu_broadcast_2x3_3_not_over_rejected() {
    let x = cpu(&[2.0, 3.0, 4.0, 1.5, 2.5, 5.0], &[2, 3]);
    let q = cpu(&[1.0, 2.0, 1.0], &[3]);

    let out = special::zeta(&x, &q)
        .expect("all-CPU broadcast zeta must NOT be rejected with NotImplementedOnCuda");

    assert_eq!(
        out.shape(),
        &[2, 3],
        "all-CPU broadcast zeta must broadcast to [2,3] like torch"
    );
    assert!(
        !out.is_cuda(),
        "all-CPU zeta result must stay on CPU (no device promotion)"
    );

    let d = out.data().unwrap();
    // Live torch 2.11.0 oracle (f64).
    let want = [
        1.644_934_058_189_392,
        0.202_056_899_666_786_2,
        1.082_323_193_550_109_9,
        2.612_375_259_399_414,
        0.341_487_258_672_714_23,
        1.036_927_700_042_724_6,
    ];
    for i in 0..6 {
        assert!(
            (d[i] - want[i]).abs() <= 1e-4 * (1.0 + want[i].abs()),
            "cpu broadcast zeta idx {i}: got {} want {} (torch 2.11)",
            d[i],
            want[i]
        );
    }
}

/// Over-rejection guard: all-CPU rank-1 broadcasting `zeta(CPU[3], CPU[1])`
/// (the CPU analogue of the pinned CUDA `divergence_zeta_broadcast_cuda_3_1`)
/// must broadcast on the host, NOT reject.
///
/// Upstream: `torch.special.zeta([2.,3.,4.], [1.])` -> [3]
/// = [1.6449340668482264, 1.2020569031595942, 1.0823232337111381] (torch f64).
#[test]
fn cpu_broadcast_3_1_not_over_rejected() {
    let x = cpu(&[2.0, 3.0, 4.0], &[3]);
    let q = cpu(&[1.0], &[1]);

    let out = special::zeta(&x, &q)
        .expect("all-CPU [3]+[1] broadcast zeta must NOT be rejected with NotImplementedOnCuda");
    assert_eq!(out.shape(), &[3], "zeta [3]+[1] broadcast -> [3] like torch");
    assert!(!out.is_cuda());

    let d = out.data().unwrap();
    // zeta(_, 1) == Riemann zeta: zeta(2)=pi^2/6, zeta(4)=pi^4/90 (symbolic,
    // R-CHAR-3 (b)); zeta(3) is Apery's constant (live torch).
    let want = [
        std::f64::consts::PI * std::f64::consts::PI / 6.0,
        1.202_056_903_159_594_2,
        std::f64::consts::PI.powi(4) / 90.0,
    ];
    for i in 0..3 {
        assert!(
            (d[i] - want[i]).abs() <= 1e-9 * (1.0 + want[i].abs()),
            "cpu broadcast [3]+[1] zeta idx {i}: got {} want {}",
            d[i],
            want[i]
        );
    }
}

/// Same-shape same-device CPU `zeta` still works (the fix left this path
/// untouched). `zeta(2,1) == pi^2/6` (Basel), symbolic constant per
/// R-CHAR-3 (b).
#[test]
fn cpu_same_shape_still_works() {
    let out = special::zeta(&cpu(&[2.0], &[1]), &cpu(&[1.0], &[1])).expect("same-shape CPU zeta");
    assert_eq!(out.shape(), &[1]);
    assert!(!out.is_cuda());
    let got = out.data().unwrap()[0];
    let want = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    assert!(
        (got - want).abs() <= 1e-12 * (1.0 + want.abs()),
        "zeta(2,1) got {got} want pi^2/6 = {want}"
    );
}
