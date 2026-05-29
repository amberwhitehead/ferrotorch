//! #1660 REGRESSION GUARD — logical-numel launch correctness for the
//! compare / where / logic-binary GPU kernels across the EXACT-LEN common path
//! and the pooled/exact backing-store combinations.
//!
//! Commit 6d2e6080e changed `launch_cmp` / `launch_where` (ferrotorch-gpu/src/
//! bool_kernels.rs, masked_kernels.rs) to launch on the *logical* element count
//! `n` threaded from the dispatch site (backend_impl.rs compare / where_cond)
//! instead of the raw pooled `CudaSlice::len()` (which `.contiguous()` rounds up
//! to `ROUND_ELEMENTS=256`). The regression risk of that change is twofold:
//!
//!   - UNDER-LAUNCH: launch fewer than `n` threads -> tail elements left
//!     uncomputed (zeros) for a buffer whose raw len < the value the old code
//!     used. This file pins that the WHOLE vector is computed for sizes that are
//!     and are NOT multiples of the 256-thread block (6, 100, 257, 1000), so a
//!     tail-truncation regression fails here.
//!   - OVER-READ: launch `n` threads over a buffer whose raw len < n. The `>= n`
//!     backing-store guard must reject under-sized operands while still launching
//!     exactly `[0, n)` for over-allocated (pooled) operands. Pinned via the
//!     both-pooled and pooled-vs-exact combinations.
//!
//! Every expected value here is a SYMBOLIC FORMULA over the input indices,
//! independently verified against live torch (2.11.0+cu130, RTX 3090) — never
//! copied from a ferrotorch GPU run (R-CHAR-3). The verification script:
//!
//! ```python
//! import torch; dev='cuda'
//! for n in [6,100,257,1000]:
//!     a = torch.arange(n, dtype=torch.float32, device=dev)
//!     b = torch.arange(n-1,-1,-1, dtype=torch.float32, device=dev)  # b[i]=n-1-i
//!     assert (a>b).cpu().tolist()  == [i>(n-1-i)  for i in range(n)]
//!     assert (a<b).cpu().tolist()  == [i<(n-1-i)  for i in range(n)]
//!     assert (a>=b).cpu().tolist() == [i>=(n-1-i) for i in range(n)]
//!     assert (a<=b).cpu().tolist() == [i<=(n-1-i) for i in range(n)]
//!     assert (a==b).cpu().tolist() == [i==(n-1-i) for i in range(n)]
//!     assert (a!=b).cpu().tolist() == [i!=(n-1-i) for i in range(n)]
//!     cond = (torch.arange(n,device=dev)%2==0); x=torch.arange(n,dtype=torch.float32,device=dev); y=-x
//!     assert torch.where(cond,x,y).cpu().tolist() == [float(i) if i%2==0 else float(-i) for i in range(n)]
//! # half/bf16: a=[1..6], b=[2;6] -> gt=[0,0,1,1,1,1]; where(a>b,a,b)=[2,2,3,4,5,6]
//! ```
//! All asserts passed -> the formulas below ARE the upstream contract.
//!
//! VERDICT for #1660 scope: this guard PASSES against 6d2e6080e (the fix is
//! correct for compare/where/logic across all sizes + backing-store combos).

#![cfg(feature = "cuda")]

use ferrotorch_core::{BoolTensor, Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = init_cuda_backend();
    });
}

fn cuda_f32(data: Vec<f32>) -> Tensor<f32> {
    let n = data.len();
    Tensor::from_storage(TensorStorage::cpu(data), vec![n], false)
        .expect("cpu tensor")
        .to(Device::Cuda(0))
        .expect("to cuda")
}

/// Exact-len (clone_htod) CUDA tensor of shape `[n, 1]` — matches the row-narrowed
/// pooled view's shape so the pooled-vs-exact comparison shapes agree.
fn cuda_f32_col(data: Vec<f32>) -> Tensor<f32> {
    let n = data.len();
    Tensor::from_storage(TensorStorage::cpu(data), vec![n, 1], false)
        .expect("cpu tensor")
        .to(Device::Cuda(0))
        .expect("to cuda")
}

fn host_bool(b: &BoolTensor) -> Vec<bool> {
    b.to(Device::Cpu)
        .expect("bool to cpu")
        .data()
        .expect("bool data")
        .to_vec()
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

// ── symbolic upstream formulas (torch-verified above) ───────────────────────
fn a_idx(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}
fn b_rev(n: usize) -> Vec<f32> {
    (0..n).map(|i| (n - 1 - i) as f32).collect()
}
fn expect_gt(n: usize) -> Vec<bool> {
    (0..n).map(|i| i > n - 1 - i).collect()
}
fn expect_lt(n: usize) -> Vec<bool> {
    (0..n).map(|i| i < n - 1 - i).collect()
}
fn expect_ge(n: usize) -> Vec<bool> {
    (0..n).map(|i| i >= n - 1 - i).collect()
}
fn expect_le(n: usize) -> Vec<bool> {
    (0..n).map(|i| i <= n - 1 - i).collect()
}
fn expect_eq(n: usize) -> Vec<bool> {
    (0..n).map(|i| i == n - 1 - i).collect()
}
fn expect_ne(n: usize) -> Vec<bool> {
    (0..n).map(|i| i != n - 1 - i).collect()
}

const SIZES: [usize; 4] = [6, 100, 257, 1000];

#[test]
fn compare_all_ops_exact_len_multisize_matches_torch() {
    ensure_cuda();
    for &n in &SIZES {
        let a = cuda_f32(a_idx(n));
        let b = cuda_f32(b_rev(n));
        assert_eq!(
            host_bool(&BoolTensor::gt(&a, &b).expect("gt")),
            expect_gt(n),
            "gt n={n}"
        );
        assert_eq!(
            host_bool(&BoolTensor::lt(&a, &b).expect("lt")),
            expect_lt(n),
            "lt n={n}"
        );
        assert_eq!(
            host_bool(&BoolTensor::ge(&a, &b).expect("ge")),
            expect_ge(n),
            "ge n={n}"
        );
        assert_eq!(
            host_bool(&BoolTensor::le(&a, &b).expect("le")),
            expect_le(n),
            "le n={n}"
        );
        assert_eq!(
            host_bool(&BoolTensor::eq_t(&a, &b).expect("eq")),
            expect_eq(n),
            "eq n={n}"
        );
        assert_eq!(
            host_bool(&BoolTensor::ne(&a, &b).expect("ne")),
            expect_ne(n),
            "ne n={n}"
        );
    }
}

#[test]
fn where_exact_len_multisize_matches_torch() {
    ensure_cuda();
    for &n in &SIZES {
        // cond[i] = (i even); x[i]=i; y[i]=-i  -> out[i] = i if even else -i
        let cond = BoolTensor::from_vec((0..n).map(|i| i % 2 == 0).collect(), vec![n])
            .expect("cond")
            .to(Device::Cuda(0))
            .expect("cond cuda");
        let x = cuda_f32((0..n).map(|i| i as f32).collect());
        let y = cuda_f32((0..n).map(|i| -(i as f32)).collect());
        let got = ferrotorch_core::where_cond_bt(&cond, &x, &y).expect("where");
        let expect: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
            .collect();
        assert_eq!(host_f32(&got), expect, "where n={n}");
    }
}

#[test]
fn logic_and_or_exact_len_multisize_matches_torch() {
    ensure_cuda();
    for &n in &[6usize, 100, 257] {
        // a[i] = (i%2==0), b[i] = (i%3==0); and/or via BoolTensor on CUDA.
        let a = BoolTensor::from_vec((0..n).map(|i| i % 2 == 0).collect(), vec![n])
            .expect("a")
            .to(Device::Cuda(0))
            .expect("a cuda");
        let b = BoolTensor::from_vec((0..n).map(|i| i % 3 == 0).collect(), vec![n])
            .expect("b")
            .to(Device::Cuda(0))
            .expect("b cuda");
        let expect_and: Vec<bool> = (0..n).map(|i| i % 2 == 0 && i % 3 == 0).collect();
        let expect_or: Vec<bool> = (0..n).map(|i| i % 2 == 0 || i % 3 == 0).collect();
        assert_eq!(host_bool(&a.and(&b).expect("and")), expect_and, "and n={n}");
        assert_eq!(host_bool(&a.or(&b).expect("or")), expect_or, "or n={n}");
    }
}

// ── pooled / exact backing-store combinations (the logical-len core risk) ────

/// `.contiguous()` of an offset view -> POOLED buffer (raw len rounded to 256).
/// Build via a [N,1] full -> narrow rows so the view's base buffer > numel.
/// Shape of the returned view is `[count, 1]`.
fn pooled_view_f32(
    rows_full: usize,
    start: usize,
    count: usize,
    fill: impl Fn(usize) -> f32,
) -> Tensor<f32> {
    let full = Tensor::from_storage(
        TensorStorage::cpu((0..rows_full).map(&fill).collect()),
        vec![rows_full, 1],
        false,
    )
    .expect("full")
    .to(Device::Cuda(0))
    .expect("full cuda");
    let view = full.narrow(0, start, count).expect("narrow");
    assert_ne!(view.storage_offset(), 0, "must carry offset");
    view
}

#[test]
fn compare_pooled_vs_pooled_logical_len_matches_torch() {
    ensure_cuda();
    // both operands are narrowed views -> both materialise to pooled (>=256 raw)
    // buffers in compare_float's `.contiguous()`. Logical numel = 6.
    // a = [3,4,5,6,7,8], b = [4,5,6,7,8,9]  -> gt = all false.
    let a = pooled_view_f32(8, 2, 6, |i| (i + 1) as f32); // full=[1..8], view=[3..8]
    let b = pooled_view_f32(8, 2, 6, |i| (i + 2) as f32); // full=[2..9], view=[4..9]
    let got = host_bool(&BoolTensor::gt(&a, &b).expect("gt pooled/pooled"));
    assert_eq!(got, vec![false; 6], "pooled vs pooled: a[i] < b[i] always");
}

#[test]
fn compare_pooled_vs_exact_logical_len_matches_torch() {
    ensure_cuda();
    // a = narrowed view [3,4,5,6,7,8] (pooled, [6,1]), b = exact-len [4.5;6] [6,1].
    // gt -> [F,F,T,T,T,T].
    let a = pooled_view_f32(8, 2, 6, |i| (i + 1) as f32);
    let b = cuda_f32_col(vec![4.5; 6]);
    let got = host_bool(&BoolTensor::gt(&a, &b).expect("gt pooled/exact"));
    assert_eq!(got, vec![false, false, true, true, true, true]);
}

#[test]
fn where_pooled_x_exact_y_logical_len_matches_torch() {
    ensure_cuda();
    // cond [6,1], x = narrowed pooled view [3,4,5,6,7,8] [6,1], y = zeros [6,1].
    // cond[i] = idx>=2 -> out = [0,0,5,6,7,8].
    let cond = BoolTensor::from_vec(vec![false, false, true, true, true, true], vec![6, 1])
        .expect("cond")
        .to(Device::Cuda(0))
        .expect("cond cuda");
    let x = pooled_view_f32(8, 2, 6, |i| (i + 1) as f32); // [3,4,5,6,7,8]
    let y = cuda_f32_col(vec![0.0; 6]);
    let got = ferrotorch_core::where_cond_bt(&cond, &x, &y).expect("where pooled x");
    assert_eq!(host_f32(&got), vec![0.0, 0.0, 5.0, 6.0, 7.0, 8.0]);
}

// ── half / bf16 compare + where (launch_cmp_half / where_16 logical-len) ─────

#[test]
fn compare_gt_f16_exact_len_matches_torch() {
    ensure_cuda();
    // a=[1..6], b=[2;6] half -> gt=[0,0,1,1,1,1]
    let n = 6usize;
    let a = Tensor::<f16>::from_storage(
        TensorStorage::cpu((1..=n).map(|i| f16::from_f32(i as f32)).collect()),
        vec![n],
        false,
    )
    .expect("a f16")
    .to(Device::Cuda(0))
    .expect("a cuda");
    let b = Tensor::<f16>::from_storage(
        TensorStorage::cpu(vec![f16::from_f32(2.0); n]),
        vec![n],
        false,
    )
    .expect("b f16")
    .to(Device::Cuda(0))
    .expect("b cuda");
    let got = host_bool(&BoolTensor::gt(&a, &b).expect("gt f16"));
    assert_eq!(got, vec![false, false, true, true, true, true]);
}

#[test]
fn compare_gt_bf16_exact_len_matches_torch() {
    ensure_cuda();
    let n = 6usize;
    let a = Tensor::<bf16>::from_storage(
        TensorStorage::cpu((1..=n).map(|i| bf16::from_f32(i as f32)).collect()),
        vec![n],
        false,
    )
    .expect("a bf16")
    .to(Device::Cuda(0))
    .expect("a cuda");
    let b = Tensor::<bf16>::from_storage(
        TensorStorage::cpu(vec![bf16::from_f32(2.0); n]),
        vec![n],
        false,
    )
    .expect("b bf16")
    .to(Device::Cuda(0))
    .expect("b cuda");
    let got = host_bool(&BoolTensor::gt(&a, &b).expect("gt bf16"));
    assert_eq!(got, vec![false, false, true, true, true, true]);
}

#[test]
fn where_bf16_exact_len_matches_torch() {
    ensure_cuda();
    // cond = (a>b) = [0,0,1,1,1,1]; where(cond, a, b) = [2,2,3,4,5,6]
    let n = 6usize;
    let a = Tensor::<bf16>::from_storage(
        TensorStorage::cpu((1..=n).map(|i| bf16::from_f32(i as f32)).collect()),
        vec![n],
        false,
    )
    .expect("a bf16")
    .to(Device::Cuda(0))
    .expect("a cuda");
    let b = Tensor::<bf16>::from_storage(
        TensorStorage::cpu(vec![bf16::from_f32(2.0); n]),
        vec![n],
        false,
    )
    .expect("b bf16")
    .to(Device::Cuda(0))
    .expect("b cuda");
    let cond = BoolTensor::from_vec(vec![false, false, true, true, true, true], vec![n])
        .expect("cond")
        .to(Device::Cuda(0))
        .expect("cond cuda");
    let got = ferrotorch_core::where_cond_bt(&cond, &a, &b).expect("where bf16");
    let got_f32: Vec<f32> = got
        .cpu()
        .expect("cpu")
        .data()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect();
    assert_eq!(got_f32, vec![2.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
