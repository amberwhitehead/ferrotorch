//! Divergence: the fused-affine family (`addmm`, `addmv`, `addr`, `addbmm`,
//! `baddbmm`) does NOT honour PyTorch's `beta == 0` NaN/Inf-suppression
//! contract. Upstream, when `beta == 0` the `self`/bias term is DROPPED
//! ENTIRELY (the `self` buffer is never read), so non-finite values in `self`
//! must NOT propagate to the output — the result is `alpha * (product)`.
//!
//! ferrotorch's forwards (`ferrotorch-core/src/grad_fns/linalg.rs`) PREVIOUSLY
//! computed `beta * self + alpha * product` LITERALLY in all 5 forwards, so
//! `beta == 0.0f32` with a NaN/Inf in `self` yielded `0.0 * NaN = NaN`
//! (IEEE-754), propagating the poison torch deliberately drops. The fix
//! (#1598) branches on `beta == 0` in each forward and computes
//! `alpha * product` only, never reading the `self`/bias buffer.
//!
//! Upstream contract (verified against torch 2.11.0+cu130 live oracle — every
//! expected value below is `alpha * product`, the self term absent):
//!
//! - addr: `aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp:53-55` — "when
//!   beta == 0, values in self should be ignored, nans and infs in self should
//!   not propagate." + `:55 if (beta_val == zero_val) {` + `:60 return
//!   alpha_val * vec1_val * vec2_val;`
//! - addmm/addbmm/baddbmm GEMM: the `self`/`c` term is skipped when beta==0 —
//!   `aten/src/ATen/native/cpu/BlasKernel.cpp:161-162` `if (beta ==
//!   opmath_t(0)) { c[...] = alpha * dot; }` and addmm only copies `self` into
//!   the result when beta != 0: `aten/src/ATen/native/LinearAlgebra.cpp:1442`
//!   `if (beta.toComplexDouble() != 0.0 && !self.is_same(result)) {
//!   result.copy_(self); }`. addbmm/baddbmm:
//!   `aten/src/ATen/native/LinearAlgebra.cpp:1682-1684` "For beta == 0, the
//!   r's value will be ignored, especially for nan value." + `if (beta ==
//!   opmath_t{0}) { r2[j] = alpha * acc_value; }`
//! - addmv: `aten/src/ATen/native/Blas.cpp:77-79` — "By definition, when
//!   beta==0, values in self should be ignored. nans and infs should not
//!   propagate" + `:90 if (!result.is_same(*self_) && betaval != 0.0)` (self
//!   copied only when beta != 0).
//!
//! Each test below feeds `self` full of NaN with `beta = 0.0, alpha = 1.0` and
//! asserts the output equals `alpha * product` with NO NaN.
//!
//! Expected values are NOT copied from the ferrotorch side (R-CHAR-3): they
//! are the pure product term, recomputed here from the input operands by the
//! same closed-form torch documents (`outer`, `mm`, `mv`, `bmm`, `sum_b bmm`),
//! and cross-checked against the live torch 2.11.0 oracle.
//!
//! Tracking: #1598 (blocker, fixed). These tests were `#[ignore]`'d while the
//! divergence was pinned; with the forwards fixed they are now permanent
//! regression coverage. The parity-sweep `addr` runner arm (commit e8350c48b)
//! still SKIPS the op_db addr `{beta:0,alpha:0,self:NaN}` sample via `Ok(None)`,
//! masking this in the sweep; removing that skip is the orchestrator's
//! follow-up (a different file: tools/parity-sweep/runner/src/main.rs).

use ferrotorch_core::Tensor;
use ferrotorch_core::linalg;
use ferrotorch_core::storage::TensorStorage;

fn nog(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

#[track_caller]
fn assert_finite_eq(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.is_finite(),
            "{label}: DIVERGENCE — output[{i}] = {a} is non-finite; torch drops \
             the beta==0 self term and returns finite {e} (NaN/Inf must not propagate)"
        );
        let diff = (a - e).abs();
        assert!(
            diff <= 1e-4 + 1e-4 * e.abs(),
            "{label}: output[{i}] = {a} != torch {e} (diff={diff})"
        );
    }
}

// ---------------------------------------------------------------------------
// addr(self=NaN[2,3], vec1=[2], vec2=[3], beta=0, alpha=1)
// torch 2.11 -> [[3,4,5],[6,8,10]]  (alpha * outer(vec1, vec2), self dropped)
// ---------------------------------------------------------------------------
#[test]
fn addr_beta0_nan_self_dropped() {
    let nan = f32::NAN;
    let self_t = nog(&[nan, nan, nan, nan, nan, nan], &[2, 3]);
    let v1d = [1.0_f32, 2.0];
    let v2d = [3.0_f32, 4.0, 5.0];
    let v1 = nog(&v1d, &[2]);
    let v2 = nog(&v2d, &[3]);

    let out = linalg::addr(&self_t, &v1, &v2, 0.0_f32, 1.0_f32).unwrap();

    // expected = alpha * outer(v1, v2), self absent (LinearAlgebraKernel.cpp:60)
    let mut expected = vec![0.0_f32; 6];
    for i in 0..2 {
        for j in 0..3 {
            expected[i * 3 + j] = 1.0 * v1d[i] * v2d[j];
        }
    }
    assert_finite_eq(out.data().unwrap(), &expected, "addr beta=0 nan-self");
}

// ---------------------------------------------------------------------------
// addmm(self=NaN[2,2], mat1=[2,3], mat2=[3,2], beta=0, alpha=1)
// torch 2.11 -> [[4,5],[10,11]]  (alpha * (mat1 @ mat2), self dropped)
// ---------------------------------------------------------------------------
#[test]
fn addmm_beta0_nan_self_dropped() {
    let nan = f32::NAN;
    let self_t = nog(&[nan, nan, nan, nan], &[2, 2]);
    let m1 = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let m2 = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2
    let mat1 = nog(&m1, &[2, 3]);
    let mat2 = nog(&m2, &[3, 2]);

    let out = linalg::addmm(&self_t, &mat1, &mat2, 0.0_f32, 1.0_f32).unwrap();

    // expected = alpha * (mat1 @ mat2), self absent (BlasKernel.cpp:162)
    let mut expected = vec![0.0_f32; 4];
    for i in 0..2 {
        for j in 0..2 {
            let mut acc = 0.0_f32;
            for k in 0..3 {
                acc += m1[i * 3 + k] * m2[k * 2 + j];
            }
            expected[i * 2 + j] = 1.0 * acc;
        }
    }
    assert_finite_eq(out.data().unwrap(), &expected, "addmm beta=0 nan-self");
}

// ---------------------------------------------------------------------------
// addmv(self=NaN[2], mat=[2,3], vec=[3], beta=0, alpha=1)
// torch 2.11 -> [6,15]  (alpha * (mat @ vec), self dropped)
// ---------------------------------------------------------------------------
#[test]
fn addmv_beta0_nan_self_dropped() {
    let nan = f32::NAN;
    let self_t = nog(&[nan, nan], &[2]);
    let m = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let v = [1.0_f32, 1.0, 1.0]; // 3
    let mat = nog(&m, &[2, 3]);
    let vec = nog(&v, &[3]);

    let out = linalg::addmv(&self_t, &mat, &vec, 0.0_f32, 1.0_f32).unwrap();

    // expected = alpha * (mat @ vec), self absent (Blas.cpp:77-79)
    let mut expected = vec![0.0_f32; 2];
    for i in 0..2 {
        let mut acc = 0.0_f32;
        for k in 0..3 {
            acc += m[i * 3 + k] * v[k];
        }
        expected[i] = 1.0 * acc;
    }
    assert_finite_eq(out.data().unwrap(), &expected, "addmv beta=0 nan-self");
}

// ---------------------------------------------------------------------------
// addbmm(self=NaN[2,2], batch1=[2,2,3], batch2=[2,3,2], beta=0, alpha=1)
// torch 2.11 -> [[6,5],[10,13]]  (alpha * sum_b(b1[b]@b2[b]), self dropped)
// ---------------------------------------------------------------------------
#[test]
fn addbmm_beta0_nan_self_dropped() {
    let nan = f32::NAN;
    let self_t = nog(&[nan, nan, nan, nan], &[2, 2]);
    // batch1 [2,2,3]
    let b1 = [
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // batch 1
    ];
    // batch2 [2,3,2]
    let b2 = [
        1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, // batch 0
        2.0, 0.0, 0.0, 2.0, 0.0, 0.0, // batch 1
    ];
    let batch1 = nog(&b1, &[2, 2, 3]);
    let batch2 = nog(&b2, &[2, 3, 2]);

    let out = linalg::addbmm(&self_t, &batch1, &batch2, 0.0_f32, 1.0_f32).unwrap();

    // expected = alpha * sum_b(b1[b] @ b2[b]) (LinearAlgebra.cpp:1682-1684)
    let mut expected = vec![0.0_f32; 4];
    for bi in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0_f32;
                for k in 0..3 {
                    acc += b1[bi * 6 + i * 3 + k] * b2[bi * 6 + k * 2 + j];
                }
                expected[i * 2 + j] += acc;
            }
        }
    }
    assert_finite_eq(out.data().unwrap(), &expected, "addbmm beta=0 nan-self");
}

// ---------------------------------------------------------------------------
// baddbmm(self=NaN[2,2,2], batch1=[2,2,3], batch2=[2,3,2], beta=0, alpha=1)
// torch 2.11 -> [[[4,5],[10,11]],[[2,0],[0,2]]] (alpha * bmm, self dropped)
// ---------------------------------------------------------------------------
#[test]
fn baddbmm_beta0_nan_self_dropped() {
    let nan = f32::NAN;
    let self_t = nog(&[nan, nan, nan, nan, nan, nan, nan, nan], &[2, 2, 2]);
    // batch1 [2,2,3]
    let b1 = [
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // batch 1
    ];
    // batch2 [2,3,2]
    let b2 = [
        1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, // batch 0
        2.0, 0.0, 0.0, 2.0, 0.0, 0.0, // batch 1
    ];
    let batch1 = nog(&b1, &[2, 2, 3]);
    let batch2 = nog(&b2, &[2, 3, 2]);

    let out = linalg::baddbmm(&self_t, &batch1, &batch2, 0.0_f32, 1.0_f32).unwrap();

    // expected = alpha * bmm(batch1, batch2), per-batch, self dropped
    let mut expected = vec![0.0_f32; 8];
    for bi in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0_f32;
                for k in 0..3 {
                    acc += b1[bi * 6 + i * 3 + k] * b2[bi * 6 + k * 2 + j];
                }
                expected[bi * 4 + i * 2 + j] = acc;
            }
        }
    }
    assert_finite_eq(out.data().unwrap(), &expected, "baddbmm beta=0 nan-self");
}
