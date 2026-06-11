//! Red-then-green regression tests for audit finding CORE-140 (crosslink
//! #1834): f16 matmul accumulates in f16 in the `mm_raw*` small-matrix
//! paths (bf16 already has an f32 accumulator there), and `dot` / `mv` /
//! `vm` / `bmm` accumulate in storage precision for ALL dtypes — both
//! diverging from PyTorch, which computes Half/BFloat16 CPU matmuls in
//! `opmath_type = float` (aten/src/ATen/OpMathType.h: `opmath_type<Half>`
//! and `opmath_type<BFloat16>` are `float`).
//!
//! Every numerical expectation below is quoted from a LIVE torch session
//! (torch 2.11.0+cu130, R-ORACLE-1 path (b)); the generating snippet is
//! pasted next to each expected block. Inputs use the deterministic
//! pattern `gen(i, mult) = 0.25 + ((i*i + mult*i) % 63) / 64`, whose values
//! lie in [0.25, 1.21875] and carry at most 6 fractional bits — exactly
//! representable in BOTH f16 (10-bit mantissa) and bf16 (8-bit mantissa),
//! so ferrotorch and torch see bit-identical inputs.
//!
//! # Tolerance derivation (R-ORACLE-5)
//!
//! With the inputs exact in the storage dtype, an opmath(f32)-correct
//! kernel computes each output as round_to_storage(f32 sum of exact
//! products). Each f16*f16 product carries <= 21 mantissa bits (exact in
//! f32); the f32 accumulation over k = 128 terms of magnitude <= 1.5
//! differs from torch's f32 accumulation only by summation order:
//! |delta| <= k * eps_f32 * sum|terms| ~= 128 * 1.19e-7 * 190 ~= 3e-3,
//! far below half of one storage ULP at the result magnitude ~70 (f16 ULP
//! = 2^-10 * 2^6 = 0.0625; bf16 ULP = 2^-7 * 2^6 = 0.5). The final
//! rounding therefore lands within 1 storage ULP of torch's value:
//! rel tol = 2^-10 for f16, 2^-7 for bf16. Pre-fix, storage-precision
//! accumulation drifts by ~0.5 % (f16) / several % (bf16) — outside both.

use ferrotorch_core::grad_fns::linalg::bmm as gf_bmm;
use ferrotorch_core::ops::linalg::{
    bmm as ops_bmm, dot as ops_dot, matmul as ops_matmul, mv as ops_mv,
};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use half::{bf16, f16};

/// Deterministic input pattern, exact in f16 AND bf16 (see module docs).
fn pat(i: usize, mult: usize) -> f32 {
    0.25 + ((i * i + mult * i) % 63) as f32 / 64.0
}

fn t_f16(data: Vec<f16>, shape: &[usize]) -> Tensor<f16> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).unwrap()
}
fn t_bf16(data: Vec<bf16>, shape: &[usize]) -> Tensor<bf16> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).unwrap()
}
fn gen_f16(n: usize, mult: usize) -> Vec<f16> {
    (0..n).map(|i| f16::from_f32(pat(i, mult))).collect()
}
fn gen_bf16(n: usize, mult: usize) -> Vec<bf16> {
    (0..n).map(|i| bf16::from_f32(pat(i, mult))).collect()
}

/// 1 f16 ULP relative (2^-10); see module-level tolerance derivation.
const REL_F16: f32 = 1.0 / 1024.0;
/// 1 bf16 ULP relative (2^-7); see module-level tolerance derivation.
const REL_BF16: f32 = 1.0 / 128.0;

fn assert_close_f16(actual: &[f16], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let a = a.to_f32();
        assert!(
            (a - e).abs() <= e.abs() * REL_F16,
            "{label}: index {i}: got {a}, torch oracle {e} \
             (diff {}, rel {} > 1 f16 ULP = {REL_F16})",
            (a - e).abs(),
            (a - e).abs() / e.abs()
        );
    }
}
fn assert_close_bf16(actual: &[bf16], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let a = a.to_f32();
        assert!(
            (a - e).abs() <= e.abs() * REL_BF16,
            "{label}: index {i}: got {a}, torch oracle {e} \
             (diff {}, rel {} > 1 bf16 ULP = {REL_BF16})",
            (a - e).abs(),
            (a - e).abs() / e.abs()
        );
    }
}

const K: usize = 128;

// ===========================================================================
// mm — the `mm_raw` small path (max_dim = 128 <= DIRECT_MM_THRESHOLD)
// ===========================================================================

// torch oracle (torch 2.11.0+cu130):
//   >>> gen = lambda n, m: [0.25 + ((i*i + m*i) % 63)/64.0 for i in range(n)]
//   >>> a = torch.tensor(gen(4*128, 37), dtype=torch.float16).reshape(4, 128)
//   >>> b = torch.tensor(gen(128*4, 53), dtype=torch.float16).reshape(128, 4)
//   >>> (a @ b).flatten().tolist()
//   [71.6875, 73.875, 72.125, 69.9375, 70.9375, 70.4375, 70.375, 70.75,
//    70.5, 71.1875, 70.375, 70.25, 70.0, 71.1875, 71.0625, 71.5625]
#[test]
fn core140_mm_f16_k128_opmath_f32() {
    let a = t_f16(gen_f16(4 * K, 37), &[4, K]);
    let b = t_f16(gen_f16(K * 4, 53), &[K, 4]);
    let c = a.matmul(&b).expect("f16 mm forward");
    assert_close_f16(
        c.data().unwrap(),
        &[
            71.6875, 73.875, 72.125, 69.9375, 70.9375, 70.4375, 70.375, 70.75, //
            70.5, 71.1875, 70.375, 70.25, 70.0, 71.1875, 71.0625, 71.5625,
        ],
        "mm f16 k=128",
    );
}

// torch oracle (same session):
//   >>> a = torch.tensor(gen(4*128, 37), dtype=torch.bfloat16).reshape(4, 128)
//   >>> b = torch.tensor(gen(128*4, 53), dtype=torch.bfloat16).reshape(128, 4)
//   >>> (a @ b).float().flatten().tolist()
//   [71.5, 74.0, 72.0, 70.0, 71.0, 70.5, 70.5, 71.0,
//    70.5, 71.0, 70.5, 70.0, 70.0, 71.0, 71.0, 71.5]
// Guard test: the bf16 mm_raw small path already accumulates in f32; this
// pins the behaviour the f16 fix must replicate.
#[test]
fn core140_mm_bf16_k128_opmath_f32_guard() {
    let a = t_bf16(gen_bf16(4 * K, 37), &[4, K]);
    let b = t_bf16(gen_bf16(K * 4, 53), &[K, 4]);
    let c = a.matmul(&b).expect("bf16 mm forward");
    assert_close_bf16(
        c.data().unwrap(),
        &[
            71.5, 74.0, 72.0, 70.0, 71.0, 70.5, 70.5, 71.0, //
            70.5, 71.0, 70.5, 70.0, 70.0, 71.0, 71.0, 71.5,
        ],
        "mm bf16 k=128",
    );
}

// ===========================================================================
// dot — k = 128 precision + intermediate-overflow probes
// ===========================================================================

// torch oracle (same session):
//   >>> x = torch.tensor(gen(128, 37), dtype=torch.float16)
//   >>> y = torch.tensor(gen(128, 53), dtype=torch.float16)
//   >>> (x @ y).item()
//   70.5         # f64 reference of the same f16 inputs: 70.486328125
#[test]
fn core140_dot_f16_k128_tensor_matmul() {
    let x = t_f16(gen_f16(K, 37), &[K]);
    let y = t_f16(gen_f16(K, 53), &[K]);
    let c = x.matmul(&y).expect("f16 dot via Tensor::matmul");
    assert_close_f16(c.data().unwrap(), &[70.5], "dot f16 k=128 (Tensor::matmul)");
}

#[test]
fn core140_dot_f16_k128_ops_dot() {
    let x = t_f16(gen_f16(K, 37), &[K]);
    let y = t_f16(gen_f16(K, 53), &[K]);
    let c = ops_dot(&x, &y).expect("f16 ops::dot");
    assert_close_f16(c.data().unwrap(), &[70.5], "dot f16 k=128 (ops::dot)");
}

// torch oracle (same session):
//   >>> xb = torch.tensor(gen(128, 37), dtype=torch.bfloat16)
//   >>> yb = torch.tensor(gen(128, 53), dtype=torch.bfloat16)
//   >>> (xb @ yb).float().item()
//   70.5
#[test]
fn core140_dot_bf16_k128_tensor_matmul() {
    let x = t_bf16(gen_bf16(K, 37), &[K]);
    let y = t_bf16(gen_bf16(K, 53), &[K]);
    let c = x.matmul(&y).expect("bf16 dot via Tensor::matmul");
    assert_close_bf16(
        c.data().unwrap(),
        &[70.5],
        "dot bf16 k=128 (Tensor::matmul)",
    );
}

#[test]
fn core140_dot_bf16_k128_ops_dot() {
    let x = t_bf16(gen_bf16(K, 37), &[K]);
    let y = t_bf16(gen_bf16(K, 53), &[K]);
    let c = ops_dot(&x, &y).expect("bf16 ops::dot");
    assert_close_bf16(c.data().unwrap(), &[70.5], "dot bf16 k=128 (ops::dot)");
}

// Intermediate-sum overflow: in f16 the running sum hits
// 240*240 + 240*240 = 115200 > 65504 (f16 max) and poisons the result to
// +inf; in f32 opmath the sum stays finite and the true value 57600 is
// exactly representable in f16.
//
// torch oracle (same session):
//   >>> x = torch.tensor([240.0, 240.0, -240.0], dtype=torch.float16)
//   >>> y = torch.tensor([240.0, 240.0, 240.0], dtype=torch.float16)
//   >>> (x @ y).item()
//   57600.0
#[test]
fn core140_dot_f16_intermediate_overflow() {
    let x = t_f16(
        vec![
            f16::from_f32(240.0),
            f16::from_f32(240.0),
            f16::from_f32(-240.0),
        ],
        &[3],
    );
    let y = t_f16(vec![f16::from_f32(240.0); 3], &[3]);
    let c = x.matmul(&y).expect("f16 dot forward");
    let got = c.data().unwrap()[0].to_f32();
    assert!(
        got.is_finite() && (got - 57600.0).abs() <= 57600.0 * REL_F16,
        "dot f16 overflow: got {got}, torch oracle 57600.0 \
         (f16 accumulation overflows the 115200 partial sum to inf)"
    );
}

// ===========================================================================
// mv / vm
// ===========================================================================

// torch oracle (same session):
//   >>> a = torch.tensor(gen(4*128, 37), dtype=torch.float16).reshape(4, 128)
//   >>> v = torch.tensor(gen(128, 53), dtype=torch.float16)
//   >>> (a @ v).tolist()
//   [70.5, 72.875, 70.5, 72.1875]
#[test]
fn core140_mv_f16_k128_tensor_matmul() {
    let a = t_f16(gen_f16(4 * K, 37), &[4, K]);
    let v = t_f16(gen_f16(K, 53), &[K]);
    let c = a.matmul(&v).expect("f16 mv via Tensor::matmul");
    assert_close_f16(
        c.data().unwrap(),
        &[70.5, 72.875, 70.5, 72.1875],
        "mv f16 k=128 (Tensor::matmul)",
    );
}

#[test]
fn core140_mv_f16_k128_ops_mv() {
    let a = t_f16(gen_f16(4 * K, 37), &[4, K]);
    let v = t_f16(gen_f16(K, 53), &[K]);
    let c = ops_mv(&a, &v).expect("f16 ops::mv");
    assert_close_f16(
        c.data().unwrap(),
        &[70.5, 72.875, 70.5, 72.1875],
        "mv f16 k=128 (ops::mv)",
    );
}

// torch oracle (same session):
//   >>> ab = torch.tensor(gen(4*128, 37), dtype=torch.bfloat16).reshape(4, 128)
//   >>> vb = torch.tensor(gen(128, 53), dtype=torch.bfloat16)
//   >>> (ab @ vb).float().tolist()
//   [70.5, 73.0, 70.5, 72.0]
#[test]
fn core140_mv_bf16_k128_tensor_matmul() {
    let a = t_bf16(gen_bf16(4 * K, 37), &[4, K]);
    let v = t_bf16(gen_bf16(K, 53), &[K]);
    let c = a.matmul(&v).expect("bf16 mv via Tensor::matmul");
    assert_close_bf16(
        c.data().unwrap(),
        &[70.5, 73.0, 70.5, 72.0],
        "mv bf16 k=128 (Tensor::matmul)",
    );
}

// torch oracle (same session):
//   >>> u = torch.tensor(gen(128, 37), dtype=torch.float16)
//   >>> b = torch.tensor(gen(128*4, 53), dtype=torch.float16).reshape(128, 4)
//   >>> (u @ b).tolist()
//   [71.6875, 73.875, 72.125, 69.9375]
#[test]
fn core140_vm_f16_k128_tensor_matmul() {
    let u = t_f16(gen_f16(K, 37), &[K]);
    let b = t_f16(gen_f16(K * 4, 53), &[K, 4]);
    let c = u.matmul(&b).expect("f16 vm via Tensor::matmul");
    assert_close_f16(
        c.data().unwrap(),
        &[71.6875, 73.875, 72.125, 69.9375],
        "vm f16 k=128 (Tensor::matmul)",
    );
}

#[test]
fn core140_vm_f16_k128_ops_matmul() {
    let u = t_f16(gen_f16(K, 37), &[K]);
    let b = t_f16(gen_f16(K * 4, 53), &[K, 4]);
    let c = ops_matmul(&u, &b).expect("f16 vm via ops::matmul");
    assert_close_f16(
        c.data().unwrap(),
        &[71.6875, 73.875, 72.125, 69.9375],
        "vm f16 k=128 (ops::matmul)",
    );
}

// torch oracle (same session):
//   >>> ub = torch.tensor(gen(128, 37), dtype=torch.bfloat16)
//   >>> bb = torch.tensor(gen(128*4, 53), dtype=torch.bfloat16).reshape(128, 4)
//   >>> (ub @ bb).float().tolist()
//   [71.5, 74.0, 72.0, 70.0]
#[test]
fn core140_vm_bf16_k128_tensor_matmul() {
    let u = t_bf16(gen_bf16(K, 37), &[K]);
    let b = t_bf16(gen_bf16(K * 4, 53), &[K, 4]);
    let c = u.matmul(&b).expect("bf16 vm via Tensor::matmul");
    assert_close_bf16(
        c.data().unwrap(),
        &[71.5, 74.0, 72.0, 70.0],
        "vm bf16 k=128 (Tensor::matmul)",
    );
}

// ===========================================================================
// bmm — both the ops triple loop and the grad_fns mm_raw-routed kernel
// ===========================================================================

// torch oracle (same session):
//   >>> a3 = torch.tensor(gen(2*2*128, 37), dtype=torch.float16).reshape(2,2,128)
//   >>> b3 = torch.tensor(gen(2*128*2, 53), dtype=torch.float16).reshape(2,128,2)
//   >>> torch.bmm(a3, b3).flatten().tolist()
//   [74.0, 71.0625, 70.1875, 69.3125, 70.1875, 68.9375, 71.3125, 71.0625]
#[test]
fn core140_bmm_f16_k128_ops_bmm() {
    let a = t_f16(gen_f16(2 * 2 * K, 37), &[2, 2, K]);
    let b = t_f16(gen_f16(2 * K * 2, 53), &[2, K, 2]);
    let c = ops_bmm(&a, &b).expect("f16 ops::bmm");
    assert_close_f16(
        c.data().unwrap(),
        &[
            74.0, 71.0625, 70.1875, 69.3125, 70.1875, 68.9375, 71.3125, 71.0625,
        ],
        "bmm f16 k=128 (ops::bmm)",
    );
}

#[test]
fn core140_bmm_f16_k128_grad_fns_bmm() {
    let a = t_f16(gen_f16(2 * 2 * K, 37), &[2, 2, K]);
    let b = t_f16(gen_f16(2 * K * 2, 53), &[2, K, 2]);
    let c = gf_bmm(&a, &b).expect("f16 grad_fns::bmm");
    assert_close_f16(
        c.data().unwrap(),
        &[
            74.0, 71.0625, 70.1875, 69.3125, 70.1875, 68.9375, 71.3125, 71.0625,
        ],
        "bmm f16 k=128 (grad_fns::bmm)",
    );
}

// torch oracle (same session):
//   >>> a3b = torch.tensor(gen(2*2*128, 37), dtype=torch.bfloat16).reshape(2,2,128)
//   >>> b3b = torch.tensor(gen(2*128*2, 53), dtype=torch.bfloat16).reshape(2,128,2)
//   >>> torch.bmm(a3b, b3b).float().flatten().tolist()
//   [74.0, 71.0, 70.0, 69.5, 70.0, 69.0, 71.5, 71.0]
#[test]
fn core140_bmm_bf16_k128_ops_bmm() {
    let a = t_bf16(gen_bf16(2 * 2 * K, 37), &[2, 2, K]);
    let b = t_bf16(gen_bf16(2 * K * 2, 53), &[2, K, 2]);
    let c = ops_bmm(&a, &b).expect("bf16 ops::bmm");
    assert_close_bf16(
        c.data().unwrap(),
        &[74.0, 71.0, 70.0, 69.5, 70.0, 69.0, 71.5, 71.0],
        "bmm bf16 k=128 (ops::bmm)",
    );
}
