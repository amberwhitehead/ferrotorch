//! Regression tests for audit finding CORE-044 (crosslink #1738).
//!
//! `rrelu(training=true)` silently executed inference behavior: the
//! `_training` flag was accepted and ignored, and the deterministic mean
//! slope `(lower + upper) / 2` was always applied. PyTorch's training path
//! (`_rrelu_with_noise_train` at `aten/src/ATen/native/Activation.cpp:578-608`)
//! draws an independent slope `r ~ Uniform[lower, upper]` for every element
//! with `x <= 0` (zero INCLUDED), saves it as the `noise` tensor, and the
//! backward applies the saved per-element slopes (`grad * noise`).
//!
//! RNG contract (R-ORACLE-1): the per-element draw is
//! `at::uniform_real_distribution<double>(lower, upper)` which consumes one
//! u64 (two MT19937 u32 calls) per draw and maps it as
//! `next_uniform_f64() * (upper - lower) + lower`
//! (`aten/src/ATen/core/DistributionsHelper.h:60-70`). Because
//! `ferrotorch_core::rng::Generator` is byte-identical to torch's CPU
//! MT19937, the training-mode forward is BIT-EXACT against torch CPU under
//! `manual_seed` — the expectations below are pinned bit patterns from a
//! live torch session (snippets quoted per test).
//!
//! All expectations below were captured live from:
//!   Python 3, torch 2.11.0+cu130, CPU default generator.

use ferrotorch_core::grad_fns::activation::rrelu;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::{Tensor, TensorStorage, manual_seed};
use std::sync::{Mutex, MutexGuard};

const LOWER: f64 = 0.125;
const UPPER: f64 = 1.0 / 3.0;

fn cpu_f64(data: &[f64], shape: &[usize], rg: bool) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), rg).unwrap()
}

fn cpu_f32(data: &[f32], shape: &[usize], rg: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), rg).unwrap()
}

fn default_rng_test_lock() -> MutexGuard<'static, ()> {
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// CORE-044 core red test: training mode must be stochastic — two different
/// seeds must produce different negative-branch outputs. Pre-fix, both calls
/// returned the deterministic mean-slope output and this failed.
#[test]
fn rrelu_training_distinct_seeds_distinct_outputs() {
    let _guard = default_rng_test_lock();
    let x = cpu_f64(&[-1.0, -2.0, 3.0], &[3], false);
    manual_seed(42).unwrap();
    let a = rrelu(&x, LOWER, UPPER, true).unwrap().data_vec().unwrap();
    manual_seed(43).unwrap();
    let b = rrelu(&x, LOWER, UPPER, true).unwrap().data_vec().unwrap();
    assert_ne!(
        a, b,
        "rrelu(training=true) is deterministic across seeds (CORE-044)"
    );
}

/// Same seed, same input → identical output (manual_seed reproducibility).
#[test]
fn rrelu_training_manual_seed_reproducible() {
    let _guard = default_rng_test_lock();
    let x = cpu_f64(&[-1.0, -2.0, 3.0, -0.5], &[4], false);
    manual_seed(1234).unwrap();
    let a = rrelu(&x, LOWER, UPPER, true).unwrap().data_vec().unwrap();
    manual_seed(1234).unwrap();
    let b = rrelu(&x, LOWER, UPPER, true).unwrap().data_vec().unwrap();
    let a_bits: Vec<u64> = a.iter().map(|v| v.to_bits()).collect();
    let b_bits: Vec<u64> = b.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        a_bits, b_bits,
        "manual_seed must make training mode reproducible"
    );
}

/// Bit-exact torch parity, f64 (R-ORACLE-1(b) — live torch snippet):
/// ```python
/// >>> torch.manual_seed(42)
/// >>> F.rrelu(torch.tensor([-1.0,-2.0,3.0], dtype=torch.float64),
/// ...         lower=0.125, upper=1/3, training=True)
/// # bits: [0xbfc18d00549888c0, 0xbfd1ad777c77518c, 0x4008000000000000]
/// # vals: [-0.1371155179086312, -0.27621256976024067, 3.0]
/// ```
/// (torch 2.11.0+cu130, CPU)
#[test]
fn rrelu_training_matches_torch_seed42_bitexact_f64() {
    let _guard = default_rng_test_lock();
    let x = cpu_f64(&[-1.0, -2.0, 3.0], &[3], false);
    manual_seed(42).unwrap();
    let y = rrelu(&x, LOWER, UPPER, true).unwrap().data_vec().unwrap();
    let got: Vec<u64> = y.iter().map(|v| v.to_bits()).collect();
    let expected: Vec<u64> = vec![
        0xbfc1_8d00_5498_88c0,
        0xbfd1_ad77_7c77_518c,
        0x4008_0000_0000_0000,
    ];
    assert_eq!(
        got, expected,
        "training-mode forward must be bit-exact vs torch CPU MT19937 (got {y:?})"
    );
}

/// Bit-exact torch parity, f32. torch draws the slope in DOUBLE precision
/// regardless of dtype (`at::uniform_real_distribution<double>` inside
/// `_rrelu_with_noise_train`), then casts to the tensor dtype:
/// ```python
/// >>> torch.manual_seed(42)
/// >>> F.rrelu(torch.tensor([-1.0,-2.0,3.0], dtype=torch.float32),
/// ...         lower=0.125, upper=1/3, training=True)
/// # bits: [0xbe0c6803, 0xbe8d6bbc, 0x40400000]
/// # vals: [-0.13711552321910858, -0.27621257305145264, 3.0]
/// ```
/// (torch 2.11.0+cu130, CPU)
#[test]
fn rrelu_training_matches_torch_seed42_bitexact_f32() {
    let _guard = default_rng_test_lock();
    let x = cpu_f32(&[-1.0, -2.0, 3.0], &[3], false);
    manual_seed(42).unwrap();
    let y = rrelu(&x, LOWER, UPPER, true).unwrap().data_vec().unwrap();
    let got: Vec<u32> = y.iter().map(|v| v.to_bits()).collect();
    let expected: Vec<u32> = vec![0xbe0c_6803, 0xbe8d_6bbc, 0x4040_0000];
    assert_eq!(
        got, expected,
        "f32 training forward must cast the f64 draw like torch (got {y:?})"
    );
}

/// Backward applies the SAVED per-element slopes — not a fresh draw, not the
/// mean slope. Live torch oracle:
/// ```python
/// >>> torch.manual_seed(42)
/// >>> xg = torch.tensor([-1.0,-2.0,3.0], dtype=torch.float64, requires_grad=True)
/// >>> F.rrelu(xg, 0.125, 1/3, training=True).sum().backward()
/// >>> xg.grad
/// tensor([0.1371155179086312, 0.13810628488012033, 1.0])
/// ```
/// (torch 2.11.0+cu130, CPU; grad[i] == noise[i] == -out[i]/|x[i]|)
#[test]
fn rrelu_training_backward_applies_saved_noise() {
    let _guard = default_rng_test_lock();
    let x = cpu_f64(&[-1.0, -2.0, 3.0], &[3], true);
    manual_seed(42).unwrap();
    let out = rrelu(&x, LOWER, UPPER, true).unwrap();
    let loss = reduce_sum(&out).unwrap();
    loss.backward().unwrap();
    let g = x.grad().unwrap().expect("grad must flow to the leaf");
    let got: Vec<u64> = g.data_vec().unwrap().iter().map(|v| v.to_bits()).collect();
    let expected: Vec<u64> = vec![
        0.137_115_517_908_631_2f64.to_bits(),
        0.138_106_284_880_120_33f64.to_bits(),
        1.0f64.to_bits(),
    ];
    assert_eq!(
        got,
        expected,
        "backward must use the noise drawn in forward (got {:?})",
        g.data_vec().unwrap()
    );
}

/// Negative-branch-only check, with torch's exact boundary semantics:
/// elements with `x <= 0` get a random slope (x == 0 INCLUDED — it consumes
/// a draw and its grad is the random slope), strictly positive elements pass
/// through untouched with grad 1. Live torch oracle:
/// ```python
/// >>> torch.manual_seed(7)
/// >>> x0 = torch.tensor([0.0,-1.0,5.0], dtype=torch.float64, requires_grad=True)
/// >>> o = F.rrelu(x0, 0.125, 1/3, training=True); o.sum().backward()
/// >>> o, x0.grad
/// ([0.0, -0.18201952367745117, 5.0],
///  [0.18320425623050082, 0.18201952367745117, 1.0])
/// ```
/// (torch 2.11.0+cu130, CPU)
#[test]
fn rrelu_training_x_le_zero_draws_positive_passthrough() {
    let _guard = default_rng_test_lock();
    let x = cpu_f64(&[0.0, -1.0, 5.0], &[3], true);
    manual_seed(7).unwrap();
    let out = rrelu(&x, LOWER, UPPER, true).unwrap();
    let fwd = out.data_vec().unwrap();
    let fwd_bits: Vec<u64> = fwd.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        fwd_bits,
        vec![
            0.0f64.to_bits(),
            (-0.182_019_523_677_451_17f64).to_bits(),
            5.0f64.to_bits(),
        ],
        "forward mismatch (got {fwd:?})"
    );
    reduce_sum(&out).unwrap().backward().unwrap();
    let g = x.grad().unwrap().expect("grad must flow");
    let g_bits: Vec<u64> = g.data_vec().unwrap().iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        g_bits,
        vec![
            0.183_204_256_230_500_82f64.to_bits(),
            0.182_019_523_677_451_17f64.to_bits(),
            1.0f64.to_bits(),
        ],
        "x==0 must carry the random slope; x>0 must carry grad 1 (got {:?})",
        g.data_vec().unwrap()
    );
}

/// Distribution sanity: every implied slope lies inside `[lower, upper]` and
/// the sample mean is near the midpoint. Tolerance is analytic
/// (R-ORACLE-5): slopes are iid Uniform[l, u] with std `(u-l)/sqrt(12)`;
/// for N = 10_000 the standard error of the mean is
/// `(u-l)/sqrt(12*N) ≈ 6.02e-4`; we allow 5σ ≈ 3.01e-3.
#[test]
fn rrelu_training_slope_distribution_sanity() {
    let _guard = default_rng_test_lock();
    const N: usize = 10_000;
    let data = vec![-1.0f64; N];
    let x = cpu_f64(&data, &[N], false);
    manual_seed(99).unwrap();
    let y = rrelu(&x, LOWER, UPPER, true).unwrap().data_vec().unwrap();
    // slope_i = out_i / x_i = -out_i for x_i = -1.
    let slopes: Vec<f64> = y.iter().map(|&v| -v).collect();
    for (i, &s) in slopes.iter().enumerate() {
        assert!(
            (LOWER..UPPER).contains(&s),
            "slope[{i}] = {s} outside [lower, upper) = [{LOWER}, {UPPER})"
        );
    }
    let mean = slopes.iter().sum::<f64>() / N as f64;
    let midpoint = f64::midpoint(LOWER, UPPER);
    let five_sigma = 5.0 * (UPPER - LOWER) / (12.0 * N as f64).sqrt();
    assert!(
        (mean - midpoint).abs() < five_sigma,
        "sample mean {mean} departs from midpoint {midpoint} by more than 5σ = {five_sigma}"
    );
}

/// Eval mode (training=false) stays the deterministic mean slope and matches
/// torch exactly. Live torch oracle:
/// ```python
/// >>> F.rrelu(torch.tensor([-1.0,-2.0,3.0], dtype=torch.float64),
/// ...         lower=0.125, upper=1/3, training=False)
/// tensor([-0.22916666666666666, -0.4583333333333333, 3.0])
/// >>> F.leaky_relu(torch.tensor([-1.0,-2.0,3.0], dtype=torch.float64), (0.125+1/3)/2)
/// tensor([-0.22916666666666666, -0.4583333333333333, 3.0])  # identical
/// ```
/// (torch 2.11.0+cu130, CPU — eval delegates to leaky_relu with slope
/// `(lower+upper)/2` per `aten/src/ATen/native/Activation.cpp:624-630`)
#[test]
fn rrelu_eval_matches_torch_mean_slope_exactly() {
    let _guard = default_rng_test_lock();
    let x = cpu_f64(&[-1.0, -2.0, 3.0], &[3], false);
    // Eval must NOT consume RNG state: seed, draw eval, then check the next
    // training call still matches the seed-42 stream from element 0.
    manual_seed(42).unwrap();
    let y = rrelu(&x, LOWER, UPPER, false).unwrap().data_vec().unwrap();
    let got: Vec<u64> = y.iter().map(|v| v.to_bits()).collect();
    let expected: Vec<u64> = vec![
        (-0.229_166_666_666_666_66f64).to_bits(),
        (-0.458_333_333_333_333_3f64).to_bits(),
        3.0f64.to_bits(),
    ];
    assert_eq!(got, expected, "eval path diverged from torch (got {y:?})");

    let t = rrelu(&x, LOWER, UPPER, true).unwrap().data_vec().unwrap();
    assert_eq!(
        t[0].to_bits(),
        0xbfc1_8d00_5498_88c0u64,
        "eval mode must not consume RNG state (training draw after eval drifted)"
    );
}
