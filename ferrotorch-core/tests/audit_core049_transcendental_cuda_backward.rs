//! Regression tests for audit finding CORE-049 (crosslink #1743).
//!
//! The unary transcendental family in `grad_fns/transcendental.rs` computes
//! its CUDA forward through `unary_map` (documented host round trip, output
//! restored to CUDA), but pre-fix the backwards of `tan asin acos atan sinh
//! cosh asinh acosh atanh exp2 expm1 log2 log10 log1p frac sinc`
//! unconditionally called the CPU-only `data()` on the saved tensors and the
//! incoming gradient — so a successful CUDA forward was followed by a
//! `GpuTensorNotAccessible` backward. The rounding/sign family (`ceil floor
//! round trunc sign`) instead routed through `zeros_like_tensor`, which
//! unconditionally created CPU storage — backward "succeeded" but silently
//! delivered a CPU gradient for a CUDA leaf.
//!
//! Post-fix contract (the `leaky_relu`/CORE-045 house pattern at
//! `activation.rs`): the backward builds the derivative factor via
//! `unary_map` (documented host round trip per R-LOUD-2) on the saved
//! input/output and combines it with `grad_output` using the CUDA-aware
//! `grad_fns::arithmetic` ops; `zeros_like_tensor` preserves the saved
//! input's device.
//!
//! Every numerical expectation below is from a live torch session
//! (R-ORACLE-1(b)):
//!
//! ```python
//! # torch 2.11.0+cu130
//! >>> t = torch.tensor(X, dtype=torch.float64, requires_grad=True)
//! >>> o = torch.OP(t); o.sum().backward()
//! >>> o, t.grad   # fwd/grad pairs quoted per test
//! ```
//!
//! Device assertions follow R-ORACLE-3: the result AND the gradient must
//! reside on `Cuda(0)` — readback transparency is not accepted.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::grad_fns::transcendental as tr;
use ferrotorch_core::{Device, FerrotorchResult, Tensor, TensorStorage};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the CORE-049 regression suite");
    });
}

// Domain-respecting input vectors (shared with the torch oracle session).
/// Whole-line domain: tan/atan/sinh/cosh/asinh/exp2/expm1/sinc.
const GEN: [f64; 5] = [-2.0, -0.5, 0.0, 0.5, 2.0];
/// Open interval (-1, 1): asin/acos/atanh.
const OPEN: [f64; 5] = [-0.75, -0.5, 0.0, 0.5, 0.75];
/// x > 1: acosh.
const GT1: [f64; 4] = [1.5, 2.0, 3.0, 5.0];
/// x > 0: log2/log10.
const POS: [f64; 4] = [0.5, 1.0, 2.0, 4.0];
/// x > -1: log1p.
const GTM1: [f64; 4] = [-0.5, 0.0, 0.5, 2.0];
/// Mixed-sign fractional values: frac + rounding/sign family.
const RND: [f64; 5] = [-1.7, -0.5, 0.0, 0.3, 2.5];

/// Build a true CUDA LEAF.
///
/// CORE-012 (#1706): `.to(device)` of a requires-grad leaf is a
/// differentiable copy — a NON-leaf whose gradients accumulate on the
/// ORIGINAL CPU leaf. These tests assert CUDA-resident `.grad()` on the
/// uploaded tensor, so they need a real CUDA leaf — torch's
/// `x.to('cuda').detach().requires_grad_(True)` idiom.
fn cuda_leaf_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("upload to cuda:0")
        .requires_grad_(true)
}

fn cuda_leaf_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("upload to cuda:0")
        .requires_grad_(true)
}

fn read_back_f64(t: &Tensor<f64>, what: &str) -> Vec<f64> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "{what} expected on Cuda(0) but resides on {:?} — silent CPU fallback (CORE-049)",
        t.device()
    );
    t.cpu().expect("D2H readback").data_vec().expect("read")
}

fn read_back_f32(t: &Tensor<f32>, what: &str) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "{what} expected on Cuda(0) but resides on {:?} — silent CPU fallback (CORE-049)",
        t.device()
    );
    t.cpu().expect("D2H readback").data_vec().expect("read")
}

/// f64 lane tolerance rationale (R-ORACLE-5): values transit one lossless
/// f64 D2H/H2D round trip; each forward/backward evaluates at most two libm
/// transcendentals on the host (≤ 1 ulp each ≈ 1.6e-15 relative at the max
/// magnitude 7.39 = expm1'(2)). 1e-12 absolute gives > 2.5 orders of margin.
const TOL_F64: f64 = 1e-12;

/// f32 lane tolerance rationale (R-ORACLE-5): f32 eps = 1.19e-7; the largest
/// expectation magnitude is 7.39 (ulp ≈ 4.8e-7) and forward+backward chain at
/// most two f32 transcendental evaluations plus one multiply (a few ulp,
/// ≤ ~2e-6 absolute at these magnitudes). 1e-5 absolute gives ≥ 5x margin
/// while still rejecting any wrong-formula or wrong-device value.
const TOL_F32: f32 = 1e-5;

fn assert_close_f64(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < TOL_F64,
            "{label}: index {i} diverges (actual={a:.17}, expected={e:.17})"
        );
    }
}

fn assert_close_f32(actual: &[f32], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (f64::from(a) - e).abs() < f64::from(TOL_F32),
            "{label}: index {i} diverges (actual={a:.9}, expected={e:.17})"
        );
    }
}

type OpF64 = fn(&Tensor<f64>) -> FerrotorchResult<Tensor<f64>>;
type OpF32 = fn(&Tensor<f32>) -> FerrotorchResult<Tensor<f32>>;

/// Shared lane (f64 + f32): CUDA forward with requires_grad=true, backward
/// through `sum`, forward values + gradient VALUES against the live torch
/// oracle, result and gradient both device-asserted (R-ORACLE-3). Pre-fix the
/// 16 transcendental ops fail the `backward()` call with
/// `GpuTensorNotAccessible`; the 5 rounding/sign ops fail the gradient device
/// assertion (silent CPU grad).
fn run_cuda_grad_lanes(name: &str, op64: OpF64, op32: OpF32, x: &[f64], fwd: &[f64], grad: &[f64]) {
    ensure_cuda_backend();

    // f64 lane
    let xt = cuda_leaf_f64(x);
    let out = op64(&xt).unwrap_or_else(|e| panic!("{name} f64: CUDA forward failed: {e:?}"));
    assert_close_f64(&read_back_f64(&out, &format!("{name} f64 fwd")), fwd, name);
    let loss = reduce_sum(&out).expect("sum-to-scalar");
    loss.backward()
        .unwrap_or_else(|e| panic!("{name} f64: CUDA backward failed (CORE-049): {e:?}"));
    let g = xt.grad().unwrap().expect("grad must reach the CUDA leaf");
    assert_close_f64(
        &read_back_f64(&g, &format!("{name} f64 grad")),
        grad,
        &format!("{name} f64 grad"),
    );

    // f32 lane (same oracle values, f32 tolerance band)
    let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let xt = cuda_leaf_f32(&x32);
    let out = op32(&xt).unwrap_or_else(|e| panic!("{name} f32: CUDA forward failed: {e:?}"));
    assert_close_f32(&read_back_f32(&out, &format!("{name} f32 fwd")), fwd, name);
    let loss = reduce_sum(&out).expect("sum-to-scalar");
    loss.backward()
        .unwrap_or_else(|e| panic!("{name} f32: CUDA backward failed (CORE-049): {e:?}"));
    let g = xt.grad().unwrap().expect("grad must reach the CUDA leaf");
    assert_close_f32(
        &read_back_f32(&g, &format!("{name} f32 grad")),
        grad,
        &format!("{name} f32 grad"),
    );
}

// ---------------------------------------------------------------------------
// The 16 error-backward ops (pre-fix: GpuTensorNotAccessible on backward)
// ---------------------------------------------------------------------------

/// ```python
/// >>> torch.tan  # X = GEN
/// fwd  = [2.185039863261519, -0.5463024898437905, 0.0, 0.5463024898437905, -2.185039863261519]
/// grad = [5.774399204041917, 1.2984464104095248, 1.0, 1.2984464104095248, 5.774399204041917]
/// ```
#[test]
fn cuda_grad_tan() {
    run_cuda_grad_lanes(
        "tan",
        tr::tan,
        tr::tan,
        &GEN,
        &[
            2.185_039_863_261_519,
            -0.546_302_489_843_790_5,
            0.0,
            0.546_302_489_843_790_5,
            -2.185_039_863_261_519,
        ],
        &[
            5.774_399_204_041_917,
            1.298_446_410_409_524_8,
            1.0,
            1.298_446_410_409_524_8,
            5.774_399_204_041_917,
        ],
    );
}

/// ```python
/// >>> torch.asin  # X = OPEN
/// fwd  = [-0.848062078981481, -0.5235987755982989, 0.0, 0.5235987755982989, 0.848062078981481]
/// grad = [1.5118578920369088, 1.1547005383792517, 1.0, 1.1547005383792517, 1.5118578920369088]
/// ```
#[test]
#[allow(
    clippy::approx_constant,
    reason = "asin(±0.5) = ±π/6 is the live torch oracle value quoted above, \
              not a hand-typed approximation of FRAC_PI_6"
)]
fn cuda_grad_asin() {
    run_cuda_grad_lanes(
        "asin",
        tr::asin,
        tr::asin,
        &OPEN,
        &[
            -0.848_062_078_981_481,
            -0.523_598_775_598_298_9,
            0.0,
            0.523_598_775_598_298_9,
            0.848_062_078_981_481,
        ],
        &[
            1.511_857_892_036_908_8,
            1.154_700_538_379_251_7,
            1.0,
            1.154_700_538_379_251_7,
            1.511_857_892_036_908_8,
        ],
    );
}

/// ```python
/// >>> torch.acos  # X = OPEN
/// fwd  = [2.4188584057763776, 2.0943951023931957, 1.5707963267948966, 1.0471975511965976, 0.7227342478134156]
/// grad = [-1.5118578920369088, -1.1547005383792517, -1.0, -1.1547005383792517, -1.5118578920369088]
/// ```
#[test]
fn cuda_grad_acos() {
    run_cuda_grad_lanes(
        "acos",
        tr::acos,
        tr::acos,
        &OPEN,
        &[
            2.418_858_405_776_377_6,
            2.094_395_102_393_195_7,
            std::f64::consts::FRAC_PI_2,
            1.047_197_551_196_597_6,
            0.722_734_247_813_415_6,
        ],
        &[
            -1.511_857_892_036_908_8,
            -1.154_700_538_379_251_7,
            -1.0,
            -1.154_700_538_379_251_7,
            -1.511_857_892_036_908_8,
        ],
    );
}

/// ```python
/// >>> torch.atan  # X = GEN
/// fwd  = [-1.1071487177940904, -0.4636476090008061, 0.0, 0.4636476090008061, 1.1071487177940904]
/// grad = [0.2, 0.8, 1.0, 0.8, 0.2]
/// ```
#[test]
fn cuda_grad_atan() {
    run_cuda_grad_lanes(
        "atan",
        tr::atan,
        tr::atan,
        &GEN,
        &[
            -1.107_148_717_794_090_4,
            -0.463_647_609_000_806_1,
            0.0,
            0.463_647_609_000_806_1,
            1.107_148_717_794_090_4,
        ],
        &[0.2, 0.8, 1.0, 0.8, 0.2],
    );
}

/// ```python
/// >>> torch.sinh  # X = GEN
/// fwd  = [-3.626860407847019, -0.5210953054937474, 0.0, 0.5210953054937474, 3.626860407847019]
/// grad = [3.7621956910836314, 1.1276259652063807, 1.0, 1.1276259652063807, 3.7621956910836314]
/// ```
#[test]
fn cuda_grad_sinh() {
    run_cuda_grad_lanes(
        "sinh",
        tr::sinh,
        tr::sinh,
        &GEN,
        &[
            -3.626_860_407_847_019,
            -0.521_095_305_493_747_4,
            0.0,
            0.521_095_305_493_747_4,
            3.626_860_407_847_019,
        ],
        &[
            3.762_195_691_083_631_4,
            1.127_625_965_206_380_7,
            1.0,
            1.127_625_965_206_380_7,
            3.762_195_691_083_631_4,
        ],
    );
}

/// ```python
/// >>> torch.cosh  # X = GEN
/// fwd  = [3.7621956910836314, 1.1276259652063807, 1.0, 1.1276259652063807, 3.7621956910836314]
/// grad = [-3.626860407847019, -0.5210953054937474, 0.0, 0.5210953054937474, 3.626860407847019]
/// ```
#[test]
fn cuda_grad_cosh() {
    run_cuda_grad_lanes(
        "cosh",
        tr::cosh,
        tr::cosh,
        &GEN,
        &[
            3.762_195_691_083_631_4,
            1.127_625_965_206_380_7,
            1.0,
            1.127_625_965_206_380_7,
            3.762_195_691_083_631_4,
        ],
        &[
            -3.626_860_407_847_019,
            -0.521_095_305_493_747_4,
            0.0,
            0.521_095_305_493_747_4,
            3.626_860_407_847_019,
        ],
    );
}

/// ```python
/// >>> torch.asinh  # X = GEN
/// fwd  = [-1.4436354751788103, -0.48121182505960347, 0.0, 0.48121182505960347, 1.4436354751788103]
/// grad = [0.4472135954999579, 0.8944271909999159, 1.0, 0.8944271909999159, 0.4472135954999579]
/// ```
#[test]
fn cuda_grad_asinh() {
    run_cuda_grad_lanes(
        "asinh",
        tr::asinh,
        tr::asinh,
        &GEN,
        &[
            -1.443_635_475_178_810_3,
            -0.481_211_825_059_603_47,
            0.0,
            0.481_211_825_059_603_47,
            1.443_635_475_178_810_3,
        ],
        &[
            0.447_213_595_499_957_9,
            0.894_427_190_999_915_9,
            1.0,
            0.894_427_190_999_915_9,
            0.447_213_595_499_957_9,
        ],
    );
}

/// ```python
/// >>> torch.acosh  # X = GT1
/// fwd  = [0.9624236501192069, 1.3169578969248166, 1.762747174039086, 2.2924316695611777]
/// grad = [0.8944271909999159, 0.5773502691896258, 0.35355339059327373, 0.20412414523193154]
/// ```
#[test]
fn cuda_grad_acosh() {
    run_cuda_grad_lanes(
        "acosh",
        tr::acosh,
        tr::acosh,
        &GT1,
        &[
            0.962_423_650_119_206_9,
            1.316_957_896_924_816_6,
            1.762_747_174_039_086,
            2.292_431_669_561_177_7,
        ],
        &[
            0.894_427_190_999_915_9,
            0.577_350_269_189_625_8,
            0.353_553_390_593_273_73,
            0.204_124_145_231_931_54,
        ],
    );
}

/// ```python
/// >>> torch.atanh  # X = OPEN
/// fwd  = [-0.9729550745276566, -0.5493061443340548, 0.0, 0.5493061443340548, 0.9729550745276566]
/// grad = [2.2857142857142856, 1.3333333333333333, 1.0, 1.3333333333333333, 2.2857142857142856]
/// ```
#[test]
fn cuda_grad_atanh() {
    run_cuda_grad_lanes(
        "atanh",
        tr::atanh,
        tr::atanh,
        &OPEN,
        &[
            -0.972_955_074_527_656_6,
            -0.549_306_144_334_054_8,
            0.0,
            0.549_306_144_334_054_8,
            0.972_955_074_527_656_6,
        ],
        &[
            2.285_714_285_714_285_6,
            1.333_333_333_333_333_3,
            1.0,
            1.333_333_333_333_333_3,
            2.285_714_285_714_285_6,
        ],
    );
}

/// ```python
/// >>> torch.exp2  # X = GEN
/// fwd  = [0.25, 0.7071067811865476, 1.0, 1.4142135623730951, 4.0]
/// grad = [0.17328679513998632, 0.4901290717342736, 0.6931471805599453, 0.9802581434685472, 2.772588722239781]
/// ```
#[test]
fn cuda_grad_exp2() {
    run_cuda_grad_lanes(
        "exp2",
        tr::exp2,
        tr::exp2,
        &GEN,
        &[
            0.25,
            std::f64::consts::FRAC_1_SQRT_2,
            1.0,
            std::f64::consts::SQRT_2,
            4.0,
        ],
        &[
            0.173_286_795_139_986_32,
            0.490_129_071_734_273_6,
            std::f64::consts::LN_2,
            0.980_258_143_468_547_2,
            2.772_588_722_239_781,
        ],
    );
}

/// ```python
/// >>> torch.expm1  # X = GEN
/// fwd  = [-0.8646647167633873, -0.3934693402873666, 0.0, 0.6487212707001282, 6.38905609893065]
/// grad = [0.1353352832366127, 0.6065306597126334, 1.0, 1.6487212707001282, 7.38905609893065]
/// ```
#[test]
fn cuda_grad_expm1() {
    run_cuda_grad_lanes(
        "expm1",
        tr::expm1,
        tr::expm1,
        &GEN,
        &[
            -0.864_664_716_763_387_3,
            -0.393_469_340_287_366_6,
            0.0,
            0.648_721_270_700_128_2,
            6.389_056_098_930_65,
        ],
        &[
            0.135_335_283_236_612_7,
            0.606_530_659_712_633_4,
            1.0,
            1.648_721_270_700_128_2,
            7.389_056_098_930_65,
        ],
    );
}

/// ```python
/// >>> torch.log2  # X = POS
/// fwd  = [-1.0, 0.0, 1.0, 2.0]
/// grad = [2.8853900817779268, 1.4426950408889634, 0.7213475204444817, 0.36067376022224085]
/// ```
#[test]
fn cuda_grad_log2() {
    run_cuda_grad_lanes(
        "log2",
        tr::log2,
        tr::log2,
        &POS,
        &[-1.0, 0.0, 1.0, 2.0],
        &[
            2.885_390_081_777_926_8,
            std::f64::consts::LOG2_E,
            0.721_347_520_444_481_7,
            0.360_673_760_222_240_85,
        ],
    );
}

/// ```python
/// >>> torch.log10  # X = POS
/// fwd  = [-0.3010299956639812, 0.0, 0.3010299956639812, 0.6020599913279624]
/// grad = [0.8685889638065037, 0.43429448190325187, 0.21714724095162594, 0.10857362047581297]
/// ```
#[test]
#[allow(
    clippy::approx_constant,
    reason = "log10(±2) = ±log10(2) is the live torch oracle value quoted \
              above, not a hand-typed approximation of LOG10_2"
)]
fn cuda_grad_log10() {
    run_cuda_grad_lanes(
        "log10",
        tr::log10,
        tr::log10,
        &POS,
        &[
            -0.301_029_995_663_981_2,
            0.0,
            0.301_029_995_663_981_2,
            0.602_059_991_327_962_4,
        ],
        &[
            0.868_588_963_806_503_7,
            0.434_294_481_903_251_87,
            0.217_147_240_951_625_94,
            0.108_573_620_475_812_97,
        ],
    );
}

/// ```python
/// >>> torch.log1p  # X = GTM1
/// fwd  = [-0.6931471805599453, 0.0, 0.4054651081081644, 1.0986122886681098]
/// grad = [2.0, 1.0, 0.6666666666666666, 0.3333333333333333]
/// ```
#[test]
fn cuda_grad_log1p() {
    run_cuda_grad_lanes(
        "log1p",
        tr::log1p,
        tr::log1p,
        &GTM1,
        &[
            -std::f64::consts::LN_2,
            0.0,
            0.405_465_108_108_164_4,
            1.098_612_288_668_109_8,
        ],
        &[2.0, 1.0, 0.666_666_666_666_666_6, 0.333_333_333_333_333_3],
    );
}

/// ```python
/// >>> torch.frac  # X = RND
/// fwd  = [-0.7, -0.5, 0.0, 0.3, 0.5]
/// grad = [1.0, 1.0, 1.0, 1.0, 1.0]
/// ```
#[test]
fn cuda_grad_frac() {
    run_cuda_grad_lanes(
        "frac",
        tr::frac,
        tr::frac,
        &RND,
        &[-0.7, -0.5, 0.0, 0.3, 0.5],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
    );
}

/// ```python
/// >>> torch.sinc  # X = GEN
/// fwd  = [-3.8981718325193755e-17, 0.6366197723675814, 1.0, 0.6366197723675814, -3.8981718325193755e-17]
/// grad = [-0.5, 1.2732395447351625, 0.0, -1.2732395447351625, 0.5]
/// ```
#[test]
#[allow(
    clippy::approx_constant,
    reason = "sinc(±0.5) = 2/π is the live torch oracle value quoted above, \
              not a hand-typed approximation of FRAC_2_PI"
)]
fn cuda_grad_sinc() {
    run_cuda_grad_lanes(
        "sinc",
        tr::sinc,
        tr::sinc,
        &GEN,
        &[
            -3.898_171_832_519_375_5e-17,
            0.636_619_772_367_581_4,
            1.0,
            0.636_619_772_367_581_4,
            -3.898_171_832_519_375_5e-17,
        ],
        &[
            -0.5,
            1.273_239_544_735_162_5,
            0.0,
            -1.273_239_544_735_162_5,
            0.5,
        ],
    );
}

// ---------------------------------------------------------------------------
// The 5 silent-CPU-grad ops (pre-fix: backward Ok but gradient on Cpu)
// ---------------------------------------------------------------------------

/// ```python
/// >>> torch.ceil  # X = RND
/// fwd  = [-1.0, -0.0, 0.0, 1.0, 3.0]
/// grad = [0.0, 0.0, 0.0, 0.0, 0.0]
/// ```
#[test]
fn cuda_grad_ceil() {
    run_cuda_grad_lanes(
        "ceil",
        tr::ceil,
        tr::ceil,
        &RND,
        &[-1.0, -0.0, 0.0, 1.0, 3.0],
        &[0.0; 5],
    );
}

/// ```python
/// >>> torch.floor  # X = RND
/// fwd  = [-2.0, -1.0, 0.0, 0.0, 2.0]
/// grad = [0.0, 0.0, 0.0, 0.0, 0.0]
/// ```
#[test]
fn cuda_grad_floor() {
    run_cuda_grad_lanes(
        "floor",
        tr::floor,
        tr::floor,
        &RND,
        &[-2.0, -1.0, 0.0, 0.0, 2.0],
        &[0.0; 5],
    );
}

/// ```python
/// >>> torch.round  # X = RND  (RNE: round(2.5) == 2.0, round(-0.5) == -0.0)
/// fwd  = [-2.0, -0.0, 0.0, 0.0, 2.0]
/// grad = [0.0, 0.0, 0.0, 0.0, 0.0]
/// ```
#[test]
fn cuda_grad_round() {
    run_cuda_grad_lanes(
        "round",
        tr::round,
        tr::round,
        &RND,
        &[-2.0, -0.0, 0.0, 0.0, 2.0],
        &[0.0; 5],
    );
}

/// ```python
/// >>> torch.trunc  # X = RND
/// fwd  = [-1.0, -0.0, 0.0, 0.0, 2.0]
/// grad = [0.0, 0.0, 0.0, 0.0, 0.0]
/// ```
#[test]
fn cuda_grad_trunc() {
    run_cuda_grad_lanes(
        "trunc",
        tr::trunc,
        tr::trunc,
        &RND,
        &[-1.0, -0.0, 0.0, 0.0, 2.0],
        &[0.0; 5],
    );
}

/// ```python
/// >>> torch.sign  # X = RND
/// fwd  = [-1.0, -1.0, 0.0, 1.0, 1.0]
/// grad = [0.0, 0.0, 0.0, 0.0, 0.0]
/// ```
#[test]
fn cuda_grad_sign() {
    run_cuda_grad_lanes(
        "sign",
        tr::sign,
        tr::sign,
        &RND,
        &[-1.0, -1.0, 0.0, 1.0, 1.0],
        &[0.0; 5],
    );
}

// ---------------------------------------------------------------------------
// NaN propagation spot checks (R-ORACLE-3 + finding's "test every CUDA
// forward together with backward")
// ---------------------------------------------------------------------------

/// ```python
/// >>> t = torch.tensor([float('nan'), 0.5], dtype=torch.float64, requires_grad=True)
/// >>> torch.tan(t)   -> fwd [nan, 0.5463024898437905],  grad [nan, 1.2984464104095248]
/// >>> torch.expm1(t) -> fwd [nan, 0.6487212707001282],  grad [nan, 1.6487212707001282]
/// >>> torch.sinc(t)  -> fwd [nan, 0.6366197723675814],  grad [nan, -1.2732395447351625]
/// >>> torch.sign(t)  -> fwd [0.0, 1.0],                 grad [0.0, 0.0]
/// ```
#[test]
#[allow(
    clippy::approx_constant,
    reason = "sinc(0.5) = 2/π is the live torch oracle value quoted above, \
              not a hand-typed approximation of FRAC_2_PI"
)]
fn cuda_grad_nan_propagation_spot_checks() {
    ensure_cuda_backend();
    let nan_lane = |name: &str, op: OpF64, fwd_exp: &[f64], grad_exp: &[f64]| {
        let x = cuda_leaf_f64(&[f64::NAN, 0.5]);
        let out = op(&x).unwrap_or_else(|e| panic!("{name}: CUDA forward failed: {e:?}"));
        let fwd = read_back_f64(&out, &format!("{name} fwd"));
        for (i, (&a, &e)) in fwd.iter().zip(fwd_exp.iter()).enumerate() {
            if e.is_nan() {
                assert!(a.is_nan(), "{name} fwd[{i}]: expected NaN, got {a}");
            } else {
                assert!((a - e).abs() < TOL_F64, "{name} fwd[{i}]: {a} vs {e}");
            }
        }
        reduce_sum(&out)
            .expect("sum")
            .backward()
            .unwrap_or_else(|e| panic!("{name}: CUDA backward failed (CORE-049): {e:?}"));
        let g = read_back_f64(&x.grad().unwrap().expect("grad"), &format!("{name} grad"));
        for (i, (&a, &e)) in g.iter().zip(grad_exp.iter()).enumerate() {
            if e.is_nan() {
                assert!(a.is_nan(), "{name} grad[{i}]: expected NaN, got {a}");
            } else {
                assert!((a - e).abs() < TOL_F64, "{name} grad[{i}]: {a} vs {e}");
            }
        }
    };
    nan_lane(
        "tan",
        tr::tan,
        &[f64::NAN, 0.546_302_489_843_790_5],
        &[f64::NAN, 1.298_446_410_409_524_8],
    );
    nan_lane(
        "expm1",
        tr::expm1,
        &[f64::NAN, 0.648_721_270_700_128_2],
        &[f64::NAN, 1.648_721_270_700_128_2],
    );
    nan_lane(
        "sinc",
        tr::sinc,
        &[f64::NAN, 0.636_619_772_367_581_4],
        &[f64::NAN, -1.273_239_544_735_162_5],
    );
    // sign(NaN) = 0 (torch.sign CPU kernel semantics) and its grad is the
    // zeros_like backward — exactly 0.0, NOT NaN.
    nan_lane("sign", tr::sign, &[0.0, 1.0], &[0.0, 0.0]);
}

// ---------------------------------------------------------------------------
// Empty-tensor spot checks
// ---------------------------------------------------------------------------

/// ```python
/// >>> t = torch.empty(0, dtype=torch.float64, device='cuda', requires_grad=True)
/// >>> o = torch.tan(t); o.sum().backward()
/// >>> o.shape, t.grad.shape, t.grad.device  -> (0,), (0,), cuda:0
/// ```
/// (same for log1p and ceil — representative of the error-backward and
/// silent-CPU-grad families respectively)
#[test]
fn cuda_grad_empty_tensor_spot_checks() {
    ensure_cuda_backend();
    let empty_lane = |name: &str, op: OpF64| {
        let x = cuda_leaf_f64(&[]);
        let out = op(&x).unwrap_or_else(|e| panic!("{name}: empty CUDA forward failed: {e:?}"));
        assert_eq!(out.shape(), &[0], "{name}: empty forward shape");
        let fwd = read_back_f64(&out, &format!("{name} empty fwd"));
        assert!(fwd.is_empty(), "{name}: empty forward numel");
        reduce_sum(&out)
            .expect("sum of empty")
            .backward()
            .unwrap_or_else(|e| panic!("{name}: empty CUDA backward failed (CORE-049): {e:?}"));
        let g = x.grad().unwrap().expect("grad must reach the CUDA leaf");
        assert_eq!(g.shape(), &[0], "{name}: empty grad shape");
        let gv = read_back_f64(&g, &format!("{name} empty grad"));
        assert!(gv.is_empty(), "{name}: empty grad numel");
    };
    empty_lane("tan", tr::tan);
    empty_lane("log1p", tr::log1p);
    empty_lane("ceil", tr::ceil);
}
