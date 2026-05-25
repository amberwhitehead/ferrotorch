//! Audit: live-torch boundary cases vs ferrotorch post-f8d737500.
//!
//! Live torch references (torch 2.11.0+cu130, 2026-05-25):
//!
//! Small scale 1e-10:
//!   x=-1e-09: out=-9.999999717180685e-10, grad=1.0
//!   x= 0.0:   out= 0.0,                   grad=1.0
//!   x= 1e-09: out= 9.999999717180685e-10, grad=1.0
//!   x= 1.5e-10: out= 2.000000026702864e-10, grad=1.0
//!   x= 2.5e-10: out= 2.000000026702864e-10, grad=1.0
//!
//! Large scale 1e10:
//!   x= 1e9:  out= 0.0,             grad=1.0
//!   x= 5e9:  out= 0.0,             grad=1.0
//!   x= 1e10: out= 1e10,            grad=1.0
//!   x= 5e10: out= 49999998976.0,   grad=1.0
//!   x= 1e11: out= 99999997952.0,   grad=1.0
//!
//! Non-finite inputs, scale=0.1, zp=0, [-128, 127]:
//!   x=+inf: out=12.699999809265137,  grad=0.0
//!   x=-inf: out=-12.800000190734863, grad=0.0
//!   x= nan: out=-12.800000190734863, grad=0.0
//!
//! Per-channel non-finite: scale=0.1, zp=0:
//!   x=[+inf, -inf, nan]: out=[-12.8, -12.8, -12.8], grad=[0, 0, 0]
//!
//! NOTE for x=+inf: upstream returns out=12.699999... (qmax=127 case).
//! Per-tensor upstream uses `static_cast<int64_t>(std::fmin(std::fmax(qval_f, qmin), qmax))`:
//!   qval_f = 0 + nearbyint(+inf * 10) = nearbyint(+inf) = +inf
//!   std::fmax(+inf, -128) = +inf
//!   std::fmin(+inf, 127) = 127
//!   static_cast<int64_t>(127) = 127
//!   out = 127 * 0.1 = 12.7
//! That matches.
//!
//! Per-channel upstream uses `static_cast<int64_t>(qval_f)` BEFORE clamp:
//!   qval_f = +inf, static_cast<int64_t>(+inf) = INT64_MIN on x86-64 (saturate)
//!   fmax(INT64_MIN, -128) = -128, fmin(-128, 127) = -128
//!   out = -128 * 0.1 = -12.8
//! That matches per-channel +inf → -12.8.

use ferrotorch_core::autograd::backward;
use ferrotorch_core::grad_fns::quantize_grad::{
    fake_quantize_per_channel_affine, fake_quantize_per_tensor_affine,
};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn t(data: Vec<f32>, shape: Vec<usize>, req_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, req_grad).unwrap()
}
fn ti(data: Vec<i64>, shape: Vec<usize>) -> IntTensor<i64> {
    IntTensor::from_vec(data, shape).unwrap()
}

// -- per-tensor non-finite handling --

#[test]
fn pt_inf_input_grad_zero() {
    let x = t(vec![f32::INFINITY], vec![1], true);
    let out = fake_quantize_per_tensor_affine(&x, 0.1, 0, -128, 127).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let g = x.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    // Live torch grad = 0.0 for +inf (mask is False because qval_f=+inf > qmax=127)
    assert_eq!(gd[0], 0.0f32, "+inf grad expected 0.0 (torch); got {}", gd[0]);
}

#[test]
fn pt_neg_inf_input_grad_zero() {
    let x = t(vec![f32::NEG_INFINITY], vec![1], true);
    let out = fake_quantize_per_tensor_affine(&x, 0.1, 0, -128, 127).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let g = x.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    assert_eq!(gd[0], 0.0f32, "-inf grad expected 0.0 (torch); got {}", gd[0]);
}

#[test]
fn pt_nan_input_grad_zero() {
    let x = t(vec![f32::NAN], vec![1], true);
    let out = fake_quantize_per_tensor_affine(&x, 0.1, 0, -128, 127).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let g = x.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    assert_eq!(gd[0], 0.0f32, "NaN grad expected 0.0 (torch); got {}", gd[0]);
}

#[test]
fn pt_small_scale_grad_one() {
    // scale = 1e-10, x = 1e-9, expect grad=1.0 (in range)
    let x = t(vec![1e-9_f32], vec![1], true);
    let out = fake_quantize_per_tensor_affine(&x, 1e-10, 0, -128, 127).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let g = x.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    assert_eq!(gd[0], 1.0f32, "x=1e-9 scale=1e-10 grad expected 1.0; got {}", gd[0]);
}

#[test]
fn pt_large_scale_grad_one() {
    // scale = 1e10, x = 5e10, expect grad=1.0 (5 is in range [-128, 127])
    let x = t(vec![5e10_f32], vec![1], true);
    let out = fake_quantize_per_tensor_affine(&x, 1e10, 0, -128, 127).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let g = x.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    assert_eq!(gd[0], 1.0f32, "x=5e10 scale=1e10 grad expected 1.0; got {}", gd[0]);
}

#[test]
fn pt_huge_input_overflows_qmax() {
    // x=1e30 scale=0.1 -> qval = 1e31 which overflows qmax -> grad=0
    // This is just a check the float compare correctly handles big floats.
    let x = t(vec![1e30_f32], vec![1], true);
    let out = fake_quantize_per_tensor_affine(&x, 0.1, 0, -128, 127).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let g = x.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    assert_eq!(gd[0], 0.0f32, "huge x grad expected 0.0; got {}", gd[0]);
}

// -- per-channel non-finite handling --

#[test]
fn pc_inf_input_grad_zero() {
    let x = t(vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN], vec![1, 3], true);
    let sc = t(vec![0.1, 0.1, 0.1], vec![3], false);
    let zp = ti(vec![0, 0, 0], vec![3]);
    let out = fake_quantize_per_channel_affine(&x, &sc, &zp, 1, -128, 127).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let g = x.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    assert_eq!(gd[0], 0.0f32, "+inf grad expected 0.0; got {}", gd[0]);
    assert_eq!(gd[1], 0.0f32, "-inf grad expected 0.0; got {}", gd[1]);
    assert_eq!(gd[2], 0.0f32, "NaN grad expected 0.0; got {}", gd[2]);
}

// -- Additional boundary cases (1/3 scale ties) --
// Live torch (torch 2.11.0+cu130, 2026-05-25):
//   x=0.1666666667 scale=1/3 qmax=0: grad=1.0
//   x=0.5         scale=1/3 qmax=1: grad=0.0
//   x=0.8333333   scale=1/3 qmax=2: grad=1.0
//   x=1.1666666   scale=1/3 qmax=3: grad=0.0
//   x=1.5         scale=1/3 qmax=4: grad=1.0
//   x=1.8333333   scale=1/3 qmax=5: grad=0.0
//   x=2.1666667   scale=1/3 qmax=6: grad=1.0

#[test]
fn pt_third_scale_boundary_x0166_qmax0() {
    let x = t(vec![1.0_f32 / 6.0], vec![1], true);
    let out = fake_quantize_per_tensor_affine(&x, 1.0/3.0, 0, -128, 0).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let g = x.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    assert_eq!(gd[0], 1.0_f32, "x=1/6 scale=1/3 qmax=0: torch grad=1.0; got {}", gd[0]);
}

#[test]
fn pt_third_scale_boundary_x05_qmax1() {
    let x = t(vec![0.5_f32], vec![1], true);
    let out = fake_quantize_per_tensor_affine(&x, 1.0/3.0, 0, -128, 1).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let g = x.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    assert_eq!(gd[0], 0.0_f32, "x=0.5 scale=1/3 qmax=1: torch grad=0.0; got {}", gd[0]);
}

#[test]
fn pt_third_scale_boundary_x0833_qmax2() {
    let x = t(vec![5.0_f32 / 6.0], vec![1], true);
    let out = fake_quantize_per_tensor_affine(&x, 1.0/3.0, 0, -128, 2).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let g = x.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    assert_eq!(gd[0], 1.0_f32, "x=5/6 scale=1/3 qmax=2: torch grad=1.0; got {}", gd[0]);
}

#[test]
fn pt_third_scale_boundary_x15_qmax4() {
    let x = t(vec![1.5_f32], vec![1], true);
    let out = fake_quantize_per_tensor_affine(&x, 1.0/3.0, 0, -128, 4).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let g = x.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    assert_eq!(gd[0], 1.0_f32, "x=1.5 scale=1/3 qmax=4: torch grad=1.0; got {}", gd[0]);
}

// Per-channel multi-element with non-trivial zp and saturation.
// Live torch (torch 2.11.0+cu130, 2026-05-25):
//   x = [[[0.5, 5.0], [1.5, -5.0]]] shape [1,2,2], axis=1
//   sc = [1/3, 0.5], zp = [2, -3]
//   qmin=-10, qmax=4
//   out = [[[0.66666669, 0.66666669], [1.5, -3.5]]]
//   grad= [[[1.0, 0.0], [1.0, 0.0]]]
//
// Channel 0 element 0: x=0.5, sc=1/3, zp=2:
//   qval_f32 = 2 + nearbyint(0.5 * 3.0) = 2 + nearbyint(1.5) = 2 + 2 = 4 (banker)
//   cast i64 = 4. -10 <= 4 <= 4 → True. grad=1.0.
// Channel 0 element 1: x=5.0, sc=1/3, zp=2:
//   qval_f32 = 2 + nearbyint(5.0 * 3.0) = 2 + 15 = 17. cast i64=17.
//   -10 <= 17 <= 4 → False. grad=0.0.
// Channel 1 element 0: x=1.5, sc=0.5, zp=-3:
//   qval_f32 = -3 + nearbyint(1.5 * 2.0) = -3 + 3 = 0. cast i64=0.
//   -10 <= 0 <= 4 → True. grad=1.0.
// Channel 1 element 1: x=-5.0, sc=0.5, zp=-3:
//   qval_f32 = -3 + nearbyint(-5.0 * 2.0) = -3 + (-10) = -13. cast i64=-13.
//   -10 <= -13 → False. grad=0.0.
#[test]
fn pc_mixed_zp_saturation() {
    let x = t(
        vec![0.5, 5.0, 1.5, -5.0],
        vec![1, 2, 2],
        true,
    );
    let sc = t(vec![1.0_f32/3.0, 0.5], vec![2], false);
    let zp = ti(vec![2, -3], vec![2]);
    let out = fake_quantize_per_channel_affine(&x, &sc, &zp, 1, -10, 4).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let g = x.grad().unwrap().unwrap();
    let gd = g.data().unwrap();
    let expected: [f32; 4] = [1.0, 0.0, 1.0, 0.0];
    for i in 0..4 {
        assert_eq!(
            gd[i], expected[i],
            "per-channel mixed-zp/saturation element {i}: torch grad={}; got {}",
            expected[i], gd[i],
        );
    }
}
