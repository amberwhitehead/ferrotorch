//! Red-then-green regression tests for audit finding CORE-180 (crosslink
//! #1874): `DivBackward` computes the denominator gradient as
//! `-g * a / (b * b)`, which overflows/underflows at HALF the exponent
//! range of upstream's nested staging. Upstream
//! `pytorch/torch/csrc/autograd/FunctionsManual.cpp:697-708`:
//!
//! ```cpp
//! Tensor div_tensor_other_backward(const Tensor& grad, const Tensor& self,
//!     const Tensor& other, const std::optional<std::string_view>& rounding_mode) {
//!   ...
//!   auto result = -grad * ((self / other) / other).conj();
//!   return handle_r_to_c(other, std::move(result));
//! }
//! ```
//!
//! Every numerical expectation below is quoted from a LIVE torch session
//! (torch 2.11.0+cu130, R-ORACLE-1 path (b)); the generating snippet is
//! pasted next to each expected block, and expectations are pinned via
//! `from_bits` of the torch-printed bit pattern.
//!
//! Tolerance justification (R-ORACLE-5): NONE — assertions are bit-exact.
//! The fixed staging replicates upstream's exact operation sequence
//! `-grad * ((a / b) / b)` where every step is a single correctly-rounded
//! IEEE-754 primitive on both lanes (Rust `f32`/`f64` `/` and `*` on CPU;
//! `div.rn.f32`/`div.rn.f64` and round-to-nearest `mul` without `.ftz` in
//! the ferrotorch-gpu PTX kernels), matching the correctly-rounded
//! primitives torch uses on CPU and CUDA. With the seed cotangent
//! `g = 1.0`, `-g * q` is exact, so the result is bitwise determined.

use ferrotorch_core::grad_fns::arithmetic::div;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;

fn leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}
fn leaf_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}
fn t_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn t_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Runs `(a / b).sum().backward()` with only `b` tracked and returns
/// `b.grad`'s single element.
fn div_bgrad_f32(adata: &[f32], bdata: &[f32]) -> f32 {
    let a = t_f32(adata, &[adata.len()]);
    let b = leaf_f32(bdata, &[bdata.len()]);
    let c = div(&a, &b).expect("forward div");
    sum(&c).expect("sum").backward().expect("backward");
    let g = b.grad().unwrap().expect("b.grad present");
    g.data().unwrap()[0]
}
fn div_bgrad_f64(adata: &[f64], bdata: &[f64]) -> f64 {
    let a = t_f64(adata, &[adata.len()]);
    let b = leaf_f64(bdata, &[bdata.len()]);
    let c = div(&a, &b).expect("forward div");
    sum(&c).expect("sum").backward().expect("backward");
    let g = b.grad().unwrap().expect("b.grad present");
    g.data().unwrap()[0]
}

// torch oracle (torch 2.11.0+cu130):
//   >>> a = torch.tensor([1e10]); b = torch.tensor([1e20], requires_grad=True)
//   >>> (a / b).sum().backward()
//   >>> b.grad  # tensor([-1.0000000031710769e-30])  bits 0x8da24260
// Pre-fix ferrotorch: b*b = 1e40 overflows f32 to inf, -1e10/inf = -0.0.
#[test]
fn core180_div_bgrad_overflow_b_sq_f32() {
    let g = div_bgrad_f32(&[1e10], &[1e20]);
    let expected = f32::from_bits(0x8DA2_4260); // -1.0000000031710769e-30
    assert_eq!(
        g.to_bits(),
        expected.to_bits(),
        "got {g:e} ({:#010x}), torch oracle {expected:e} (0x8da24260)",
        g.to_bits()
    );
}

// torch oracle:
//   >>> a = torch.tensor([1.0]); b = torch.tensor([1e20], requires_grad=True)
//   >>> (a / b).sum().backward()
//   >>> b.grad  # tensor([-9.99994610111476e-41]) — SUBNORMAL, bits 0x800116c2
// Pre-fix ferrotorch: -1.0 / inf = -0.0 (the issue's "subnormal vs -0" case).
#[test]
fn core180_div_bgrad_subnormal_result_f32() {
    let g = div_bgrad_f32(&[1.0], &[1e20]);
    let expected = f32::from_bits(0x8001_16C2); // -9.99994610111476e-41
    assert_eq!(
        g.to_bits(),
        expected.to_bits(),
        "got {g:e} ({:#010x}), torch oracle subnormal -9.99994610111476e-41 (0x800116c2)",
        g.to_bits()
    );
}

// torch oracle:
//   >>> a = torch.tensor([1e-13]); b = torch.tensor([1e-25], requires_grad=True)
//   >>> (a / b).sum().backward()
//   >>> b.grad  # tensor([-9.999999299990512e+36])  bits 0xfcf0bdc1
// Pre-fix ferrotorch: b*b = 1e-50 underflows f32 to 0, -1e-13/0 = -inf
// (the issue's "finite vs inf" case).
#[test]
fn core180_div_bgrad_underflow_b_sq_f32() {
    let g = div_bgrad_f32(&[1e-13], &[1e-25]);
    let expected = f32::from_bits(0xFCF0_BDC1); // -9.999999299990512e+36
    assert_eq!(
        g.to_bits(),
        expected.to_bits(),
        "got {g:e} ({:#010x}), torch oracle -9.999999299990512e+36 (0xfcf0bdc1)",
        g.to_bits()
    );
}

// torch oracle:
//   >>> a = torch.tensor([1.0]); b = torch.tensor([-1e20], requires_grad=True)
//   >>> (a / b).sum().backward()
//   >>> b.grad  # tensor([-9.99994610111476e-41])  bits 0x800116c2
#[test]
fn core180_div_bgrad_negative_b_f32() {
    let g = div_bgrad_f32(&[1.0], &[-1e20]);
    let expected = f32::from_bits(0x8001_16C2);
    assert_eq!(
        g.to_bits(),
        expected.to_bits(),
        "got {g:e} ({:#010x}), torch oracle -9.99994610111476e-41 (0x800116c2)",
        g.to_bits()
    );
}

// Normal range must stay exact:
//   >>> a = torch.tensor([3.0]); b = torch.tensor([2.0], requires_grad=True)
//   >>> (a / b).sum().backward(); b.grad  # tensor([-0.7500])  bits 0xbf400000
#[test]
fn core180_div_bgrad_normal_range_f32() {
    let g = div_bgrad_f32(&[3.0], &[2.0]);
    assert_eq!(g.to_bits(), (-0.75f32).to_bits(), "got {g}, torch -0.75");
}

// f64 lane — torch oracle:
//   >>> a = torch.tensor([1.0], dtype=torch.float64)
//   >>> b = torch.tensor([1e160], dtype=torch.float64, requires_grad=True)
//   >>> (a / b).sum().backward()
//   >>> b.grad  # tensor([-1e-320]) — SUBNORMAL, bits 0x80000000000007e8
// Pre-fix ferrotorch: b*b = 1e320 overflows f64 to inf, -1/inf = -0.0.
#[test]
fn core180_div_bgrad_overflow_b_sq_f64() {
    let g = div_bgrad_f64(&[1.0], &[1e160]);
    let expected = f64::from_bits(0x8000_0000_0000_07E8); // -1e-320
    assert_eq!(
        g.to_bits(),
        expected.to_bits(),
        "got {g:e} ({:#018x}), torch oracle subnormal -1e-320 (0x80000000000007e8)",
        g.to_bits()
    );
}

// torch oracle:
//   >>> a = torch.tensor([1e-200], dtype=torch.float64)
//   >>> b = torch.tensor([1e-180], dtype=torch.float64, requires_grad=True)
//   >>> (a / b).sum().backward()
//   >>> b.grad  # tensor([-9.999999999999999e+159])  bits 0xe126c2d4256ffcc2
// Pre-fix ferrotorch: b*b = 1e-360 underflows f64 to 0, -1e-200/0 = -inf.
#[test]
fn core180_div_bgrad_underflow_b_sq_f64() {
    let g = div_bgrad_f64(&[1e-200], &[1e-180]);
    let expected = f64::from_bits(0xE126_C2D4_256F_FCC2); // -9.999999999999999e+159
    assert_eq!(
        g.to_bits(),
        expected.to_bits(),
        "got {g:e} ({:#018x}), torch oracle -9.999999999999999e+159 (0xe126c2d4256ffcc2)",
        g.to_bits()
    );
}

// The numerator gradient (da = g / b) and broadcasting reduction are
// untouched by the restaging — pin them so the fix can't drift them:
//   >>> a = torch.tensor([3.0, 5.0], requires_grad=True)
//   >>> b = torch.tensor([2.0], requires_grad=True)
//   >>> (a / b).sum().backward()
//   >>> a.grad  # tensor([0.5000, 0.5000])
//   >>> b.grad  # tensor([-2.])    (= -3/4 + -5/4, exact)
#[test]
fn core180_div_backward_broadcast_both_grads_f32() {
    let a = leaf_f32(&[3.0, 5.0], &[2]);
    let b = leaf_f32(&[2.0], &[1]);
    let c = div(&a, &b).expect("forward div");
    sum(&c).expect("sum").backward().expect("backward");
    let ga = a.grad().unwrap().expect("a.grad present");
    let gb = b.grad().unwrap().expect("b.grad present");
    assert_eq!(
        ga.data().unwrap(),
        &[0.5f32, 0.5],
        "a.grad vs torch [0.5, 0.5]"
    );
    assert_eq!(gb.data().unwrap(), &[-2.0f32], "b.grad vs torch [-2.0]");
}

// CUDA lane (gpu feature) — torch oracle on cuda:0 is bit-identical to the
// CPU oracle for every case below (verified live, same bits printed):
//   cuda f32 a=[1e10] b=[1e20]  -> b.grad bits 0x8da24260
//   cuda f32 a=[1.0]  b=[1e20]  -> b.grad bits 0x800116c2
//   cuda f32 a=[1e-13] b=[1e-25]-> b.grad bits 0xfcf0bdc1
//   cuda f32 a=[3.0]  b=[2.0]   -> b.grad bits 0xbf400000
#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the GPU lane of this suite");
        });
    }

    fn div_bgrad_cuda_f32(adata: &[f32], bdata: &[f32]) -> f32 {
        let dev = Device::Cuda(0);
        let a = t_f32(adata, &[adata.len()]).to(dev).expect("upload a");
        let b = leaf_f32(bdata, &[bdata.len()])
            .to(dev)
            .expect("upload b")
            .detach()
            .requires_grad_(true);
        let c = div(&a, &b).expect("forward div cuda");
        sum(&c).expect("sum").backward().expect("backward cuda");
        let g = b.grad().unwrap().expect("b.grad present");
        // R-ORACLE-3: the gradient must be CUDA-resident.
        assert_eq!(g.device(), dev, "b.grad must be CUDA-resident");
        g.cpu().expect("D2H readback").data().unwrap()[0]
    }

    #[test]
    fn core180_gpu_div_bgrad_overflow_b_sq_f32() {
        ensure_cuda_backend();
        let g = div_bgrad_cuda_f32(&[1e10], &[1e20]);
        assert_eq!(
            g.to_bits(),
            0x8DA2_4260,
            "got {g:e} ({:#010x}), torch cuda oracle 0x8da24260",
            g.to_bits()
        );
    }

    #[test]
    fn core180_gpu_div_bgrad_subnormal_result_f32() {
        ensure_cuda_backend();
        let g = div_bgrad_cuda_f32(&[1.0], &[1e20]);
        assert_eq!(
            g.to_bits(),
            0x8001_16C2,
            "got {g:e} ({:#010x}), torch cuda oracle subnormal 0x800116c2",
            g.to_bits()
        );
    }

    #[test]
    fn core180_gpu_div_bgrad_underflow_b_sq_f32() {
        ensure_cuda_backend();
        let g = div_bgrad_cuda_f32(&[1e-13], &[1e-25]);
        assert_eq!(
            g.to_bits(),
            0xFCF0_BDC1,
            "got {g:e} ({:#010x}), torch cuda oracle 0xfcf0bdc1",
            g.to_bits()
        );
    }

    #[test]
    fn core180_gpu_div_bgrad_normal_range_f32() {
        ensure_cuda_backend();
        let g = div_bgrad_cuda_f32(&[3.0], &[2.0]);
        assert_eq!(g.to_bits(), (-0.75f32).to_bits(), "got {g}, torch -0.75");
    }
}
