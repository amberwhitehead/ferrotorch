//! Divergence-coverage test for acto-critic audit of commit `0258ffb0c`
//! (which closed #1259 and #1260 by switching the FORWARD rounding chain to
//! f32). The audit observes that the FORWARD mask side at upstream
//! `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:2686`
//! (per-tensor) and `:2811`/`:2833` (per-channel) is computed from the SAME
//! f32 `qval_f` (or cast-to-i64 `qval`) as the forward output:
//!
//! ```cpp
//!   auto qval_f = z_point + std::nearbyint(*input_val * inv_scale);  // line :2683
//!   const auto qval = static_cast<int64_t>(std::fmin(std::fmax(qval_f, quant_min), quant_max));
//!   *output_val = (qval - z_point) * sc;                              // line :2685
//!   *mask_val = ((quant_min <= qval_f) && (qval_f <= quant_max));     // line :2686
//! ```
//!
//! All three of `qval_f`, the clamped output, and the mask are computed from
//! `qval_f`, which itself comes from `float inv_scale = 1.0f / sc;` at `:2665`
//! — a single f32 rounding chain that drives BOTH the forward AND the mask.
//!
//! ## Divergence
//!
//! After commit `0258ffb0c`:
//!
//! - Forward at `grad_fns/quantize_grad.rs:243-263` computes `qval_f32` at
//!   f32 precision (the fix).
//! - Backward `FakeQuantizeBackward::backward` at `:685-715` still computes
//!   the mask using an INDEPENDENT f64 chain at `:692-706`:
//!
//!   ```rust
//!   let inv_scale = 1.0_f64 / self.scale;
//!   let qval_f = zp_f64 + (x_f64 * inv_scale).round_ties_even();
//!   if qval_f >= qmin_f64 && qval_f <= qmax_f64 { g } else { zero }
//!   ```
//!
//! For inputs that land on an f32 / f64 split boundary, the forward's
//! `qval_f32` and the backward's `qval_f64` round to DIFFERENT integers,
//! and when the [qmin, qmax] window touches that integer, the masks
//! disagree: the forward clamps the output to the boundary while the
//! backward independently says "in range" (or vice versa).
//!
//! Upstream (live torch 2.11.0+cu130, 2026-05-25) computes both with the
//! same f32 chain, so the upstream backward gradient agrees with the
//! upstream forward's in-range decision.
//!
//! ## Live torch repro (captured 2026-05-25 vs torch 2.11.0+cu130)
//!
//! ```python
//! import torch
//!
//! # Case A: x=0.025, scale=0.05, zp=64, [-128, 64]
//! #   f32 chain: 0.025*20 = 0.5_f32 exact → banker → 0 → qval_f=64.
//! #   64 IS <= qmax=64, so mask=True, grad passes through.
//! x = torch.tensor([0.025], dtype=torch.float32, requires_grad=True)
//! out = torch.fake_quantize_per_tensor_affine(x, 0.05, 64, -128, 64)
//! out.sum().backward()
//! assert out.detach().item() == 0.0
//! assert x.grad.item() == 1.0      # upstream backward says "in range"
//!
//! # Case B: x=0.175, scale=0.05, zp=64, [-128, 67]
//! #   f32 chain: 0.175*20 = 3.5_f32 exact → banker → 4 → qval_f=68.
//! #   68 is NOT <= qmax=67, so mask=False, grad zeroed.
//! x = torch.tensor([0.175], dtype=torch.float32, requires_grad=True)
//! out = torch.fake_quantize_per_tensor_affine(x, 0.05, 64, -128, 67)
//! out.sum().backward()
//! assert abs(out.detach().item() - 0.15) < 1e-6
//! assert x.grad.item() == 0.0      # upstream backward says "out of range"
//! ```
//!
//! Ferrotorch (post-`0258ffb0c`):
//!
//! - Case A forward returns 0.0 (correct; f32 chain matches upstream).
//!   Backward (independent f64 chain) computes `qval_f64 = 64 + 1 = 65`,
//!   65 > 64 = qmax → mask=False → grad=0.0. **Disagrees with upstream 1.0**.
//!
//! - Case B forward returns 0.15 (clamped at qmax=67 → dq=(67-64)*0.05=0.15).
//!   Backward computes `qval_f64 = 64 + 3 = 67`, 67 <= 67 = qmax →
//!   mask=True → grad=1.0. **Disagrees with upstream 0.0**.
//!
//! ## R-CHAR-3 compliance
//!
//! Expected values were captured live from `torch.fake_quantize_per_tensor_affine`
//! against torch 2.11.0+cu130 on 2026-05-25, NOT copied from ferrotorch.
//! The repro snippet above is runnable.
//!
//! ## Tracking
//!
//! These tests are un-`#[ignore]`d because:
//! 1. The fix for `#1259`/`#1260` claimed "byte-for-byte upstream match" but
//!    the precision contract on the BACKWARD was overlooked.
//! 2. The dispatcher's instruction "Do NOT touch the backward — its mask
//!    formula is independent" is FACTUALLY WRONG: upstream's mask is NOT
//!    independent of the forward's `qval_f` — it IS the same value (cited
//!    line `:2686` reads `*mask_val = ((quant_min <= qval_f) && (qval_f <=
//!    quant_max))`, with `qval_f` defined at `:2683` from the SAME f32
//!    `inv_scale` at `:2665`).
//!
//! Reference: pytorch 2ec0222669f1bcd37b5670ce384f8608c033b158
//! `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:2665`
//! (`float inv_scale = 1.0f / sc;`), `:2683` (`qval_f = z_point +
//! std::nearbyint(*input_val * inv_scale);`), `:2686` (`*mask_val =
//! ((quant_min <= qval_f) && (qval_f <= quant_max));`).

use ferrotorch_core::autograd::backward;
use ferrotorch_core::grad_fns::quantize_grad::fake_quantize_per_tensor_affine;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn t(data: Vec<f32>, shape: Vec<usize>, req_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, req_grad).unwrap()
}

/// Case A — upstream mask = True (grad = 1.0). Ferrotorch backward (f64) = False.
#[test]
fn backward_mask_matches_torch_f32_chain_x025_qmax64() {
    // Live torch reference (torch 2.11.0+cu130, 2026-05-25):
    //   >>> x = torch.tensor([0.025], dtype=torch.float32, requires_grad=True)
    //   >>> out = torch.fake_quantize_per_tensor_affine(x, 0.05, 64, -128, 64)
    //   >>> out.sum().backward()
    //   >>> out.detach().item(), x.grad.item()
    //   (0.0, 1.0)
    //
    // Upstream `qval_f` is computed at f32 precision per
    // QuantizedOpKernels.cpp:2683 — `0.025_f32 * (1.0_f32/0.05_f32) = 0.5_f32`
    // exactly, banker-rounds to 0, +zp(64) = 64. The mask check at :2686 is
    // `(quant_min=-128 <= qval_f=64) && (qval_f=64 <= quant_max=64)` →
    // BOTH true → mask=1 → backward grad passes through unchanged (=1.0
    // because `sum().backward()` seeds grad_output=1.0 elementwise).
    let input = t(vec![0.025_f32], vec![1], true);
    let out = fake_quantize_per_tensor_affine(&input, 0.05, 64, -128, 64).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let grad = input.grad().unwrap().unwrap();
    let grad_data = grad.data().unwrap();
    let expected_torch_grad: f32 = 1.0;
    assert_eq!(
        grad_data[0], expected_torch_grad,
        "fake_quantize_per_tensor_affine backward mask at x=0.025_f32, scale=0.05, \
         zp=64, [qmin=-128, qmax=64]: torch returns grad={expected_torch_grad} (because \
         upstream's mask at QuantizedOpKernels.cpp:2686 reuses the same f32 `qval_f` \
         the forward computed = 64, which IS <= qmax=64). Ferrotorch returned \
         grad={got}. The forward at quantize_grad.rs:262 was fixed to f32; the \
         backward at quantize_grad.rs:692-706 still uses f64 `inv_scale = 1.0_f64 / \
         self.scale` and computes a DIFFERENT `qval_f64 = 65` that fails the mask. \
         Forward and backward must share precision (upstream line :2683 defines \
         qval_f ONCE and lines :2685/:2686 both reuse it).",
        got = grad_data[0],
    );
}

/// Case B — upstream mask = False (grad = 0.0). Ferrotorch backward (f64) = True.
#[test]
fn backward_mask_matches_torch_f32_chain_x175_qmax67() {
    // Live torch reference (torch 2.11.0+cu130, 2026-05-25):
    //   >>> x = torch.tensor([0.175], dtype=torch.float32, requires_grad=True)
    //   >>> out = torch.fake_quantize_per_tensor_affine(x, 0.05, 64, -128, 67)
    //   >>> out.sum().backward()
    //   >>> out.detach().item(), x.grad.item()
    //   (0.15000000596046448, 0.0)
    //
    // Upstream f32 chain: `0.175_f32 * (1.0_f32/0.05_f32) = 3.5_f32` exact,
    // banker-rounds to 4, +zp(64) = 68. The mask check is
    // `(quant_min=-128 <= 68) && (68 <= quant_max=67)` → SECOND fails →
    // mask=0 → grad=0.
    //
    // Ferrotorch's f64 backward chain produces `qval_f64 = 67`, which passes
    // the mask → grad=1.0. This is wrong vs upstream.
    let input = t(vec![0.175_f32], vec![1], true);
    let out = fake_quantize_per_tensor_affine(&input, 0.05, 64, -128, 67).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let grad = input.grad().unwrap().unwrap();
    let grad_data = grad.data().unwrap();
    let expected_torch_grad: f32 = 0.0;
    assert_eq!(
        grad_data[0], expected_torch_grad,
        "fake_quantize_per_tensor_affine backward mask at x=0.175_f32, scale=0.05, \
         zp=64, [qmin=-128, qmax=67]: torch returns grad={expected_torch_grad} (mask \
         at QuantizedOpKernels.cpp:2686 reuses f32 `qval_f`=68, which exceeds \
         qmax=67). Ferrotorch returned grad={got}: the forward correctly clamps at \
         f32 (output ≈ 0.15), but the backward at quantize_grad.rs:692-706 re-derives \
         the mask in f64 and gets qval_f64=67, which passes the mask. The \
         dispatcher's claim that 'the backward's mask formula is independent' is \
         wrong — upstream computes qval_f ONCE per element at f32 and both the \
         forward output and the mask read it.",
        got = grad_data[0],
    );
}
