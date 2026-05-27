//! Divergence-coverage test for acto-critic audit of commit `0258ffb0c`
//! (which closed #1259 and #1260 by switching the FORWARD rounding chain to
//! f32). Companion to
//! `divergence_fake_quantize_backward_mask_f32_vs_f64.rs` — same precision-
//! coupling bug, in the per-channel kernel.
//!
//! ## Upstream contract
//!
//! Upstream per-channel kernel at
//! `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:2828-2848`
//! (integer-zero-point branch). The MASK lambda is at `:2830-2834`:
//!
//! ```cpp
//!   cpu_kernel(iter_mask, [=](SelfType self, float scale, int32_t zero_point) -> bool {
//!     float inv_scale = 1.0f / scale;                                            // line 2831
//!     const auto qval = static_cast<int64_t>(zero_point + std::nearbyint(self * inv_scale));  // :2832
//!     return ((quant_min <= qval) && (qval <= quant_max));                       // :2833
//!   });
//! ```
//!
//! and the FORWARD lambda at `:2836-2848` reads the SAME f32 `inv_scale` and
//! produces the SAME `static_cast<int64_t>(zero_point + std::nearbyint(self *
//! inv_scale))`. Both are at f32 precision.
//!
//! ## Divergence
//!
//! After `0258ffb0c`:
//!
//! - `per_channel_dequantize_f64` at `grad_fns/quantize_grad.rs:325-353` was
//!   correctly switched to f32 for the forward.
//! - `per_channel_mask_in_range` at `grad_fns/quantize_grad.rs:363-379` was
//!   NOT switched — it still computes:
//!
//!   ```rust
//!   let inv_scale = 1.0 / scale_f64;
//!   let qval_f = zp_f64 + (x_f64 * inv_scale).round_ties_even();
//!   ```
//!
//! at f64. The dispatcher's "independent mask formula" framing missed that
//! both upstream lambdas share `float inv_scale = 1.0f / scale`.
//!
//! ## Live torch repro (torch 2.11.0+cu130, 2026-05-25)
//!
//! ```python
//! import torch
//!
//! # Case C: x=0.35, scale=0.1, zp=0, [-128, 3]
//! #   f32 chain: 0.35 * (1.0_f32/0.1_f32) = 3.5_f32 exact → banker → 4.
//! #     +zp(0)=4. cast i64=4. mask: -128 <= 4 <= 3 ? FALSE (4 > 3).
//! #   Therefore forward output is dq = (clamp(4,-128,3) - 0) * 0.1 = 0.3,
//! #     and backward grad = 0.0.
//! x = torch.tensor([[0.35]], dtype=torch.float32, requires_grad=True)
//! sc = torch.tensor([0.1], dtype=torch.float32)
//! zp = torch.tensor([0], dtype=torch.int32)
//! out = torch.fake_quantize_per_channel_affine(x, sc, zp, 1, -128, 3)
//! out.sum().backward()
//! assert abs(out.detach().item() - 0.30000001192092896) < 1e-6
//! assert x.grad.item() == 0.0      # upstream backward mask=False
//! ```
//!
//! Ferrotorch (post-`0258ffb0c`):
//!
//! - Forward (post-fix): `qval_i64 (f32 chain) = 4`, clamp to qmax=3, dq=0.3.
//!   Matches upstream.
//! - Backward (still buggy): `per_channel_mask_in_range` runs f64,
//!   `qval_f64 = 0 + 3.4999998882... → banker → 3`, cast i64 = 3. Mask:
//!   `-128 <= 3 <= 3` → TRUE → grad=1.0. **Disagrees with upstream 0.0**.
//!
//! ## R-CHAR-3 compliance
//!
//! Expected gradient `0.0` was captured live from
//! `torch.fake_quantize_per_channel_affine` against torch 2.11.0+cu130 on
//! 2026-05-25. Repro snippet above is runnable.
//!
//! ## Tracking
//!
//! Un-`#[ignore]`d: blocker. The forward fix is incomplete without the
//! corresponding backward-mask fix; they share a precision contract per
//! `QuantizedOpKernels.cpp:2831` (`float inv_scale`) and `:2838` (same).

use ferrotorch_core::autograd::backward;
use ferrotorch_core::grad_fns::quantize_grad::fake_quantize_per_channel_affine;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn t(data: Vec<f32>, shape: Vec<usize>, req_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, req_grad).unwrap()
}

fn ti64(data: Vec<i64>, shape: Vec<usize>) -> IntTensor<i64> {
    IntTensor::from_vec(data, shape).unwrap()
}

/// Case C — upstream mask = False (grad = 0.0). Ferrotorch backward (f64) = True.
#[test]
fn per_channel_backward_mask_matches_torch_f32_chain_x035_qmax3() {
    // Live torch reference (torch 2.11.0+cu130, 2026-05-25):
    //   >>> x = torch.tensor([[0.35]], dtype=torch.float32, requires_grad=True)
    //   >>> sc = torch.tensor([0.1], dtype=torch.float32)
    //   >>> zp = torch.tensor([0], dtype=torch.int32)
    //   >>> out = torch.fake_quantize_per_channel_affine(x, sc, zp, 1, -128, 3)
    //   >>> out.sum().backward()
    //   >>> out.detach().item(), x.grad.item()
    //   (0.30000001192092896, 0.0)
    //
    // Upstream's mask at QuantizedOpKernels.cpp:2832-2833 casts to i64 BEFORE
    // the comparison, using the same f32 `inv_scale` as the forward at
    // :2838-2845:
    //     0.35_f32 * (1.0_f32 / 0.1_f32) = 3.5_f32 exact
    //     banker rounds 3.5 → 4 (even)
    //     +zp(0) = 4. cast i64 = 4.
    //     mask: (-128 <= 4) && (4 <= 3) → FALSE.
    // Forward output: clamp(4, -128, 3) - 0 = 3, dq = 3 * 0.1 = 0.3.
    let input = t(vec![0.35_f32], vec![1, 1], true);
    let scale = t(vec![0.1_f32], vec![1], false);
    let zp = ti64(vec![0], vec![1]);
    let out = fake_quantize_per_channel_affine(&input, &scale, &zp, 1, -128, 3).unwrap();
    let s = sum(&out).unwrap();
    backward(&s).unwrap();
    let grad = input.grad().unwrap().unwrap();
    let grad_data = grad.data().unwrap();
    let expected_torch_grad: f32 = 0.0;
    assert_eq!(
        grad_data[0],
        expected_torch_grad,
        "fake_quantize_per_channel_affine backward mask at x=0.35_f32, scale=0.1, \
         zp=0, axis=1, [qmin=-128, qmax=3]: torch returns grad={expected_torch_grad} \
         (upstream's f32 mask at QuantizedOpKernels.cpp:2832 yields qval_i64=4, which \
         FAILS the `qval <= quant_max=3` check). Ferrotorch returned grad={got}: \
         the forward correctly clamps (output ≈ 0.3) but `per_channel_mask_in_range` \
         at quantize_grad.rs:363-379 still runs the mask at f64, giving qval_i64=3 \
         which incorrectly PASSES the mask. The dispatcher's claim that 'backward \
         mask is independent' is wrong — upstream :2831 reads \
         `float inv_scale = 1.0f / scale` (same as forward :2838).",
        got = grad_data[0],
    );
}

/// Companion: assert the FORWARD output is correct at the same boundary
/// (sanity check that the forward fix landed).
///
/// `#[allow(clippy::excessive_precision)]` because the expected value is the
/// f32-precision result captured live from torch and quoted with full f64
/// precision to make the upstream byte sequence visible in the source; the
/// downcast-to-f32 is intentional and traces upstream's `.item()` print.
#[test]
#[allow(clippy::excessive_precision)]
fn per_channel_forward_clamps_at_f32_boundary_x035_qmax3_control() {
    // Same inputs as the backward test above. Upstream forward = 0.3000...
    let input = t(vec![0.35_f32], vec![1, 1], false);
    let scale = t(vec![0.1_f32], vec![1], false);
    let zp = ti64(vec![0], vec![1]);
    let out = fake_quantize_per_channel_affine(&input, &scale, &zp, 1, -128, 3).unwrap();
    let actual = out.data().unwrap();
    // Upstream: dq = clamp(qval=4, -128, 3) * scale = 3 * 0.1 = 0.3 (with f32
    // rounding, the dequant tail materializes as 0.30000001192...).
    let expected_torch: f32 = 0.30000001192092896;
    assert!(
        (actual[0] - expected_torch).abs() < 1e-6,
        "forward control: torch expects {expected_torch}, got {got}. If this fails \
         the forward fix in 0258ffb0c regressed; if this passes but the companion \
         backward-mask test fails, the divergence is exclusively in \
         per_channel_mask_in_range.",
        got = actual[0],
    );
}
