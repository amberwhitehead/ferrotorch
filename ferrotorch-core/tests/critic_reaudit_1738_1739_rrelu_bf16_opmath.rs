//! ACToR critic re-audit (goal-audit-fix.md step 6) of #1738 / CORE-044
//! `rrelu(training=true)`.
//!
//! Divergence: ferrotorch's `rrelu_train` computes the negative-branch output
//! as `x * r` in the TENSOR dtype `T` after casting the f64 draw straight to
//! `T` (`let r = t_from::<T>(r64, ...); out.push(x * r)` at
//! `ferrotorch-core/src/grad_fns/activation.rs:2925-2935`).
//!
//! Upstream `_rrelu_with_noise_train` at
//! `pytorch/aten/src/ATen/native/Activation.cpp:586-600` computes the draw and
//! the multiply in `opmath_t = at::opmath_type<scalar_t>`, NOT in `scalar_t`:
//!
//! ```cpp
//! using opmath_t = at::opmath_type<scalar_t>;          // line 586
//! ...
//! at::uniform_real_distribution<double> uniform(lower, upper);   // line 597
//! const opmath_t r = (opmath_t)uniform(gen);            // line 598  <-- opmath_t, not scalar_t
//! output_data[i] = input_data[i] * r;                  // line 599  <-- promoted to opmath_t
//! noise_data[i] = r;                                    // line 600
//! ```
//!
//! For `scalar_t = bfloat16`, `opmath_type<BFloat16> == float`. So upstream:
//!   1. casts the f64 draw to `float` (not bf16),
//!   2. promotes the bf16 input to `float`,
//!   3. multiplies in `float`,
//!   4. rounds the float product back to bf16 on store.
//!
//! ferrotorch instead rounds `r` to bf16 FIRST and multiplies in bf16, so for
//! many draws the last-bit rounding differs. The generator's own docstring
//! mis-quotes the upstream as `const scalar_t r = (scalar_t)uniform(gen)`,
//! which is the source of the bug; the existing CORE-044 tests only cover f64
//! and f32 (where `opmath_t == scalar_t`, so the difference is invisible).
//! `f16` (`opmath_type<Half> == float`) is affected identically.
//!
//! Live torch oracle (Python 3, torch 2.11.0+cu130, CPU default generator):
//! ```text
//! >>> import torch
//! >>> x = torch.tensor([-2.1], dtype=torch.bfloat16)   # bits 49158 == -2.09375
//! >>> torch.manual_seed(2)
//! >>> out = torch.nn.functional.rrelu(x, lower=0.125, upper=1.0/3.0, training=True)
//! >>> float(out[0]), int(out[0].view(torch.uint16).item())
//! (-0.66015625, 48937)
//! ```
//! ferrotorch (bf16 multiply path) yields bits 48938 (-0.6640625): off by one ULP.
//!
//! Tracking: #1953 (blocker). Left un-`#[ignore]`d: bit-inexactness vs torch in
//! a supported dtype (bf16/f16, the Llama-3 precision path) is a release block.

use ferrotorch_core::grad_fns::activation::rrelu;
use ferrotorch_core::{Tensor, TensorStorage, manual_seed};
use std::sync::{Mutex, MutexGuard};

const LOWER: f64 = 0.125;
const UPPER: f64 = 1.0 / 3.0;

fn default_rng_test_lock() -> MutexGuard<'static, ()> {
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Divergence: `rrelu_train` multiplies in `scalar_t` (bf16) instead of
/// `opmath_t` (float) per `pytorch aten/src/ATen/native/Activation.cpp:598-599`.
/// Upstream (seed 2, x=-2.1 bf16) returns bf16 bits 48937 (-0.66015625);
/// ferrotorch returns bf16 bits 48938 (-0.6640625).
/// Tracking: #1953
#[test]
fn divergence_rrelu_train_bf16_opmath_multiply() {
    let _guard = default_rng_test_lock();
    // Single negative element -> exactly one f64 draw, deterministic under seed.
    let x_in = half::bf16::from_f32(-2.1f32);
    assert_eq!(
        x_in.to_bits(),
        49158,
        "input bf16 representation must match torch's bf16 cast of -2.1"
    );
    let x = Tensor::from_storage(TensorStorage::cpu(vec![x_in]), vec![1], false).unwrap();

    manual_seed(2);
    let y = rrelu(&x, LOWER, UPPER, true).expect("rrelu training bf16 forward");
    let out = y.data().expect("cpu bf16 output");

    // Bit-exact assertion against the live torch oracle quoted above.
    const TORCH_BITS: u16 = 48937; // -0.66015625, captured from torch 2.11.0+cu130
    assert_eq!(
        out[0].to_bits(),
        TORCH_BITS,
        "rrelu(training) bf16 negative branch must multiply in opmath float \
         (torch bits {TORCH_BITS} == -0.66015625); got bits {} ({})",
        out[0].to_bits(),
        out[0].to_f32()
    );
}
