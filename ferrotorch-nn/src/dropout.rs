//! Dropout regularization layers.
//!
//! [`Dropout`] randomly zeroes individual elements during training with
//! probability `p`, scaling surviving elements by `1/(1-p)` (inverted
//! dropout). [`Dropout1d`], [`Dropout2d`], and [`Dropout3d`] drop entire
//! channels instead of individual elements, for 3D, 4D, and 5D inputs
//! respectively. [`AlphaDropout`] preserves mean and variance for use
//! with SELU activations.
//!
//! All six CPU forward paths draw their keep-mask from the byte-exact
//! MT19937 `Generator` (`ferrotorch_core::rng`) with torch's exact
//! consumption — per element ([`Dropout`], [`AlphaDropout`]) or per `[N, C]`
//! channel ([`Dropout1d`]/[`Dropout2d`]/[`Dropout3d`],
//! [`FeatureAlphaDropout`]) in flat order, keep iff `next_uniform_f64() <
//! (1 - p)` — so `ferrotorch_core::manual_seed(s)` reproduces
//! `torch.manual_seed(s); F.dropout{,1d,2d,3d}` / `nn.AlphaDropout` /
//! `nn.FeatureAlphaDropout` byte-for-byte (#1634, #1635, #1636). The alpha
//! variants use torch's hardcoded `alpha = 1.7580993408473766` affine
//! (`aten/src/ATen/native/Dropout.cpp:76`).
//!
//! All modules are identity in eval mode and have zero learnable parameters.
//!
//! ## REQ status (per `.design/ferrotorch-nn/dropout.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: `pub struct Dropout<T: Float>` here with `p` / `training` fields + ctor rejecting `p` outside `[0,1)`; non-test consumer: `Dropout::<T>::new(0.5)?` invoked in `ferrotorch-vision/src/models/vgg.rs` (the VGG classifier head dropout). |
//! | REQ-2 | SHIPPED | impl: `<Dropout as Module>::forward` body with eval / `p==0` short-circuit + Bernoulli + scale here; non-test consumer: `Dropout::forward` is called on every forward pass through the VGG / Inception classifier (constructed in `vgg.rs` and `inception.rs`). |
//! | REQ-3 | SHIPPED | impl: `input.is_cuda() && backend = ferrotorch_core::gpu_dispatch::gpu_backend()` GPU branch inside `<Dropout as Module>::forward` here; non-test consumer: any vision model run on CUDA (e.g. VGG / Inception fine-tuning with parameters on GPU) triggers this on every forward step. |
//! | REQ-4 | SHIPPED | impl: `struct DropoutBackward<T>` + `GradFn` impl here; non-test consumer: every `loss.backward()` over a model containing `Dropout` traverses these nodes via the autograd engine. |
//! | REQ-5 | SHIPPED | impl: `pub struct Dropout2d<T: Float>` + `Module` impl here; per-channel keep-mask drawn from the byte-exact MT19937 `Generator` (`make_feature_noise(input).bernoulli_(1-p)`, `Dropout.cpp:73-74`, keep iff `u < 1-p`), reproducing `torch.manual_seed(s); F.dropout2d` byte-for-byte (#1635, pinned by `divergence_dropout_seed_extended_and_feature_1634.rs::dropout2d_seed42_per_channel_matches_torch` vs live torch 2.11); non-test consumer: `pub use dropout::Dropout2d` in `lib.rs` exposes for downstream vision / segmentation code. |
//! | REQ-6 | SHIPPED | impl: `pub struct Dropout1d<T: Float>` + `Module` impl here; per-channel MT19937 mask (#1635, pinned by `dropout1d_seed42_per_channel_matches_torch`); non-test consumer: `pub use dropout::Dropout1d` in `lib.rs`. |
//! | REQ-7 | SHIPPED | impl: `pub struct Dropout3d<T: Float>` + `Module` impl here; per-channel MT19937 mask (#1635, pinned by `dropout3d_seed42_per_channel_matches_torch`); non-test consumer: `pub use dropout::Dropout3d` in `lib.rs`. |
//! | REQ-8 | SHIPPED | impl: `struct Dropout2dBackward<T>` + `GradFn` impl here; non-test consumer: autograd engine traversal on any model using `Dropout2d` in training. |
//! | REQ-9 | SHIPPED | impl: `pub struct AlphaDropout<T: Float>` + torch's EXACT alpha affine inside `<AlphaDropout as Module>::forward` here — per-element MT19937 keep-mask (keep iff `u < 1-p`) + `alpha = 1.7580993408473766` (`ALPHA_DROPOUT_ALPHA`, torch's hardcoded literal at `Dropout.cpp:76`, NOT recomputed `SELU_LAMBDA*SELU_ALPHA`), `a = 1/sqrt((alpha^2*p+1)*(1-p))`, kept = `a*x+alpha*a*p`, dropped = `-alpha*a+alpha*a*p` (`Dropout.cpp:74-79`), reproducing `torch.manual_seed(s); nn.AlphaDropout(p)` byte-for-byte (#1636, pinned by `divergence_dropout_seed_extended_and_feature_1634.rs::alpha_dropout_seed42_matches_torch` vs live torch 2.11); non-test consumer: `pub use dropout::AlphaDropout` in `lib.rs`. |
//! | REQ-10 | SHIPPED | impl: `struct AlphaDropoutBackward<T>` + `GradFn` impl here; non-test consumer: autograd engine traversal on models using `AlphaDropout`. |
//! | REQ-11 | SHIPPED | impl: 5 `Module<T> for <DropoutKind><T>` impl blocks here, each returning `vec![]` for parameters; non-test consumer: `ferrotorch_optim::Optimizer` walks `Module::parameters_mut()` of containers; dropout returns an empty list (correct: dropout has no trainable parameters). |
//! | REQ-12 | SHIPPED | impl: `with_inplace` builder + `inplace` getter + `inplace` field on all six dropout structs, the autograd-safe `apply_inplace_dropout` helper (errors on grad-requiring leaf per torch `VariableTypeUtils.h:80-84`; out-of-place fallback on grad-requiring non-leaf — R-DEV-7, ferrotorch lacks torch's version counter `saved_variable.cpp:170-186`; raw `write_inplace`/`Tensor::update_data` only on the non-grad-tracked path), and the `if self.inplace { apply_inplace_dropout(input, &output_data)? }` branch in `<Dropout/Dropout1d/Dropout2d/Dropout3d as Module>::forward` here, mirroring `_VF.dropout_`/`_VF.feature_dropout_` at `torch/nn/functional.py:1449,1516,1579,1629` on the memory-opt path; `AlphaDropout`/`FeatureAlphaDropout` carry the field for ABI parity but match torch's module forward which never forwards `inplace` (`dropout.py:265-269,319-323`). Non-test production consumer: the `if self.inplace` branch is on the live forward path of `<Dropout as Module>::forward` here, exercised by `ferrotorch-nn/src/lora.rs` (LoRA input dropout), `ferrotorch-vision/src/models/vgg.rs` / `inception.rs` (classifier head), and `ferrotorch-graph/src/gcn.rs` (inter-layer dropout). Default `inplace=false` preserves existing behavior. Closes #1446, #1580, #1581. |
//! | REQ-13 | SHIPPED | impl: `pub struct FeatureAlphaDropout<T: Float>` + `FeatureAlphaDropoutBackward<T>` + `Module<T>` impl here — per-channel MT19937 keep-mask (`make_feature_noise` flat `[N,C]` Bernoulli, keep iff `u < 1-p`) broadcast over `[N, C, *]`, torch's EXACT alpha affine (`alpha = 1.7580993408473766`, kept = `a*x+alpha*a*p`, dropped = `-alpha*a+alpha*a*p`, `Dropout.cpp:73-79`), reproducing `torch.manual_seed(s); nn.FeatureAlphaDropout(p)` byte-for-byte (#1636, pinned by `divergence_dropout_seed_extended_and_feature_1634.rs::feature_alpha_dropout_seed42_matches_torch` vs live torch 2.11); closes #1448; non-test consumer: `pub use dropout::FeatureAlphaDropout` in `lib.rs` (re-export) exposes the layer to downstream self-normalising-network model code in `ferrotorch-vision` / `ferrotorch-llama`. |
//! | REQ-14 | NOT-STARTED | blocker #1441 (umbrella) — `Dropout2d` / `Dropout1d` / `Dropout3d` GPU forward absent (CUDA inputs return `NotImplementedOnCuda`). Parity-sweep runner arms also absent. |

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::gpu_dispatch::GpuRngState;
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

use crate::module::Module;
use crate::parameter::Parameter;

// ---------------------------------------------------------------------------
// Philox 4x32-10 for CPU-side mask regeneration
// ---------------------------------------------------------------------------
// We need the Philox algorithm on CPU to regenerate dropout masks during
// backward for GPU tensors (the forward mask was generated on GPU using
// the Philox state). This is a copy of the core algorithm from
// ferrotorch-gpu/src/rng.rs to avoid a dependency on the GPU crate.

#[allow(dead_code)]
const PHILOX_M0: u32 = 0xD2511F53;
#[allow(dead_code)]
const PHILOX_M1: u32 = 0xCD9E8D57;
#[allow(dead_code)]
const PHILOX_W0: u32 = 0x9E3779B9;
#[allow(dead_code)]
const PHILOX_W1: u32 = 0xBB67AE85;

#[allow(dead_code)]
#[inline]
fn philox_round(c0: u32, c1: u32, c2: u32, c3: u32, k0: u32, k1: u32) -> (u32, u32, u32, u32) {
    let prod0 = (PHILOX_M0 as u64) * (c0 as u64);
    let hi0 = (prod0 >> 32) as u32;
    let lo0 = prod0 as u32;

    let prod1 = (PHILOX_M1 as u64) * (c2 as u64);
    let hi1 = (prod1 >> 32) as u32;
    let lo1 = prod1 as u32;

    let new_c0 = hi1 ^ c1 ^ k0;
    let new_c1 = lo1;
    let new_c2 = hi0 ^ c3 ^ k1;
    let new_c3 = lo0;

    (new_c0, new_c1, new_c2, new_c3)
}

/// Philox 4x32-10: produces 4 uniform u32 values from (counter, key).
#[allow(dead_code)]
fn philox_4x32_10(counter: u64, key: u64) -> [u32; 4] {
    let mut c0 = counter as u32;
    let mut c1 = (counter >> 32) as u32;
    let mut c2 = 0u32;
    let mut c3 = 0u32;

    let mut k0 = key as u32;
    let mut k1 = (key >> 32) as u32;

    for _ in 0..9 {
        (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
        k0 = k0.wrapping_add(PHILOX_W0);
        k1 = k1.wrapping_add(PHILOX_W1);
    }
    // Round 10 (final, no key advance)
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);

    [c0, c1, c2, c3]
}

/// Generate a dropout mask using the Philox algorithm, matching the GPU kernel's
/// behavior. The mask uses `(counter ^ seed)` as a derived u32 seed and applies
/// the same xorshift-multiply hash that the GPU dropout kernel uses.
///
/// This ensures backward mask matches the forward mask generated on GPU.
fn philox_dropout_mask<T: Float>(
    numel: usize,
    threshold: u32,
    scale: T,
    rng_state: &GpuRngState,
) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let derived_seed = (rng_state.counter() ^ rng_state.seed()) as u32;

    (0..numel)
        .map(|i| {
            let mut r = (i as u32).wrapping_mul(2654435761) ^ derived_seed;
            r ^= r << 13;
            r ^= r >> 17;
            r ^= r << 5;
            if r < threshold { zero } else { scale }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// In-place storage write
// ---------------------------------------------------------------------------

/// Whether the in-place dropout policy actually mutated the input storage, or
/// suppressed the mutation for autograd safety. See [`apply_inplace_dropout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InplaceOutcome {
    /// The input storage was mutated in place (`inplace=true` honored).
    Mutated,
    /// The mutation was suppressed for autograd safety; the caller must build
    /// the output out-of-place from the freshly-allocated `output_data` buffer.
    FellBackToOutOfPlace,
}

/// Apply the in-place dropout policy, mutating `input`'s storage where it is
/// autograd-safe to do so and matching torch's observable error contract where
/// ferrotorch can.
///
/// # Autograd safety (R-DEV-7 deviation — documented)
///
/// torch enforces in-place autograd correctness with two mechanisms that
/// ferrotorch's autograd engine does NOT have:
///
/// 1. A **leaf in-place guard** — mutating a leaf that requires grad raises
///    `"a leaf Variable that requires grad is being used in an in-place
///    operation."` from `check_inplace`
///    (`torch/csrc/autograd/VariableTypeUtils.h:61-63,80-84`).
/// 2. A **version counter** — every saved tensor records the storage version it
///    was saved at; if an in-place op bumps that version before backward,
///    `SavedVariable::unpack` raises `"one of the variables needed for gradient
///    computation has been modified by an inplace operation"`
///    (`torch/csrc/autograd/saved_variable.cpp:170-186`).
///
/// ferrotorch has neither (no `version` field on `TensorInner`; `Tensor::clone`
/// shares the `Arc<TensorInner>` storage). Without a version counter it cannot
/// *detect* that another backward node saved the pre-mutation storage, so an
/// unconditional in-place write silently corrupts that branch's gradient
/// (#1580). To eliminate the corruption rather than risk it, this helper adopts
/// a conservative policy on the grad-tracked path:
///
/// * **Leaf requiring grad, grad enabled** → return an `Err` mirroring torch's
///   leaf-guard message. (Matches torch exactly; pins #1581.)
/// * **Non-leaf requiring grad, grad enabled** → do NOT mutate; signal
///   [`InplaceOutcome::FellBackToOutOfPlace`] so the caller builds a fresh
///   output. The result tensor is numerically identical and the gradient is
///   correct (no shared-storage corruption); this is *more permissive* than
///   torch's version-counter `RuntimeError` — ferrotorch cannot prove the
///   storage is unused by another backward without a version counter, so it
///   declines to mutate instead of erroring. (Eliminates #1580's corruption.)
/// * **Grad disabled, or input does not require grad** → mutate in place. This
///   is the real memory-optimization case; no autograd node observes the
///   storage, so it is graph-safe and matches torch's `_VF.dropout_`.
///
/// The deviation preserves torch's *observable result* (identical output,
/// correct gradient) while declining to replicate torch's runtime error on the
/// non-leaf path, because ferrotorch lacks the version-counter infrastructure
/// that error depends on.
fn apply_inplace_dropout<T: Float>(
    input: &Tensor<T>,
    new_data: &[T],
) -> FerrotorchResult<InplaceOutcome> {
    if is_grad_enabled() && input.requires_grad() {
        if input.is_leaf() {
            // Match torch's leaf in-place guard
            // (`torch/csrc/autograd/VariableTypeUtils.h:80-84`).
            return Err(FerrotorchError::InvalidArgument {
                message:
                    "a leaf Variable that requires grad is being used in an in-place operation."
                        .to_string(),
            });
        }
        // Non-leaf requiring grad: ferrotorch has no version counter to prove
        // the shared storage is unused by another saved-for-backward node, so
        // fall back to out-of-place rather than risk corrupting that branch's
        // gradient (#1580). The caller builds the output from `new_data`.
        return Ok(InplaceOutcome::FellBackToOutOfPlace);
    }

    // Grad disabled or input does not require grad: the genuine
    // memory-optimization case. No autograd node can observe the storage, so
    // the in-place write is graph-safe and matches torch's `_VF.dropout_`.
    write_inplace(input, new_data)?;
    Ok(InplaceOutcome::Mutated)
}

/// Write `new_data` over `input`'s storage in place, mirroring torch's
/// `_VF.dropout_` family (`torch/nn/functional.py:1449,1516,1579,1629`)
/// which mutate the input tensor's buffer rather than allocating a fresh
/// output.
///
/// This is the raw write; the autograd-safety policy that decides *whether* a
/// write is permitted lives in [`apply_inplace_dropout`]. Callers must route
/// through that helper and never call this directly on a grad-tracked path.
fn write_inplace<T: Float>(input: &Tensor<T>, new_data: &[T]) -> FerrotorchResult<()> {
    // SAFETY: `update_data` requires exclusive access to the input's storage
    // for the duration of the write. The dropout forward holds the only live
    // borrow of the input data (consumed into `new_data` by the caller before
    // this call). The autograd-safety policy in `apply_inplace_dropout`
    // guarantees this is only reached when grad is disabled or the input does
    // not require grad, so no backward node has saved (and could later read) a
    // version of this storage. `new_data.len() == input.numel()` is guaranteed
    // by the callers (the mask and input share numel). PyTorch performs this
    // exact mutation in `_VF.dropout_` (`torch/nn/functional.py:1449`).
    #[allow(
        clippy::undocumented_unsafe_blocks,
        reason = "SAFETY comment above documents the exclusive-access invariant; apply_inplace_dropout gates this to the non-grad-tracked path where no backward node observes the storage"
    )]
    unsafe {
        input.update_data(new_data)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// DropoutBackward
// ---------------------------------------------------------------------------

/// Backward node for elementwise dropout.
///
/// Reapplies the same binary mask scaled by `1/(1-p)` to the upstream
/// gradient, routing gradients only through surviving elements.
///
/// The mask is stored as a [`Tensor<T>`] on the same device as the
/// forward input so backward reduces to a single `mul` that stays
/// GPU-native when the input is on CUDA.
#[derive(Debug)]
struct DropoutBackward<T: Float> {
    input: Tensor<T>,
    /// Mask tensor with elements in `{0, 1/(1-p)}`. Lives on the same
    /// device as `input`, so `mul(grad_output, scaled_mask)` in the
    /// backward routes entirely through GPU ops when training on CUDA.
    scaled_mask: Tensor<T>,
}

impl<T: Float> GradFn<T> for DropoutBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let g = ferrotorch_core::grad_fns::arithmetic::mul(grad_output, &self.scaled_mask)?;
            Some(g)
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "DropoutBackward"
    }
}

// ---------------------------------------------------------------------------
// Dropout2dBackward
// ---------------------------------------------------------------------------

/// Backward node for channel-wise dropout.
///
/// Identical to [`DropoutBackward`] — the mask shape already encodes the
/// channel-level structure (all spatial positions in a dropped channel are 0).
#[derive(Debug)]
struct Dropout2dBackward<T: Float> {
    input: Tensor<T>,
    scaled_mask: Vec<T>,
}

impl<T: Float> GradFn<T> for Dropout2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "dropout2d backward",
            });
        }
        let da = if self.input.requires_grad() {
            let go_data = grad_output.data_vec()?;
            let grad_a: Vec<T> = go_data
                .iter()
                .zip(self.scaled_mask.iter())
                .map(|(&g, &m)| g * m)
                .collect();
            let g = Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.input.shape().to_vec(),
                false,
            )?;
            Some(g)
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "Dropout2dBackward"
    }
}

// ===========================================================================
// Dropout
// ===========================================================================

/// Randomly zeroes elements with probability `p` during training.
///
/// During training, each element is independently set to zero with probability
/// `p` and scaled by `1/(1-p)` so that the expected value is preserved
/// (inverted dropout).  During evaluation (`eval()` mode), the input is
/// returned unchanged.
///
/// # Panics
///
/// The constructor returns an error if `p` is outside `[0, 1)`.
#[derive(Debug)]
pub struct Dropout<T: Float> {
    p: f64,
    training: bool,
    /// When `true`, the forward mutates the input tensor's storage in place
    /// (mask + scale written back over the input) instead of allocating a
    /// fresh output buffer. Mirrors `_DropoutNd.inplace` at
    /// `torch/nn/modules/dropout.py:29` and the `inplace` branch of
    /// `F.dropout` at `torch/nn/functional.py:1448-1450`
    /// (`_VF.dropout_(input, p, training) if inplace`).
    inplace: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Dropout<T> {
    /// Create a new `Dropout` layer.
    ///
    /// `p` is the probability of an element being zeroed. Must be in `[0, 1)`.
    pub fn new(p: f64) -> FerrotorchResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("dropout probability must be in [0, 1), got {p}"),
            });
        }
        Ok(Self {
            p,
            training: true,
            inplace: false,
            _marker: std::marker::PhantomData,
        })
    }

    /// Set the `inplace` flag, mirroring `torch.nn.Dropout(p, inplace=...)`
    /// at `torch/nn/modules/dropout.py:22-29`. When `true`, training-mode
    /// forward mutates the input storage instead of allocating a new buffer.
    #[must_use]
    pub fn with_inplace(mut self, inplace: bool) -> Self {
        self.inplace = inplace;
        self
    }

    /// Returns the `inplace` flag.
    pub fn inplace(&self) -> bool {
        self.inplace
    }

    /// Override the dropout probability after construction. Same
    /// validation as [`Self::new`]: `p` must be in `[0, 1)`.
    ///
    /// Use case: MC-dropout-style inference where a model loaded with
    /// `p=0` (eval-time default) is temporarily reactivated with a
    /// non-zero rate to draw stochastic samples without rebuilding
    /// the module hierarchy.
    pub fn set_p(&mut self, p: f64) -> FerrotorchResult<()> {
        if !(0.0..1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("dropout probability must be in [0, 1), got {p}"),
            });
        }
        self.p = p;
        Ok(())
    }

    /// Read the current dropout probability.
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl<T: Float> Module<T> for Dropout<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Eval mode or p == 0: identity.
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let numel = input.numel();
        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        // GPU fast path: run dropout kernel entirely on device using the
        // Philox CBRNG. This integrates with the global GPU RNG state so
        // that gradient checkpointing can reproduce identical masks.
        if input.is_cuda() {
            if let Some(backend) = ferrotorch_core::gpu_dispatch::gpu_backend() {
                let threshold = (self.p * u32::MAX as f64) as u32;
                let scale_f32 = 1.0f32 / (1.0 - self.p as f32);

                let (handle, rng_state) =
                    backend.dropout_philox_f32(input.gpu_handle()?, threshold, scale_f32)?;

                // For backward, we need the mask. Regenerate it from the saved
                // Philox RNG state using the same deterministic hash that the
                // GPU kernel uses. This is reproducible across checkpoint
                // save/restore because the Philox state is deterministic.
                if is_grad_enabled() && input.requires_grad() {
                    let scaled_mask_vec = philox_dropout_mask(numel, threshold, scale, &rng_state);
                    // Upload the mask to the input's device so the
                    // backward `mul` runs on-device without a CPU
                    // round-trip.
                    let mask_cpu = Tensor::from_storage(
                        TensorStorage::cpu(scaled_mask_vec),
                        input.shape().to_vec(),
                        false,
                    )?;
                    let scaled_mask = mask_cpu.to(input.device())?;
                    return Tensor::from_operation(
                        TensorStorage::gpu(handle),
                        input.shape().to_vec(),
                        Arc::new(DropoutBackward {
                            input: input.clone(),
                            scaled_mask,
                        }),
                    );
                } else {
                    return Tensor::from_storage(
                        TensorStorage::gpu(handle),
                        input.shape().to_vec(),
                        false,
                    );
                }
            }
        }

        // CPU path — draw the keep-mask from the byte-exact MT19937
        // `Generator` (`ferrotorch_core::rng`) using torch's EXACT CPU dropout
        // consumption, so `ferrotorch_core::manual_seed(s); Dropout::forward`
        // reproduces `torch.manual_seed(s); F.dropout(...)` byte-for-byte
        // (#1634). torch draws the mask via `noise.bernoulli_(1 - p)`
        // (`aten/src/ATen/native/Dropout.cpp:74`); the scalar bernoulli kernel
        // (`aten/src/ATen/native/cpu/DistributionTemplates.h:388-399`)
        // evaluates per element in flat order
        // `transformation::bernoulli<double>(uniform_real<double>(gen->random64(), 0, 1), 1 - p)`
        // = `uniform64 < (1 - p)` (keep == 1)
        // (`DistributionsHelper.h:107-113,219-222`,
        // `TransformationHelper.h:84-89,171-173`).
        // `uniform_real<double>(random64(), 0, 1)` is exactly
        // `Generator::next_uniform_f64` (rng.rs REQ-5, byte-exact); survivors
        // are scaled by `1/(1-p)` (`Dropout.cpp:81` `noise.div_(1 - p)`).
        let keep_prob = 1.0 - self.p;
        let scaled_mask_vec: Vec<T> = ferrotorch_core::rng::with_thread_rng(|g| {
            (0..numel)
                .map(|_| {
                    if g.next_uniform_f64() < keep_prob {
                        scale
                    } else {
                        zero
                    }
                })
                .collect()
        });

        let input_data = input.data()?;
        let output_data: Vec<T> = input_data
            .iter()
            .zip(scaled_mask_vec.iter())
            .map(|(&x, &m)| x * m)
            .collect();

        // In-place branch, mirroring `_VF.dropout_(input, p, training)` at
        // `torch/nn/functional.py:1449`. `apply_inplace_dropout` applies the
        // autograd-safe policy: it errors on a grad-requiring leaf (matching
        // torch), falls back to out-of-place for a grad-requiring non-leaf
        // (ferrotorch lacks torch's version counter, so it declines to mutate
        // shared storage), and mutates in place only when no autograd node can
        // observe the storage. The out-of-place output below is always built
        // from `output_data`, so the fallback needs no special handling here.
        if self.inplace {
            apply_inplace_dropout(input, &output_data)?;
        }

        if is_grad_enabled() && input.requires_grad() {
            let scaled_mask = Tensor::from_storage(
                TensorStorage::cpu(scaled_mask_vec),
                input.shape().to_vec(),
                false,
            )?;
            Tensor::from_operation(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                Arc::new(DropoutBackward {
                    input: input.clone(),
                    scaled_mask,
                }),
            )
        } else {
            Tensor::from_storage(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                false,
            )
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Dropout2d
// ===========================================================================

/// Randomly zeroes entire channels with probability `p` during training.
///
/// Expects input of shape `[B, C, ...]` (at least 2 dimensions). During
/// training, each channel (the entire `[H, W, ...]` slice for a given `b, c`)
/// is independently set to zero with probability `p` and surviving channels
/// are scaled by `1/(1-p)`.  During evaluation the input is returned unchanged.
///
/// # Panics
///
/// The constructor returns an error if `p` is outside `[0, 1)`.
#[derive(Debug)]
pub struct Dropout2d<T: Float> {
    p: f64,
    training: bool,
    /// In-place flag, mirroring `_DropoutNd.inplace` at
    /// `torch/nn/modules/dropout.py:29` and the `inplace` branch of
    /// `F.dropout2d` at `torch/nn/functional.py:1578-1582`
    /// (`_VF.feature_dropout_(input, p, training) if inplace`).
    inplace: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Dropout2d<T> {
    /// Create a new `Dropout2d` layer.
    ///
    /// `p` is the probability of an entire channel being zeroed. Must be in `[0, 1)`.
    pub fn new(p: f64) -> FerrotorchResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("dropout2d probability must be in [0, 1), got {p}"),
            });
        }
        Ok(Self {
            p,
            training: true,
            inplace: false,
            _marker: std::marker::PhantomData,
        })
    }

    /// Set the `inplace` flag, mirroring `torch.nn.Dropout2d(p, inplace=...)`.
    /// When `true`, training-mode forward mutates the input storage.
    #[must_use]
    pub fn with_inplace(mut self, inplace: bool) -> Self {
        self.inplace = inplace;
        self
    }

    /// Returns the `inplace` flag.
    pub fn inplace(&self) -> bool {
        self.inplace
    }
}

impl<T: Float> Module<T> for Dropout2d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Eval mode or p == 0: identity.
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let shape = input.shape();
        if shape.len() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Dropout2d expects at least 2D input [B, C, ...], got shape {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        // Product of empty slice is 1, so no special case needed for 2-D inputs.
        let spatial: usize = shape[2..].iter().product();

        let numel = input.numel();
        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        // GPU tensors are not yet supported for Dropout2d — needs a fused
        // channel-broadcast dropout kernel.
        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "Dropout2d" });
        }

        // CPU path — draw the per-channel keep mask from the byte-exact
        // MT19937 `Generator` (`ferrotorch_core::rng`), matching torch's
        // `make_feature_noise(input).bernoulli_(1 - p)`
        // (`aten/src/ATen/native/Dropout.cpp:73-74`). torch reduces the input
        // to a `[N, C, 1, 1...]` noise tensor and draws ONE Bernoulli per
        // `[N, C]` entry in flat order, then broadcasts over the spatial dims
        // and scales survivors by `1/(1-p)` (`Dropout.cpp:81` `noise.div_(1-p)`).
        // The scalar bernoulli kernel keeps iff `next_uniform_f64() < (1 - p)`
        // (`DistributionTemplates.h` / `TransformationHelper.h:171-173`), so a
        // shared `ferrotorch_core::manual_seed(s)` reproduces
        // `torch.manual_seed(s); F.dropout2d(...)` byte-for-byte (#1635).
        let keep_prob = 1.0 - self.p;
        let channel_mask: Vec<bool> = ferrotorch_core::rng::with_thread_rng(|g| {
            (0..batch * channels)
                .map(|_| g.next_uniform_f64() < keep_prob)
                .collect()
        });

        // Expand channel mask to full element mask.
        let scaled_mask: Vec<T> = {
            let mut mask = Vec::with_capacity(numel);
            for &cm in &channel_mask {
                let val = if cm { scale } else { zero };
                for _ in 0..spatial {
                    mask.push(val);
                }
            }
            mask
        };

        let input_data = input.data_vec()?;
        let output_data: Vec<T> = input_data
            .iter()
            .zip(scaled_mask.iter())
            .map(|(&x, &m)| x * m)
            .collect();

        // In-place branch mirrors `_VF.feature_dropout_` at
        // `torch/nn/functional.py:1579`. Routed through the autograd-safe
        // policy (`apply_inplace_dropout`): errors on a grad-requiring leaf,
        // falls back to out-of-place on a grad-requiring non-leaf, mutates only
        // when no autograd node observes the storage.
        if self.inplace {
            apply_inplace_dropout(input, &output_data)?;
        }

        let result = if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                Arc::new(Dropout2dBackward {
                    input: input.clone(),
                    scaled_mask,
                }),
            )?
        } else {
            Tensor::from_storage(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                false,
            )?
        };
        Ok(result)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Dropout1d — CL-433
// ===========================================================================

/// Randomly zeroes entire 1D channels with probability `p` during training.
///
/// Expects input of shape `[B, C, L]` (3 dimensions). During training,
/// each channel (the entire length-`L` slice for a given `b, c`) is
/// independently set to zero with probability `p` and surviving channels
/// are scaled by `1/(1-p)`. During evaluation the input is returned unchanged.
///
/// This is the 1D analogue of [`Dropout2d`].
///
/// Matches `torch.nn.Dropout1d`.
#[derive(Debug)]
pub struct Dropout1d<T: Float> {
    p: f64,
    training: bool,
    /// In-place flag, mirroring `_DropoutNd.inplace` at
    /// `torch/nn/modules/dropout.py:29` and the `inplace` branch of
    /// `F.dropout1d` at `torch/nn/functional.py:1515-1519`
    /// (`_VF.feature_dropout_(input, p, training) if inplace`).
    inplace: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Dropout1d<T> {
    /// Create a new `Dropout1d` layer.
    ///
    /// `p` is the probability of an entire channel being zeroed. Must be in `[0, 1)`.
    pub fn new(p: f64) -> FerrotorchResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("dropout1d probability must be in [0, 1), got {p}"),
            });
        }
        Ok(Self {
            p,
            training: true,
            inplace: false,
            _marker: std::marker::PhantomData,
        })
    }

    /// Set the `inplace` flag, mirroring `torch.nn.Dropout1d(p, inplace=...)`.
    /// When `true`, training-mode forward mutates the input storage.
    #[must_use]
    pub fn with_inplace(mut self, inplace: bool) -> Self {
        self.inplace = inplace;
        self
    }

    /// Returns the `inplace` flag.
    pub fn inplace(&self) -> bool {
        self.inplace
    }
}

impl<T: Float> Module<T> for Dropout1d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let shape = input.shape();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Dropout1d expects 3D input [B, C, L], got shape {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        let length = shape[2];

        let numel = input.numel();
        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "Dropout1d" });
        }

        // Per-channel keep mask from the byte-exact MT19937 `Generator`,
        // matching torch's `make_feature_noise(input).bernoulli_(1 - p)`
        // (`aten/src/ATen/native/Dropout.cpp:73-74`): one Bernoulli draw per
        // `[N, C]` channel in flat order, keep iff `next_uniform_f64() < (1-p)`,
        // broadcast over the length-`L` dim, survivors scaled by `1/(1-p)`.
        // Reproducible under `ferrotorch_core::manual_seed` (#1635).
        let keep_prob = 1.0 - self.p;
        let channel_mask: Vec<bool> = ferrotorch_core::rng::with_thread_rng(|g| {
            (0..batch * channels)
                .map(|_| g.next_uniform_f64() < keep_prob)
                .collect()
        });

        let scaled_mask: Vec<T> = {
            let mut mask = Vec::with_capacity(numel);
            for &cm in &channel_mask {
                let val = if cm { scale } else { zero };
                for _ in 0..length {
                    mask.push(val);
                }
            }
            mask
        };

        let input_data = input.data_vec()?;
        let output_data: Vec<T> = input_data
            .iter()
            .zip(scaled_mask.iter())
            .map(|(&x, &m)| x * m)
            .collect();

        // In-place branch mirrors `_VF.feature_dropout_` at
        // `torch/nn/functional.py:1516`. Routed through the autograd-safe
        // policy (`apply_inplace_dropout`): errors on a grad-requiring leaf,
        // falls back to out-of-place on a grad-requiring non-leaf, mutates only
        // when no autograd node observes the storage.
        if self.inplace {
            apply_inplace_dropout(input, &output_data)?;
        }

        let result = if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                Arc::new(Dropout2dBackward {
                    input: input.clone(),
                    scaled_mask,
                }),
            )?
        } else {
            Tensor::from_storage(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                false,
            )?
        };
        Ok(result)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Dropout3d — CL-433
// ===========================================================================

/// Randomly zeroes entire 3D channels with probability `p` during training.
///
/// Expects input of shape `[B, C, D, H, W]` (5 dimensions). During training,
/// each channel (the entire `D * H * W` volume for a given `b, c`) is
/// independently set to zero with probability `p` and surviving channels
/// are scaled by `1/(1-p)`. During evaluation the input is returned unchanged.
///
/// Matches `torch.nn.Dropout3d`.
#[derive(Debug)]
pub struct Dropout3d<T: Float> {
    p: f64,
    training: bool,
    /// In-place flag, mirroring `_DropoutNd.inplace` at
    /// `torch/nn/modules/dropout.py:29` and the `inplace` branch of
    /// `F.dropout3d` at `torch/nn/functional.py:1628-1632`
    /// (`_VF.feature_dropout_(input, p, training) if inplace`).
    inplace: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Dropout3d<T> {
    /// Create a new `Dropout3d` layer.
    ///
    /// `p` is the probability of an entire channel being zeroed. Must be in `[0, 1)`.
    pub fn new(p: f64) -> FerrotorchResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("dropout3d probability must be in [0, 1), got {p}"),
            });
        }
        Ok(Self {
            p,
            training: true,
            inplace: false,
            _marker: std::marker::PhantomData,
        })
    }

    /// Set the `inplace` flag, mirroring `torch.nn.Dropout3d(p, inplace=...)`.
    /// When `true`, training-mode forward mutates the input storage.
    #[must_use]
    pub fn with_inplace(mut self, inplace: bool) -> Self {
        self.inplace = inplace;
        self
    }

    /// Returns the `inplace` flag.
    pub fn inplace(&self) -> bool {
        self.inplace
    }
}

impl<T: Float> Module<T> for Dropout3d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let shape = input.shape();
        if shape.len() != 5 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Dropout3d expects 5D input [B, C, D, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        let spatial: usize = shape[2..].iter().product();

        let numel = input.numel();
        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "Dropout3d" });
        }

        // Per-channel keep mask from the byte-exact MT19937 `Generator`,
        // matching torch's `make_feature_noise(input).bernoulli_(1 - p)`
        // (`aten/src/ATen/native/Dropout.cpp:73-74`): one Bernoulli draw per
        // `[N, C]` channel in flat order, keep iff `next_uniform_f64() < (1-p)`,
        // broadcast over the `D*H*W` volume, survivors scaled by `1/(1-p)`.
        // Reproducible under `ferrotorch_core::manual_seed` (#1635).
        let keep_prob = 1.0 - self.p;
        let channel_mask: Vec<bool> = ferrotorch_core::rng::with_thread_rng(|g| {
            (0..batch * channels)
                .map(|_| g.next_uniform_f64() < keep_prob)
                .collect()
        });

        let scaled_mask: Vec<T> = {
            let mut mask = Vec::with_capacity(numel);
            for &cm in &channel_mask {
                let val = if cm { scale } else { zero };
                for _ in 0..spatial {
                    mask.push(val);
                }
            }
            mask
        };

        let input_data = input.data_vec()?;
        let output_data: Vec<T> = input_data
            .iter()
            .zip(scaled_mask.iter())
            .map(|(&x, &m)| x * m)
            .collect();

        // In-place branch mirrors `_VF.feature_dropout_` at
        // `torch/nn/functional.py:1629`. Routed through the autograd-safe
        // policy (`apply_inplace_dropout`): errors on a grad-requiring leaf,
        // falls back to out-of-place on a grad-requiring non-leaf, mutates only
        // when no autograd node observes the storage.
        if self.inplace {
            apply_inplace_dropout(input, &output_data)?;
        }

        let result = if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                Arc::new(Dropout2dBackward {
                    input: input.clone(),
                    scaled_mask,
                }),
            )?
        } else {
            Tensor::from_storage(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                false,
            )?
        };
        Ok(result)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// AlphaDropout — CL-433
// ===========================================================================

/// Alpha Dropout for use with SELU activations.
///
/// Unlike standard dropout, `AlphaDropout` preserves the self-normalizing
/// property of SELU by maintaining the mean and variance of the input.
/// Dropped elements are set to the SELU saturation value rather than zero,
/// and the output is affinely transformed to restore the original mean and
/// variance.
///
/// During training, mirroring `aten/src/ATen/native/Dropout.cpp:74-79`:
/// 1. A per-element Bernoulli keep-mask is drawn at probability `1 - p` from
///    the byte-exact MT19937 `Generator` (keep iff `next_uniform_f64() < 1-p`).
/// 2. With `alpha = 1.7580993408473766` and
///    `a = 1/sqrt((alpha^2 * p + 1) * (1 - p))`:
///    - kept elements map to `a*x + alpha*a*p`,
///    - dropped elements map to the constant `-alpha*a + alpha*a*p`.
///
/// During evaluation, the input is returned unchanged.
///
/// Matches `torch.nn.AlphaDropout`. Reproducible under
/// `ferrotorch_core::manual_seed` (#1636).
#[derive(Debug)]
pub struct AlphaDropout<T: Float> {
    p: f64,
    training: bool,
    /// In-place flag, carried for API parity with `_DropoutNd.inplace`
    /// (`torch/nn/modules/dropout.py:29`).
    ///
    /// NOTE — faithful upstream behaviour: `AlphaDropout.forward` at
    /// `torch/nn/modules/dropout.py:265-269` calls
    /// `F.alpha_dropout(input, self.p, self.training)` and does **not** pass
    /// `self.inplace`, so torch's `nn.AlphaDropout(p, inplace=True)` does NOT
    /// mutate in place at the module level — the `inplace` field exists on the
    /// struct (inherited from `_DropoutNd.__init__`) but the module forward
    /// drops it. We mirror that exactly: the field is stored for ABI parity,
    /// but [`AlphaDropout::forward`] never mutates the input. (The functional
    /// `F.alpha_dropout` does accept `inplace`, but the module never forwards
    /// it.)
    inplace: bool,
    _marker: std::marker::PhantomData<T>,
}

/// The alpha-dropout affine constant torch hardcodes at
/// `aten/src/ATen/native/Dropout.cpp:76`
/// (`constexpr double alpha = 1.7580993408473766;`). This is the SELU-derived
/// `lambda * alpha` magnitude, but used VERBATIM as torch's literal — NOT
/// recomputed as `SELU_LAMBDA * SELU_ALPHA`, which differs in the last ULPs and
/// would shift the affine away from torch byte-for-byte (#1636).
const ALPHA_DROPOUT_ALPHA: f64 = 1.7580993408473766;

impl<T: Float> AlphaDropout<T> {
    /// Create a new `AlphaDropout` layer.
    ///
    /// `p` is the probability of an element being dropped. Must be in `[0, 1)`.
    pub fn new(p: f64) -> FerrotorchResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("alpha_dropout probability must be in [0, 1), got {p}"),
            });
        }
        Ok(Self {
            p,
            training: true,
            inplace: false,
            _marker: std::marker::PhantomData,
        })
    }

    /// Set the `inplace` flag for API parity with
    /// `torch.nn.AlphaDropout(p, inplace=...)`.
    ///
    /// Like upstream, the module `forward` does NOT mutate in place even when
    /// this is `true` — `torch.nn.AlphaDropout.forward` never forwards
    /// `self.inplace` to `F.alpha_dropout` (`dropout.py:265-269`). The flag is
    /// retained so the constructor surface matches torch field-for-field.
    #[must_use]
    pub fn with_inplace(mut self, inplace: bool) -> Self {
        self.inplace = inplace;
        self
    }

    /// Returns the `inplace` flag.
    pub fn inplace(&self) -> bool {
        self.inplace
    }
}

/// Backward node for AlphaDropout.
///
/// The affine correction factor `a` is baked into the scaled_mask:
/// surviving elements get `a`, dropped elements get `0`.
/// Gradient routing: grad_input = grad_output * scaled_mask.
#[derive(Debug)]
struct AlphaDropoutBackward<T: Float> {
    input: Tensor<T>,
    /// Mask with `a` for kept elements and `0` for dropped elements.
    grad_mask: Vec<T>,
}

impl<T: Float> GradFn<T> for AlphaDropoutBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "AlphaDropout backward",
            });
        }
        let da = if self.input.requires_grad() {
            let go_data = grad_output.data_vec()?;
            let grad_a: Vec<T> = go_data
                .iter()
                .zip(self.grad_mask.iter())
                .map(|(&g, &m)| g * m)
                .collect();
            let g = Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.input.shape().to_vec(),
                false,
            )?;
            Some(g)
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AlphaDropoutBackward"
    }
}

impl<T: Float> Module<T> for AlphaDropout<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "AlphaDropout" });
        }

        let numel = input.numel();
        let p = self.p;

        // torch's EXACT alpha affine, `aten/src/ATen/native/Dropout.cpp:74-79`:
        //   noise.bernoulli_(1 - p)                 // 1.0 kept, 0.0 dropped
        //   constexpr double alpha = 1.7580993408473766;
        //   double a = 1. / sqrt((alpha*alpha*p + 1) * (1 - p));
        //   b = noise.add(-1).mul_(alpha*a).add_(alpha*a*p);
        //   noise.mul_(a);                          // a kept, 0 dropped
        //   out = input * noise + b
        // Folding the per-element `b = (noise-1)*alpha*a + alpha*a*p`:
        //   kept  (noise=1): out = a*x + alpha*a*p
        //   dropped(noise=0): out = -alpha*a + alpha*a*p   (constant in x)
        // We use torch's hardcoded `alpha` constant verbatim — NOT a recomputed
        // `-SELU_LAMBDA*SELU_ALPHA` (= -1.7580993..., same magnitude but the
        // recomputed value diverges in the last ULPs and changes the affine).
        let alpha = ALPHA_DROPOUT_ALPHA;
        let a_f64 = 1.0 / ((alpha * alpha * p + 1.0) * (1.0 - p)).sqrt();
        let dropped_f64 = -alpha * a_f64 + alpha * a_f64 * p;
        let kept_b_f64 = alpha * a_f64 * p;

        let a = T::from(a_f64).unwrap();
        let kept_b = T::from(kept_b_f64).unwrap();
        let dropped_v = T::from(dropped_f64).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        // Per-element keep mask from the byte-exact MT19937 `Generator`,
        // matching `at::empty_like(input).bernoulli_(1 - p)` (alpha_dropout is
        // element-wise, NOT feature noise; `Dropout.cpp:73`). Keep iff
        // `next_uniform_f64() < (1 - p)`; reproducible under
        // `ferrotorch_core::manual_seed` (#1636).
        let keep_prob = 1.0 - p;
        let keep: Vec<bool> = ferrotorch_core::rng::with_thread_rng(|g| {
            (0..numel)
                .map(|_| g.next_uniform_f64() < keep_prob)
                .collect()
        });

        let input_data = input.data()?;
        let mut output_data = Vec::with_capacity(numel);
        let mut grad_mask = Vec::with_capacity(numel);

        for (i, &x) in input_data.iter().enumerate() {
            if keep[i] {
                // Kept element: a * x + alpha*a*p
                output_data.push(a * x + kept_b);
                grad_mask.push(a);
            } else {
                // Dropped element: -alpha*a + alpha*a*p (independent of x).
                output_data.push(dropped_v);
                grad_mask.push(zero);
            }
        }

        if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                Arc::new(AlphaDropoutBackward {
                    input: input.clone(),
                    grad_mask,
                }),
            )
        } else {
            Tensor::from_storage(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                false,
            )
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// FeatureAlphaDropout — closes #1448
// ===========================================================================

/// Randomly masks entire feature-channels with the SELU saturation value
/// during training, mirroring `torch.nn.FeatureAlphaDropout`
/// (`torch/nn/modules/dropout.py:233-281`).
///
/// Unlike [`AlphaDropout`], which drops individual elements, this layer
/// drops every spatial position within a `(b, c)` feature-channel as a unit
/// — the dropout decision is sampled once per channel and broadcast over
/// the trailing spatial dims. Used in self-normalising convolutional
/// networks where per-feature decorrelation must be preserved while
/// maintaining mean/variance.
///
/// During training, mirroring `aten/src/ATen/native/Dropout.cpp:73-79`
/// (`_dropout_impl<feature=true, alpha=true>`):
/// 1. A per-channel Bernoulli keep-mask is drawn over the reduced
///    `[N, C, 1, 1...]` noise tensor at probability `1 - p` from the
///    byte-exact MT19937 `Generator` (keep iff `next_uniform_f64() < 1-p`),
///    in flat `[N, C]` order, then broadcast over the spatial volume.
/// 2. With `alpha = 1.7580993408473766` and
///    `a = 1/sqrt((alpha^2 * p + 1) * (1 - p))`, kept channels map to
///    `a*x + alpha*a*p` and dropped channels to `-alpha*a + alpha*a*p`.
///
/// During evaluation, the input is returned unchanged.
///
/// Expects input of shape `[N, C, *]` (at least 2-D). Reproducible under
/// `ferrotorch_core::manual_seed` (#1636).
#[derive(Debug)]
pub struct FeatureAlphaDropout<T: Float> {
    p: f64,
    training: bool,
    /// In-place flag, carried for API parity with `_DropoutNd.inplace`
    /// (`torch/nn/modules/dropout.py:29`).
    ///
    /// NOTE — faithful upstream behaviour: `FeatureAlphaDropout.forward` at
    /// `torch/nn/modules/dropout.py:319-323` calls
    /// `F.feature_alpha_dropout(input, self.p, self.training)` and does **not**
    /// pass `self.inplace`, so torch's `nn.FeatureAlphaDropout(p,
    /// inplace=True)` does NOT mutate in place at the module level. We mirror
    /// that exactly: the field is stored for ABI parity, but
    /// [`FeatureAlphaDropout::forward`] never mutates the input.
    inplace: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> FeatureAlphaDropout<T> {
    /// Create a new `FeatureAlphaDropout` layer.
    ///
    /// `p` is the probability of an entire feature-channel being dropped.
    /// Must be in `[0, 1)`.
    pub fn new(p: f64) -> FerrotorchResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("feature_alpha_dropout probability must be in [0, 1), got {p}"),
            });
        }
        Ok(Self {
            p,
            training: true,
            inplace: false,
            _marker: std::marker::PhantomData,
        })
    }

    /// Set the `inplace` flag for API parity with
    /// `torch.nn.FeatureAlphaDropout(p, inplace=...)`.
    ///
    /// Like upstream, the module `forward` does NOT mutate in place even when
    /// this is `true` — `torch.nn.FeatureAlphaDropout.forward` never forwards
    /// `self.inplace` to `F.feature_alpha_dropout` (`dropout.py:319-323`).
    #[must_use]
    pub fn with_inplace(mut self, inplace: bool) -> Self {
        self.inplace = inplace;
        self
    }

    /// Returns the `inplace` flag.
    pub fn inplace(&self) -> bool {
        self.inplace
    }
}

/// Backward node for `FeatureAlphaDropout`.
///
/// The affine factor `a` is baked into the broadcast mask: kept channels
/// receive `a`, dropped channels receive `0`. Gradient routes as
/// `grad_input = grad_output * grad_mask`.
#[derive(Debug)]
struct FeatureAlphaDropoutBackward<T: Float> {
    input: Tensor<T>,
    /// Full-shape mask with `a` for kept channels, `0` for dropped.
    grad_mask: Vec<T>,
}

impl<T: Float> GradFn<T> for FeatureAlphaDropoutBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "FeatureAlphaDropout backward",
            });
        }
        let da = if self.input.requires_grad() {
            let go_data = grad_output.data_vec()?;
            let grad_a: Vec<T> = go_data
                .iter()
                .zip(self.grad_mask.iter())
                .map(|(&g, &m)| g * m)
                .collect();
            let g = Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.input.shape().to_vec(),
                false,
            )?;
            Some(g)
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "FeatureAlphaDropoutBackward"
    }
}

impl<T: Float> Module<T> for FeatureAlphaDropout<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let shape = input.shape();
        if shape.len() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "FeatureAlphaDropout expects at least 2D input [N, C, ...], got shape {:?}",
                    shape
                ),
            });
        }

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "FeatureAlphaDropout",
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        // Spatial dims (D, H, W, ...). For a 2-D `[N, C]` input the product
        // of the empty suffix is 1, matching torch's broadcast behaviour.
        let spatial: usize = shape[2..].iter().product();

        let numel = input.numel();
        let p = self.p;

        // torch's EXACT feature-alpha affine: `feature_alpha_dropout` calls
        // `_dropout_impl<feature=true, alpha=true>`, so the noise is a
        // PER-CHANNEL `make_feature_noise` tensor (`Dropout.cpp:73`) drawn with
        // `bernoulli_(1 - p)`, then the alpha affine
        // (`Dropout.cpp:76-79`): `alpha = 1.7580993408473766`,
        // `a = 1/sqrt((alpha^2*p + 1)*(1-p))`,
        // kept (noise=1) -> `a*x + alpha*a*p`,
        // dropped (noise=0) -> `-alpha*a + alpha*a*p` (constant in x).
        let alpha = ALPHA_DROPOUT_ALPHA;
        let a_f64 = 1.0 / ((alpha * alpha * p + 1.0) * (1.0 - p)).sqrt();
        let dropped_f64 = -alpha * a_f64 + alpha * a_f64 * p;
        let kept_b_f64 = alpha * a_f64 * p;

        let a = T::from(a_f64).unwrap();
        let kept_b = T::from(kept_b_f64).unwrap();
        let dropped_v = T::from(dropped_f64).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        // Per-channel keep mask: one Bernoulli draw per `[N, C]` entry in flat
        // order from the byte-exact MT19937 `Generator`, keep iff
        // `next_uniform_f64() < (1 - p)`, broadcast over the trailing spatial
        // volume. Reproducible under `ferrotorch_core::manual_seed` (#1636).
        let keep_prob = 1.0 - p;
        let keep_channel: Vec<bool> = ferrotorch_core::rng::with_thread_rng(|g| {
            (0..batch * channels)
                .map(|_| g.next_uniform_f64() < keep_prob)
                .collect()
        });

        let input_data = input.data_vec()?;
        let mut output_data = Vec::with_capacity(numel);
        let mut grad_mask = Vec::with_capacity(numel);

        // For each channel: emit `spatial` masked elements at once.
        for bc in 0..batch * channels {
            let keep = keep_channel[bc];
            let base = bc * spatial;
            for s in 0..spatial {
                let x = input_data[base + s];
                if keep {
                    output_data.push(a * x + kept_b);
                    grad_mask.push(a);
                } else {
                    output_data.push(dropped_v);
                    grad_mask.push(zero);
                }
            }
        }

        if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                Arc::new(FeatureAlphaDropoutBackward {
                    input: input.clone(),
                    grad_mask,
                }),
            )
        } else {
            Tensor::from_storage(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                false,
            )
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a leaf tensor with given data and shape.
    fn leaf_tensor(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // Dropout
    // -----------------------------------------------------------------------

    #[test]
    fn test_dropout_rate_approximately_correct() {
        let d = Dropout::<f32>::new(0.5).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[100_000]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        // Count zeros — should be roughly 50%.
        let zeros = data.iter().filter(|&&x| x == 0.0).count();
        let rate = zeros as f64 / data.len() as f64;
        assert!(
            (rate - 0.5).abs() < 0.05,
            "dropout rate = {rate}, expected ~0.5"
        );

        // Surviving elements should be scaled by 1/(1-0.5) = 2.0.
        let non_zero: Vec<f32> = data.iter().copied().filter(|&x| x != 0.0).collect();
        assert!(!non_zero.is_empty());
        for &v in &non_zero {
            assert!(
                (v - 2.0).abs() < 1e-6,
                "surviving element = {v}, expected 2.0"
            );
        }
    }

    #[test]
    fn test_dropout_eval_is_identity() {
        let mut d = Dropout::<f32>::new(0.5).unwrap();
        d.eval();
        assert!(!d.is_training());

        let input = ferrotorch_core::ones::<f32>(&[100]).unwrap();
        let output = d.forward(&input).unwrap();

        // In eval mode the output should be the exact same Arc (identity).
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_dropout_zero_prob_is_identity() {
        let d = Dropout::<f32>::new(0.0).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[100]).unwrap();
        let output = d.forward(&input).unwrap();
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_dropout_invalid_p() {
        assert!(Dropout::<f32>::new(1.0).is_err());
        assert!(Dropout::<f32>::new(-0.1).is_err());
        assert!(Dropout::<f32>::new(1.5).is_err());
    }

    #[test]
    fn test_dropout_backward_routes_through_surviving() {
        let d = Dropout::<f32>::new(0.5).unwrap();
        let input = leaf_tensor(&[1.0; 1000], &[1000], true);
        let output = d.forward(&input).unwrap();

        // To backward we need a scalar loss. Sum the output manually.
        let out_data = output.data().unwrap().to_vec();
        let total: f32 = out_data.iter().sum();

        // Build a SumBackward so we can call backward.
        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let t = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(t)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward {
                input: output.clone(),
            }),
        )
        .unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();

        // Every gradient element should be either 0 (dropped) or 1/(1-p) = 2.0 (survived).
        for &g in grad_data {
            assert!(
                g == 0.0 || (g - 2.0).abs() < 1e-6,
                "gradient element = {g}, expected 0.0 or 2.0"
            );
        }

        // The dropout mask for forward and backward should match: output zero
        // iff gradient zero.
        let out_data = output.data().unwrap();
        for (i, (&o, &g)) in out_data.iter().zip(grad_data.iter()).enumerate() {
            assert_eq!(
                o == 0.0,
                g == 0.0,
                "mismatch at index {i}: output={o}, grad={g}"
            );
        }
    }

    #[test]
    fn test_dropout_no_parameters() {
        let d = Dropout::<f32>::new(0.3).unwrap();
        assert!(d.parameters().is_empty());
        assert!(d.named_parameters().is_empty());
    }

    #[test]
    fn test_dropout_train_eval_toggle() {
        let mut d = Dropout::<f32>::new(0.5).unwrap();
        assert!(d.is_training());
        d.eval();
        assert!(!d.is_training());
        d.train();
        assert!(d.is_training());
    }

    #[test]
    fn test_dropout_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Dropout<f32>>();
        assert_send_sync::<Dropout<f64>>();
    }

    // -----------------------------------------------------------------------
    // Dropout2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_dropout2d_drops_whole_channels() {
        let d = Dropout2d::<f32>::new(0.5).unwrap();
        // Shape: [2, 10, 4, 4] — 2 batches, 10 channels, 4x4 spatial.
        let input = ferrotorch_core::ones::<f32>(&[2, 10, 4, 4]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        let spatial = 4 * 4;
        // Check that each channel is either entirely zero or entirely scaled.
        for b in 0..2 {
            for c in 0..10 {
                let start = (b * 10 + c) * spatial;
                let end = start + spatial;
                let channel = &data[start..end];

                let first = channel[0];
                assert!(
                    channel.iter().all(|&x| (x - first).abs() < 1e-6),
                    "channel (b={b}, c={c}) is not uniform: first={first}, channel={channel:?}"
                );
                // Value should be 0 or 1/(1-0.5) = 2.0.
                assert!(
                    first == 0.0 || (first - 2.0).abs() < 1e-6,
                    "channel value = {first}, expected 0.0 or 2.0"
                );
            }
        }
    }

    #[test]
    fn test_dropout2d_rate_approximately_correct() {
        let d = Dropout2d::<f32>::new(0.5).unwrap();
        // Many channels to get a good statistical sample.
        let input = ferrotorch_core::ones::<f32>(&[1, 1000, 2, 2]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        let spatial = 2 * 2;
        let mut dropped = 0;
        for c in 0..1000 {
            let start = c * spatial;
            if data[start] == 0.0 {
                dropped += 1;
            }
        }
        let rate = dropped as f64 / 1000.0;
        assert!(
            (rate - 0.5).abs() < 0.05,
            "dropout2d rate = {rate}, expected ~0.5"
        );
    }

    #[test]
    fn test_dropout2d_eval_is_identity() {
        let mut d = Dropout2d::<f32>::new(0.5).unwrap();
        d.eval();
        let input = ferrotorch_core::ones::<f32>(&[2, 3, 4, 4]).unwrap();
        let output = d.forward(&input).unwrap();
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_dropout2d_invalid_p() {
        assert!(Dropout2d::<f32>::new(1.0).is_err());
        assert!(Dropout2d::<f32>::new(-0.1).is_err());
    }

    #[test]
    fn test_dropout2d_requires_2d_input() {
        let d = Dropout2d::<f32>::new(0.3).unwrap();
        let input_1d = ferrotorch_core::ones::<f32>(&[10]).unwrap();
        assert!(d.forward(&input_1d).is_err());
    }

    #[test]
    fn test_dropout2d_backward_routes_through_surviving_channels() {
        let d = Dropout2d::<f32>::new(0.5).unwrap();
        // [1, 20, 3, 3]
        let input = leaf_tensor(&[1.0; 20 * 3 * 3], &[1, 20, 3, 3], true);
        let output = d.forward(&input).unwrap();

        let out_data = output.data().unwrap().to_vec();
        let total: f32 = out_data.iter().sum();

        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let t = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(t)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward {
                input: output.clone(),
            }),
        )
        .unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();
        let out_data = output.data().unwrap();

        // Gradient mask must match output mask.
        for (i, (&o, &g)) in out_data.iter().zip(grad_data.iter()).enumerate() {
            assert_eq!(
                o == 0.0,
                g == 0.0,
                "mismatch at index {i}: output={o}, grad={g}"
            );
        }

        // Gradients should be channel-uniform.
        let spatial = 3 * 3;
        for c in 0..20 {
            let start = c * spatial;
            let end = start + spatial;
            let channel_grad = &grad_data[start..end];
            let first = channel_grad[0];
            assert!(
                channel_grad.iter().all(|&g| (g - first).abs() < 1e-6),
                "gradient channel {c} is not uniform"
            );
        }
    }

    #[test]
    fn test_dropout2d_no_parameters() {
        let d = Dropout2d::<f32>::new(0.3).unwrap();
        assert!(d.parameters().is_empty());
        assert!(d.named_parameters().is_empty());
    }

    #[test]
    fn test_dropout2d_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Dropout2d<f32>>();
        assert_send_sync::<Dropout2d<f64>>();
    }

    // -----------------------------------------------------------------------
    // Dropout1d — CL-433
    // -----------------------------------------------------------------------

    #[test]
    fn test_dropout1d_drops_whole_channels() {
        let d = Dropout1d::<f32>::new(0.5).unwrap();
        // Shape: [2, 10, 8] — 2 batches, 10 channels, length 8.
        let input = ferrotorch_core::ones::<f32>(&[2, 10, 8]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        let length = 8;
        for b in 0..2 {
            for c in 0..10 {
                let start = (b * 10 + c) * length;
                let end = start + length;
                let channel = &data[start..end];

                let first = channel[0];
                assert!(
                    channel.iter().all(|&x| (x - first).abs() < 1e-6),
                    "channel (b={b}, c={c}) is not uniform"
                );
                assert!(
                    first == 0.0 || (first - 2.0).abs() < 1e-6,
                    "channel value = {first}, expected 0.0 or 2.0"
                );
            }
        }
    }

    #[test]
    fn test_dropout1d_rate_approximately_correct() {
        let d = Dropout1d::<f32>::new(0.5).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[1, 1000, 4]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        let length = 4;
        let mut dropped = 0;
        for c in 0..1000 {
            if data[c * length] == 0.0 {
                dropped += 1;
            }
        }
        let rate = dropped as f64 / 1000.0;
        assert!(
            (rate - 0.5).abs() < 0.05,
            "dropout1d rate = {rate}, expected ~0.5"
        );
    }

    #[test]
    fn test_dropout1d_eval_is_identity() {
        let mut d = Dropout1d::<f32>::new(0.5).unwrap();
        d.eval();
        let input = ferrotorch_core::ones::<f32>(&[2, 3, 8]).unwrap();
        let output = d.forward(&input).unwrap();
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_dropout1d_invalid_p() {
        assert!(Dropout1d::<f32>::new(1.0).is_err());
        assert!(Dropout1d::<f32>::new(-0.1).is_err());
    }

    #[test]
    fn test_dropout1d_requires_3d_input() {
        let d = Dropout1d::<f32>::new(0.3).unwrap();
        let input_2d = ferrotorch_core::ones::<f32>(&[10, 5]).unwrap();
        assert!(d.forward(&input_2d).is_err());
    }

    #[test]
    fn test_dropout1d_no_parameters() {
        let d = Dropout1d::<f32>::new(0.3).unwrap();
        assert!(d.parameters().is_empty());
    }

    #[test]
    fn test_dropout1d_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Dropout1d<f32>>();
        assert_send_sync::<Dropout1d<f64>>();
    }

    // -----------------------------------------------------------------------
    // Dropout3d — CL-433
    // -----------------------------------------------------------------------

    #[test]
    fn test_dropout3d_drops_whole_channels() {
        let d = Dropout3d::<f32>::new(0.5).unwrap();
        // Shape: [2, 10, 2, 2, 2] — 2 batches, 10 channels, 2x2x2 spatial.
        let input = ferrotorch_core::ones::<f32>(&[2, 10, 2, 2, 2]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        let spatial = 2 * 2 * 2;
        for b in 0..2 {
            for c in 0..10 {
                let start = (b * 10 + c) * spatial;
                let end = start + spatial;
                let channel = &data[start..end];

                let first = channel[0];
                assert!(
                    channel.iter().all(|&x| (x - first).abs() < 1e-6),
                    "channel (b={b}, c={c}) is not uniform"
                );
                assert!(
                    first == 0.0 || (first - 2.0).abs() < 1e-6,
                    "channel value = {first}, expected 0.0 or 2.0"
                );
            }
        }
    }

    #[test]
    fn test_dropout3d_rate_approximately_correct() {
        let d = Dropout3d::<f32>::new(0.5).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[1, 1000, 2, 2, 2]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        let spatial = 2 * 2 * 2;
        let mut dropped = 0;
        for c in 0..1000 {
            if data[c * spatial] == 0.0 {
                dropped += 1;
            }
        }
        let rate = dropped as f64 / 1000.0;
        assert!(
            (rate - 0.5).abs() < 0.05,
            "dropout3d rate = {rate}, expected ~0.5"
        );
    }

    #[test]
    fn test_dropout3d_eval_is_identity() {
        let mut d = Dropout3d::<f32>::new(0.5).unwrap();
        d.eval();
        let input = ferrotorch_core::ones::<f32>(&[2, 3, 2, 2, 2]).unwrap();
        let output = d.forward(&input).unwrap();
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_dropout3d_invalid_p() {
        assert!(Dropout3d::<f32>::new(1.0).is_err());
        assert!(Dropout3d::<f32>::new(-0.1).is_err());
    }

    #[test]
    fn test_dropout3d_requires_5d_input() {
        let d = Dropout3d::<f32>::new(0.3).unwrap();
        let input_4d = ferrotorch_core::ones::<f32>(&[2, 3, 4, 4]).unwrap();
        assert!(d.forward(&input_4d).is_err());
    }

    #[test]
    fn test_dropout3d_no_parameters() {
        let d = Dropout3d::<f32>::new(0.3).unwrap();
        assert!(d.parameters().is_empty());
    }

    #[test]
    fn test_dropout3d_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Dropout3d<f32>>();
        assert_send_sync::<Dropout3d<f64>>();
    }

    // -----------------------------------------------------------------------
    // AlphaDropout — CL-433
    // -----------------------------------------------------------------------

    #[test]
    fn test_alpha_dropout_preserves_mean_approx() {
        // With large sample, mean should be approximately preserved.
        let d = AlphaDropout::<f64>::new(0.5).unwrap();
        // Generate input with known mean.
        let n = 100_000;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) - 0.5).collect();
        let input_mean: f64 = data.iter().sum::<f64>() / n as f64;

        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![1, n], false).unwrap();
        let output = d.forward(&input).unwrap();
        let out_data = output.data().unwrap();
        let out_mean: f64 = out_data.iter().sum::<f64>() / n as f64;

        // Mean should be roughly preserved (within statistical tolerance).
        assert!(
            (out_mean - input_mean).abs() < 0.05,
            "AlphaDropout mean = {out_mean}, input mean = {input_mean}"
        );
    }

    #[test]
    fn test_alpha_dropout_eval_is_identity() {
        let mut d = AlphaDropout::<f32>::new(0.5).unwrap();
        d.eval();
        let input = ferrotorch_core::ones::<f32>(&[100]).unwrap();
        let output = d.forward(&input).unwrap();
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_alpha_dropout_zero_prob_is_identity() {
        let d = AlphaDropout::<f32>::new(0.0).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[100]).unwrap();
        let output = d.forward(&input).unwrap();
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_alpha_dropout_invalid_p() {
        assert!(AlphaDropout::<f32>::new(1.0).is_err());
        assert!(AlphaDropout::<f32>::new(-0.1).is_err());
        assert!(AlphaDropout::<f32>::new(1.5).is_err());
    }

    #[test]
    fn test_alpha_dropout_no_parameters() {
        let d = AlphaDropout::<f32>::new(0.3).unwrap();
        assert!(d.parameters().is_empty());
    }

    #[test]
    fn test_alpha_dropout_backward_routes_gradient() {
        let d = AlphaDropout::<f32>::new(0.5).unwrap();
        let input = leaf_tensor(&[1.0; 1000], &[1000], true);
        let output = d.forward(&input).unwrap();

        let out_data = output.data().unwrap().to_vec();
        let total: f32 = out_data.iter().sum();

        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let t = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(t)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward {
                input: output.clone(),
            }),
        )
        .unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();

        // Gradient should have two types of values: 0 for dropped, `a` for kept.
        let mut seen_zero = false;
        let mut seen_nonzero = false;
        for &g in grad_data {
            if g == 0.0 {
                seen_zero = true;
            } else {
                seen_nonzero = true;
            }
        }
        assert!(
            seen_zero,
            "some elements should have zero gradient (dropped)"
        );
        assert!(
            seen_nonzero,
            "some elements should have nonzero gradient (kept)"
        );
    }

    #[test]
    fn test_alpha_dropout_train_eval_toggle() {
        let mut d = AlphaDropout::<f32>::new(0.5).unwrap();
        assert!(d.is_training());
        d.eval();
        assert!(!d.is_training());
        d.train();
        assert!(d.is_training());
    }

    #[test]
    fn test_alpha_dropout_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AlphaDropout<f32>>();
        assert_send_sync::<AlphaDropout<f64>>();
    }

    // -----------------------------------------------------------------------
    // inplace=true — blocker #1446
    //
    // Mirrors torch's `_VF.dropout_` / `_VF.feature_dropout_` family
    // (`torch/nn/functional.py:1449,1516,1579,1629`): with `inplace=True` and
    // training, the input tensor's storage is mutated (mask + scale written
    // back) instead of a fresh buffer being allocated. The mask-based backward
    // keeps autograd correct.
    // -----------------------------------------------------------------------

    /// A minimal sum-reduction backward node used to drive `.backward()` in
    /// the in-place gradient tests below.
    #[derive(Debug)]
    struct SumBackward<T: Float> {
        input: Tensor<T>,
    }
    impl<T: Float> GradFn<T> for SumBackward<T> {
        fn backward(&self, _grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
            let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
            let t =
                Tensor::from_storage(TensorStorage::cpu(ones), self.input.shape().to_vec(), false)?;
            Ok(vec![Some(t)])
        }
        fn inputs(&self) -> Vec<&Tensor<T>> {
            vec![&self.input]
        }
        fn name(&self) -> &'static str {
            "SumBackward"
        }
    }

    // (a) inplace=true mutates the SAME input storage. The input buffer (all
    //     ones before forward) is overwritten with the masked / scaled values
    //     {0, 2.0}. Verified by reading `input.data()` AFTER forward.
    #[test]
    fn test_dropout_inplace_mutates_input_storage() {
        let d = Dropout::<f32>::new(0.5).unwrap().with_inplace(true);
        assert!(d.inplace());

        // Leaf without grad so we can re-read the input storage directly.
        let buf = vec![1.0f32; 10_000];
        let input = leaf_tensor(&buf, &[10_000], false);
        // Before forward: every element is 1.0.
        assert!(input.data().unwrap().iter().all(|&x| x == 1.0));

        let output = d.forward(&input).unwrap();

        // After forward: the INPUT storage itself has been mutated to the
        // post-dropout values (0.0 dropped, 2.0 = 1/(1-0.5) survivors). This
        // is the load-bearing in-place observation.
        let in_after = input.data().unwrap();
        assert!(
            in_after.contains(&0.0),
            "inplace forward must have zeroed some input elements"
        );
        for &x in in_after {
            assert!(
                x == 0.0 || (x - 2.0).abs() < 1e-6,
                "mutated input element = {x}, expected 0.0 or 2.0"
            );
        }

        // (b) The output equals the mutated input element-for-element: the
        //     in-place write and the returned buffer carry the identical mask.
        let out_data = output.data().unwrap();
        assert_eq!(out_data.len(), in_after.len());
        for (i, (&o, &x)) in out_data.iter().zip(in_after.iter()).enumerate() {
            assert_eq!(o, x, "output/input mismatch at {i}: out={o}, in={x}");
        }
    }

    // (d) eval-mode inplace is identity — torch's `F.dropout(.., training=False,
    //     inplace=True)` returns the input untouched (the `_VF.dropout_` branch
    //     is never reached because training is False; see functional.py:1448).
    #[test]
    fn test_dropout_inplace_eval_is_identity() {
        let mut d = Dropout::<f32>::new(0.5).unwrap().with_inplace(true);
        d.eval();
        let input = leaf_tensor(&[1.0; 100], &[100], false);
        let output = d.forward(&input).unwrap();
        // Identity: same tensor object returned, input storage untouched.
        assert!(output.is_same(&input));
        assert!(input.data().unwrap().iter().all(|&x| x == 1.0));
    }

    // p == 0 with inplace=true is also identity.
    #[test]
    fn test_dropout_inplace_p_zero_is_identity() {
        let d = Dropout::<f32>::new(0.0).unwrap().with_inplace(true);
        let input = leaf_tensor(&[1.0; 100], &[100], false);
        let output = d.forward(&input).unwrap();
        assert!(output.is_same(&input));
        assert!(input.data().unwrap().iter().all(|&x| x == 1.0));
    }

    // (c) backward through an in-place dropout on a grad-tracked NON-LEAF is
    //     correct: the autograd-safe policy falls back to out-of-place (no
    //     version counter to prove the shared storage is unused), so the input
    //     storage is NOT mutated, but the gradient still routes only through
    //     surviving elements (0 for dropped, 2.0 for kept) and the grad mask
    //     matches the output mask, exactly as the out-of-place path.
    #[test]
    fn test_dropout_inplace_backward_routes_through_surviving() {
        use ferrotorch_core::grad_fns::arithmetic::mul;

        let d = Dropout::<f32>::new(0.5).unwrap().with_inplace(true);
        // Non-leaf grad-tracked input: `t = x * 1` requires grad but is not a
        // leaf, so `apply_inplace_dropout` takes the out-of-place fallback
        // rather than erroring on the leaf guard.
        let x = leaf_tensor(&[1.0; 1000], &[1000], true);
        let ones = leaf_tensor(&[1.0; 1000], &[1000], false);
        let input = mul(&x, &ones).unwrap();
        assert!(input.requires_grad() && !input.is_leaf());
        let input_before = input.data().unwrap().to_vec();

        let output = d.forward(&input).unwrap();

        // Safe fallback: the grad-tracked non-leaf storage is left UNMUTATED.
        let input_after = input.data().unwrap().to_vec();
        assert_eq!(
            input_before, input_after,
            "in-place dropout on a grad-tracked non-leaf must fall back to \
             out-of-place and leave the input storage untouched (no version \
             counter to prove the shared storage is unused)"
        );

        let out_data = output.data().unwrap().to_vec();
        let total: f32 = out_data.iter().sum();
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward {
                input: output.clone(),
            }),
        )
        .unwrap();
        loss.backward().unwrap();

        // Gradient flows back to the leaf `x` through the out-of-place dropout.
        let grad = x.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();
        for &g in grad_data {
            assert!(
                g == 0.0 || (g - 2.0).abs() < 1e-6,
                "gradient element = {g}, expected 0.0 or 2.0"
            );
        }
        // grad mask matches output mask: dropped iff zero gradient.
        for (i, (&o, &g)) in out_data.iter().zip(grad_data.iter()).enumerate() {
            assert_eq!(
                o == 0.0,
                g == 0.0,
                "mismatch at index {i}: out={o}, grad={g}"
            );
        }
    }

    // (c2) in-place dropout on a grad-requiring LEAF errors, matching torch's
    //      leaf in-place guard (`torch/csrc/autograd/VariableTypeUtils.h:80-84`,
    //      "a leaf Variable that requires grad is being used in an in-place
    //      operation."). Pins #1581.
    #[test]
    fn test_dropout_inplace_on_grad_leaf_errors() {
        let original = vec![1.0f32; 100];
        let d = Dropout::<f32>::new(0.5).unwrap().with_inplace(true);
        let input = leaf_tensor(&original, &[100], true);
        assert!(input.is_leaf() && input.requires_grad());

        let err = d.forward(&input).unwrap_err();
        match err {
            FerrotorchError::InvalidArgument { message } => assert!(
                message.contains("leaf Variable that requires grad"),
                "expected torch leaf-guard message, got: {message}"
            ),
            other => panic!("expected InvalidArgument leaf-guard error, got {other:?}"),
        }
        // The leaf storage is left untouched (no partial mutation before error).
        assert_eq!(input.data().unwrap().to_vec(), original);
    }

    // (e) all four standard dropout variants honor inplace: the input storage
    //     is mutated channel-wise (or element-wise for `Dropout`).
    #[test]
    fn test_dropout2d_inplace_mutates_input_storage() {
        let d = Dropout2d::<f32>::new(0.5).unwrap().with_inplace(true);
        assert!(d.inplace());
        let input = leaf_tensor(&[1.0; 2 * 500 * 4], &[2, 500, 2, 2], false);
        let _ = d.forward(&input).unwrap();
        let in_after = input.data().unwrap();
        // Channel-wise: each (b, c) block of 4 spatial elems is uniform.
        let spatial = 4;
        let mut saw_dropped = false;
        for blk in in_after.chunks(spatial) {
            let first = blk[0];
            assert!(blk.iter().all(|&x| (x - first).abs() < 1e-6));
            assert!(first == 0.0 || (first - 2.0).abs() < 1e-6);
            if first == 0.0 {
                saw_dropped = true;
            }
        }
        assert!(
            saw_dropped,
            "inplace dropout2d must have zeroed some channels"
        );
    }

    #[test]
    fn test_dropout1d_inplace_mutates_input_storage() {
        let d = Dropout1d::<f32>::new(0.5).unwrap().with_inplace(true);
        assert!(d.inplace());
        let input = leaf_tensor(&[1.0; 500 * 4], &[1, 500, 4], false);
        let _ = d.forward(&input).unwrap();
        let in_after = input.data().unwrap();
        let mut saw_dropped = false;
        for blk in in_after.chunks(4) {
            let first = blk[0];
            assert!(blk.iter().all(|&x| (x - first).abs() < 1e-6));
            assert!(first == 0.0 || (first - 2.0).abs() < 1e-6);
            if first == 0.0 {
                saw_dropped = true;
            }
        }
        assert!(
            saw_dropped,
            "inplace dropout1d must have zeroed some channels"
        );
    }

    #[test]
    fn test_dropout3d_inplace_mutates_input_storage() {
        let d = Dropout3d::<f32>::new(0.5).unwrap().with_inplace(true);
        assert!(d.inplace());
        let input = leaf_tensor(&[1.0; 500 * 8], &[1, 500, 2, 2, 2], false);
        let _ = d.forward(&input).unwrap();
        let in_after = input.data().unwrap();
        let mut saw_dropped = false;
        for blk in in_after.chunks(8) {
            let first = blk[0];
            assert!(blk.iter().all(|&x| (x - first).abs() < 1e-6));
            assert!(first == 0.0 || (first - 2.0).abs() < 1e-6);
            if first == 0.0 {
                saw_dropped = true;
            }
        }
        assert!(
            saw_dropped,
            "inplace dropout3d must have zeroed some channels"
        );
    }

    // The non-inplace path is the default and leaves the input untouched —
    // confirms inplace=false (existing behavior) is preserved.
    #[test]
    fn test_dropout_default_is_not_inplace() {
        let d = Dropout::<f32>::new(0.5).unwrap();
        assert!(!d.inplace());
        let input = leaf_tensor(&[1.0; 1000], &[1000], false);
        let _ = d.forward(&input).unwrap();
        // Input untouched: still all ones.
        assert!(input.data().unwrap().iter().all(|&x| x == 1.0));
    }

    // AlphaDropout / FeatureAlphaDropout carry the `inplace` field for ABI
    // parity but — matching torch's module forward (`dropout.py:265-269`,
    // `319-323`, which never pass `self.inplace` to the functional) — do NOT
    // mutate the input even when inplace=true. The field is observable via the
    // `inplace()` getter.
    #[test]
    fn test_alpha_dropout_inplace_field_does_not_mutate() {
        let d = AlphaDropout::<f32>::new(0.5).unwrap().with_inplace(true);
        assert!(d.inplace(), "field is retained for API parity");
        let input = leaf_tensor(&[1.0; 1000], &[1000], false);
        let _ = d.forward(&input).unwrap();
        // Matching torch: the module forward ignores inplace, input untouched.
        assert!(
            input.data().unwrap().iter().all(|&x| x == 1.0),
            "AlphaDropout module forward must not mutate in place (matches torch dropout.py:265-269)"
        );
    }

    #[test]
    fn test_feature_alpha_dropout_inplace_field_does_not_mutate() {
        let d = FeatureAlphaDropout::<f32>::new(0.5)
            .unwrap()
            .with_inplace(true);
        assert!(d.inplace(), "field is retained for API parity");
        let input = leaf_tensor(&[1.0; 1000], &[1, 1000], false);
        let _ = d.forward(&input).unwrap();
        assert!(
            input.data().unwrap().iter().all(|&x| x == 1.0),
            "FeatureAlphaDropout module forward must not mutate in place (matches torch dropout.py:319-323)"
        );
    }

    // -----------------------------------------------------------------------
    // Seed-reproducible byte-match vs LIVE torch 2.11 (#1635 / #1636).
    //
    // Reference values produced by live torch under `torch.manual_seed(42)`
    // — NOT copied from the ferrotorch side (R-CHAR-3). The per-channel /
    // per-element masks come from the byte-exact MT19937 `Generator`, so a
    // shared `ferrotorch_core::manual_seed(42)` reproduces torch's stream.
    // -----------------------------------------------------------------------

    fn ones_shape_t(shape: &[usize]) -> Tensor<f32> {
        let n: usize = shape.iter().product();
        Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; n]), shape.to_vec(), false).unwrap()
    }

    /// `torch.manual_seed(42); F.dropout2d(ones(1,8,1,1),0.5,True)` per-channel
    /// -> survivors scaled by 1/(1-0.5)=2 in the MT19937 keep pattern
    /// [keep,keep,keep,keep,DROP,keep,DROP,DROP].
    #[test]
    fn test_dropout2d_seed42_matches_torch() {
        let want = [2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0];
        ferrotorch_core::rng::manual_seed(42);
        let d = Dropout2d::<f32>::new(0.5).unwrap();
        let y = d.forward(&ones_shape_t(&[1, 8, 1, 1])).unwrap();
        assert_eq!(y.data().unwrap(), &want);
    }

    /// `torch.manual_seed(42); F.dropout1d(ones(1,6,3),0.5,True)` per-channel
    /// -> [2,2,2,2,0,2], broadcast over the length-3 dim.
    #[test]
    fn test_dropout1d_seed42_matches_torch() {
        let want = [2.0, 2.0, 2.0, 2.0, 0.0, 2.0];
        ferrotorch_core::rng::manual_seed(42);
        let d = Dropout1d::<f32>::new(0.5).unwrap();
        let y = d.forward(&ones_shape_t(&[1, 6, 3])).unwrap();
        let data = y.data().unwrap();
        let per_chan: Vec<f32> = (0..6).map(|c| data[c * 3]).collect();
        assert_eq!(per_chan.as_slice(), &want);
    }

    /// `torch.manual_seed(42); F.dropout3d(ones(1,6,1,1,1),0.5,True)` per-channel
    /// -> [2,2,2,2,0,2].
    #[test]
    fn test_dropout3d_seed42_matches_torch() {
        let want = [2.0, 2.0, 2.0, 2.0, 0.0, 2.0];
        ferrotorch_core::rng::manual_seed(42);
        let d = Dropout3d::<f32>::new(0.5).unwrap();
        let y = d.forward(&ones_shape_t(&[1, 6, 1, 1, 1])).unwrap();
        assert_eq!(y.data().unwrap(), &want);
    }

    /// Two seeded `Dropout2d` forwards under the SAME `manual_seed(42)` produce
    /// the SAME mask (MT19937 reset on manual_seed; no system-time entropy).
    #[test]
    fn test_dropout2d_reproducible_under_manual_seed() {
        let d = Dropout2d::<f32>::new(0.5).unwrap();
        ferrotorch_core::rng::manual_seed(42);
        let y1 = d.forward(&ones_shape_t(&[1, 64, 1, 1])).unwrap();
        ferrotorch_core::rng::manual_seed(42);
        let y2 = d.forward(&ones_shape_t(&[1, 64, 1, 1])).unwrap();
        assert_eq!(y1.data().unwrap(), y2.data().unwrap());
    }

    /// `torch.manual_seed(42); nn.AlphaDropout(0.5).train()(ones(10))`
    /// -> kept = 1.6655989, dropped = -0.7791939 in the MT19937 keep pattern.
    /// kept/dropped values from torch's exact affine (`Dropout.cpp:74-79`),
    /// alpha = 1.7580993408473766.
    #[test]
    fn test_alpha_dropout_seed42_matches_torch() {
        let want = [
            1.6655989, 1.6655989, 1.6655989, 1.6655989, -0.7791939, 1.6655989, -0.7791939,
            -0.7791939, 1.6655989, 1.6655989,
        ];
        ferrotorch_core::rng::manual_seed(42);
        let d = AlphaDropout::<f32>::new(0.5).unwrap();
        let y = d.forward(&ones_shape_t(&[10])).unwrap();
        let got = y.data().unwrap();
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-4, "elem {i}: got {g} want {w}");
        }
    }

    /// `torch.manual_seed(42); nn.FeatureAlphaDropout(0.5).train()(ones(1,6,1,1))`
    /// per-channel -> [1.6655989 ×4, -0.7791939, 1.6655989].
    #[test]
    fn test_feature_alpha_dropout_seed42_matches_torch() {
        let want = [
            1.6655989, 1.6655989, 1.6655989, 1.6655989, -0.7791939, 1.6655989,
        ];
        ferrotorch_core::rng::manual_seed(42);
        let d = FeatureAlphaDropout::<f32>::new(0.5).unwrap();
        let y = d.forward(&ones_shape_t(&[1, 6, 1, 1])).unwrap();
        let got = y.data().unwrap();
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-4, "elem {i}: got {g} want {w}");
        }
    }
}
