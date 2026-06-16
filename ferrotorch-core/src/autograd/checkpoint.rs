//! ## REQ status (per `.design/ferrotorch-core/autograd/checkpoint.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub fn `checkpoint`<T, F>` at `checkpoint.rs:64-102` + `struct `CheckpointBackward`<T>` at `:190-201` + `impl GradFn` at `:216-269`. Existing pub API — boundary-API grandfathering. |
//! | REQ-2 | SHIPPED | `pub fn `checkpoint_multi`<T, F>` at `checkpoint.rs:110-145` + `struct `CheckpointMultiBackward`<T>` at `:275-281` + `impl GradFn` at `:296-358`. Existing pub API — boundary-API grandfathering. |
//! | REQ-3 | SHIPPED | `saved_autocast: AutocastSnapshot` at `checkpoint.rs:200, :280`; `current_autocast_snapshot()` at `:81, :125`; `with_autocast_state` recompute wraps at `:240, :312`; consumer: every checkpoint call. |
//! | REQ-4 | SHIPPED | `fn `save_checkpoint_rng_state`` captures CPU plus CUDA input-device RNG state; consumer: every checkpoint. |
//! | REQ-5 | SHIPPED | `struct `CheckpointRngGuard`` restores saved state for recompute and explicitly restores caller state after recompute. |
//! | REQ-6 | SHIPPED | `CheckpointBackward.input` field at `checkpoint.rs:192` populated by `input.clone()` at `:93`; consumer: `input_with_grad.grad()` read at `:257`. |
//! | REQ-7 | SHIPPED | Weighted-sum recompute trick at `checkpoint.rs:248-256`; consumer: every backward of a checkpointed output. |
//! | REQ-8 | SHIPPED | Skip-attach at `checkpoint.rs:86-88, :129-132`; consumer: every inference-time checkpoint call. |
//!

use std::sync::Arc;

use crate::autograd::autocast::{AutocastSnapshot, current_autocast_snapshot, with_autocast_state};
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::GpuRngState;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Type alias for a checkpointable function: takes an input tensor and produces an output tensor.
type CheckpointFn<T> = Arc<dyn Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

/// Type alias for a multi-input checkpointable function.
type CheckpointMultiFn<T> = Arc<dyn Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

/// Run a function with gradient checkpointing.
///
/// During the forward pass, intermediate activations are **not** saved.
/// During the backward pass, the forward function is re-executed to
/// recompute them, trading compute for memory.
///
/// This is useful for very deep networks where storing all activations
/// would exceed available memory.
///
/// # Arguments
///
/// * `f` - The forward function to checkpoint. It receives the input tensor
///   and returns the output tensor.
/// * `input` - The input tensor. Must have `requires_grad = true`.
///
/// # Returns
///
/// The output tensor, with a grad_fn that will recompute `f` during backward.
///
/// # Saved inputs and storage sharing
///
/// The checkpoint stores a clone of the input tensor. Because `Tensor` is an
/// `Arc`-wrapped type, the clone shares the same underlying `TensorStorage`.
/// If the caller mutates the storage in-place between the forward and backward
/// passes (which is unusual but possible via unsafe code), the recomputation
/// will see the mutated data. This is the same behavior as PyTorch.
///
/// # RNG reproducibility
///
/// The checkpoint saves the current thread's CPU RNG state before the forward
/// pass and restores it during backward recomputation. If CUDA inputs are
/// present, it also saves the RNG state for each distinct CUDA input device.
/// This mirrors PyTorch's `preserve_rng_state=True` default: stochastic
/// operations like dropout produce identical masks during forward and
/// recomputation, while the caller's surrounding RNG state is restored after
/// backward recomputation.
///
/// # Thread-local state and rayon
///
/// **Warning:** Both [`no_grad`] and `GRAD_ENABLED` use `thread_local!`
/// storage. When `f` spawns work onto rayon worker threads (e.g., via
/// parallel iterators), those threads will **not** inherit the calling
/// thread's gradient-enabled flag. This means operations executed on rayon
/// threads inside a `no_grad` block may still record gradients. This is a
/// known limitation — fixing it properly requires per-rayon-thread state
/// propagation which is a larger effort.
pub fn checkpoint<T, F>(f: F, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
{
    use crate::autograd::no_grad::no_grad;

    // Save RNG state before the forward pass so we can restore it during
    // backward recomputation. This ensures stochastic masks are identical.
    let saved_rng = save_checkpoint_rng_state(std::slice::from_ref(input))?;

    // Capture the autocast state at forward time so the recomputation during
    // backward runs in the same mixed-precision context. Without this, a
    // checkpoint declared inside `autocast(F16, ...)` would produce f32
    // matmul outputs during recompute (different from forward) and the
    // gradients would be numerically inconsistent.
    let saved_autocast = current_autocast_snapshot();

    // Forward pass without recording the graph (saves memory).
    let output = no_grad(|| f(input))?;

    if !input.requires_grad() {
        return Ok(output);
    }

    // Wrap in a CheckpointBackward that re-runs f during backward.
    let checkpoint_fn = Arc::new(CheckpointBackward {
        func: Arc::new(f),
        input: input.clone(),
        output_shape: output.shape().to_vec(),
        saved_rng,
        saved_autocast,
    });

    let (storage, shape) = checkpoint_output_storage(output)?;
    Tensor::from_operation(storage, shape, checkpoint_fn)
}

/// Gradient checkpointing for functions with multiple tensor inputs.
///
/// Like [`checkpoint`], but the function `f` receives a slice of tensors.
/// Gradients are computed for all inputs that have `requires_grad = true`.
///
/// CPU RNG state is always saved/restored. CUDA RNG state is saved/restored
/// for each distinct CUDA input device.
pub fn checkpoint_multi<T, F>(f: F, inputs: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
{
    use crate::autograd::no_grad::no_grad;

    if inputs.is_empty() {
        return Err(crate::error::FerrotorchError::InvalidArgument {
            message: "checkpoint_multi: at least one input required".into(),
        });
    }

    let saved_rng = save_checkpoint_rng_state(inputs)?;
    let saved_autocast = current_autocast_snapshot();

    let output = no_grad(|| f(inputs))?;

    let any_requires_grad = inputs.iter().any(|t| t.requires_grad());
    if !any_requires_grad {
        return Ok(output);
    }

    let checkpoint_fn = Arc::new(CheckpointMultiBackward {
        func: Arc::new(f),
        inputs: inputs.to_vec(),
        output_shape: output.shape().to_vec(),
        saved_rng,
        saved_autocast,
    });

    let (storage, shape) = checkpoint_output_storage(output)?;
    Tensor::from_operation(storage, shape, checkpoint_fn)
}

fn checkpoint_output_storage<T: Float>(
    output: Tensor<T>,
) -> FerrotorchResult<(TensorStorage<T>, Vec<usize>)> {
    let exact_contiguous = output.is_contiguous()
        && output.storage_offset() == 0
        && output.storage_len() == output.numel();
    let packed = if exact_contiguous {
        output
    } else {
        crate::methods::contiguous_t(&output)?
    };
    packed.into_storage_and_shape()
}

#[derive(Debug, Clone)]
struct CheckpointRngState {
    cpu: crate::rng::Generator,
    gpu: Vec<GpuRngState>,
}

/// Save CPU RNG plus every distinct CUDA input-device RNG state.
fn save_checkpoint_rng_state<T: Float>(
    tensors: &[Tensor<T>],
) -> FerrotorchResult<CheckpointRngState> {
    let mut cuda_devices = Vec::new();
    for tensor in tensors {
        if let crate::device::Device::Cuda(device) = tensor.device()
            && !cuda_devices.contains(&device)
        {
            cuda_devices.push(device);
        }
    }

    Ok(CheckpointRngState {
        cpu: crate::rng::thread_rng_state(),
        gpu: save_gpu_rng_states(&cuda_devices)?,
    })
}

fn save_gpu_rng_states(devices: &[usize]) -> FerrotorchResult<Vec<GpuRngState>> {
    if devices.is_empty() {
        return Ok(Vec::new());
    }
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    devices
        .iter()
        .map(|&device| backend.save_rng_state(device))
        .collect()
}

fn restore_gpu_rng_states(states: &[GpuRngState]) -> FerrotorchResult<()> {
    if states.is_empty() {
        return Ok(());
    }
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    for &state in states {
        backend.restore_rng_state(state)?;
    }
    Ok(())
}

fn finish_rng_guarded<R>(
    result: FerrotorchResult<R>,
    restore: FerrotorchResult<()>,
    context: &'static str,
) -> FerrotorchResult<R> {
    match (result, restore) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(err), Ok(())) | (Ok(_), Err(err)) => Err(err),
        (Err(work_err), Err(restore_err)) => Err(FerrotorchError::Internal {
            message: format!(
                "{context} failed ({work_err}); RNG state restore also failed ({restore_err})"
            ),
        }),
    }
}

/// Fork-style RNG guard. `activate` restores the forward RNG state for
/// recomputation; `restore` returns the caller's pre-backward RNG state and
/// propagates restore errors. `Drop` is a best-effort fallback for panic paths.
struct CheckpointRngGuard {
    previous: CheckpointRngState,
    restored: bool,
}

impl CheckpointRngGuard {
    fn activate(saved: &CheckpointRngState) -> FerrotorchResult<Self> {
        let devices: Vec<usize> = saved.gpu.iter().map(|state| state.device()).collect();
        let previous = CheckpointRngState {
            cpu: crate::rng::thread_rng_state(),
            gpu: save_gpu_rng_states(&devices)?,
        };

        crate::rng::set_thread_rng_state(saved.cpu.clone());
        if let Err(err) = restore_gpu_rng_states(&saved.gpu) {
            crate::rng::set_thread_rng_state(previous.cpu.clone());
            let _ = restore_gpu_rng_states(&previous.gpu);
            return Err(err);
        }

        Ok(Self {
            previous,
            restored: false,
        })
    }

    fn restore(&mut self) -> FerrotorchResult<()> {
        if self.restored {
            return Ok(());
        }
        crate::rng::set_thread_rng_state(self.previous.cpu.clone());
        restore_gpu_rng_states(&self.previous.gpu)?;
        self.restored = true;
        Ok(())
    }
}

impl Drop for CheckpointRngGuard {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

/// Internal backward node for gradient checkpointing.
///
/// # TensorId aliasing invariant
///
/// The `input` field stores a clone of the original input tensor. Because
/// `Tensor::clone()` is an `Arc` clone, the stored tensor shares the same
/// `TensorId` as the original. This is **intentional**: the autograd engine
/// uses `TensorId` to accumulate gradients, so the checkpoint's input must
/// have the same identity as the user's input tensor. If `TensorId` were
/// reassigned on clone, gradients computed during recomputation would be
/// written to a different identity and the user would never see them.
struct CheckpointBackward<T: Float> {
    func: CheckpointFn<T>,
    input: Tensor<T>,
    output_shape: Vec<usize>,
    /// CPU plus CUDA input-device RNG state saved before the forward pass.
    /// Restored during backward recomputation so stochastic ops produce
    /// identical masks.
    saved_rng: CheckpointRngState,
    /// Autocast (enabled, dtype) state captured at forward time. Restored
    /// for the duration of the recomputation so mixed-precision ops produce
    /// numerically identical activations.
    saved_autocast: AutocastSnapshot,
}

impl<T: Float> std::fmt::Debug for CheckpointBackward<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointBackward")
            .field("func", &"<closure>")
            .field("input_shape", &self.input.shape())
            .field("output_shape", &self.output_shape)
            .field("gpu_rng_states", &self.saved_rng.gpu.len())
            .field("autocast_enabled", &self.saved_autocast.enabled)
            .field("autocast_dtype", &self.saved_autocast.dtype)
            .finish()
    }
}

impl<T: Float> crate::tensor::GradFn<T> for CheckpointBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // Re-run the forward function WITH gradient tracking to build the graph.
        //
        let mut rng_guard = CheckpointRngGuard::activate(&self.saved_rng)?;

        // Run the recomputation inside an autocast context that exactly
        // matches the forward pass and force gradient tracking on even if
        // the caller invoked backward from inside `no_grad`. Both guards are
        // RAII-backed, so caller autocast and grad-mode state are restored
        // after recomputation, including panic unwind.
        let result = crate::autograd::no_grad::enable_grad(|| {
            with_autocast_state(self.saved_autocast, || {
                let input_with_grad = self.input.clone().requires_grad_(true);
                let recomputed = (self.func)(&input_with_grad)?;

                // Use autograd to compute gradients with grad_output as the
                // upstream gradient. We compute the scalar
                // sum(recomputed * grad_output) and backprop through that;
                // this correctly propagates grad_output through chain rule.
                use crate::grad_fns::arithmetic::mul;
                use crate::grad_fns::reduction::sum;
                let weighted = mul(
                    &recomputed,
                    &grad_output.clone().requires_grad_(false).detach(),
                )?;
                let scalar = sum(&weighted)?;
                scalar.backward()?;

                let input_grad = input_with_grad.grad()?;
                Ok(vec![input_grad])
            })
        });

        let restore = rng_guard.restore();
        finish_rng_guarded(result, restore, "checkpoint backward recomputation")
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "CheckpointBackward"
    }
}

// ---------------------------------------------------------------------------
// Multi-input checkpoint backward
// ---------------------------------------------------------------------------

struct CheckpointMultiBackward<T: Float> {
    func: CheckpointMultiFn<T>,
    inputs: Vec<Tensor<T>>,
    output_shape: Vec<usize>,
    saved_rng: CheckpointRngState,
    saved_autocast: AutocastSnapshot,
}

impl<T: Float> std::fmt::Debug for CheckpointMultiBackward<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointMultiBackward")
            .field("func", &"<closure>")
            .field("num_inputs", &self.inputs.len())
            .field("output_shape", &self.output_shape)
            .field("gpu_rng_states", &self.saved_rng.gpu.len())
            .field("autocast_enabled", &self.saved_autocast.enabled)
            .field("autocast_dtype", &self.saved_autocast.dtype)
            .finish()
    }
}

impl<T: Float> crate::tensor::GradFn<T> for CheckpointMultiBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let mut rng_guard = CheckpointRngGuard::activate(&self.saved_rng)?;

        // Run recomputation under the same autocast state as the forward
        // pass and force gradient tracking on independent of the caller's
        // ambient grad mode. Both guards restore caller state on exit,
        // including panic unwind.
        let result = crate::autograd::no_grad::enable_grad(|| {
            with_autocast_state(self.saved_autocast, || {
                // Re-run forward with grad tracking on all inputs that need it.
                let inputs_with_grad: Vec<Tensor<T>> = self
                    .inputs
                    .iter()
                    .map(|t| {
                        if t.requires_grad() {
                            t.clone().requires_grad_(true)
                        } else {
                            t.clone()
                        }
                    })
                    .collect();

                let recomputed = (self.func)(&inputs_with_grad)?;

                // Backprop via weighted sum trick.
                use crate::grad_fns::arithmetic::mul;
                use crate::grad_fns::reduction::sum;
                let weighted = mul(
                    &recomputed,
                    &grad_output.clone().requires_grad_(false).detach(),
                )?;
                let scalar = sum(&weighted)?;
                scalar.backward()?;

                // Collect gradients for each input.
                let mut grads = Vec::with_capacity(self.inputs.len());
                for (orig, with_grad) in self.inputs.iter().zip(inputs_with_grad.iter()) {
                    if orig.requires_grad() {
                        grads.push(with_grad.grad()?);
                    } else {
                        grads.push(None);
                    }
                }
                Ok(grads)
            })
        });

        let restore = rng_guard.restore();
        finish_rng_guarded(result, restore, "checkpoint_multi backward recomputation")
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        self.inputs.iter().collect()
    }

    fn name(&self) -> &'static str {
        "CheckpointMultiBackward"
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::autocast::{AutocastDtype, autocast, is_autocast_enabled};
    use crate::creation::{from_slice, scalar};
    use crate::grad_fns::arithmetic::{add, mul};
    use crate::grad_fns::reduction::sum;
    use crate::storage::TensorStorage;

    fn leaf_grad(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    // -----------------------------------------------------------------------
    // Single-input checkpoint correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_single_input_basic() {
        // f(x) = (x * x) + x  -- df/dx = 2x + 1
        // For x = [1, 2, 3], grad should be [3, 5, 7].
        let x = leaf_grad(&[1.0, 2.0, 3.0], &[3]);
        let y = checkpoint(
            |t: &Tensor<f32>| {
                let sq = mul(t, t)?;
                add(&sq, t)
            },
            &x,
        )
        .unwrap();
        // sum(y) = 1+1 + 4+2 + 9+3 = 20
        let s = sum(&y).unwrap();
        assert!((s.item().unwrap() - 20.0).abs() < 1e-5);

        s.backward().unwrap();
        let g = x.grad().unwrap().expect("x should have a gradient");
        let gd = g.data().unwrap();
        assert!((gd[0] - 3.0).abs() < 1e-5);
        assert!((gd[1] - 5.0).abs() < 1e-5);
        assert!((gd[2] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_checkpoint_no_grad_input_returns_output_only() {
        // When input does not require grad, checkpoint should still produce
        // the correct output but skip wrapping in a backward node.
        let x = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let y = checkpoint(
            |t: &Tensor<f32>| {
                let two = scalar(2.0f32)?;
                mul(t, &two)
            },
            &x,
        )
        .unwrap();
        let yd = y.data().unwrap();
        assert_eq!(yd, &[2.0, 4.0, 6.0]);
        // No grad_fn since input had no grad.
        assert!(y.grad_fn().is_none());
    }

    // -----------------------------------------------------------------------
    // Multi-input checkpoint correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_multi_two_inputs_both_grad() {
        // f(a, b) = a * b + a  -- df/da = b + 1, df/db = a
        let a = leaf_grad(&[1.0, 2.0, 3.0], &[3]);
        let b = leaf_grad(&[4.0, 5.0, 6.0], &[3]);
        let y = checkpoint_multi(
            |ts: &[Tensor<f32>]| {
                let prod = mul(&ts[0], &ts[1])?;
                add(&prod, &ts[0])
            },
            &[a.clone(), b.clone()],
        )
        .unwrap();
        // y = [4+1, 10+2, 18+3] = [5, 12, 21]
        let s = sum(&y).unwrap();
        s.backward().unwrap();

        // df/da = b + 1 = [5, 6, 7]
        let ga = a.grad().unwrap().expect("a should have a gradient");
        let gad = ga.data().unwrap();
        assert!((gad[0] - 5.0).abs() < 1e-5);
        assert!((gad[1] - 6.0).abs() < 1e-5);
        assert!((gad[2] - 7.0).abs() < 1e-5);

        // df/db = a = [1, 2, 3]
        let gb = b.grad().unwrap().expect("b should have a gradient");
        let gbd = gb.data().unwrap();
        assert!((gbd[0] - 1.0).abs() < 1e-5);
        assert!((gbd[1] - 2.0).abs() < 1e-5);
        assert!((gbd[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_checkpoint_multi_partial_grad() {
        // Only the second input requires grad.
        let a = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let b = leaf_grad(&[4.0, 5.0, 6.0], &[3]);
        let y = checkpoint_multi(
            |ts: &[Tensor<f32>]| mul(&ts[0], &ts[1]),
            &[a.clone(), b.clone()],
        )
        .unwrap();
        let s = sum(&y).unwrap();
        s.backward().unwrap();

        // a has no grad, b's grad should be a.
        assert!(a.grad().unwrap().is_none());
        let gb = b.grad().unwrap().expect("b should have a gradient");
        let gbd = gb.data().unwrap();
        assert_eq!(gbd, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_checkpoint_multi_empty_inputs_errors() {
        let result = checkpoint_multi(|_: &[Tensor<f32>]| panic!("should not run"), &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_multi_no_grad_inputs_returns_output_only() {
        // None of the inputs need grad — output is computed but no backward.
        let a = from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let b = from_slice(&[3.0f32, 4.0], &[2]).unwrap();
        let y = checkpoint_multi(|ts: &[Tensor<f32>]| add(&ts[0], &ts[1]), &[a, b]).unwrap();
        let yd = y.data().unwrap();
        assert_eq!(yd, &[4.0, 6.0]);
        assert!(y.grad_fn().is_none());
    }

    // -----------------------------------------------------------------------
    // Autocast snapshot helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_current_autocast_snapshot_outside_region() {
        let snap = current_autocast_snapshot();
        assert!(!snap.enabled);
    }

    #[test]
    fn test_current_autocast_snapshot_inside_region() {
        autocast(AutocastDtype::BF16, || {
            let snap = current_autocast_snapshot();
            assert!(snap.enabled);
            assert_eq!(snap.dtype, AutocastDtype::BF16);
        });
    }

    #[test]
    fn test_with_autocast_state_restores_disabled() {
        // Snapshot disabled state, then call with_autocast_state from
        // inside an enabled region — the closure should see disabled.
        let disabled = AutocastSnapshot {
            enabled: false,
            dtype: AutocastDtype::F16,
        };
        autocast(AutocastDtype::F16, || {
            assert!(is_autocast_enabled());
            with_autocast_state(disabled, || {
                assert!(!is_autocast_enabled());
            });
            // After the closure, the surrounding autocast region is restored.
            assert!(is_autocast_enabled());
        });
    }

    #[test]
    fn test_with_autocast_state_overrides_dtype() {
        let f16_state = AutocastSnapshot {
            enabled: true,
            dtype: AutocastDtype::F16,
        };
        autocast(AutocastDtype::BF16, || {
            with_autocast_state(f16_state, || {
                assert!(is_autocast_enabled());
                assert_eq!(
                    crate::autograd::autocast::autocast_dtype(),
                    AutocastDtype::F16
                );
            });
            // Restored.
            assert_eq!(
                crate::autograd::autocast::autocast_dtype(),
                AutocastDtype::BF16
            );
        });
    }

    // -----------------------------------------------------------------------
    // Checkpoint preserves autocast state across backward recomputation
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_captures_autocast_snapshot() {
        // When checkpoint is called inside an autocast region, the saved
        // snapshot should reflect that. We can verify this by checking the
        // Debug output of the backward node — its `autocast_enabled` field
        // tracks the snapshot state.
        let x = leaf_grad(&[1.0f32, 2.0, 3.0], &[3]);
        let y_inside = autocast(AutocastDtype::F16, || {
            checkpoint(|t: &Tensor<f32>| mul(t, t), &x)
        })
        .unwrap();
        let dbg = format!("{:?}", y_inside.grad_fn().unwrap());
        assert!(
            dbg.contains("autocast_enabled: true"),
            "expected captured autocast=true in debug repr, got {dbg}"
        );
    }

    #[test]
    fn test_checkpoint_outside_autocast_captures_disabled() {
        let x = leaf_grad(&[1.0f32, 2.0, 3.0], &[3]);
        let y = checkpoint(|t: &Tensor<f32>| mul(t, t), &x).unwrap();
        let dbg = format!("{:?}", y.grad_fn().unwrap());
        assert!(
            dbg.contains("autocast_enabled: false"),
            "expected captured autocast=false in debug repr, got {dbg}"
        );
    }

    #[test]
    fn test_checkpoint_recomputation_uses_saved_autocast() {
        // The checkpoint is created inside autocast(F16). Backward is called
        // OUTSIDE any autocast region. During recomputation, autocast must
        // be re-enabled (with F16) so the inner ops see the same context.
        // We verify by inspecting the autocast state from inside the
        // recomputation closure via a shared flag.
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let saw_autocast = StdArc::new(AtomicBool::new(false));
        let saw_autocast_clone = StdArc::clone(&saw_autocast);

        let x = leaf_grad(&[1.0f32, 2.0, 3.0], &[3]);
        let y = autocast(AutocastDtype::F16, || {
            checkpoint(
                move |t: &Tensor<f32>| {
                    saw_autocast_clone.store(is_autocast_enabled(), Ordering::SeqCst);
                    mul(t, t)
                },
                &x,
            )
        })
        .unwrap();

        // Reset the flag — forward set it to true (we were in autocast).
        saw_autocast.store(false, Ordering::SeqCst);

        // Backward runs OUTSIDE any autocast region.
        assert!(!is_autocast_enabled());
        let s = sum(&y).unwrap();
        s.backward().unwrap();

        // The recomputation closure should have observed autocast = true,
        // because the saved snapshot was restored before the recomputation.
        assert!(
            saw_autocast.load(Ordering::SeqCst),
            "checkpoint backward should re-enable autocast during recomputation"
        );

        // After backward returns, the caller's autocast state is restored
        // (still disabled outside the region).
        assert!(!is_autocast_enabled());
    }

    #[test]
    fn test_checkpoint_multi_recomputation_uses_saved_autocast() {
        // Same as the single-input test but for checkpoint_multi.
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let observed = StdArc::new(AtomicUsize::new(0));
        let observed_clone = StdArc::clone(&observed);

        let a = leaf_grad(&[1.0f32, 2.0], &[2]);
        let b = leaf_grad(&[3.0f32, 4.0], &[2]);

        let y = autocast(AutocastDtype::BF16, || {
            checkpoint_multi(
                move |ts: &[Tensor<f32>]| {
                    let dtype = crate::autograd::autocast::autocast_dtype();
                    let val = if is_autocast_enabled() {
                        match dtype {
                            AutocastDtype::F16 => 1,
                            AutocastDtype::BF16 => 2,
                        }
                    } else {
                        0
                    };
                    observed_clone.store(val, Ordering::SeqCst);
                    add(&ts[0], &ts[1])
                },
                &[a.clone(), b.clone()],
            )
        })
        .unwrap();

        // Forward observed BF16 (= 2). Reset.
        observed.store(0, Ordering::SeqCst);

        let s = sum(&y).unwrap();
        s.backward().unwrap();

        // Backward recomputation should also observe BF16.
        assert_eq!(
            observed.load(Ordering::SeqCst),
            2,
            "expected recomputation to see autocast(BF16), got code {}",
            observed.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn test_checkpoint_recomputation_does_not_leak_autocast() {
        // If the checkpoint was created INSIDE autocast and backward is
        // called OUTSIDE autocast, after backward returns we should still
        // be outside autocast (the with_autocast_state RAII guard restores).
        let x = leaf_grad(&[1.0f32, 2.0], &[2]);
        let y = autocast(AutocastDtype::F16, || {
            checkpoint(|t: &Tensor<f32>| mul(t, t), &x)
        })
        .unwrap();

        assert!(!is_autocast_enabled());
        let s = sum(&y).unwrap();
        s.backward().unwrap();
        assert!(
            !is_autocast_enabled(),
            "checkpoint backward should not leak autocast state to caller"
        );
    }

    #[test]
    fn test_checkpoint_recomputation_inside_different_autocast() {
        // Forward in F16, backward called from inside BF16 region.
        // Recomputation should TEMPORARILY switch to F16, then restore BF16.
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicU8, Ordering};

        let observed = StdArc::new(AtomicU8::new(0));
        let observed_clone = StdArc::clone(&observed);

        let x = leaf_grad(&[1.0f32, 2.0], &[2]);
        let y = autocast(AutocastDtype::F16, || {
            checkpoint(
                move |t: &Tensor<f32>| {
                    let code: u8 = if is_autocast_enabled() {
                        match crate::autograd::autocast::autocast_dtype() {
                            AutocastDtype::F16 => 1,
                            AutocastDtype::BF16 => 2,
                        }
                    } else {
                        0
                    };
                    observed_clone.store(code, Ordering::SeqCst);
                    mul(t, t)
                },
                &x,
            )
        })
        .unwrap();

        observed.store(0, Ordering::SeqCst);

        autocast(AutocastDtype::BF16, || {
            let s = sum(&y).unwrap();
            s.backward().unwrap();
            // The surrounding BF16 region must be restored after backward
            // (the saved F16 snapshot only applies during the recomputation
            // closure).
            assert_eq!(
                crate::autograd::autocast::autocast_dtype(),
                AutocastDtype::BF16,
                "with_autocast_state should restore caller's BF16 state"
            );
        });

        assert_eq!(
            observed.load(Ordering::SeqCst),
            1,
            "expected recomputation to see F16 (saved snapshot), got code {}",
            observed.load(Ordering::SeqCst)
        );
    }
}
