//! Gradient checkpointing (activation checkpointing / rematerialization).
//!
//! Wraps [`ferrotorch_core::autograd::checkpoint::checkpoint`] with
//! higher-level utilities for training:
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`checkpoint`] | Re-export of the core single-input checkpoint |
//! | [`checkpoint_sequential`] | Checkpoint a sequence of `Arc<dyn Module<T>>` in segments |
//!
//! # How it works
//!
//! Each segment is routed through `ferrotorch_core`'s `checkpoint` primitive:
//! during the forward pass the segment's intermediate activations are **not**
//! saved; during the backward pass the segment's forward is re-executed to
//! recompute them, trading compute for memory. On CUDA, the core primitive
//! also saves and restores the GPU RNG state so that stochastic ops
//! (e.g. dropout) produce identical masks during recomputation, and it
//! preserves the surrounding autocast state.
//!
//! Each segment gets its own `CheckpointBackward` node — `segments` controls
//! the compute/memory trade-off (more segments => more savings, more
//! recomputation). Nested checkpoints are supported because each segment's
//! closure is itself a `checkpoint` call: an outer `checkpoint` re-running
//! a closure that itself calls `checkpoint` will trigger the inner
//! save/restore from within the outer recomputation.
//!
//! # Why `Arc<dyn Module<T>>`
//!
//! `ferrotorch_core::autograd::checkpoint::checkpoint` requires its closure
//! to be `'static + Send + Sync` — the closure is stored on the
//! `CheckpointBackward` node and called during backward, which may happen
//! at any later point. Therefore we cannot capture a borrow of a `&[M]`
//! slice from the caller. The honest signature takes the modules **by
//! value** as a `Vec<Arc<dyn Module<T>>>`: each segment's closure clones
//! the relevant `Arc`s into itself, satisfying `'static`.
//!
//! # Examples
//!
//! ```ignore
//! use std::sync::Arc;
//! use ferrotorch_nn::Module;
//! use ferrotorch_train::checkpoint;
//!
//! // Wrap a single expensive layer.
//! let output = checkpoint(|x| layer.forward(x), &input)?;
//!
//! // Wrap a sequence of modules: each segment gets its own checkpoint.
//! let modules: Vec<Arc<dyn Module<f32>>> = layers
//!     .into_iter()
//!     .map(|m| Arc::new(m) as Arc<dyn Module<f32>>)
//!     .collect();
//! let output = checkpoint_sequential(modules, 3, &input)?;
//! ```
//!
//! [CL-334] Add gradient checkpointing, autocast context, gradient clipping, and EMA callback
//! [CL-1108] Pass 5.C.1: route `checkpoint_sequential` through the core checkpoint primitive
//!
//! ## REQ status (per `.design/ferrotorch-train/checkpoint.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: `pub use ferrotorch_core::autograd::checkpoint::checkpoint;` at `ferrotorch-train/src/checkpoint.rs:61`; consumer: `ferrotorch-train/src/checkpoint.rs:140` invokes `checkpoint(move |x| { ... }, &current)` — same-file production consumer. |
//! | REQ-2 | NOT-STARTED | open prereq blocker #1502 — `pub fn checkpoint_sequential` at `:102-153` is shipped but no in-tree caller invokes it outside the unit-test module. |
//! | REQ-3 | NOT-STARTED | open prereq blocker #1502 — no-grad shortcut at `:126-134` is shipped but only exercised by `test_checkpoint_sequential_no_grad_skips_checkpoint`. |
//! | REQ-4 | NOT-STARTED | open prereq blocker #1502 — `checkpoint(move |x| { ... }, &current)` at `:140-149` is the production-side wiring for the segment wrap, reachable only through `checkpoint_sequential` which has no production caller. |
//! | REQ-5 | NOT-STARTED | open prereq blocker #1502 — the `move` closure at `:141-147` captures `Vec<Arc<dyn Module<T>>>` satisfying `'static + Send + Sync`, but the only caller of the surrounding `checkpoint_sequential` is the unit test. |
//! | REQ-6 | NOT-STARTED | open prereq blocker #1502 — recomputation-on-backward behavior is pinned by `test_checkpoint_sequential_real_checkpoint_grad_fn` at `:336-407`, but no production training loop drives the recomputation. |

use std::sync::Arc;

pub use ferrotorch_core::autograd::checkpoint::checkpoint;

use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::Module;

/// Apply gradient checkpointing to a sequence of modules in segments.
///
/// Splits `modules` into `segments` roughly equal contiguous groups and wraps
/// each group in a [`checkpoint`] call. This is useful for models like
/// ResNets or Transformers where the backbone is a long sequence of repeated
/// blocks: storing every intermediate activation across the whole sequence
/// is expensive, but storing only the segment-boundary activations and
/// recomputing within each segment during backward keeps peak memory
/// proportional to a single segment's depth.
///
/// Each segment produces its own `CheckpointBackward` grad_fn node — the
/// returned tensor's grad_fn chain therefore reports `"CheckpointBackward"`
/// for the last segment (and one per segment further up the chain), which
/// is the structural signal that distinguishes a real checkpoint from a
/// straight-through forward.
///
/// # Arguments
///
/// * `modules` - The modules to run in sequence, as `Arc<dyn Module<T>>`.
///   Taking ownership of `Arc`-wrapped trait objects is what makes each
///   segment's closure `'static + Send + Sync` (required by the core
///   primitive).
/// * `segments` - Number of checkpoint segments. Each segment saves/restores
///   independently. More segments => more memory savings but more
///   recomputation.
/// * `input` - The input tensor.
///
/// # Returns
///
/// The output tensor. If `input.requires_grad()`, the grad_fn chain
/// contains one `CheckpointBackward` per segment; otherwise the forward is
/// just chained directly (no autograd work to do).
///
/// # Panics
///
/// Panics if `segments == 0` or `modules` is empty.
pub fn checkpoint_sequential<T: Float>(
    modules: Vec<Arc<dyn Module<T>>>,
    segments: usize,
    input: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    assert!(segments > 0, "segments must be > 0");
    assert!(!modules.is_empty(), "modules must not be empty");

    let n = modules.len();
    let seg_size = n.div_ceil(segments); // ceil division — last segment may be smaller

    let mut current = input.clone();

    // Iterate contiguous segments. We move ownership of `modules` into the
    // segment closures by draining `seg_size` modules at a time from the
    // front. Each segment closure captures only the modules it needs, so
    // the lifetime of unused modules is bounded by this function (no
    // retained references in the grad graph beyond what each segment
    // actually uses).
    let mut remaining = modules;
    while !remaining.is_empty() {
        let take = seg_size.min(remaining.len());
        let segment: Vec<Arc<dyn Module<T>>> = remaining.drain(..take).collect();

        if !current.requires_grad() {
            // No autograd to build — just chain forwards directly. The
            // core `checkpoint` already short-circuits in this case but
            // we skip it explicitly to avoid the `no_grad` wrap allocation.
            for module in &segment {
                current = module.forward(&current)?;
            }
            continue;
        }

        // Route through the core checkpoint primitive. The closure owns
        // the segment's `Arc`s, so it is `'static + Send + Sync` —
        // `Arc<dyn Module<T>>` is `Send + Sync` because `Module<T>:
        // Send + Sync` (see ferrotorch-nn/src/module.rs).
        current = checkpoint(
            move |x: &Tensor<T>| {
                let mut h = x.clone();
                for module in &segment {
                    h = module.forward(&h)?;
                }
                Ok(h)
            },
            &current,
        )?;
    }

    Ok(current)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::grad_fns::reduction::sum;
    use ferrotorch_nn::Parameter;

    #[test]
    fn test_checkpoint_reexported() {
        // Verify the re-export exists. The actual checkpoint logic is tested
        // exhaustively in ferrotorch-core. Here we just confirm the symbol
        // is accessible.
        type CheckpointFn = fn(
            fn(&Tensor<f32>) -> FerrotorchResult<Tensor<f32>>,
            &Tensor<f32>,
        ) -> FerrotorchResult<Tensor<f32>>;
        let _f: CheckpointFn = checkpoint;
    }

    // -- checkpoint_sequential -----------------------------------------------

    /// Minimal pass-through scaling module for testing.
    struct ScaleModule {
        factor: f32,
    }

    impl Module<f32> for ScaleModule {
        fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
            use ferrotorch_core::grad_fns::arithmetic::mul;
            let s = ferrotorch_core::scalar(self.factor)?;
            mul(input, &s)
        }

        fn parameters(&self) -> Vec<&Parameter<f32>> {
            vec![]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
            vec![]
        }

        fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
            vec![]
        }

        // Stateless test module — `train()` / `eval()` have nothing to
        // toggle (no dropout, no BN running stats). The `Module` trait
        // requires these methods; for a pure-functional `x * factor`
        // op there is no mode to switch. Matches the convention in
        // `tests/conformance_train.rs:ScaleModule`.
        fn train(&mut self) {
            // Stateless: no training-mode state to set. The `is_training`
            // accessor returns the fixed value `true` below — there is no
            // boolean field to mutate.
            let _ = self;
        }
        fn eval(&mut self) {
            // Stateless: no eval-mode state to set. Symmetric with `train`.
            let _ = self;
        }
        fn is_training(&self) -> bool {
            true
        }
    }

    fn scale_modules(factors: &[f32]) -> Vec<Arc<dyn Module<f32>>> {
        factors
            .iter()
            .map(|&f| Arc::new(ScaleModule { factor: f }) as Arc<dyn Module<f32>>)
            .collect()
    }

    #[test]
    fn test_checkpoint_sequential_single_segment() {
        let modules = scale_modules(&[2.0, 3.0]);
        let input = ferrotorch_core::scalar(1.0_f32).unwrap();
        let output = checkpoint_sequential(modules, 1, &input).unwrap();
        let val = output.item().unwrap();
        // 1.0 * 2.0 * 3.0 = 6.0
        assert!((val - 6.0).abs() < 1e-5, "expected 6.0, got {val}");
    }

    #[test]
    fn test_checkpoint_sequential_multiple_segments() {
        let modules = scale_modules(&[2.0, 3.0, 4.0]);
        let input = ferrotorch_core::scalar(1.0_f32).unwrap();
        let output = checkpoint_sequential(modules, 2, &input).unwrap();
        let val = output.item().unwrap();
        // 1.0 * 2.0 * 3.0 * 4.0 = 24.0
        assert!((val - 24.0).abs() < 1e-5, "expected 24.0, got {val}");
    }

    #[test]
    fn test_checkpoint_sequential_more_segments_than_modules() {
        let modules = scale_modules(&[5.0, 2.0]);
        let input = ferrotorch_core::scalar(1.0_f32).unwrap();
        // 10 segments for 2 modules — each module is its own segment.
        let output = checkpoint_sequential(modules, 10, &input).unwrap();
        let val = output.item().unwrap();
        // 1.0 * 5.0 * 2.0 = 10.0
        assert!((val - 10.0).abs() < 1e-5, "expected 10.0, got {val}");
    }

    #[test]
    #[should_panic(expected = "segments must be > 0")]
    fn test_checkpoint_sequential_zero_segments_panics() {
        let modules = scale_modules(&[1.0]);
        let input = ferrotorch_core::scalar(1.0_f32).unwrap();
        let _ = checkpoint_sequential(modules, 0, &input);
    }

    #[test]
    #[should_panic(expected = "modules must not be empty")]
    fn test_checkpoint_sequential_empty_modules_panics() {
        let modules: Vec<Arc<dyn Module<f32>>> = vec![];
        let input = ferrotorch_core::scalar(1.0_f32).unwrap();
        let _ = checkpoint_sequential(modules, 1, &input);
    }

    // -----------------------------------------------------------------------
    // Discriminating test (#1108).
    //
    // The audit found that the previous implementation had if/else branches
    // that were byte-for-byte identical — neither branch routed through the
    // core checkpoint primitive. The sabotage signature is therefore: an
    // input that requires grad runs through checkpoint_sequential, but the
    // output's grad_fn does **not** contain `CheckpointBackward` anywhere
    // in its chain (the chain reflects only the underlying module ops:
    // Mul, Add, etc.).
    //
    // This test asserts that:
    //   1. The output's grad_fn exists.
    //   2. The top-level grad_fn is `CheckpointBackward` (the outermost
    //      segment's checkpoint wrapper).
    //   3. Backward recomputation actually runs the module forwards a
    //      second time (we observe this via a side-channel counter
    //      captured inside the module's `forward`).
    //   4. The computed gradient still matches the analytic value.
    // -----------------------------------------------------------------------

    /// Module that counts how many times its `forward` is called. The
    /// counter is `Arc<AtomicUsize>` so it can be shared between the
    /// `Module` and the assertion code without `&mut`.
    struct CountingScale {
        factor: f32,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Module<f32> for CountingScale {
        fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
            use ferrotorch_core::grad_fns::arithmetic::mul;
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let s = ferrotorch_core::scalar(self.factor)?;
            mul(input, &s)
        }

        fn parameters(&self) -> Vec<&Parameter<f32>> {
            vec![]
        }
        fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
            vec![]
        }
        fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
            vec![]
        }
        // Stateless: `CountingScale` has only a `factor` constant and the
        // shared call-counter `Arc`. There is no train/eval flag to toggle.
        fn train(&mut self) {
            let _ = self;
        }
        fn eval(&mut self) {
            let _ = self;
        }
        fn is_training(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_checkpoint_sequential_real_checkpoint_grad_fn() {
        use ferrotorch_core::storage::TensorStorage;

        // Input that requires grad.
        let input = Tensor::from_storage(TensorStorage::cpu(vec![3.0_f32]), vec![1], true).unwrap();

        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let modules: Vec<Arc<dyn Module<f32>>> = vec![
            Arc::new(CountingScale {
                factor: 2.0,
                calls: Arc::clone(&calls),
            }),
            Arc::new(CountingScale {
                factor: 5.0,
                calls: Arc::clone(&calls),
            }),
        ];

        // Single segment so we get exactly one CheckpointBackward wrapping
        // both modules — this is the case the old implementation got wrong
        // (it ran the loop and never wrapped anything, leaving the raw
        // Mul grad_fns at the top of the chain).
        let output = checkpoint_sequential(modules, 1, &input).unwrap();

        // 3.0 * 2.0 * 5.0 = 30.0
        assert!((output.item().unwrap() - 30.0).abs() < 1e-5);

        // Forward should have called each module exactly once.
        let forward_calls = calls.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            forward_calls, 2,
            "forward should call each module once, got {forward_calls}"
        );

        // SABOTAGE-CATCHING ASSERTION: the top-level grad_fn must be
        // `CheckpointBackward`, not the inner `Mul` grad_fn. If the
        // implementation regresses to a straight-through forward (the
        // exact bug #1108 caught), this assertion fails because the
        // grad_fn chain would surface `Mul` at the top instead.
        let grad_fn = output
            .grad_fn()
            .expect("checkpoint_sequential output must carry a grad_fn when input requires grad");
        assert_eq!(
            grad_fn.name(),
            "CheckpointBackward",
            "checkpoint_sequential must wrap each segment in CheckpointBackward — \
             got top-level grad_fn '{}', which is the audit-flagged \
             straight-through-forward regression from #1108",
            grad_fn.name()
        );

        // Running backward should re-execute the forward (recomputation).
        // We verify this by checking the call counter increased after
        // backward, which is the *behavioral* signal that the checkpoint
        // is real (not just structurally tagged).
        let s = sum(&output).unwrap();
        s.backward().unwrap();

        let total_calls = calls.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            total_calls > forward_calls,
            "backward should re-execute the segment's forward to recompute \
             activations — got {total_calls} calls after backward, expected > {forward_calls}. \
             A straight-through forward (the #1108 bug) would leave the counter at {forward_calls}."
        );

        // Gradient correctness: d/dx [x * 2 * 5] = 10.
        let grad = input.grad().unwrap().expect("input should have a gradient");
        let g = grad.item().unwrap();
        assert!((g - 10.0).abs() < 1e-5, "expected grad=10.0, got {g}");
    }

    #[test]
    fn test_checkpoint_sequential_no_grad_skips_checkpoint() {
        // If the input does not require grad, there is no autograd graph
        // to build and we don't need to pay the checkpoint overhead. The
        // output should still be numerically correct, just without a
        // grad_fn.
        let input = ferrotorch_core::scalar(2.0_f32).unwrap();
        assert!(!input.requires_grad());
        let modules = scale_modules(&[3.0, 4.0]);
        let output = checkpoint_sequential(modules, 1, &input).unwrap();
        assert!((output.item().unwrap() - 24.0).abs() < 1e-5);
        assert!(
            output.grad_fn().is_none(),
            "no_grad path should not attach a grad_fn"
        );
    }

    #[test]
    fn test_checkpoint_sequential_multi_segment_each_wraps() {
        // With 2 modules and 2 segments, each segment gets its own
        // CheckpointBackward wrapper. The top-level grad_fn is still
        // CheckpointBackward (the second segment's wrapper), with the
        // first segment's CheckpointBackward nested inside.
        use ferrotorch_core::storage::TensorStorage;

        let input = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32]), vec![1], true).unwrap();
        let modules = scale_modules(&[2.0, 7.0]);
        let output = checkpoint_sequential(modules, 2, &input).unwrap();
        assert!((output.item().unwrap() - 14.0).abs() < 1e-5);
        assert_eq!(
            output.grad_fn().expect("must have grad_fn").name(),
            "CheckpointBackward",
        );

        // Gradient correctness through nested checkpoints.
        let s = sum(&output).unwrap();
        s.backward().unwrap();
        let g = input.grad().unwrap().unwrap().item().unwrap();
        assert!((g - 14.0).abs() < 1e-5, "expected grad=14.0, got {g}");
    }
}
