//! Gradient clipping utilities.
//!
//! This module previously hosted hand-rolled CPU-only forks of
//! `clip_grad_norm_` and `clip_grad_value_`. Pass 5.B.2 (#1104) deduplicates
//! those forks by re-exporting the canonical, device-dispatching
//! implementations from `ferrotorch_nn::utils`.
//!
//! Both functions mirror PyTorch's `torch.nn.utils.clip_grad_norm_` /
//! `torch.nn.utils.clip_grad_value_`:
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`clip_grad_norm_`] | Clip total gradient norm across all parameters |
//! | [`clip_grad_value_`] | Clamp each gradient element to `[-clip_value, clip_value]` |
//!
//! Device dispatch (CPU / CUDA f32+f64 / mixed-device error) is handled by the
//! underlying impl in `ferrotorch_nn::utils`. See that module for the dispatch
//! policy and CUDA-kernel boundaries.
//!
//! [CL-334] Add gradient checkpointing, autocast context, gradient clipping, and EMA callback
//! [CL-1104] Deduplicate train's clip_grad_* with ferrotorch-nn's device-dispatching versions
//!
//! ## REQ status (per `.design/ferrotorch-train/grad_utils.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: `pub use ferrotorch_nn::utils::{clip_grad_norm_, clip_grad_value_};` at `ferrotorch-train/src/grad_utils.rs:23`; consumer: `ferrotorch-train/src/lib.rs:179` `pub use grad_utils::{clip_grad_norm_, clip_grad_value_};` re-exports at the crate root. |
//! | REQ-2 | SHIPPED | impl: structural — `pub use` IS the deduplication; consumer: `ferrotorch-train/src/grad_utils.rs:277, :295` use `std::ptr::fn_addr_eq` to assert the symbol identity matches `ferrotorch_nn::clip_grad_norm_` / `clip_grad_value_`; production usage at `lib.rs:179` consumes the deduplicated re-export. |
//! | REQ-3 | SHIPPED | impl: behavioral contract owned by `ferrotorch_nn::utils::clip_grad_norm_`; consumer: `ferrotorch-train/src/lib.rs:179` re-export ladder. |
//! | REQ-4 | SHIPPED | impl: `clip_grad_value_` re-exported from `ferrotorch_nn::utils`; consumer: same `lib.rs:179` ladder. |
//! | REQ-5 | SHIPPED | impl: no-grad handling owned by `ferrotorch_nn::utils`; consumer: same `lib.rs:179` ladder; behavior pinned by `ferrotorch-train/src/grad_utils.rs:166`. |
//! | REQ-6 | SHIPPED | impl: device dispatch owned by `ferrotorch_nn::utils`; consumer: same `lib.rs:179` ladder. |

pub use ferrotorch_nn::utils::{clip_grad_norm_, clip_grad_value_};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// The behavioral tests below exercise the re-exported `ferrotorch_nn::utils`
// implementation through this module's public re-export path. They pinned the
// CPU semantics that the old train fork implemented and continue to pin them
// against the canonical impl. The two `*_is_nn_*` tests are the structural
// discriminator: they assert that after deduplication the train and nn names
// resolve to the same function symbol — if a future change reintroduces a
// fork those tests fail loudly.

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{FerrotorchResult, Float, Tensor, TensorStorage};
    use ferrotorch_nn::Parameter;

    fn make_param_with_grad(data: &[f32], grad_data: &[f32], shape: &[usize]) -> Parameter<f32> {
        let p = Parameter::from_slice(data, shape).unwrap();
        let grad = Tensor::from_storage(
            TensorStorage::cpu(grad_data.to_vec()),
            shape.to_vec(),
            false,
        )
        .unwrap();
        p.set_grad(Some(grad)).unwrap();
        p
    }

    // -- clip_grad_norm_ with L2 norm ----------------------------------------

    #[test]
    fn test_clip_grad_norm_l2_clips_when_above() {
        // Gradient = [3.0, 4.0], L2 norm = 5.0. Max norm = 2.5.
        let p = make_param_with_grad(&[1.0, 2.0], &[3.0, 4.0], &[2]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        let total_norm = clip_grad_norm_(&params, 2.5, 2.0).unwrap();

        assert!(
            (total_norm - 5.0).abs() < 1e-5,
            "total norm should be 5.0, got {total_norm}"
        );

        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        // clip_coef ~= 2.5 / 5.0 = 0.5 (the canonical nn impl uses no epsilon;
        // the previous train fork used (total_norm + 1e-6) which differed by
        // ~1e-7 — within the existing 1e-4 tolerance below).
        // clipped = [3.0 * 0.5, 4.0 * 0.5] = [1.5, 2.0]
        let expected_coef = 2.5 / 5.0;
        assert!(
            (data[0] - 3.0 * expected_coef as f32).abs() < 1e-4,
            "expected ~1.5, got {}",
            data[0]
        );
        assert!(
            (data[1] - 4.0 * expected_coef as f32).abs() < 1e-4,
            "expected ~2.0, got {}",
            data[1]
        );
    }

    #[test]
    fn test_clip_grad_norm_l2_no_clip_when_below() {
        // Gradient = [0.1, 0.2], L2 norm ~= 0.2236. Max norm = 1.0.
        let p = make_param_with_grad(&[1.0, 2.0], &[0.1, 0.2], &[2]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        let total_norm = clip_grad_norm_(&params, 1.0, 2.0).unwrap();

        let expected_norm = (0.01_f64 + 0.04).sqrt();
        assert!(
            (total_norm - expected_norm).abs() < 1e-5,
            "total norm should be {expected_norm}, got {total_norm}"
        );

        // Gradients should be unchanged.
        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 0.1).abs() < 1e-6);
        assert!((data[1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_norm_multiple_params() {
        // Two parameters: grads = [3.0] and [4.0]. Total L2 norm = 5.0.
        let p1 = make_param_with_grad(&[1.0], &[3.0], &[1]);
        let p2 = make_param_with_grad(&[2.0], &[4.0], &[1]);
        let params: Vec<&Parameter<f32>> = vec![&p1, &p2];

        let total_norm = clip_grad_norm_(&params, 2.5, 2.0).unwrap();
        assert!((total_norm - 5.0).abs() < 1e-5);

        let coef = 2.5 / 5.0;
        let g1 = p1.grad().unwrap().unwrap().data().unwrap()[0];
        let g2 = p2.grad().unwrap().unwrap().data().unwrap()[0];
        assert!((g1 - 3.0 * coef as f32).abs() < 1e-4);
        assert!((g2 - 4.0 * coef as f32).abs() < 1e-4);
    }

    // -- clip_grad_norm_ with L1 norm ----------------------------------------

    #[test]
    fn test_clip_grad_norm_l1() {
        // Gradient = [3.0, -4.0], L1 norm = 7.0. Max norm = 3.5.
        let p = make_param_with_grad(&[1.0, 2.0], &[3.0, -4.0], &[2]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        let total_norm = clip_grad_norm_(&params, 3.5, 1.0).unwrap();
        assert!((total_norm - 7.0).abs() < 1e-5);

        let coef = 3.5 / 7.0;
        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 3.0 * coef as f32).abs() < 1e-4);
        assert!((data[1] - (-4.0 * coef as f32)).abs() < 1e-4);
    }

    // -- clip_grad_norm_ with inf norm ---------------------------------------

    #[test]
    fn test_clip_grad_norm_inf() {
        // Gradient = [3.0, -7.0], inf norm = 7.0. Max norm = 3.5.
        let p = make_param_with_grad(&[1.0, 2.0], &[3.0, -7.0], &[2]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        let total_norm = clip_grad_norm_(&params, 3.5, f64::INFINITY).unwrap();
        assert!((total_norm - 7.0).abs() < 1e-5);

        let coef = 3.5 / 7.0;
        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 3.0 * coef as f32).abs() < 1e-4);
        assert!((data[1] - (-7.0 * coef as f32)).abs() < 1e-4);
    }

    // -- clip_grad_norm_ with no gradients -----------------------------------

    #[test]
    fn test_clip_grad_norm_no_gradients() {
        let p = Parameter::<f32>::zeros(&[3]).unwrap();
        // No gradient set.
        let params: Vec<&Parameter<f32>> = vec![&p];

        let total_norm = clip_grad_norm_(&params, 1.0, 2.0).unwrap();
        assert!((total_norm - 0.0).abs() < 1e-12);
    }

    // -- clip_grad_norm_ with zero gradients ---------------------------------

    #[test]
    fn test_clip_grad_norm_zero_gradients() {
        let p = make_param_with_grad(&[1.0, 2.0], &[0.0, 0.0], &[2]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        let total_norm = clip_grad_norm_(&params, 1.0, 2.0).unwrap();
        assert!(total_norm < 1e-12);

        // Gradients remain zero.
        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0]).abs() < 1e-12);
        assert!((data[1]).abs() < 1e-12);
    }

    // -- clip_grad_value_ ----------------------------------------------------

    #[test]
    fn test_clip_grad_value_clips_large() {
        let p = make_param_with_grad(&[1.0, 2.0, 3.0], &[10.0, -10.0, 0.5], &[3]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        clip_grad_value_(&params, 1.0).unwrap();

        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!(
            (data[0] - 1.0).abs() < 1e-6,
            "expected 1.0, got {}",
            data[0]
        );
        assert!(
            (data[1] - (-1.0)).abs() < 1e-6,
            "expected -1.0, got {}",
            data[1]
        );
        assert!(
            (data[2] - 0.5).abs() < 1e-6,
            "expected 0.5, got {}",
            data[2]
        );
    }

    #[test]
    fn test_clip_grad_value_no_clip_needed() {
        let p = make_param_with_grad(&[1.0, 2.0], &[0.3, -0.3], &[2]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        clip_grad_value_(&params, 1.0).unwrap();

        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 0.3).abs() < 1e-6);
        assert!((data[1] - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_value_no_gradients() {
        let p = Parameter::<f32>::zeros(&[2]).unwrap();
        let params: Vec<&Parameter<f32>> = vec![&p];

        // Should succeed with no-op.
        clip_grad_value_(&params, 1.0).unwrap();
    }

    #[test]
    fn test_clip_grad_value_multiple_params() {
        let p1 = make_param_with_grad(&[1.0], &[5.0], &[1]);
        let p2 = make_param_with_grad(&[2.0], &[-5.0], &[1]);
        let params: Vec<&Parameter<f32>> = vec![&p1, &p2];

        clip_grad_value_(&params, 2.0).unwrap();

        assert!((p1.grad().unwrap().unwrap().data().unwrap()[0] - 2.0).abs() < 1e-6);
        assert!((p2.grad().unwrap().unwrap().data().unwrap()[0] - (-2.0)).abs() < 1e-6);
    }

    // -- Send + Sync ---------------------------------------------------------

    #[test]
    fn test_functions_are_callable() {
        // Smoke test: the functions exist and have the expected signatures.
        fn _test_norm<T: Float>(params: &[&Parameter<T>]) {
            let _ = clip_grad_norm_(params, 1.0, 2.0);
        }
        fn _test_value<T: Float>(params: &[&Parameter<T>]) {
            let _ = clip_grad_value_(params, 1.0);
        }
    }

    // -----------------------------------------------------------------------
    // Deduplication discriminators (Pass 5.B.2 / #1104)
    //
    // After the train re-export, `ferrotorch_train::clip_grad_norm_` and
    // `ferrotorch_nn::clip_grad_norm_` MUST resolve to the same function
    // symbol. If a future change reintroduces a fork (a thin wrapper, an
    // alternative impl, etc.) these tests fail.
    // -----------------------------------------------------------------------

    #[test]
    fn train_clip_grad_norm_is_nn_clip_grad_norm() {
        // From inside the train crate's own #[cfg(test)] module the crate
        // name `ferrotorch_train` is not importable (E0433); use `crate::`
        // which resolves to the same path through `lib.rs`'s `pub use
        // grad_utils::{clip_grad_norm_, clip_grad_value_}` re-export.
        let train_fn: fn(&[&Parameter<f32>], f64, f64) -> FerrotorchResult<f64> =
            crate::clip_grad_norm_::<f32>;
        let nn_fn: fn(&[&Parameter<f32>], f64, f64) -> FerrotorchResult<f64> =
            ferrotorch_nn::clip_grad_norm_::<f32>;
        assert!(
            std::ptr::fn_addr_eq(train_fn, nn_fn),
            "after deduplication, ferrotorch_train::clip_grad_norm_ and \
             ferrotorch_nn::clip_grad_norm_ must resolve to the same function \
             symbol; otherwise we still have a duplicate-and-drift bug"
        );
    }

    #[test]
    fn train_clip_grad_value_is_nn_clip_grad_value() {
        let train_fn: fn(&[&Parameter<f32>], f64) -> FerrotorchResult<()> =
            crate::clip_grad_value_::<f32>;
        let nn_fn: fn(&[&Parameter<f32>], f64) -> FerrotorchResult<()> =
            ferrotorch_nn::clip_grad_value_::<f32>;
        assert!(
            std::ptr::fn_addr_eq(train_fn, nn_fn),
            "after deduplication, ferrotorch_train::clip_grad_value_ and \
             ferrotorch_nn::clip_grad_value_ must resolve to the same function \
             symbol; otherwise we still have a duplicate-and-drift bug"
        );
    }
}
