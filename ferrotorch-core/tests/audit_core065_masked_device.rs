//! Regression tests for audit finding CORE-065 (#1759): despite the module's
//! "no silent CPU↔GPU round trips" claim, several public `MaskedTensor` ops
//! silently changed device based on the operation or edge case:
//!
//! - `filled` / `to_tensor` always returned CPU tensors for CUDA-backed data.
//! - `masked_mean` downloaded its GPU sum and always returned a CPU scalar.
//! - `masked_count` always returned a CPU scalar.
//! - CUDA `masked_min` / `masked_max` normally returned CUDA scalars, but the
//!   all-masked case returned a CPU NaN scalar.
//!
//! DEVICE CONTRACT under test (matches torch, probed live on 2.11.0+cu130,
//! 2026-06-11): every value-producing op returns a tensor ON THE DATA
//! TENSOR'S DEVICE.
//!
//! ```python
//! >>> tc = torch.tensor([1.0, 2.0, 3.0], device="cuda")
//! >>> mc = torch.tensor([True, False, True], device="cuda")
//! >>> torch.masked.sum(tc, mask=mc).device
//! device(type='cuda', index=0)
//! >>> torch.masked.mean(tc, mask=mc).device
//! device(type='cuda', index=0)
//! >>> torch.masked.amin(tc, mask=mc).device
//! device(type='cuda', index=0)
//! >>> masked_tensor(tc, mc).to_tensor(0.0).device
//! device(type='cuda', index=0)
//! >>> mc_none = torch.tensor([False, False, False], device="cuda")
//! >>> torch.masked.amax(tc, mask=mc_none)        # edge case stays on device
//! tensor(-inf, device='cuda:0')
//! >>> torch.masked.mean(tc, mask=mc_none)
//! tensor(nan, device='cuda:0')
//! ```
//!
//! The all-masked extremum VALUE stays under the #1924 NaN pin (ferrotorch
//! NaN sentinel vs torch ±inf identity); this suite asserts only that the
//! pinned NaN now lives on the data device instead of silently demoting to
//! CPU. `masked_count` has no torch.masked free-function counterpart; the
//! module device contract (result on the data tensor's device) governs it.
//!
//! Values compared bit-exact (integers, NaN-bit checks) — no tolerance needed.

use ferrotorch_core::masked::{
    MaskedTensor, masked_count, masked_max, masked_mean, masked_min, masked_sum,
};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn cpu_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}

// ---------------------------------------------------------------------------
// CPU lane: every value-producing op returns CPU for CPU data (contract guard).
// ---------------------------------------------------------------------------

#[test]
fn cpu_data_yields_cpu_results_for_every_op() {
    let x = cpu_f32(&[1.0, 2.0, 3.0]);
    let mt = MaskedTensor::new(x, vec![true, false, true]).unwrap();
    assert!(masked_sum(&mt).unwrap().is_cpu(), "masked_sum");
    assert!(masked_mean(&mt).unwrap().is_cpu(), "masked_mean");
    assert!(masked_min(&mt).unwrap().is_cpu(), "masked_min");
    assert!(masked_max(&mt).unwrap().is_cpu(), "masked_max");
    assert!(masked_count(&mt).unwrap().is_cpu(), "masked_count");
    assert!(mt.filled().unwrap().is_cpu(), "filled");
    assert!(mt.to_tensor().unwrap().is_cpu(), "to_tensor");
}

// ---------------------------------------------------------------------------
// GPU lanes — every value-producing op must return on the data device,
// including the all-masked edge cases (the pre-fix silent CPU demotions).
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::device::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the gpu masked device tests");
        });
    }

    fn cuda_f32(data: &[f32]) -> Tensor<f32> {
        cpu_f32(data).to(Device::Cuda(0)).unwrap()
    }

    fn cuda_f64(data: &[f64]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap()
    }

    /// `filled` on CUDA f32 data: result on CUDA with the same values the CPU
    /// walk produces. Oracle: `masked_tensor(tc, mc).to_tensor(0.0)` →
    /// `tensor([1., 0., 3.], device='cuda:0')`.
    #[test]
    #[allow(clippy::float_cmp, reason = "fill substitution is bit-exact")]
    fn gpu_filled_returns_cuda_with_cpu_values() {
        ensure_cuda_backend();
        let x = cuda_f32(&[1.0, 2.0, 3.0]);
        let mt = MaskedTensor::new(x, vec![true, false, true]).unwrap();
        let f = mt.filled().unwrap();
        assert_eq!(
            f.device(),
            Device::Cuda(0),
            "filled must return on the data device (CORE-065 silent CPU demotion)"
        );
        assert_eq!(f.cpu().unwrap().data().unwrap(), &[1.0, 0.0, 3.0]);
    }

    /// `to_tensor` (alias) with a fill override, f64 lane.
    #[test]
    #[allow(clippy::float_cmp, reason = "fill substitution is bit-exact")]
    fn gpu_to_tensor_fill_override_returns_cuda() {
        ensure_cuda_backend();
        let x = cuda_f64(&[1.0, 2.0, 3.0]);
        let mt = MaskedTensor::new(x, vec![true, false, true])
            .unwrap()
            .with_fill_value(-99.0);
        let f = mt.to_tensor().unwrap();
        assert_eq!(f.device(), Device::Cuda(0), "to_tensor device");
        assert_eq!(f.cpu().unwrap().data().unwrap(), &[1.0, -99.0, 3.0]);
    }

    /// `masked_mean` on CUDA: the division happens on-device; the result is a
    /// CUDA scalar (pre-fix: GPU sum downloaded, CPU scalar returned).
    #[test]
    #[allow(clippy::float_cmp, reason = "30.0 is exactly representable")]
    fn gpu_masked_mean_returns_cuda() {
        ensure_cuda_backend();
        let x = cuda_f32(&[10.0, 0.0, 30.0, 0.0, 50.0]);
        let mt = MaskedTensor::new(x, vec![true, false, true, false, true]).unwrap();
        let m = masked_mean(&mt).unwrap();
        assert_eq!(
            m.device(),
            Device::Cuda(0),
            "masked_mean must return on the data device (CORE-065)"
        );
        assert_eq!(m.cpu().unwrap().data().unwrap(), &[30.0]);
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "30.0 is exactly representable")]
    fn gpu_masked_mean_f64_returns_cuda() {
        ensure_cuda_backend();
        let x = cuda_f64(&[10.0, 0.0, 30.0, 0.0, 50.0]);
        let mt = MaskedTensor::new(x, vec![true, false, true, false, true]).unwrap();
        let m = masked_mean(&mt).unwrap();
        assert_eq!(m.device(), Device::Cuda(0), "masked_mean f64 device");
        assert_eq!(m.cpu().unwrap().data().unwrap(), &[30.0]);
    }

    /// `masked_count` on CUDA data: count is computed from the host-resident
    /// mask (by design) but the RESULT lands on the data device.
    #[test]
    #[allow(clippy::float_cmp, reason = "integer count cast to float is exact")]
    fn gpu_masked_count_returns_cuda() {
        ensure_cuda_backend();
        let x = cuda_f32(&[1.0, 2.0, 3.0, 4.0]);
        let mt = MaskedTensor::new(x, vec![true, false, true, true]).unwrap();
        let c = masked_count(&mt).unwrap();
        assert_eq!(
            c.device(),
            Device::Cuda(0),
            "masked_count must return on the data device (CORE-065)"
        );
        assert_eq!(c.cpu().unwrap().data().unwrap(), &[3.0]);
    }

    /// Regression guards: the paths that already returned CUDA keep doing so.
    #[test]
    #[allow(clippy::float_cmp, reason = "values are exact small integers")]
    fn gpu_sum_min_max_partial_stay_cuda() {
        ensure_cuda_backend();
        let x = cuda_f32(&[5.0, 1.0, 9.0, 2.0]);
        let mt = MaskedTensor::new(x, vec![true, false, false, true]).unwrap();
        let s = masked_sum(&mt).unwrap();
        assert_eq!(s.device(), Device::Cuda(0), "masked_sum device");
        assert_eq!(s.cpu().unwrap().data().unwrap(), &[7.0]);
        let mn = masked_min(&mt).unwrap();
        assert_eq!(mn.device(), Device::Cuda(0), "masked_min device");
        assert_eq!(mn.cpu().unwrap().data().unwrap(), &[2.0]);
        let mx = masked_max(&mt).unwrap();
        assert_eq!(mx.device(), Device::Cuda(0), "masked_max device");
        assert_eq!(mx.cpu().unwrap().data().unwrap(), &[5.0]);
    }

    /// THE all-masked edge from the finding: CUDA `masked_min` / `masked_max`
    /// must return their #1924-pinned NaN sentinel ON CUDA, not silently
    /// demote to a CPU scalar. (Torch keeps its ±inf identity payload on
    /// `cuda:0`; the VALUE divergence stays pinned under #1924 — only the
    /// device is asserted here against the module contract.)
    #[test]
    fn gpu_all_masked_extremum_returns_cuda_nan() {
        ensure_cuda_backend();
        type MaskedRed = fn(&MaskedTensor<f32>) -> ferrotorch_core::FerrotorchResult<Tensor<f32>>;
        for (name, op) in [
            ("masked_min", masked_min as MaskedRed),
            ("masked_max", masked_max as MaskedRed),
        ] {
            let x = cuda_f32(&[1.0, 2.0]);
            let mt = MaskedTensor::new(x, vec![false, false]).unwrap();
            let r: Tensor<f32> = op(&mt).unwrap();
            assert_eq!(
                r.device(),
                Device::Cuda(0),
                "{name} all-masked must stay on the data device (CORE-065 edge)"
            );
            let v = r.cpu().unwrap().data().unwrap()[0];
            assert!(
                v.is_nan(),
                "{name} all-masked value must remain the #1924-pinned NaN, got {v}"
            );
        }
    }

    /// All-masked `masked_mean` edge: NaN on the data device.
    #[test]
    fn gpu_all_masked_mean_returns_cuda_nan() {
        ensure_cuda_backend();
        let x = cuda_f32(&[1.0, 2.0]);
        let mt = MaskedTensor::new(x, vec![false, false]).unwrap();
        let m = masked_mean(&mt).unwrap();
        assert_eq!(
            m.device(),
            Device::Cuda(0),
            "masked_mean all-masked must stay on the data device (CORE-065 edge)"
        );
        let v = m.cpu().unwrap().data().unwrap()[0];
        assert!(v.is_nan(), "all-masked mean must be NaN, got {v}");
    }
}
