//! Audit CORE-103 (crosslink #1797): empty CUDA integer `sum` / `prod` must
//! return CUDA-resident scalars, not silently demote to CPU.
//!
//! At HEAD, the CUDA arm of `reduce_op` special-cases `numel() == 0` with
//! `IntTensor::scalar(id)`, which always builds CPU storage — so the result
//! device changes solely because an input dimension is zero, while non-empty
//! CUDA reductions stay device-resident.
//!
//! Live PyTorch oracle (torch 2.11.0+cu130, RTX 3090) — R-ORACLE-1 path (b):
//!
//! ```text
//! >>> s = torch.zeros(0, dtype=torch.int32, device='cuda').sum()
//! >>> s.item(), s.device, tuple(s.shape)
//! (0, device(type='cuda', index=0), ())
//! >>> p = torch.zeros(0, dtype=torch.int32, device='cuda').prod()
//! >>> p.item(), p.device
//! (1, device(type='cuda', index=0))
//! >>> e64 = torch.zeros(0, dtype=torch.int64, device='cuda')
//! >>> e64.sum().device, e64.prod().device, e64.sum().item(), e64.prod().item()
//! (device(type='cuda', index=0), device(type='cuda', index=0), 0, 1)
//! ```
//!
//! (torch's default integer `sum`/`prod` promote dtype to int64;
//! `IntTensor::sum`/`prod` are documented same-width. The contract pinned
//! here is the RESULT DEVICE plus the identity values 0 / 1.)

use ferrotorch_core::int_tensor::IntTensor;

fn empty32() -> IntTensor<i32> {
    IntTensor::from_vec(Vec::new(), vec![0]).unwrap()
}

fn empty64() -> IntTensor<i64> {
    IntTensor::from_vec(Vec::new(), vec![0]).unwrap()
}

// ── CPU regression guards: empty reductions stay CPU with the identity ─────

#[test]
fn cpu_empty_sum_is_cpu_zero() {
    let r = empty32().sum().unwrap();
    assert!(!r.is_cuda());
    assert_eq!(r.shape(), &[] as &[usize]);
    assert_eq!(r.data().unwrap(), &[0]);
}

#[test]
fn cpu_empty_prod_is_cpu_one() {
    let r = empty64().prod().unwrap();
    assert!(!r.is_cuda());
    assert_eq!(r.data().unwrap(), &[1]);
}

// ── CUDA lane: the finding ──────────────────────────────────────────────────

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::device::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialise for CORE-103 gpu lane");
        });
    }

    #[test]
    fn cuda_empty_sum_i32_stays_cuda_resident() {
        ensure_cuda_backend();
        let g = empty32().to(Device::Cuda(0)).unwrap();
        assert!(g.is_cuda(), "precondition: input is CUDA-resident");
        let r = g.sum().unwrap();
        // R-ORACLE-3: assert the result device. torch: cuda:0.
        assert!(
            r.is_cuda(),
            "empty CUDA sum must stay CUDA-resident, got device {:?}",
            r.device()
        );
        assert_eq!(r.shape(), &[] as &[usize], "sum reduces to a 0-d scalar");
        assert_eq!(r.to(Device::Cpu).unwrap().data().unwrap(), &[0]);
    }

    #[test]
    fn cuda_empty_prod_i32_stays_cuda_resident() {
        ensure_cuda_backend();
        let g = empty32().to(Device::Cuda(0)).unwrap();
        let r = g.prod().unwrap();
        assert!(
            r.is_cuda(),
            "empty CUDA prod must stay CUDA-resident, got device {:?}",
            r.device()
        );
        assert_eq!(r.shape(), &[] as &[usize]);
        assert_eq!(r.to(Device::Cpu).unwrap().data().unwrap(), &[1]);
    }

    #[test]
    fn cuda_empty_sum_i64_stays_cuda_resident() {
        ensure_cuda_backend();
        let g = empty64().to(Device::Cuda(0)).unwrap();
        let r = g.sum().unwrap();
        assert!(
            r.is_cuda(),
            "empty CUDA i64 sum must stay CUDA-resident, got device {:?}",
            r.device()
        );
        assert_eq!(r.to(Device::Cpu).unwrap().data().unwrap(), &[0]);
    }

    #[test]
    fn cuda_empty_prod_i64_stays_cuda_resident() {
        ensure_cuda_backend();
        let g = empty64().to(Device::Cuda(0)).unwrap();
        let r = g.prod().unwrap();
        assert!(
            r.is_cuda(),
            "empty CUDA i64 prod must stay CUDA-resident, got device {:?}",
            r.device()
        );
        assert_eq!(r.to(Device::Cpu).unwrap().data().unwrap(), &[1]);
    }

    #[test]
    fn cuda_empty_min_max_still_error() {
        // PyTorch: min/max of an empty tensor is an error on every device.
        // Pin that the device-residency fix does not change this contract.
        ensure_cuda_backend();
        let g = empty32().to(Device::Cuda(0)).unwrap();
        assert!(g.min().is_err(), "empty CUDA min must error");
        assert!(g.max().is_err(), "empty CUDA max must error");
    }

    #[test]
    fn cuda_nonempty_sum_still_cuda_resident_guard() {
        // Guard: the non-empty path was already resident; keep it that way.
        ensure_cuda_backend();
        let g = IntTensor::<i32>::from_vec(vec![1, 2, 3], vec![3])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let r = g.sum().unwrap();
        assert!(r.is_cuda());
        assert_eq!(r.to(Device::Cpu).unwrap().data().unwrap(), &[6]);
    }
}
