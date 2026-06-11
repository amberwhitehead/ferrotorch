//! Audit CORE-101 (crosslink #1795): integer arithmetic must wrap at the
//! element width, as documented and as PyTorch does.
//!
//! At HEAD, the CPU paths for `add`, `sub`, and `sum` widen to i64, wrap
//! there, and convert back with `unwrap_or(<left operand / accumulator>)` —
//! so `i32::MAX + 1` returns `i32::MAX` instead of wrapping to `i32::MIN`,
//! and a sum that overflows simply drops the overflowing element. The
//! `mul` / shift helpers already compute at the concrete width.
//!
//! Every numerical expectation below was read from a live PyTorch session
//! (torch 2.11.0+cu130, RTX 3090) — R-ORACLE-1 path (b):
//!
//! ```text
//! >>> (torch.tensor([2147483647], dtype=torch.int32)
//! ...  + torch.tensor([1], dtype=torch.int32)).item()
//! -2147483648
//! >>> (torch.tensor([-2147483648], dtype=torch.int32)
//! ...  - torch.tensor([1], dtype=torch.int32)).item()
//! 2147483647
//! >>> (torch.tensor([2147483647], dtype=torch.int32)
//! ...  * torch.tensor([2], dtype=torch.int32)).item()
//! -2
//! >>> (-torch.tensor([-2147483648], dtype=torch.int32)).item()
//! -2147483648
//! >>> torch.tensor([2147483647, 1], dtype=torch.int32).sum(dtype=torch.int32).item()
//! -2147483648
//! >>> torch.tensor([2147483647, 1, 5], dtype=torch.int32).sum(dtype=torch.int32).item()
//! -2147483643
//! >>> torch.tensor([-2147483648, -1], dtype=torch.int32).sum(dtype=torch.int32).item()
//! 2147483647
//! >>> torch.tensor([2147483647, 2], dtype=torch.int32).prod(dtype=torch.int32).item()
//! -2
//! >>> (torch.tensor([9223372036854775807]) + torch.tensor([1])).item()
//! -9223372036854775808
//! >>> (torch.tensor([-9223372036854775808]) - torch.tensor([1])).item()
//! 9223372036854775807
//! >>> torch.tensor([9223372036854775807, 1]).sum().item()
//! -9223372036854775808
//! >>> torch.tensor([-9223372036854775808, -1]).sum().item()
//! 9223372036854775807
//! >>> torch.tensor([9223372036854775807, 2]).prod().item()
//! -2
//! >>> (torch.tensor([2147483647], dtype=torch.int32, device='cuda')
//! ...  + torch.tensor([1], dtype=torch.int32, device='cuda')).item()
//! -2147483648
//! ```
//!
//! Note on `sum`/`prod` dtype: torch's default integer `sum()`/`prod()`
//! promote to int64; `IntTensor::sum`/`prod` are documented same-width
//! (wrapping accumulator), so the oracle passes `dtype=torch.int32`
//! explicitly for the i32 cases — same-width accumulation is the contract
//! under test.

use ferrotorch_core::int_tensor::IntTensor;

fn t32(v: &[i32]) -> IntTensor<i32> {
    IntTensor::from_vec(v.to_vec(), vec![v.len()]).unwrap()
}

fn t64(v: &[i64]) -> IntTensor<i64> {
    IntTensor::from_vec(v.to_vec(), vec![v.len()]).unwrap()
}

// ── i32 elementwise ─────────────────────────────────────────────────────────

#[test]
fn add_i32_wraps_max_plus_one_to_min() {
    let r = t32(&[i32::MAX]).add(&t32(&[1])).unwrap();
    assert_eq!(r.data().unwrap(), &[i32::MIN]);
}

#[test]
fn add_i32_wraps_min_plus_minus_one_to_max() {
    let r = t32(&[i32::MIN]).add(&t32(&[-1])).unwrap();
    assert_eq!(r.data().unwrap(), &[i32::MAX]);
}

#[test]
fn sub_i32_wraps_min_minus_one_to_max() {
    let r = t32(&[i32::MIN]).sub(&t32(&[1])).unwrap();
    assert_eq!(r.data().unwrap(), &[i32::MAX]);
}

#[test]
fn sub_i32_wraps_max_minus_minus_one_to_min() {
    let r = t32(&[i32::MAX]).sub(&t32(&[-1])).unwrap();
    assert_eq!(r.data().unwrap(), &[i32::MIN]);
}

#[test]
fn mul_i32_wraps_max_times_two() {
    let r = t32(&[i32::MAX]).mul(&t32(&[2])).unwrap();
    assert_eq!(r.data().unwrap(), &[-2]);
}

#[test]
fn neg_i32_min_is_min() {
    let r = t32(&[i32::MIN]).neg().unwrap();
    assert_eq!(r.data().unwrap(), &[i32::MIN]);
}

// ── i32 reductions ──────────────────────────────────────────────────────────

#[test]
fn sum_i32_wraps_positive_overflow() {
    let r = t32(&[i32::MAX, 1]).sum().unwrap();
    assert_eq!(r.data().unwrap(), &[i32::MIN]);
}

#[test]
fn sum_i32_does_not_drop_overflowing_element() {
    // HEAD bug: the first overflow returns the stale accumulator, silently
    // ignoring the element — [MAX, 1, 5] summed to MAX + 5 instead of the
    // wrapped MIN + 5. Oracle: -2147483643.
    let r = t32(&[i32::MAX, 1, 5]).sum().unwrap();
    assert_eq!(r.data().unwrap(), &[-2147483643]);
}

#[test]
fn sum_i32_wraps_negative_overflow() {
    let r = t32(&[i32::MIN, -1]).sum().unwrap();
    assert_eq!(r.data().unwrap(), &[i32::MAX]);
}

#[test]
fn prod_i32_wraps_overflow() {
    let r = t32(&[i32::MAX, 2]).prod().unwrap();
    assert_eq!(r.data().unwrap(), &[-2]);
}

// ── i64 elementwise ─────────────────────────────────────────────────────────

#[test]
fn add_i64_wraps_max_plus_one_to_min() {
    let r = t64(&[i64::MAX]).add(&t64(&[1])).unwrap();
    assert_eq!(r.data().unwrap(), &[i64::MIN]);
}

#[test]
fn sub_i64_wraps_min_minus_one_to_max() {
    let r = t64(&[i64::MIN]).sub(&t64(&[1])).unwrap();
    assert_eq!(r.data().unwrap(), &[i64::MAX]);
}

#[test]
fn mul_i64_wraps_max_times_two() {
    let r = t64(&[i64::MAX]).mul(&t64(&[2])).unwrap();
    assert_eq!(r.data().unwrap(), &[-2]);
}

#[test]
fn neg_i64_min_is_min() {
    let r = t64(&[i64::MIN]).neg().unwrap();
    assert_eq!(r.data().unwrap(), &[i64::MIN]);
}

// ── i64 reductions ──────────────────────────────────────────────────────────

#[test]
fn sum_i64_wraps_positive_overflow() {
    let r = t64(&[i64::MAX, 1]).sum().unwrap();
    assert_eq!(r.data().unwrap(), &[i64::MIN]);
}

#[test]
fn sum_i64_wraps_negative_overflow() {
    let r = t64(&[i64::MIN, -1]).sum().unwrap();
    assert_eq!(r.data().unwrap(), &[i64::MAX]);
}

#[test]
fn prod_i64_wraps_overflow() {
    let r = t64(&[i64::MAX, 2]).prod().unwrap();
    assert_eq!(r.data().unwrap(), &[-2]);
}

// ── CUDA lane: GPU kernels wrap; CPU must agree (CPU/GPU parity pin) ───────
//
// The PTX kernels (`add.s32` etc.) already wrap at the element width; the
// CORE-101 bug was CPU-only. These pin (a) GPU residency of the result
// (R-ORACLE-3) and (b) bit-identical CPU/GPU boundary behaviour.

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::device::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialise for CORE-101 gpu lane");
        });
    }

    fn check_i32(
        op: impl Fn(&IntTensor<i32>, &IntTensor<i32>) -> IntTensor<i32>,
        a: &[i32],
        b: &[i32],
        expected: &[i32],
    ) {
        ensure_cuda_backend();
        let ag = t32(a).to(Device::Cuda(0)).unwrap();
        let bg = t32(b).to(Device::Cuda(0)).unwrap();
        let r = op(&ag, &bg);
        assert!(r.is_cuda(), "result must stay CUDA-resident");
        assert_eq!(r.to(Device::Cpu).unwrap().data().unwrap(), expected);
        // CPU path must agree bit-for-bit at the boundary.
        assert_eq!(op(&t32(a), &t32(b)).data().unwrap(), expected);
    }

    fn check_i64(
        op: impl Fn(&IntTensor<i64>, &IntTensor<i64>) -> IntTensor<i64>,
        a: &[i64],
        b: &[i64],
        expected: &[i64],
    ) {
        ensure_cuda_backend();
        let ag = t64(a).to(Device::Cuda(0)).unwrap();
        let bg = t64(b).to(Device::Cuda(0)).unwrap();
        let r = op(&ag, &bg);
        assert!(r.is_cuda(), "result must stay CUDA-resident");
        assert_eq!(r.to(Device::Cpu).unwrap().data().unwrap(), expected);
        assert_eq!(op(&t64(a), &t64(b)).data().unwrap(), expected);
    }

    #[test]
    fn gpu_add_i32_wraps_and_matches_cpu() {
        // oracle: torch cuda i32 MAX+1 -> -2147483648
        check_i32(|a, b| a.add(b).unwrap(), &[i32::MAX], &[1], &[i32::MIN]);
    }

    #[test]
    fn gpu_sub_i32_wraps_and_matches_cpu() {
        check_i32(|a, b| a.sub(b).unwrap(), &[i32::MIN], &[1], &[i32::MAX]);
    }

    #[test]
    fn gpu_mul_i32_wraps_and_matches_cpu() {
        check_i32(|a, b| a.mul(b).unwrap(), &[i32::MAX], &[2], &[-2]);
    }

    #[test]
    fn gpu_add_i64_wraps_and_matches_cpu() {
        check_i64(|a, b| a.add(b).unwrap(), &[i64::MAX], &[1], &[i64::MIN]);
    }

    #[test]
    fn gpu_sub_i64_wraps_and_matches_cpu() {
        check_i64(|a, b| a.sub(b).unwrap(), &[i64::MIN], &[1], &[i64::MAX]);
    }

    #[test]
    fn gpu_sum_i32_wraps_and_matches_cpu() {
        ensure_cuda_backend();
        let g = t32(&[i32::MAX, 1, 5]).to(Device::Cuda(0)).unwrap();
        let r = g.sum().unwrap();
        assert!(r.is_cuda(), "sum result must stay CUDA-resident");
        assert_eq!(r.to(Device::Cpu).unwrap().data().unwrap(), &[-2147483643]);
        assert_eq!(
            t32(&[i32::MAX, 1, 5]).sum().unwrap().data().unwrap(),
            &[-2147483643]
        );
    }

    #[test]
    fn gpu_sum_i64_wraps_and_matches_cpu() {
        ensure_cuda_backend();
        let g = t64(&[i64::MAX, 1]).to(Device::Cuda(0)).unwrap();
        let r = g.sum().unwrap();
        assert!(r.is_cuda(), "sum result must stay CUDA-resident");
        assert_eq!(r.to(Device::Cpu).unwrap().data().unwrap(), &[i64::MIN]);
        assert_eq!(
            t64(&[i64::MAX, 1]).sum().unwrap().data().unwrap(),
            &[i64::MIN]
        );
    }

    #[test]
    fn gpu_prod_i32_wraps_and_matches_cpu() {
        ensure_cuda_backend();
        let g = t32(&[i32::MAX, 2]).to(Device::Cuda(0)).unwrap();
        let r = g.prod().unwrap();
        assert!(r.is_cuda(), "prod result must stay CUDA-resident");
        assert_eq!(r.to(Device::Cpu).unwrap().data().unwrap(), &[-2]);
    }
}
