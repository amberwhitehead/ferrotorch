//! Audit CORE-102 (crosslink #1796): CPU integer division / remainder by
//! zero must be a structured error, not a fabricated `0`.
//!
//! PyTorch's per-device contract, probed live (torch 2.11.0+cu130, RTX 3090),
//! R-ORACLE-1 path (b):
//!
//! ```text
//! >>> torch.floor_divide(torch.tensor([7], dtype=torch.int32),
//! ...                    torch.tensor([0], dtype=torch.int32))
//! RuntimeError: ZeroDivisionError
//! >>> torch.remainder(torch.tensor([7], dtype=torch.int32),
//! ...                 torch.tensor([0], dtype=torch.int32))
//! RuntimeError: ZeroDivisionError
//! >>> torch.floor_divide(torch.tensor([7, 8], dtype=torch.int32),
//! ...                    torch.tensor([2, 0], dtype=torch.int32))
//! RuntimeError: ZeroDivisionError        # ANY zero divisor errors on CPU
//! >>> torch.floor_divide(torch.tensor([7,-7,0,2147483647], dtype=torch.int32,
//! ...                    device='cuda'), torch.tensor([0,0,0,0], ...)).cpu()
//! tensor([-1, -2, -1, -1], dtype=torch.int32)   # CUDA: no trap, UNSPECIFIED
//! >>> torch.remainder(<same cuda operands>).cpu()
//! tensor([-1, -1, -1, -1], dtype=torch.int32)
//! ```
//!
//! So the pinned contract is per-device (R-ORACLE-4 — exactly one contract
//! per lane):
//! - **CPU**: any zero divisor element → structured `Err` carrying
//!   `ZeroDivisionError` (torch CPU parity). At HEAD this returned
//!   `Ok([0, ...])` — fabricated data.
//! - **CUDA**: no trap; the result stays CUDA-resident; zero-divisor lanes
//!   are unspecified (torch CUDA gives garbage like `-1`/`4294967295`, our
//!   PTX kernel gives `-2`/`-1`; neither value is part of any contract and
//!   the zero lanes are deliberately NOT asserted); nonzero-divisor lanes
//!   are exact.

use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::int_tensor::IntTensor;

fn t32(v: &[i32]) -> IntTensor<i32> {
    IntTensor::from_vec(v.to_vec(), vec![v.len()]).unwrap()
}

fn t64(v: &[i64]) -> IntTensor<i64> {
    IntTensor::from_vec(v.to_vec(), vec![v.len()]).unwrap()
}

fn assert_zero_division_err(r: Result<IntTensor<i32>, FerrotorchError>) {
    match r {
        Err(FerrotorchError::InvalidArgument { message }) => {
            assert!(
                message.contains("ZeroDivisionError"),
                "error message must carry ZeroDivisionError (torch CPU parity), got: {message}"
            );
        }
        Err(other) => panic!("expected InvalidArgument(ZeroDivisionError), got {other:?}"),
        Ok(t) => panic!(
            "expected Err on zero divisor, got fabricated Ok({:?})",
            t.data().unwrap()
        ),
    }
}

fn assert_zero_division_err64(r: Result<IntTensor<i64>, FerrotorchError>) {
    match r {
        Err(FerrotorchError::InvalidArgument { message }) => {
            assert!(
                message.contains("ZeroDivisionError"),
                "error message must carry ZeroDivisionError (torch CPU parity), got: {message}"
            );
        }
        Err(other) => panic!("expected InvalidArgument(ZeroDivisionError), got {other:?}"),
        Ok(t) => panic!(
            "expected Err on zero divisor, got fabricated Ok({:?})",
            t.data().unwrap()
        ),
    }
}

// ── CPU: zero divisor is a structured error ────────────────────────────────

#[test]
fn cpu_floor_div_i32_by_zero_errors() {
    assert_zero_division_err(t32(&[7]).floor_div(&t32(&[0])));
}

#[test]
fn cpu_remainder_i32_by_zero_errors() {
    assert_zero_division_err(t32(&[7]).remainder(&t32(&[0])));
}

#[test]
fn cpu_floor_div_i32_mixed_divisors_errors() {
    // torch CPU errors when ANY divisor element is zero.
    assert_zero_division_err(t32(&[7, 8]).floor_div(&t32(&[2, 0])));
}

#[test]
fn cpu_remainder_i32_mixed_divisors_errors() {
    assert_zero_division_err(t32(&[7, 8]).remainder(&t32(&[2, 0])));
}

#[test]
fn cpu_floor_div_i64_by_zero_errors() {
    assert_zero_division_err64(t64(&[7, -7]).floor_div(&t64(&[0, 0])));
}

#[test]
fn cpu_remainder_i64_by_zero_errors() {
    assert_zero_division_err64(t64(&[7, -7]).remainder(&t64(&[0, 0])));
}

#[test]
fn cpu_zero_dividend_zero_divisor_still_errors() {
    // 0 // 0 is just as invalid as 7 // 0 (torch: ZeroDivisionError).
    assert_zero_division_err(t32(&[0]).floor_div(&t32(&[0])));
}

// ── CPU: nonzero divisors keep working (regression guard) ──────────────────

#[test]
fn cpu_floor_div_nonzero_divisors_unaffected() {
    // torch.floor_divide([-7, 7, -7, 7], [2, 2, -2, -2]) -> [-4, 3, 3, -4]
    let r = t32(&[-7, 7, -7, 7])
        .floor_div(&t32(&[2, 2, -2, -2]))
        .unwrap();
    assert_eq!(r.data().unwrap(), &[-4, 3, 3, -4]);
}

#[test]
fn cpu_remainder_nonzero_divisors_unaffected() {
    // torch.remainder([-7, 7, 7], [2, 2, -2]) -> [1, 1, -1]
    let r = t32(&[-7, 7, 7]).remainder(&t32(&[2, 2, -2])).unwrap();
    assert_eq!(r.data().unwrap(), &[1, 1, -1]);
}

// ── CUDA: torch CUDA parity — no trap, resident result, exact nonzero lanes ─

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::device::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialise for CORE-102 gpu lane");
        });
    }

    #[test]
    fn cuda_floor_div_zero_divisor_no_trap_resident_nonzero_lanes_exact() {
        ensure_cuda_backend();
        let a = t32(&[7, -7, 0, 9]).to(Device::Cuda(0)).unwrap();
        let b = t32(&[2, 0, 3, 0]).to(Device::Cuda(0)).unwrap();
        // torch CUDA contract: integer floor_divide with zero divisors does
        // not error and does not trap the device.
        let r = a
            .floor_div(&b)
            .expect("CUDA floor_div must not error on zero divisors");
        assert!(r.is_cuda(), "result must stay CUDA-resident");
        let host = r.to(Device::Cpu).unwrap();
        let vals = host.data().unwrap();
        // Nonzero lanes are exact: 7//2=3, 0//3=0. Zero-divisor lanes
        // (indices 1, 3) are UNSPECIFIED per the torch CUDA contract and
        // are deliberately not asserted.
        assert_eq!(vals[0], 3);
        assert_eq!(vals[2], 0);
    }

    #[test]
    fn cuda_remainder_zero_divisor_no_trap_resident_nonzero_lanes_exact() {
        ensure_cuda_backend();
        let a = t32(&[7, -7, -7, 9]).to(Device::Cuda(0)).unwrap();
        let b = t32(&[2, 0, 2, 0]).to(Device::Cuda(0)).unwrap();
        let r = a
            .remainder(&b)
            .expect("CUDA remainder must not error on zero divisors");
        assert!(r.is_cuda(), "result must stay CUDA-resident");
        let host = r.to(Device::Cpu).unwrap();
        let vals = host.data().unwrap();
        // torch.remainder(7, 2) = 1; torch.remainder(-7, 2) = 1.
        assert_eq!(vals[0], 1);
        assert_eq!(vals[2], 1);
    }

    #[test]
    fn cuda_i64_div_rem_zero_divisor_no_trap() {
        ensure_cuda_backend();
        let a = t64(&[7, -7]).to(Device::Cuda(0)).unwrap();
        let z = t64(&[0, 0]).to(Device::Cuda(0)).unwrap();
        let d = a.floor_div(&z).expect("CUDA i64 floor_div must not error");
        let m = a.remainder(&z).expect("CUDA i64 remainder must not error");
        assert!(
            d.is_cuda() && m.is_cuda(),
            "results must stay CUDA-resident"
        );
        // Values unspecified (torch CUDA contract); residency + no-trap is
        // the pinned contract. Reading back must still succeed.
        let _ = d.to(Device::Cpu).unwrap();
        let _ = m.to(Device::Cpu).unwrap();
    }
}
