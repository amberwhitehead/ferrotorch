//! CORE-111 (#1805, CLASS-U) regression suite: `index_select` / `gather`
//! must validate every integer index VALUE against the selected axis before
//! any CPU compute or GPU kernel launch.
//!
//! Pre-fix behavior observed at HEAD (the red run for this suite):
//! - CPU: `sel as usize` on a negative/`>= axis_len` index panicked inside
//!   the fallible API (`index out of bounds: the len is 12 but the index is
//!   18446744073709551612` at ops/phase2c.rs), or silently returned a value
//!   from the WRONG row when the unchecked address stayed inside the buffer.
//! - CUDA (RTX 3090): small OOB returned `Ok` with garbage readback
//!   (`[0.0, 0.0, 0.0, 0.0]`); a negative index produced
//!   `CUDA_ERROR_ILLEGAL_ADDRESS` at the next sync AND poisoned the context
//!   for every subsequent CUDA op in the process.
//!
//! Oracle (PyTorch 2.11.0+cu130, live session — negative indices are NOT
//! wrapped by either op):
//! ```text
//! >>> torch.index_select(x, 0, torch.tensor([-1]))
//! IndexError: index out of range in self
//! >>> torch.index_select(x, 0, torch.tensor([3]))      # x: [3,4]
//! IndexError: index out of range in self
//! >>> torch.gather(x, 1, torch.tensor([[-1],[0],[0]]))
//! RuntimeError: index -1 is out of bounds for dimension 1 with size 4
//! >>> torch.gather(x, 1, torch.tensor([[4],[0],[0]]))
//! RuntimeError: index 4 is out of bounds for dimension 1 with size 4
//! ```
//! Upstream contract:
//! - aten/src/ATen/native/TensorAdvancedIndexing.cpp:1704-1706
//!   `TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size),
//!    "index out of range in self");`
//! - aten/src/ATen/native/cpu/ScatterGatherKernel.cpp:116-120
//!   `TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound, "index ", ...,
//!    " is out of bounds for dimension ", dim, " with size ", index_upper_bound);`

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::tensor::Tensor;

/// `[3,4]` float tensor `0..12` (the oracle session's `x`).
fn x_f32() -> Tensor<f32> {
    from_vec::<f32>((0..12).map(|v| v as f32).collect(), &[3, 4]).unwrap()
}

/// `[3,4]` i32 tensor `0..12` (the oracle session's `xi`).
fn x_i32() -> IntTensor<i32> {
    IntTensor::<i32>::from_vec((0..12).collect(), vec![3, 4]).unwrap()
}

fn idx1(vals: Vec<i64>) -> IntTensor<i64> {
    let n = vals.len();
    IntTensor::<i64>::from_vec(vals, vec![n]).unwrap()
}

/// Column-shaped `[3,1]` gather index along dim=1 of a `[3,4]` input.
fn gidx_col(vals: [i64; 3]) -> IntTensor<i64> {
    IntTensor::<i64>::from_vec(vals.to_vec(), vec![3, 1]).unwrap()
}

/// Row-shaped `[1,4]` gather index along dim=0 of a `[3,4]` input.
fn gidx_row(vals: [i64; 4]) -> IntTensor<i64> {
    IntTensor::<i64>::from_vec(vals.to_vec(), vec![1, 4]).unwrap()
}

/// The structured-error contract: `InvalidArgument` carrying the offending
/// index value, its flat position, the dimension, and the axis size
/// (mirroring torch's "index {v} is out of bounds for dimension {d} with
/// size {n}" — R-ORACLE-4: exactly one accepted outcome).
fn expect_oob(res: Result<impl std::fmt::Debug, FerrotorchError>, parts: &[&str]) {
    match res {
        Err(FerrotorchError::InvalidArgument { message }) => {
            for p in parts {
                assert!(
                    message.contains(p),
                    "expected {p:?} in error message {message:?}"
                );
            }
        }
        Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        Ok(v) => panic!("expected InvalidArgument, got Ok({v:?})"),
    }
}

// ── CPU lane: Tensor::index_select ──────────────────────────────────────────

#[test]
fn cpu_tensor_index_select_rejects_negative() {
    expect_oob(
        x_f32().index_select(0, &idx1(vec![-1])),
        &[
            "index_select",
            "index -1 is out of bounds for dimension 0 with size 3",
            "position 0",
        ],
    );
}

#[test]
fn cpu_tensor_index_select_rejects_eq_axis_len() {
    expect_oob(
        x_f32().index_select(0, &idx1(vec![0, 3])),
        &[
            "index 3 is out of bounds for dimension 0 with size 3",
            "position 1",
        ],
    );
}

#[test]
fn cpu_tensor_index_select_rejects_large_positive() {
    expect_oob(
        x_f32().index_select(1, &idx1(vec![9999])),
        &["index 9999 is out of bounds for dimension 1 with size 4"],
    );
}

// ── CPU lane: Tensor::gather ────────────────────────────────────────────────

#[test]
fn cpu_tensor_gather_rejects_negative() {
    expect_oob(
        x_f32().gather(1, &gidx_col([-1, 0, 0])),
        &[
            "gather",
            "index -1 is out of bounds for dimension 1 with size 4",
            "position 0",
        ],
    );
}

#[test]
fn cpu_tensor_gather_rejects_eq_axis_len() {
    // Pre-fix this case did NOT panic: the unchecked address landed inside
    // the buffer and silently returned x[1,0] for x[0,4] (probe output:
    // `Ok(Tensor { shape: [3,4] ... })`). Torch raises (oracle above).
    expect_oob(
        x_f32().gather(1, &gidx_col([4, 0, 0])),
        &[
            "index 4 is out of bounds for dimension 1 with size 4",
            "position 0",
        ],
    );
}

#[test]
fn cpu_tensor_gather_rejects_large_positive() {
    expect_oob(
        x_f32().gather(0, &gidx_row([0, 999_999, 0, 0])),
        &[
            "index 999999 is out of bounds for dimension 0 with size 3",
            "position 1",
        ],
    );
}

// ── CPU lane: IntTensor::index_select / IntTensor::gather ──────────────────

#[test]
fn cpu_inttensor_index_select_rejects_negative() {
    expect_oob(
        x_i32().index_select(0, &idx1(vec![-1])),
        &["index -1 is out of bounds for dimension 0 with size 3"],
    );
}

#[test]
fn cpu_inttensor_index_select_rejects_eq_axis_len() {
    expect_oob(
        x_i32().index_select(0, &idx1(vec![3])),
        &["index 3 is out of bounds for dimension 0 with size 3"],
    );
}

#[test]
fn cpu_inttensor_index_select_rejects_large_positive() {
    expect_oob(
        x_i32().index_select(1, &idx1(vec![12345])),
        &["index 12345 is out of bounds for dimension 1 with size 4"],
    );
}

#[test]
fn cpu_inttensor_gather_rejects_negative() {
    expect_oob(
        x_i32().gather(1, &gidx_col([0, -1, 0])),
        &[
            "index -1 is out of bounds for dimension 1 with size 4",
            "position 1",
        ],
    );
}

#[test]
fn cpu_inttensor_gather_rejects_eq_axis_len() {
    expect_oob(
        x_i32().gather(1, &gidx_col([4, 0, 0])),
        &["index 4 is out of bounds for dimension 1 with size 4"],
    );
}

#[test]
fn cpu_inttensor_gather_rejects_large_positive() {
    expect_oob(
        x_i32().gather(0, &gidx_row([0, 0, 0, 777_777])),
        &[
            "index 777777 is out of bounds for dimension 0 with size 3",
            "position 3",
        ],
    );
}

// ── CPU lane: valid boundary (axis_len - 1) still computes torch's values ──

#[test]
fn cpu_boundary_values_match_torch() {
    // torch 2.11 live session (quoted in the module doc-comment context):
    //   torch.index_select(x, 0, torch.tensor([2])) -> [[8., 9., 10., 11.]]
    //   torch.index_select(x, 1, torch.tensor([3, 0]))
    //       -> [[3., 0.], [7., 4.], [11., 8.]]
    //   torch.gather(x, 1, torch.tensor([[3],[3],[3]])) -> [[3.], [7.], [11.]]
    //   torch.gather(x, 0, torch.tensor([[2,2,2,2]])) -> [[8., 9., 10., 11.]]
    let x = x_f32();
    let sel0 = x.index_select(0, &idx1(vec![2])).unwrap();
    assert_eq!(sel0.shape(), [1, 4]);
    assert_eq!(
        sel0.data_vec().unwrap(),
        vec![8.0f32, 9.0, 10.0, 11.0],
        "index_select dim=0 boundary"
    );
    let sel1 = x.index_select(1, &idx1(vec![3, 0])).unwrap();
    assert_eq!(sel1.shape(), [3, 2]);
    assert_eq!(
        sel1.data_vec().unwrap(),
        vec![3.0f32, 0.0, 7.0, 4.0, 11.0, 8.0],
        "index_select dim=1 boundary"
    );
    let g1 = x.gather(1, &gidx_col([3, 3, 3])).unwrap();
    assert_eq!(g1.shape(), [3, 1]);
    assert_eq!(
        g1.data_vec().unwrap(),
        vec![3.0f32, 7.0, 11.0],
        "gather dim=1 boundary"
    );
    let g0 = x.gather(0, &gidx_row([2, 2, 2, 2])).unwrap();
    assert_eq!(g0.shape(), [1, 4]);
    assert_eq!(
        g0.data_vec().unwrap(),
        vec![8.0f32, 9.0, 10.0, 11.0],
        "gather dim=0 boundary"
    );
}

#[test]
fn cpu_inttensor_boundary_values_match_torch() {
    // torch 2.11 live session:
    //   torch.index_select(xi, 0, torch.tensor([2])) -> [[8, 9, 10, 11]]
    //   torch.index_select(xi, 1, torch.tensor([3, 0]))
    //       -> [[3, 0], [7, 4], [11, 8]]
    //   torch.gather(xi, 1, torch.tensor([[3],[3],[3]])) -> [[3], [7], [11]]
    //   torch.gather(xi, 0, torch.tensor([[2,2,2,2]])) -> [[8, 9, 10, 11]]
    let xi = x_i32();
    let sel0 = xi.index_select(0, &idx1(vec![2])).unwrap();
    assert_eq!(sel0.shape(), [1, 4]);
    assert_eq!(sel0.data().unwrap(), [8i32, 9, 10, 11]);
    let sel1 = xi.index_select(1, &idx1(vec![3, 0])).unwrap();
    assert_eq!(sel1.shape(), [3, 2]);
    assert_eq!(sel1.data().unwrap(), [3i32, 0, 7, 4, 11, 8]);
    let g1 = xi.gather(1, &gidx_col([3, 3, 3])).unwrap();
    assert_eq!(g1.shape(), [3, 1]);
    assert_eq!(g1.data().unwrap(), [3i32, 7, 11]);
    let g0 = xi.gather(0, &gidx_row([2, 2, 2, 2])).unwrap();
    assert_eq!(g0.shape(), [1, 4]);
    assert_eq!(g0.data().unwrap(), [8i32, 9, 10, 11]);
}

// ── CUDA lane ───────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::device::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();
    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend().expect("CUDA backend init for CORE-111 suite");
        });
    }

    fn x_f32_gpu() -> Tensor<f32> {
        ensure_cuda_backend();
        x_f32().to(Device::Cuda(0)).unwrap()
    }
    fn x_i32_gpu() -> IntTensor<i32> {
        ensure_cuda_backend();
        x_i32().to(Device::Cuda(0)).unwrap()
    }
    fn gpu_idx(t: IntTensor<i64>) -> IntTensor<i64> {
        t.to(Device::Cuda(0)).unwrap()
    }

    // Invalid indices must be rejected BEFORE any kernel launch — pre-fix the
    // negative-index case raised CUDA_ERROR_ILLEGAL_ADDRESS and poisoned the
    // context for the rest of the process (probe output in module docs).

    #[test]
    fn gpu_tensor_index_select_rejects_negative() {
        expect_oob(
            x_f32_gpu().index_select(0, &gpu_idx(idx1(vec![-1]))),
            &["index -1 is out of bounds for dimension 0 with size 3"],
        );
    }

    #[test]
    fn gpu_tensor_index_select_rejects_eq_axis_len() {
        // Pre-fix: Ok with garbage readback [0.0, 0.0, 0.0, 0.0] (3090 probe).
        expect_oob(
            x_f32_gpu().index_select(0, &gpu_idx(idx1(vec![3]))),
            &["index 3 is out of bounds for dimension 0 with size 3"],
        );
    }

    #[test]
    fn gpu_tensor_index_select_rejects_large_positive() {
        expect_oob(
            x_f32_gpu().index_select(1, &gpu_idx(idx1(vec![9999]))),
            &["index 9999 is out of bounds for dimension 1 with size 4"],
        );
    }

    #[test]
    fn gpu_tensor_gather_rejects_negative() {
        expect_oob(
            x_f32_gpu().gather(1, &gpu_idx(gidx_col([-1, 0, 0]))),
            &["index -1 is out of bounds for dimension 1 with size 4"],
        );
    }

    #[test]
    fn gpu_tensor_gather_rejects_eq_axis_len() {
        expect_oob(
            x_f32_gpu().gather(1, &gpu_idx(gidx_col([4, 0, 0]))),
            &["index 4 is out of bounds for dimension 1 with size 4"],
        );
    }

    #[test]
    fn gpu_tensor_gather_rejects_large_positive() {
        // Pre-fix: Ok with garbage readback (silent OOB device read, 3090).
        expect_oob(
            x_f32_gpu().gather(0, &gpu_idx(gidx_row([999_999, 0, 0, 0]))),
            &["index 999999 is out of bounds for dimension 0 with size 3"],
        );
    }

    #[test]
    fn gpu_inttensor_index_select_rejects_negative() {
        expect_oob(
            x_i32_gpu().index_select(0, &gpu_idx(idx1(vec![-1]))),
            &["index -1 is out of bounds for dimension 0 with size 3"],
        );
    }

    #[test]
    fn gpu_inttensor_index_select_rejects_eq_axis_len() {
        expect_oob(
            x_i32_gpu().index_select(0, &gpu_idx(idx1(vec![3]))),
            &["index 3 is out of bounds for dimension 0 with size 3"],
        );
    }

    #[test]
    fn gpu_inttensor_index_select_rejects_large_positive() {
        expect_oob(
            x_i32_gpu().index_select(1, &gpu_idx(idx1(vec![12345]))),
            &["index 12345 is out of bounds for dimension 1 with size 4"],
        );
    }

    #[test]
    fn gpu_inttensor_gather_rejects_negative() {
        expect_oob(
            x_i32_gpu().gather(1, &gpu_idx(gidx_col([0, -1, 0]))),
            &["index -1 is out of bounds for dimension 1 with size 4"],
        );
    }

    #[test]
    fn gpu_inttensor_gather_rejects_eq_axis_len() {
        expect_oob(
            x_i32_gpu().gather(1, &gpu_idx(gidx_col([4, 0, 0]))),
            &["index 4 is out of bounds for dimension 1 with size 4"],
        );
    }

    #[test]
    fn gpu_inttensor_gather_rejects_large_positive() {
        expect_oob(
            x_i32_gpu().gather(0, &gpu_idx(gidx_row([0, 0, 777_777, 0]))),
            &["index 777777 is out of bounds for dimension 0 with size 3"],
        );
    }

    // Valid boundary (axis_len - 1): values still match the torch oracle and
    // the result stays CUDA-resident (R-ORACLE-3 device assertion).

    #[test]
    fn gpu_boundary_values_match_torch_and_stay_resident() {
        let x = x_f32_gpu();
        let sel0 = x.index_select(0, &gpu_idx(idx1(vec![2]))).unwrap();
        assert!(
            sel0.is_cuda(),
            "index_select result must stay CUDA-resident"
        );
        assert_eq!(sel0.shape(), [1, 4]);
        assert_eq!(
            sel0.to(Device::Cpu).unwrap().data_vec().unwrap(),
            vec![8.0f32, 9.0, 10.0, 11.0],
            "torch.index_select(x, 0, [2]) -> [[8., 9., 10., 11.]]"
        );

        let g1 = x.gather(1, &gpu_idx(gidx_col([3, 3, 3]))).unwrap();
        assert!(g1.is_cuda(), "gather result must stay CUDA-resident");
        assert_eq!(g1.shape(), [3, 1]);
        assert_eq!(
            g1.to(Device::Cpu).unwrap().data_vec().unwrap(),
            vec![3.0f32, 7.0, 11.0],
            "torch.gather(x, 1, [[3],[3],[3]]) -> [[3.], [7.], [11.]]"
        );

        let xi = x_i32_gpu();
        let isel = xi.index_select(0, &gpu_idx(idx1(vec![2]))).unwrap();
        assert!(isel.is_cuda(), "IntTensor index_select must stay resident");
        assert_eq!(
            isel.to(Device::Cpu).unwrap().data().unwrap(),
            [8i32, 9, 10, 11],
            "torch.index_select(xi, 0, [2]) -> [[8, 9, 10, 11]]"
        );
        let ig = xi.gather(0, &gpu_idx(gidx_row([2, 2, 2, 2]))).unwrap();
        assert!(ig.is_cuda(), "IntTensor gather must stay resident");
        assert_eq!(
            ig.to(Device::Cpu).unwrap().data().unwrap(),
            [8i32, 9, 10, 11],
            "torch.gather(xi, 0, [[2,2,2,2]]) -> [[8, 9, 10, 11]]"
        );
    }
}
