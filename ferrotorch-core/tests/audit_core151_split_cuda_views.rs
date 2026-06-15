//! Regression tests for CORE-151 (#1845) — CLASS-V, High.
//!
//! `split_t` / `chunk_t` must be PyTorch-style view operations:
//! metadata-only, storage-sharing, dtype-generic, and resident on the
//! input device. The old CUDA f32 fast path passed only a raw base
//! buffer plus logical extents to `strided_split_f32`, so non-zero
//! storage offsets and non-contiguous strides were dropped. These
//! probes pin the corrected behavior: the split/chunk outputs preserve
//! view geometry and only the explicit `.cpu()` readback materializes
//! the logical values.
//!
//! Oracle (R-ORACLE-1(b)): live torch 2.11.0+cu130 on cuda:0
//! (RTX 3090), 2026-06-11:
//!
//! ```python
//! x = torch.arange(24., device='cuda:0', dtype=torch.float32).reshape(4, 6)
//! v = x.narrow(0, 1, 2)              # (2,6), storage_offset 6, contiguous
//! parts = v.split([3, 3], dim=1)     # 2 x (2,3)
//! # parts[0].flatten() -> [6., 7., 8., 12., 13., 14.]
//! # parts[1].flatten() -> [9., 10., 11., 15., 16., 17.]
//! w = x.t()                          # (6,4), non-contiguous
//! ch = w.chunk(2, 0)                 # 2 x (3,4)
//! # ch[0].flatten() -> [0., 6., 12., 18., 1., 7., 13., 19., 2., 8., 14., 20.]
//! # ch[1].flatten() -> [3., 9., 15., 21., 4., 10., 16., 22., 5., 11., 17., 23.]
//! ```

#![cfg(feature = "gpu")]

use std::sync::{Arc, Once};

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::{Device, Tensor};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the CORE-151 regression suite");
    });
}

/// `arange(24).reshape(4, 6)` uploaded to cuda:0.
fn arange_4x6_cuda() -> Tensor<f32> {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    from_vec(data, &[4, 6])
        .expect("construct cpu tensor")
        .to(Device::Cuda(0))
        .expect("cpu->gpu upload")
}

fn cuda(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    from_vec(data.to_vec(), shape)
        .expect("construct cpu tensor")
        .to(Device::Cuda(0))
        .expect("cpu->gpu upload")
}

/// `split` of a `narrow` view (contiguous strides, storage_offset = 6).
/// Pre-fix the fast path reads from storage element 0 and returns the
/// first two ROWS' columns (`[0,1,2,6,7,8]` / `[3,4,5,9,10,11]`)
/// instead of rows 1..3.
#[test]
// reason: split is pure data movement (no arithmetic) — every element
// round-trips bit-exactly, so float equality is the right check.
#[allow(clippy::float_cmp)]
fn split_of_cuda_narrow_view_matches_torch() {
    ensure_cuda_backend();
    let x = arange_4x6_cuda();
    let v = x.narrow(0, 1, 2).expect("narrow view");
    assert_eq!(v.storage_offset(), 6, "narrow view must carry its offset");
    assert!(v.is_contiguous(), "row-narrow keeps contiguous strides");

    let parts = v.split(&[3, 3], 1).expect("split along dim 1");
    assert_eq!(parts.len(), 2);

    // torch: parts[0] -> [6,7,8,12,13,14]; parts[1] -> [9,10,11,15,16,17]
    let want: [&[f32]; 2] = [
        &[6.0, 7.0, 8.0, 12.0, 13.0, 14.0],
        &[9.0, 10.0, 11.0, 15.0, 16.0, 17.0],
    ];
    for (i, (part, want)) in parts.iter().zip(want).enumerate() {
        assert!(
            part.is_cuda(),
            "split chunk {i} of a CUDA view must stay on CUDA"
        );
        assert!(
            Arc::ptr_eq(v.inner_storage_arc(), part.inner_storage_arc()),
            "split chunk {i} must share storage with the input view"
        );
        assert_eq!(part.shape(), &[2, 3], "chunk {i} shape");
        let host = part.cpu().expect("gpu->cpu readback");
        assert_eq!(
            host.data().expect("host slice"),
            want,
            "split chunk {i} of a CUDA narrow view gathered wrong elements (CORE-151)"
        );
    }
}

/// `chunk` of a transpose view (non-contiguous strides, offset 0).
/// Pre-fix the fast path treats the base buffer as if it were laid out
/// in the view's logical order and returns the first half of the
/// ROW-MAJOR base buffer (`[0..12]`) instead of the transposed rows.
#[test]
// reason: chunk is pure data movement (no arithmetic) — every element
// round-trips bit-exactly, so float equality is the right check.
#[allow(clippy::float_cmp)]
fn chunk_of_cuda_transpose_view_matches_torch() {
    ensure_cuda_backend();
    let x = arange_4x6_cuda();
    let w = x.transpose(0, 1).expect("transpose view"); // (6,4)
    assert!(!w.is_contiguous(), "transpose view must be non-contiguous");

    let chunks = w.chunk(2, 0).expect("chunk along dim 0");
    assert_eq!(chunks.len(), 2);

    // torch: ch[0] -> [0,6,12,18,1,7,13,19,2,8,14,20]
    //        ch[1] -> [3,9,15,21,4,10,16,22,5,11,17,23]
    let want: [&[f32]; 2] = [
        &[
            0.0, 6.0, 12.0, 18.0, 1.0, 7.0, 13.0, 19.0, 2.0, 8.0, 14.0, 20.0,
        ],
        &[
            3.0, 9.0, 15.0, 21.0, 4.0, 10.0, 16.0, 22.0, 5.0, 11.0, 17.0, 23.0,
        ],
    ];
    for (i, (chunk, want)) in chunks.iter().zip(want).enumerate() {
        assert!(
            chunk.is_cuda(),
            "chunk {i} of a CUDA view must stay on CUDA"
        );
        assert!(
            Arc::ptr_eq(w.inner_storage_arc(), chunk.inner_storage_arc()),
            "chunk {i} must share storage with the input view"
        );
        assert_eq!(chunk.shape(), &[3, 4], "chunk {i} shape");
        let host = chunk.cpu().expect("gpu->cpu readback");
        assert_eq!(
            host.data().expect("host slice"),
            want,
            "chunk {i} of a CUDA transpose view gathered wrong elements (CORE-151)"
        );
    }
}

/// `SplitBackward` fed a non-contiguous CUDA `grad_output` must copy the
/// gradient's logical order into the original tensor, not raw storage order.
/// Torch oracle:
/// ```python
/// x = torch.arange(8., device='cuda:0').reshape(4,2).requires_grad_()
/// y = x.split([2,2], 0)[1]
/// y.backward(torch.arange(4., device='cuda:0').reshape(2,2).t())
/// x.grad.flatten()  # [0,0,0,0,0,2,1,3]
/// ```
#[test]
// reason: split backward is pure data movement, so exact float comparison is appropriate.
#[allow(clippy::float_cmp)]
fn split_cuda_backward_with_noncontiguous_grad_output_matches_torch() {
    ensure_cuda_backend();
    let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x = cuda(&data, &[4, 2]).requires_grad_(true);
    let parts = x.split(&[2, 2], 0).expect("split along dim 0");
    let go_base = cuda(&[0.0, 1.0, 2.0, 3.0], &[2, 2]);
    let go_t = go_base.transpose(0, 1).expect("transposed grad_output");
    assert!(!go_t.is_contiguous());

    parts[1]
        .backward_with_gradient(&go_t)
        .expect("backward with non-contiguous CUDA grad_output");

    let grad = x.grad().unwrap().expect("grad must reach CUDA leaf");
    assert_eq!(
        grad.device(),
        Device::Cuda(0),
        "split backward grad must remain CUDA-resident"
    );
    assert_eq!(grad.shape(), &[4, 2]);
    assert_eq!(
        grad.data_vec().expect("gpu readback"),
        &[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 3.0]
    );
}
