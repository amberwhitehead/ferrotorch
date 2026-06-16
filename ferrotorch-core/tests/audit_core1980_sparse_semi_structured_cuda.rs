//! Regression coverage for #1980: PyTorch-style semi-structured sparse packing.
//!
//! Live PyTorch oracle used for the CUTLASS f32 metadata expectation:
//!
//! ```python
//! import torch
//! from torch.sparse import SparseSemiStructuredTensor
//! from torch.sparse._semi_structured_conversions import sparse_semi_structured_from_dense_cutlass
//! SparseSemiStructuredTensor._FORCE_CUTLASS = True
//! x = torch.zeros((32, 32), device="cuda", dtype=torch.float32)
//! for r in range(32):
//!     for c in range(0, 32, 2):
//!         x[r, c] = r * 100 + c + 1
//! values, meta = sparse_semi_structured_from_dense_cutlass(x)
//! values[0].cpu().tolist()
//! # [1.0, 3.0, 5.0, ..., 31.0]
//! torch.unique(meta.cpu())
//! # tensor([17476], dtype=torch.int16)
//! ```

use ferrotorch_core::{FerrotorchError, from_vec, to_sparse_semi_structured_cutlass};

#[test]
fn to_sparse_semi_structured_rejects_cpu_input_at_public_boundary() {
    let dense = from_vec::<f32>(vec![0.0; 32 * 32], &[32, 32]).expect("cpu dense");
    let err = to_sparse_semi_structured_cutlass(&dense).expect_err("CPU input must reject");
    match err {
        FerrotorchError::InvalidArgument { message } => {
            assert!(
                message.contains("only CUDA tensors are supported"),
                "unexpected error message: {message}"
            );
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[cfg(feature = "gpu")]
mod cuda {
    use std::sync::Once;

    use super::*;
    use ferrotorch_core::{Device, SemiStructuredSparseBackend};

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for sparse semi-structured probe");
        });
    }

    fn oracle_input() -> Vec<f32> {
        let mut data = vec![0.0_f32; 32 * 32];
        for r in 0..32 {
            for c in (0..32).step_by(2) {
                data[r * 32 + c] = (r * 100 + c + 1) as f32;
            }
        }
        data
    }

    #[test]
    fn cutlass_f32_pack_matches_torch_values_and_metadata_on_cuda() {
        ensure_cuda_backend();
        let dense_cpu = from_vec::<f32>(oracle_input(), &[32, 32]).expect("cpu dense");
        let dense = dense_cpu.to(Device::Cuda(0)).expect("dense -> cuda");

        let sparse =
            to_sparse_semi_structured_cutlass(&dense).expect("CUTLASS semi-structured pack");
        assert_eq!(sparse.backend(), SemiStructuredSparseBackend::Cutlass);
        assert_eq!(sparse.shape(), &[32, 32]);
        assert_eq!(sparse.values_shape(), &[32, 16]);
        assert_eq!(sparse.indices_shape(), &[32, 4]);
        assert_eq!(sparse.packed().device(), Device::Cuda(0));
        assert_eq!(sparse.indices().device(), Device::Cuda(0));

        let values = sparse.values().expect("values view");
        assert_eq!(values.device(), Device::Cuda(0));
        let values_host = values
            .cpu()
            .expect("values -> cpu")
            .data_vec()
            .expect("values data");
        let expected: Vec<f32> = (0..32)
            .flat_map(|r| (0..32).step_by(2).map(move |c| (r * 100 + c + 1) as f32))
            .collect();
        assert_eq!(values_host, expected);

        let metadata_cpu = sparse.indices().to(Device::Cpu).expect("metadata -> cpu");
        let metadata = metadata_cpu.data().expect("metadata data").to_vec();
        assert_eq!(metadata.len(), 32 * 4);
        assert!(
            metadata.iter().all(|&m| m == 17_476),
            "metadata must match torch CUTLASS f32 encoding 0x4444"
        );
    }
}
