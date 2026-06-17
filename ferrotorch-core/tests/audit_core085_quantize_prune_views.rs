//! CORE-085 (#1779): quantization and pruning CPU paths must consume the
//! logical tensor view, not the packed storage slice.
//!
//! PyTorch oracles (local `/home/doll/pytorch`, torch 2.11.0+cu130):
//!
//! ```python
//! x = torch.arange(1., 13.).reshape(3, 4).t()
//! obs = torch.ao.quantization.observer.MinMaxObserver(
//!     dtype=torch.qint8, qscheme=torch.per_tensor_affine)
//! obs(x)
//! scale, zp = obs.calculate_qparams()
//! torch.quantize_per_tensor(x, float(scale), int(zp),
//!                           dtype=torch.qint8).int_repr().reshape(-1)
//! # [-107, -22, 63, -86, 0, 84, -64, 21, 106, -43, 42, 127]
//!
//! v = torch.tensor([-9., 1., -8., -2., 7., 3., 6., -4.]).as_strided((4,), (2,), 1)
//! mask = torch.nn.utils.prune.L1Unstructured(0.375).compute_mask(v, torch.ones_like(v))
//! v * mask
//! # [0.0, -0.0, 3.0, -4.0]
//!
//! y = torch.tensor([[1., 2.], [-3., 4.], [5., -6.], [7., 8.]]).t()
//! scores = y.reshape(-1, 4) * y.reshape(-1, 4)
//! idx = torch.topk(scores, k=2, dim=1, largest=False).indices
//! torch.ones_like(scores).scatter(1, idx, 0).view_as(y) * y
//! # [[0.0, -0.0, 5.0, 7.0], [0.0, 0.0, -6.0, 8.0]]
//! ```

use ferrotorch_core::pruning::{apply_2_4_mask, magnitude_prune, sparsity_ratio};
use ferrotorch_core::quantize::{QuantDtype, QuantScheme, quantize};
use ferrotorch_core::{Tensor, TensorStorage};

#[cfg(feature = "gpu")]
use ferrotorch_core::{FerrotorchError, device::Device};

fn cpu_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).expect("cpu tensor")
}

fn assert_f32_bits(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: actual={actual:?}, expected={expected:?}"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{label}: index {i} bit mismatch (actual={a:?}, expected={e:?})"
        );
    }
}

#[test]
fn quantize_per_tensor_reads_transposed_view_in_logical_order() {
    let base = cpu_tensor((1..=12).map(|v| v as f32).collect(), vec![3, 4]);
    let view = base
        .as_strided(&[4, 3], &[1, 4], Some(0))
        .expect("transpose view");
    assert!(
        !view.is_contiguous(),
        "probe must exercise a non-contiguous view"
    );
    assert_eq!(
        view.data_vec().expect("logical view values"),
        vec![
            1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0
        ]
    );

    let qt = quantize(&view, QuantScheme::PerTensor, QuantDtype::Int8).expect("quantize view");
    let codes: Vec<i32> = qt.data().iter().map(|&v| i32::from(v)).collect();

    assert_eq!(qt.shape(), &[4, 3]);
    assert_eq!(
        codes,
        &[-107, -22, 63, -86, 0, 84, -64, 21, 106, -43, 42, 127]
    );
}

#[test]
fn quantize_per_channel_reads_transposed_view_in_logical_order() {
    let base = cpu_tensor(
        vec![
            -6.0, 1.0, -2.0, 3.0, 4.0, -8.0, 7.0, -5.0, 9.0, 10.0, -11.0, 12.0,
        ],
        vec![3, 4],
    );
    let view = base
        .as_strided(&[4, 3], &[1, 4], Some(0))
        .expect("transpose view");
    assert!(!view.is_contiguous());

    let materialized = cpu_tensor(
        view.data_vec().expect("materialized logical view"),
        vec![4, 3],
    );
    let view_q =
        quantize(&view, QuantScheme::PerChannel(1), QuantDtype::Int8).expect("view quantize");
    let mat_q = quantize(&materialized, QuantScheme::PerChannel(1), QuantDtype::Int8)
        .expect("materialized quantize");

    assert_eq!(view_q.shape(), mat_q.shape());
    assert_eq!(view_q.data(), mat_q.data());
    assert_eq!(view_q.zero_point(), mat_q.zero_point());
    assert_eq!(view_q.scale(), mat_q.scale());
}

#[test]
fn magnitude_prune_reads_offset_strided_view_and_preserves_signed_zero() {
    let base = cpu_tensor(vec![-9.0, 1.0, -8.0, -2.0, 7.0, 3.0, 6.0, -4.0], vec![8]);
    let view = base
        .as_strided(&[4], &[2], Some(1))
        .expect("offset stride-2 view");
    assert_eq!(
        view.data_vec().expect("logical view"),
        vec![1.0, -2.0, 3.0, -4.0]
    );

    let no_prune = magnitude_prune(&view, 0.125).expect("half-even rounds 0.5 to 0");
    assert_f32_bits(
        &no_prune.data_vec().expect("no prune values"),
        &[1.0, -2.0, 3.0, -4.0],
        "0.125 prune count",
    );

    let pruned = magnitude_prune(&view, 0.375).expect("half-even rounds 1.5 to 2");
    assert_eq!(pruned.shape(), &[4]);
    assert_f32_bits(
        &pruned.data_vec().expect("pruned values"),
        &[0.0, -0.0, 3.0, -4.0],
        "offset strided magnitude prune",
    );
}

#[test]
fn apply_2_4_mask_reads_transposed_rows_without_crossing_logical_rows() {
    let base = cpu_tensor(vec![1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, 8.0], vec![4, 2]);
    let view = base
        .as_strided(&[2, 4], &[1, 2], Some(0))
        .expect("2x4 transpose view");
    assert!(!view.is_contiguous());
    assert_eq!(
        view.data_vec().expect("logical 2:4 input"),
        vec![1.0, -3.0, 5.0, 7.0, 2.0, 4.0, -6.0, 8.0]
    );

    let masked = apply_2_4_mask(&view).expect("apply 2:4 mask");
    assert_eq!(masked.shape(), &[2, 4]);
    assert_f32_bits(
        &masked.data_vec().expect("masked values"),
        &[0.0, -0.0, 5.0, 7.0, 0.0, 0.0, -6.0, 8.0],
        "transposed 2:4 mask",
    );
}

#[test]
fn sparsity_ratio_counts_logical_values_in_strided_view() {
    let base = cpu_tensor(vec![9.0, 0.0, 8.0, -0.0, 7.0, 3.0, 6.0, 0.0], vec![8]);
    let view = base
        .as_strided(&[4], &[2], Some(1))
        .expect("offset stride-2 view");
    assert_eq!(
        view.data_vec().expect("logical values"),
        vec![0.0, -0.0, 3.0, 0.0]
    );

    let ratio = sparsity_ratio(&view).expect("sparsity ratio");
    assert_eq!(ratio, 0.75);
}

#[cfg(feature = "gpu")]
#[test]
fn quantize_cuda_still_rejects_without_host_readback() {
    ferrotorch_gpu::init_cuda_backend().expect("CUDA backend");
    let gpu = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4])
        .to(Device::Cuda(0))
        .expect("upload");

    let err = quantize(&gpu, QuantScheme::PerTensor, QuantDtype::Int8)
        .expect_err("quantize must reject CUDA tensors");
    assert!(
        matches!(
            err,
            FerrotorchError::NotImplementedOnCuda { op: "quantize" }
        ),
        "quantize CUDA must report a structured no-host-readback error, got {err:?}"
    );
}
