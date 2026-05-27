//! Divergence test: `Embedding::forward` rejects N-D index inputs.
//!
//! `<Embedding as Module>::forward` (`ferrotorch-nn/src/embedding.rs:591-597`)
//! hard-rejects any input whose `ndim() != 1`:
//!
//! ```text
//! if input.ndim() != 1 {
//!     return Err(FerrotorchError::InvalidArgument { ... });
//! }
//! ```
//!
//! PyTorch's `F.embedding` (`torch/nn/functional.py` -> `aten` `embedding`,
//! `aten/src/ATen/native/Embedding.cpp:43-53`) accepts an index tensor of ANY
//! shape and returns shape `(*input.shape, embedding_dim)` via
//! `weight.index_select(0, indices.reshape(-1)).view_symint(size)`.
//!
//! This also contradicts the module's OWN design contract, REQ-3 in
//! `.design/ferrotorch-nn/embedding.md`:
//! "accepts an indices tensor (any shape) ... Output shape is
//! `(*input_shape, embedding_dim)`."
//!
//! Expected shape is taken from live torch 2.11.0 (`F.embedding` of a
//! `[2,2]` index tensor against a `[6,2]` weight returns `[2,2,2]`),
//! R-CHAR-3 — no tautologies.
//!
//! Tracking: #1565 (pre-existing; surfaced during the #1445 re-audit).

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::Embedding;
use ferrotorch_nn::module::Module;

/// Divergence: ferrotorch's `Embedding::forward` diverges from
/// `pytorch aten/src/ATen/native/Embedding.cpp:43-53` for a 2-D index input.
///
/// Input: weight `[6, 2]`, index tensor of shape `[2, 2]` = `[[0,1],[2,3]]`.
/// Upstream `F.embedding` returns shape `[2, 2, 2]` (live torch 2.11.0).
/// ferrotorch returns `Err(InvalidArgument)` ("Embedding input must be 1-D").
/// Tracking: #1565
#[test]
fn divergence_embedding_2d_index_input_shape() {
    let weight_data: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let weight = Tensor::from_storage(TensorStorage::cpu(weight_data), vec![6, 2], true).unwrap();
    let emb = Embedding::from_pretrained(weight, None).unwrap();

    // 2-D index tensor [[0,1],[2,3]] -> torch output shape [2,2,2].
    let idx = Tensor::from_storage(
        TensorStorage::cpu(vec![0.0f64, 1.0, 2.0, 3.0]),
        vec![2, 2],
        false,
    )
    .unwrap();

    let out = emb
        .forward(&idx)
        .expect("torch F.embedding accepts N-D index input; ferrotorch rejected it");
    assert_eq!(
        out.shape(),
        &[2, 2, 2],
        "torch returns (*input_shape, embedding_dim) = [2,2,2]"
    );

    // The gathered values are the flattened rows of `weight`: rows 0,1,2,3.
    // weight rows: 0->[0,1] 1->[2,3] 2->[4,5] 3->[6,7]. Flattened output
    // (row-major) is [0,1, 2,3, 4,5, 6,7].
    let data = out.data().unwrap();
    assert_eq!(data, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
}

/// Backward for an N-D index must scatter the per-element grad rows back to
/// the weight exactly as the flattened 1-D path does — the grad to `weight`
/// has shape `[num_embeddings, embedding_dim]` regardless of index rank,
/// mirroring upstream `embedding_dense_backward_cpu`
/// (`aten/src/ATen/native/Embedding.cpp:112-179`), which `view`s the grad to
/// `{numel, grad.size(-1)}` and scatter-adds by the flattened index.
#[test]
fn divergence_embedding_2d_index_backward_to_weight() {
    // weight [4,2]; 2-D index [[1,1],[0,3]] (shape [2,2]) -> output [2,2,2].
    let weight_data: Vec<f64> = vec![
        10.0, 20.0, // row 0
        30.0, 40.0, // row 1
        50.0, 60.0, // row 2
        70.0, 80.0, // row 3
    ];
    let weight = Tensor::from_storage(TensorStorage::cpu(weight_data), vec![4, 2], true).unwrap();
    let emb = Embedding::from_pretrained(weight, None).unwrap();

    let idx = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0f64, 1.0, 0.0, 3.0]),
        vec![2, 2],
        false,
    )
    .unwrap();

    let out = emb.forward(&idx).unwrap();
    assert_eq!(out.shape(), &[2, 2, 2]);
    assert!(out.requires_grad());

    // grad_output has the SAME shape as the output: [2,2,2]. Flattened
    // (row-major) it is the per-position grad rows for indices [1,1,0,3].
    // Rows: pos0(idx1)=[1,1] pos1(idx1)=[2,2] pos2(idx0)=[3,3] pos3(idx3)=[4,4].
    let grad_output = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0f64, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]),
        vec![2, 2, 2],
        false,
    )
    .unwrap();

    let grad_fn = out.grad_fn().unwrap();
    let grads = grad_fn.backward(&grad_output).unwrap();
    let grad_weight = grads[0].as_ref().unwrap();

    // grad_weight is [num_embeddings, embedding_dim] = [4, 2] (NOT the index shape).
    assert_eq!(grad_weight.shape(), &[4, 2]);
    let gd = grad_weight.data().unwrap();
    // Row 0: idx 0 appears once (pos2) -> [3,3].
    // Row 1: idx 1 appears twice (pos0,pos1) -> [1+2, 1+2] = [3,3].
    // Row 2: never accessed -> [0,0].
    // Row 3: idx 3 appears once (pos3) -> [4,4].
    assert_eq!(gd, &[3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 4.0, 4.0]);
}
