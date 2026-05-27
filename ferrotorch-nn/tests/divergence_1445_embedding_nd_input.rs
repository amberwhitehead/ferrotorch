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
use ferrotorch_nn::module::Module;
use ferrotorch_nn::Embedding;

/// Divergence: ferrotorch's `Embedding::forward` diverges from
/// `pytorch aten/src/ATen/native/Embedding.cpp:43-53` for a 2-D index input.
///
/// Input: weight `[6, 2]`, index tensor of shape `[2, 2]` = `[[0,1],[2,3]]`.
/// Upstream `F.embedding` returns shape `[2, 2, 2]` (live torch 2.11.0).
/// ferrotorch returns `Err(InvalidArgument)` ("Embedding input must be 1-D").
/// Tracking: #1565
#[test]
#[ignore = "divergence: Embedding::forward rejects N-D index input (pytorch returns (*input_shape, embedding_dim)); pre-existing; tracking #1565"]
fn divergence_embedding_2d_index_input_shape() {
    let weight_data: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let weight =
        Tensor::from_storage(TensorStorage::cpu(weight_data), vec![6, 2], true).unwrap();
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
}
