//! Critic audit for wave-G uncommitted nn changes:
//!   - #1443 Conv padding_mode kwarg ("zeros"/"reflect"/"replicate"/"circular")
//!   - #1446 Dropout inplace=true kwarg
//!   - #1448 FeatureAlphaDropout NEW module
//!
//! Each probe pins one observable behaviour from upstream
//! `torch/nn/modules/{conv,dropout}.py` against the ferrotorch surface.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::{Conv2d, Dropout};

fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

// ============================================================================
// #1443 Conv padding_mode kwarg
// ============================================================================

/// #1443: Conv2d::with_padding_mode("reflect") should reflect-pad the input.
/// Upstream `torch/nn/modules/conv.py` `_ConvNd.__init__` accepts
/// `padding_mode ∈ {"zeros","reflect","replicate","circular"}` and threads
/// it through `F.pad(input, self._reversed_padding_repeated_twice, self.padding_mode)`
/// when padding_mode != "zeros".
///
/// Behavioral expectation: under reflect padding a tensor like [[1,2,3]] padded
/// by 1 on the left produces [[2,1,2,3]], whereas zero-pad gives [[0,1,2,3]].
/// We probe by constructing the conv with an identity-shaped kernel and
/// confirming the output of reflect-pad differs from zero-pad.
#[test]
fn audit_1443_conv_padding_mode_reflect_observably_differs_from_zero() {
    // We can't call a method that doesn't exist; compile-fail confirms the
    // kwarg surface is absent. Attempt a heuristic API discovery instead.
    let conv = Conv2d::<f32>::new(1, 1, (3, 3), (1, 1), (1, 1), false).unwrap();
    // The forward path uses zero padding only. Probe input where the
    // boundary value differs depending on padding mode: a single-channel
    // 1x1x3x3 with [[1,2,3],[4,5,6],[7,8,9]] yields a different result
    // under reflect vs zero pad along the top-left corner.
    let inp = cpu_tensor(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
    );
    // Run forward — succeeds, but the result is the zero-padded one.
    let out = conv.forward(&inp).unwrap();
    assert_eq!(out.shape(), &[1, 1, 3, 3]);

    // The blocker claim cannot be exercised at all: there is no
    // `padding_mode` parameter to set. This test PASSES only because the
    // public surface for #1443 doesn't exist; the divergence is the
    // *absence* of the API. To make this test meaningful, expose
    // `Conv2d::with_padding_mode(&str)` (or a `padding_mode` field on
    // `new_full`) and rerun.
    let methods_exist = false; // placeholder for ".with_padding_mode" / "padding_mode" field
    assert!(
        methods_exist,
        "Conv2d has no padding_mode kwarg / setter; #1443 is VOCAB-ONLY (not even VOCAB — surface absent)"
    );
}

// ============================================================================
// #1446 Dropout inplace=true kwarg
// ============================================================================

/// #1446: Dropout::new_inplace(p, inplace=true) should match PyTorch's
/// inplace semantics. The current implementation accepts the flag but is
/// documented (and implemented) as a no-op — input tensors are not mutated.
///
/// Upstream `torch/nn/modules/dropout.py:53-58`:
///   class Dropout(_DropoutNd):
///       def forward(self, input: Tensor) -> Tensor:
///           return F.dropout(input, self.p, self.training, self.inplace)
///
/// And `torch/nn/functional.py::dropout(.., inplace=True)` mutates `input`
/// in place — observable via input.data_ptr() and via the input tensor
/// reading back as the same storage post-call.
///
/// Probe: construct Dropout with inplace=true, capture the input identity,
/// run forward, and assert the input was mutated. ferrotorch's
/// new_inplace silently accepts the flag as a no-op — the input remains
/// unchanged after forward.
#[test]
fn audit_1446_dropout_inplace_true_actually_mutates_input() {
    let d = Dropout::<f32>::new_inplace(0.5, true).unwrap();
    let inp = cpu_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]);
    let inp_before = inp.data_vec().unwrap();
    let _out = d.forward(&inp).unwrap();
    let inp_after = inp.data_vec().unwrap();
    // Upstream contract: with inplace=true, AT LEAST ONE element of
    // `input` should have been overwritten (zeros or scaled). If
    // ferrotorch's "inplace" is a documented no-op, this fails.
    let any_mutated = inp_before
        .iter()
        .zip(inp_after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-8);
    assert!(
        any_mutated,
        "Dropout::new_inplace(_, true) accepted inplace=true but did not mutate input; \
         this is a VOCAB-ONLY divergence vs torch.nn.Dropout(inplace=True)"
    );
}

// ============================================================================
// #1448 FeatureAlphaDropout NEW module
// ============================================================================

/// #1448: FeatureAlphaDropout should be a separate module.
/// Upstream `torch/nn/modules/dropout.py` defines it as a class:
///   class FeatureAlphaDropout(_DropoutNd):
///       def forward(self, input): return F.feature_alpha_dropout(input, ...)
///
/// Currently ferrotorch-nn does not export FeatureAlphaDropout. The
/// re-export at `ferrotorch-nn/src/lib.rs:214` lists only `AlphaDropout,
/// Dropout, Dropout1d, Dropout2d, Dropout3d` — no FeatureAlphaDropout.
///
/// Because the type does not exist, a compile-time check is the only
/// meaningful probe. We use a `type _ = ferrotorch_nn::FeatureAlphaDropout<f32>`
/// assertion: if it doesn't compile, the audit verdict is "absent".
#[test]
fn audit_1448_feature_alpha_dropout_exists() {
    // If ferrotorch-nn re-exports FeatureAlphaDropout, this constant compiles.
    // Otherwise the file would not compile (`unresolved import`). Without
    // resorting to compile_fail we runtime-assert the absence directly:
    let exists = false; // ferrotorch_nn::FeatureAlphaDropout<f32> does not resolve
    assert!(
        exists,
        "ferrotorch-nn does not export FeatureAlphaDropout (#1448); REQ-13 still \
         NOT-STARTED per dropout.rs:28"
    );
}
