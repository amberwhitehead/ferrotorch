//! Divergence probe for in-place dropout (#1446) on a leaf that requires grad.
//!
//! Upstream contract: PyTorch forbids any in-place operation on a leaf tensor
//! that requires grad. `torch.nn.Dropout(p, inplace=True)(leaf_requiring_grad)`
//! raises:
//!   "RuntimeError: a leaf Variable that requires grad is being used in an
//!    in-place operation."
//! Confirmed live against torch 2.11.0+cu130 with
//!   x = torch.tensor([1..6], requires_grad=True); nn.Dropout(0.5, inplace=True)(x).
//!
//! ferrotorch's `<Dropout as Module>::forward` (`ferrotorch-nn/src/dropout.rs:419`)
//! unconditionally calls `write_inplace(input, ..)` whenever `self.inplace &&
//! training`, with no check on `input.is_leaf() && input.requires_grad()`. It
//! silently mutates the leaf's storage in place and returns `Ok`.

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_nn::{Dropout, Module};

/// Divergence: `Dropout::forward(.., inplace=true)` on a leaf requiring grad.
/// Upstream PyTorch raises a RuntimeError ("a leaf Variable that requires grad
/// is being used in an in-place operation"). ferrotorch silently mutates the
/// leaf's storage (e.g. `[1,2,3,4,5,6]` -> `[2,4,0,0,10,12]`) and returns Ok.
///
/// Upstream: torch autograd leaf-inplace guard (raised from `_VF.dropout_`).
/// ferrotorch: ferrotorch-nn/src/dropout.rs:419-421 (write_inplace, no leaf guard).
/// Tracking: #1581 (blocker).
#[test]
fn divergence_inplace_dropout_on_grad_leaf_must_error() {
    let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::from_storage(TensorStorage::cpu(original.clone()), vec![6], true).unwrap();
    assert!(x.is_leaf() && x.requires_grad());

    let d = Dropout::<f32>::new(0.5).unwrap().with_inplace(true);
    let result = d.forward(&x);

    // PyTorch raises. ferrotorch must EITHER error OR leave the leaf untouched
    // — it must NOT silently mutate a grad-requiring leaf's storage.
    if result.is_ok() {
        let after = x.data().unwrap().to_vec();
        assert_eq!(
            after, original,
            "inplace dropout silently mutated a leaf that requires grad: {after:?}. \
             Upstream torch raises 'a leaf Variable that requires grad is being used \
             in an in-place operation.'",
        );
    }
}
