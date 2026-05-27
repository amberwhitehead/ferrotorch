//! Divergences in commit `8e98ee0d2` (#1253 take, #1254 put):
//!
//! The 16+32 skipped samples in the parity smoke for take/put are skipped
//! by the RUNNER (not the impl) via narrow-contract pre-filters at
//! `tools/parity-sweep/runner/src/main.rs:1276-1296` (take) and `:1304-1339`
//! (put). The pre-filters skip:
//!   - 0-d input (`input.numel() == 0` — buggy condition; numel of a 0-d
//!     tensor is 1, not 0, so this filter doesn't actually fire on 0-d!)
//!   - Any negative-index value
//! Without parity coverage for these cases the impl is unverified there.
//!
//! D7: `pub fn take` at `indexing.rs:3110-3168` correctly wraps negative
//! indices (matches upstream `take`), and live torch oracle confirms upstream
//! also wraps:
//!     >>> torch.take(t([[1,2],[3,4]]), t([-1,-4]))
//!     tensor([4., 1.])
//! Test below pins this AS WORKING; the divergence is that the runner
//! gratuitously skips it, hiding the (correct) behavior from the smoke and
//! preventing parity-sweep from catching a future regression.
//! Promoted to a unit test here.
//!
//! D9: `pub fn take` claims to skip the 0-d input case in the runner via
//! `if input.numel() == 0 { return Ok(None); }`. But `Tensor::numel()` for
//! a 0-d tensor (shape `[]`) returns 1 (empty product), so this filter
//! never fires! The runner DOES forward 0-d input to `take`, and the impl
//! handles it (input_data has 1 element, idx_usize wraps). Test pins this
//! as working.
//!
//! D10: `pub fn put` backward at `indexing.rs:3204-3265` with
//! `accumulate=true` and duplicate indices: VJP for source is
//! `grad.take(index)`. Each occurrence of an idx in index produces one
//! grad-source entry. Live oracle:
//!     >>> inp = t([1,2,3,4], rg=T); idx = t([0,0]); src = t([10,20], rg=T)
//!     >>> out = inp.put(idx, src, accumulate=True); out.sum().backward()
//!     >>> src.grad -> tensor([1., 1.])
//!     >>> inp.grad -> tensor([1., 1., 1., 1.])
//! Ferrotorch impl matches; pinned here as regression guard.

use ferrotorch_core::grad_fns::indexing::{put, take};
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn idx(d: Vec<i64>, s: Vec<usize>) -> IntTensor<i64> {
    IntTensor::from_vec(d, s).unwrap()
}

/// D7 / D9 regression pin: take with negative indices on a 2-D input.
/// Live oracle: torch.take(t([[1,2],[3,4]]), t([-1,-4])) = t([4., 1.]).
/// (Indices wrap via `idx + input.numel()`.)
/// This SHOULD pass — included as a regression pin against future changes
/// since the runner pre-filter at `runner/src/main.rs:1292-1295` skips it.
#[test]
fn take_negative_index_wraps_correctly_pin() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![2, 2],
        false,
    )
    .unwrap();
    let i = idx(vec![-1, -4], vec![2]);
    let out = take(&input, &i).unwrap();
    assert_eq!(out.data().unwrap(), &[4.0_f32, 1.0]);
}

/// take with duplicate flat indices: grad accumulates at the dup.
/// Live oracle:
///   inp = t([10,20,30], rg=T); idx = t([0,0,2]); out=take(inp,idx)
///   out.sum().backward() -> inp.grad = t([2., 0., 1.])
#[test]
fn take_backward_accumulates_grad_on_duplicate_indices() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![10.0_f32, 20.0, 30.0]),
        vec![3],
        true,
    )
    .unwrap();
    let i = idx(vec![0, 0, 2], vec![3]);
    let out = take(&input, &i).unwrap();
    let gf = out
        .grad_fn()
        .expect("take must attach TakeBackward when input requires_grad");
    let go =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32, 1.0, 1.0]), vec![3], false).unwrap();
    let grads = gf.backward(&go).expect("take backward must not panic");
    let g = grads[0].as_ref().expect("Some(grad)");
    assert_eq!(
        g.data().unwrap(),
        &[2.0_f32, 0.0, 1.0],
        "take backward must accumulate (idx[0]=idx[1]=0 → grad[0]=2)"
    );
}

/// D9: 0-d input + 1-d index. Runner filter `input.numel() == 0` doesn't
/// fire (0-d has numel=1), so this path runs. Live oracle:
///   torch.take(tensor(5.0), tensor([0])) = tensor([5.])
#[test]
fn take_zero_d_input_returns_scalar_per_oracle() {
    let input = Tensor::from_storage(TensorStorage::cpu(vec![5.0_f32]), vec![], false).unwrap();
    let i = idx(vec![0], vec![1]);
    let out = take(&input, &i).unwrap();
    assert_eq!(out.data().unwrap(), &[5.0_f32]);
    assert_eq!(out.shape(), &[1]);
}

/// D10: put accumulate=true with dup indices. Live oracle:
///   inp = t([1,2,3,4], rg=T); idx = t([0,0]); src = t([10,20], rg=T)
///   out = inp.put(idx, src, accumulate=True)
///   out.sum().backward()
///   inp.grad = t([1., 1., 1., 1.])   # identity for accumulate=true
///   src.grad = t([1., 1.])           # each src element contributes once
#[test]
fn put_accumulate_true_duplicate_index_backward_pin() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![4],
        true,
    )
    .unwrap();
    let i = idx(vec![0, 0], vec![2]);
    let source =
        Tensor::from_storage(TensorStorage::cpu(vec![10.0_f32, 20.0]), vec![2], true).unwrap();
    let out = put(&input, &i, &source, true).unwrap();
    // Forward value: out[0] = 1 + 10 + 20 = 31, rest unchanged
    assert_eq!(out.data().unwrap(), &[31.0_f32, 2.0, 3.0, 4.0]);
    let gf = out.grad_fn().expect("put attaches PutBackward");
    let go = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 1.0, 1.0, 1.0]),
        vec![4],
        false,
    )
    .unwrap();
    let grads = gf.backward(&go).unwrap();
    let g_input = grads[0].as_ref().expect("Some(grad_input)");
    assert_eq!(
        g_input.data().unwrap(),
        &[1.0_f32, 1.0, 1.0, 1.0],
        "put accumulate=true input VJP is identity (`grad` per derivatives.yaml:1422)"
    );
    let g_src = grads[1].as_ref().expect("Some(grad_src)");
    assert_eq!(
        g_src.data().unwrap(),
        &[1.0_f32, 1.0],
        "put accumulate=true source VJP = grad.take(index)"
    );
}

/// put accumulate=False backward. Live oracle:
///   inp=t([1,2,3,4],rg=T); idx=t([0,2]); src=t([100,200],rg=T)
///   out = inp.put(idx, src, accumulate=False); out.sum().backward()
///   inp.grad = t([0., 1., 0., 1.])   # zeros at written positions
///   src.grad = t([1., 1.])
#[test]
fn put_accumulate_false_backward_pin() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![4],
        true,
    )
    .unwrap();
    let i = idx(vec![0, 2], vec![2]);
    let source =
        Tensor::from_storage(TensorStorage::cpu(vec![100.0_f32, 200.0]), vec![2], true).unwrap();
    let out = put(&input, &i, &source, false).unwrap();
    let gf = out.grad_fn().expect("put attaches PutBackward");
    let go = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 1.0, 1.0, 1.0]),
        vec![4],
        false,
    )
    .unwrap();
    let grads = gf.backward(&go).unwrap();
    let g_input = grads[0].as_ref().expect("Some(grad_input)");
    assert_eq!(
        g_input.data().unwrap(),
        &[0.0_f32, 1.0, 0.0, 1.0],
        "put accumulate=false input VJP zeros at written positions"
    );
    let g_src = grads[1].as_ref().expect("Some(grad_src)");
    assert_eq!(
        g_src.data().unwrap(),
        &[1.0_f32, 1.0],
        "put accumulate=false source VJP = grad.take(index)"
    );
}
