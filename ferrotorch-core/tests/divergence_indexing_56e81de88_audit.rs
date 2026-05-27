//! Audit divergences in commit 56e81de88 (the "fix-7-divergences" patch
//! that landed against pin #1286). These are NEW divergences introduced
//! BY the fix itself, plus pre-existing ones the fix failed to address.
//!
//! Live oracle pins (torch 2.11.0+cu130):
//!
//! A. `index_add(0-d self, 0, [0], 1-D length-1 src)` —
//!    upstream `RuntimeError: source tensor shape must match self tensor
//!    shape, excluding the specified dimension. Got self.shape = []
//!    source.shape = [1]`; ferrotorch ACCEPTS via the
//!    `source_is_0d_compatible` permissive branch at
//!    `grad_fns/indexing.rs:2701-2702` whose docstring claims it mirrors
//!    upstream `:1280-1287` — it does not.
//!
//! B. `index_copy(1-D self, 0, [1], 0-d src)` —
//!    upstream RESULT `tensor([1., 99., 3., 4.])` (broadcasts the scalar);
//!    ferrotorch REJECTS via `strict_index_add_copy_validate` at
//!    `:2412-2422`. The shared helper enforces the strict `source_ndim ==
//!    0 && ndim > 0 → error` rule, but per upstream meta at
//!    `TensorAdvancedIndexing.cpp:285-300` index_copy specifically
//!    PERMITS 0-d source — it's index_add that forbids it. Sharing the
//!    helper between the two ops conflates the contracts.
//!
//! C. `scatter_reduce(reduce='amax'|'amin'|'prod', requires_grad=True)`
//!    grad_fn attach — upstream attaches `ScatterReduceBackward0` and
//!    `r.sum().backward()` produces VALID grads (verified live: amax
//!    src.grad=[1,1], prod src.grad=[1,2]). Ferrotorch's "fix" at
//!    `:2241-2257` and `:2275-2289` (D2 path) deliberately skips the
//!    grad_fn attach for non-sum modes, so `.backward()` on the user's
//!    chain through a non-sum scatter_reduce silently produces wrong /
//!    no grads (the result tensor has `requires_grad=false`, breaking
//!    the chain). This is a NEW divergence introduced by 56e81de88 —
//!    the prior (8e98ee0d2) impl unconditionally attached but errored
//!    inside backward; this impl silently breaks the autograd chain.
//!
//! D. Runner pre-filter masking. `tools/parity-sweep/runner/src/main.rs`
//!    at `:1190-1218` (index_add) and `:1253-1281` (index_copy) was
//!    EXTENDED to skip the exact inputs that the strict-validate helper
//!    now rejects (negative-idx, source-size mismatch, 0-d-source on N-D
//!    self). The parity sweep's "0 failed" report therefore does NOT
//!    exercise these paths — they're filtered out before reaching the
//!    impl. The harness gap is not a divergence per se but it hides
//!    Divergences A/B/C from the parity sweep.

use ferrotorch_core::GradFn;
use ferrotorch_core::grad_fns::indexing::{ScatterReduce, index_add, index_copy, scatter_reduce};
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn idx(d: Vec<i64>, s: Vec<usize>) -> IntTensor<i64> {
    IntTensor::from_vec(d, s).unwrap()
}

/// Divergence A: ferrotorch accepts 1-D length-1 source on 0-d self.
/// Upstream errors: "source tensor shape must match self tensor shape,
/// excluding the specified dimension. Got self.shape = [] source.shape = [1]"
/// Live oracle: torch.index_add(t(5.), 0, t([0]), t([99.])) -> RuntimeError.
#[test]
fn audit_56e81de88_index_add_0d_self_1d_len1_source_should_error() {
    let input = Tensor::from_storage(TensorStorage::cpu(vec![5.0_f32]), vec![], false).unwrap();
    let i = idx(vec![0], vec![1]);
    let source = Tensor::from_storage(TensorStorage::cpu(vec![99.0_f32]), vec![1], false).unwrap();
    let res = index_add(&input, 0, &i, &source, 1.0);
    assert!(
        res.is_err(),
        "index_add with 0-D self and 1-D length-1 source must error per \
         upstream `source tensor shape must match self tensor shape, \
         excluding the specified dimension. Got self.shape = [] source.shape \
         = [1]`; ferrotorch ACCEPTS via the `source_is_0d_compatible` \
         permissive branch at grad_fns/indexing.rs:2701-2702 whose comment \
         claims it mirrors upstream :1280-1287 but does not."
    );
}

/// Divergence B: index_copy with 0-d source on 1-D self.
/// Upstream broadcasts scalar; ferrotorch's shared helper rejects.
/// Live oracle:
///   >>> torch.tensor([1.,2.,3.,4.]).index_copy(0, torch.tensor([1]), torch.tensor(99.))
///   tensor([1., 99., 3., 4.])
#[test]
fn audit_56e81de88_index_copy_0d_source_on_1d_self_should_accept() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![4],
        false,
    )
    .unwrap();
    let i = idx(vec![1], vec![1]);
    let source = Tensor::from_storage(TensorStorage::cpu(vec![99.0_f32]), vec![], false).unwrap();
    let res = index_copy(&input, 0, &i, &source);
    let out = res.expect(
        "index_copy with 0-D source on 1-D self must succeed per upstream \
         (broadcasts the scalar -> [1., 99., 3., 4.]); ferrotorch's \
         `strict_index_add_copy_validate` at indexing.rs:2412-2422 erroneously \
         routes index_copy through the `source_ndim == 0 && ndim > 0` \
         rejection that only applies to index_add. Per upstream meta at \
         `aten/src/ATen/native/TensorAdvancedIndexing.cpp:285-300` \
         index_copy PERMITS 0-d source. Conflating index_add and index_copy \
         contracts in a single helper produces this regression.",
    );
    assert_eq!(out.data().unwrap(), &[1.0_f32, 99.0, 3.0, 4.0]);
}

/// Divergence A-cont: 0-d self + multi-element index + 0-d source.
/// Upstream errors "Dimension specified as 0 but tensor has no dimensions";
/// ferrotorch accepts the literal-0 indices and loops the accumulator.
#[test]
fn audit_56e81de88_index_add_0d_self_multi_index_should_error() {
    let input = Tensor::from_storage(TensorStorage::cpu(vec![5.0_f32]), vec![], false).unwrap();
    let i = idx(vec![0, 0], vec![2]);
    let source = Tensor::from_storage(TensorStorage::cpu(vec![99.0_f32]), vec![], false).unwrap();
    let res = index_add(&input, 0, &i, &source, 1.0);
    assert!(
        res.is_err(),
        "index_add 0-d self with multi-element index must error per upstream \
         `IndexError: Dimension specified as 0 but tensor has no dimensions`; \
         ferrotorch accepts via the `n_indices != 1 && n_indices != 0` \
         pre-check at indexing.rs:2722 — which permits n_indices == 1 \
         but the upstream rejection is on `dim 0 vs ndim 0`, not on n_indices."
    );
}

/// Divergence C: scatter_reduce non-sum mode breaks the autograd chain.
/// Upstream attaches `ScatterReduceBackward0` for ALL reduce modes; the
/// backward for amax/amin/prod IS implemented in upstream (verified live:
/// amax `src.grad = [1.,1.]`, prod `src.grad = [1.,2.]`). The 56e81de88
/// "fix" SKIPS the grad_fn attach for non-sum modes — so a downstream
/// `requires_grad=True` operand of a non-sum scatter_reduce result has
/// no path back. This is a NEW regression introduced by the D2 fix.
#[test]
fn audit_56e81de88_scatter_reduce_amax_breaks_autograd_chain() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0]),
        vec![3],
        true, // requires_grad
    )
    .unwrap();
    let index: Vec<usize> = vec![0, 1];
    let src = Tensor::from_storage(TensorStorage::cpu(vec![5.0_f32, 6.0]), vec![2], true).unwrap();
    let out = scatter_reduce(&input, 0, &index, &[2], &src, ScatterReduce::Amax, true)
        .expect("amax forward must not error");
    // Per upstream: r.requires_grad should be True and r.grad_fn should be set.
    assert!(
        out.grad_fn().is_some(),
        "scatter_reduce amax with requires_grad operands MUST attach a \
         grad_fn per upstream `ScatterReduceBackward0` (live oracle: \
         `r.grad_fn = <ScatterReduceBackward0 ...>`, `r.requires_grad = \
         True`, and `r.sum().backward()` succeeds). The 56e81de88 fix at \
         indexing.rs:2241-2257 / :2275-2289 deliberately skips the \
         grad_fn attach for non-sum modes — breaking the autograd chain. \
         The fix should EITHER (a) implement non-sum backward (preferred, \
         matches upstream), or (b) attach a grad_fn whose backward returns \
         a useful error mentioning the mode. Silently dropping the chain \
         loses information the user needs to know."
    );
    assert!(
        out.requires_grad(),
        "scatter_reduce amax result.requires_grad must be true when inputs \
         require_grad — upstream live oracle confirms `r.requires_grad = \
         True`."
    );
}

/// Divergence C-cont: same for prod (the test that is currently mis-pinned
/// against the wrong upstream behavior in the existing critic test
/// `divergence_scatter_reduce_prod_docstring_claims_no_grad_fn_but_code_attaches_one`).
/// The PRIOR critic test asserts `out.grad_fn().is_none()` based on the
/// docstring's claim — but that docstring claim was WRONG against upstream
/// from day one. This new test pins the correct upstream behavior.
#[test]
fn audit_56e81de88_scatter_reduce_prod_must_attach_grad_fn_per_upstream() {
    let input =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0]), vec![3], true).unwrap();
    let index: Vec<usize> = vec![0, 2];
    let src = Tensor::from_storage(TensorStorage::cpu(vec![2.0_f32, 3.0]), vec![2], true).unwrap();
    let out = scatter_reduce(&input, 0, &index, &[2], &src, ScatterReduce::Prod, true)
        .expect("prod forward must not error");
    // Live upstream:
    //   inp.requires_grad=True, src.requires_grad=True
    //   r = inp.scatter_reduce(0, idx, src, reduce='prod', include_self=True)
    //   r.grad_fn -> <ScatterReduceBackward0 ...>   (attached!)
    //   r.sum().backward() -> inp.grad=[2,1,3], src.grad=[1,2]
    assert!(
        out.grad_fn().is_some(),
        "scatter_reduce prod result MUST carry a grad_fn per upstream \
         (the backward IS implemented in `derivatives.yaml:3074-3077`); \
         the 56e81de88 commit message claims this matches upstream but \
         live torch DOES attach `ScatterReduceBackward0` for all reduce \
         modes. The docstring at indexing.rs:2082-2091 that 56e81de88 \
         'reconciled' was itself a divergence statement from day one."
    );
}

/// Pre-existing critic test `divergence_scatter_reduce_prod_docstring_claims_no_grad_fn_but_code_attaches_one`
/// asserts the OPPOSITE of the upstream truth above. That test now passes
/// not because the bug is fixed but because the bug was made worse (chain
/// silently broken instead of erroring inside backward).
///
/// This audit-pin demonstrates the inconsistency: BOTH assertions cannot
/// be simultaneously consistent with upstream.
#[test]
#[ignore = "documentation pin only: shows the prior critic test is wrong"]
fn audit_56e81de88_prior_critic_test_was_pinned_against_wrong_truth() {
    // Intentional left blank — the body of this test would just be a long
    // comment block. See module docstring section C above and the live
    // oracle output in the audit report.
}
