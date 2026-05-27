//! Divergences in commit `8e98ee0d2` (S1 batch closure of #1245 scatter_reduce):
//!
//! D1: `pub fn scatter_reduce` reads `src_data[i]` with flat `i` over
//! `index_numel`. When `src` is BIGGER than the index along non-`dim` axes
//! (upstream's allowed contract — src.size(d) >= index.size(d) for all d), the
//! flat-i indexing reads the WRONG element. Upstream walks src using the
//! index-shape coordinates (`src[coords]` where coords iterates over
//! index.shape) — ferrotorch's `src_data[i]` only matches when src.shape() ==
//! index.shape() at every position.
//!
//! Upstream: `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2354
//! TORCH_IMPL_FUNC(scatter_reduce_two)` → `scatter_meta_impl` validates
//! `src.size(d) >= index.size(d)`; the kernel uses the index-shape walk via
//! `cpu_scatter_kernel` to look up src at the same coords.
//!
//! Ferrotorch: `ferrotorch-core/src/grad_fns/indexing.rs:2244`:
//!   `out[dst_flat] = apply_reduce(reduce, out[dst_flat], src_data[i]);`
//! reads `src_data[i]` with `i` from 0..index_numel; for src.shape() ==
//! [2,3] and index.shape() == [2,2], the flat-i traversal hits row-1 src
//! elements starting at flat-3 not flat-2.
//!
//! D2: docstring at `indexing.rs:2082-2085` claims "[non-sum modes] produce
//! a result tensor without a grad_fn attached" but the code at `:2221`,
//! `:2252` unconditionally attaches a `ScatterReduceBackward` grad_fn for
//! all reduce modes. Backward then errors at runtime — which is a worse UX
//! than the docstring promises (no-grad-fn = clean .backward()=no-op on a
//! sum; grad-fn that errors = the user's whole .backward() chain dies).
//!
//! These tests fail against HEAD `8e98ee0d2`.

use ferrotorch_core::GradFn;
use ferrotorch_core::grad_fns::indexing::{ScatterReduce, ScatterReduceBackward, scatter_reduce};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

/// D1: src shape [2,3], index shape [2,2]. Upstream PyTorch produces
/// [[11.0, 52.0, 3.0], [44.0, 25.0, 6.0]] (verified via live oracle).
/// Ferrotorch's flat-i src indexing reads the wrong src elements.
#[test]
fn divergence_scatter_reduce_src_bigger_than_index_along_inner_dim() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
        false,
    )
    .unwrap();
    // index [[0, 1], [1, 0]] — shape [2,2]
    let index: Vec<usize> = vec![0, 1, 1, 0];
    let index_shape = vec![2, 2];
    // src [[10, 20, 99], [40, 50, 99]] — shape [2,3], BIGGER than index in dim 1
    let src = Tensor::from_storage(
        TensorStorage::cpu(vec![10.0_f32, 20.0, 99.0, 40.0, 50.0, 99.0]),
        vec![2, 3],
        false,
    )
    .unwrap();
    let out = scatter_reduce(
        &input,
        0,
        &index,
        &index_shape,
        &src,
        ScatterReduce::Sum,
        true,
    )
    .expect("scatter_reduce forward must not error for src.size(d) >= index.size(d)");
    let got = out.data().unwrap().to_vec();
    // Live torch oracle:
    //   inp.scatter_reduce(0, idx, src, reduce="sum", include_self=True)
    //   -> tensor([[11., 52.,  3.], [44., 25.,  6.]])
    // Per-element derivation:
    //   coords (0,0): idx=0 -> out[0,0] += src[0,0]=10 -> 11
    //   coords (0,1): idx=1 -> out[1,1] += src[0,1]=20 -> 25
    //   coords (1,0): idx=1 -> out[1,0] += src[1,0]=40 -> 44
    //   coords (1,1): idx=0 -> out[0,1] += src[1,1]=50 -> 52
    // Ferrotorch's flat-i traversal would read src[2]=99 at coords (1,0)
    // and src[3]=40 at coords (1,1) — wrong.
    let expected = vec![11.0_f32, 52.0, 3.0, 44.0, 25.0, 6.0];
    assert_eq!(
        got, expected,
        "scatter_reduce must walk src by index-shape coords, not by flat-i"
    );
}

/// D2: prod-mode scatter_reduce attaches a ScatterReduceBackward grad_fn
/// per `indexing.rs:2221/2252`, but the docstring at `:2082-2085` claims
/// non-sum modes produce a tensor "without a grad_fn attached". Verify the
/// docstring claim against the code.
#[test]
fn divergence_scatter_reduce_prod_docstring_claims_no_grad_fn_but_code_attaches_one() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0]),
        vec![3],
        true, // requires_grad
    )
    .unwrap();
    let index: Vec<usize> = vec![0, 2];
    let src = Tensor::from_storage(TensorStorage::cpu(vec![2.0_f32, 3.0]), vec![2], true).unwrap();
    let out = scatter_reduce(&input, 0, &index, &[2], &src, ScatterReduce::Prod, true)
        .expect("prod forward must not error");
    // Per the docstring at `indexing.rs:2082-2085`: "[non-sum modes] produce a
    // result tensor without a grad_fn attached". Therefore:
    assert!(
        out.grad_fn().is_none(),
        "docstring promises non-sum modes attach no grad_fn — but \
         scatter_reduce at indexing.rs:2252 unconditionally attaches \
         ScatterReduceBackward for ALL reduce modes (prod here). Either the \
         docstring is wrong (then users hit a runtime error inside \
         .backward() instead of a no-op) or the code should match the \
         docstring (skip grad_fn for non-sum)."
    );
}

/// D2-companion: confirm the error message from non-sum backward is the
/// documented "only sum implemented" form — not a panic, not a silent zero.
/// (This test PASSES today; included as a behavioral pin so a future "fix"
/// that changes the message or panics is caught.)
#[test]
fn scatter_reduce_amax_backward_returns_invalid_argument_not_panic() {
    let input =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0]), vec![3], true).unwrap();
    let src = Tensor::from_storage(TensorStorage::cpu(vec![5.0_f32, 6.0]), vec![2], true).unwrap();
    let go =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32, 1.0, 1.0]), vec![3], false).unwrap();
    let bw: ScatterReduceBackward<f32> = ScatterReduceBackward {
        input,
        src,
        dim: 0,
        index: vec![0, 1],
        index_shape: vec![2],
        reduce: ScatterReduce::Amax,
        include_self: true,
    };
    let err = GradFn::<f32>::backward(&bw, &go).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("only implemented for reduce='sum'") || msg.contains("only implemented"),
        "amax backward should error with informative message, got: {msg}"
    );
}
