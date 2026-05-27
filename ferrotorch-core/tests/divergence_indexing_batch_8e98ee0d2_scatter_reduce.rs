//! scatter_reduce forward + backward characterization (originally S1 batch
//! closure of #1245; updated post-`c013b5432` which IMPLEMENTED value-aware
//! prod/amax/amin VJPs).
//!
//! D1 (forward, still a live divergence pin): `scatter_reduce` must walk `src`
//! by index-shape coordinates, not by flat-i, when `src` is bigger than the
//! index along a non-`dim` axis. Upstream:
//! `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2354
//! TORCH_IMPL_FUNC(scatter_reduce_two)` → `scatter_meta_impl` validates
//! `src.size(d) >= index.size(d)` and the kernel reads src at the same coords.
//!
//! BACKWARD (post-c013b5432, CHARACTERIZATION — these are regression guards):
//! commit `c013b5432` added `ScatterReduceBackward::result` and implemented the
//! value-aware VJPs for prod/amax/amin per
//! `torch/csrc/autograd/FunctionsManual.cpp:7194-7279`. An independent live
//! `torch==2.11.0` audit (forward + `.backward()`, float64, reading
//! `input.grad` / `src.grad`) found ferrotorch matches torch EXACTLY on all
//! 21 audited cases — including prod-with-zeros (single/double/self zero) and
//! amax/amin tie-splitting. The two tests previously asserting the *superseded*
//! behavior (prod attaches no grad_fn; amax backward errors "only sum") are now
//! rewritten to PIN the correct torch-matching gradients. Every expected value
//! below is sourced from the live-torch oracle, NOT copied from ferrotorch
//! (R-CHAR-3). Oracle reproduction:
//!   inp = torch.tensor(<inp>, dtype=torch.float64, requires_grad=True)
//!   src = torch.tensor(<src>, dtype=torch.float64, requires_grad=True)
//!   r = inp.scatter_reduce(0, torch.tensor(<idx>), src,
//!                          reduce=<reduce>, include_self=<inc>)
//!   r.backward(torch.arange(1, r.numel()+1, dtype=torch.float64))
//!   inp.grad, src.grad

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
    let expected = vec![11.0_f32, 52.0, 3.0, 44.0, 25.0, 6.0];
    assert_eq!(
        got, expected,
        "scatter_reduce must walk src by index-shape coords, not by flat-i"
    );
}

/// Helper: build a 1-D `ScatterReduceBackward`, run it, and return
/// `(inp.grad, src.grad)`. The `result` buffer matches what the forward saves
/// (verified separately via the full forward+engine-backward integration path).
fn bw_1d(
    inp: Vec<f64>,
    index: Vec<usize>,
    src: Vec<f64>,
    result: Vec<f64>,
    go: Vec<f64>,
    reduce: ScatterReduce,
    include_self: bool,
) -> (Vec<f64>, Vec<f64>) {
    let n = inp.len();
    let input = Tensor::from_storage(TensorStorage::cpu(inp), vec![n], true).unwrap();
    let sl = src.len();
    let srct = Tensor::from_storage(TensorStorage::cpu(src), vec![sl], true).unwrap();
    let got = Tensor::from_storage(TensorStorage::cpu(go), vec![n], false).unwrap();
    let bw: ScatterReduceBackward<f64> = ScatterReduceBackward {
        input,
        src: srct,
        dim: 0,
        index_shape: vec![index.len()],
        index,
        reduce,
        include_self,
        result,
    };
    let grads = GradFn::<f64>::backward(&bw, &got).unwrap();
    let gi = grads[0].as_ref().unwrap().data_vec().unwrap();
    let gs = grads[1].as_ref().unwrap().data_vec().unwrap();
    (gi, gs)
}

/// CHARACTERIZATION (was: `..._prod_docstring_claims_no_grad_fn_but_code_...`).
///
/// Pre-c013b5432 this asserted prod attaches NO grad_fn (per a now-removed
/// docstring). c013b5432 implemented the prod VJP; torch attaches
/// `ScatterReduceBackward0` for every mode. This now pins the torch-matching
/// prod gradient AND asserts the grad_fn IS attached (the docstring was wrong;
/// the code is right).
#[test]
fn scatter_reduce_prod_attaches_grad_fn_and_matches_torch() {
    // grad_fn must be present for prod when grad is enabled (torch:
    // r.grad_fn is <ScatterReduceBackward0>).
    let input =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0]), vec![3], true).unwrap();
    let src = Tensor::from_storage(TensorStorage::cpu(vec![2.0_f32, 3.0]), vec![2], true).unwrap();
    let out = scatter_reduce(&input, 0, &[0, 2], &[2], &src, ScatterReduce::Prod, true)
        .expect("prod forward must not error");
    assert!(
        out.grad_fn().is_some(),
        "torch attaches ScatterReduceBackward0 for reduce='prod'; ferrotorch must too"
    );

    // prod, NO zeros, include_self=True. Oracle:
    //   inp=[1,2,3] idx=[0,2] src=[2,3] result=[2,2,9] go=[1,2,3]
    //   inp.grad=[2,2,9]  src.grad=[1,9]
    let (gi, gs) = bw_1d(
        vec![1., 2., 3.],
        vec![0, 2],
        vec![2., 3.],
        vec![2., 2., 9.],
        vec![1., 2., 3.],
        ScatterReduce::Prod,
        true,
    );
    assert_eq!(gi, vec![2., 2., 9.], "prod inp.grad (no zeros, include_self)");
    assert_eq!(gs, vec![1., 9.], "prod src.grad (no zeros, include_self)");

    // prod with a SINGLE zero in src scattering into slot 0, include_self=True.
    // Oracle: inp=[4,5,6] idx=[0,0,2] src=[2,0,3] result=[0,5,18] go=[1,2,3]
    //   inp.grad=[0,2,9]  src.grad=[0,8,18]
    let (gi, gs) = bw_1d(
        vec![4., 5., 6.],
        vec![0, 0, 2],
        vec![2., 0., 3.],
        vec![0., 5., 18.],
        vec![1., 2., 3.],
        ScatterReduce::Prod,
        true,
    );
    assert_eq!(gi, vec![0., 2., 9.], "prod inp.grad (single src zero)");
    assert_eq!(gs, vec![0., 8., 18.], "prod src.grad (single src zero)");

    // prod with a zero in SELF, include_self=True. Oracle:
    //   inp=[0,5,6] idx=[0,2] src=[2,3] result=[0,5,18] go=[1,2,3]
    //   inp.grad=[2,2,9]  src.grad=[0,18]
    let (gi, gs) = bw_1d(
        vec![0., 5., 6.],
        vec![0, 2],
        vec![2., 3.],
        vec![0., 5., 18.],
        vec![1., 2., 3.],
        ScatterReduce::Prod,
        true,
    );
    assert_eq!(gi, vec![2., 2., 9.], "prod inp.grad (self zero)");
    assert_eq!(gs, vec![0., 18.], "prod src.grad (self zero)");

    // prod with TWO src zeros into slot 0, include_self=True. Oracle:
    //   inp=[4,5,6] idx=[0,0,2] src=[0,0,3] result=[0,5,18] go=[1,2,3]
    //   inp.grad=[0,2,9]  src.grad=[0,0,18]
    let (gi, gs) = bw_1d(
        vec![4., 5., 6.],
        vec![0, 0, 2],
        vec![0., 0., 3.],
        vec![0., 5., 18.],
        vec![1., 2., 3.],
        ScatterReduce::Prod,
        true,
    );
    assert_eq!(gi, vec![0., 2., 9.], "prod inp.grad (double src zero)");
    assert_eq!(gs, vec![0., 0., 18.], "prod src.grad (double src zero)");

    // prod, NO zeros, include_self=False. Oracle:
    //   inp=[1,2,3] idx=[0,2] src=[2,3] result=[2,2,3] go=[1,2,3]
    //   inp.grad=[0,2,0]  src.grad=[1,3]
    let (gi, gs) = bw_1d(
        vec![1., 2., 3.],
        vec![0, 2],
        vec![2., 3.],
        vec![2., 2., 3.],
        vec![1., 2., 3.],
        ScatterReduce::Prod,
        false,
    );
    assert_eq!(gi, vec![0., 2., 0.], "prod inp.grad (no zeros, !include_self)");
    assert_eq!(gs, vec![1., 3.], "prod src.grad (no zeros, !include_self)");
}

/// CHARACTERIZATION (was:
/// `scatter_reduce_amax_backward_returns_invalid_argument_not_panic`).
///
/// Pre-c013b5432 amax backward errored "only implemented for reduce='sum'".
/// c013b5432 implemented the amax/amin tie-splitting VJP
/// (`FunctionsManual.cpp:7256-7265`). torch distributes the gradient evenly
/// across every position (self + scattered src) whose value equals the
/// per-slot max/min. This pins the torch-matching VALUES, including ties.
#[test]
fn scatter_reduce_amax_amin_backward_matches_torch_incl_ties() {
    // amax TIE: self[0]=5 ties with src[0]=5 into slot 0 -> grad 1.0 split 0.5/0.5.
    // Oracle: inp=[5,2,3] idx=[0,1] src=[5,6] result=[5,6,3] go=[1,2,3]
    //   include_self=True : inp.grad=[0.5,0,3]  src.grad=[0.5,2]
    let (gi, gs) = bw_1d(
        vec![5., 2., 3.],
        vec![0, 1],
        vec![5., 6.],
        vec![5., 6., 3.],
        vec![1., 2., 3.],
        ScatterReduce::Amax,
        true,
    );
    assert_eq!(gi, vec![0.5, 0., 3.], "amax tie inp.grad (include_self)");
    assert_eq!(gs, vec![0.5, 2.], "amax tie src.grad (include_self)");

    // amax TIE, include_self=False: self contribution zeroed by post-processing.
    // Oracle: inp.grad=[0,0,3]  src.grad=[0.5,2]
    let (gi, gs) = bw_1d(
        vec![5., 2., 3.],
        vec![0, 1],
        vec![5., 6.],
        vec![5., 6., 3.],
        vec![1., 2., 3.],
        ScatterReduce::Amax,
        false,
    );
    assert_eq!(gi, vec![0., 0., 3.], "amax tie inp.grad (!include_self)");
    assert_eq!(gs, vec![0.5, 2.], "amax tie src.grad (!include_self)");

    // amax NO tie, include_self=True: self[0]=1 != max 5, so inp.grad[0]=0.
    // Oracle: inp=[1,2,3] idx=[0,1] src=[5,6] result=[5,6,3] go=[1,2,3]
    //   inp.grad=[0,0,3]  src.grad=[1,2]
    let (gi, gs) = bw_1d(
        vec![1., 2., 3.],
        vec![0, 1],
        vec![5., 6.],
        vec![5., 6., 3.],
        vec![1., 2., 3.],
        ScatterReduce::Amax,
        true,
    );
    assert_eq!(gi, vec![0., 0., 3.], "amax notie inp.grad (include_self)");
    assert_eq!(gs, vec![1., 2.], "amax notie src.grad (include_self)");

    // amax MULTI-SRC tie: src[0]=src[1]=7 both scatter into slot 0; self[0]=1
    // loses. grad_out[0]=1 splits 0.5/0.5 across the two src; inp.grad[0]=0.
    // Oracle: inp=[1,2,3] idx=[0,0,2] src=[7,7,4] result=[7,2,4] go=[1,2,3]
    //   inp.grad=[0,2,0]  src.grad=[0.5,0.5,3]
    let (gi, gs) = bw_1d(
        vec![1., 2., 3.],
        vec![0, 0, 2],
        vec![7., 7., 4.],
        vec![7., 2., 4.],
        vec![1., 2., 3.],
        ScatterReduce::Amax,
        true,
    );
    assert_eq!(gi, vec![0., 2., 0.], "amax multi-src tie inp.grad");
    assert_eq!(gs, vec![0.5, 0.5, 3.], "amax multi-src tie src.grad");

    // amin TIE: self[0]=5 ties with src[0]=5 (the min into slot 0).
    // Oracle: inp=[5,2,3] idx=[0,1] src=[5,1] result=[5,1,3] go=[1,2,3]
    //   include_self=True : inp.grad=[0.5,0,3]  src.grad=[0.5,2]
    let (gi, gs) = bw_1d(
        vec![5., 2., 3.],
        vec![0, 1],
        vec![5., 1.],
        vec![5., 1., 3.],
        vec![1., 2., 3.],
        ScatterReduce::Amin,
        true,
    );
    assert_eq!(gi, vec![0.5, 0., 3.], "amin tie inp.grad (include_self)");
    assert_eq!(gs, vec![0.5, 2.], "amin tie src.grad (include_self)");
}

/// CHARACTERIZATION: 2-D dim=0 amax with a tie, full forward+backward via the
/// hand-built grad_fn (the `result` buffer matches the forward's saved buffer —
/// confirmed against the integration path). Stresses the index-shape coords
/// walk in 2-D. Oracle:
///   inp=[[9,2,3],[4,5,6]] idx=[[0,1,0],[1,0,1]] src=[[9,8,7],[1,1,1]]
///   reduce=amax include_self=True go=[[1,2,3],[4,5,6]]
///   result=[[9,2,7],[4,8,6]]
///   inp.grad=[[0.5,2,0],[4,0,6]]  src.grad=[[0.5,5,3],[0,0,0]]
#[test]
fn scatter_reduce_2d_amax_tie_matches_torch() {
    let input =
        Tensor::from_storage(TensorStorage::cpu(vec![9., 2., 3., 4., 5., 6.]), vec![2, 3], true)
            .unwrap();
    let src =
        Tensor::from_storage(TensorStorage::cpu(vec![9., 8., 7., 1., 1., 1.]), vec![2, 3], true)
            .unwrap();
    let go = Tensor::from_storage(
        TensorStorage::cpu(vec![1., 2., 3., 4., 5., 6.]),
        vec![2, 3],
        false,
    )
    .unwrap();
    let bw: ScatterReduceBackward<f64> = ScatterReduceBackward {
        input,
        src,
        dim: 0,
        index: vec![0, 1, 0, 1, 0, 1],
        index_shape: vec![2, 3],
        reduce: ScatterReduce::Amax,
        include_self: true,
        result: vec![9., 2., 7., 4., 8., 6.],
    };
    let grads = GradFn::<f64>::backward(&bw, &go).unwrap();
    let gi = grads[0].as_ref().unwrap().data_vec().unwrap();
    let gs = grads[1].as_ref().unwrap().data_vec().unwrap();
    assert_eq!(gi, vec![0.5, 2., 0., 4., 0., 6.], "2d amax tie inp.grad");
    assert_eq!(gs, vec![0.5, 5., 3., 0., 0., 0.], "2d amax tie src.grad");
}
