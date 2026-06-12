//! Red regression tests for CORE-054 (#1748) — CLASS-V, Medium.
//!
//! `repeat` (`src/grad_fns/shape.rs`) documents that a repeat count of zero
//! collapses the corresponding axis to size zero (torch semantics:
//! `aten/src/ATen/native/TensorShape.cpp` `Tensor repeat(...)` — result size
//! along axis `i` is `input_size[i] * repeats[i]`). Its zero branch instead
//! calls `reshape` on the existing NON-empty tensor with a zero-sized shape;
//! `reshape` requires the element count to stay unchanged, so
//! `repeat(non_empty, [..., 0, ...])` returns a shape-mismatch error instead
//! of an empty tensor. `tile` delegates to the same path.
//!
//! Oracle (R-ORACLE-1b): live torch 2.11.0+cu130, 2026-06-11:
//!
//! ```python
//! x = torch.tensor([1.,2.,3.]); y = torch.arange(6.).reshape(2,3)
//! x.repeat(0).shape          # torch.Size([0])
//! y.repeat(0,2).shape        # torch.Size([0, 6])
//! y.repeat(2,0).shape        # torch.Size([4, 0])
//! y.repeat(0,0).shape        # torch.Size([0, 0])
//! x.repeat(0,2).shape        # torch.Size([0, 6])   (leading new dim)
//! x.repeat(2,0).shape        # torch.Size([2, 0])
//! torch.tensor(5.).repeat(0).shape       # torch.Size([0])
//! torch.tensor(5.).repeat(2,0).shape     # torch.Size([2, 0])
//! torch.empty(0).repeat(0).shape         # torch.Size([0])
//! torch.empty(2,0).repeat(3,5).shape     # torch.Size([6, 0])
//! a = torch.tensor([1.,2.], requires_grad=True)
//! out = a.repeat(0)          # requires_grad=True, grad_fn=RepeatBackward0
//! out.sum().backward(); a.grad           # tensor([0., 0.])
//! b = torch.arange(6., requires_grad=True); bb = b.reshape(2,3)
//! o2 = bb.repeat(0,2)        # (0,6), requires_grad=True
//! o2.sum().backward(); b.grad            # tensor([0.,0.,0.,0.,0.,0.])
//! torch.tile(x,(0,)).shape   # torch.Size([0])
//! torch.tile(y,(0,)).shape   # torch.Size([2, 0])  (reps left-padded with 1)
//! torch.tile(y,(2,0)).shape  # torch.Size([4, 0])
//! ```

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn plain(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

/// Assert a repeat/tile result is a genuine empty tensor of `want_shape`:
/// correct shape, zero elements, and readable (empty) host data.
fn assert_empty(got: &Tensor<f32>, want_shape: &[usize], label: &str) {
    assert_eq!(got.shape(), want_shape, "{label}: shape");
    assert_eq!(got.numel(), 0, "{label}: numel");
    assert_eq!(
        got.data_vec()
            .unwrap_or_else(|e| panic!("{label}: result must be readable: {e:?}")),
        Vec::<f32>::new(),
        "{label}: data"
    );
}

// torch: x.repeat(0) -> torch.Size([0])
#[test]
fn repeat_zero_1d() {
    let x = plain(&[1.0, 2.0, 3.0], &[3]);
    let out = x
        .repeat_t(&[0])
        .expect("repeat([0]) of a non-empty 1-D tensor must succeed (CORE-054)");
    assert_empty(&out, &[0], "repeat([0])");
}

// torch: y.repeat(0,2) -> (0,6); y.repeat(2,0) -> (4,0); y.repeat(0,0) -> (0,0)
#[test]
fn repeat_zero_each_axis_2d() {
    let y = plain(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    let cases: [(&[isize], &[usize]); 3] =
        [(&[0, 2], &[0, 6]), (&[2, 0], &[4, 0]), (&[0, 0], &[0, 0])];
    for (reps, want) in cases {
        let out = y
            .repeat_t(reps)
            .unwrap_or_else(|e| panic!("repeat({reps:?}) must succeed (CORE-054): {e:?}"));
        assert_empty(&out, want, &format!("repeat({reps:?})"));
    }
}

// torch: x.repeat(0,2) -> (0,6); x.repeat(2,0) -> (2,0)  (leading new dim)
#[test]
fn repeat_zero_with_leading_new_dim() {
    let x = plain(&[1.0, 2.0, 3.0], &[3]);
    let out = x
        .repeat_t(&[0, 2])
        .expect("repeat([0,2]) with a new leading dim must succeed (CORE-054)");
    assert_empty(&out, &[0, 6], "repeat([0,2])");
    let out = x
        .repeat_t(&[2, 0])
        .expect("repeat([2,0]) with a new leading dim must succeed (CORE-054)");
    assert_empty(&out, &[2, 0], "repeat([2,0])");
}

// torch: torch.tensor(5.).repeat(0) -> (0,); .repeat(2,0) -> (2,0)
#[test]
fn repeat_zero_scalar() {
    let s = plain(&[5.0], &[]);
    let out = s
        .repeat_t(&[0])
        .expect("scalar repeat([0]) must succeed (CORE-054)");
    assert_empty(&out, &[0], "scalar repeat([0])");
    let out = s
        .repeat_t(&[2, 0])
        .expect("scalar repeat([2,0]) must succeed (CORE-054)");
    assert_empty(&out, &[2, 0], "scalar repeat([2,0])");
}

// torch: torch.empty(0).repeat(0) -> (0,); torch.empty(2,0).repeat(3,5) -> (6,0)
#[test]
fn repeat_zero_and_empty_inputs() {
    let e = plain(&[], &[0]);
    let out = e
        .repeat_t(&[0])
        .expect("empty.repeat([0]) must succeed (CORE-054)");
    assert_empty(&out, &[0], "empty repeat([0])");

    let e2 = plain(&[], &[2, 0]);
    let out = e2
        .repeat_t(&[3, 5])
        .expect("empty(2,0).repeat([3,5]) must succeed (CORE-054)");
    assert_empty(&out, &[6, 0], "empty(2,0) repeat([3,5])");
}

// torch: a=[1.,2.] (leaf); a.repeat(0) requires_grad=True, RepeatBackward0;
//        backward -> a.grad == [0., 0.]  (gradient FLOW per R-ORACLE-3)
#[test]
// reason: the gradient of a zero-repeat is exactly zero (no copies of the
// input survive in the output) — bit-exact equality is the right check.
#[allow(clippy::float_cmp)]
fn repeat_zero_tracked_grad_is_zeros() {
    let a = leaf(&[1.0, 2.0], &[2]);
    let out = a
        .repeat_t(&[0])
        .expect("tracked repeat([0]) must succeed (CORE-054)");
    assert_empty(&out, &[0], "tracked repeat([0])");
    assert!(
        out.requires_grad(),
        "torch keeps requires_grad=True on a zero-repeat output (RepeatBackward0)"
    );
    let go = plain(&[], &[0]);
    out.backward_with_gradient(&go).expect("backward");
    let g = a
        .grad()
        .unwrap()
        .expect("zero-repeat backward must deliver a (zero) grad to the leaf");
    assert_eq!(g.shape(), &[2], "grad shape");
    assert_eq!(
        g.data_vec().unwrap(),
        vec![0.0, 0.0],
        "torch: a.grad == [0., 0.]"
    );
}

// torch: bb=(2,3) view of leaf b; bb.repeat(0,2) -> (0,6) tracked;
//        backward -> b.grad == zeros(6)
#[test]
// reason: zero-repeat gradient is exactly zero — bit-exact equality.
#[allow(clippy::float_cmp)]
fn repeat_zero_2d_tracked_grad_is_zeros() {
    let b = leaf(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    let out = b
        .repeat_t(&[0, 2])
        .expect("tracked repeat([0,2]) must succeed (CORE-054)");
    assert_empty(&out, &[0, 6], "tracked repeat([0,2])");
    assert!(out.requires_grad(), "zero-repeat output must stay tracked");
    let go = plain(&[], &[0, 6]);
    out.backward_with_gradient(&go).expect("backward");
    let g = b
        .grad()
        .unwrap()
        .expect("zero-repeat backward must deliver a (zero) grad to the leaf");
    assert_eq!(g.shape(), &[2, 3], "grad shape");
    assert_eq!(
        g.data_vec().unwrap(),
        vec![0.0; 6],
        "torch: b.grad == zeros(2,3)"
    );
}

// torch: tile(x,(0,)) -> (0,); tile(y,(0,)) -> (2,0); tile(y,(2,0)) -> (4,0)
#[test]
fn tile_zero_delegates_correctly() {
    let x = plain(&[1.0, 2.0, 3.0], &[3]);
    let out = x
        .tile_t(&[0])
        .expect("tile(x,(0,)) must succeed (CORE-054)");
    assert_empty(&out, &[0], "tile(x,(0,))");

    let y = plain(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    let out = y
        .tile_t(&[0])
        .expect("tile(y,(0,)) must succeed (CORE-054)");
    assert_empty(&out, &[2, 0], "tile(y,(0,)) — reps left-padded with 1");
    let out = y
        .tile_t(&[2, 0])
        .expect("tile(y,(2,0)) must succeed (CORE-054)");
    assert_empty(&out, &[4, 0], "tile(y,(2,0))");
}
