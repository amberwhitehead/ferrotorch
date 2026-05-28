//! ACToR discriminator re-audit of commit `09ffca9c0` (#1603) — SHAPE-INVISIBLE
//! x1<->x2 / i<->j swap detector for `Bilinear::forward_pair`.
//!
//! The dispatch flagged "mis-associating which input maps to in1 vs in2". When
//! in1 != in2 a swap is caught by a shape error. The hard case is in1 == in2
//! (square feature axes): a swap of the operands, or a transpose of the einsum
//! equations (`bi,oij->boj` vs `bj,oij->boi`), is SHAPE-INVISIBLE and only
//! detectable by VALUES — and only when the weight is non-symmetric in i/j and
//! x1 != x2.
//!
//! EXPECTED is LIVE torch 2.11.0+cu130 (driver inline), NOT from ferrotorch
//! (R-CHAR-3). The swapped torch call is shown to DIFFER, proving the chosen
//! weight is non-symmetric so the test genuinely discriminates a swap.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::Bilinear;
use ferrotorch_nn::module::Module;

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: len mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{ctx}: element {i}: ferrotorch={a} torch={e} (|d|={})",
            (a - e).abs()
        );
    }
}

/// torch driver:
///   W  = tensor([0.5,1.0,2.0,3.0, -1.0,0.25,0.75,-0.5]).reshape(2,2,2)  # out=2,in1=2,in2=2
///   b  = tensor([0.1,-0.2])
///   x1 = tensor([[1.,2.],[3.,4.]])   # (2, in1=2)
///   x2 = tensor([[5.,6.],[7.,8.]])   # (2, in2=2), distinct from x1
///   torch.nn.functional.bilinear(x1,x2,W,b) == [[64.6,-2.2],[186.6,-10.2]]
///   torch.nn.functional.bilinear(x2,x1,W,b) == [[60.6,-4.2],[182.6,-12.2]]  (differs => non-symmetric)
#[test]
fn divergence_1603_bilinear_shape_invisible_swap() {
    let mut bl = Bilinear::<f32>::new(2, 2, 2, true).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut bl);
        params[0].set_data(t(&[0.5, 1.0, 2.0, 3.0, -1.0, 0.25, 0.75, -0.5], &[2, 2, 2]));
        params[1].set_data(t(&[0.1, -0.2], &[2]));
    }
    let x1 = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let x2 = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let y = bl.forward_pair(&x1, &x2).unwrap();
    assert_eq!(y.shape(), &[2, 2]);
    // If ferrotorch swapped operands it would yield [60.6,-4.2,182.6,-12.2].
    assert_close(
        y.data().unwrap(),
        &[64.599998, -2.2, 186.600006, -10.2],
        1e-3,
        "shape-invisible swap (in1==in2, non-symmetric W)",
    );
}
