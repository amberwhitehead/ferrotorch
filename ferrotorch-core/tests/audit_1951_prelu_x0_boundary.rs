//! Regression test for crosslink #1951 (discovered mid-fix during #1739).
//!
//! `PReluBackward` routed `x == 0` to the passthrough branch (`grad * 1`).
//! PyTorch's `_prelu_kernel_backward` uses the strict comparison
//! `dx = x > 0 ? grad_out : weight * grad_out`, so `x == 0` takes the
//! WEIGHT branch (`grad * alpha`).
//!
//! Live torch oracle (R-ORACLE-1(b), torch 2.11.0+cu130, CPU):
//! ```python
//! >>> t = torch.tensor([-2.0,-0.5,0.0,0.5,2.0,4.0], dtype=torch.float64,
//! ...                  requires_grad=True)
//! >>> alpha = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
//! >>> F.prelu(t, alpha).sum().backward()
//! >>> t.grad
//! tensor([0.25, 0.25, 0.25, 1.0, 1.0, 1.0])   # index 2 (x==0) is alpha
//! >>> alpha.grad
//! tensor([-2.5])                              # x==0 contributes 0
//! ```
//!
//! The conformance prelu fixtures never caught this because their inputs
//! are all-negative (see #1951) — this test pins the boundary explicitly.

use ferrotorch_core::grad_fns::activation::prelu;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::{Tensor, TensorStorage};

#[test]
fn prelu_backward_x0_takes_weight_branch_cpu() {
    let x = Tensor::from_storage(
        TensorStorage::cpu(vec![-2.0f64, -0.5, 0.0, 0.5, 2.0, 4.0]),
        vec![6],
        true,
    )
    .unwrap();
    let alpha = Tensor::from_storage(TensorStorage::cpu(vec![0.25f64]), vec![1], true).unwrap();
    let out = prelu(&x, &alpha).expect("prelu forward");
    // Forward at x==0 is 0 under both conventions.
    assert_eq!(
        out.data_vec().unwrap(),
        vec![-0.5, -0.125, 0.0, 0.5, 2.0, 4.0]
    );

    reduce_sum(&out).unwrap().backward().unwrap();
    let gx = x.grad().unwrap().expect("grad_x");
    let gx_bits: Vec<u64> = gx.data_vec().unwrap().iter().map(|v| v.to_bits()).collect();
    let expected: Vec<u64> = [0.25f64, 0.25, 0.25, 1.0, 1.0, 1.0]
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(
        gx_bits,
        expected,
        "x==0 must take the weight branch: grad = alpha = 0.25 (got {:?})",
        gx.data_vec().unwrap()
    );

    let ga = alpha.grad().unwrap().expect("grad_alpha");
    assert_eq!(
        ga.data_vec().unwrap(),
        vec![-2.5],
        "grad_alpha must exclude the x==0 cell (contributes x*g = 0)"
    );
}
