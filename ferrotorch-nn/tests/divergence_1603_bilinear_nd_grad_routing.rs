//! ACToR discriminator re-audit of commit `09ffca9c0` (#1603 Bilinear N-D
//! input via flatten-to-2D + reshape-back, `ferrotorch-nn/src/linear.rs`).
//!
//! The generator's in-crate tests cover 3-D forward, 3-D backward, and 4-D
//! forward. BUT the 3-D backward test seeds the output gradient with
//! `loss = y.sum()`, i.e. a UNIFORM upstream gradient of all-ones. A uniform
//! upstream gradient masks a leading-dim transpose / mis-routed reshape-back
//! in the gradient path: every batch position receives the same upstream grad,
//! so swapping positions during the flatten/unflatten leaves grad_x1/x2/W
//! unchanged. This file closes that gap.
//!
//! Every EXPECTED value below is the LIVE torch 2.11.0+cu130 output of the
//! matching `torch.nn.functional.bilinear` call with a NON-UNIFORM seeded
//! output gradient (`y.backward(go)` where `go = arange(...) * 0.1`), NOT
//! copied from the ferrotorch side (R-CHAR-3). The torch driver script that
//! produced each block is reproduced inline above the constants so the values
//! are regenerable.
//!
//! Design choices that make a transpose/swap divergence DETECTABLE:
//!   - Distinct feature sizes in1=3 != in2=2 (an i<->j swap changes shapes).
//!   - Non-symmetric weight W[o,i,j] (an i<->j swap changes values even when
//!     shapes happen to align).
//!   - NON-SQUARE leading dims (2,3) so a leading-dim transpose -> (3,2)
//!     yields a different element ordering after reshape-back.
//!   - NON-UNIFORM seeded output gradient so per-position grad routing matters.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::Bilinear;
use ferrotorch_nn::module::Module;

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, ctx: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{ctx}: length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{ctx}: element {i} diverges: ferrotorch={a} torch={e} (|d|={})",
            (a - e).abs()
        );
    }
}

/// Shared deterministic layer: in1=3, in2=2, out=2, NON-symmetric weight + bias.
/// These exact W/b values were fed to `torch.nn.functional.bilinear` to produce
/// every oracle constant in this file.
fn make_layer() -> Bilinear<f32> {
    let mut bl = Bilinear::<f32>::new(3, 2, 2, true).unwrap();
    // W[o,i,j], row-major flatten of [out=2, in1=3, in2=2].
    let weight: [f32; 12] = [
        0.1, 0.2, 0.3, -0.1, -0.2, 0.05, // o=0
        0.0, 0.4, -0.3, 0.2, 0.1, -0.15, // o=1
    ];
    let bias: [f32; 2] = [0.5, -0.25];
    {
        let mut params = Module::<f32>::parameters_mut(&mut bl);
        params[0].set_data(t(&weight, &[2, 3, 2]));
        params[1].set_data(t(&bias, &[2]));
    }
    bl
}

/// 3-D NON-UNIFORM-GRADIENT backward with NON-SQUARE leading dims (2,3).
///
/// torch driver:
///   W = tensor([0.1,0.2,0.3,-0.1,-0.2,0.05, 0.0,0.4,-0.3,0.2,0.1,-0.15]).reshape(2,3,2)
///   b = tensor([0.5,-0.25])
///   x1 = arange(1,19).float().reshape(2,3,3).requires_grad_()   # (2,3,in1=3)
///   x2 = (arange(1,13).float()*0.5).reshape(2,3,2).requires_grad_()  # (2,3,in2=2)
///   y  = F.bilinear(x1,x2,W,b)                                   # (2,3,out=2)
///   go = (arange(1, y.numel()+1).float()*0.1).reshape(y.shape)
///   y.backward(go)
///
/// The existing in-crate `test_bilinear_3d_backward_matches_torch` uses
/// `y.sum()` (uniform grad of ones) on square (2,2) leading dims — both
/// choices hide a per-position grad-routing transpose. This test removes both
/// masks.
#[test]
fn divergence_1603_bilinear_3d_nonuniform_grad_routing() {
    let bl = make_layer();

    let x1_data: Vec<f32> = (1..=18).map(|i| i as f32).collect();
    let x2_data: Vec<f32> = (1..=12).map(|i| i as f32 * 0.5).collect();
    let x1 = leaf(&x1_data, &[2, 3, 3]);
    let x2 = leaf(&x2_data, &[2, 3, 2]);

    let y = bl.forward_pair(&x1, &x2).unwrap();
    assert_eq!(y.shape(), &[2, 3, 2]);

    // FWD oracle (non-square leading dims): catches a leading-dim transpose
    // even in the forward pass.
    let fwd: [f32; 12] = [
        0.7, -0.05, 2.75, 1.8, 6.9, 5.150001, 13.150001, 9.999998, 21.500004, 16.349998,
        31.950001, 24.200001,
    ];
    assert_close(y.data().unwrap(), &fwd, 1e-4, "3D fwd (non-square leading)");

    // Seeded NON-UNIFORM output gradient: go = arange(1,13)*0.1, reshaped (2,3,2).
    let go_data: Vec<f32> = (1..=12).map(|i| i as f32 * 0.1).collect();
    let go = t(&go_data, &[2, 3, 2]);
    y.backward_with_gradient(&go).unwrap();

    let g_x1 = x1.grad().unwrap().expect("x1 grad");
    assert_eq!(g_x1.shape(), &[2, 3, 3]);
    let grad_x1: [f32; 18] = [
        0.105, 0.015, -0.025, 0.485, 0.055, -0.12, 1.145, 0.135, -0.295, 2.085, 0.255, -0.55,
        3.305, 0.415, -0.885, 4.805, 0.615, -1.3,
    ];
    assert_close(g_x1.data().unwrap(), &grad_x1, 1e-4, "3D grad_x1 (non-uniform)");

    let g_x2 = x2.grad().unwrap().expect("x2 grad");
    assert_eq!(g_x2.shape(), &[2, 3, 2]);
    let grad_x2: [f32; 12] = [
        -0.05, 0.085, -0.15, 0.86, -0.25, 2.355, -0.35, 4.57, -0.45, 7.505, -0.55, 11.160001,
    ];
    assert_close(g_x2.data().unwrap(), &grad_x2, 1e-4, "3D grad_x2 (non-uniform)");

    let g_w = bl.weight.grad().unwrap().expect("W grad");
    assert_eq!(g_w.shape(), &[2, 3, 2]);
    let grad_w: [f32; 12] = [
        184.550003, 205.100006, 198.850006, 221.200012, 213.150009, 237.300018, 205.100006,
        228.200012, 221.200012, 246.400009, 237.300003, 264.600006,
    ];
    assert_close(g_w.data().unwrap(), &grad_w, 2e-3, "3D grad_W (non-uniform)");

    let g_b = bl
        .bias
        .as_ref()
        .unwrap()
        .grad()
        .unwrap()
        .expect("bias grad");
    assert_eq!(g_b.shape(), &[2]);
    // bias grad sums the seeded output grad over ALL leading dims:
    // sum over o=0 positions = 0.1+0.3+0.5+0.7+0.9+1.1 = 3.6; o=1 = 4.2.
    assert_close(g_b.data().unwrap(), &[3.6, 4.2], 1e-4, "3D grad_bias (non-uniform)");
}

/// 4-D NON-UNIFORM-GRADIENT forward + backward (no 4-D backward exists in the
/// committed in-crate tests at all).
///
/// torch driver:
///   W = tensor([0.1,0.2,0.3,-0.1,-0.2,0.05, 0.0,0.4,-0.3,0.2,0.1,-0.15]).reshape(2,3,2)
///   b = tensor([0.5,-0.25])
///   x1 = (arange(1,25).float()*0.1).reshape(2,1,4,3).requires_grad_()  # last=in1=3
///   x2 = (arange(1,17).float()*0.2 - 0.3).reshape(2,1,4,2).requires_grad_() # last=in2=2
///   y  = F.bilinear(x1,x2,W,b)                                          # (2,1,4,2)
///   go = (arange(1, y.numel()+1).float()*0.1).reshape(y.shape)
///   y.backward(go)
#[test]
fn divergence_1603_bilinear_4d_nonuniform_grad_routing() {
    let bl = make_layer();

    let x1_data: Vec<f32> = (1..=24).map(|i| i as f32 * 0.1).collect();
    let x2_data: Vec<f32> = (1..=16).map(|i| i as f32 * 0.2 - 0.3).collect();
    let x1 = leaf(&x1_data, &[2, 1, 4, 3]);
    let x2 = leaf(&x2_data, &[2, 1, 4, 2]);

    let y = bl.forward_pair(&x1, &x2).unwrap();
    assert_eq!(y.shape(), &[2, 1, 4, 2]);

    let fwd: [f32; 16] = [
        0.5005, -0.2435, 0.551, -0.192, 0.6855, -0.0805, 0.904, 0.091, 1.2065, 0.3225, 1.593,
        0.614, 2.0635, 0.9655, 2.618, 1.377,
    ];
    assert_close(y.data().unwrap(), &fwd, 1e-4, "4D fwd");

    let go_data: Vec<f32> = (1..=16).map(|i| i as f32 * 0.1).collect();
    let go = t(&go_data, &[2, 1, 4, 2]);
    y.backward_with_gradient(&go).unwrap();

    let g_x1 = x1.grad().unwrap().expect("x1 grad");
    assert_eq!(g_x1.shape(), &[2, 1, 4, 3]);
    let grad_x1: [f32; 24] = [
        0.009, 0.006, -0.0025, 0.119, 0.016, -0.0285, 0.341, 0.042, -0.0865, 0.675, 0.084,
        -0.1765, 1.121, 0.142, -0.2985, 1.679, 0.216, -0.4525, 2.349, 0.306, -0.6385, 3.131,
        0.412, -0.8565,
    ];
    assert_close(g_x1.data().unwrap(), &grad_x1, 1e-4, "4D grad_x1");

    let g_x2 = x2.grad().unwrap().expect("x2 grad");
    assert_eq!(g_x2.shape(), &[2, 1, 4, 2]);
    let grad_x2: [f32; 16] = [
        -0.005, 0.0085, -0.015, 0.086, -0.025, 0.2355, -0.035, 0.457, -0.045, 0.7505, -0.055,
        1.116, -0.065, 1.5535, -0.075, 2.063,
    ];
    assert_close(g_x2.data().unwrap(), &grad_x2, 1e-4, "4D grad_x2");

    let g_w = bl.weight.grad().unwrap().expect("W grad");
    assert_eq!(g_w.shape(), &[2, 3, 2]);
    let grad_w: [f32; 12] = [
        20.740002, 22.716002, 21.908001, 24.012001, 23.076002, 25.308002, 22.440002, 24.600002,
        23.712002, 26.016003, 24.984001, 27.432003,
    ];
    assert_close(g_w.data().unwrap(), &grad_w, 2e-3, "4D grad_W");

    let g_b = bl
        .bias
        .as_ref()
        .unwrap()
        .grad()
        .unwrap()
        .expect("bias grad");
    assert_eq!(g_b.shape(), &[2]);
    assert_close(g_b.data().unwrap(), &[6.400001, 7.2], 1e-4, "4D grad_bias");
}

/// i<->j SWAP detector: distinct in1=3 != in2=2 with a non-symmetric weight.
/// If the implementation mis-associates which input maps to in1 vs in2 (or
/// transposes W's i/j axes), the contraction `sum_ij x1[i] W[o,i,j] x2[j]`
/// yields a different scalar. A single-sample 1-D pair keeps it minimal.
///
/// torch driver:
///   W = (...same as make_layer...)   # [2,3,2]
///   x1 = tensor([1.,2.,3.])          # in1=3
///   x2 = tensor([4.,5.])             # in2=2
///   F.bilinear(x1,x2,W,b)            # -> shape (2,)
#[test]
fn divergence_1603_bilinear_ij_swap_detector() {
    let bl = make_layer();
    let x1 = leaf(&[1.0, 2.0, 3.0], &[3]);
    let x2 = leaf(&[4.0, 5.0], &[2]);
    let y = bl.forward_pair(&x1, &x2).unwrap();
    assert_eq!(y.shape(), &[2]);
    // torch: o=0: 1*(0.1*4+0.2*5)+2*(0.3*4+-0.1*5)+3*(-0.2*4+0.05*5)+0.5
    //              = 1.4 + 2*0.7 + 3*(-0.55) + 0.5 = 1.4+1.4-1.65+0.5 = 1.65
    //        o=1: 1*(0*4+0.4*5)+2*(-0.3*4+0.2*5)+3*(0.1*4+-0.15*5)+(-0.25)
    //              = 2.0 + 2*(-0.2) + 3*(-0.35) - 0.25 = 2.0-0.4-1.05-0.25 = 0.3
    assert_close(y.data().unwrap(), &[1.65, 0.3], 1e-4, "ij-swap detector");
}
