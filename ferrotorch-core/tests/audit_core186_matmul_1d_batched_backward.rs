//! Red-then-green regression tests for audit finding CORE-186 (crosslink
//! #1880): 1-D × batched matmul backward returns wrong gradients (vector
//! operand: cross-batch contamination) or errors outright (the batched
//! operand: `swap_last_two` rejects ndim < 2).
//!
//! Every numerical expectation below is quoted from a LIVE torch session
//! (torch 2.11.0+cu130, R-ORACLE-1 path (b)); the generating snippet is
//! pasted in a comment next to each expected block.
//!
//! Coverage: all four 1-D × batched arrangements (1D@3D, 3D@1D, 1D@4D,
//! 4D@1D), both gradients each, f32 + f64, CPU + CUDA (gpu feature lane).

use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;

fn leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}
fn leaf_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}
fn t_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn t_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

// Tolerances: every operand/cotangent below is a small integer, every
// expected gradient is an integer of magnitude <= 1370, and the contraction
// depth is <= 24 terms — all values and partial sums are exact in the 24-bit
// f32 mantissa (< 2^24), so the only rounding is the final sum ordering:
// 1e-3 absolute headroom on f32, 1e-9 on f64 (53-bit mantissa, fully exact).
const TOL_F32: f32 = 1e-3;
const TOL_F64: f64 = 1e-9;

fn assert_close_f32(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= TOL_F32,
            "{label}: index {i}: got {a}, torch oracle {e} (diff {})",
            (a - e).abs()
        );
    }
}
fn assert_close_f64(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= TOL_F64,
            "{label}: index {i}: got {a}, torch oracle {e} (diff {})",
            (a - e).abs()
        );
    }
}

// ===========================================================================
// 1D @ 3D — the exact audit probe (ones cotangent)
// ===========================================================================

// torch oracle (torch 2.11.0+cu130):
//   >>> a = torch.tensor([1.,2.], requires_grad=True)
//   >>> b = torch.arange(1.,13.).reshape(2,2,3).requires_grad_(True)
//   >>> c = a @ b
//   >>> c.backward(torch.ones_like(c))
//   >>> c     # shape [2,3]: [9., 12., 15., 27., 30., 33.]
//   >>> a.grad
//   tensor([30., 48.])
//   >>> b.grad
//   tensor([[[1., 1., 1.], [2., 2., 2.]], [[1., 1., 1.], [2., 2., 2.]]])
#[test]
fn core186_1d_at_3d_ones_cotangent_f32() {
    let a = leaf_f32(&[1.0, 2.0], &[2]);
    let b_data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let b = leaf_f32(&b_data, &[2, 2, 3]);
    let c = a.matmul(&b).expect("forward 1D@3D");
    assert_eq!(c.shape(), &[2, 3]);
    assert_close_f32(
        c.data().unwrap(),
        &[9.0, 12.0, 15.0, 27.0, 30.0, 33.0],
        "1D@3D fwd",
    );
    // loss = sum(c) gives the ones cotangent of the audit probe.
    let loss = sum(&c).expect("sum");
    loss.backward().expect("backward 1D@3D");
    let ga = a.grad().unwrap().expect("grad_a present");
    let gb = b.grad().unwrap().expect("grad_b present");
    // Audit probe: pre-fix this returns [60, 96] (batch-count contamination)
    // where torch returns [30, 48].
    assert_close_f32(ga.data().unwrap(), &[30.0, 48.0], "1D@3D grad_a");
    assert_close_f32(
        gb.data().unwrap(),
        &[
            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, //
            1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
        ],
        "1D@3D grad_b",
    );
}

// Same probe with ONLY `a` requiring grad: pre-fix this is the SILENT
// wrong-gradient path from the audit — backward succeeds but returns
// [60., 96.] (cross-batch contamination) instead of torch's [30., 48.].
// (With both leaves tracked the error from `swap_last_two(a)` masks it.)
#[test]
fn core186_1d_at_3d_ones_cotangent_vector_grad_only_f32() {
    let a = leaf_f32(&[1.0, 2.0], &[2]);
    let b_data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let b = t_f32(&b_data, &[2, 2, 3]); // requires_grad = false
    let c = a.matmul(&b).expect("forward 1D@3D");
    let loss = sum(&c).expect("sum");
    loss.backward().expect("backward 1D@3D (grad_a only)");
    let ga = a.grad().unwrap().expect("grad_a present");
    // torch oracle: tensor([30., 48.]) — see snippet above.
    assert_close_f32(ga.data().unwrap(), &[30.0, 48.0], "1D@3D grad_a only");
}

// ===========================================================================
// 1D @ 3D — non-uniform cotangent (cross-contamination detector)
// ===========================================================================

// torch oracle (torch 2.11.0+cu130):
//   >>> a = torch.tensor([1.,2.], requires_grad=True)
//   >>> b = torch.arange(1.,13.).reshape(2,2,3).requires_grad_(True)
//   >>> (a @ b).backward(torch.arange(1.,7.).reshape(2,3))
//   >>> a.grad
//   tensor([136., 199.])
//   >>> b.grad
//   tensor([[[ 1., 2., 3.], [ 2., 4., 6.]], [[ 4., 5., 6.], [ 8., 10., 12.]]])
#[test]
fn core186_1d_at_3d_weighted_f32() {
    let a = leaf_f32(&[1.0, 2.0], &[2]);
    let b_data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let b = leaf_f32(&b_data, &[2, 2, 3]);
    let c = a.matmul(&b).expect("forward 1D@3D");
    let w_data: Vec<f32> = (1..=6).map(|v| v as f32).collect();
    let w = t_f32(&w_data, &[2, 3]);
    let loss = sum(&c.mul_t(&w).expect("weight")).expect("sum");
    loss.backward().expect("backward 1D@3D weighted");
    let ga = a.grad().unwrap().expect("grad_a present");
    let gb = b.grad().unwrap().expect("grad_b present");
    assert_close_f32(ga.data().unwrap(), &[136.0, 199.0], "1D@3D w grad_a");
    assert_close_f32(
        gb.data().unwrap(),
        &[
            1.0, 2.0, 3.0, 2.0, 4.0, 6.0, //
            4.0, 5.0, 6.0, 8.0, 10.0, 12.0,
        ],
        "1D@3D w grad_b",
    );
}

#[test]
fn core186_1d_at_3d_weighted_f64() {
    let a = leaf_f64(&[1.0, 2.0], &[2]);
    let b_data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
    let b = leaf_f64(&b_data, &[2, 2, 3]);
    let c = a.matmul(&b).expect("forward 1D@3D");
    let w_data: Vec<f64> = (1..=6).map(|v| v as f64).collect();
    let w = t_f64(&w_data, &[2, 3]);
    let loss = sum(&c.mul_t(&w).expect("weight")).expect("sum");
    loss.backward().expect("backward 1D@3D weighted");
    let ga = a.grad().unwrap().expect("grad_a present");
    let gb = b.grad().unwrap().expect("grad_b present");
    assert_close_f64(ga.data().unwrap(), &[136.0, 199.0], "1D@3D w grad_a");
    assert_close_f64(
        gb.data().unwrap(),
        &[
            1.0, 2.0, 3.0, 2.0, 4.0, 6.0, //
            4.0, 5.0, 6.0, 8.0, 10.0, 12.0,
        ],
        "1D@3D w grad_b",
    );
}

// ===========================================================================
// 3D @ 1D
// ===========================================================================

// torch oracle (torch 2.11.0+cu130):
//   >>> a = torch.arange(1.,13.).reshape(2,2,3).requires_grad_(True)
//   >>> v = torch.tensor([1.,2.,3.], requires_grad=True)
//   >>> c = a @ v   # shape [2,2]: [14., 32., 50., 68.]
//   >>> c.backward(torch.arange(1.,5.).reshape(2,2))
//   >>> a.grad
//   tensor([[[ 1., 2., 3.], [ 2., 4., 6.]], [[ 3., 6., 9.], [ 4., 8., 12.]]])
//   >>> v.grad
//   tensor([70., 80., 90.])
#[test]
fn core186_3d_at_1d_weighted_f32() {
    let a_data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let a = leaf_f32(&a_data, &[2, 2, 3]);
    let v = leaf_f32(&[1.0, 2.0, 3.0], &[3]);
    let c = a.matmul(&v).expect("forward 3D@1D");
    assert_eq!(c.shape(), &[2, 2]);
    assert_close_f32(c.data().unwrap(), &[14.0, 32.0, 50.0, 68.0], "3D@1D fwd");
    let w = t_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let loss = sum(&c.mul_t(&w).expect("weight")).expect("sum");
    loss.backward().expect("backward 3D@1D weighted");
    let ga = a.grad().unwrap().expect("grad_a present");
    let gv = v.grad().unwrap().expect("grad_v present");
    assert_close_f32(
        ga.data().unwrap(),
        &[
            1.0, 2.0, 3.0, 2.0, 4.0, 6.0, //
            3.0, 6.0, 9.0, 4.0, 8.0, 12.0,
        ],
        "3D@1D w grad_a",
    );
    assert_close_f32(gv.data().unwrap(), &[70.0, 80.0, 90.0], "3D@1D w grad_v");
}

#[test]
fn core186_3d_at_1d_weighted_f64() {
    let a_data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
    let a = leaf_f64(&a_data, &[2, 2, 3]);
    let v = leaf_f64(&[1.0, 2.0, 3.0], &[3]);
    let c = a.matmul(&v).expect("forward 3D@1D");
    let w = t_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let loss = sum(&c.mul_t(&w).expect("weight")).expect("sum");
    loss.backward().expect("backward 3D@1D weighted");
    let ga = a.grad().unwrap().expect("grad_a present");
    let gv = v.grad().unwrap().expect("grad_v present");
    assert_close_f64(
        ga.data().unwrap(),
        &[
            1.0, 2.0, 3.0, 2.0, 4.0, 6.0, //
            3.0, 6.0, 9.0, 4.0, 8.0, 12.0,
        ],
        "3D@1D w grad_a",
    );
    assert_close_f64(gv.data().unwrap(), &[70.0, 80.0, 90.0], "3D@1D w grad_v");
}

// ===========================================================================
// 1D @ 4D
// ===========================================================================

// torch oracle (torch 2.11.0+cu130):
//   >>> v = torch.tensor([1.,2.], requires_grad=True)
//   >>> b = torch.arange(1.,25.).reshape(2,2,2,3).requires_grad_(True)
//   >>> c = v @ b   # shape [2,2,3]: [9.,12.,15.,27.,30.,33.,45.,48.,51.,63.,66.,69.]
//   >>> c.backward(torch.arange(1.,13.).reshape(2,2,3))
//   >>> v.grad
//   tensor([1136., 1370.])
//   >>> b.grad.flatten()
//   tensor([ 1.,  2.,  3.,  2.,  4.,  6.,  4.,  5.,  6.,  8., 10., 12.,
//            7.,  8.,  9., 14., 16., 18., 10., 11., 12., 20., 22., 24.])
#[test]
fn core186_1d_at_4d_weighted_f32() {
    let v = leaf_f32(&[1.0, 2.0], &[2]);
    let b_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let b = leaf_f32(&b_data, &[2, 2, 2, 3]);
    let c = v.matmul(&b).expect("forward 1D@4D");
    assert_eq!(c.shape(), &[2, 2, 3]);
    assert_close_f32(
        c.data().unwrap(),
        &[
            9.0, 12.0, 15.0, 27.0, 30.0, 33.0, //
            45.0, 48.0, 51.0, 63.0, 66.0, 69.0,
        ],
        "1D@4D fwd",
    );
    let w_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let w = t_f32(&w_data, &[2, 2, 3]);
    let loss = sum(&c.mul_t(&w).expect("weight")).expect("sum");
    loss.backward().expect("backward 1D@4D weighted");
    let gv = v.grad().unwrap().expect("grad_v present");
    let gb = b.grad().unwrap().expect("grad_b present");
    assert_close_f32(gv.data().unwrap(), &[1136.0, 1370.0], "1D@4D w grad_v");
    assert_close_f32(
        gb.data().unwrap(),
        &[
            1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, //
            7.0, 8.0, 9.0, 14.0, 16.0, 18.0, 10.0, 11.0, 12.0, 20.0, 22.0, 24.0,
        ],
        "1D@4D w grad_b",
    );
}

#[test]
fn core186_1d_at_4d_weighted_f64() {
    let v = leaf_f64(&[1.0, 2.0], &[2]);
    let b_data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
    let b = leaf_f64(&b_data, &[2, 2, 2, 3]);
    let c = v.matmul(&b).expect("forward 1D@4D");
    let w_data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let w = t_f64(&w_data, &[2, 2, 3]);
    let loss = sum(&c.mul_t(&w).expect("weight")).expect("sum");
    loss.backward().expect("backward 1D@4D weighted");
    let gv = v.grad().unwrap().expect("grad_v present");
    let gb = b.grad().unwrap().expect("grad_b present");
    assert_close_f64(gv.data().unwrap(), &[1136.0, 1370.0], "1D@4D w grad_v");
    assert_close_f64(
        gb.data().unwrap(),
        &[
            1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, //
            7.0, 8.0, 9.0, 14.0, 16.0, 18.0, 10.0, 11.0, 12.0, 20.0, 22.0, 24.0,
        ],
        "1D@4D w grad_b",
    );
}

// ===========================================================================
// 4D @ 1D
// ===========================================================================

// torch oracle (torch 2.11.0+cu130):
//   >>> a = torch.arange(1.,25.).reshape(2,2,2,3).requires_grad_(True)
//   >>> v = torch.tensor([1.,2.,3.], requires_grad=True)
//   >>> c = a @ v   # shape [2,2,2]: [14., 32., 50., 68., 86., 104., 122., 140.]
//   >>> c.backward(torch.arange(1.,9.).reshape(2,2,2))
//   >>> a.grad.flatten()
//   tensor([ 1.,  2.,  3.,  2.,  4.,  6.,  3.,  6.,  9.,  4.,  8., 12.,
//            5., 10., 15.,  6., 12., 18.,  7., 14., 21.,  8., 16., 24.])
//   >>> v.grad
//   tensor([540., 576., 612.])
#[test]
fn core186_4d_at_1d_weighted_f32() {
    let a_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let a = leaf_f32(&a_data, &[2, 2, 2, 3]);
    let v = leaf_f32(&[1.0, 2.0, 3.0], &[3]);
    let c = a.matmul(&v).expect("forward 4D@1D");
    assert_eq!(c.shape(), &[2, 2, 2]);
    assert_close_f32(
        c.data().unwrap(),
        &[14.0, 32.0, 50.0, 68.0, 86.0, 104.0, 122.0, 140.0],
        "4D@1D fwd",
    );
    let w_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let w = t_f32(&w_data, &[2, 2, 2]);
    let loss = sum(&c.mul_t(&w).expect("weight")).expect("sum");
    loss.backward().expect("backward 4D@1D weighted");
    let ga = a.grad().unwrap().expect("grad_a present");
    let gv = v.grad().unwrap().expect("grad_v present");
    assert_close_f32(
        ga.data().unwrap(),
        &[
            1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0, //
            5.0, 10.0, 15.0, 6.0, 12.0, 18.0, 7.0, 14.0, 21.0, 8.0, 16.0, 24.0,
        ],
        "4D@1D w grad_a",
    );
    assert_close_f32(gv.data().unwrap(), &[540.0, 576.0, 612.0], "4D@1D w grad_v");
}

#[test]
fn core186_4d_at_1d_weighted_f64() {
    let a_data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
    let a = leaf_f64(&a_data, &[2, 2, 2, 3]);
    let v = leaf_f64(&[1.0, 2.0, 3.0], &[3]);
    let c = a.matmul(&v).expect("forward 4D@1D");
    let w_data: Vec<f64> = (1..=8).map(|x| x as f64).collect();
    let w = t_f64(&w_data, &[2, 2, 2]);
    let loss = sum(&c.mul_t(&w).expect("weight")).expect("sum");
    loss.backward().expect("backward 4D@1D weighted");
    let ga = a.grad().unwrap().expect("grad_a present");
    let gv = v.grad().unwrap().expect("grad_v present");
    assert_close_f64(
        ga.data().unwrap(),
        &[
            1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0, //
            5.0, 10.0, 15.0, 6.0, 12.0, 18.0, 7.0, 14.0, 21.0, 8.0, 16.0, 24.0,
        ],
        "4D@1D w grad_a",
    );
    assert_close_f64(gv.data().unwrap(), &[540.0, 576.0, 612.0], "4D@1D w grad_v");
}

// ===========================================================================
// GPU lane — same four arrangements, f32 + f64, ones cotangent, with
// result-device AND gradient-device assertions (R-ORACLE-3).
// ===========================================================================

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the GPU lane of this suite");
        });
    }

    fn read_back_f32(t: &Tensor<f32>) -> Vec<f32> {
        let cpu = t.cpu().expect("D2H readback");
        cpu.data().expect("read CPU data").to_vec()
    }
    fn read_back_f64(t: &Tensor<f64>) -> Vec<f64> {
        let cpu = t.cpu().expect("D2H readback");
        cpu.data().expect("read CPU data").to_vec()
    }

    // torch oracle (torch 2.11.0+cu130, identical on cuda:0):
    //   same snippets as the CPU tests above with .cuda() inputs.
    #[test]
    fn core186_gpu_1d_at_3d_ones_cotangent_f32() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let a = leaf_f32(&[1.0, 2.0], &[2]).to(dev).expect("upload a");
        let b_data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        let b = leaf_f32(&b_data, &[2, 2, 3]).to(dev).expect("upload b");
        let c = a.matmul(&b).expect("forward 1D@3D cuda");
        assert_eq!(c.shape(), &[2, 3]);
        let loss = sum(&c).expect("sum");
        loss.backward().expect("backward 1D@3D cuda");
        let ga = a.grad().unwrap().expect("grad_a present");
        let gb = b.grad().unwrap().expect("grad_b present");
        // R-ORACLE-3: gradients must be device-resident, not CPU demotions.
        assert_eq!(ga.device(), dev, "grad_a must be CUDA-resident");
        assert_eq!(gb.device(), dev, "grad_b must be CUDA-resident");
        assert_close_f32(&read_back_f32(&ga), &[30.0, 48.0], "gpu 1D@3D grad_a");
        assert_close_f32(
            &read_back_f32(&gb),
            &[
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, //
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
            ],
            "gpu 1D@3D grad_b",
        );
    }

    #[test]
    fn core186_gpu_3d_at_1d_ones_cotangent_f32() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let a_data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        let a = leaf_f32(&a_data, &[2, 2, 3]).to(dev).expect("upload a");
        let v = leaf_f32(&[1.0, 2.0, 3.0], &[3]).to(dev).expect("upload v");
        let c = a.matmul(&v).expect("forward 3D@1D cuda");
        assert_eq!(c.shape(), &[2, 2]);
        let loss = sum(&c).expect("sum");
        loss.backward().expect("backward 3D@1D cuda");
        let ga = a.grad().unwrap().expect("grad_a present");
        let gv = v.grad().unwrap().expect("grad_v present");
        assert_eq!(ga.device(), dev, "grad_a must be CUDA-resident");
        assert_eq!(gv.device(), dev, "grad_v must be CUDA-resident");
        // torch: a.grad = ones(2,2) outer v per batch row = [[1,2,3]]*4;
        //        v.grad = sum over batch+rows of a = [22., 26., 30.]
        //   >>> a = torch.arange(1.,13.).reshape(2,2,3).requires_grad_(True)
        //   >>> v = torch.tensor([1.,2.,3.], requires_grad=True)
        //   >>> (a @ v).backward(torch.ones(2,2))
        //   >>> a.grad.flatten()  # [1.,2.,3.] x4
        //   >>> v.grad            # tensor([22., 26., 30.])
        assert_close_f32(
            &read_back_f32(&ga),
            &[
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, //
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
            ],
            "gpu 3D@1D grad_a",
        );
        assert_close_f32(&read_back_f32(&gv), &[22.0, 26.0, 30.0], "gpu 3D@1D grad_v");
    }

    #[test]
    fn core186_gpu_1d_at_4d_ones_cotangent_f64() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let v = leaf_f64(&[1.0, 2.0], &[2]).to(dev).expect("upload v");
        let b_data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let b = leaf_f64(&b_data, &[2, 2, 2, 3]).to(dev).expect("upload b");
        let c = v.matmul(&b).expect("forward 1D@4D cuda");
        assert_eq!(c.shape(), &[2, 2, 3]);
        let loss = sum(&c).expect("sum");
        loss.backward().expect("backward 1D@4D cuda");
        let gv = v.grad().unwrap().expect("grad_v present");
        let gb = b.grad().unwrap().expect("grad_b present");
        assert_eq!(gv.device(), dev, "grad_v must be CUDA-resident");
        assert_eq!(gb.device(), dev, "grad_b must be CUDA-resident");
        // torch:
        //   >>> v = torch.tensor([1.,2.], requires_grad=True)
        //   >>> b = torch.arange(1.,25.).reshape(2,2,2,3).requires_grad_(True)
        //   >>> (v @ b).backward(torch.ones(2,2,3))
        //   >>> v.grad   # tensor([132., 168.])
        //   >>> b.grad   # rows alternate [1,1,1] / [2,2,2]
        assert_close_f64(&read_back_f64(&gv), &[132.0, 168.0], "gpu 1D@4D grad_v");
        assert_close_f64(
            &read_back_f64(&gb),
            &[
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, //
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
            ],
            "gpu 1D@4D grad_b",
        );
    }

    #[test]
    fn core186_gpu_4d_at_1d_ones_cotangent_f64() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let a_data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let a = leaf_f64(&a_data, &[2, 2, 2, 3]).to(dev).expect("upload a");
        let v = leaf_f64(&[1.0, 2.0, 3.0], &[3]).to(dev).expect("upload v");
        let c = a.matmul(&v).expect("forward 4D@1D cuda");
        assert_eq!(c.shape(), &[2, 2, 2]);
        let loss = sum(&c).expect("sum");
        loss.backward().expect("backward 4D@1D cuda");
        let ga = a.grad().unwrap().expect("grad_a present");
        let gv = v.grad().unwrap().expect("grad_v present");
        assert_eq!(ga.device(), dev, "grad_a must be CUDA-resident");
        assert_eq!(gv.device(), dev, "grad_v must be CUDA-resident");
        // torch:
        //   >>> a = torch.arange(1.,25.).reshape(2,2,2,3).requires_grad_(True)
        //   >>> v = torch.tensor([1.,2.,3.], requires_grad=True)
        //   >>> (a @ v).backward(torch.ones(2,2,2))
        //   >>> a.grad.flatten()  # [1.,2.,3.] x8
        //   >>> v.grad            # tensor([92., 100., 108.])
        assert_close_f64(
            &read_back_f64(&ga),
            &[
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, //
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
            ],
            "gpu 4D@1D grad_a",
        );
        assert_close_f64(
            &read_back_f64(&gv),
            &[92.0, 100.0, 108.0],
            "gpu 4D@1D grad_v",
        );
    }
}
