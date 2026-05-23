//! Phase 3d sentinel (GPU dtype-parity epic, crosslink #1187): the BACKWARD
//! pass of the masked ops is GPU-resident. Forward was made resident in Phase
//! 3c; here the VJPs of `masked_fill`, `where_cond`, and `masked_select` run on
//! CUDA through real PTX kernels with a GPU-resident `BoolTensor` mask — NO CPU
//! round trip for the mask or the gradients.
//!
//! What this probe asserts, for masked_fill / where_cond / masked_select:
//!   (a) Each resulting grad tensor `.is_cuda()` (the VJP stays resident).
//!   (b) Grad values match a CPU reference computed independently.
//! Plus the attention case: where_cond grad routing to BOTH x and y, and an
//! end-to-end `.backward()` through the autograd graph for masked_select
//! (resident grad accumulated into the GPU leaf's `.grad()`).
//!
//! The kernels exercised:
//!   - MaskedFillBackward  → backend.masked_fill_dt(grad, mask, 0)   (resident)
//!   - WhereCondBackward   → backend.where_cond(cond, grad, zeros)   (resident)
//!   - MaskedSelectBackward→ backend.masked_scatter(grad, mask, n)   (NEW kernel)
//!
//! The only host crossing in the op paths is the masked_select forward's
//! output-length integer (the data-dependent SHAPE, PyTorch parity). The probe
//! pulls grads to host ONLY for value assertions (explicit `.to(Cpu)`), which is
//! the test reading the answer, not the backward detouring through host.
//!
//! Prints a PASS/FAIL table ending `PASS: N, FAIL: 0`. Requires the `gpu`
//! feature + a real CUDA device (run on the host RTX 3090).

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Tensor;
use ferrotorch_core::autograd::graph::backward_with_grad;
use ferrotorch_core::bool_tensor::BoolTensor;
use ferrotorch_core::device::Device;
use ferrotorch_core::grad_fns::indexing::{
    MaskedFillBackward, MaskedSelectBackward, WhereCondBackward,
};
use ferrotorch_core::ops::indexing::masked_select;
use ferrotorch_core::tensor::GradFn;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialise for Phase 3d masked-backward probe");
    });
}

fn record(label: &str, ok: bool, detail: &str, pass: &mut usize, fail: &mut usize) {
    if ok {
        *pass += 1;
        println!("PASS  {label:<46} {detail}");
    } else {
        *fail += 1;
        println!("FAIL  {label:<46} {detail}");
    }
}

/// GPU f32 tensor with requires_grad.
fn gpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    ferrotorch_core::creation::from_slice::<f32>(data, shape)
        .unwrap()
        .requires_grad_(true)
        .to(Device::Cuda(0))
        .unwrap()
}

/// GPU f32 tensor without requires_grad (e.g. a grad_output seed).
fn gpu_f32_plain(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    ferrotorch_core::creation::from_slice::<f32>(data, shape)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
}

fn gpu_bool(data: &[bool], shape: &[usize]) -> BoolTensor {
    BoolTensor::from_slice(data, shape)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
}

fn read(t: &Tensor<f32>) -> Vec<f32> {
    t.to(Device::Cpu).unwrap().data_vec().unwrap()
}

fn approx_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(&x, &y)| {
            x == y
                || (x.is_infinite()
                    && y.is_infinite()
                    && x.is_sign_negative() == y.is_sign_negative())
                || (x - y).abs() < 1e-5
        })
}

/// MaskedFillBackward: grad_input[i] = mask[i] ? 0 : grad_output[i].
fn check_masked_fill_backward(pass: &mut usize, fail: &mut usize) {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mask_h = [false, true, false, true, true];
    let go = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let shape = [5];

    let expected: Vec<f32> = mask_h
        .iter()
        .zip(&go)
        .map(|(&m, &g)| if m { 0.0 } else { g })
        .collect();

    let input_g = gpu_f32(&input, &shape);
    let mask_g = gpu_bool(&mask_h, &shape);
    let grad_fn = MaskedFillBackward {
        input: input_g.clone(),
        mask: mask_g,
    };
    let go_g = gpu_f32_plain(&go, &shape);
    let grads = grad_fn.backward(&go_g).expect("masked_fill backward gpu");
    let grad_input = grads[0].as_ref().unwrap();

    let resident = grad_input.is_cuda();
    let vals = read(grad_input);
    record(
        "masked_fill backward grad_input.is_cuda()",
        resident,
        &format!("is_cuda={resident}"),
        pass,
        fail,
    );
    record(
        "masked_fill backward values vs CPU ref",
        approx_eq(&vals, &expected),
        &format!("vals={vals:?} expected={expected:?}"),
        pass,
        fail,
    );
}

/// WhereCondBackward: grad_x[i] = cond[i] ? g[i] : 0 ; grad_y[i] = cond[i] ? 0 : g[i].
/// Both x and y require grad (the attention routing case).
fn check_where_cond_backward(pass: &mut usize, fail: &mut usize) {
    let x = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let y = [-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let cond_h = [true, false, true, false, true, false];
    let go = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
    let shape = [6];

    let expected_x: Vec<f32> = cond_h
        .iter()
        .zip(&go)
        .map(|(&c, &g)| if c { g } else { 0.0 })
        .collect();
    let expected_y: Vec<f32> = cond_h
        .iter()
        .zip(&go)
        .map(|(&c, &g)| if c { 0.0 } else { g })
        .collect();

    let x_g = gpu_f32(&x, &shape);
    let y_g = gpu_f32(&y, &shape);
    let cond_g = gpu_bool(&cond_h, &shape);
    let grad_fn = WhereCondBackward {
        x: x_g.clone(),
        y: y_g.clone(),
        condition: cond_g,
    };
    let go_g = gpu_f32_plain(&go, &shape);
    let grads = grad_fn.backward(&go_g).expect("where_cond backward gpu");
    let grad_x = grads[0].as_ref().unwrap();
    let grad_y = grads[1].as_ref().unwrap();

    let res_x = grad_x.is_cuda();
    let res_y = grad_y.is_cuda();
    record(
        "where_cond backward grad_x/grad_y both .is_cuda()",
        res_x && res_y,
        &format!("grad_x.is_cuda={res_x} grad_y.is_cuda={res_y}"),
        pass,
        fail,
    );
    let vx = read(grad_x);
    let vy = read(grad_y);
    record(
        "where_cond backward grad_x vs CPU ref",
        approx_eq(&vx, &expected_x),
        &format!("vals={vx:?} expected={expected_x:?}"),
        pass,
        fail,
    );
    record(
        "where_cond backward grad_y vs CPU ref",
        approx_eq(&vy, &expected_y),
        &format!("vals={vy:?} expected={expected_y:?}"),
        pass,
        fail,
    );
}

/// MaskedSelectBackward: scatter the compacted grad back into a zeros tensor of
/// input.numel() at the true mask positions (NEW masked_scatter kernel).
fn check_masked_select_backward(pass: &mut usize, fail: &mut usize) {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mask_h = [true, false, true, true, false, false, true];
    let shape = [7];
    // Compacted grad_output (length = #true = 4): distinct values to verify the
    // scatter lands them at the right input positions.
    let go_compact = [11.0f32, 33.0, 44.0, 77.0];

    // CPU reference: grad_input[i] = go_compact[j++] where mask[i], else 0.
    let mut expected = vec![0.0f32; 7];
    let mut j = 0usize;
    for (i, &m) in mask_h.iter().enumerate() {
        if m {
            expected[i] = go_compact[j];
            j += 1;
        }
    }

    let input_g = gpu_f32(&input, &shape);
    let mask_g = gpu_bool(&mask_h, &shape);
    let grad_fn = MaskedSelectBackward {
        input: input_g.clone(),
        mask: mask_g,
    };
    let go_g = gpu_f32_plain(&go_compact, &[go_compact.len()]);
    let grads = grad_fn.backward(&go_g).expect("masked_select backward gpu");
    let grad_input = grads[0].as_ref().unwrap();

    let resident = grad_input.is_cuda();
    let shape_ok = grad_input.shape() == [7];
    let vals = read(grad_input);
    record(
        "masked_select backward grad_input.is_cuda()",
        resident,
        &format!("is_cuda={resident} shape={:?}", grad_input.shape()),
        pass,
        fail,
    );
    record(
        "masked_select backward values vs CPU ref",
        shape_ok && approx_eq(&vals, &expected),
        &format!("vals={vals:?} expected={expected:?}"),
        pass,
        fail,
    );
}

/// End-to-end: forward masked_select on a GPU leaf, then `.backward()` through
/// the autograd graph; the leaf's accumulated `.grad()` must be resident and
/// match the scatter reference. Confirms the grad_fn is wired into the graph
/// (not just callable directly) and stays on-device through accumulation.
fn check_masked_select_e2e_graph(pass: &mut usize, fail: &mut usize) {
    let input = [2.0f32, 4.0, 6.0, 8.0];
    let mask_h = [true, false, true, false];
    let shape = [4];

    let input_g = gpu_f32(&input, &shape);
    let mask_g = gpu_bool(&mask_h, &shape);
    let selected = masked_select(&input_g, &mask_g).expect("masked_select forward");
    // selected = [2, 6]. Seed an explicit GPU grad_output = [1, 1] and run the
    // graph backward; the scatter lands [1,1] back -> [1, 0, 1, 0] at the leaf.
    let seed = gpu_f32_plain(&[1.0, 1.0], &[selected.shape()[0]]);
    backward_with_grad(&selected, Some(&seed)).expect("graph backward");

    let grad = input_g.grad().expect("grad result").expect("grad present");
    let resident = grad.is_cuda();
    let vals = read(&grad);
    let expected = [1.0f32, 0.0, 1.0, 0.0];
    record(
        "masked_select e2e .backward() leaf grad resident + correct",
        resident && approx_eq(&vals, &expected),
        &format!("is_cuda={resident} vals={vals:?} expected={expected:?}"),
        pass,
        fail,
    );
}

#[test]
fn probe_phase3d_masked_backward() {
    ensure_cuda_backend();

    let mut pass = 0usize;
    let mut fail = 0usize;

    println!("── 3d masked_fill backward (resident masked_fill_dt) ──");
    check_masked_fill_backward(&mut pass, &mut fail);
    println!("── 3d where_cond backward (resident where_cond, x+y) ──");
    check_where_cond_backward(&mut pass, &mut fail);
    println!("── 3d masked_select backward (NEW masked_scatter) ─────");
    check_masked_select_backward(&mut pass, &mut fail);
    println!("── 3d masked_select end-to-end .backward() graph ──────");
    check_masked_select_e2e_graph(&mut pass, &mut fail);

    println!("───────────────────────────────────────────────────────");
    println!("PASS: {pass}, FAIL: {fail}");
    assert_eq!(fail, 0, "Phase 3d masked-backward probe had failures");
}
