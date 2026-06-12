//! Red-then-green regression tests for audit finding CORE-066 (crosslink
//! #1760, CLASS-S): the nested dense conversions and the nested attention
//! helper silently detach component graphs. The design
//! (`.design/ferrotorch-core/nested.md`) sells the components-list layout
//! on "component-level autograd graph independence", yet `to_padded`,
//! `from_padded`, and `nested_scaled_dot_product_attention` all built
//! fresh `requires_grad = false` outputs on every path.
//!
//! Post-fix contract: when grad is enabled and an input tracks gradients,
//! the conversions compose from the differentiable primitives
//! (cat/unsqueeze for padding; narrow/contiguous/reshape for unpadding;
//! mm_bt/mul/softmax/matmul for attention) so gradients FLOW to the
//! original leaves (R-ORACLE-3 — flags alone prove nothing).
//!
//! Every numerical expectation is quoted from a LIVE torch session
//! (torch 2.11.0+cu130, R-ORACLE-1(b)); snippets are pasted per test.
//! `torch.nested.as_nested_tensor` preserves component autograd history
//! and `torch.nested.to_padded_tensor` is differentiable — verified live:
//!
//! ```python
//! >>> t1 = torch.tensor([[1.,2.],[3.,4.],[5.,6.]], requires_grad=True)
//! >>> t2 = torch.tensor([[7.,8.]], requires_grad=True)
//! >>> nt = torch.nested.as_nested_tensor([t1, t2])
//! >>> padded = torch.nested.to_padded_tensor(nt, 0.0)
//! >>> padded.requires_grad
//! True
//! ```

use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::nested::{
    NestedTensor, PackedNestedTensor, nested_scaled_dot_product_attention,
};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn leaf_f32(data: &[f32], shape: &[usize], rg: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), rg).unwrap()
}

fn leaf_f64(data: &[f64], shape: &[usize], rg: bool) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), rg).unwrap()
}

/// `to_padded` must preserve per-component gradient flow.
///
/// torch oracle:
/// ```python
/// >>> t1 = torch.tensor([[1.,2.],[3.,4.],[5.,6.]], requires_grad=True)
/// >>> t2 = torch.tensor([[7.,8.]], requires_grad=True)
/// >>> nt = torch.nested.as_nested_tensor([t1, t2])
/// >>> padded = torch.nested.to_padded_tensor(nt, 0.0)
/// >>> w = torch.arange(1., 13.).reshape(2,3,2)
/// >>> (padded * w).sum().backward()
/// >>> t1.grad.flatten().tolist(), t2.grad.flatten().tolist()
/// ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0])
/// ```
#[test]
// reason: the backward of to_padded is a pure gather of the cotangent (the
// weights) — no arithmetic — so gradients are bit-identical to the weights.
#[allow(clippy::float_cmp)]
fn core066_to_padded_component_gradients_cpu() {
    let t1 = leaf_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true);
    let t2 = leaf_f32(&[7.0, 8.0], &[1, 2], true);
    let nt = NestedTensor::new(vec![t1.clone(), t2.clone()], 0).unwrap();

    let padded = nt.to_padded(0.0).unwrap();
    assert_eq!(padded.shape(), &[2, 3, 2]);
    assert!(
        padded.requires_grad(),
        "torch.nested.to_padded_tensor is differentiable; a detached padded \
         tensor breaks variable-length training"
    );

    let w: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let w = leaf_f32(&w, &[2, 3, 2], false);
    let loss = sum(&mul(&padded, &w).unwrap()).unwrap();
    loss.backward().unwrap();

    // R-ORACLE-3: gradient FLOW — values reaching the original leaves.
    let g1 = t1.grad().unwrap().expect("t1.grad present");
    assert_eq!(
        g1.data_vec().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "torch oracle t1.grad"
    );
    let g2 = t2.grad().unwrap().expect("t2.grad present");
    assert_eq!(
        g2.data_vec().unwrap(),
        vec![7.0, 8.0],
        "torch oracle t2.grad"
    );
}

/// A non-zero pad value is a CONSTANT — gradients to the components are
/// unchanged (same torch oracle as above; the pad value does not appear
/// in any component's gradient path).
#[test]
#[allow(clippy::float_cmp)] // reason: pure cotangent gather, bit-exact.
fn core066_to_padded_grad_independent_of_pad_value() {
    let t1 = leaf_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true);
    let t2 = leaf_f32(&[7.0, 8.0], &[1, 2], true);
    let nt = NestedTensor::new(vec![t1.clone(), t2.clone()], 0).unwrap();

    let padded = nt.to_padded(-3.5).unwrap();
    let w: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let w = leaf_f32(&w, &[2, 3, 2], false);
    sum(&mul(&padded, &w).unwrap()).unwrap().backward().unwrap();

    let g2 = t2.grad().unwrap().expect("t2.grad present");
    assert_eq!(
        g2.data_vec().unwrap(),
        vec![7.0, 8.0],
        "pad slots are constants; component grads are pad-value-independent"
    );
}

/// `from_padded` must keep components connected to the padded source.
///
/// torch oracle (narrow-based equivalent; `padded[b, :len]` is the same
/// differentiable slicing torch.nested performs):
/// ```python
/// >>> padded2 = torch.arange(1., 13.).reshape(2,3,2).clone().requires_grad_(True)
/// >>> c0 = padded2[0, :3]; c1 = padded2[1, :1]
/// >>> w0 = torch.tensor([[1.,2.],[3.,4.],[5.,6.]]); w1 = torch.tensor([[7.,8.]])
/// >>> ((c0*w0).sum() + (c1*w1).sum()).backward()
/// >>> padded2.grad.flatten().tolist()
/// [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0]
/// ```
#[test]
#[allow(clippy::float_cmp)] // reason: pure cotangent scatter, bit-exact.
fn core066_from_padded_gradient_reaches_padded_source_cpu() {
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let padded = leaf_f32(&data, &[2, 3, 2], true);

    let nt = NestedTensor::from_padded(&padded, &[3, 1], 0).unwrap();
    assert!(
        nt.tensors()[0].requires_grad(),
        "components sliced from a grad-tracking padded tensor must stay connected"
    );

    let w0 = leaf_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);
    let w1 = leaf_f32(&[7.0, 8.0], &[1, 2], false);
    let l0 = sum(&mul(&nt.tensors()[0], &w0).unwrap()).unwrap();
    let l1 = sum(&mul(&nt.tensors()[1], &w1).unwrap()).unwrap();
    let loss = add(&l0, &l1).unwrap();
    loss.backward().unwrap();

    let g = padded.grad().unwrap().expect("padded.grad present");
    assert_eq!(
        g.data_vec().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0],
        "torch oracle padded.grad — zeros in the pad region"
    );
}

/// Round-trip `to_padded ∘ from_padded` keeps gradient flow end to end.
#[test]
#[allow(clippy::float_cmp)] // reason: pure cotangent gather, bit-exact.
fn core066_round_trip_gradient_flow_cpu() {
    let t1 = leaf_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true);
    let t2 = leaf_f32(&[7.0, 8.0], &[1, 2], true);
    let nt = NestedTensor::new(vec![t1.clone(), t2.clone()], 0).unwrap();
    let padded = nt.to_padded(0.0).unwrap();
    let back = NestedTensor::from_padded(&padded, &[3, 1], 0).unwrap();

    let w1 = leaf_f32(&[7.0, 8.0], &[1, 2], false);
    let loss = sum(&mul(&back.tensors()[1], &w1).unwrap()).unwrap();
    loss.backward().unwrap();

    let g2 = t2
        .grad()
        .unwrap()
        .expect("t2.grad present after round-trip");
    assert_eq!(g2.data_vec().unwrap(), vec![7.0, 8.0]);
    // t1 was not part of the loss — its grad is absent or all-zero.
    if let Some(g1) = t1.grad().unwrap() {
        assert_eq!(g1.data_vec().unwrap(), vec![0.0; 6]);
    }
}

/// The attention helper must be differentiable end to end.
///
/// torch oracle (float64):
/// ```python
/// >>> q = torch.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.]], dtype=torch.float64, requires_grad=True)
/// >>> k = torch.tensor([[1.,1.,0.,0.],[0.,0.,1.,1.],[1.,0.,1.,0.]], dtype=torch.float64, requires_grad=True)
/// >>> v = torch.tensor([[1.,2.],[3.,4.],[5.,6.]], dtype=torch.float64, requires_grad=True)
/// >>> out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
/// >>> out.flatten().tolist()
/// [3.3555882856328187, 4.35558828563282, 2.698089612856696, 3.698089612856696]
/// >>> (out * torch.tensor([[1.,2.],[3.,4.]], dtype=torch.float64)).sum().backward()
/// >>> q.grad.flatten().tolist()
/// [0.14618338559658664, -0.968389242780179, 0.9683892427801779, -0.1461833855965878,
///  -0.405399549421784, -2.280162568912073, 2.2801625689120715, 0.4053995494217827]
/// >>> k.grad.flatten().tolist()
/// [-0.968389242780179, -2.280162568912073, -0.968389242780179, -2.280162568912073,
///  -0.1461833855965878, 0.4053995494217827, -0.1461833855965878, 0.4053995494217827,
///  1.1145726283767656, 1.8747630194902887, 1.1145726283767656, 1.8747630194902887]
/// >>> v.grad.flatten().tolist()
/// [1.4250238126328492, 2.082744162884597, 1.4250238126328492, 2.082744162884597,
///  1.149952374734302, 1.8345116742308067]
/// ```
#[test]
fn core066_attention_gradients_cpu_f64() {
    let q = leaf_f64(&[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], &[2, 4], true);
    let k = leaf_f64(
        &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        &[3, 4],
        true,
    );
    let v = leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true);

    let qn = NestedTensor::new(vec![q.clone()], 0).unwrap();
    let kn = NestedTensor::new(vec![k.clone()], 0).unwrap();
    let vn = NestedTensor::new(vec![v.clone()], 0).unwrap();

    let out = nested_scaled_dot_product_attention(&qn, &kn, &vn).unwrap();
    let out0 = &out.tensors()[0];
    assert!(
        out0.requires_grad(),
        "attention over grad-tracking q/k/v must stay in the graph"
    );

    // Forward parity first.
    let got = out0.data_vec().unwrap();
    let want_out = [
        3.3555882856328187,
        4.35558828563282,
        2.698089612856696,
        3.698089612856696,
    ];
    for (i, (g, w)) in got.iter().zip(want_out.iter()).enumerate() {
        // f64 eps over 4-term dots + 3-way softmax + 3-term weighted sums.
        assert!((g - w).abs() < 1e-12, "out[{i}]: got {g}, want {w}");
    }

    let w = leaf_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    sum(&mul(out0, &w).unwrap()).unwrap().backward().unwrap();

    let want_q = [
        0.14618338559658664,
        -0.968389242780179,
        0.9683892427801779,
        -0.1461833855965878,
        -0.405399549421784,
        -2.280162568912073,
        2.2801625689120715,
        0.4053995494217827,
    ];
    let want_k = [
        -0.968389242780179,
        -2.280162568912073,
        -0.968389242780179,
        -2.280162568912073,
        -0.1461833855965878,
        0.4053995494217827,
        -0.1461833855965878,
        0.4053995494217827,
        1.1145726283767656,
        1.8747630194902887,
        1.1145726283767656,
        1.8747630194902887,
    ];
    let want_v = [
        1.4250238126328492,
        2.082744162884597,
        1.4250238126328492,
        2.082744162884597,
        1.149952374734302,
        1.8345116742308067,
    ];
    let gq = q
        .grad()
        .unwrap()
        .expect("q.grad present")
        .data_vec()
        .unwrap();
    let gk = k
        .grad()
        .unwrap()
        .expect("k.grad present")
        .data_vec()
        .unwrap();
    let gv = v
        .grad()
        .unwrap()
        .expect("v.grad present")
        .data_vec()
        .unwrap();
    for (i, (g, w)) in gq.iter().zip(want_q.iter()).enumerate() {
        // f64 eps through softmax backward (products of O(1) terms) — 1e-12.
        assert!((g - w).abs() < 1e-12, "q.grad[{i}]: got {g}, want {w}");
    }
    for (i, (g, w)) in gk.iter().zip(want_k.iter()).enumerate() {
        assert!((g - w).abs() < 1e-12, "k.grad[{i}]: got {g}, want {w}");
    }
    for (i, (g, w)) in gv.iter().zip(want_v.iter()).enumerate() {
        assert!((g - w).abs() < 1e-12, "v.grad[{i}]: got {g}, want {w}");
    }
}

/// Non-tracking inputs keep producing detached outputs (no spurious
/// graph nodes, matching torch's behavior with requires_grad=False).
#[test]
fn core066_non_tracking_inputs_stay_detached() {
    let t1 = leaf_f32(&[1.0, 2.0], &[1, 2], false);
    let nt = NestedTensor::new(vec![t1], 0).unwrap();
    let padded = nt.to_padded(0.0).unwrap();
    assert!(!padded.requires_grad());
    let back = NestedTensor::from_padded(&padded, &[1], 0).unwrap();
    assert!(!back.tensors()[0].requires_grad());
}

/// PackedNestedTensor is documented as a no-autograd storage layout; its
/// constructor must reject grad-tracking components LOUDLY rather than
/// silently detaching them (R-LOUD-3), OR preserve flow. The shipped
/// contract is the structured error (path (a) for the conversions above;
/// the packed layout remains autograd-free by design).
#[test]
fn core066_packed_from_nested_does_not_silently_detach() {
    let t1 = leaf_f32(&[1.0, 2.0], &[1, 2], true);
    let nt = NestedTensor::new(vec![t1], 0).unwrap();
    let r = PackedNestedTensor::from_nested(&nt);
    assert!(
        r.is_err(),
        "from_nested on grad-tracking components must error loudly (the packed \
         layout drops graphs by design), got silent detach: {r:?}"
    );
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the CORE-066 regression suite");
        });
    }

    /// True CUDA leaf (CORE-012 idiom: upload detached, then mark leaf).
    fn cuda_leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
            .unwrap()
            .to(Device::Cuda(0))
            .expect("upload to cuda:0")
            .requires_grad_(true)
    }

    fn assert_cuda<T: ferrotorch_core::Float>(t: &Tensor<T>, what: &str) {
        assert_eq!(
            t.device(),
            Device::Cuda(0),
            "{what} expected on Cuda(0) but resides on {:?} — silent CPU fallback",
            t.device()
        );
    }

    /// Same torch oracle as the CPU to_padded test; R-ORACLE-3 asserts
    /// the padded result AND the gradients live on Cuda(0).
    #[test]
    #[allow(clippy::float_cmp)] // reason: pure cotangent gather, bit-exact.
    fn core066_to_padded_component_gradients_cuda() {
        ensure_cuda_backend();
        let t1 = cuda_leaf_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let t2 = cuda_leaf_f32(&[7.0, 8.0], &[1, 2]);
        let nt = NestedTensor::new(vec![t1.clone(), t2.clone()], 0).unwrap();

        let padded = nt.to_padded(0.0).unwrap();
        assert_cuda(&padded, "to_padded result");
        assert!(
            padded.requires_grad(),
            "CUDA padded tensor must stay in graph"
        );

        let wdata: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let w = Tensor::from_storage(TensorStorage::cpu(wdata), vec![2, 3, 2], false)
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        sum(&mul(&padded, &w).unwrap()).unwrap().backward().unwrap();

        let g1 = t1.grad().unwrap().expect("t1.grad present");
        assert_cuda(&g1, "t1.grad");
        assert_eq!(g1.data_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let g2 = t2.grad().unwrap().expect("t2.grad present");
        assert_cuda(&g2, "t2.grad");
        assert_eq!(g2.data_vec().unwrap(), vec![7.0, 8.0]);
    }

    /// Same torch oracle as the CPU from_padded test, on CUDA.
    #[test]
    #[allow(clippy::float_cmp)] // reason: pure cotangent scatter, bit-exact.
    fn core066_from_padded_gradient_reaches_padded_source_cuda() {
        ensure_cuda_backend();
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let padded = cuda_leaf_f32(&data, &[2, 3, 2]);

        let nt = NestedTensor::from_padded(&padded, &[3, 1], 0).unwrap();
        assert_cuda(&nt.tensors()[0], "from_padded component 0");
        assert!(nt.tensors()[0].requires_grad());

        let w0 = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![3, 2],
            false,
        )
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
        let w1 = Tensor::from_storage(TensorStorage::cpu(vec![7.0f32, 8.0]), vec![1, 2], false)
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let l0 = sum(&mul(&nt.tensors()[0], &w0).unwrap()).unwrap();
        let l1 = sum(&mul(&nt.tensors()[1], &w1).unwrap()).unwrap();
        add(&l0, &l1).unwrap().backward().unwrap();

        let g = padded.grad().unwrap().expect("padded.grad present");
        assert_cuda(&g, "padded.grad");
        assert_eq!(
            g.data_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    /// Attention with grad-tracking CUDA inputs INSIDE the flash regime
    /// (d ≤ 128): the non-differentiable flash kernel must not swallow
    /// the graph. Same torch f64 oracle as the CPU test; f32 here, so
    /// tolerance 1e-5 (f32 eps 1.2e-7 over 4-term dots, a 3-way softmax,
    /// and 3-term weighted sums of O(1..6) magnitudes).
    #[test]
    fn core066_attention_gradients_cuda_f32() {
        ensure_cuda_backend();
        let q = cuda_leaf_f32(&[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], &[2, 4]);
        let k = cuda_leaf_f32(
            &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            &[3, 4],
        );
        let v = cuda_leaf_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

        let qn = NestedTensor::new(vec![q.clone()], 0).unwrap();
        let kn = NestedTensor::new(vec![k.clone()], 0).unwrap();
        let vn = NestedTensor::new(vec![v.clone()], 0).unwrap();
        let out = nested_scaled_dot_product_attention(&qn, &kn, &vn).unwrap();
        let out0 = &out.tensors()[0];
        assert_cuda(out0, "attention output");
        assert!(out0.requires_grad(), "graph must survive the CUDA dispatch");

        let want_out = [
            3.3555882856328187f64,
            4.35558828563282,
            2.698089612856696,
            3.698089612856696,
        ];
        let got = out0.data_vec().unwrap();
        for (i, (g, w)) in got.iter().zip(want_out.iter()).enumerate() {
            assert!(
                (*g as f64 - w).abs() < 1e-5,
                "out[{i}]: got {g}, torch oracle {w}"
            );
        }

        let w = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
            vec![2, 2],
            false,
        )
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
        sum(&mul(out0, &w).unwrap()).unwrap().backward().unwrap();

        let want_q = [
            0.14618338559658664f64,
            -0.968389242780179,
            0.9683892427801779,
            -0.1461833855965878,
            -0.405399549421784,
            -2.280162568912073,
            2.2801625689120715,
            0.4053995494217827,
        ];
        let want_v = [
            1.4250238126328492f64,
            2.082744162884597,
            1.4250238126328492,
            2.082744162884597,
            1.149952374734302,
            1.8345116742308067,
        ];
        let gq = q.grad().unwrap().expect("q.grad present");
        assert_cuda(&gq, "q.grad");
        for (i, (g, w)) in gq.data_vec().unwrap().iter().zip(want_q.iter()).enumerate() {
            assert!(
                (*g as f64 - w).abs() < 1e-5,
                "q.grad[{i}]: got {g}, torch oracle {w}"
            );
        }
        let gv = v.grad().unwrap().expect("v.grad present");
        assert_cuda(&gv, "v.grad");
        for (i, (g, w)) in gv.data_vec().unwrap().iter().zip(want_v.iter()).enumerate() {
            assert!(
                (*g as f64 - w).abs() < 1e-5,
                "v.grad[{i}]: got {g}, torch oracle {w}"
            );
        }
        let gk = k.grad().unwrap().expect("k.grad present");
        assert_cuda(&gk, "k.grad");
    }
}
