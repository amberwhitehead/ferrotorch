//! Red-then-green regression tests for audit finding CORE-013 (crosslink
//! #1707): the physical materialization paths of `to_memory_format` /
//! `contiguous_in` (GPU strided-copy fast path AND host fallback) construct
//! fresh tensors with `grad_fn: None`, `is_leaf: true`, and `requires_grad`
//! copied from the input — differentiable-looking but disconnected from the
//! input graph. The gradient of a memory-format change is the identity on
//! logical values, so a backward edge must be attached.
//!
//! Every expectation below is quoted from a LIVE torch session
//! (torch 2.11.0+cu130, RTX 3090; R-ORACLE-1 path (b)):
//!
//! ```text
//! >>> w = torch.arange(1., 17.).reshape(2,2,2,2)
//! >>> lf = torch.arange(16.).reshape(2,2,2,2).requires_grad_(True)
//! >>> y = lf.to(memory_format=torch.channels_last)
//! >>> y.is_leaf, type(y.grad_fn).__name__
//! (False, 'ToCopyBackward0')
//! >>> (y * w).sum().backward(); torch.equal(lf.grad, w)
//! True                                  # identity gradient
//! >>> zc = (lf * 2).contiguous(memory_format=torch.channels_last)
//! >>> type(zc.grad_fn).__name__         # 'CloneBackward0'; lf.grad == 2*w
//! >>> y2 = y.contiguous(memory_format=torch.contiguous_format)
//! >>> type(y2.grad_fn).__name__         # 'CloneBackward0'; lf.grad == w
//! >>> with torch.no_grad(): yn = lf.to(memory_format=torch.channels_last)
//! >>> yn.requires_grad, yn.is_leaf, yn.grad_fn
//! (False, True, None)
//! >>> # channels_last_3d: grad_fn ToCopyBackward0, grads identity;
//! >>> # on cuda the leaf grads are cuda:0-resident (torch.equal vs w: True)
//! ```
//!
//! Tolerance justification (R-ORACLE-5): NONE — all values are small
//! integers exactly representable in f32; exact equality throughout.

use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{MemoryFormat, Tensor};

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;

fn leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn plain_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn iota(n: usize, start: f32) -> Vec<f32> {
    (0..n).map(|i| start + i as f32).collect()
}

// torch oracle: y = lf.to(memory_format=channels_last) -> is_leaf False,
// grad_fn present; (y*w).sum().backward() -> lf.grad == w (identity).
#[test]
fn core013_cpu_channels_last_leaf_grad_identity() {
    let lf = leaf_f32(&iota(16, 0.0), &[2, 2, 2, 2]);
    let y = lf
        .to_memory_format(MemoryFormat::ChannelsLast)
        .expect("to_memory_format");
    assert!(y.requires_grad(), "requires_grad must propagate");
    assert!(
        !y.is_leaf(),
        "materialized format conversion must be a NON-leaf (torch: is_leaf=False) — CORE-013"
    );
    assert!(
        y.grad_fn().is_some(),
        "materialized format conversion must carry a grad_fn (torch: ToCopyBackward0) — CORE-013"
    );
    let w = plain_f32(&iota(16, 1.0), &[2, 2, 2, 2]);
    // CPU elementwise kernels still reject non-contiguous (channels-last)
    // operands with a structured error — open issue #1826 (CORE-132),
    // Phase-2 scope. Hop through the graph-preserving `.contiguous()`
    // (ContiguousBackward, identity) so this test pins ONLY the CORE-013
    // mechanism: gradient flow through MemoryFormatBackward to the leaf.
    let yc = y.contiguous().expect("graph-preserving contiguous hop");
    let loss = sum(&mul(&yc, &w).expect("y*w")).expect("sum");
    loss.backward().expect("backward");
    let g = lf
        .grad()
        .expect("grad access")
        .expect("source leaf must receive the gradient (CORE-013)");
    assert_eq!(
        g.data().unwrap(),
        &iota(16, 1.0)[..],
        "torch oracle: lf.grad == w (identity gradient through the layout change)"
    );
}

// torch oracle: l3.to(memory_format=channels_last_3d) -> grad_fn present;
// grads identity.
#[test]
fn core013_cpu_channels_last_3d_leaf_grad_identity() {
    let lf = leaf_f32(&iota(16, 0.0), &[1, 2, 2, 2, 2]);
    let y = lf
        .to_memory_format(MemoryFormat::ChannelsLast3d)
        .expect("to_memory_format 3d");
    assert!(
        !y.is_leaf() && y.grad_fn().is_some(),
        "ChannelsLast3d materialization must stay connected (CORE-013)"
    );
    let w = plain_f32(&iota(16, 1.0), &[1, 2, 2, 2, 2]);
    // `.contiguous()` hop: CPU elementwise rejects channels-last operands
    // (#1826, CORE-132); ContiguousBackward is identity, so the CORE-013
    // gradient path is still the one under test.
    let yc = y.contiguous().expect("graph-preserving contiguous hop");
    let loss = sum(&mul(&yc, &w).expect("y*w")).expect("sum");
    loss.backward().expect("backward");
    let g = lf.grad().expect("grad access").expect("leaf grad present");
    assert_eq!(g.data().unwrap(), &iota(16, 1.0)[..]);
}

// Non-leaf input through `contiguous_in`. torch oracle:
// zc = (lf*2).contiguous(memory_format=channels_last) -> CloneBackward0;
// (zc*w).sum().backward() -> lf.grad == 2*w.
#[test]
fn core013_cpu_contiguous_in_nonleaf_chain() {
    let lf = leaf_f32(&iota(16, 0.0), &[2, 2, 2, 2]);
    let two = plain_f32(&[2.0; 16], &[2, 2, 2, 2]);
    let z = mul(&lf, &two).expect("lf*2");
    let zc = z
        .contiguous_in(MemoryFormat::ChannelsLast)
        .expect("contiguous_in");
    assert!(
        zc.grad_fn().is_some(),
        "contiguous_in on a non-leaf must keep the graph (torch: CloneBackward0) — CORE-013"
    );
    let w = plain_f32(&iota(16, 1.0), &[2, 2, 2, 2]);
    // `.contiguous()` hop: CPU elementwise rejects channels-last operands
    // (#1826, CORE-132); ContiguousBackward is identity, so the CORE-013
    // gradient path is still the one under test.
    let zcc = zc.contiguous().expect("graph-preserving contiguous hop");
    let loss = sum(&mul(&zcc, &w).expect("zc*w")).expect("sum");
    loss.backward().expect("backward");
    let g = lf.grad().expect("grad access").expect("leaf grad present");
    let expected: Vec<f32> = iota(16, 1.0).iter().map(|v| 2.0 * v).collect();
    assert_eq!(
        g.data().unwrap(),
        &expected[..],
        "torch oracle: lf.grad == 2*w through the non-leaf chain"
    );
}

// Contiguous-format materialization (round trip back from channels-last).
// torch oracle: y2 = y.contiguous(memory_format=contiguous_format) ->
// CloneBackward0; backward -> lf.grad == w.
#[test]
fn core013_cpu_contiguous_roundtrip_grad_identity() {
    let lf = leaf_f32(&iota(16, 0.0), &[2, 2, 2, 2]);
    let y = lf
        .to_memory_format(MemoryFormat::ChannelsLast)
        .expect("to channels_last");
    let y2 = y
        .to_memory_format(MemoryFormat::Contiguous)
        .expect("back to contiguous");
    assert!(
        y2.grad_fn().is_some(),
        "Contiguous materialization of a tracked channels-last tensor must keep the graph"
    );
    let w = plain_f32(&iota(16, 1.0), &[2, 2, 2, 2]);
    let loss = sum(&mul(&y2, &w).expect("y2*w")).expect("sum");
    loss.backward().expect("backward");
    let g = lf.grad().expect("grad access").expect("leaf grad present");
    assert_eq!(g.data().unwrap(), &iota(16, 1.0)[..]);
}

// torch no_grad oracle: requires_grad False, is_leaf True, grad_fn None —
// never a bare copied flag on a fresh leaf (R-LOUD-3).
#[test]
fn core013_cpu_no_grad_does_not_track() {
    let lf = leaf_f32(&iota(16, 0.0), &[2, 2, 2, 2]);
    let y = ferrotorch_core::autograd::no_grad::no_grad(|| {
        lf.to_memory_format(MemoryFormat::ChannelsLast)
    })
    .expect("to_memory_format under no_grad");
    assert!(
        !y.requires_grad(),
        "under no_grad the conversion must not require grad (torch: False)"
    );
    assert!(y.is_leaf());
    assert!(y.grad_fn().is_none());
}

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

    fn cuda_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        plain_f32(data, shape)
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(true)
    }

    // torch oracle (cuda): y = lc.to(memory_format=channels_last) ->
    // ToCopyBackward0, device cuda:0; backward -> lc.grad cuda:0, == w.
    #[test]
    fn core013_gpu_channels_last_leaf_grad_identity() {
        ensure_cuda_backend();
        let lf = cuda_leaf(&iota(16, 0.0), &[2, 2, 2, 2]);
        let y = lf
            .to_memory_format(MemoryFormat::ChannelsLast)
            .expect("to_memory_format cuda");
        assert_eq!(y.device(), Device::Cuda(0), "conversion must stay CUDA");
        assert!(
            !y.is_leaf() && y.grad_fn().is_some(),
            "CUDA format materialization must stay connected (CORE-013)"
        );
        let w = plain_f32(&iota(16, 1.0), &[2, 2, 2, 2])
            .to(Device::Cuda(0))
            .expect("upload w");
        let loss = sum(&mul(&y, &w).expect("y*w")).expect("sum");
        loss.backward().expect("backward");
        let g = lf
            .grad()
            .expect("grad access")
            .expect("CUDA leaf must receive the gradient (CORE-013)");
        assert_eq!(
            g.device(),
            Device::Cuda(0),
            "leaf grad must be CUDA-resident, like torch's"
        );
        assert_eq!(
            g.cpu().expect("D2H").data().unwrap(),
            &iota(16, 1.0)[..],
            "torch oracle: lc.grad == w"
        );
    }

    // torch oracle (cuda, 5D): channels_last_3d grads identity, cuda:0.
    #[test]
    fn core013_gpu_channels_last_3d_leaf_grad_identity() {
        ensure_cuda_backend();
        let lf = cuda_leaf(&iota(16, 0.0), &[1, 2, 2, 2, 2]);
        let y = lf
            .to_memory_format(MemoryFormat::ChannelsLast3d)
            .expect("to_memory_format 3d cuda");
        assert_eq!(y.device(), Device::Cuda(0));
        assert!(
            !y.is_leaf() && y.grad_fn().is_some(),
            "CUDA ChannelsLast3d materialization must stay connected (CORE-013)"
        );
        let w = plain_f32(&iota(16, 1.0), &[1, 2, 2, 2, 2])
            .to(Device::Cuda(0))
            .expect("upload w");
        let loss = sum(&mul(&y, &w).expect("y*w")).expect("sum");
        loss.backward().expect("backward");
        let g = lf.grad().expect("grad access").expect("leaf grad present");
        assert_eq!(g.device(), Device::Cuda(0));
        assert_eq!(
            g.cpu().expect("D2H").data().unwrap(),
            &iota(16, 1.0)[..],
            "torch oracle: lc.grad == w3"
        );
    }

    // Non-leaf chain on CUDA: zc = (lf*2).contiguous_in(channels_last);
    // backward -> lf.grad == 2*w on cuda:0 (torch: CloneBackward0 chain).
    #[test]
    fn core013_gpu_contiguous_in_nonleaf_chain() {
        ensure_cuda_backend();
        let lf = cuda_leaf(&iota(16, 0.0), &[2, 2, 2, 2]);
        let two = plain_f32(&[2.0; 16], &[2, 2, 2, 2])
            .to(Device::Cuda(0))
            .expect("upload two");
        let z = mul(&lf, &two).expect("lf*2");
        let zc = z
            .contiguous_in(MemoryFormat::ChannelsLast)
            .expect("contiguous_in cuda");
        assert_eq!(zc.device(), Device::Cuda(0));
        assert!(
            zc.grad_fn().is_some(),
            "CUDA contiguous_in on a non-leaf must keep the graph (CORE-013)"
        );
        let w = plain_f32(&iota(16, 1.0), &[2, 2, 2, 2])
            .to(Device::Cuda(0))
            .expect("upload w");
        let loss = sum(&mul(&zc, &w).expect("zc*w")).expect("sum");
        loss.backward().expect("backward");
        let g = lf.grad().expect("grad access").expect("leaf grad present");
        assert_eq!(g.device(), Device::Cuda(0));
        let expected: Vec<f32> = iota(16, 1.0).iter().map(|v| 2.0 * v).collect();
        assert_eq!(g.cpu().expect("D2H").data().unwrap(), &expected[..]);
    }
}
