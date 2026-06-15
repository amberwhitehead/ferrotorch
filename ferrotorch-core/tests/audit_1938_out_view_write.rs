//! #1938 regression suite — `out=` / in-place writes into sub-views must
//! write INTO the view's existing storage region, never swap the whole
//! shared buffer.
//!
//! Root cause (pinned by `divergence_core001_critic_out_view_swap.rs`):
//! `Tensor::update_storage` -> `TensorStorage::replace_buffer_aliased`
//! replaced the ENTIRE shared buffer with the freshly-computed result
//! buffer (length == view numel). When the target was a sub-view
//! (`storage_offset != 0` or `numel != storage_len`) or a non-contiguous
//! view, this shrank/reordered the shared storage and destroyed the base
//! tensor's other elements. Upstream
//! `pytorch aten/src/ATen/native/Resize.cpp:27` returns `false` (no
//! resize) on matched sizes — the TensorIterator writes elementwise into
//! `out`'s storage at its storage_offset, honoring `out`'s strides.
//!
//! Every expected value below is torch-derived (torch 2.11 semantics:
//! elementwise write through the view; the base keeps all elements the
//! view does not cover), NOT copied from ferrotorch's own output
//! (R-CHAR-3 / R-ORACLE-1).
//!
//! Tracking: #1938 (CORE-001 residual).

use ferrotorch_core::grad_fns::arithmetic::add_out;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn cpu_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

/// Non-contiguous (stride-2) `out` view: torch writes elementwise through
/// the strides.
///
/// ```python
/// # torch 2.11:
/// base = torch.tensor([1., 2., 3., 4., 5., 6.])
/// v = base.as_strided((3,), (2,))          # elements 0, 2, 4
/// torch.add(torch.tensor([10., 20., 30.]),
///           torch.tensor([100., 200., 300.]), out=v)
/// # v    -> [110., 220., 330.]
/// # base -> [110., 2., 220., 4., 330., 6.]
/// ```
#[test]
fn add_out_strided_view_writes_through_strides() {
    let base = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
    let v = base.stride_view(vec![3], vec![2], 0);
    let a = cpu_tensor(vec![10.0, 20.0, 30.0], vec![3]);
    let b = cpu_tensor(vec![100.0, 200.0, 300.0], vec![3]);

    add_out(&v, &a, &b).unwrap();

    assert_eq!(v.data_vec().unwrap(), &[110.0, 220.0, 330.0]);
    assert_eq!(
        base.data_vec().unwrap(),
        &[110.0, 2.0, 220.0, 4.0, 330.0, 6.0],
        "add_out into a stride-2 view must scatter through the view's strides"
    );
    assert_eq!(base.storage().len(), 6);
}

/// In-place broadcast path (`mul_`) on a contiguous sub-view: the
/// broadcast arm of `Tensor::mul_` routes through `update_storage`
/// (inplace.rs) with a fresh result buffer of the VIEW's numel — same
/// unguarded swap as `add_out`.
///
/// ```python
/// # torch 2.11:
/// base = torch.arange(1., 9.)              # [1..8]
/// v = base.as_strided((2, 2), (2, 1), 4)   # elements 5, 6, 7, 8
/// v.mul_(torch.tensor([[10., 100.]]))      # broadcast [1,2] -> [2,2]
/// # base -> [1., 2., 3., 4., 50., 600., 70., 800.]
/// ```
#[test]
fn mul_inplace_broadcast_on_subview_preserves_base() {
    let base = cpu_tensor((1..=8).map(|i| i as f32).collect(), vec![8]);
    let v = base.stride_view(vec![2, 2], vec![2, 1], 4);
    let other = cpu_tensor(vec![10.0, 100.0], vec![1, 2]);

    v.mul_(&other).unwrap();

    assert_eq!(v.data_vec().unwrap(), &[50.0, 600.0, 70.0, 800.0]);
    assert_eq!(
        base.data_vec().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 50.0, 600.0, 70.0, 800.0],
        "mul_ broadcast path on a sub-view must write the view's region in place"
    );
    assert_eq!(base.storage().len(), 8);
}

/// `Tensor::update_data` on a NON-contiguous view must scatter through the
/// strides, not write a contiguous run at the storage offset.
///
/// torch analogue (`v.copy_(src)` through a strided view, torch 2.11):
/// `base = [1..6]; v = base.as_strided((3,), (2,)); v.copy_(tensor([9., 8., 7.]))`
/// -> `base == [9., 2., 8., 4., 7., 6.]`.
#[test]
fn update_data_noncontiguous_view_scatters_through_strides() {
    let base = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
    let v = base.stride_view(vec![3], vec![2], 0);

    // SAFETY: `v` and `base` are local to this test; no other thread
    // touches the storage and no borrow of the data is held across the
    // call (update_data's documented exclusive-access contract).
    unsafe { v.update_data(&[9.0, 8.0, 7.0]).unwrap() };

    assert_eq!(v.data_vec().unwrap(), &[9.0, 8.0, 7.0]);
    assert_eq!(
        base.data_vec().unwrap(),
        &[9.0, 2.0, 8.0, 4.0, 7.0, 6.0],
        "update_data through a stride-2 view must scatter, not overwrite [0..3)"
    );
}

/// Internal-overlap guard: a stride-0 (expanded) `out` maps several
/// logical elements onto one memory location. torch raises
/// `RuntimeError: unsupported operation: more than one element of the
/// written-to tensor refers to a single memory location. Please clone()
/// the tensor before performing the operation.` — ferrotorch must return
/// a structured error, never write a plausible value.
#[test]
fn add_out_overlapping_expanded_view_is_rejected() {
    let base = cpu_tensor(vec![1.0, 2.0], vec![2]);
    let v = base.stride_view(vec![2], vec![0], 0); // both elements -> slot 0
    let a = cpu_tensor(vec![10.0, 20.0], vec![2]);
    let b = cpu_tensor(vec![100.0, 200.0], vec![2]);

    let err = add_out(&v, &a, &b);
    assert!(
        err.is_err(),
        "out= into an internally-overlapping (stride-0) view must error \
         (torch: 'more than one element of the written-to tensor refers to \
         a single memory location'), got {err:?}"
    );
    // The guard must fire BEFORE any write: base is untouched.
    assert_eq!(base.data_vec().unwrap(), &[1.0, 2.0]);
}

/// Whole-storage, contiguous `out` keeps the (sound) swap fast path:
/// values match `torch.add(a, b, out=out)` with `out` a plain tensor.
#[test]
fn add_out_whole_tensor_still_writes_values() {
    let out = cpu_tensor(vec![f32::NAN, f32::NAN], vec![2]);
    let a = cpu_tensor(vec![10.0, 20.0], vec![2]);
    let b = cpu_tensor(vec![100.0, 200.0], vec![2]);

    add_out(&out, &a, &b).unwrap();
    // torch: torch.add([10,20],[100,200]) == [110, 220]; NaN sentinels gone.
    assert_eq!(out.data_vec().unwrap(), &[110.0, 220.0]);
}

#[cfg(feature = "gpu")]
mod gpu {
    use std::sync::Once;

    use ferrotorch_core::Device;
    use ferrotorch_core::creation::from_vec;
    use ferrotorch_core::grad_fns::arithmetic::add_out;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the #1938 GPU suite");
        });
    }

    /// CUDA mirror of the critic's offset-view divergence: the f32 path
    /// must write the view's region in the EXISTING device buffer
    /// (`strided_scatter_f32`), leaving the base's other elements intact —
    /// the pre-fix handle swap left aliased views pointing at a freed
    /// device region.
    ///
    /// torch (2.11, cuda): identical to the CPU case —
    /// `base=[1,2,3,4] (cuda); v=base.narrow(0,2,2);
    ///  torch.add([10,20],[100,200] (cuda), out=v)`
    /// -> `base == [1,2,110,220]`, storage stays 4 elements, `v.is_cuda`.
    #[test]
    fn add_out_offset_view_cuda_writes_in_place() {
        ensure_cuda_backend();

        let base = from_vec::<f32>(vec![1.0, 2.0, 3.0, 4.0], &[4])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let v = base.stride_view(vec![2], vec![1], 2);
        let a = from_vec::<f32>(vec![10.0, 20.0], &[2])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let b = from_vec::<f32>(vec![100.0, 200.0], &[2])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();

        add_out(&v, &a, &b).unwrap();

        // R-ORACLE-3: the write must not demote anything off-device.
        assert!(v.is_cuda(), "out view must stay CUDA-resident");
        assert!(base.is_cuda(), "base must stay CUDA-resident");
        assert_eq!(
            base.storage().len(),
            4,
            "shared device storage must keep len 4"
        );

        let base_host = base.to(Device::Cpu).unwrap();
        assert_eq!(
            base_host.data_vec().unwrap(),
            &[1.0, 2.0, 110.0, 220.0],
            "CUDA out= into a narrowed view must scatter into the existing \
             device buffer at the view's offset"
        );
        let v_host = v.to(Device::Cpu).unwrap();
        assert_eq!(v_host.data_vec().unwrap(), &[110.0, 220.0]);
    }

    /// CUDA sub-view `out=` for bf16 must use the u16 strided-scatter
    /// kernel and preserve all non-view elements in the shared base
    /// storage. This mirrors torch 2.11 CUDA:
    /// `base=[1,2,3,4] (bf16 cuda); v=base.narrow(0,2,2);
    ///  torch.add([10,20],[100,200], out=v)`
    /// -> `base == [1,2,110,220]`.
    #[test]
    fn add_out_subview_cuda_bf16_writes_in_place() {
        ensure_cuda_backend();

        let mk = |vals: &[f32], shape: &[usize]| {
            let bf: Vec<half::bf16> = vals.iter().copied().map(half::bf16::from_f32).collect();
            from_vec::<half::bf16>(bf, shape)
                .unwrap()
                .to(Device::Cuda(0))
                .unwrap()
        };
        let base = mk(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let v = base.stride_view(vec![2], vec![1], 2);
        let a = mk(&[10.0, 20.0], &[2]);
        let b = mk(&[100.0, 200.0], &[2]);

        add_out(&v, &a, &b).unwrap();

        assert!(v.is_cuda(), "out view must stay CUDA-resident");
        assert!(base.is_cuda(), "base must stay CUDA-resident");
        assert_eq!(base.storage().len(), 4);

        let base_host = base.to(Device::Cpu).unwrap();
        let base_vals: Vec<f32> = base_host
            .data_vec()
            .unwrap()
            .iter()
            .map(|x| x.to_f32())
            .collect();
        assert_eq!(base_vals, vec![1.0, 2.0, 110.0, 220.0]);

        let v_host = v.to(Device::Cpu).unwrap();
        let v_vals: Vec<f32> = v_host
            .data_vec()
            .unwrap()
            .iter()
            .map(|x| x.to_f32())
            .collect();
        assert_eq!(v_vals, vec![110.0, 220.0]);
    }
}
