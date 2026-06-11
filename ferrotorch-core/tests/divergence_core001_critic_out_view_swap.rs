//! CORE-001 critic re-audit (#1695 fix at 8ae145258) — divergence #1938.
//!
//! The matched-shape branch of `add_scaled_out` / `add_out`
//! (`grad_fns/arithmetic.rs:721`) routes the write through
//! `Tensor::update_storage` -> `TensorStorage::replace_buffer_aliased`,
//! which swaps the ENTIRE shared storage buffer with the freshly-computed
//! result buffer. That is only sound for the design's documented matched
//! precondition (`.design/.../arithmetic.md:400`: `storage_len == numel &&
//! storage_offset == 0`), but NOTHING enforces it: when `out` is an offset
//! sub-view of a larger storage, the swap replaces the whole shared buffer
//! with a shorter one, silently corrupting the base tensor (and every other
//! alias) instead of writing the view's region in place.
//!
//! Upstream PyTorch (`aten/src/ATen/native/Resize.cpp:27`):
//!   `if (at::symint::sizes<T>(output).equals(shape)) { return false; }`
//! i.e. when `out.sizes()` already equals the result shape, NO resize/swap
//! happens — the TensorIterator writes elementwise into `out`'s existing
//! storage at its storage_offset. The backing storage is never replaced or
//! shrunk; other views of the same storage keep their elements.
//!
//! Live torch (2.11.0+cu130), confirmed via the parity oracle:
//!   base = [1,2,3,4]; v = base.narrow(0,2,2)        # offset-2 view, len 2
//!   torch.add([10,20],[100,200], out=v)
//!   -> v    == [110.0, 220.0]
//!   -> base == [1.0, 2.0, 110.0, 220.0]   (storage stays len 4)
//!
//! ferrotorch: `add_out(&v, a, b)` returns Ok, but replaces the shared
//! storage with a len-2 buffer; `base` (shape [4]) can no longer be read
//! (`base.data()` errors "tensor view extends beyond storage") and the
//! base elements [1,2] are destroyed. This is silent shared-state
//! corruption reachable from safe public API.
//!
//! Tracking: #1938.

use ferrotorch_core::grad_fns::arithmetic::add_out;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn cpu_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

/// Divergence: ferrotorch's `add_out` into an offset sub-view diverges from
/// `pytorch aten/src/ATen/native/Resize.cpp:27` (matched-shape => no resize,
/// in-place elementwise write). Upstream leaves base == [1,2,110,220] with
/// storage len 4; ferrotorch swaps the whole shared buffer to len 2,
/// corrupting base.
///
/// Expected values are torch-derived (oracle: torch 2.11.0+cu130), NOT
/// copied from the ferrotorch side (R-CHAR-3).
#[test]
fn divergence_core001_critic_add_out_offset_view_corrupts_base() {
    let base = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]); // storage len 4
    let v = base.stride_view(vec![2], vec![1], 2); // offset 2, numel 2 (== base[2:4])
    let a = cpu_tensor(vec![10.0, 20.0], vec![2]);
    let b = cpu_tensor(vec![100.0, 200.0], vec![2]);

    // matched-shape branch: v.shape()==[2]==broadcast([2],[2]).
    add_out(&v, &a, &b).unwrap();

    // Torch: the view's region is written in place; the base keeps its
    // first two elements and sees the new tail.
    assert_eq!(
        base.data_vec().unwrap(),
        &[1.0, 2.0, 110.0, 220.0],
        "out= into a narrowed view must write in place (Resize.cpp:27), \
         not swap/shrink the shared storage"
    );
    assert_eq!(v.data_vec().unwrap(), &[110.0, 220.0]);
    // Storage length is preserved upstream (untyped_storage stays 4 elems).
    assert_eq!(base.storage().len(), 4);
}

/// Tighter variant: an offset-0 sub-view (`base[0:2]` of a `[4]`). torch
/// writes in place leaving base == [110, 220, 3, 4]; ferrotorch shrinks the
/// shared storage to len 2, dropping base elements [3,4].
///
/// Expected values torch-derived (oracle), not copied from ferrotorch.
#[test]
fn divergence_core001_critic_add_out_head_subview_corrupts_tail() {
    let base = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let head = base.stride_view(vec![2], vec![1], 0); // base[0:2]
    let a = cpu_tensor(vec![10.0, 20.0], vec![2]);
    let b = cpu_tensor(vec![100.0, 200.0], vec![2]);

    add_out(&head, &a, &b).unwrap();

    assert_eq!(
        base.data_vec().unwrap(),
        &[110.0, 220.0, 3.0, 4.0],
        "out= into base[0:2] must leave base[2:4] == [3,4] (storage not swapped)"
    );
    assert_eq!(base.storage().len(), 4);
}
