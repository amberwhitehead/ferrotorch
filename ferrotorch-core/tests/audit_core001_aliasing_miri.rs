//! CORE-001 (#1695) — permanent Miri gate for aliased in-place mutation.
//!
//! `Tensor<T>` is `Arc<TensorInner>`-shared and `TensorInner.storage` is
//! `Arc<TensorStorage>`-shared (views/clones share storage by design —
//! PyTorch parity per `c10/core/StorageImpl.h:38` "storage is supposed to
//! uniquely own a data pointer"). In-place ops (`fill_`, `add_`, ...,
//! `update_data` / `update_storage`) therefore mutate memory that other
//! `Tensor` handles can reach. The audit found that the mutation paths
//! manufactured `&mut TensorStorage` / `&mut [T]` behind those aliased
//! `Arc`s — instant aliasing UB under Stacked Borrows the moment any other
//! handle held a live `&TensorStorage` (or had one outstanding).
//!
//! The fix routes every aliased mutation through `UnsafeCell`-based interior
//! mutability at the storage-buffer layer (`StorageBuffer::Cpu(CpuBuffer)`
//! plus `TensorStorage.data: UnsafeCell<StorageBuffer>`), so the *supported*
//! aliasing patterns below are UB-free. This file pins them under Miri.
//!
//! ## How to run (requires nightly + miri component)
//!
//! ```bash
//! cargo +nightly miri test -p ferrotorch-core --test audit_core001_aliasing_miri
//! ```
//!
//! All tests are CPU-only and tiny (Miri-compatible: no GPU backend, no
//! file IO, numel <= 8). They also run (and must pass) under plain
//! `cargo test`.
//!
//! ## Documented residual contract (NOT covered by these tests)
//!
//! Two patterns remain forbidden by the crate's documented synchronization
//! contract (the same informal contract PyTorch has for its storages):
//!
//! 1. Holding a `&[T]` obtained from `Tensor::data()` ACROSS an in-place
//!    mutation performed through another alias, then reading it. The
//!    element write invalidates the outstanding shared slice (and a
//!    storage-swapping op such as the broadcast `add_scaled_` path frees
//!    the buffer it points into). Sequence reads after writes instead —
//!    every test below does exactly that.
//! 2. Unsynchronized cross-thread access: one thread mutating through an
//!    alias while another reads. That is a data race; callers must provide
//!    external synchronization (PyTorch's contract, documented on
//!    `TensorStorage`).

use ferrotorch_core::grad_fns::arithmetic::{add_out, add_scaled_out};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn cpu_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

// ---------------------------------------------------------------------------
// Cloned-alias variants: mutate through one handle, read through the other.
// This is THE PyTorch semantic model (clones share storage and identity).
// ---------------------------------------------------------------------------

#[test]
fn clone_alias_add_scalar_sequenced() {
    let t1 = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let t2 = t1.clone();
    t1.add_scalar_(10.0).unwrap();
    assert_eq!(t2.data().unwrap(), &[11.0, 12.0, 13.0]);
}

#[test]
fn clone_alias_fill_then_zero_sequenced() {
    let t1 = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t2 = t1.clone();
    t1.fill_(7.0).unwrap();
    assert_eq!(t2.data().unwrap(), &[7.0, 7.0, 7.0, 7.0]);
    t2.zero_().unwrap();
    assert_eq!(t1.data().unwrap(), &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn clone_alias_tensor_add_and_mul_sequenced() {
    let t1 = cpu_tensor(vec![1.0, 2.0], vec![2]);
    let t2 = t1.clone();
    let rhs = cpu_tensor(vec![10.0, 20.0], vec![2]);
    // Same-shape CPU fast path (update_data).
    t1.add_(&rhs).unwrap();
    assert_eq!(t2.data().unwrap(), &[11.0, 22.0]);
    // mul_ same-shape CPU fast path, written through the OTHER alias.
    t2.mul_(&rhs).unwrap();
    assert_eq!(t1.data().unwrap(), &[110.0, 440.0]);
}

#[test]
fn clone_alias_broadcast_add_scaled_storage_swap_sequenced() {
    // alpha != 1.0 forces the broadcast/scaled path, which swaps the whole
    // storage buffer in-place (update_storage). The alias must observe the
    // swapped values afterwards.
    let t1 = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let t2 = t1.clone();
    let rhs = cpu_tensor(vec![1.0, 1.0, 1.0], vec![3]);
    t1.add_scaled_(&rhs, 2.0).unwrap();
    assert_eq!(t2.data().unwrap(), &[3.0, 4.0, 5.0]);
}

// ---------------------------------------------------------------------------
// View-alias variants: views share the storage Arc with the base.
// ---------------------------------------------------------------------------

#[test]
fn view_alias_fill_sequenced() {
    let base = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let view = base.view_reshape(vec![4]).unwrap();
    base.fill_(9.0).unwrap();
    assert_eq!(view.data().unwrap(), &[9.0, 9.0, 9.0, 9.0]);
}

#[test]
fn offset_view_alias_partial_write_sequenced() {
    // A narrowed stride-view writes only its own region of the shared
    // buffer (update_data honors storage_offset). PyTorch parity: the base
    // observes a partial update.
    let base = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let tail = base.stride_view(vec![2], vec![1], 2); // elements [3.0, 4.0]
    tail.fill_(0.0).unwrap();
    assert_eq!(base.data().unwrap(), &[1.0, 2.0, 0.0, 0.0]);
}

// ---------------------------------------------------------------------------
// Metadata borrows held ACROSS an aliased write are supported (this was the
// observed pre-fix UB: manufacturing `&mut TensorStorage` invalidated every
// outstanding `&TensorStorage`). Post-fix, element writes and buffer swaps
// go through interior mutability and never invalidate metadata borrows.
// ---------------------------------------------------------------------------

#[test]
fn metadata_borrow_survives_aliased_elementwise_write() {
    let t1 = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let t2 = t1.clone();
    let st = t2.storage(); // &TensorStorage held across the write
    let dev_before = st.device();
    t1.add_scalar_(1.0).unwrap();
    // Pre-fix: UB under Miri (st's tag was popped by `&mut TensorStorage`).
    let dev_after = st.device();
    assert_eq!(dev_before, dev_after);
    assert_eq!(t2.data().unwrap(), &[2.0, 3.0, 4.0]);
}

#[test]
fn metadata_borrow_survives_aliased_storage_swap() {
    let t1 = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let t2 = t1.clone();
    let st = t2.storage(); // &TensorStorage held across the buffer swap
    let rhs = cpu_tensor(vec![1.0, 1.0, 1.0], vec![3]);
    t1.add_scaled_(&rhs, 2.0).unwrap(); // update_storage path
    // Pre-fix: UB under Miri (ptr::replace through Arc::as_ptr popped st).
    assert!(st.is_cpu());
    assert_eq!(st.len(), 3);
    assert_eq!(t2.data().unwrap(), &[3.0, 4.0, 5.0]);
}

#[test]
fn shape_borrow_survives_aliased_elementwise_write() {
    let t1 = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let t2 = t1.clone();
    let shape = t2.shape(); // &[usize] into TensorInner, held across write
    t1.fill_(5.0).unwrap();
    assert_eq!(shape, &[2, 2]);
}

// ---------------------------------------------------------------------------
// out= writes (grad_fns::arithmetic::add_out / add_scaled_out).
// ---------------------------------------------------------------------------

#[test]
fn add_out_into_cloned_alias_sequenced() {
    let a = cpu_tensor(vec![1.0, 2.0], vec![2]);
    let b = cpu_tensor(vec![10.0, 20.0], vec![2]);
    let out = cpu_tensor(vec![0.0, 0.0], vec![2]);
    let out_alias = out.clone();
    add_out(&out, &a, &b).unwrap();
    assert_eq!(out_alias.data().unwrap(), &[11.0, 22.0]);
}

#[test]
fn add_scaled_out_resize_unique_out() {
    // out.shape() != broadcast shape and out is UNIQUE → torch-parity
    // silent resize (Tensor.resize_ analog).
    let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cpu_tensor(vec![1.0, 1.0], vec![2]);
    let out = cpu_tensor(vec![0.0], vec![1]);
    add_scaled_out(&out, &a, &b, 1.0).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.data().unwrap(), &[2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn add_scaled_out_resize_aliased_out_errors() {
    // Resizing rewrites shape/strides inside the shared TensorInner. With
    // a second handle alive that is not soundly expressible (the alias may
    // be concurrently observing the metadata), so the resize branch must
    // return a structured error instead of mutating behind the alias.
    // CORE-001 (#1695): pre-fix this silently resized behind the alias.
    let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = cpu_tensor(vec![1.0, 1.0], vec![2]);
    let out = cpu_tensor(vec![0.0], vec![1]);
    let alias = out.clone();
    let err = add_scaled_out(&out, &a, &b, 1.0).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("alias"),
        "expected an aliased-resize error, got: {msg}"
    );
    // The alias must be untouched.
    assert_eq!(alias.shape(), &[1]);
    assert_eq!(alias.data().unwrap(), &[0.0]);
}

// ---------------------------------------------------------------------------
// The unsafe entry points used per their contracts (optimizer pattern):
// genuinely exclusive access, sequenced reads.
// ---------------------------------------------------------------------------

#[test]
fn data_mut_exclusive_optimizer_pattern() {
    let t = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    {
        // SAFETY: `t` has no clones and no outstanding borrows of its data;
        // the &mut [T] is dropped before any other access.
        let slice = unsafe { t.data_mut() }.unwrap();
        for x in slice.iter_mut() {
            *x *= 2.0;
        }
    }
    assert_eq!(t.data().unwrap(), &[2.0, 4.0, 6.0]);
}

#[test]
fn update_data_then_alias_read_sequenced() {
    let t1 = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let t2 = t1.clone();
    // SAFETY: no outstanding borrows of the storage; single thread; the
    // alias t2 only reads after this call returns (sequenced access).
    unsafe { t1.update_data(&[7.0, 8.0, 9.0]) }.unwrap();
    assert_eq!(t2.data().unwrap(), &[7.0, 8.0, 9.0]);
}

#[test]
fn update_storage_swap_visible_to_alias_sequenced() {
    let t1 = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let t2 = t1.clone();
    let fresh = TensorStorage::cpu(vec![4.0, 5.0, 6.0]);
    // SAFETY: no outstanding borrows of the storage; single thread; the
    // alias t2 only reads after this call returns.
    unsafe { t1.update_storage(fresh) }.unwrap();
    assert_eq!(t2.data().unwrap(), &[4.0, 5.0, 6.0]);
}
