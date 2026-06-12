//! Red regression tests for CORE-055 (#1749), CPU lane — CLASS-V, High.
//!
//! The CPU `cat` path (`src/grad_fns/shape.rs`) calls `data()` on each
//! input; `data()` explicitly rejects non-contiguous tensors, so valid
//! inputs such as transposes / permutes / column-narrows fail with
//! `InvalidArgument` instead of concatenating their logical values.
//! `CatBackward`'s CPU path makes the same `data()` call on `grad_output`,
//! so a non-contiguous upstream gradient errors out of the backward.
//!
//! Oracle (R-ORACLE-1b): live torch 2.11.0+cu130, 2026-06-11:
//!
//! ```python
//! x = torch.arange(24.).reshape(4,6)
//! w = x.t()                                   # (6,4) non-contiguous
//! torch.cat([w,w],0).flatten()
//! # [0,6,12,18, 1,7,13,19, 2,8,14,20, 3,9,15,21, 4,10,16,22, 5,11,17,23] * 2
//! v = x.narrow(0,1,2)                         # (2,6) storage_offset 6
//! torch.cat([v,v],1).flatten()
//! # [6..11, 6..11, 12..17, 12..17]
//! n = x.narrow(1,2,3)                         # (4,3) offset 2, non-contiguous
//! torch.cat([n,n],0).flatten()
//! # [2,3,4, 8,9,10, 14,15,16, 20,21,22] * 2
//! y = torch.arange(24.).reshape(2,3,4); p = y.permute(2,0,1)   # (4,2,3)
//! torch.cat([p,p],0).flatten()
//! # [0,4,8,12,16,20, 1,5,9,13,17,21, 2,6,10,14,18,22, 3,7,11,15,19,23] * 2
//! torch.cat([w, torch.ones(2,4)],0).flatten()
//! # [0,6,12,18, ..., 5,11,17,23, 1,1,1,1, 1,1,1,1]
//!
//! a = torch.arange(24.).reshape(4,6).detach().requires_grad_(True)
//! b = (torch.arange(24.)*0.5).reshape(6,4).detach().requires_grad_(True)
//! out = torch.cat([a.t(), b], 0)              # (12,4)
//! out.backward(torch.arange(48.).reshape(12,4))
//! a.grad.flatten()  # [0,4,8,12,16,20, 1,5,9,13,17,21, 2,6,10,14,18,22, 3,7,11,15,19,23]
//! b.grad.flatten()  # [24..47]
//!
//! c = torch.arange(24.).reshape(4,6).detach().requires_grad_(True)
//! out2 = torch.cat([c.narrow(0,1,2), torch.ones(2,6)], 0)     # (4,6)
//! out2.backward(torch.arange(24.).reshape(4,6) + 1.0)
//! c.grad.flatten()  # [0,0,0,0,0,0, 1..6, 7..12, 0,0,0,0,0,0]
//!
//! d = torch.arange(8.).reshape(2,4).detach().requires_grad_(True)
//! e = (torch.arange(8.)+10.).reshape(2,4).detach().requires_grad_(True)
//! out3 = torch.cat([d, e], 0)                 # (4,4)
//! out3.backward(torch.arange(16.).reshape(4,4).t())   # non-contiguous go
//! d.grad.flatten()  # [0,4,8,12, 1,5,9,13]
//! e.grad.flatten()  # [2,6,10,14, 3,7,11,15]
//! ```

use ferrotorch_core::cat;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn plain(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn arange(n: usize) -> Vec<f32> {
    (0..n).map(|v| v as f32).collect()
}

// cat is pure data movement (no arithmetic): every element round-trips
// bit-exactly, so float equality is the right check throughout this file.
fn assert_exact(got: &Tensor<f32>, want_shape: &[usize], want: &[f32], label: &str) {
    assert_eq!(got.shape(), want_shape, "{label}: shape");
    let host = got.data_vec().expect("readback");
    assert_eq!(host.len(), want.len(), "{label}: numel");
    for (i, (a, e)) in host.iter().zip(want).enumerate() {
        assert!(
            a == e,
            "{label}[{i}]: got {a}, torch oracle {e} (full: {host:?})"
        );
    }
}

/// Pre-fix: `data()` on the transpose view returns
/// `InvalidArgument("tensor is not contiguous")` — torch concatenates it.
#[test]
fn cat_of_cpu_transpose_views_matches_torch() {
    let x = plain(&arange(24), &[4, 6]);
    let w = x.transpose(0, 1).expect("transpose view"); // (6,4)
    assert!(!w.is_contiguous(), "transpose view must be non-contiguous");
    let out = cat(&[w.clone(), w], 0).expect("cat of CPU transpose views must succeed (CORE-055)");
    let half: Vec<f32> = vec![
        0.0, 6.0, 12.0, 18.0, 1.0, 7.0, 13.0, 19.0, 2.0, 8.0, 14.0, 20.0, 3.0, 9.0, 15.0, 21.0,
        4.0, 10.0, 16.0, 22.0, 5.0, 11.0, 17.0, 23.0,
    ];
    let want: Vec<f32> = half.iter().chain(half.iter()).copied().collect();
    assert_exact(&out, &[12, 4], &want, "cat([t,t],0)");
}

/// Row-narrow (offset 6, contiguous strides) — works at HEAD because CPU
/// `data()` honours `storage_offset`; pinned so the materialization gate
/// never regresses it.
#[test]
fn cat_of_cpu_narrow_offset_views_matches_torch() {
    let x = plain(&arange(24), &[4, 6]);
    let v = x.narrow(0, 1, 2).expect("narrow view"); // (2,6), offset 6
    assert_eq!(v.storage_offset(), 6, "narrow view must carry its offset");
    let out = cat(&[v.clone(), v], 1).expect("cat of CPU narrow views must succeed (CORE-055)");
    let want: Vec<f32> = vec![
        6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    ];
    assert_exact(&out, &[2, 12], &want, "cat([narrow,narrow],1)");
}

/// Column-narrow (offset 2, NON-contiguous strides). Pre-fix: rejected.
#[test]
fn cat_of_cpu_column_narrow_views_matches_torch() {
    let x = plain(&arange(24), &[4, 6]);
    let n = x.narrow(1, 2, 3).expect("column narrow"); // (4,3), offset 2
    assert!(!n.is_contiguous(), "column narrow must be non-contiguous");
    let out =
        cat(&[n.clone(), n], 0).expect("cat of CPU column-narrow views must succeed (CORE-055)");
    let half: Vec<f32> = vec![
        2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 14.0, 15.0, 16.0, 20.0, 21.0, 22.0,
    ];
    let want: Vec<f32> = half.iter().chain(half.iter()).copied().collect();
    assert_exact(&out, &[8, 3], &want, "cat([ncol,ncol],0)");
}

/// 3-D permute view. Pre-fix: rejected.
#[test]
fn cat_of_cpu_permute_views_matches_torch() {
    let y = plain(&arange(24), &[2, 3, 4]);
    let p = y.permute(&[2, 0, 1]).expect("permute view"); // (4,2,3)
    assert!(!p.is_contiguous(), "permute view must be non-contiguous");
    let out = cat(&[p.clone(), p], 0).expect("cat of CPU permute views must succeed (CORE-055)");
    let half: Vec<f32> = vec![
        0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0,
        18.0, 22.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0,
    ];
    let want: Vec<f32> = half.iter().chain(half.iter()).copied().collect();
    assert_exact(&out, &[8, 2, 3], &want, "cat([perm,perm],0)");
}

/// Mixed layouts: a transpose view next to a plain contiguous tensor.
#[test]
fn cat_of_cpu_transpose_and_contiguous_matches_torch() {
    let x = plain(&arange(24), &[4, 6]);
    let w = x.transpose(0, 1).expect("transpose view"); // (6,4)
    let ones = plain(&[1.0; 8], &[2, 4]);
    let out = cat(&[w, ones], 0).expect("cat of transpose + contiguous must succeed (CORE-055)");
    let want: Vec<f32> = vec![
        0.0, 6.0, 12.0, 18.0, 1.0, 7.0, 13.0, 19.0, 2.0, 8.0, 14.0, 20.0, 3.0, 9.0, 15.0, 21.0,
        4.0, 10.0, 16.0, 22.0, 5.0, 11.0, 17.0, 23.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    assert_exact(&out, &[8, 4], &want, "cat([t,ones],0)");
}

/// Backward through cat over a transpose VIEW of a tracking leaf: the grad
/// must reach the ORIGINAL leaf with torch's values (R-ORACLE-3 grad flow,
/// never `requires_grad` flags alone). Pre-fix the forward already errors.
#[test]
fn cat_backward_over_transpose_view_reaches_original_leaf() {
    let a = leaf(&arange(24), &[4, 6]);
    let b = leaf(
        &arange(24).iter().map(|v| v * 0.5).collect::<Vec<_>>(),
        &[6, 4],
    );
    let at = a.transpose(0, 1).expect("transpose view of leaf"); // (6,4)
    let out = cat(&[at, b.clone()], 0).expect("cat over a transpose view must succeed (CORE-055)");
    assert_eq!(out.shape(), &[12, 4]);

    let go = plain(&arange(48), &[12, 4]);
    out.backward_with_gradient(&go).expect("backward");

    let ga = a
        .grad()
        .unwrap()
        .expect("grad must reach the ORIGINAL transposed leaf");
    assert_exact(
        &ga,
        &[4, 6],
        &[
            0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0,
            18.0, 22.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0,
        ],
        "a.grad",
    );
    let gb = b.grad().unwrap().expect("grad must reach leaf b");
    let want_b: Vec<f32> = (24..48).map(|v| v as f32).collect();
    assert_exact(&gb, &[6, 4], &want_b, "b.grad");
}

/// Backward through cat over a narrow-offset VIEW of a tracking leaf:
/// the grad scatters into the narrowed rows of the original leaf.
#[test]
fn cat_backward_over_narrow_view_reaches_original_leaf() {
    let c = leaf(&arange(24), &[4, 6]);
    let vc = c.narrow(0, 1, 2).expect("narrow view of leaf"); // (2,6)
    let ones = plain(&[1.0; 12], &[2, 6]);
    let out = cat(&[vc, ones], 0).expect("cat over a narrow view must succeed (CORE-055)");
    assert_eq!(out.shape(), &[4, 6]);

    let go_data: Vec<f32> = (0..24).map(|v| v as f32 + 1.0).collect();
    let go = plain(&go_data, &[4, 6]);
    out.backward_with_gradient(&go).expect("backward");

    let gc = c
        .grad()
        .unwrap()
        .expect("grad must reach the ORIGINAL narrowed leaf");
    assert_exact(
        &gc,
        &[4, 6],
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
            12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        "c.grad",
    );
}

/// `CatBackward` fed a NON-contiguous `grad_output` (a transpose view).
/// Pre-fix: the CPU backward calls `data()` on it and errors.
#[test]
fn cat_backward_with_noncontiguous_grad_output_matches_torch() {
    let d = leaf(&arange(8), &[2, 4]);
    let e_data: Vec<f32> = (0..8).map(|v| v as f32 + 10.0).collect();
    let e = leaf(&e_data, &[2, 4]);
    let out = cat(&[d.clone(), e.clone()], 0).expect("cat forward");
    assert_eq!(out.shape(), &[4, 4]);

    let go_base = plain(&arange(16), &[4, 4]);
    let go_t = go_base.transpose(0, 1).expect("transposed grad_output");
    assert!(!go_t.is_contiguous());
    out.backward_with_gradient(&go_t)
        .expect("backward with a non-contiguous grad_output must succeed (CORE-055)");

    let gd = d.grad().unwrap().expect("grad must reach leaf d");
    assert_exact(
        &gd,
        &[2, 4],
        &[0.0, 4.0, 8.0, 12.0, 1.0, 5.0, 9.0, 13.0],
        "d.grad",
    );
    let ge = e.grad().unwrap().expect("grad must reach leaf e");
    assert_exact(
        &ge,
        &[2, 4],
        &[2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0],
        "e.grad",
    );
}
