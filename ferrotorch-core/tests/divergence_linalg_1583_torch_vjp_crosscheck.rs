//! R-BUILD-4 adversarial re-audit of commit `211b0af56` (#1583 linalg
//! grad-aware public-forward wiring: addmm/addbmm/baddbmm/addmv/addr +
//! diagonal/diag/tril/triu + kron).
//!
//! The builder's own #1583 FD tests are *self-consistent* against the same
//! ferrotorch forward (FD of `linalg_fwd::op` vs the analytic backward), and
//! several of them used a UNIFORM `sum` loss (notably `diagonal`), which masks
//! a wrong scatter POSITION or a forward that itself diverges from torch.
//!
//! Per R-CHAR-3 the expected values below are NOT copied from the ferrotorch
//! side: each is the `.grad` / forward output of a LIVE `torch` float64 run
//! captured by the parity-sweep oracle (torch 2.11.0+cu130). The exact
//! reproduction script + outputs are pinned in the comment above each block.
//! Driving the now-grad-aware PUBLIC forwards with NON-default beta/alpha,
//! NON-zero offsets/diagonals, input-broadcast bias, and NON-uniform weighted
//! losses, each test asserts the torch value. A divergence in any VJP, in the
//! forward index-mapping, or in the bias-broadcast sum-reduce fails here.

use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::linalg as linalg_fwd;
use ferrotorch_core::ops::tensor_ops;
use ferrotorch_core::storage::TensorStorage;

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn no_grad_leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{label}[{i}]: ferrotorch={a}, torch={e}, diff={}",
            (a - e).abs()
        );
    }
}

/// Sum of `op_out * weight` as the scalar loss, so the upstream grad is the
/// (non-uniform) weight tensor rather than all-ones — this exposes wrong
/// scatter/mask POSITIONS that a `sum`-only loss cannot see.
fn weighted_backward(out: &Tensor<f64>, weight: &[f64], wshape: &[usize]) {
    let w = no_grad_leaf(weight, wshape);
    let prod = mul(out, &w).unwrap();
    let loss = reduce_sum(&prod).unwrap();
    loss.backward().unwrap();
}

// =====================================================================
// KRON — index-mapping of the Kronecker VJP (the bug surface per brief).
//
// torch 2.11.0 float64:
//   A = [[1,2],[-1,0.5]]; B = [[2,-1],[0.5,1]]
//   K = kron(A,B); W = arange(1..17).reshape(4,4); (K*W).sum().backward()
//   kron dA = [8.5, 13.5, 28.5, 33.5]
//   kron dB = [3.5, 6.0, 13.5, 16.0]
// =====================================================================
#[test]
fn kron_dA_dB_match_torch_weighted() {
    let a = leaf(&[1.0, 2.0, -1.0, 0.5], &[2, 2]);
    let b = leaf(&[2.0, -1.0, 0.5, 1.0], &[2, 2]);
    let k = linalg_fwd::kron(&a, &b).unwrap();
    assert_eq!(k.shape(), &[4, 4]);
    let w: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    weighted_backward(&k, &w, &[4, 4]);
    let ga = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    let gb = b.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(&ga, &[8.5, 13.5, 28.5, 33.5], 1e-9, "kron dA vs torch");
    assert_close(&gb, &[3.5, 6.0, 13.5, 16.0], 1e-9, "kron dB vs torch");
}

// =====================================================================
// DIAGONAL — non-zero offset scatter POSITION.
//
// torch 2.11.0 float64:
//   A = arange(1..10).reshape(3,3)
//   offset=1:  d = diagonal(A,1) -> len 2; w=[10,100]; (d*w).sum().backward()
//     dA = [0,10,0, 0,0,100, 0,0,0]
//   offset=-1: w=[10,100]; dA = [0,0,0, 10,0,0, 0,100,0]
// =====================================================================
#[test]
fn diagonal_offset_pos1_scatter_position_matches_torch() {
    let a = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    let d = linalg_fwd::diagonal(&a, 1).unwrap();
    // forward value must also match torch diagonal(A,1) = [2,6]
    assert_close(
        &d.data().unwrap().to_vec(),
        &[2.0, 6.0],
        1e-9,
        "diagonal(A,1) fwd",
    );
    weighted_backward(&d, &[10.0, 100.0], &[2]);
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g,
        &[0.0, 10.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        1e-9,
        "diagonal offset=1 scatter vs torch",
    );
}

#[test]
fn diagonal_offset_neg1_scatter_position_matches_torch() {
    let a = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    let d = linalg_fwd::diagonal(&a, -1).unwrap();
    assert_close(
        &d.data().unwrap().to_vec(),
        &[4.0, 8.0],
        1e-9,
        "diagonal(A,-1) fwd",
    );
    weighted_backward(&d, &[10.0, 100.0], &[2]);
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g,
        &[0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 100.0, 0.0],
        1e-9,
        "diagonal offset=-1 scatter vs torch",
    );
}

// =====================================================================
// DIAG — both directions, weighted; plus non-zero diagonal construct.
//
// torch 2.11.0 float64:
//   extract: A=arange(1..10).reshape(3,3); d=diag(A,0); w=[10,100,1000];
//     dA = [10,0,0, 0,100,0, 0,0,1000]
//   construct: a=[1,2,3]; d=diag(a,0)=3x3; W=diag([1,2,3]); dA=[1,2,3]
//   construct offset=1: a=[1,2,3]; d=diag(a,1)=4x4 superdiag;
//     W=arange(1..17).reshape(4,4); dA=[2,7,12]
// =====================================================================
#[test]
fn diag_extract_2d_to_1d_matches_torch_weighted() {
    let a = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    let d = tensor_ops::diag(&a, 0).unwrap();
    assert_eq!(d.shape(), &[3]);
    weighted_backward(&d, &[10.0, 100.0, 1000.0], &[3]);
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g,
        &[10.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1000.0],
        1e-9,
        "diag extract dA vs torch",
    );
}

#[test]
fn diag_construct_1d_to_2d_matches_torch_weighted() {
    let a = leaf(&[1.0, 2.0, 3.0], &[3]);
    let d = tensor_ops::diag(&a, 0).unwrap();
    assert_eq!(d.shape(), &[3, 3]);
    weighted_backward(&d, &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3]);
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(&g, &[1.0, 2.0, 3.0], 1e-9, "diag construct dA vs torch");
}

#[test]
fn diag_construct_1d_to_2d_offset1_matches_torch_weighted() {
    let a = leaf(&[1.0, 2.0, 3.0], &[3]);
    let d = tensor_ops::diag(&a, 1).unwrap();
    assert_eq!(d.shape(), &[4, 4]);
    let w: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    weighted_backward(&d, &w, &[4, 4]);
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g,
        &[2.0, 7.0, 12.0],
        1e-9,
        "diag construct offset=1 dA vs torch",
    );
}

// =====================================================================
// TRIL / TRIU — non-zero diagonal mask BOUNDARY.
//
// torch 2.11.0 float64:
//   A=arange(1..10).reshape(3,3); W=arange(1..10).reshape(3,3)
//   tril(A,1): (t*W).sum().backward(); dA = [1,2,0, 4,5,6, 7,8,9]
//   triu(A,1): W=arange(1..10).reshape(3,3); dA = [0,2,3, 0,0,6, 0,0,0]
// =====================================================================
#[test]
fn tril_diag1_mask_boundary_matches_torch() {
    let a = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    let t = tensor_ops::tril(&a, 1).unwrap();
    let w: Vec<f64> = (1..=9).map(|x| x as f64).collect();
    weighted_backward(&t, &w, &[3, 3]);
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g,
        &[1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        1e-9,
        "tril(A,1) mask vs torch",
    );
}

#[test]
fn triu_diag1_mask_boundary_matches_torch() {
    let a = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    let t = tensor_ops::triu(&a, 1).unwrap();
    let w: Vec<f64> = (1..=9).map(|x| x as f64).collect();
    weighted_backward(&t, &w, &[3, 3]);
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g,
        &[0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0],
        1e-9,
        "triu(A,1) mask vs torch",
    );
}

// =====================================================================
// ADDMM — VECTOR input broadcast to [m,n]; dbias must sum-reduce over the
// broadcast (row) dim scaled by beta. beta=0.5 alpha=2.0.
//
// torch 2.11.0 float64:
//   bias=[0.5,-1.0] (shape [2] -> [2,2]); m1=[[1,2,-1],[0.5,3,1]] [2,3];
//   m2=[[2,-1],[0.5,1],[1.5,-0.5]] [3,2]; out=addmm(bias,m1,m2,0.5,2.0).sum().backward()
//   out = [3.25, 2.5, 8.25, 3.5]
//   dbias = [1.0, 1.0]   (= beta * sum over 2 rows of grad=1)
//   dm1 = [2,3,2, 2,3,2]
//   dm2 = [3,3, 10,10, 0,0]
// =====================================================================
#[test]
fn addmm_vector_bias_broadcast_grad_matches_torch() {
    let bias = leaf(&[0.5, -1.0], &[2]); // broadcast to [2,2]
    let m1 = leaf(&[1.0, 2.0, -1.0, 0.5, 3.0, 1.0], &[2, 3]);
    let m2 = leaf(&[2.0, -1.0, 0.5, 1.0, 1.5, -0.5], &[3, 2]);
    let c = linalg_fwd::addmm(&bias, &m1, &m2, 0.5, 2.0).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_close(
        &c.data().unwrap().to_vec(),
        &[3.25, 2.5, 8.25, 3.5],
        1e-9,
        "addmm fwd (vec bias)",
    );
    let loss = reduce_sum(&c).unwrap();
    loss.backward().unwrap();
    let gb = bias.grad().unwrap().unwrap();
    assert_eq!(
        gb.shape(),
        &[2],
        "addmm dbias must keep the [2] vector shape"
    );
    assert_close(
        &gb.data().unwrap().to_vec(),
        &[1.0, 1.0],
        1e-9,
        "addmm dbias vs torch",
    );
    let g1 = m1.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g1,
        &[2.0, 3.0, 2.0, 2.0, 3.0, 2.0],
        1e-9,
        "addmm dm1 vs torch",
    );
    let g2 = m2.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g2,
        &[3.0, 3.0, 10.0, 10.0, 0.0, 0.0],
        1e-9,
        "addmm dm2 vs torch",
    );
}

// =====================================================================
// ADDMV — beta=0.5 alpha=2.0.
//
// torch 2.11.0 float64:
//   bias=[0.5,-1]; mat=[[1,2,-1],[0.5,3,1]] [2,3]; v=[2,-1,0.5];
//   addmv(bias,mat,v,0.5,2.0).sum().backward()
//   dbias=[0.5,0.5]; dmat=[4,-2,1, 4,-2,1]; dvec=[3,10,0]
// =====================================================================
#[test]
fn addmv_grad_matches_torch_nondefault_scaling() {
    let bias = leaf(&[0.5, -1.0], &[2]);
    let mat = leaf(&[1.0, 2.0, -1.0, 0.5, 3.0, 1.0], &[2, 3]);
    let v = leaf(&[2.0, -1.0, 0.5], &[3]);
    let c = linalg_fwd::addmv(&bias, &mat, &v, 0.5, 2.0).unwrap();
    assert_eq!(c.shape(), &[2]);
    let loss = reduce_sum(&c).unwrap();
    loss.backward().unwrap();
    let gb = bias.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(&gb, &[0.5, 0.5], 1e-9, "addmv dbias vs torch");
    let gm = mat.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &gm,
        &[4.0, -2.0, 1.0, 4.0, -2.0, 1.0],
        1e-9,
        "addmv dmat vs torch",
    );
    let gv = v.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(&gv, &[3.0, 10.0, 0.0], 1e-9, "addmv dvec vs torch");
}

// =====================================================================
// ADDR — beta=1.0 alpha=0.5.
//
// torch 2.11.0 float64:
//   bias=[[0.5,-1,2],[1.5,0,-0.5]] [2,3]; v1=[1.5,-2]; v2=[2,1,-1.5];
//   addr(bias,v1,v2,1.0,0.5).sum().backward()
//   dbias=[1,1,1,1,1,1]; dv1=[0.75,0.75]; dv2=[-0.25,-0.25,-0.25]
// =====================================================================
#[test]
fn addr_grad_matches_torch_nondefault_alpha() {
    let bias = leaf(&[0.5, -1.0, 2.0, 1.5, 0.0, -0.5], &[2, 3]);
    let v1 = leaf(&[1.5, -2.0], &[2]);
    let v2 = leaf(&[2.0, 1.0, -1.5], &[3]);
    let c = linalg_fwd::addr(&bias, &v1, &v2, 1.0, 0.5).unwrap();
    assert_eq!(c.shape(), &[2, 3]);
    let loss = reduce_sum(&c).unwrap();
    loss.backward().unwrap();
    let gb = bias.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &gb,
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        1e-9,
        "addr dbias vs torch",
    );
    let gv1 = v1.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(&gv1, &[0.75, 0.75], 1e-9, "addr dv1 vs torch");
    let gv2 = v2.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(&gv2, &[-0.25, -0.25, -0.25], 1e-9, "addr dv2 vs torch");
}

// =====================================================================
// ADDBMM — VECTOR bias broadcast to [m,n]; grad shared over batch.
// beta=0.5 alpha=1.5.
//
// torch 2.11.0 float64:
//   bias=[0.5,-1] ([2]->[2,2]);
//   b1=[[[1,2],[-1,.5]],[[.5,-1],[2,1]]] [2,2,2];
//   b2=[[[2,-1],[.5,1]],[[1,0],[-.5,2]]] [2,2,2];
//   addbmm(bias,b1,b2,0.5,1.5).sum().backward()
//   out=[6.25,-2,-0.125,4.75]
//   dbias=[1,1]; db1=[1.5,2.25,1.5,2.25, 1.5,2.25,1.5,2.25];
//   db2=[0,0,3.75,3.75, 3.75,3.75,0,0]
// =====================================================================
#[test]
fn addbmm_vector_bias_broadcast_grad_matches_torch() {
    let bias = leaf(&[0.5, -1.0], &[2]);
    let b1 = leaf(&[1.0, 2.0, -1.0, 0.5, 0.5, -1.0, 2.0, 1.0], &[2, 2, 2]);
    let b2 = leaf(&[2.0, -1.0, 0.5, 1.0, 1.0, 0.0, -0.5, 2.0], &[2, 2, 2]);
    let c = linalg_fwd::addbmm(&bias, &b1, &b2, 0.5, 1.5).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_close(
        &c.data().unwrap().to_vec(),
        &[6.25, -2.0, -0.125, 4.75],
        1e-9,
        "addbmm fwd (vec bias)",
    );
    let loss = reduce_sum(&c).unwrap();
    loss.backward().unwrap();
    let gb = bias.grad().unwrap().unwrap();
    assert_eq!(gb.shape(), &[2], "addbmm dbias must keep [2] vector shape");
    assert_close(
        &gb.data().unwrap().to_vec(),
        &[1.0, 1.0],
        1e-9,
        "addbmm dbias vs torch",
    );
    let g1 = b1.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g1,
        &[1.5, 2.25, 1.5, 2.25, 1.5, 2.25, 1.5, 2.25],
        1e-9,
        "addbmm db1 vs torch",
    );
    let g2 = b2.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g2,
        &[0.0, 0.0, 3.75, 3.75, 3.75, 3.75, 0.0, 0.0],
        1e-9,
        "addbmm db2 vs torch",
    );
}

// =====================================================================
// BADDBMM — full 3D bias, beta=1.0 alpha=0.75.
//
// torch 2.11.0 float64:
//   bias=[[[.5,-1],[2,1.5]],[[0,1],[-.5,2]]] [2,2,2];
//   b1=[[[1,2],[-1,.5]],[[.5,-1],[2,1]]];
//   b2=[[[2,-1],[.5,1]],[[1,0],[-.5,2]]];
//   baddbmm(bias,b1,b2,1.0,0.75).sum().backward()
//   db1=[0.75,1.125,0.75,1.125, 0.75,1.125,0.75,1.125]
//   db2=[0,0,1.875,1.875, 1.875,1.875,0,0]
// =====================================================================
#[test]
fn baddbmm_grad_matches_torch_nondefault_alpha() {
    let bias = leaf(&[0.5, -1.0, 2.0, 1.5, 0.0, 1.0, -0.5, 2.0], &[2, 2, 2]);
    let b1 = leaf(&[1.0, 2.0, -1.0, 0.5, 0.5, -1.0, 2.0, 1.0], &[2, 2, 2]);
    let b2 = leaf(&[2.0, -1.0, 0.5, 1.0, 1.0, 0.0, -0.5, 2.0], &[2, 2, 2]);
    let c = linalg_fwd::baddbmm(&bias, &b1, &b2, 1.0, 0.75).unwrap();
    assert_eq!(c.shape(), &[2, 2, 2]);
    let loss = reduce_sum(&c).unwrap();
    loss.backward().unwrap();
    let g1 = b1.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g1,
        &[0.75, 1.125, 0.75, 1.125, 0.75, 1.125, 0.75, 1.125],
        1e-9,
        "baddbmm db1 vs torch",
    );
    let g2 = b2.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g2,
        &[0.0, 0.0, 1.875, 1.875, 1.875, 1.875, 0.0, 0.0],
        1e-9,
        "baddbmm db2 vs torch",
    );
}
