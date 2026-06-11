//! Red-then-green regression tests for audit findings CORE-161..CORE-165
//! (crosslink #1855 #1856 #1857 #1858 #1859) — the einsum mechanism batch.
//!
//! Every numerical expectation below is quoted from a LIVE torch session
//! (torch 2.11.0+cu130, R-ORACLE-1 path (b)); the generating snippet is
//! pasted in a comment next to each expected value.
//!
//! * CORE-161 / #1855 — two-input CPU einsum silently drops summation over
//!   lone indices (subscript in one operand, absent from the output).
//! * CORE-162 / #1856 — backward panics (index OOB) when an operand has
//!   repeated subscripts; the textually-swapped gradient equation is also
//!   mathematically wrong (true gradient needs a diagonal-embed scatter).
//! * CORE-163 / #1857 — stored equations not whitespace-normalized;
//!   forward accepts "ii -> i" but backward re-parses the raw string and
//!   panics (`char_val[&' ']`) or errors.
//! * CORE-164 / #1858 — implicit-mode equations (no "->") are not
//!   differentiable; the backward treats the output as empty instead of
//!   the sorted once-occurring labels.
//! * CORE-165 / #1859 — repeated OUTPUT subscripts accepted and produce
//!   garbage; torch rejects with a structured error.

use ferrotorch_core::einsum::{einsum, einsum_differentiable};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;
#[cfg(feature = "gpu")]
use std::sync::Once;

#[cfg(feature = "gpu")]
static GPU_INIT: Once = Once::new();

#[cfg(feature = "gpu")]
fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU lane of this suite");
    });
}

fn t_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn t_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}
fn leaf_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

// f32 tolerance: values here are exact small-integer products (well inside
// the 24-bit mantissa), so the only rounding is the contraction-order sum
// over <= 4 terms: 1e-4 relative head-room on O(1e3) magnitudes.
const TOL_F32: f32 = 1e-3;
// f64: same analysis at 53-bit mantissa.
const TOL_F64: f64 = 1e-9;

fn assert_close_f32(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= TOL_F32,
            "{label}: index {i}: {a} vs {e} (diff {})",
            (a - e).abs()
        );
    }
}
fn assert_close_f64(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= TOL_F64,
            "{label}: index {i}: {a} vs {e} (diff {})",
            (a - e).abs()
        );
    }
}

// ===========================================================================
// CORE-165 / #1859 — repeated output subscripts must be rejected
// ===========================================================================

// torch oracle:
//   >>> torch.einsum("i->ii", torch.tensor([1.,2.,3.]))
//   RuntimeError: einsum(): output subscript i appears more than once in the output
#[test]
fn core165_repeated_output_subscript_rejected_single_input_f32() {
    let v = t_f32(&[1.0, 2.0, 3.0], &[3]);
    let r = einsum("i->ii", &[&v]);
    assert!(
        r.is_err(),
        "einsum(\"i->ii\") must be rejected (torch: 'output subscript i appears more \
         than once in the output'); got Ok with shape {:?}",
        r.as_ref().map(|t| t.shape().to_vec())
    );
    let msg = format!("{}", r.unwrap_err());
    assert!(
        msg.contains("more than once"),
        "error must identify the repeated output subscript; got: {msg}"
    );
}

#[test]
fn core165_repeated_output_subscript_rejected_single_input_f64() {
    let v = t_f64(&[1.0, 2.0, 3.0], &[3]);
    assert!(einsum("i->ii", &[&v]).is_err());
}

// torch oracle:
//   >>> torch.einsum("ij,jk->iik", torch.ones(2,2), torch.ones(2,2))
//   RuntimeError: einsum(): output subscript i appears more than once in the output
#[test]
fn core165_repeated_output_subscript_rejected_two_input_f32() {
    let a = t_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = t_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    assert!(
        einsum("ij,jk->iik", &[&a, &b]).is_err(),
        "einsum(\"ij,jk->iik\") must be rejected (torch errors)"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn core165_repeated_output_subscript_rejected_cuda() {
    ensure_cuda_backend();
    let v = t_f32(&[1.0, 2.0, 3.0], &[3]).to(Device::Cuda(0)).unwrap();
    assert!(
        einsum("i->ii", &[&v]).is_err(),
        "CUDA: einsum(\"i->ii\") must be rejected, not produce garbage"
    );
}

// ===========================================================================
// CORE-161 / #1855 — lone-index summation on the CPU two-input path
// ===========================================================================

// torch oracle:
//   >>> A = torch.tensor([[1.,2.],[3.,4.]]); B = torch.tensor([10.,100.])
//   >>> torch.einsum("ij,j->j", A, B).tolist()
//   [40.0, 600.0]
#[test]
fn core161_lone_a_index_is_summed_ij_j_to_j_f32() {
    let a = t_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = t_f32(&[10.0, 100.0], &[2]);
    let c = einsum("ij,j->j", &[&a, &b]).unwrap();
    assert_eq!(c.shape(), &[2]);
    assert_close_f32(c.data().unwrap(), &[40.0, 600.0], "ij,j->j f32");
}

#[test]
fn core161_lone_a_index_is_summed_ij_j_to_j_f64() {
    let a = t_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = t_f64(&[10.0, 100.0], &[2]);
    let c = einsum("ij,j->j", &[&a, &b]).unwrap();
    assert_close_f64(c.data().unwrap(), &[40.0, 600.0], "ij,j->j f64");
}

// torch oracle:
//   >>> v = torch.tensor([1.,2.]); M = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
//   >>> torch.einsum("i,ij->i", v, M).tolist()
//   [6.0, 30.0]
#[test]
fn core161_lone_b_index_is_summed_i_ij_to_i_f32() {
    let v = t_f32(&[1.0, 2.0], &[2]);
    let m = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = einsum("i,ij->i", &[&v, &m]).unwrap();
    assert_eq!(c.shape(), &[2]);
    assert_close_f32(c.data().unwrap(), &[6.0, 30.0], "i,ij->i f32");
}

#[test]
fn core161_lone_b_index_is_summed_i_ij_to_i_f64() {
    let v = t_f64(&[1.0, 2.0], &[2]);
    let m = t_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = einsum("i,ij->i", &[&v, &m]).unwrap();
    assert_close_f64(c.data().unwrap(), &[6.0, 30.0], "i,ij->i f64");
}

// torch oracle:
//   >>> P = torch.tensor([[1.,2.],[3.,4.]])
//   >>> Q = torch.tensor([[5.,6.,7.],[8.,9.,10.]])
//   >>> torch.einsum("ab,cd->ad", P, Q).tolist()
//   [[39.0, 45.0, 51.0], [91.0, 105.0, 119.0]]
#[test]
fn core161_lone_indices_both_operands_ab_cd_to_ad_f32() {
    let p = t_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let q = t_f32(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3]);
    let c = einsum("ab,cd->ad", &[&p, &q]).unwrap();
    assert_eq!(c.shape(), &[2, 3]);
    assert_close_f32(
        c.data().unwrap(),
        &[39.0, 45.0, 51.0, 91.0, 105.0, 119.0],
        "ab,cd->ad f32",
    );
}

// torch oracle:
//   >>> torch.einsum("ij,j->j", torch.zeros(0,2), torch.tensor([10.,100.])).tolist()
//   [0.0, 0.0]   (shape [2], no error)
#[test]
fn core161_zero_size_lone_dim_sums_to_zero_no_panic_f32() {
    let a = t_f32(&[], &[0, 2]);
    let b = t_f32(&[10.0, 100.0], &[2]);
    let c = einsum("ij,j->j", &[&a, &b]).unwrap();
    assert_eq!(c.shape(), &[2]);
    assert_close_f32(c.data().unwrap(), &[0.0, 0.0], "zero-size lone dim");
}

// GPU lane: reduce_lone_axes already handles this on CUDA — pin it so the
// CPU fix cannot regress the device path, and assert the result device
// (R-ORACLE-3).
#[cfg(feature = "gpu")]
#[test]
fn core161_lone_index_cuda_parity_unregressed() {
    ensure_cuda_backend();
    let a = t_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let b = t_f32(&[10.0, 100.0], &[2]).to(Device::Cuda(0)).unwrap();
    let c = einsum("ij,j->j", &[&a, &b]).unwrap();
    assert!(c.is_cuda(), "result must stay on CUDA");
    // torch oracle (same snippet as the CPU test): [40.0, 600.0]
    assert_close_f32(
        c.cpu().unwrap().data().unwrap(),
        &[40.0, 600.0],
        "ij,j->j cuda",
    );
}

// ===========================================================================
// CORE-163 / #1857 — whitespace-tolerant equations must be differentiable
// ===========================================================================

// torch oracle:
//   >>> A = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
//   >>> d = torch.einsum("ii -> i", A); d.tolist()
//   [1.0, 4.0]
//   >>> d.sum().backward(); A.grad.tolist()
//   [[1.0, 0.0], [0.0, 1.0]]
#[test]
fn core163_spaced_equation_diag_backward_f32() {
    let a = leaf_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let d = einsum_differentiable("ii -> i", &[&a]).unwrap();
    assert_close_f32(d.data().unwrap(), &[1.0, 4.0], "ii -> i forward");
    let loss = d.sum_all().unwrap();
    loss.backward().unwrap();
    let g = a.grad().unwrap().expect("a must receive a gradient");
    assert_close_f32(
        g.cpu().unwrap().data().unwrap(),
        &[1.0, 0.0, 0.0, 1.0],
        "ii -> i grad",
    );
}

// torch oracle:
//   >>> A = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
//   >>> B = torch.tensor([[5.,6.],[7.,8.]], requires_grad=True)
//   >>> torch.einsum("ij, jk -> ik", A, B).sum().backward()
//   >>> A.grad.tolist(), B.grad.tolist()
//   ([[11.0, 15.0], [11.0, 15.0]], [[4.0, 4.0], [6.0, 6.0]])
#[test]
fn core163_spaced_equation_mm_backward_f64() {
    let a = leaf_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = leaf_f64(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c = einsum_differentiable("ij, jk -> ik", &[&a, &b]).unwrap();
    let loss = c.sum_all().unwrap();
    loss.backward().unwrap();
    let ga = a.grad().unwrap().expect("a grad");
    let gb = b.grad().unwrap().expect("b grad");
    assert_close_f64(
        ga.cpu().unwrap().data().unwrap(),
        &[11.0, 15.0, 11.0, 15.0],
        "spaced mm grad_A",
    );
    assert_close_f64(
        gb.cpu().unwrap().data().unwrap(),
        &[4.0, 4.0, 6.0, 6.0],
        "spaced mm grad_B",
    );
}

// ===========================================================================
// CORE-164 / #1858 — implicit-mode equations must be differentiable
// ===========================================================================

// torch oracle:
//   >>> A = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
//   >>> B = torch.tensor([[5.,6.],[7.,8.]], requires_grad=True)
//   >>> c = torch.einsum("ij,jk", A, B); c.tolist()
//   [[19.0, 22.0], [43.0, 50.0]]
//   >>> c.sum().backward(); A.grad.tolist(), B.grad.tolist()
//   ([[11.0, 15.0], [11.0, 15.0]], [[4.0, 4.0], [6.0, 6.0]])
#[test]
fn core164_implicit_two_input_backward_f32() {
    let a = leaf_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = leaf_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c = einsum_differentiable("ij,jk", &[&a, &b]).unwrap();
    assert_close_f32(
        c.data().unwrap(),
        &[19.0, 22.0, 43.0, 50.0],
        "implicit mm forward",
    );
    let loss = c.sum_all().unwrap();
    loss.backward().unwrap();
    let ga = a.grad().unwrap().expect("a grad");
    let gb = b.grad().unwrap().expect("b grad");
    assert_close_f32(
        ga.cpu().unwrap().data().unwrap(),
        &[11.0, 15.0, 11.0, 15.0],
        "implicit mm grad_A",
    );
    assert_close_f32(
        gb.cpu().unwrap().data().unwrap(),
        &[4.0, 4.0, 6.0, 6.0],
        "implicit mm grad_B",
    );
}

// torch oracle:
//   >>> A = torch.tensor([[1.,2.,3.],[4.,5.,6.]], requires_grad=True)
//   >>> tr = torch.einsum("ji", A)   # implicit output "ij" -> transpose
//   >>> tr.tolist()
//   [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
//   >>> tr.backward(torch.tensor([[1.,2.],[3.,4.],[5.,6.]]))
//   >>> A.grad.tolist()
//   [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
#[test]
fn core164_implicit_single_input_transpose_backward_f64() {
    let a = leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tr = einsum_differentiable("ji", &[&a]).unwrap();
    assert_eq!(tr.shape(), &[3, 2]);
    assert_close_f64(
        tr.data().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        "implicit ji forward",
    );
    let g = t_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    tr.backward_with_gradient(&g).unwrap();
    let ga = a.grad().unwrap().expect("a grad");
    assert_close_f64(
        ga.cpu().unwrap().data().unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
        "implicit ji grad_A",
    );
}

// ===========================================================================
// CORE-162 / #1856 — repeated INPUT subscripts: correct (diagonal-embedded)
// gradients instead of an index-OOB panic
// ===========================================================================

// torch oracle:
//   >>> A = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
//   >>> B = torch.tensor([10.,100.,1000.], requires_grad=True)
//   >>> C = torch.einsum("ii,j->ij", A, B); C.tolist()
//   [[10.0, 100.0, 1000.0], [40.0, 400.0, 4000.0]]
//   >>> C.sum().backward()
//   >>> A.grad.tolist()
//   [[1110.0, 0.0], [0.0, 1110.0]]
//   >>> B.grad.tolist()
//   [5.0, 5.0, 5.0]
#[test]
fn core162_repeated_input_subscript_backward_uniform_f32() {
    let a = leaf_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = leaf_f32(&[10.0, 100.0, 1000.0], &[3]);
    let c = einsum_differentiable("ii,j->ij", &[&a, &b]).unwrap();
    assert_close_f32(
        c.data().unwrap(),
        &[10.0, 100.0, 1000.0, 40.0, 400.0, 4000.0],
        "ii,j->ij forward",
    );
    let loss = c.sum_all().unwrap();
    loss.backward().unwrap();
    let ga = a.grad().unwrap().expect("a grad");
    let gb = b.grad().unwrap().expect("b grad");
    assert_close_f32(
        ga.cpu().unwrap().data().unwrap(),
        &[1110.0, 0.0, 0.0, 1110.0],
        "ii,j->ij grad_A (diag-embed)",
    );
    assert_close_f32(
        gb.cpu().unwrap().data().unwrap(),
        &[5.0, 5.0, 5.0],
        "ii,j->ij grad_B",
    );
}

// torch oracle (non-uniform upstream gradient):
//   >>> A2 = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
//   >>> B2 = torch.tensor([10.,100.,1000.], requires_grad=True)
//   >>> C2 = torch.einsum("ii,j->ij", A2, B2)
//   >>> C2.backward(torch.tensor([[1.,2.,3.],[4.,5.,6.]]))
//   >>> A2.grad.tolist()
//   [[3210.0, 0.0], [0.0, 6540.0]]
//   >>> B2.grad.tolist()
//   [17.0, 22.0, 27.0]
#[test]
fn core162_repeated_input_subscript_backward_weighted_f64() {
    let a = leaf_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = leaf_f64(&[10.0, 100.0, 1000.0], &[3]);
    let c = einsum_differentiable("ii,j->ij", &[&a, &b]).unwrap();
    let g = t_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    c.backward_with_gradient(&g).unwrap();
    let ga = a.grad().unwrap().expect("a grad");
    let gb = b.grad().unwrap().expect("b grad");
    assert_close_f64(
        ga.cpu().unwrap().data().unwrap(),
        &[3210.0, 0.0, 0.0, 6540.0],
        "weighted grad_A (diag-embed)",
    );
    assert_close_f64(
        gb.cpu().unwrap().data().unwrap(),
        &[17.0, 22.0, 27.0],
        "weighted grad_B",
    );
}

// Lone-index equation gradients: the forward sums over the lone subscript
// (CORE-161); its gradient broadcasts back along it. Exercises the
// missing-char expand path of the generalized backward.
//
// torch oracle:
//   >>> A = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
//   >>> B = torch.tensor([10.,100.], requires_grad=True)
//   >>> C = torch.einsum("ij,j->j", A, B)
//   >>> C.backward(torch.tensor([2.,3.]))
//   >>> A.grad.tolist()
//   [[20.0, 300.0], [20.0, 300.0]]
//   >>> B.grad.tolist()
//   [8.0, 18.0]
#[test]
fn core162_lone_index_backward_broadcasts_f32() {
    let a = leaf_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = leaf_f32(&[10.0, 100.0], &[2]);
    let c = einsum_differentiable("ij,j->j", &[&a, &b]).unwrap();
    let g = t_f32(&[2.0, 3.0], &[2]);
    c.backward_with_gradient(&g).unwrap();
    let ga = a.grad().unwrap().expect("a grad");
    let gb = b.grad().unwrap().expect("b grad");
    assert_close_f32(
        ga.cpu().unwrap().data().unwrap(),
        &[20.0, 300.0, 20.0, 300.0],
        "ij,j->j grad_A (broadcast)",
    );
    assert_close_f32(
        gb.cpu().unwrap().data().unwrap(),
        &[8.0, 18.0],
        "ij,j->j grad_B",
    );
}

// torch oracle:
//   >>> a = torch.tensor([1.,2.,3.], requires_grad=True)
//   >>> c = torch.tensor([4.,5.], requires_grad=True)
//   >>> out = torch.einsum("a,c->", a, c); out.item()
//   54.0
//   >>> out.backward(); a.grad.tolist(), c.grad.tolist()
//   ([9.0, 9.0, 9.0], [6.0, 6.0])
#[test]
fn core162_fully_lone_scalar_backward_f64() {
    let a = leaf_f64(&[1.0, 2.0, 3.0], &[3]);
    let c = leaf_f64(&[4.0, 5.0], &[2]);
    let out = einsum_differentiable("a,c->", &[&a, &c]).unwrap();
    assert_close_f64(out.data().unwrap(), &[54.0], "a,c-> forward");
    out.backward().unwrap();
    let ga = a.grad().unwrap().expect("a grad");
    let gc = c.grad().unwrap().expect("c grad");
    assert_close_f64(
        ga.cpu().unwrap().data().unwrap(),
        &[9.0, 9.0, 9.0],
        "a,c-> grad_a",
    );
    assert_close_f64(
        gc.cpu().unwrap().data().unwrap(),
        &[6.0, 6.0],
        "a,c-> grad_c",
    );
}

// CUDA lane for the repeated-subscript backward (the audit reports the
// same backward returns an Internal error on CUDA). R-ORACLE-3: assert
// gradient device as well as values.
#[cfg(feature = "gpu")]
#[test]
fn core162_repeated_input_subscript_backward_cuda() {
    ensure_cuda_backend();
    let a_host = leaf_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let a = a_host.to(Device::Cuda(0)).unwrap();
    // `.to()` on a leaf produces a graph edge back to the host leaf; mark
    // the device tensor itself as the observed leaf if `.to()` detaches.
    let b_host = leaf_f32(&[10.0, 100.0, 1000.0], &[3]);
    let b = b_host.to(Device::Cuda(0)).unwrap();
    let c = einsum_differentiable("ii,j->ij", &[&a, &b]).unwrap();
    assert!(c.is_cuda(), "forward result must stay on CUDA");
    let loss = c.sum_all().unwrap();
    loss.backward().unwrap();
    // Gradient flow must reach a leaf (R-ORACLE-3): accept either the
    // device tensor or the host leaf carrying the grad, depending on
    // whether `.to()` is a graph edge in this build.
    let ga_holder = if a.grad().unwrap().is_some() {
        &a
    } else {
        &a_host
    };
    let gb_holder = if b.grad().unwrap().is_some() {
        &b
    } else {
        &b_host
    };
    let ga = ga_holder
        .grad()
        .unwrap()
        .expect("grad must flow to a leaf for A");
    let gb = gb_holder
        .grad()
        .unwrap()
        .expect("grad must flow to a leaf for B");
    // torch oracle (same snippet as the uniform CPU test):
    //   A.grad = [[1110, 0], [0, 1110]], B.grad = [5, 5, 5]
    assert_close_f32(
        ga.cpu().unwrap().data().unwrap(),
        &[1110.0, 0.0, 0.0, 1110.0],
        "cuda ii,j->ij grad_A",
    );
    assert_close_f32(
        gb.cpu().unwrap().data().unwrap(),
        &[5.0, 5.0, 5.0],
        "cuda ii,j->ij grad_B",
    );
}
