//! Permanent regression sentinel for #819: GPU broadcast/4D matmul kernel.
//!
//! Pre-fix: shapes that fall through to `linalg::matmul` (anything other
//! than 2D x 2D, 1D x 1D, 2D x 1D, 1D x 2D, or matching 3D x 3D) hit
//! `.data()?` on a CUDA tensor and surface as `Err(GpuTensorNotAccessible)`.
//! PyTorch supports all of these on CUDA.
//!
//! Post-fix (this probe):
//! - 4D bmm (`(2,3,4,5) @ (2,3,5,6) -> (2,3,4,6)`) — no broadcast.
//! - 3D x 2D (`(4,3,5) @ (5,6) -> (4,3,6)`) — RHS implicit batch.
//! - 2D x 3D (`(3,5) @ (4,5,6) -> (4,3,6)`) — LHS implicit batch.
//! - 4D outer-axis broadcast (`(2,1,3,5) @ (2,4,5,6) -> (2,4,3,6)`).
//! - Empty batch (`(0,3,5) @ (5,6) -> (0,3,6)`).
//!
//! All routes through the new `broadcast_bmm_f{32,64}` GPU backend methods.
//! Result `is_cuda()` is true (no CPU detour).
//!
//! Tolerance constants are inlined here because the workspace-wide
//! constants live as private items inside `tests/conformance_linalg.rs`.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::Tensor;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::linalg::matmul_differentiable;

const F32_MATMUL_GPU: f32 = 1e-3;
const F64_MATMUL_GPU: f64 = 1e-9;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

fn read_back_f32(t: &Tensor<f32>) -> Vec<f32> {
    let cpu = if t.is_cuda() {
        t.cpu().expect("gpu->cpu copy")
    } else {
        t.clone()
    };
    cpu.data().expect("read_back").to_vec()
}

fn read_back_f64(t: &Tensor<f64>) -> Vec<f64> {
    let cpu = if t.is_cuda() {
        t.cpu().expect("gpu->cpu copy")
    } else {
        t.clone()
    };
    cpu.data().expect("read_back").to_vec()
}

fn vec_f32(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect()
}

fn vec_f64(n: usize, seed: u32) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            ((x as f64) / (u32::MAX as f64)) * 2.0 - 1.0
        })
        .collect()
}

/// Compute the broadcasted leading shape per numpy/PyTorch rules.
/// Result picks `db` when `da == 1`, else `da`. This handles the 0-axis
/// case correctly: 0 broadcasts with 1 -> 0 (not 1).
fn broadcast_lead(a_lead: &[usize], b_lead: &[usize]) -> Vec<usize> {
    let max_len = a_lead.len().max(b_lead.len());
    let mut out = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let da = if i < max_len - a_lead.len() {
            1
        } else {
            a_lead[i - (max_len - a_lead.len())]
        };
        let db = if i < max_len - b_lead.len() {
            1
        } else {
            b_lead[i - (max_len - b_lead.len())]
        };
        let pick = if db == 1 { da } else { db };
        out.push(pick);
    }
    out
}

/// Map a flat index in `lead` to a flat batch offset in `src_lead` after
/// broadcasting size-1 axes (and missing-prefix axes).
fn broadcast_offset(flat: usize, src_lead: &[usize], lead: &[usize]) -> usize {
    if src_lead.is_empty() {
        return 0;
    }
    let mut idx = vec![0usize; lead.len()];
    let mut rem = flat;
    for i in 0..lead.len() {
        let stride: usize = lead[i + 1..].iter().product();
        idx[i] = rem / stride.max(1);
        rem %= stride.max(1);
    }
    let offset = lead.len() - src_lead.len();
    let mut src_off = 0usize;
    let mut src_strides = vec![1usize; src_lead.len()];
    for i in (0..src_lead.len().saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_lead[i + 1];
    }
    for (i, src_stride) in src_strides.iter().enumerate().take(src_lead.len()) {
        let out_axis = i + offset;
        let dim = src_lead[i];
        let coord = if dim == 1 { 0 } else { idx[out_axis] };
        src_off += coord * src_stride;
    }
    src_off
}

/// CPU reference for a broadcast bmm: leading axes broadcast pairwise.
/// `a_full = a_lead + [m, k]`, `b_full = b_lead + [k, n]`, output is
/// `out_lead + [m, n]` where `out_lead = broadcast(a_lead, b_lead)`.
fn cpu_broadcast_bmm_f32(
    a: &[f32],
    a_lead: &[usize],
    b: &[f32],
    b_lead: &[usize],
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<usize>) {
    let lead = broadcast_lead(a_lead, b_lead);
    let batch: usize = lead.iter().product();
    let mut out = vec![0.0f32; batch * m * n];
    for flat in 0..batch {
        let a_off = broadcast_offset(flat, a_lead, &lead) * m * k;
        let b_off = broadcast_offset(flat, b_lead, &lead) * k * n;
        let c_off = flat * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += a[a_off + i * k + p] as f64 * b[b_off + p * n + j] as f64;
                }
                out[c_off + i * n + j] = acc as f32;
            }
        }
    }
    let mut full = lead;
    full.push(m);
    full.push(n);
    (out, full)
}

fn cpu_broadcast_bmm_f64(
    a: &[f64],
    a_lead: &[usize],
    b: &[f64],
    b_lead: &[usize],
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f64>, Vec<usize>) {
    let lead = broadcast_lead(a_lead, b_lead);
    let batch: usize = lead.iter().product();
    let mut out = vec![0.0f64; batch * m * n];
    for flat in 0..batch {
        let a_off = broadcast_offset(flat, a_lead, &lead) * m * k;
        let b_off = broadcast_offset(flat, b_lead, &lead) * k * n;
        let c_off = flat * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += a[a_off + i * k + p] * b[b_off + p * n + j];
                }
                out[c_off + i * n + j] = acc;
            }
        }
    }
    let mut full = lead;
    full.push(m);
    full.push(n);
    (out, full)
}

#[derive(Clone, Copy, Debug)]
struct ShapeCase {
    name: &'static str,
    a_lead: &'static [usize],
    b_lead: &'static [usize],
    m: usize,
    k: usize,
    n: usize,
}

const CASES: &[ShapeCase] = &[
    ShapeCase {
        name: "4d_bmm",
        a_lead: &[2, 3],
        b_lead: &[2, 3],
        m: 4,
        k: 5,
        n: 6,
    },
    ShapeCase {
        name: "3d_x_2d",
        a_lead: &[4],
        b_lead: &[],
        m: 3,
        k: 5,
        n: 6,
    },
    ShapeCase {
        name: "2d_x_3d",
        a_lead: &[],
        b_lead: &[4],
        m: 3,
        k: 5,
        n: 6,
    },
    ShapeCase {
        name: "4d_outer_broadcast",
        a_lead: &[2, 1],
        b_lead: &[2, 4],
        m: 3,
        k: 5,
        n: 6,
    },
    ShapeCase {
        name: "empty_batch",
        a_lead: &[0],
        b_lead: &[],
        m: 3,
        k: 5,
        n: 6,
    },
];

#[test]
fn gpu_broadcast_bmm_f32_matches_cpu() {
    ensure_cuda_backend();
    for case in CASES {
        let a_lead_prod: usize = case.a_lead.iter().product();
        let b_lead_prod: usize = case.b_lead.iter().product();
        let a_n = a_lead_prod.max(1) * case.m * case.k;
        let b_n = b_lead_prod.max(1) * case.k * case.n;

        let a_vals = vec_f32(a_n, 0xD00);
        let b_vals = vec_f32(b_n, 0xE00);

        let mut a_shape = case.a_lead.to_vec();
        a_shape.push(case.m);
        a_shape.push(case.k);
        let mut b_shape = case.b_lead.to_vec();
        b_shape.push(case.k);
        b_shape.push(case.n);

        let a_cpu = from_vec::<f32>(a_vals.clone(), &a_shape).expect("cpu A");
        let b_cpu = from_vec::<f32>(b_vals.clone(), &b_shape).expect("cpu B");
        let a_gpu = a_cpu.to(Device::Cuda(0)).expect("A gpu");
        let b_gpu = b_cpu.to(Device::Cuda(0)).expect("B gpu");

        let y = matmul_differentiable(&a_gpu, &b_gpu)
            .unwrap_or_else(|e| panic!("case {}: matmul failed: {e:?}", case.name));
        assert!(
            y.is_cuda(),
            "case {}: result must stay on GPU (got non-CUDA)",
            case.name
        );

        let (want, want_shape) = cpu_broadcast_bmm_f32(
            &a_vals,
            &if case.a_lead.is_empty() {
                vec![1usize; 0]
            } else {
                case.a_lead.to_vec()
            },
            &b_vals,
            &if case.b_lead.is_empty() {
                vec![1usize; 0]
            } else {
                case.b_lead.to_vec()
            },
            case.m,
            case.k,
            case.n,
        );

        assert_eq!(
            y.shape(),
            want_shape.as_slice(),
            "case {}: shape mismatch",
            case.name
        );

        let got = read_back_f32(&y);
        assert_eq!(got.len(), want.len(), "case {}: numel mismatch", case.name);
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let d = (g - w).abs();
            assert!(
                d <= F32_MATMUL_GPU,
                "case {} f32 elem {i}: got={g} want={w} diff={d} gate={F32_MATMUL_GPU}",
                case.name
            );
        }
    }
}

#[test]
fn gpu_broadcast_bmm_f64_matches_cpu() {
    ensure_cuda_backend();
    for case in CASES {
        let a_lead_prod: usize = case.a_lead.iter().product();
        let b_lead_prod: usize = case.b_lead.iter().product();
        let a_n = a_lead_prod.max(1) * case.m * case.k;
        let b_n = b_lead_prod.max(1) * case.k * case.n;

        let a_vals = vec_f64(a_n, 0xF00);
        let b_vals = vec_f64(b_n, 0x1100);

        let mut a_shape = case.a_lead.to_vec();
        a_shape.push(case.m);
        a_shape.push(case.k);
        let mut b_shape = case.b_lead.to_vec();
        b_shape.push(case.k);
        b_shape.push(case.n);

        let a_cpu = from_vec::<f64>(a_vals.clone(), &a_shape).expect("cpu A");
        let b_cpu = from_vec::<f64>(b_vals.clone(), &b_shape).expect("cpu B");
        let a_gpu = a_cpu.to(Device::Cuda(0)).expect("A gpu");
        let b_gpu = b_cpu.to(Device::Cuda(0)).expect("B gpu");

        let y = matmul_differentiable(&a_gpu, &b_gpu)
            .unwrap_or_else(|e| panic!("case {}: matmul f64 failed: {e:?}", case.name));
        assert!(
            y.is_cuda(),
            "case {} f64: result must stay on GPU",
            case.name
        );

        let (want, want_shape) = cpu_broadcast_bmm_f64(
            &a_vals,
            &if case.a_lead.is_empty() {
                vec![1usize; 0]
            } else {
                case.a_lead.to_vec()
            },
            &b_vals,
            &if case.b_lead.is_empty() {
                vec![1usize; 0]
            } else {
                case.b_lead.to_vec()
            },
            case.m,
            case.k,
            case.n,
        );

        assert_eq!(
            y.shape(),
            want_shape.as_slice(),
            "case {} f64: shape mismatch",
            case.name
        );
        let got = read_back_f64(&y);
        assert_eq!(
            got.len(),
            want.len(),
            "case {} f64: numel mismatch",
            case.name
        );
        let gate = F64_MATMUL_GPU.max(1e-12 * (case.k as f64).max(1.0));
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let d = (g - w).abs();
            assert!(
                d <= gate,
                "case {} f64 elem {i}: got={g} want={w} diff={d} gate={gate}",
                case.name
            );
        }
    }
}
