//! ADVERSARIAL RE-AUDIT of the dim-aware GPU gather/scatter family
//! (commit b2793d6a9, #1545 / sub #1535) and the strided-view fix
//! (commit 02fcd71a1, #1655). The original builder verified only "nice"
//! contiguous cases; the #1655 fix made the four CUDA fast paths
//! `.contiguous()`-materialise input/self/src on-device before dispatch.
//! This file (a) pins the transposed-view divergences the fix targets and
//! (b) hunts HARDER strided cases (non-zero storage_offset, permuted 3D)
//! the fix might not cover.
//!
//! # FIX UNDER AUDIT — non-contiguous CUDA input must honor strides
//!
//! `ferrotorch_core::ops::indexing::{gather,scatter,scatter_value,scatter_add}`
//! formerly took a CUDA fast path guarded only by `input.is_cuda()`, passing
//! `input.gpu_handle()` (the RAW physical buffer, ignoring `strides()` and
//! `storage_offset()`) into PTX launchers that assume a C-contiguous
//! `[outer, axis, inner]` layout. For a transposed/permuted/narrowed view the
//! logical shape != physical layout, so every computed address was wrong.
//!
//! The fix (`ferrotorch-core/src/ops/indexing.rs:190,351-352,498,653-654`)
//! calls `input.contiguous()` (and `src.contiguous()` for scatter/scatter_add)
//! before dispatch. For a non-contiguous CUDA tensor `.contiguous()` dispatches
//! to the backend `strided_copy_f32/f64` kernel, an ON-DEVICE copy that
//! honors `src_strides` AND `src_offset`
//! (`ferrotorch-core/src/methods.rs:1583-1589`) — no host round trip; result
//! stays GPU-resident. Upstream parallel: torch's CUDA scatter/gather restrides
//! via TensorIterator honoring `self.strides()`
//! (`pytorch aten/src/ATen/native/cuda/ScatterGatherKernel.cu:196,205-207`).
//!
//! NB: in this API the `index` is a flat row-major `&[usize]` host slice with
//! `index_shape` — there is NO index Tensor that could be non-contiguous, so
//! "transposed index view" is not expressible at this surface; the index is
//! always the logical row-major order torch's `index` tensor would present.
//!
//! # R-CHAR-3 provenance (live torch 2.11.0+cu130, CUDA — RTX 3090)
//!
//! ```python
//! import torch; d="cuda"   # values identical on cpu unless noted nondeterministic
//!
//! # --- transposed gather (fix target) ---
//! base = torch.tensor([[1.,2.,3.],[4.,5.,6.]], device=d)   # [2,3]
//! tt = base.t()                                            # [3,2] view, NOT contiguous
//! idx = torch.tensor([[0,1],[1,0],[0,1]], device=d)
//! torch.gather(tt, 1, idx).cpu().tolist()  #   [[1.,4.],[5.,2.],[3.,6.]]
//!
//! # --- transposed scatter OVERWRITE, UNIQUE target offsets, NON-ZERO self (deterministic) ---
//! st  = torch.tensor([[1.,2.,3.],[4.,5.,6.]], device=d).t()  # [3,2] view [[1,4],[2,5],[3,6]]
//! src = torch.tensor([[10.,20.],[30.,40.]], device=d)        # [2,2]
//! sidx= torch.tensor([[0,1],[2,0]], device=d)                # dim0, unique per col: c0{0,2} c1{1,0}
//! torch.scatter(st, 0, sidx, src).cpu().tolist()
//! #   [[10.,40.],[2.,20.],[30.,6.]]   (cpu == cuda; the 2 & 6 are PRESERVED self values
//! #                                    a contiguous-misread would corrupt)
//!
//! # --- transposed scatter OVERWRITE, DUPLICATE target offsets (NONDETERMINISTIC on CUDA) ---
//! z = torch.zeros(2,3, device=d).t()                       # [3,2] view
//! src2 = torch.tensor([[10.,20.],[30.,40.],[50.,60.]], device=d)
//! sidx2= torch.tensor([[0,1],[1,0],[0,1]], device=d)       # col0 rows {0,1,0}, col1 {1,0,1}
//! torch.scatter(z, 0, sidx2, src2).cpu().tolist()
//! #   cpu : [[50.,40.],[30.,60.],[0.,0.]]
//! #   cuda: [[10.,40.],[30.,20.],[0.,0.]]   (stable on this HW; torch docs: unspecified)
//!
//! # --- transposed scatter_add, NON-ZERO self (fix target) ---
//! za   = torch.tensor([[1.,2.,3.],[4.,5.,6.]], device=d).t()   # [3,2] view
//! aidx = torch.tensor([[0,0],[1,1],[0,1]], device=d)
//! asrc = torch.tensor([[1.,2.],[3.,4.],[5.,6.]], device=d)
//! torch.scatter_add(za, 1, aidx, asrc).cpu().tolist()  #   [[4.,4.],[2.,12.],[8.,12.]]
//!
//! # --- HARDER: narrowed (nonzero storage_offset) non-contiguous gather ---
//! big = torch.arange(0.,24.,device=d).reshape(4,6)
//! v   = big.narrow(1,1,3).narrow(0,1,2)    # [2,3] view, strides (6,1), offset 7, NOT contig
//! gidx= torch.tensor([[0,2,1],[1,0,2]], device=d)
//! torch.gather(v, 1, gidx).cpu().tolist()  #   [[7.,9.,8.],[14.,13.,15.]]
//!
//! # --- HARDER: transposed-narrowed non-contiguous scatter_add ---
//! base = torch.arange(1.,25.,device=d).reshape(4,6)
//! nc   = base.narrow(1,0,2).narrow(0,0,3).t()   # [2,3] view strides (1,6) offset 0, NOT contig
//!                                                # == [[1,7,13],[2,8,14]]
//! saidx= torch.tensor([[0,0,1],[1,1,0]], device=d)
//! sasrc= torch.tensor([[1.,2.,3.],[4.,5.,6.]], device=d)
//! torch.scatter_add(nc, 0, saidx, sasrc).cpu().tolist()  #   [[2.,9.,19.],[6.,13.,17.]]
//!
//! # --- HARDER: 3D permuted non-contiguous gather at the permuted dim ---
//! t3 = torch.arange(0.,24.,device=d).reshape(2,3,4)
//! p  = t3.permute(2,0,1)                  # [4,2,3] view, strides (1,12,4), NOT contig
//! g3 = torch.zeros(4,2,3,dtype=torch.long,device=d); g3[:,1,:]=1
//! torch.gather(p, 1, g3).flatten().cpu().tolist()
//! #   [0,4,8, 12,16,20, 1,5,9, 13,17,21, 2,6,10, 14,18,22, 3,7,11, 15,19,23]
//!
//! # --- clean (regression-guard) fixtures, contiguous inputs ---
//! inp = torch.arange(1.,13.,device=d).reshape(3,4)
//! idx = torch.tensor([[0,3],[1,2],[3,0]], device=d)
//! torch.gather(inp,1,idx).cpu().tolist()  #   [[1.,4.],[6.,7.],[12.,9.]]
//! z = torch.zeros(3,4,device=d)
//! src = torch.tensor([[1.,2.],[3.,4.],[5.,6.]],device=d)
//! torch.scatter(z,1,torch.tensor([[0,3],[1,2],[3,0]],device=d),src).cpu().tolist()
//! #   [[1.,0.,0.,2.],[0.,3.,4.,0.],[6.,0.,0.,5.]]
//! za = torch.zeros(3,device=d); aidx=torch.zeros(1000,dtype=torch.long,device=d)
//! torch.scatter_add(za,0,aidx,torch.ones(1000,device=d)).cpu().tolist()[0]  #   1000.0
//! ```

#![cfg(feature = "cuda")]

use ferrotorch_core::ops::indexing::scatter_value;
use ferrotorch_core::{Device, Tensor, TensorStorage, gather, scatter, scatter_add};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
}

fn cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f64 tensor")
}

fn cpu_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(f16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu f16 tensor")
}

fn cpu_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(bf16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu bf16 tensor")
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

fn host_f16(t: &Tensor<f16>) -> Vec<f32> {
    t.cpu()
        .expect("cpu()")
        .data()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn host_bf16(t: &Tensor<bf16>) -> Vec<f32> {
    t.cpu()
        .expect("cpu()")
        .data()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn cuda_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    cpu_f16(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda f16")
        .requires_grad_(requires_grad)
}

fn cuda_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    cpu_bf16(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda bf16")
        .requires_grad_(requires_grad)
}

// ===========================================================================
// FIX TARGET: non-contiguous (transposed) CUDA input — strides honored.
// These exercise the #1655 .contiguous()-materialise fix. They must PASS:
// GPU result == torch == ferrotorch-CPU (all deterministic cases). Each FAILS
// against the pre-fix (b2793d6a9) kernel (negative control verified live).
// ===========================================================================

/// `gather` on a transposed (non-contiguous) CUDA tensor must match
/// `torch.gather`. Pins the #1655 fix at
/// `ferrotorch-core/src/ops/indexing.rs:190` (input.contiguous()).
/// torch returns `[[1,4],[5,2],[3,6]]`; pre-fix GPU returned `[1,2,4,3,5,6]`.
#[test]
fn divergence_gather_transposed_cuda_input_f32() {
    ensure_cuda();
    let base = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let tt = base.transpose(0, 1).expect("transpose"); // [3,2] view, non-contig
    assert!(tt.is_cuda());
    assert!(
        !tt.is_contiguous(),
        "transposed view must be non-contiguous"
    );

    let index = [0usize, 1, 1, 0, 0, 1];
    let out = gather(&tt, 1, &index, &[3, 2]).expect("gpu gather transposed");
    assert!(out.is_cuda(), "result must stay GPU-resident");

    let cpu_view = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .transpose(0, 1)
        .expect("cpu transpose");
    let cpu_out = gather(&cpu_view, 1, &index, &[3, 2]).expect("cpu gather transposed");
    assert_eq!(
        cpu_out.data_vec().unwrap(),
        vec![1.0, 4.0, 5.0, 2.0, 3.0, 6.0],
        "CPU reference must equal torch (sanity)"
    );

    // torch.gather(base.t(), 1, idx) == [[1,4],[5,2],[3,6]].
    assert_eq!(
        host_f32(&out),
        vec![1.0, 4.0, 5.0, 2.0, 3.0, 6.0],
        "GPU gather on transposed input must match torch (and CPU)"
    );
}

/// `scatter` (OVERWRITE) on a transposed (non-contiguous) CUDA `self` with a
/// **UNIQUE** target-offset index set AND a **NON-ZERO** self — deterministic,
/// so GPU==CPU==torch, while the preserved non-zero self values expose the
/// stride bug.
///
/// (Rewritten from the prior ill-posed fixture which used DUPLICATE target
/// offsets into a zeros self: scatter-overwrite with duplicate indices is
/// documented NONDETERMINISTIC on CUDA per torch, so torch-CPU and torch-CUDA
/// legitimately differ and GPU==CPU is impossible; a zeros self ALSO masks the
/// stride bug. This rewrite uses unique indices into a non-zero self so the
/// case is both well-posed AND still pins the stride fix. The duplicate-index
/// nondeterminism is covered separately by
/// `scatter_transposed_dup_idx_cuda_matches_torch_cuda`.)
///
/// self = `[[1,2,3],[4,5,6]].t()` = `[[1,4],[2,5],[3,6]]` ([3,2] view, dim0
/// extent 3). dim=0 scatter, idx `[[0,1],[2,0]]` (unique per column),
/// src `[[10,20],[30,40]]`. The `2` at [1][0] and `6` at [2][1] are PRESERVED
/// self values; a contiguous-misread of the physical buffer corrupts them.
/// torch (cpu == cuda): `[[10,40],[2,20],[30,6]]`.
#[test]
fn divergence_scatter_transposed_cuda_self_f32() {
    ensure_cuda();
    let base = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let st = base.transpose(0, 1).expect("transpose"); // [3,2] view [[1,4],[2,5],[3,6]]
    assert!(st.is_cuda());
    assert!(
        !st.is_contiguous(),
        "transposed self must be non-contiguous"
    );
    let src = cpu_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2])
        .to(Device::Cuda(0))
        .expect("src cuda");
    // torch sidx = [[0,1],[2,0]] -> row-major flat. UNIQUE per column.
    let index = [0usize, 1, 2, 0];
    let out = scatter(&st, 0, &index, &[2, 2], &src).expect("gpu scatter transposed unique");
    assert!(out.is_cuda(), "result must stay GPU-resident");

    let st_cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .transpose(0, 1)
        .expect("cpu t");
    let src_cpu = cpu_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let cpu_out = scatter(&st_cpu, 0, &index, &[2, 2], &src_cpu).expect("cpu scatter t");
    assert_eq!(
        cpu_out.data_vec().unwrap(),
        vec![10.0, 40.0, 2.0, 20.0, 30.0, 6.0],
        "CPU reference must equal torch (sanity)"
    );

    // torch.scatter(base.t(), 0, [[0,1],[2,0]], src) == [[10,40],[2,20],[30,6]].
    assert_eq!(
        host_f32(&out),
        vec![10.0, 40.0, 2.0, 20.0, 30.0, 6.0],
        "GPU scatter into transposed non-zero self (unique idx) must match torch (and CPU)"
    );
}

/// Coverage of the DUPLICATE-index scatter-overwrite case the original
/// (ill-posed) test conflated. scatter-overwrite with duplicate target offsets
/// is NONDETERMINISTIC on CUDA per torch docs: torch-CPU and torch-CUDA give
/// DIFFERENT results, so the correct cross-check is GPU == **torch-CUDA**, NOT
/// GPU == CPU. On the RTX 3090 / torch 2.11.0+cu130 the CUDA value is stably
/// `[[10,40],[30,20],[0,0]]` (cpu would be `[[50,40],[30,60],[0,0]]`).
#[test]
fn scatter_transposed_dup_idx_cuda_matches_torch_cuda() {
    ensure_cuda();
    let z = cpu_f32(&[0.0; 6], &[2, 3])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let zt = z.transpose(0, 1).expect("transpose"); // [3,2] view
    assert!(!zt.is_contiguous());
    let src = cpu_f32(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2])
        .to(Device::Cuda(0))
        .expect("src cuda");
    // torch sidx = [[0,1],[1,0],[0,1]] -> DUPLICATE: col0 rows {0,1,0}, col1 {1,0,1}.
    let index = [0usize, 1, 1, 0, 0, 1];
    let out = scatter(&zt, 0, &index, &[3, 2], &src).expect("gpu scatter transposed dup");
    assert!(out.is_cuda(), "result must stay GPU-resident");

    // torch CUDA value (NOT cpu): [[10,40],[30,20],[0,0]].
    assert_eq!(
        host_f32(&out),
        vec![10.0, 40.0, 30.0, 20.0, 0.0, 0.0],
        "GPU scatter dup-idx must match torch-CUDA (nondeterministic-on-CUDA op)"
    );
}

/// `scatter_add` on a transposed (non-contiguous) CUDA `self` with NON-ZERO
/// `self` data must match `torch.scatter_add`. A zeros-`self` would MASK the
/// bug (all-zeros reads identically contiguous or strided). torch returns
/// `[[4,4],[2,12],[8,12]]`.
#[test]
fn divergence_scatter_add_transposed_cuda_nonzero_self_f32() {
    ensure_cuda();
    let z = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let zt = z.transpose(0, 1).expect("transpose"); // [3,2] view
    assert!(!zt.is_contiguous());
    let src = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
        .to(Device::Cuda(0))
        .expect("src cuda");
    let index = [0usize, 0, 1, 1, 0, 1];
    let out = scatter_add(&zt, 1, &index, &[3, 2], &src).expect("gpu scatter_add t");
    assert!(out.is_cuda());

    let zt_cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .transpose(0, 1)
        .expect("cpu t");
    let src_cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let cpu_out = scatter_add(&zt_cpu, 1, &index, &[3, 2], &src_cpu).expect("cpu sa t");
    assert_eq!(
        cpu_out.data_vec().unwrap(),
        vec![4.0, 4.0, 2.0, 12.0, 8.0, 12.0]
    );

    // torch.scatter_add(self.t(), 1, idx, src) == [[4,4],[2,12],[8,12]].
    assert_eq!(
        host_f32(&out),
        vec![4.0, 4.0, 2.0, 12.0, 8.0, 12.0],
        "GPU scatter_add into transposed nonzero self must match torch (and CPU)"
    );
}

// ===========================================================================
// HARDER strided cases the fixer might not cover: non-zero storage_offset
// (narrowed views) and 3D permutation. .contiguous()->strided_copy passes
// BOTH src_strides AND src_offset (methods.rs:1583-1589); these prove it.
// Each FAILS against the pre-fix kernel (negative control verified live).
// ===========================================================================

/// `gather` on a NARROWED (non-zero `storage_offset`, non-contiguous) CUDA
/// view. `big[4,6].narrow(1,1,3).narrow(0,1,2)` is a `[2,3]` view with strides
/// `(6,1)` and storage_offset 7 (== `[[7,8,9],[13,14,15]]`). If
/// `.contiguous()` ignored `storage_offset` the gather would read from offset
/// 0 and corrupt the result. torch.gather(v,1,[[0,2,1],[1,0,2]]) ==
/// `[[7,9,8],[14,13,15]]`.
#[test]
fn gather_narrowed_storage_offset_cuda_f32() {
    ensure_cuda();
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let big = cpu_f32(&data, &[4, 6]).to(Device::Cuda(0)).unwrap();
    let v = big
        .narrow(1, 1, 3)
        .expect("narrow dim1")
        .narrow(0, 1, 2)
        .expect("narrow dim0"); // [2,3] view, offset 7, strides (6,1)
    assert!(v.is_cuda());
    assert!(
        !v.is_contiguous(),
        "narrowed inner-dim view must be non-contiguous"
    );
    assert_ne!(v.storage_offset(), 0, "fixture must have nonzero offset");

    let index = [0usize, 2, 1, 1, 0, 2];
    let out = gather(&v, 1, &index, &[2, 3]).expect("gpu gather narrowed");
    assert!(out.is_cuda(), "result must stay GPU-resident");

    // torch.gather(big[1:3,1:4], 1, [[0,2,1],[1,0,2]]) == [[7,9,8],[14,13,15]].
    assert_eq!(
        host_f32(&out),
        vec![7.0, 9.0, 8.0, 14.0, 13.0, 15.0],
        "GPU gather on narrowed (offset!=0) input must match torch"
    );

    let v_cpu = cpu_f32(&data, &[4, 6])
        .narrow(1, 1, 3)
        .unwrap()
        .narrow(0, 1, 2)
        .unwrap();
    let cpu_out = gather(&v_cpu, 1, &index, &[2, 3]).unwrap();
    assert_eq!(host_f32(&out), cpu_out.data_vec().unwrap(), "GPU == CPU");
}

/// `scatter_add` on a TRANSPOSED-NARROWED non-contiguous CUDA `self`.
/// `base[4,6].narrow(1,0,2).narrow(0,0,3).t()` is a `[2,3]` view, strides
/// `(1,6)`, == `[[1,7,13],[2,8,14]]`. dim=0 scatter_add with
/// idx `[[0,0,1],[1,1,0]]`, src `[[1,2,3],[4,5,6]]`.
/// torch (cpu == cuda) == `[[2,9,19],[6,13,17]]`.
#[test]
fn scatter_add_transposed_narrowed_cuda_f32() {
    ensure_cuda();
    let data: Vec<f32> = (1..=24).map(|v| v as f32).collect();
    let base = cpu_f32(&data, &[4, 6]).to(Device::Cuda(0)).unwrap();
    let nc = base
        .narrow(1, 0, 2)
        .expect("narrow dim1")
        .narrow(0, 0, 3)
        .expect("narrow dim0")
        .transpose(0, 1)
        .expect("transpose"); // [2,3] view, strides (1,6)
    assert!(nc.is_cuda());
    assert!(!nc.is_contiguous());

    let src = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .to(Device::Cuda(0))
        .unwrap();
    let index = [0usize, 0, 1, 1, 1, 0];
    let out = scatter_add(&nc, 0, &index, &[2, 3], &src).expect("gpu scatter_add tnarrow");
    assert!(out.is_cuda());

    // torch.scatter_add([[1,7,13],[2,8,14]], 0, [[0,0,1],[1,1,0]], src) == [[2,9,19],[6,13,17]].
    assert_eq!(
        host_f32(&out),
        vec![2.0, 9.0, 19.0, 6.0, 13.0, 17.0],
        "GPU scatter_add on transposed-narrowed self must match torch"
    );

    let nc_cpu = cpu_f32(&data, &[4, 6])
        .narrow(1, 0, 2)
        .unwrap()
        .narrow(0, 0, 3)
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let src_cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let cpu_out = scatter_add(&nc_cpu, 0, &index, &[2, 3], &src_cpu).unwrap();
    assert_eq!(host_f32(&out), cpu_out.data_vec().unwrap(), "GPU == CPU");
}

/// 3D PERMUTED non-contiguous CUDA gather at the permuted (scattered) dim.
/// `arange(24).reshape(2,3,4).permute(2,0,1)` is a `[4,2,3]` view, strides
/// `(1,12,4)`. gather dim=1 with index selecting row 1 of the permuted axis.
/// torch flat == `[0,4,8, 12,16,20, 1,5,9, 13,17,21, 2,6,10, 14,18,22,
/// 3,7,11, 15,19,23]`.
#[test]
fn gather_3d_permuted_cuda_f32() {
    ensure_cuda();
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let t3 = cpu_f32(&data, &[2, 3, 4]).to(Device::Cuda(0)).unwrap();
    let p = t3.permute(&[2, 0, 1]).expect("permute"); // [4,2,3] view, strides (1,12,4)
    assert!(p.is_cuda());
    assert!(!p.is_contiguous(), "permuted view must be non-contiguous");
    assert_eq!(p.shape(), &[4, 2, 3]);

    // index [4,2,3]: row 0 of dim1 -> 0, row 1 -> 1 (g3[:,1,:]=1).
    let mut index = vec![0usize; 24];
    // row-major over [4,2,3]: positions where dim1==1 are flat idx (o*6 + 1*3 + k).
    for o in 0..4 {
        for k in 0..3 {
            index[o * 6 + 3 + k] = 1;
        }
    }
    let out = gather(&p, 1, &index, &[4, 2, 3]).expect("gpu gather 3d permuted");
    assert!(out.is_cuda());

    let expected = vec![
        0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0,
        18.0, 22.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0,
    ];
    assert_eq!(
        host_f32(&out),
        expected,
        "GPU gather on 3D permuted input must match torch"
    );

    let p_cpu = cpu_f32(&data, &[2, 3, 4]).permute(&[2, 0, 1]).unwrap();
    let cpu_out = gather(&p_cpu, 1, &index, &[4, 2, 3]).unwrap();
    assert_eq!(host_f32(&out), cpu_out.data_vec().unwrap(), "GPU == CPU");
}

// ===========================================================================
// REGRESSION GUARDS — hard cases that ARE clean on contiguous inputs.
// These PASS; they pin the harder coverage the builder skipped so a future
// regression is caught.
// ===========================================================================

/// Index smaller than input along the non-gathered axis: input [3,4], gather
/// dim=1 with index [3,2]. Kernel must iterate index-extent, not input-extent.
/// torch: [[1,4],[6,7],[12,9]].
#[test]
fn gather_smaller_index_than_input_f32() {
    ensure_cuda();
    let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let gpu = cpu_f32(&data, &[3, 4]).to(Device::Cuda(0)).unwrap();
    let index = [0usize, 3, 1, 2, 3, 0];
    let out = gather(&gpu, 1, &index, &[3, 2]).expect("gpu gather smaller idx");
    assert!(out.is_cuda());
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(host_f32(&out), vec![1.0, 4.0, 6.0, 7.0, 12.0, 9.0]);
    let cpu_out = gather(&cpu_f32(&data, &[3, 4]), 1, &index, &[3, 2]).unwrap();
    assert_eq!(host_f32(&out), cpu_out.data().unwrap().to_vec());
}

/// Scatter with index smaller than self along non-scattered axis: self [3,4]
/// zeros, dim=1, index [3,2], src [3,2]. Untouched positions must stay 0
/// (in-place clone semantics). torch:
/// [[1,0,0,2],[0,3,4,0],[6,0,0,5]].
#[test]
fn scatter_smaller_index_preserves_untouched_f32() {
    ensure_cuda();
    let gpu = cpu_f32(&[0.0; 12], &[3, 4]).to(Device::Cuda(0)).unwrap();
    let src = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let index = [0usize, 3, 1, 2, 3, 0];
    let out = scatter(&gpu, 1, &index, &[3, 2], &src).expect("gpu scatter smaller idx");
    assert!(out.is_cuda());
    assert_eq!(
        host_f32(&out),
        vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 6.0, 0.0, 0.0, 5.0]
    );
}

/// Large-scale atomic accumulation: 1000 src=1.0 all targeting slot 0 of a
/// [3] zeros along dim 0. Atomic add must lose no updates -> slot0 == 1000.0
/// (exactly f32-representable). A non-atomic / last-write-wins kernel fails.
#[test]
fn scatter_add_large_atomic_no_lost_updates_f32() {
    ensure_cuda();
    let gpu = cpu_f32(&[0.0, 0.0, 0.0], &[3]).to(Device::Cuda(0)).unwrap();
    let src = cpu_f32(&vec![1.0f32; 1000], &[1000])
        .to(Device::Cuda(0))
        .unwrap();
    let index = vec![0usize; 1000];
    let out = scatter_add(&gpu, 0, &index, &[1000], &src).expect("gpu large atomic");
    assert!(out.is_cuda());
    let h = host_f32(&out);
    assert_eq!(h[0], 1000.0, "atomic add must accumulate all 1000 ones");
    assert_eq!(h[1], 0.0);
    assert_eq!(h[2], 0.0);
}

/// f64 large-scale atomic accumulation companion.
#[test]
fn scatter_add_large_atomic_no_lost_updates_f64() {
    ensure_cuda();
    let gpu = cpu_f64(&[0.0, 0.0, 0.0], &[3]).to(Device::Cuda(0)).unwrap();
    let src = cpu_f64(&vec![1.0f64; 1000], &[1000])
        .to(Device::Cuda(0))
        .unwrap();
    let index = vec![0usize; 1000];
    let out = scatter_add(&gpu, 0, &index, &[1000], &src).expect("gpu large atomic f64");
    assert!(out.is_cuda());
    assert_eq!(host_f64(&out)[0], 1000.0);
}

/// scatter vs scatter_value distinction: scatter writes a SRC tensor,
/// scatter_value writes a SCALAR. Confirm the two are not swapped and each
/// matches torch on the SAME index/self.
/// self zeros [3,2] dim=0 idx=[[0,1],[2,0]] (shape [2,2]):
///   scatter src=[[7,8],[9,10]] -> [[7,10],[0,8],[9,0]]
///   scatter_value 5.0          -> [[5,5],[0,5],[5,0]]
#[test]
fn scatter_vs_scatter_value_not_swapped_f32() {
    ensure_cuda();
    let base = cpu_f32(&[0.0; 6], &[3, 2]);
    let gpu = base.clone().to(Device::Cuda(0)).unwrap();
    let index = [0usize, 1, 2, 0];

    // scatter (tensor src)
    let src = cpu_f32(&[7.0, 8.0, 9.0, 10.0], &[2, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let s_out = scatter(&gpu, 0, &index, &[2, 2], &src).expect("gpu scatter");
    let s_ref = scatter(
        &base,
        0,
        &index,
        &[2, 2],
        &cpu_f32(&[7.0, 8.0, 9.0, 10.0], &[2, 2]),
    )
    .unwrap();
    assert_eq!(host_f32(&s_out), s_ref.data().unwrap().to_vec());
    assert_eq!(host_f32(&s_out), vec![7.0, 10.0, 0.0, 8.0, 9.0, 0.0]);

    // scatter_value (scalar)
    let gpu2 = base.clone().to(Device::Cuda(0)).unwrap();
    let v_out = scatter_value(&gpu2, 0, &index, &[2, 2], 5.0f32).expect("gpu scatter_value");
    assert_eq!(host_f32(&v_out), vec![5.0, 5.0, 0.0, 5.0, 5.0, 0.0]);
}

/// scatter must PRESERVE the untouched positions of `self` (in-place clone
/// semantics, not zero-fill). self = [1..6] reshaped [2,3], dim=1, write
/// src=[[10],[20]] at idx [[2],[0]]. torch: [[1,2,10],[20,5,6]].
#[test]
fn scatter_preserves_self_untouched_positions_f32() {
    ensure_cuda();
    let base = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let gpu = base.clone().to(Device::Cuda(0)).unwrap();
    let src = cpu_f32(&[10.0, 20.0], &[2, 1]).to(Device::Cuda(0)).unwrap();
    let index = [2usize, 0];
    let out = scatter(&gpu, 1, &index, &[2, 1], &src).expect("gpu scatter preserve");
    assert!(out.is_cuda());
    assert_eq!(host_f32(&out), vec![1.0, 2.0, 10.0, 20.0, 5.0, 6.0]);
}

/// 3D non-square [2,5,3], gather at each axis. Verifies the
/// outer/axis/inner factorisation does not swap a row/col stride.
#[test]
fn gather_3d_nonsquare_each_axis_f32() {
    ensure_cuda();
    let data: Vec<f32> = (0..30).map(|v| v as f32).collect(); // [2,5,3]
    let gpu = cpu_f32(&data, &[2, 5, 3]).to(Device::Cuda(0)).unwrap();
    let cpu = cpu_f32(&data, &[2, 5, 3]);

    // dim=0, index [1,5,3] all zeros -> picks slab 0 == arange(0..15).
    let i0 = vec![0usize; 15];
    let o0 = gather(&gpu, 0, &i0, &[1, 5, 3]).unwrap();
    let r0 = gather(&cpu, 0, &i0, &[1, 5, 3]).unwrap();
    assert_eq!(host_f32(&o0), r0.data().unwrap().to_vec());
    assert_eq!(o0.shape(), &[1, 5, 3]);

    // dim=1, index [2,1,3] all zeros -> first row of each slab.
    let i1 = vec![0usize; 6];
    let o1 = gather(&gpu, 1, &i1, &[2, 1, 3]).unwrap();
    let r1 = gather(&cpu, 1, &i1, &[2, 1, 3]).unwrap();
    assert_eq!(host_f32(&o1), r1.data().unwrap().to_vec());

    // dim=2, index [2,5,1] all index 2 -> last col of each row.
    let i2 = vec![2usize; 10];
    let o2 = gather(&gpu, 2, &i2, &[2, 5, 1]).unwrap();
    let r2 = gather(&cpu, 2, &i2, &[2, 5, 1]).unwrap();
    assert_eq!(host_f32(&o2), r2.data().unwrap().to_vec());
}

/// PyTorch CUDA supports `gather` for f16 and bf16, including duplicate-index
/// gradient accumulation. This used to be a stale rejection assertion from the
/// pre-#1822 surface; keep it as a real parity guard so the CUDA half paths
/// cannot regress or silently demote gradients to host.
#[test]
fn gather_bf16_f16_cuda_forward_backward_matches_torch() {
    ensure_cuda();

    let f16_input = cuda_f16(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let bf16_input = cuda_bf16(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let index = [0usize, 0, 1, 0]; // torch.gather(x, 0, [[0,0],[1,0]])

    let f16_out = gather(&f16_input, 0, &index, &[2, 2]).expect("f16 gather");
    assert!(
        f16_out.is_cuda(),
        "f16 gather output must stay CUDA-resident"
    );
    assert_eq!(host_f16(&f16_out), vec![1.0, 2.0, 3.0, 2.0]);

    let bf16_out = gather(&bf16_input, 0, &index, &[2, 2]).expect("bf16 gather");
    assert!(
        bf16_out.is_cuda(),
        "bf16 gather output must stay CUDA-resident"
    );
    assert_eq!(host_bf16(&bf16_out), vec![1.0, 2.0, 3.0, 2.0]);

    let f16_grad_out = cuda_f16(&[1.0; 4], &[2, 2], false);
    let f16_grads = f16_out
        .grad_fn()
        .expect("tracked f16 gather must carry grad_fn")
        .backward(&f16_grad_out)
        .expect("f16 gather backward");
    let f16_grad = f16_grads[0].as_ref().expect("f16 input grad");
    assert!(
        f16_grad.is_cuda(),
        "f16 gather grad must stay CUDA-resident"
    );
    assert_eq!(host_f16(f16_grad), vec![1.0, 2.0, 1.0, 0.0]);

    let bf16_grad_out = cuda_bf16(&[1.0; 4], &[2, 2], false);
    let bf16_grads = bf16_out
        .grad_fn()
        .expect("tracked bf16 gather must carry grad_fn")
        .backward(&bf16_grad_out)
        .expect("bf16 gather backward");
    let bf16_grad = bf16_grads[0].as_ref().expect("bf16 input grad");
    assert!(
        bf16_grad.is_cuda(),
        "bf16 gather grad must stay CUDA-resident"
    );
    assert_eq!(host_bf16(bf16_grad), vec![1.0, 2.0, 1.0, 0.0]);
}
