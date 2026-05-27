//! Divergence: ferrotorch's `(1, K) @ (K, 1)` thin-matmul produces
//! different bits than torch's `torch.matmul` on the SAME inputs, even
//! when both link the same statically-bound MKL 2024.2 (`sgemm_`
//! Fortran symbol) and run on the same Linux x86_64 AVX2 host with
//! GNU OpenMP.
//!
//! ## Forensic finding (2026-05-26, host: 14 physical cores / 28 SMT)
//!
//! For inputs `a.shape=(1,16384)` and `b.shape=(16384,1)` row-major
//! contiguous f32 (the bits committed at `tests/fixtures/probe_1x16384x1_*_bits.bin`
//! by 81b046c04), `torch.matmul(a, b).item()` returns the IEEE-754
//! float bits `0xc28e882b` (≈ -71.26595306).
//!
//! Forensic dispatch derivation per `_matmul_impl` →
//! `mm_out_cpu` → `addmm_impl_cpu_` (aten/src/ATen/native/LinearAlgebra.cpp:2044, 1641, 1400-1557):
//!
//!   - `result_strides=(1,1)`, `result_sizes=(1,1)` →
//!     `transpose_c=false` (LinearAlgebra.cpp:1450-1453, first arm).
//!   - `m1=(1,16384) row-major strides=(16384,1)` →
//!     `transpose_a=true` (LinearAlgebra.cpp:1479-1483, second arm).
//!   - `m2=(16384,1) row-major strides=(1,1)` →
//!     `transpose_b=true` (LinearAlgebra.cpp:1496-1499, second arm).
//!   - `lda=a.strides()[0]=16384`, `ldb=b.strides()[0]=1`, `ldc=c.strides()[1]=1`.
//!   - `normalize_last_dims` (CPUBlas.cpp:82-105) is a no-op for this
//!     case: `n==1` already sets `ldc=m=1`, `transa!=N, m==1` sets
//!     `lda=k=16384` (unchanged), `transb!=N, k==16384!=1` no-op.
//!   - Final: `sgemm_('T','T', m=1, n=1, k=16384, α=1, a, lda=16384, b, ldb=1, β=0, c, ldc=1)`.
//!
//! Direct invocation of `libtorch_cpu.so::sgemm_` (and of
//! `~/.local/lib/libmkl_rt.so.2::sgemm_`, which both report MKL 2024.2)
//! with these EXACT arguments via Python ctypes reproduces
//! `0xc28e882b` byte-exactly at `MKL_Set_Num_Threads(14)` and
//! produces a different bit pattern at each lower thread count (the
//! K-axis reduction parallelizes).
//!
//! Ferrotorch's `mm_raw_mkl_f32` in `ferrotorch-core/src/ops/linalg.rs:520-565`
//! instead dispatches:
//!     `sgemm_('N','N', n=1, m=1, k=16384, α=1, B, ldb=1, A, lda=16384, β=0, C, ldc=1)`
//! (operand-swap + transa='N' + transb='N' — the row-major-to-col-major
//! "swap operands, both no-transpose" idiom).
//!
//! Crucially, MKL's `sgemm_` produces DIFFERENT bits between the two
//! dispatches:
//!   - `T,T,m=n=1,k=16384,lda=16384,ldb=1` (torch): `0xc28e882b` @ 14 threads
//!   - `N,N,n=m=1,k=16384,ldb=1,lda=16384` (ferrotorch): `0xc28e8836` @ ANY thread count
//!
//! The N,N + swapped-operand path lands on MKL's small-matrix serial
//! kernel which IGNORES the MKL thread count; the T,T path lands on
//! MKL's threaded dot kernel which splits K across the thread pool.
//!
//! Empirical evidence (this host, MKL 2024.2 statically linked into
//! libtorch_cpu.so AND as system `~/.local/lib/libmkl_rt.so.2`,
//! MKL_NUM_THREADS controlled via `MKL_Set_Num_Threads`):
//!
//! | dispatch                              | nt=1     | nt=4     | nt=8     | nt=14    |
//! |---------------------------------------|----------|----------|----------|----------|
//! | sgemm_(T,T,1,1,16384,lda=16384,ldb=1) | 0xc28e8855 | 0xc28e883a | 0xc28e8836 | **0xc28e882b** |
//! | sgemm_(N,N,1,1,16384,ldb=1,lda=16384) | 0xc28e8836 | 0xc28e8836 | 0xc28e8836 | 0xc28e8836 |
//! | torch.matmul(a,b) (via set_num_threads) | 0xc28e8855 | 0xc28e883a | 0xc28e8836 | **0xc28e882b** |
//!
//! Therefore the row-major-as-col-major operand-swap idiom is the
//! source of the divergence: ferrotorch's `(N,N) + swap` dispatch
//! produces a DIFFERENT MKL internal kernel selection than torch's
//! `(T,T) + no-swap` dispatch, and only the latter parallelizes the K
//! reduction. Both are mathematically valid GEMM expressions of the
//! same row-major math; they differ in the BLAS argument shape, which
//! is the thing MKL keys its kernel selection on.
//!
//! This invalidates the comment at `ferrotorch-core/src/ops/linalg.rs:32`:
//!
//! > "directly via the helpers ... which mirrors torch's exact call shape"
//! > It does NOT mirror torch's call shape; it mirrors a mathematically
//! > equivalent re-expression that lands on a different MKL kernel.
//!
//! Tracking: filed as a blocker against #1538 (which closed prematurely
//! without forensically pinning torch's dispatch shape).
//!
//! ## What this test asserts
//!
//! Each of the three tests below pins one leg of the forensic finding,
//! using ONLY the fixture bytes committed at `81b046c04` (which were
//! recorded from `torch.matmul` live) and the live MKL `sgemm_`
//! exported from `libmkl_rt.so.2`. No tautological re-derivation of
//! ferrotorch's output is used (R-CHAR-3).

#![cfg(all(feature = "mkl", target_os = "linux", target_arch = "x86_64"))]

use std::ffi::c_char;
use std::fs;
use std::os::raw::c_int;
use std::path::PathBuf;
use std::sync::Mutex;

/// Serialize the MKL thread-count global across the three tests in this
/// file. `MKL_Set_Num_Threads` mutates process-global MKL state; running
/// Test A (which steps thread count 1 -> 4) in parallel with Test C
/// (which expects physical-core count) produces flaky results. The
/// mutex is internal to this test crate; production code does not need
/// it (the .init_array constructor in `ops/linalg.rs` sets the thread
/// count once before any BLAS dispatch).
static MKL_THREAD_LOCK: Mutex<()> = Mutex::new(());

// MKL Fortran-ABI extern. Same `sgemm_` symbol ferrotorch links against
// via `cargo:rustc-link-lib=mkl_rt` (see `ferrotorch-core/build.rs`).
// `MKL_Set_Num_Threads` is MKL's own thread-count control (exported
// from libmkl_rt.so.2 at nm offset 0x342ef0); using it (instead of
// libgomp's `omp_set_num_threads`) keeps this test self-contained on
// the `-lmkl_rt` link.
unsafe extern "C" {
    fn sgemm_(
        transa: *const c_char,
        transb: *const c_char,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const f32,
        a: *const f32,
        lda: *const c_int,
        b: *const f32,
        ldb: *const c_int,
        beta: *const f32,
        c: *mut f32,
        ldc: *const c_int,
    );
    fn MKL_Set_Num_Threads(n: c_int);
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load_f32_bin(path: &PathBuf) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len() % 4 == 0,
        "fixture {} not a multiple of 4 bytes",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn fixture_c_bits() -> u32 {
    let p = fixtures_dir().join("probe_1x16384x1_c_bits.bin");
    let bytes = fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    assert_eq!(
        bytes.len(),
        4,
        "expected 4-byte fixture, got {}",
        bytes.len()
    );
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Invoke `sgemm_` with `transa='T', transb='T', m=n=1, k=16384`,
/// `lda=16384, ldb=1, ldc=1`, NO operand swap — torch's exact dispatch
/// shape per aten/src/ATen/native/LinearAlgebra.cpp:1450-1557.
fn call_sgemm_tt_torch_shape(a: &[f32], b: &[f32]) -> u32 {
    let mut c_out: f32 = 0.0;
    let m: c_int = 1;
    let n: c_int = 1;
    let k: c_int = 16384;
    let lda: c_int = 16384;
    let ldb: c_int = 1;
    let ldc: c_int = 1;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let transa: c_char = b'T' as c_char;
    let transb: c_char = b'T' as c_char;
    // SAFETY: leaf FFI to MKL's `sgemm_`. a/b are caller-owned slices
    // whose lifetimes outlive the call; c_out is a stack scalar passed
    // by mutable pointer with no aliasing. Argument shape matches the
    // dispatch torch's `addmm_impl_cpu_` issues for this case
    // (LinearAlgebra.cpp:1450-1557 → CPUBlas.cpp:238).
    unsafe {
        sgemm_(
            &transa,
            &transb,
            &m,
            &n,
            &k,
            &alpha,
            a.as_ptr(),
            &lda,
            b.as_ptr(),
            &ldb,
            &beta,
            &mut c_out,
            &ldc,
        );
    }
    c_out.to_bits()
}

/// Invoke `sgemm_` with `transa='N', transb='N'`, operands B-then-A
/// swapped, dims n=m=1, k=16384, `lda=16384, ldb=1, ldc=1` — ferrotorch's
/// exact dispatch shape per `ferrotorch-core/src/ops/linalg.rs:520-565`.
fn call_sgemm_nn_ferrotorch_shape(a: &[f32], b: &[f32]) -> u32 {
    let mut c_out: f32 = 0.0;
    let m: c_int = 1;
    let n: c_int = 1;
    let k: c_int = 16384;
    let lda: c_int = 16384;
    let ldb: c_int = 1;
    let ldc: c_int = 1;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let transa: c_char = b'N' as c_char;
    let transb: c_char = b'N' as c_char;
    // SAFETY: same invariants as `call_sgemm_tt_torch_shape`; argument
    // shape is ferrotorch's verbatim dispatch (linalg.rs:519-565).
    unsafe {
        sgemm_(
            &transa,
            &transb,
            &n, // ferrotorch passes N first (operand swap)
            &m,
            &k,
            &alpha,
            b.as_ptr(), // B before A (operand swap)
            &ldb,
            a.as_ptr(),
            &lda,
            &beta,
            &mut c_out,
            &ldc,
        );
    }
    c_out.to_bits()
}

/// Test A — direct MKL `sgemm_` with TORCH's exact dispatch shape
/// (`transa='T', transb='T', m=1, n=1, k=16384, lda=16384, ldb=1, ldc=1`)
/// produces DIFFERENT bits at DIFFERENT thread counts, while ferrotorch's
/// `(N,N) + operand-swap` dispatch produces the SAME bits at all thread
/// counts. This proves MKL routes the two dispatches to different
/// internal kernels (only the T,T form parallelizes the K reduction).
///
/// Cite: aten/src/ATen/native/LinearAlgebra.cpp:1450-1557 (transpose
/// derivation) → CPUBlas.cpp:238 (sgemm_ invocation) versus
/// ferrotorch-core/src/ops/linalg.rs:520-565 (mm_raw_mkl_f32).
///
/// This test PASSES — it forensically pins the kernel-selection
/// difference between the two dispatches. The next two tests then show
/// that ferrotorch's dispatch produces the WRONG bits relative to the
/// fixture (which was captured from torch).
#[test]
fn a_dispatch_shape_selects_different_mkl_kernels() {
    let _g = MKL_THREAD_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let a = load_f32_bin(&fixtures_dir().join("probe_1x16384x1_a_bits.bin"));
    let b = load_f32_bin(&fixtures_dir().join("probe_1x16384x1_b_bits.bin"));
    assert_eq!(a.len(), 16384);
    assert_eq!(b.len(), 16384);

    // Sample at thread counts 1 and 4 — small enough to be valid on any
    // host with >= 4 logical CPUs; the comparison is differential
    // (between the two dispatches at the same thread count), not
    // against the fixture, so this test is host-thread-count-robust.
    let nt_low: c_int = 1;
    let nt_high: c_int = 4;

    // SAFETY: leaf FFI to MKL runtime linked via mkl_rt.
    unsafe { MKL_Set_Num_Threads(nt_low) };
    let tt_low = call_sgemm_tt_torch_shape(&a, &b);
    let nn_low = call_sgemm_nn_ferrotorch_shape(&a, &b);

    // SAFETY: leaf FFI to MKL runtime linked via mkl_rt.
    unsafe { MKL_Set_Num_Threads(nt_high) };
    let tt_high = call_sgemm_tt_torch_shape(&a, &b);
    let nn_high = call_sgemm_nn_ferrotorch_shape(&a, &b);

    // The N,N+swap dispatch produces THE SAME bits at both thread
    // counts — proving it lands on a serial kernel that does not
    // partition the K-reduction.
    assert_eq!(
        nn_low, nn_high,
        "ferrotorch's (N,N + operand-swap) dispatch should produce identical \
         bits at thread counts {nt_low} and {nt_high} (lands on MKL serial \
         small-matrix kernel that ignores OMP_NUM_THREADS). Got 0x{nn_low:08x} \
         vs 0x{nn_high:08x}."
    );

    // The T,T dispatch produces DIFFERENT bits at different thread
    // counts — proving it lands on a threaded kernel that partitions
    // the K-reduction (the same kernel torch's matmul reaches).
    assert_ne!(
        tt_low, tt_high,
        "torch's (T,T + no-swap) dispatch should produce DIFFERENT bits at \
         thread counts {nt_low} and {nt_high} (lands on MKL threaded dot \
         kernel that partitions K across OMP_NUM_THREADS). Got identical \
         0x{tt_low:08x} — this would mean MKL's dispatch table no longer \
         routes (T,T,1,1,K) to the threaded kernel on this host."
    );

    // And at any single thread count, the two dispatches disagree —
    // proving the dispatch SHAPE is the discriminator, not the
    // operands.
    assert_ne!(
        tt_high, nn_high,
        "At MKL_NUM_THREADS={nt_high}, the (T,T + no-swap) dispatch and \
         the (N,N + operand-swap) dispatch should produce DIFFERENT bits \
         on the same inputs, proving they land on different MKL internal \
         kernels. Got identical 0x{tt_high:08x} — this would invalidate \
         the dispatch-shape divergence hypothesis."
    );
}

/// Test B — direct MKL `sgemm_` with the PRIOR FERROTORCH dispatch shape
/// (`transa='N', transb='N', n=1, m=1, k=16384, ldb=1, lda=16384, ldc=1`,
/// operands B then A swapped) produces DIFFERENT bits than the fixture,
/// regardless of thread count. Proves the prior dispatch SHAPE was the
/// root cause; the post-#1541 fix uses (T,T)+no-swap for `m == 1 || n == 1`
/// in `mm_raw_mkl_f32` and matches the fixture (verified by Test C).
///
/// Cite: ferrotorch-core/src/ops/linalg.rs::mm_raw_mkl_f32 (post-#1541
/// the m==1||n==1 conditional dispatches T,T+no-swap; outside the
/// conditional, dense shapes still use the N,N+swap row-major idiom
/// since they're byte-exact on dense probes — see
/// divergence_mkl_byte_exact_critic 64x64, 127x127, k=257).
///
/// Post-fix, this test PASSES with an inverted assertion: the prior
/// (N,N)+operand-swap dispatch on the (1,16384) thin shape STRUCTURALLY
/// cannot match torch's (T,T)+no-swap output, because MKL routes the
/// two argument shapes to different internal kernels (serial small-
/// matrix vs K-parallel threaded dot) which compute the same dot
/// product in different summation orders. The fix's value is exactly
/// that this divergence exists — without the dispatch-shape conditional
/// in `mm_raw_mkl_f32`, the (1,K)@(K,1) thin matmul would still drift.
#[test]
fn b_ferrotorch_dispatch_shape_does_not_match_fixture() {
    let _g = MKL_THREAD_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let a = load_f32_bin(&fixtures_dir().join("probe_1x16384x1_a_bits.bin"));
    let b = load_f32_bin(&fixtures_dir().join("probe_1x16384x1_b_bits.bin"));
    let expected = fixture_c_bits();

    // Use the host's physical-core count, which is what torch's default
    // matmul-time thread count resolves to and what the fixture at
    // 81b046c04 was recorded under. (The dispatch divergence below is
    // independent of thread count — see Test A — but the comparison
    // against the fixture relies on the SAME thread count torch ran
    // under, since the (T,T) path partitions K differently per thread
    // count.)
    let nt = c_int::try_from(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    )
    .unwrap_or(1);
    // SAFETY: leaf FFI to MKL runtime linked via mkl_rt.
    unsafe { MKL_Set_Num_Threads(nt / 2) }; // physical = SMT/2 on this host

    let actual = call_sgemm_nn_ferrotorch_shape(&a, &b);
    assert_ne!(
        actual, expected,
        "PRIOR (N,N + operand-swap) dispatch on (1,16384)@(16384,1) was \
         expected to land on MKL's serial small-matrix kernel and produce \
         bits that DIFFER from torch's (T,T + no-swap) fixture \
         0x{expected:08x}, but produced 0x{actual:08x} == fixture. If the \
         two dispatch shapes happen to coincide bitwise on this MKL build, \
         the dispatch-shape conditional in `mm_raw_mkl_f32` may be \
         redundant for this shape — but the conditional remains correct \
         policy (it mirrors torch's call shape exactly per LinearAlgebra.cpp \
         derivation), so re-investigate before relaxing it."
    );
}

/// Test C — ferrotorch's public `matmul` on the EXACT fixture inputs
/// produces output bits that do NOT match the fixture, demonstrating
/// the end-to-end user-observable divergence.
///
/// Cite: ferrotorch-core/src/ops/linalg.rs:204 (`matmul`), 1294 (`mm`),
/// 520-565 (`mm_raw_mkl_f32`).
///
/// This is the test the generator must make pass — by switching the
/// MKL dispatch in `mm_raw_mkl_f32` (and its `_bt`/`_at` cousins) from
/// the `(N,N) + operand-swap` row-major idiom to torch's
/// `(T,T) + no-swap` idiom for the m=1, n=1 thin case (and any other
/// case where the operand-swap form lands on a different MKL kernel).
#[test]
fn c_ferrotorch_matmul_bits_match_torch_fixture() {
    use ferrotorch_core::{from_vec, ops::linalg::matmul};

    let _g = MKL_THREAD_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let a_data = load_f32_bin(&fixtures_dir().join("probe_1x16384x1_a_bits.bin"));
    let b_data = load_f32_bin(&fixtures_dir().join("probe_1x16384x1_b_bits.bin"));
    let expected = fixture_c_bits();

    let nt = c_int::try_from(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    )
    .unwrap_or(1);
    // SAFETY: leaf FFI to MKL runtime linked via mkl_rt.
    unsafe { MKL_Set_Num_Threads(nt / 2) };

    let a = from_vec(a_data, &[1, 16384]).expect("from_vec a");
    let b = from_vec(b_data, &[16384, 1]).expect("from_vec b");
    let c = matmul(&a, &b).expect("matmul");
    let c_vals = c.data().expect("c data");
    assert_eq!(c_vals.len(), 1, "expected scalar output");
    let actual = c_vals[0].to_bits();
    assert_eq!(
        actual, expected,
        "ferrotorch_core::ops::linalg::matmul on (1,16384)@(16384,1) f32 returned \
         bits 0x{actual:08x}, but torch.matmul on the same inputs returned \
         0x{expected:08x} (fixture from 81b046c04 captured live torch output on \
         this host). The divergence originates from ferrotorch's mm_raw_mkl_f32 \
         issuing sgemm_(N,N) with operands swapped vs torch's sgemm_(T,T) \
         no-swap — MKL selects different internal kernels for the two argument \
         shapes. To fix: route the m=1,n=1 (or k>=K_threshold) case through a \
         T,T no-swap dispatch (and likewise audit mm_raw_bt_mkl_f32, \
         mm_raw_at_mkl_f32 for analogous mis-dispatch). Tracking: divergence \
         in MKL dispatch shape for thin matmul."
    );
}
