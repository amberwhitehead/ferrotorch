//! Linear algebra operations bridging to ferray-linalg.
//!
//! Constructs ferray `Array` views from tensor data slices, calls
//! ferray-linalg operations, and wraps the results back into tensors.
//!
//! ## REQ status (per `.design/ferrotorch-core/ops/linalg.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`matmul` shape dispatcher) | SHIPPED | mirrors `Tensor matmul` and the six-arm dispatch in `_matmul_impl`; non-test consumer is the CPU fallback `linalg::matmul` invocation inside `grad_fns::linalg::matmul_differentiable` after the GPU branches are exhausted; in-file tests cover AC-1..AC-10. |
//! | REQ-2 (`broadcast_matmul`) | SHIPPED-with-known-drift | mirrors the `expand_batch_product` + `bmm` path; reached from `matmul` for the `_ => broadcast_matmul(a, b)` (>=3D) arm; known drift vs PyTorch BLAS (~1.5e-5 on f32 k=10) tracked under blocker #1347; the plan routes per-batch slices through `mm_raw`. |
//! | REQ-3 (`bmm`) | SHIPPED-with-known-drift | mirrors `TORCH_IMPL_FUNC(bmm_out_cpu)`; carries the same accumulation-drift property as REQ-2; reached from `matmul`'s `broadcast_matmul` path; the autograd wrapper `bmm_differentiable` reimplements the CPU triple loop inline today — the planned #1347 fix routes BOTH `broadcast_matmul` and that fallback through `mm_raw`; indirect parity `[bmm] 8/8 passed`. |
//! | REQ-4 (`mm`) | SHIPPED | mirrors `TORCH_IMPL_FUNC(mm_out_cpu)`; non-test consumers `complex_tensor::matmul` and the `matmul` dispatcher's `(2, 2) => mm(a, b)` arm; indirect parity via the `grad_fns` runner arm. |
//! | REQ-5 (`mm_raw`) | SHIPPED | the BLAS-routed workhorse; non-test consumers `mm` itself, `grad_fns::linalg::mm_differentiable`, and `MmBackward`; mirrors upstream's gemm-via-cpublas size-gated dispatch. |
//! | REQ-6 (`mm_raw_bt`) | SHIPPED | fused `A @ B^T`; non-test consumer `MmBackward::backward` for `dA = grad_C @ B^T` plus two other grad-fns sites; mirrors upstream `gemm_transb_` family. |
//! | REQ-7 (`mm_raw_at`) | SHIPPED | fused `A^T @ B`; non-test consumer `MmBackward::backward` for `dB = A^T @ grad_C` plus two other grad-fns sites; mirrors upstream `gemm_transa_` family. |
//! | REQ-8 (`mv`) | SHIPPED | mirrors `Tensor mv`; non-test consumers `MvBackward::backward`, `MmBackward` (mat @ vec case), and `matmul`'s `(2, 1) => mv(a, b)` arm. |
//! | REQ-9 (`dot`) | SHIPPED | mirrors `Tensor dot`; non-test consumer is `matmul`'s `(1, 1) => dot(a, b)` arm; in-file `test_dot`. |
//! | REQ-10 (`transpose`) | SHIPPED | materialises a row-major 2-D transpose (R-DEV-7 deviation from upstream's view); non-test consumer `MmBackward::backward` for `A^T @ grad_C` setup. |
//! | REQ-11 (private broadcast helpers) | SHIPPED | `broadcast_batch_shapes`, `broadcast_strides`, `batch_linear_index`; consumed only by `broadcast_matmul`; exercised indirectly through `test_matmul_*_broadcast` and `test_matmul_4d`. |
//! | REQ-12 (bf16 precision helpers) | SHIPPED | `is_bf16::<T>()`, `as_bf16_slice`, `write_f32_as_bf16` — all `#[inline(always)]`; consumed by all three small-matrix paths in `mm_raw` / `mm_raw_bt` / `mm_raw_at`; per-block `// SAFETY:` comments name the `TypeId`-guard invariant. |
//! | REQ-13 (`MKL_ENABLED` runtime cfg probe + Fortran sgemm_/dgemm_ FFI path) | SHIPPED under `--features mkl` | const `MKL_ENABLED: bool` in `ops/linalg.rs` (true iff built with `--features mkl`); `mm_raw` / `mm_raw_bt` / `mm_raw_at` f32 and f64 branches gain a `#[cfg(feature = "mkl")]` fork that calls the Fortran `sgemm_` / `dgemm_` symbols of system MKL 2024.x directly via the helpers `mm_raw_mkl_f32` / `mm_raw_bt_mkl_f32` / `mm_raw_at_mkl_f32` and f64 mirrors. The dispatcher mirrors torch's exact call shape at `aten/src/ATen/native/CPUBlas.cpp:215-247` (raw `sgemm_` with operand-swap + dim-swap + lda/ldb-swap to convert ferrotorch's row-major into the col-major equivalent torch dispatches). Non-test production consumers identical to REQ-5/6/7 (the same `grad_fns::linalg` and `MmBackward` call-sites pick up the MKL path transparently when the feature is on). The parity-sweep runner `tolerance_for` reads `MKL_ENABLED` at runtime to tighten the matmul-family envelope from `rtol=1e-4` (faer fallback) to `tol_f32()` (the default `(1e-5, 1e-7)`). Closes #1538 and #1348. |

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Compile-time-resolved flag exposed to runtime callers (notably the
/// parity-sweep runner's `tolerance_for`). When `--features mkl` is on
/// for ferrotorch-core, the `mm_raw` family routes f32/f64 through the
/// Fortran `sgemm_`/`dgemm_` symbols of system MKL 2024.x using torch's
/// exact dispatch shape, so parity vs PyTorch's MKL link is
/// byte-for-byte; when off, faer is the BLAS backend and the
/// matmul-family parity envelope widens to `rtol=1e-4` to absorb the
/// cross-BLAS-implementation f32 ULP variance documented in
/// `tools/parity-sweep/runner/src/main.rs::tolerance_for`.
///
/// Read by `tools/parity-sweep/runner/src/main.rs::tolerance_for` at
/// runtime so the downstream parity-sweep crate does not need its own
/// `mkl` Cargo feature declaration; passing `--features
/// ferrotorch-core/mkl` to a `cargo run -p parity-sweep-runner`
/// invocation is sufficient to flip both the FFI path AND the parity
/// envelope. Closes #1538 + #1348.
#[cfg(feature = "mkl")]
pub const MKL_ENABLED: bool = true;

/// See `MKL_ENABLED` doc-comment for the `--features mkl` branch.
#[cfg(not(feature = "mkl"))]
pub const MKL_ENABLED: bool = false;

// FFI declarations for the Fortran-ABI BLAS symbols `sgemm_` and
// `dgemm_` as exported by `libmkl_rt.so.2`. These are the same symbols
// PyTorch's `aten/src/ATen/native/CPUBlas.cpp:215-247` calls into for
// CPU `at::mm` / `at::bmm` on Linux. Calling the Fortran symbol
// directly (rather than `cblas_sgemm`) is critical for byte-exact
// parity: the cblas row-major wrapper picks different MKL micro-
// kernels than the raw Fortran path even on the same column-major-
// equivalent shapes (root cause of issue #1538's 1-ULP `mm_raw_at`
// drift in the prior dispatch).
//
// Fortran calling convention: all scalar arguments are passed
// pass-by-reference (i.e. as `*const T` / `*mut T`), including the
// integer dimensions, the alpha/beta scalars, and the trans*
// character flags. The `c_char` ASCII byte must be `'N'` (`0x4E`) or
// `'T'` (`0x54`) per the BLAS spec.
#[cfg(feature = "mkl")]
unsafe extern "C" {
    fn sgemm_(
        transa: *const std::ffi::c_char,
        transb: *const std::ffi::c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        b: *const f32,
        ldb: *const i32,
        beta: *const f32,
        c: *mut f32,
        ldc: *const i32,
    );
    fn dgemm_(
        transa: *const std::ffi::c_char,
        transb: *const std::ffi::c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
}

/// MKL runtime alignment for byte-exact parity with torch (#1541, #1538).
///
/// Mirrors three pieces of torch's MKL initialization so the same
/// `sgemm_(T,T,...)` thin-matmul dispatch produces byte-identical
/// output between ferrotorch and torch on the same host:
///
/// 1. **Threading layer = GNU** (`MKL_THREADING_LAYER=GNU`). MKL ships
///    two K-parallel kernel variants: `intel_thread` (libiomp5 OpenMP)
///    and `gnu_thread` (libgomp OpenMP). They use DIFFERENT
///    K-reduction summation orders. PyTorch's CPU wheel is built with
///    libgomp (verified via `MKL_VERBOSE=1` showing `gnu_thread` in
///    torch's banner). Without this env, MKL defaults to `intel_thread`
///    on Linux when libiomp5 is present, producing 2-ULP drift vs
///    torch on the (1,16384)@(16384,1) thin probe at the same thread
///    count.
///
/// 2. **Dynamic disabled** (`MKL_DYNAMIC=FALSE`). PyTorch sets
///    `MKL_DYNAMIC=FALSE` so the dispatcher uses the exact thread
///    count requested rather than letting MKL trim threads per call.
///    The dynamic adjustment can change the K partition between calls
///    even at the same `MKL_NUM_THREADS`, producing run-to-run drift.
///    Verified via `MKL_VERBOSE` output `Dyn:0` (torch) vs `Dyn:1`
///    (ferrotorch default).
///
/// 3. **Thread count = physical cores**. Torch's
///    `intraop_default_num_threads()`
///    (`aten/src/ATen/ParallelCommon.cpp:103-133`) resolves to the
///    cpuinfo physical-core count (14 on this host's 14C/28T CPU)
///    when neither `OMP_NUM_THREADS` nor `MKL_NUM_THREADS` is set.
///    libgomp's default would be `sysconf(_SC_NPROCESSORS_ONLN) = 28`
///    (logical cores) — the wrong K partition. We count distinct
///    `(physical id, core id)` tuples in `/proc/cpuinfo` (mirrors
///    cpuinfo's `cores_count` semantics that
///    `c10/core/thread_pool.cpp::defaultNumThreads` consumes) and
///    set both `OMP_NUM_THREADS` and `MKL_NUM_THREADS`.
///
/// All three settings are skipped when the user has explicitly set the
/// corresponding env var (matches torch's resolution chain at
/// ParallelCommon.cpp:110-111, which respects pre-existing
/// `OMP_NUM_THREADS` / `MKL_NUM_THREADS`).
///
/// This `.init_array` constructor runs before MKL's own static ctors
/// read these env vars, so the settings are honored on the first BLAS
/// dispatch in the process.
///
/// Gated `target_os = "linux"` only — Linux ships `/proc/cpuinfo` and
/// is the only OS where the `.init_array` ELF section is honored.
/// macOS / Windows are handled by torch via different `defaultNumThreads`
/// branches (see ParallelCommon.cpp:116-127 for Apple Silicon) and are
/// not part of the #1538 byte-exact host set.
#[cfg(all(feature = "mkl", target_os = "linux"))]
mod mkl_thread_align {
    unsafe extern "C" {
        fn setenv(
            name: *const std::ffi::c_char,
            value: *const std::ffi::c_char,
            overwrite: i32,
        ) -> i32;
        fn getenv(name: *const std::ffi::c_char) -> *const std::ffi::c_char;
        fn sysconf(name: i32) -> isize;
    }

    /// Linux glibc value for `_SC_NPROCESSORS_ONLN` (number of
    /// online logical processors). Used only as a fallback when
    /// `/proc/cpuinfo` lacks `physical id`/`core id` lines (e.g. some
    /// container / VM shapes); under that fallback we assume 2-way SMT
    /// and divide by 2.
    const SC_NPROCESSORS_ONLN: i32 = 84;

    /// Count distinct `(physical id, core id)` tuples in `/proc/cpuinfo`.
    /// Mirrors cpuinfo's `cores_count` semantics, which is what
    /// `c10/core/thread_pool.cpp::defaultNumThreads` consumes via the
    /// `cpuinfo_get_cores_count()` call.
    fn physical_cores() -> usize {
        let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") else {
            return logical_fallback();
        };
        let mut tuples: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();
        let mut current_phys: Option<String> = None;
        for line in content.lines() {
            let line = line.trim();
            if let Some(v) = line.strip_prefix("physical id") {
                current_phys = v
                    .split(':')
                    .nth(1)
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
            } else if let Some(v) = line.strip_prefix("core id") {
                let core = v.split(':').nth(1).map(|s| s.trim().to_string());
                if let (Some(p), Some(c)) = (current_phys.as_ref(), core)
                    && !c.is_empty()
                {
                    tuples.insert((p.clone(), c));
                }
            } else if line.is_empty() {
                // Blank line terminates a processor block in /proc/cpuinfo.
                current_phys = None;
            }
        }
        if tuples.is_empty() {
            logical_fallback()
        } else {
            tuples.len()
        }
    }

    fn logical_fallback() -> usize {
        // SAFETY: leaf FFI to libc `sysconf` with a static const arg.
        let n = unsafe { sysconf(SC_NPROCESSORS_ONLN) };
        if n > 1 {
            // Assume 2-way SMT when /proc/cpuinfo doesn't expose
            // physical topology (matches torch's behavior on shapes
            // where cpuinfo reports only logical processors).
            ((n as usize) / 2).max(1)
        } else {
            1
        }
    }

    /// SAFETY: helper that calls libc `setenv` with the given static
    /// NUL-terminated name and value byte arrays. setenv copies the
    /// value internally, so the caller-provided byte buffer need only
    /// be live for the duration of the call.
    fn set_env_if_unset(name: &'static [u8], value: &[u8]) {
        // SAFETY: leaf FFI to libc `getenv` with a static NUL-terminated
        // name. Return pointer is read-only; we only null-check.
        let already_set = unsafe { !getenv(name.as_ptr().cast()).is_null() };
        if already_set {
            return;
        }
        // SAFETY: leaf FFI to libc `setenv` with a NUL-terminated name
        // (static) and NUL-terminated value (caller invariant). setenv
        // copies into its own storage; no aliasing concerns.
        unsafe {
            setenv(name.as_ptr().cast(), value.as_ptr().cast(), 1);
        }
    }

    /// Match torch's CPU MKL initialization: set the threading layer
    /// to GNU OpenMP, disable MKL's dynamic thread adjustment, and set
    /// the thread count to the physical-core count (mirrors torch's
    /// `intraop_default_num_threads()` resolution). All three settings
    /// are skipped if the user has already set the corresponding env
    /// var.
    extern "C" fn ferrotorch_mkl_align_threads() {
        // Threading layer must be GNU OpenMP (libgomp) — torch's CPU
        // build links libgomp, and `intel_thread` vs `gnu_thread`
        // produces 2-ULP K-reduction drift at k=16384.
        set_env_if_unset(b"MKL_THREADING_LAYER\0", b"GNU\0");
        // Disable dynamic thread adjustment — torch sets this to FALSE
        // so the dispatcher uses the exact requested thread count
        // (Dyn:0 in MKL_VERBOSE output).
        set_env_if_unset(b"MKL_DYNAMIC\0", b"FALSE\0");

        let omp_name = b"OMP_NUM_THREADS\0";
        let mkl_name = b"MKL_NUM_THREADS\0";
        // SAFETY: leaf FFI to libc `getenv` with static null-terminated
        // strings. Returned pointer is read-only and we only check for
        // null — we never dereference it.
        unsafe {
            if !getenv(omp_name.as_ptr().cast()).is_null()
                || !getenv(mkl_name.as_ptr().cast()).is_null()
            {
                // User already set thread count; respect it (matches
                // torch's resolution chain at ParallelCommon.cpp:110-111).
                return;
            }
        }
        let physical = physical_cores();
        // Manual itoa for the small physical-core counts we expect
        // (1..=4096 covers any plausible host). Avoids pulling in the
        // `itoa` crate per goal.md R-FIX-4 (no new workspace deps).
        let mut digits = [0u8; 5]; // up to 4 decimal digits + NUL
        let mut n = physical.min(9999);
        let mut len = 0usize;
        if n == 0 {
            digits[0] = b'1';
            len = 1;
        } else {
            while n > 0 {
                digits[len] = b'0' + (n % 10) as u8;
                len += 1;
                n /= 10;
            }
            digits[..len].reverse();
        }
        let mut buf = [0u8; 6];
        buf[..len].copy_from_slice(&digits[..len]);
        buf[len] = 0; // NUL terminator

        // SAFETY: leaf FFI to libc `setenv` with static null-terminated
        // name + a stack-resident NUL-terminated value buffer. setenv
        // copies the value internally, so the stack lifetime is sound.
        unsafe {
            setenv(omp_name.as_ptr().cast(), buf.as_ptr().cast(), 1);
            setenv(mkl_name.as_ptr().cast(), buf.as_ptr().cast(), 1);
        }
    }

    /// `.init_array` ELF section: function pointers placed here are
    /// invoked by the dynamic loader before `main()` and before any
    /// `__attribute__((constructor))` from a downstream shared library
    /// runs. This guarantees our thread-count alignment lands before
    /// MKL's own static ctors read `OMP_NUM_THREADS` / `MKL_NUM_THREADS`.
    #[used]
    #[unsafe(link_section = ".init_array")]
    static FERROTORCH_MKL_THREAD_INIT: extern "C" fn() = ferrotorch_mkl_align_threads;
}

/// Row-major-to-col-major-dispatch helper for the `sgemm_` call. The
/// arguments encode the **column-major Fortran view** of the GEMM, with
/// operands already swapped so that the row-major problem is presented
/// to MKL as the column-major equivalent torch uses.
///
/// # Safety
///
/// The caller must guarantee `a` has at least `lda * (if transa == 'N'
/// { k } else { m })` floats, `b` has at least `ldb * (if transb ==
/// 'N' { n } else { k })` floats (in column-major terms), and `c` has
/// at least `ldc * n` floats. The caller also guarantees no aliasing
/// between `a`/`b` (immutable borrows) and `c` (unique mutable
/// borrow). Internal helper — never pub.
#[cfg(feature = "mkl")]
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors Fortran sgemm_ signature 1:1; arg count is upstream-fixed"
)]
#[allow(
    clippy::borrow_as_ptr,
    reason = "Fortran ABI passes scalars by-ref; the &val syntax is exactly that pass-by-ref convention, not an accidental implicit borrow-to-pointer"
)]
unsafe fn call_sgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    let transa_c = transa as std::ffi::c_char;
    let transb_c = transb as std::ffi::c_char;
    // SAFETY: leaf FFI shim to MKL's Fortran `sgemm_` (permitted under
    // goal.md R-CODE-1). All scalar arguments are passed by-ref as the
    // Fortran ABI requires; the temporaries live on the stack for the
    // duration of the call. Pointer / lda / ldb invariants are the
    // caller's responsibility per the SAFETY doc above.
    unsafe {
        sgemm_(
            &transa_c, &transb_c, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
        );
    }
}

/// f64 mirror of `call_sgemm`.
///
/// # Safety
///
/// Identical invariants to `call_sgemm` but for `dgemm_`.
#[cfg(feature = "mkl")]
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors Fortran dgemm_ signature 1:1; arg count is upstream-fixed"
)]
#[allow(
    clippy::borrow_as_ptr,
    reason = "Fortran ABI passes scalars by-ref; the &val syntax is exactly that pass-by-ref convention, not an accidental implicit borrow-to-pointer"
)]
unsafe fn call_dgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    let transa_c = transa as std::ffi::c_char;
    let transb_c = transb as std::ffi::c_char;
    // SAFETY: leaf FFI shim to MKL's Fortran `dgemm_`; see `call_sgemm`.
    unsafe {
        dgemm_(
            &transa_c, &transb_c, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
        );
    }
}

/// Matrix multiplication: C = A @ B.
///
/// Follows PyTorch `torch.matmul` semantics exactly:
///
/// - 1D x 1D: dot product -> scalar
/// - 2D x 1D: matrix-vector multiply (M,K) @ (K,) -> (M,)
/// - 1D x 2D: vector-matrix multiply (K,) @ (K,N) -> (N,)
/// - 2D x 2D: standard matrix multiply (M,K) @ (K,N) -> (M,N)
/// - ≥3D: batched matmul with NumPy-style broadcasting over leading dims.
///   If one input is 1D, it is promoted (prepend dim for LHS, append dim for RHS)
///   and the added dimension is squeezed from the output.
pub fn matmul<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::matmul(a, b)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("matmul", "linalg", &[a.shape(), b.shape()], || {
        match (a.ndim(), b.ndim()) {
            (0, _) | (_, 0) => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "matmul: scalar operands not supported, got shapes {:?} and {:?}",
                    a.shape(),
                    b.shape()
                ),
            }),
            (1, 1) => dot(a, b),
            (2, 1) => mv(a, b),
            (1, 2) => vm(a, b),
            (2, 2) => mm(a, b),
            _ => broadcast_matmul(a, b),
        }
    })
}

/// Broadcast leading dimensions of two shapes according to NumPy rules.
/// Returns the broadcasted batch shape.
fn broadcast_batch_shapes(a: &[usize], b: &[usize]) -> FerrotorchResult<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let da = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let db = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };
        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!("matmul: batch dimensions cannot be broadcast: {a:?} vs {b:?}"),
            });
        }
    }
    Ok(result)
}

/// Batched matrix multiply with NumPy-style broadcast over leading dimensions.
///
/// Handles all cases where at least one operand has ndim ≥ 3 (and the other
/// is at least 1D). 1D operands are promoted before dispatch and the added
/// dimension is squeezed from the output, matching `torch.matmul`.
fn broadcast_matmul<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let device = a.device();
    // --- 1D promotion ------------------------------------------------
    // If a is 1D (K,) → treat as (1, K) and squeeze row from output.
    // If b is 1D (K,) → treat as (K, 1) and squeeze col from output.
    let squeeze_row = a.ndim() == 1;
    let squeeze_col = b.ndim() == 1;

    let a_shape: Vec<usize> = if squeeze_row {
        let mut s = vec![1];
        s.extend_from_slice(a.shape());
        s
    } else {
        a.shape().to_vec()
    };
    let b_shape: Vec<usize> = if squeeze_col {
        let mut s = b.shape().to_vec();
        s.push(1);
        s
    } else {
        b.shape().to_vec()
    };

    let a_nd = a_shape.len();
    let b_nd = b_shape.len();

    // Matrix dims (last two of each).
    let m = a_shape[a_nd - 2];
    let k_a = a_shape[a_nd - 1];
    let k_b = b_shape[b_nd - 2];
    let n = b_shape[b_nd - 1];

    if k_a != k_b {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "matmul: inner dimensions mismatch: {:?} @ {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }
    let k = k_a;

    // Batch dimensions.
    let a_batch = &a_shape[..a_nd - 2];
    let b_batch = &b_shape[..b_nd - 2];
    let batch_shape = broadcast_batch_shapes(a_batch, b_batch)?;
    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);

    // Compute strides for broadcasting iteration.
    let a_batch_strides = broadcast_strides(a_batch, &batch_shape);
    let b_batch_strides = broadcast_strides(b_batch, &batch_shape);

    let a_mat_size = m * k;
    let b_mat_size = k * n;
    let c_mat_size = m * n;

    let a_data = a.data_vec()?;
    let b_data = b.data_vec()?;
    let mut result = vec![<T as num_traits::Zero>::zero(); batch_size * c_mat_size];

    for bi in 0..batch_size {
        // Map flat batch index to a/b offsets using broadcast strides.
        let a_off = batch_linear_index(bi, &a_batch_strides, &batch_shape) * a_mat_size;
        let b_off = batch_linear_index(bi, &b_batch_strides, &batch_shape) * b_mat_size;
        let c_off = bi * c_mat_size;

        // Route each per-batch (M,K)@(K,N) slab through the faer-backed
        // `mm_raw` workhorse. This consolidates the cross-BLAS-implementation
        // accumulation behavior — the naive (i,j,p) triple-loop diverged from
        // PyTorch's MKL block-summation by ~1.5e-5 on f32 with k>=10 (verified
        // 2026-05-26 on op_db sample matmul seed=7 i=6); routing through
        // faer at least matches the same well-known cross-BLAS f32 ULP
        // variance envelope. Byte-for-byte parity vs MKL requires the
        // future-epic MKL/OpenBLAS FFI path; this commit acknowledges that
        // reality by widening the matmul-family runner tolerance to
        // rtol=1e-4 (see `tools/parity-sweep/runner/src/main.rs`
        // `tolerance_for`) rather than masking it.
        let a_slice = &a_data[a_off..a_off + a_mat_size];
        let b_slice = &b_data[b_off..b_off + b_mat_size];
        let c_slab = mm_raw(a_slice, b_slice, m, k, n);
        result[c_off..c_off + c_mat_size].copy_from_slice(&c_slab);
    }

    // Output shape = batch_shape + [m, n], then squeeze promoted dims.
    let mut out_shape = batch_shape;
    out_shape.push(m);
    out_shape.push(n);

    if squeeze_row {
        // Remove the m=1 dimension (second-to-last).
        let pos = out_shape.len() - 2;
        out_shape.remove(pos);
    }
    if squeeze_col {
        // Remove the n=1 dimension (last).
        out_shape.pop();
    }

    let t = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
    Ok(if device.is_cuda() { t.to(device)? } else { t })
}

/// Compute the strides needed to map a flat index in the broadcast shape
/// back to a flat index in the (possibly smaller) source batch shape.
fn broadcast_strides(src: &[usize], broadcast: &[usize]) -> Vec<usize> {
    let offset = broadcast.len() - src.len();
    let mut strides = vec![0usize; broadcast.len()];

    // Compute row-major strides for the source shape.
    if !src.is_empty() {
        let mut src_strides = vec![1usize; src.len()];
        for i in (0..src.len() - 1).rev() {
            src_strides[i] = src_strides[i + 1] * src[i + 1];
        }

        for (i, stride) in strides.iter_mut().enumerate() {
            if i < offset {
                // Dimension doesn't exist in source — broadcast (stride 0).
                *stride = 0;
            } else {
                let si = i - offset;
                if src[si] == 1 {
                    // Size-1 dimension — broadcast (stride 0).
                    *stride = 0;
                } else {
                    *stride = src_strides[si];
                }
            }
        }
    }

    strides
}

/// Convert a flat batch index into a flat source index using broadcast strides.
fn batch_linear_index(flat: usize, strides: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0;
    let mut remaining = flat;
    // Decompose flat index into multi-index, then dot with strides.
    for i in (0..shape.len()).rev() {
        let coord = remaining % shape[i];
        remaining /= shape[i];
        idx += coord * strides[i];
    }
    idx
}

/// Dot product of two 1-D tensors -> scalar.
pub fn dot<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "dot requires 1-D tensors, got {:?} and {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }
    if a.shape()[0] != b.shape()[0] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "dot product dimension mismatch: {} vs {}",
                a.shape()[0],
                b.shape()[0]
            ),
        });
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let result = a_data
        .iter()
        .zip(b_data.iter())
        .fold(<T as num_traits::Zero>::zero(), |acc, (&x, &y)| acc + x * y);

    Tensor::from_storage(TensorStorage::cpu(vec![result]), vec![], false)
}

/// Threshold for switching from direct ikj loop to faer.
/// For matrices at or below this size, the naive loop avoids faer overhead.
const DIRECT_MM_THRESHOLD: usize = 128;

/// Whether `T` is `half::bf16`. Used to route bf16 kernels through an
/// f32 accumulator to avoid the catastrophic precision loss of summing
/// hundreds of 7-bit-mantissa values in bf16.
#[allow(clippy::inline_always)] // reason: trivial TypeId compare; must inline to constant-fold the bf16 dispatch in hot matmul
#[inline(always)]
fn is_bf16<T: 'static>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>()
}

/// Reinterpret a bf16 slice as `&[half::bf16]`. Only call when
/// `is_bf16::<T>()` is true.
#[allow(clippy::inline_always)] // reason: zero-cost reinterpret cast; must inline so caller sees through to the bf16 slice
#[inline(always)]
unsafe fn as_bf16_slice<T>(data: &[T]) -> &[half::bf16] {
    // SAFETY: caller guarantees T is half::bf16 (same size, same repr).
    unsafe { &*(std::ptr::from_ref::<[T]>(data) as *const [half::bf16]) }
}

/// Write f32 results into a freshly-zeroed T slice (only valid when T=bf16).
#[allow(clippy::inline_always)] // reason: tight zip-loop fused into matmul epilogue; must inline to vectorize the bf16 store
#[inline(always)]
unsafe fn write_f32_as_bf16<T>(dst: &mut [T], src: &[f32]) {
    // SAFETY: caller guarantees T is half::bf16.
    let dst_bf16 = unsafe { &mut *(std::ptr::from_mut::<[T]>(dst) as *mut [half::bf16]) };
    for (d, &s) in dst_bf16.iter_mut().zip(src.iter()) {
        *d = half::bf16::from_f32(s);
    }
}

/// Choose parallelism for faer matmul.  For medium matrices use sequential
/// to avoid thread-pool overhead; for large matrices use rayon.
#[inline]
fn faer_par(m: usize, k: usize, n: usize) -> faer::Par {
    // Rough heuristic: only pay rayon spawn cost when there's enough work.
    if m * k * n >= 512 * 512 * 512 {
        faer::Par::rayon(0)
    } else {
        faer::Par::Seq
    }
}

// MKL Fortran-symbol helpers — used by `mm_raw`/`mm_raw_bt`/`mm_raw_at`
// under `--features mkl` to route the entire f32/f64 path (any matrix
// size) through MKL's `sgemm_`/`dgemm_` Fortran symbols for byte-for-
// byte parity with PyTorch's MKL CPU build (when torch links MKL).
// The cross-implementation f32 ULP drift originates from k>=10 dot
// products, which means it hits even the small-matrix sizes that
// op_db samples — we therefore unconditionally dispatch every f32/f64
// GEMM through MKL when the feature is on. Each helper takes the
// caller's T-typed slices and reinterprets them at the FFI boundary;
// the TypeId guard at the dispatch site upstream proves T == f32 or
// T == f64.

/// Row-major mm dispatch: `C = A @ B` where A is (M,K) row-major and B
/// is (K,N) row-major. Mirrors torch's CPUBlas dispatch at
/// `aten/src/ATen/native/CPUBlas.cpp:215-247` exactly.
///
/// Derivation. Row-major (M,K) is bit-identical to col-major (K,M);
/// row-major (K,N) is bit-identical to col-major (N,K); the desired
/// row-major (M,N) is bit-identical to col-major (N,M). So in
/// col-major terms we want
/// `C_col[n,m] = sum_k A_col[k,m] * B_col[n,k] = (B_col @ A_col)[n,m]`,
/// i.e. an unswapped no-trans GEMM with the operand order **swapped**:
///
/// ```text
///   sgemm_('N', 'N', N, M, K, 1.0, B, N, A, K, 0.0, C, N)
/// ```
///
/// This is the exact pattern torch uses when `at::mm` lowers a
/// row-major-contiguous result/op pair to `cpublas::gemm` (the
/// `transpose_c=true; std::swap(m1, m2)` branch in
/// `aten/src/ATen/native/LinearAlgebra.cpp:1454-1465` followed by the
/// per-operand `transpose_a=false` / `transpose_b=false` arms at
/// `:1475-1499`).
#[cfg(feature = "mkl")]
fn mm_raw_mkl_f32<T: 'static + num_traits::Zero + Clone>(
    a_data: &[T],
    b_data: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<T> {
    debug_assert_eq!(std::any::TypeId::of::<T>(), std::any::TypeId::of::<f32>());
    let zero_t = <T as num_traits::Zero>::zero();
    let mut result: Vec<T> = vec![zero_t; m * n];
    // MKL rejects degenerate shapes; for an empty contraction (k=0)
    // the result is all-zero (matches torch's empty-reduction
    // semantics); for m=0/n=0 the result is empty.
    if m == 0 || n == 0 || k == 0 {
        return result;
    }
    // SAFETY: TypeId guard at caller proves T == f32; &[T] reinterprets
    // as &[f32] with identical layout.
    let a_f32 = unsafe { &*(std::ptr::from_ref::<[T]>(a_data) as *const [f32]) };
    // SAFETY: same as a_f32.
    let b_f32 = unsafe { &*(std::ptr::from_ref::<[T]>(b_data) as *const [f32]) };
    // SAFETY: T == f32 by TypeId guard; `result` is fresh and unaliased.
    let c_f32 = unsafe { &mut *(std::ptr::from_mut::<[T]>(result.as_mut_slice()) as *mut [f32]) };
    // Thin-matmul dispatch-shape conditional (#1541, #1538).
    //
    // For `m == 1 || n == 1` the row-major result has strides that make
    // torch's `addmm_impl_cpu_` (aten/src/ATen/native/LinearAlgebra.cpp:1450-1557)
    // take the `transpose_c=false` first arm and emit
    // `sgemm_('T','T', m, n, k, alpha, A, lda=k, B, ldb=n, beta, C, ldc=m)`
    // (no operand swap). MKL's kernel dispatcher keys on (transa, transb,
    // m, n, k, lda, ldb, ldc); the (T,T)+no-swap form lands on the
    // K-parallel threaded dot kernel, while the (N,N)+operand-swap form
    // used for dense GEMM lands on a SERIAL small-matrix kernel that
    // ignores the thread pool. The two compute the same dot product in
    // different summation orders, producing 5-11 ULP drift at k=O(10^4).
    // For thin shapes we mirror torch's dispatch exactly; for dense
    // shapes the (N,N)+swap form is byte-exact vs torch (#1538 audit
    // confirmed on 64x64, 127x127, k=257 probes).
    //
    // For `m == 1 || n == 1` the memory layout of the result coincides
    // between row-major (M,N) and col-major (M,N): both store M*N
    // contiguous floats with no leading-dim stride dependency, so
    // writing into `c_f32` with ldc=m is bit-identical to row-major.
    if m == 1 || n == 1 {
        // SAFETY: leaf FFI to MKL's `sgemm_`. (T,T)+no-swap mirrors
        // torch's `addmm_impl_cpu_` derivation at LinearAlgebra.cpp:1450-1557
        // → CPUBlas.cpp:238 for thin matmul. m/n/k cast to i32 is sound
        // for any host-memory-bounded tensor shape.
        unsafe {
            call_sgemm(
                b'T',
                b'T',
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a_f32.as_ptr(),
                k as i32,
                b_f32.as_ptr(),
                n as i32,
                0.0,
                c_f32.as_mut_ptr(),
                m as i32,
            );
        }
        return result;
    }
    // SAFETY: leaf FFI shim to MKL's Fortran `sgemm_`. Dispatch is the
    // operand-swap + dim-swap pattern documented in this function's
    // doc-comment. m/n/k cast to i32 is sound for any host-memory-
    // bounded tensor shape. beta=0.0 writes (not accumulates).
    unsafe {
        call_sgemm(
            b'N',
            b'N',
            n as i32,
            m as i32,
            k as i32,
            1.0,
            b_f32.as_ptr(),
            n as i32,
            a_f32.as_ptr(),
            k as i32,
            0.0,
            c_f32.as_mut_ptr(),
            n as i32,
        );
    }
    result
}

/// f64 mirror of `mm_raw_mkl_f32`.
#[cfg(feature = "mkl")]
fn mm_raw_mkl_f64<T: 'static + num_traits::Zero + Clone>(
    a_data: &[T],
    b_data: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<T> {
    debug_assert_eq!(std::any::TypeId::of::<T>(), std::any::TypeId::of::<f64>());
    let zero_t = <T as num_traits::Zero>::zero();
    let mut result: Vec<T> = vec![zero_t; m * n];
    if m == 0 || n == 0 || k == 0 {
        return result;
    }
    // SAFETY: T == f64 by caller's TypeId guard.
    let a_f64 = unsafe { &*(std::ptr::from_ref::<[T]>(a_data) as *const [f64]) };
    // SAFETY: same as a_f64.
    let b_f64 = unsafe { &*(std::ptr::from_ref::<[T]>(b_data) as *const [f64]) };
    // SAFETY: T == f64; result is fresh and unaliased.
    let c_f64 = unsafe { &mut *(std::ptr::from_mut::<[T]>(result.as_mut_slice()) as *mut [f64]) };
    // Thin-matmul dispatch-shape conditional — see `mm_raw_mkl_f32`'s
    // body for the full derivation (#1541, #1538). `dgemm_` has the
    // same kernel-dispatch structure as `sgemm_`, so the same
    // (T,T)+no-swap mirror of torch's dispatch applies.
    if m == 1 || n == 1 {
        // SAFETY: leaf FFI to MKL's `dgemm_`; mirror of the f32 path.
        unsafe {
            call_dgemm(
                b'T',
                b'T',
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a_f64.as_ptr(),
                k as i32,
                b_f64.as_ptr(),
                n as i32,
                0.0,
                c_f64.as_mut_ptr(),
                m as i32,
            );
        }
        return result;
    }
    // SAFETY: leaf FFI shim to MKL's `dgemm_`; same invariants as
    // `mm_raw_mkl_f32` but f64.
    unsafe {
        call_dgemm(
            b'N',
            b'N',
            n as i32,
            m as i32,
            k as i32,
            1.0,
            b_f64.as_ptr(),
            n as i32,
            a_f64.as_ptr(),
            k as i32,
            0.0,
            c_f64.as_mut_ptr(),
            n as i32,
        );
    }
    result
}

/// `mm_raw_bt` MKL path for f32: A is (M,K) row-major, B is (N,K)
/// row-major, computes `C = A @ B^T` of shape (M,N).
///
/// Derivation. Row-major B (N,K) is bit-identical to col-major (K,N).
/// `C_col[n,m] = sum_k A_col[k,m] * B_col[k,n] = (B_col^T @ A_col)[n,m]`.
/// Fortran call:
///
/// ```text
///   sgemm_('T', 'N', N, M, K, 1.0, B, K, A, K, 0.0, C, N)
/// ```
#[cfg(feature = "mkl")]
fn mm_raw_bt_mkl_f32<T: 'static + num_traits::Zero + Clone>(
    a_data: &[T],
    b_data: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<T> {
    debug_assert_eq!(std::any::TypeId::of::<T>(), std::any::TypeId::of::<f32>());
    let zero_t = <T as num_traits::Zero>::zero();
    let mut result: Vec<T> = vec![zero_t; m * n];
    if m == 0 || n == 0 || k == 0 {
        return result;
    }
    // SAFETY: T == f32 by caller's TypeId guard.
    let a_f32 = unsafe { &*(std::ptr::from_ref::<[T]>(a_data) as *const [f32]) };
    // SAFETY: same as a_f32.
    let b_f32 = unsafe { &*(std::ptr::from_ref::<[T]>(b_data) as *const [f32]) };
    // SAFETY: T == f32; result is fresh and unaliased.
    let c_f32 = unsafe { &mut *(std::ptr::from_mut::<[T]>(result.as_mut_slice()) as *mut [f32]) };
    // SAFETY: leaf FFI to MKL's `sgemm_`. transa='T' applied to the
    // col-major B view (of row-major (N,K), which is col-major (K,N)).
    unsafe {
        call_sgemm(
            b'T',
            b'N',
            n as i32,
            m as i32,
            k as i32,
            1.0,
            b_f32.as_ptr(),
            k as i32,
            a_f32.as_ptr(),
            k as i32,
            0.0,
            c_f32.as_mut_ptr(),
            n as i32,
        );
    }
    result
}

/// f64 mirror of `mm_raw_bt_mkl_f32`.
#[cfg(feature = "mkl")]
fn mm_raw_bt_mkl_f64<T: 'static + num_traits::Zero + Clone>(
    a_data: &[T],
    b_data: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<T> {
    debug_assert_eq!(std::any::TypeId::of::<T>(), std::any::TypeId::of::<f64>());
    let zero_t = <T as num_traits::Zero>::zero();
    let mut result: Vec<T> = vec![zero_t; m * n];
    if m == 0 || n == 0 || k == 0 {
        return result;
    }
    // SAFETY: T == f64 by caller's TypeId guard.
    let a_f64 = unsafe { &*(std::ptr::from_ref::<[T]>(a_data) as *const [f64]) };
    // SAFETY: same as a_f64.
    let b_f64 = unsafe { &*(std::ptr::from_ref::<[T]>(b_data) as *const [f64]) };
    // SAFETY: T == f64; result is fresh and unaliased.
    let c_f64 = unsafe { &mut *(std::ptr::from_mut::<[T]>(result.as_mut_slice()) as *mut [f64]) };
    // SAFETY: leaf FFI to MKL's `dgemm_`; mirror of the f32 path.
    unsafe {
        call_dgemm(
            b'T',
            b'N',
            n as i32,
            m as i32,
            k as i32,
            1.0,
            b_f64.as_ptr(),
            k as i32,
            a_f64.as_ptr(),
            k as i32,
            0.0,
            c_f64.as_mut_ptr(),
            n as i32,
        );
    }
    result
}

/// `mm_raw_at` MKL path for f32: A is (K,M) row-major, B is (K,N)
/// row-major, computes `C = A^T @ B` of shape (M,N).
///
/// Derivation. Row-major A (K,M) is bit-identical to col-major (M,K);
/// row-major B (K,N) is bit-identical to col-major (N,K).
/// `C_col[n,m] = sum_k A_col[m,k] * B_col[n,k] = (B_col @ A_col^T)[n,m]`:
///
/// ```text
///   sgemm_('N', 'T', N, M, K, 1.0, B, N, A, M, 0.0, C, N)
/// ```
///
/// **Why this matters (#1538 root cause)**. The prior dispatch called
/// `cblas_sgemm(RowMajor, transa=Trans, transb=NoTrans, lda=m, ldb=n)`.
/// MKL's cblas row-major wrapper picks a different micro-kernel than
/// raw Fortran `sgemm_('N', 'T', ...)` for the same math, producing
/// 1-ULP-different block-summation rounds vs torch on `mm_raw_at`'s
/// inputs. Calling the raw Fortran symbol with the swap pattern torch
/// itself uses fixes the drift.
#[cfg(feature = "mkl")]
fn mm_raw_at_mkl_f32<T: 'static + num_traits::Zero + Clone>(
    a_data: &[T],
    b_data: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<T> {
    debug_assert_eq!(std::any::TypeId::of::<T>(), std::any::TypeId::of::<f32>());
    let zero_t = <T as num_traits::Zero>::zero();
    let mut result: Vec<T> = vec![zero_t; m * n];
    if m == 0 || n == 0 || k == 0 {
        return result;
    }
    // SAFETY: T == f32 by caller's TypeId guard.
    let a_f32 = unsafe { &*(std::ptr::from_ref::<[T]>(a_data) as *const [f32]) };
    // SAFETY: same as a_f32.
    let b_f32 = unsafe { &*(std::ptr::from_ref::<[T]>(b_data) as *const [f32]) };
    // SAFETY: T == f32; result is fresh and unaliased.
    let c_f32 = unsafe { &mut *(std::ptr::from_mut::<[T]>(result.as_mut_slice()) as *mut [f32]) };
    // SAFETY: leaf FFI to MKL's `sgemm_`. transb='T' applied to the
    // col-major A view (of row-major (K,M), which is col-major (M,K)).
    unsafe {
        call_sgemm(
            b'N',
            b'T',
            n as i32,
            m as i32,
            k as i32,
            1.0,
            b_f32.as_ptr(),
            n as i32,
            a_f32.as_ptr(),
            m as i32,
            0.0,
            c_f32.as_mut_ptr(),
            n as i32,
        );
    }
    result
}

/// f64 mirror of `mm_raw_at_mkl_f32`.
#[cfg(feature = "mkl")]
fn mm_raw_at_mkl_f64<T: 'static + num_traits::Zero + Clone>(
    a_data: &[T],
    b_data: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<T> {
    debug_assert_eq!(std::any::TypeId::of::<T>(), std::any::TypeId::of::<f64>());
    let zero_t = <T as num_traits::Zero>::zero();
    let mut result: Vec<T> = vec![zero_t; m * n];
    if m == 0 || n == 0 || k == 0 {
        return result;
    }
    // SAFETY: T == f64 by caller's TypeId guard.
    let a_f64 = unsafe { &*(std::ptr::from_ref::<[T]>(a_data) as *const [f64]) };
    // SAFETY: same as a_f64.
    let b_f64 = unsafe { &*(std::ptr::from_ref::<[T]>(b_data) as *const [f64]) };
    // SAFETY: T == f64; result is fresh and unaliased.
    let c_f64 = unsafe { &mut *(std::ptr::from_mut::<[T]>(result.as_mut_slice()) as *mut [f64]) };
    // SAFETY: leaf FFI to MKL's `dgemm_`; mirror of the f32 path.
    unsafe {
        call_dgemm(
            b'N',
            b'T',
            n as i32,
            m as i32,
            k as i32,
            1.0,
            b_f64.as_ptr(),
            n as i32,
            a_f64.as_ptr(),
            m as i32,
            0.0,
            c_f64.as_mut_ptr(),
            n as i32,
        );
    }
    result
}

/// Raw matrix multiply on borrowed slices: (M,K) @ (K,N) -> Vec<T>.
/// Zero input allocations — operates directly on the borrowed data.
/// This is the hot-path workhorse used by both `mm` and `mm_differentiable`.
pub fn mm_raw<T: Float>(a_data: &[T], b_data: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    // Under `--features mkl`, every f32 and f64 multiply (regardless of
    // matrix size) routes through `cblas_sgemm`/`cblas_dgemm` for
    // byte-for-byte parity with PyTorch's MKL CPU build (closes #1348).
    // We deliberately bypass the DIRECT_MM_THRESHOLD small-matrix loop
    // here because the small ikj loop is precisely where the cross-
    // implementation drift originates (a k=10 dot in ferrotorch's loop
    // vs MKL's k=10 dot diverge by ~3e-6 at f32 due to different
    // accumulation orders and FMA fusion). The bf16 small-matrix path
    // keeps its f32-accumulator route (MKL has no bf16 sgemm via cblas
    // anyway), and the f16 large-matrix upcast fallback stays.
    #[cfg(feature = "mkl")]
    {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            return mm_raw_mkl_f32::<T>(a_data, b_data, m, k, n);
        }
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            return mm_raw_mkl_f64::<T>(a_data, b_data, m, k, n);
        }
    }
    let max_dim = m.max(n).max(k);
    let zero = <T as num_traits::Zero>::zero();

    if max_dim <= DIRECT_MM_THRESHOLD {
        // Direct ikj loop — cache-friendly, zero intermediate allocations.
        // Uses unsafe get_unchecked to eliminate bounds checks in the hot loop.
        let mut result = vec![zero; m * n];
        if is_bf16::<T>() {
            // bf16 fast path with f32 accumulator. Summing up to
            // DIRECT_MM_THRESHOLD bf16 values in bf16 loses ~7 bits of
            // precision per dot; accumulating in f32 preserves them.
            // SAFETY: is_bf16::<T>() returned true (TypeId::of::<T>() ==
            // TypeId::of::<half::bf16>()), so T == half::bf16. This satisfies
            // the contracts of as_bf16_slice (T == bf16) and write_f32_as_bf16
            // (dst is bf16-layout). All get_unchecked calls index in [0, m*k),
            // [0, k*n), or [0, m*n), which are within the allocated buffers
            // of length m*k, k*n, m*n respectively (acc and result are
            // explicitly sized m*n; a_data/b_data are guaranteed by caller).
            unsafe {
                let a_bf16 = as_bf16_slice(a_data);
                let b_bf16 = as_bf16_slice(b_data);
                let mut acc = vec![0.0f32; m * n];
                for i in 0..m {
                    let a_row = i * k;
                    let r_row = i * n;
                    for p in 0..k {
                        let a_ip = a_bf16.get_unchecked(a_row + p).to_f32();
                        let b_row = p * n;
                        for j in 0..n {
                            *acc.get_unchecked_mut(r_row + j) +=
                                a_ip * b_bf16.get_unchecked(b_row + j).to_f32();
                        }
                    }
                }
                write_f32_as_bf16(&mut result, &acc);
            }
        } else {
            // SAFETY: a_data has length m*k and b_data has length k*n by
            // function contract; result was allocated with `vec![zero; m*n]`.
            // Index arithmetic `i*k + p` with i<m, p<k stays in [0, m*k);
            // `p*n + j` with p<k, j<n stays in [0, k*n); `i*n + j` stays in
            // [0, m*n). All get_unchecked accesses are in bounds.
            unsafe {
                for i in 0..m {
                    let a_row = i * k;
                    let r_row = i * n;
                    for p in 0..k {
                        let a_ip = *a_data.get_unchecked(a_row + p);
                        let b_row = p * n;
                        for j in 0..n {
                            let r = result.get_unchecked_mut(r_row + j);
                            *r += a_ip * *b_data.get_unchecked(b_row + j);
                        }
                    }
                }
            }
        }
        result
    } else {
        // Large matrices — use faer GEMM for high-performance BLAS.
        // faer supports arbitrary strides natively and auto-vectorises with AVX/SSE.
        let mut result = vec![zero; m * n];
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // SAFETY: TypeId guard above proves T == f32, so &[T] and &[f32]
            // have identical layout (size, alignment, niche). The cast is a
            // no-op reinterpretation; lifetimes are tied to a_data/b_data/result.
            let a_f32 = unsafe { &*(std::ptr::from_ref::<[T]>(a_data) as *const [f32]) };
            // SAFETY: same as a_f32 above (T == f32 by the TypeId guard).
            let b_f32 = unsafe { &*(std::ptr::from_ref::<[T]>(b_data) as *const [f32]) };
            // SAFETY: same as a_f32 above (T == f32 by the TypeId guard).
            // `result` was just allocated and is not aliased by a_data/b_data
            // (which are immutable borrows from the caller), so producing a
            // unique &mut [f32] view is sound.
            let c_f32 =
                unsafe { &mut *(std::ptr::from_mut::<[T]>(result.as_mut_slice()) as *mut [f32]) };
            // Under `--features mkl` the function head short-circuited
            // f32 through `mm_raw_mkl_f32`, so this large-matrix faer
            // branch only runs in the no-mkl build.
            let a_mat = faer::mat::MatRef::from_row_major_slice(a_f32, m, k);
            let b_mat = faer::mat::MatRef::from_row_major_slice(b_f32, k, n);
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(c_f32, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f32,
                par,
            );
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            // SAFETY: TypeId guard above proves T == f64, so &[T] and &[f64]
            // have identical layout; the cast is a layout-preserving reinterpret.
            let a_f64 = unsafe { &*(std::ptr::from_ref::<[T]>(a_data) as *const [f64]) };
            // SAFETY: same as a_f64 above (T == f64 by the TypeId guard).
            let b_f64 = unsafe { &*(std::ptr::from_ref::<[T]>(b_data) as *const [f64]) };
            // SAFETY: T == f64 by the TypeId guard. `result` was just allocated
            // and is not aliased by the immutable a_data/b_data borrows, so the
            // produced &mut [f64] is unique for the duration of this block.
            let c_f64 =
                unsafe { &mut *(std::ptr::from_mut::<[T]>(result.as_mut_slice()) as *mut [f64]) };
            // Under `--features mkl` the function head short-circuited
            // f64 through `mm_raw_mkl_f64`, so this faer branch only
            // runs in the no-mkl build.
            let a_mat = faer::mat::MatRef::from_row_major_slice(a_f64, m, k);
            let b_mat = faer::mat::MatRef::from_row_major_slice(b_f64, k, n);
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(c_f64, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f64,
                par,
            );
        } else {
            // Fallback for f16/bf16: upcast to f64, run faer, downcast.
            let a_f64: Vec<f64> = a_data.iter().map(|&v| v.to_f64().unwrap()).collect();
            let b_f64: Vec<f64> = b_data.iter().map(|&v| v.to_f64().unwrap()).collect();
            let mut r_f64 = vec![0.0f64; m * n];
            let a_mat = faer::mat::MatRef::from_row_major_slice(&a_f64, m, k);
            let b_mat = faer::mat::MatRef::from_row_major_slice(&b_f64, k, n);
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(&mut r_f64, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f64,
                par,
            );
            for (r, &v) in result.iter_mut().zip(r_f64.iter()) {
                *r = T::from(v).unwrap();
            }
        }
        result
    }
}

/// Matrix multiply with B transposed: A @ B^T.
/// A is (M,K), B is (N,K) stored row-major, result is (M,N).
/// For small matrices, uses a direct loop. For large matrices, uses faer GEMM
/// with a zero-copy transposed view of B.
pub fn mm_raw_bt<T: Float>(a_data: &[T], b_data: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    // Under `--features mkl`, route every f32/f64 multiply through MKL
    // for byte-for-byte parity vs torch. See the equivalent guard in
    // `mm_raw` for the full rationale (closes #1348).
    #[cfg(feature = "mkl")]
    {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            return mm_raw_bt_mkl_f32::<T>(a_data, b_data, m, k, n);
        }
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            return mm_raw_bt_mkl_f64::<T>(a_data, b_data, m, k, n);
        }
    }
    let max_dim = m.max(n).max(k);
    let zero = <T as num_traits::Zero>::zero();

    if max_dim <= DIRECT_MM_THRESHOLD {
        // Direct loop — both A row and B row are accessed sequentially.
        // B is (N,K) row-major, so B[j][p] = b_data[j*k + p].
        // C[i][j] = sum_p A[i][p] * B[j][p]
        let mut result = vec![zero; m * n];
        if is_bf16::<T>() {
            // bf16 fast path with f32 accumulator (see note in `mm_raw`).
            // SAFETY: is_bf16::<T>() returned true so T == half::bf16, which
            // satisfies as_bf16_slice and write_f32_as_bf16 contracts. A is
            // (M,K) and B is (N,K), so a_row+p < m*k and b_row+p < n*k for
            // i<m, j<n, p<k. acc_buf and result both have length m*n, so
            // r_row+j = i*n+j < m*n. All get_unchecked accesses are in bounds.
            unsafe {
                let a_bf16 = as_bf16_slice(a_data);
                let b_bf16 = as_bf16_slice(b_data);
                let mut acc_buf = vec![0.0f32; m * n];
                for i in 0..m {
                    let a_row = i * k;
                    let r_row = i * n;
                    for j in 0..n {
                        let b_row = j * k;
                        let mut acc = 0.0f32;
                        for p in 0..k {
                            acc += a_bf16.get_unchecked(a_row + p).to_f32()
                                * b_bf16.get_unchecked(b_row + p).to_f32();
                        }
                        *acc_buf.get_unchecked_mut(r_row + j) = acc;
                    }
                }
                write_f32_as_bf16(&mut result, &acc_buf);
            }
        } else {
            // SAFETY: A is (M,K) and B is (N,K) row-major (function contract);
            // result was allocated with `vec![zero; m*n]`. Index arithmetic
            // i*k+p < m*k for i<m,p<k; j*k+p < n*k for j<n,p<k; i*n+j < m*n.
            // All get_unchecked accesses are in bounds for the respective slices.
            unsafe {
                for i in 0..m {
                    let a_row = i * k;
                    let r_row = i * n;
                    for j in 0..n {
                        let b_row = j * k;
                        let mut acc = zero;
                        for p in 0..k {
                            acc +=
                                *a_data.get_unchecked(a_row + p) * *b_data.get_unchecked(b_row + p);
                        }
                        *result.get_unchecked_mut(r_row + j) = acc;
                    }
                }
            }
        }
        result
    } else {
        // Large matrices — use faer GEMM with zero-copy transposed B view.
        // B is (N,K) row-major. Wrap as (N,K) MatRef then .transpose() to get (K,N).
        let mut result = vec![zero; m * n];
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // SAFETY: TypeId guard above proves T == f32, so &[T] and &[f32]
            // have identical layout; the cast is a layout-preserving reinterpret.
            let a_f32 = unsafe { &*(std::ptr::from_ref::<[T]>(a_data) as *const [f32]) };
            // SAFETY: same as a_f32 above (T == f32 by the TypeId guard).
            let b_f32 = unsafe { &*(std::ptr::from_ref::<[T]>(b_data) as *const [f32]) };
            // SAFETY: T == f32 by the TypeId guard. `result` was just allocated
            // and is not aliased by the immutable a_data/b_data borrows, so the
            // produced &mut [f32] is unique for the duration of this block.
            let c_f32 =
                unsafe { &mut *(std::ptr::from_mut::<[T]>(result.as_mut_slice()) as *mut [f32]) };
            // Under `--features mkl` the function head short-circuited
            // f32 through `mm_raw_bt_mkl_f32`, so this faer branch only
            // runs in the no-mkl build.
            let a_mat = faer::mat::MatRef::from_row_major_slice(a_f32, m, k);
            // B is (N,K) row-major; transpose gives (K,N) view — zero copy.
            let b_mat = faer::mat::MatRef::from_row_major_slice(b_f32, n, k).transpose();
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(c_f32, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f32,
                par,
            );
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            // SAFETY: TypeId guard above proves T == f64, so &[T] and &[f64]
            // have identical layout; the cast is a layout-preserving reinterpret.
            let a_f64 = unsafe { &*(std::ptr::from_ref::<[T]>(a_data) as *const [f64]) };
            // SAFETY: same as a_f64 above (T == f64 by the TypeId guard).
            let b_f64 = unsafe { &*(std::ptr::from_ref::<[T]>(b_data) as *const [f64]) };
            // SAFETY: T == f64 by the TypeId guard. `result` was just allocated
            // and is not aliased by the immutable a_data/b_data borrows, so the
            // produced &mut [f64] is unique for the duration of this block.
            let c_f64 =
                unsafe { &mut *(std::ptr::from_mut::<[T]>(result.as_mut_slice()) as *mut [f64]) };
            // Under `--features mkl` the function head short-circuited
            // f64 through `mm_raw_bt_mkl_f64`, so this faer branch only
            // runs in the no-mkl build.
            let a_mat = faer::mat::MatRef::from_row_major_slice(a_f64, m, k);
            let b_mat = faer::mat::MatRef::from_row_major_slice(b_f64, n, k).transpose();
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(c_f64, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f64,
                par,
            );
        } else {
            // Fallback for f16: upcast to f64, run faer, downcast.
            // (`a_f64`/`b_f64` use `to_f64` -> Option; for the supported
            // dtype set this never returns None — `T::from(f64)` on the
            // way back may legitimately fail but the original code chose
            // unwrap; preserving that contract.)
            let a_f64: Vec<f64> = a_data
                .iter()
                .map(|&v| num_traits::cast::<T, f64>(v).unwrap_or(0.0))
                .collect();
            let b_f64: Vec<f64> = b_data
                .iter()
                .map(|&v| num_traits::cast::<T, f64>(v).unwrap_or(0.0))
                .collect();
            let mut r_f64 = vec![0.0f64; m * n];
            let a_mat = faer::mat::MatRef::from_row_major_slice(&a_f64, m, k);
            let b_mat = faer::mat::MatRef::from_row_major_slice(&b_f64, n, k).transpose();
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(&mut r_f64, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f64,
                par,
            );
            for (r, &v) in result.iter_mut().zip(r_f64.iter()) {
                *r = T::from(v).unwrap();
            }
        }
        result
    }
}

/// Matrix multiply with A transposed: A^T @ B.
/// A is (K,M) stored row-major, B is (K,N) row-major, result is (M,N).
/// Computes C[i,j] = sum_k A[k,i] * B[k,j] = A^T @ B without materializing the transpose.
pub fn mm_raw_at<T: Float>(a_data: &[T], b_data: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    // Under `--features mkl`, route every f32/f64 multiply through MKL
    // for byte-for-byte parity vs torch. See the equivalent guard in
    // `mm_raw` for the full rationale (closes #1348).
    #[cfg(feature = "mkl")]
    {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            return mm_raw_at_mkl_f32::<T>(a_data, b_data, m, k, n);
        }
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            return mm_raw_at_mkl_f64::<T>(a_data, b_data, m, k, n);
        }
    }
    let max_dim = m.max(n).max(k);
    let zero = <T as num_traits::Zero>::zero();

    if max_dim <= DIRECT_MM_THRESHOLD {
        // Direct loop: A is (K,M) row-major, B is (K,N) row-major.
        // C[i,j] = sum_p A[p,i] * B[p,j]
        let mut result = vec![zero; m * n];
        if is_bf16::<T>() {
            // bf16 fast path with f32 accumulator (see note in `mm_raw`).
            // SAFETY: is_bf16::<T>() returned true so T == half::bf16,
            // satisfying as_bf16_slice and write_f32_as_bf16 contracts.
            // A is (K,M) and B is (K,N), so for p<k, i<m, j<n we have
            // a_row+i = p*m+i < k*m and b_row+j = p*n+j < k*n. acc_buf and
            // result both have length m*n, so r_row+j = i*n+j < m*n. All
            // get_unchecked accesses are in bounds.
            unsafe {
                let a_bf16 = as_bf16_slice(a_data);
                let b_bf16 = as_bf16_slice(b_data);
                let mut acc_buf = vec![0.0f32; m * n];
                for p in 0..k {
                    let a_row = p * m;
                    let b_row = p * n;
                    for i in 0..m {
                        let a_val = a_bf16.get_unchecked(a_row + i).to_f32();
                        let r_row = i * n;
                        for j in 0..n {
                            *acc_buf.get_unchecked_mut(r_row + j) +=
                                a_val * b_bf16.get_unchecked(b_row + j).to_f32();
                        }
                    }
                }
                write_f32_as_bf16(&mut result, &acc_buf);
            }
        } else {
            // SAFETY: A is (K,M) and B is (K,N) row-major (function contract);
            // result was allocated with `vec![zero; m*n]`. p*m+i < k*m for
            // p<k,i<m; p*n+j < k*n for p<k,j<n; i*n+j < m*n for i<m,j<n.
            // All get_unchecked accesses are in bounds.
            unsafe {
                for p in 0..k {
                    let a_row = p * m;
                    let b_row = p * n;
                    for i in 0..m {
                        let a_val = *a_data.get_unchecked(a_row + i);
                        let r_row = i * n;
                        for j in 0..n {
                            let r = result.get_unchecked_mut(r_row + j);
                            *r += a_val * *b_data.get_unchecked(b_row + j);
                        }
                    }
                }
            }
        }
        result
    } else {
        // Large matrices — use faer GEMM with zero-copy transposed A view.
        // A is (K,M) row-major. Wrap as (K,M) MatRef then .transpose() to get (M,K).
        let mut result = vec![zero; m * n];
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // SAFETY: TypeId guard above proves T == f32, so &[T] and &[f32]
            // have identical layout; the cast is a layout-preserving reinterpret.
            let a_f32 = unsafe { &*(std::ptr::from_ref::<[T]>(a_data) as *const [f32]) };
            // SAFETY: same as a_f32 above (T == f32 by the TypeId guard).
            let b_f32 = unsafe { &*(std::ptr::from_ref::<[T]>(b_data) as *const [f32]) };
            // SAFETY: T == f32 by the TypeId guard. `result` was just allocated
            // and is not aliased by the immutable a_data/b_data borrows, so the
            // produced &mut [f32] is unique for the duration of this block.
            let c_f32 =
                unsafe { &mut *(std::ptr::from_mut::<[T]>(result.as_mut_slice()) as *mut [f32]) };
            // Under `--features mkl` the function head short-circuited
            // f32 through `mm_raw_at_mkl_f32`, so this faer branch only
            // runs in the no-mkl build.
            // A is (K,M) row-major; transpose gives (M,K) view — zero copy.
            let a_mat = faer::mat::MatRef::from_row_major_slice(a_f32, k, m).transpose();
            let b_mat = faer::mat::MatRef::from_row_major_slice(b_f32, k, n);
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(c_f32, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f32,
                par,
            );
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            // SAFETY: TypeId guard above proves T == f64, so &[T] and &[f64]
            // have identical layout; the cast is a layout-preserving reinterpret.
            let a_f64 = unsafe { &*(std::ptr::from_ref::<[T]>(a_data) as *const [f64]) };
            // SAFETY: same as a_f64 above (T == f64 by the TypeId guard).
            let b_f64 = unsafe { &*(std::ptr::from_ref::<[T]>(b_data) as *const [f64]) };
            // SAFETY: T == f64 by the TypeId guard. `result` was just allocated
            // and is not aliased by the immutable a_data/b_data borrows, so the
            // produced &mut [f64] is unique for the duration of this block.
            let c_f64 =
                unsafe { &mut *(std::ptr::from_mut::<[T]>(result.as_mut_slice()) as *mut [f64]) };
            // Under `--features mkl` the function head short-circuited
            // f64 through `mm_raw_at_mkl_f64`, so this faer branch only
            // runs in the no-mkl build.
            let a_mat = faer::mat::MatRef::from_row_major_slice(a_f64, k, m).transpose();
            let b_mat = faer::mat::MatRef::from_row_major_slice(b_f64, k, n);
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(c_f64, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f64,
                par,
            );
        } else {
            // Fallback for f16: upcast to f64, run faer, downcast.
            let a_f64: Vec<f64> = a_data
                .iter()
                .map(|&v| num_traits::cast::<T, f64>(v).unwrap_or(0.0))
                .collect();
            let b_f64: Vec<f64> = b_data
                .iter()
                .map(|&v| num_traits::cast::<T, f64>(v).unwrap_or(0.0))
                .collect();
            let mut r_f64 = vec![0.0f64; m * n];
            let a_mat = faer::mat::MatRef::from_row_major_slice(&a_f64, k, m).transpose();
            let b_mat = faer::mat::MatRef::from_row_major_slice(&b_f64, k, n);
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(&mut r_f64, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f64,
                par,
            );
            for (r, &v) in result.iter_mut().zip(r_f64.iter()) {
                *r = T::from(v).unwrap();
            }
        }
        result
    }
}

/// Matrix-matrix multiply: (M,K) @ (K,N) -> (M,N).
pub fn mm<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "mm requires 2-D tensors, got {:?} and {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }

    // Materialize non-contiguous views (e.g. from transpose/permute).
    let a = if a.is_contiguous() {
        a.clone()
    } else {
        a.contiguous()?
    };
    let b = if b.is_contiguous() {
        b.clone()
    } else {
        b.contiguous()?
    };

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    if k != b.shape()[0] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "mm: inner dimensions mismatch: ({},{}) @ ({},{})",
                m,
                k,
                b.shape()[0],
                n
            ),
        });
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let result = mm_raw(a_data, b_data, m, k, n);

    Tensor::from_storage(TensorStorage::cpu(result), vec![m, n], false)
}

/// Matrix-vector multiply: (M,K) @ (K,) -> (M,).
pub fn mv<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 2 || b.ndim() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "mv requires (2-D, 1-D), got {:?} and {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }

    let m = a.shape()[0];
    let k = a.shape()[1];

    if k != b.shape()[0] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "mv: dimension mismatch: ({},{}) @ ({},)",
                m,
                k,
                b.shape()[0]
            ),
        });
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut result = vec![<T as num_traits::Zero>::zero(); m];

    for i in 0..m {
        let mut acc = <T as num_traits::Zero>::zero();
        for p in 0..k {
            acc += a_data[i * k + p] * b_data[p];
        }
        result[i] = acc;
    }

    Tensor::from_storage(TensorStorage::cpu(result), vec![m], false)
}

/// Vector-matrix multiply: (K,) @ (K,N) -> (N,).
fn vm<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let k = a.shape()[0];
    let n = b.shape()[1];

    if k != b.shape()[0] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "vm: dimension mismatch: ({},) @ ({},{})",
                k,
                b.shape()[0],
                n
            ),
        });
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut result = vec![<T as num_traits::Zero>::zero(); n];

    for j in 0..n {
        let mut acc = <T as num_traits::Zero>::zero();
        for p in 0..k {
            acc += a_data[p] * b_data[p * n + j];
        }
        result[j] = acc;
    }

    Tensor::from_storage(TensorStorage::cpu(result), vec![n], false)
}

/// Batched matrix multiply: [B, M, K] @ [B, K, N] -> [B, M, N].
///
/// Loops over the batch dimension and calls `mm` for each slice.
pub fn bmm<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 3 || b.ndim() != 3 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "bmm requires 3-D tensors, got {:?} and {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }

    let batch = a.shape()[0];
    let m = a.shape()[1];
    let k = a.shape()[2];
    let n = b.shape()[2];

    if b.shape()[0] != batch {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "bmm: batch dimensions mismatch: {} vs {}",
                batch,
                b.shape()[0]
            ),
        });
    }
    if k != b.shape()[1] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "bmm: inner dimensions mismatch: ({},{},{}) @ ({},{},{})",
                batch,
                m,
                k,
                b.shape()[0],
                b.shape()[1],
                n
            ),
        });
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let slice_a = m * k;
    let slice_b = k * n;
    let slice_c = m * n;
    let mut result = vec![<T as num_traits::Zero>::zero(); batch * slice_c];

    for bi in 0..batch {
        let a_off = bi * slice_a;
        let b_off = bi * slice_b;
        let c_off = bi * slice_c;
        for i in 0..m {
            for j in 0..n {
                let mut acc = <T as num_traits::Zero>::zero();
                for p in 0..k {
                    acc += a_data[a_off + i * k + p] * b_data[b_off + p * n + j];
                }
                result[c_off + i * n + j] = acc;
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(result), vec![batch, m, n], false)
}

/// Transpose a 2-D tensor.
pub fn transpose<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("transpose requires 2-D tensor, got {:?}", input.shape()),
        });
    }

    let m = input.shape()[0];
    let n = input.shape()[1];
    let data = input.data()?;
    let mut result = vec![<T as num_traits::Zero>::zero(); m * n];

    for i in 0..m {
        for j in 0..n {
            result[j * m + i] = data[i * n + j];
        }
    }

    Tensor::from_storage(TensorStorage::cpu(result), vec![n, m], false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_dot() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[4.0, 5.0, 6.0], &[3]);
        let c = dot(&a, &b).unwrap();
        assert!(c.is_scalar());
        assert!((c.item().unwrap() - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_mm() {
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = mm(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let d = c.data().unwrap();
        assert!((d[0] - 19.0).abs() < 1e-6);
        assert!((d[1] - 22.0).abs() < 1e-6);
        assert!((d[2] - 43.0).abs() < 1e-6);
        assert!((d[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_mv() {
        // [[1, 2], [3, 4]] @ [5, 6] = [17, 39]
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0], &[2]);
        let c = mv(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2]);
        let d = c.data().unwrap();
        assert!((d[0] - 17.0).abs() < 1e-6);
        assert!((d[1] - 39.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_dispatch() {
        // 1D x 1D -> dot
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[4.0, 5.0, 6.0], &[3]);
        let c = matmul(&a, &b).unwrap();
        assert!(c.is_scalar());

        // 2D x 2D -> mm
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    // -------------------------------------------------------------------
    // broadcast_matmul tests
    // -------------------------------------------------------------------

    #[test]
    fn test_matmul_3d_3d_same_batch() {
        // (2, 2, 3) @ (2, 3, 2) -> (2, 2, 2)
        // Batch 0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,0]] = [[4,2],[10,5]]
        // Batch 1: identity-like
        #[rustfmt::skip]
        let a = t(&[
            1.0, 2.0, 3.0,  4.0, 5.0, 6.0,   // batch 0
            1.0, 0.0, 0.0,  0.0, 1.0, 0.0,   // batch 1
        ], &[2, 2, 3]);
        #[rustfmt::skip]
        let b = t(&[
            1.0, 0.0,  0.0, 1.0,  1.0, 0.0,  // batch 0
            1.0, 2.0,  3.0, 4.0,  5.0, 6.0,  // batch 1
        ], &[2, 3, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);
        let d = c.data().unwrap();
        // Batch 0: [[1*1+2*0+3*1, 1*0+2*1+3*0], [4*1+5*0+6*1, 4*0+5*1+6*0]]
        //        = [[4, 2], [10, 5]]
        assert!((d[0] - 4.0).abs() < 1e-6);
        assert!((d[1] - 2.0).abs() < 1e-6);
        assert!((d[2] - 10.0).abs() < 1e-6);
        assert!((d[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_3d_2d_broadcast() {
        // (2, 3, 4) @ (4, 2) -> (2, 3, 2)
        // The 2D right operand broadcasts over the batch dim.
        let a = t(&[1.0; 2 * 3 * 4], &[2, 3, 4]);
        let b = t(&[1.0; 4 * 2], &[4, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3, 2]);
        // Each element = sum of 4 ones = 4.0
        for &v in c.data().unwrap() {
            assert!((v - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_2d_3d_broadcast() {
        // (3, 4) @ (2, 4, 2) -> (2, 3, 2)
        let a = t(&[1.0; 3 * 4], &[3, 4]);
        let b = t(&[1.0; 2 * 4 * 2], &[2, 4, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3, 2]);
    }

    #[test]
    fn test_matmul_batch_broadcast_1_vs_n() {
        // (1, 2, 3) @ (4, 3, 2) -> (4, 2, 2) — batch dim 1 broadcasts to 4
        let a = t(&[1.0; 2 * 3], &[1, 2, 3]);
        let b = t(&[1.0; 4 * 3 * 2], &[4, 3, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[4, 2, 2]);
    }

    #[test]
    fn test_matmul_4d() {
        // (2, 3, 2, 4) @ (2, 3, 4, 5) -> (2, 3, 2, 5)
        let a = t(&[1.0; 2 * 3 * 2 * 4], &[2, 3, 2, 4]);
        let b = t(&vec![1.0; 2 * 3 * 4 * 5], &[2, 3, 4, 5]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3, 2, 5]);
    }

    #[test]
    fn test_matmul_3d_1d() {
        // (2, 3, 4) @ (4,) -> (2, 3) — 1D promoted to (4,1), col squeezed
        let a = t(&[1.0; 2 * 3 * 4], &[2, 3, 4]);
        let b = t(&[1.0; 4], &[4]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        for &v in c.data().unwrap() {
            assert!((v - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_1d_3d() {
        // (4,) @ (2, 4, 3) -> (2, 3) — 1D promoted to (1,4), row squeezed
        let a = t(&[1.0; 4], &[4]);
        let b = t(&[1.0; 2 * 4 * 3], &[2, 4, 3]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_matmul_broadcast_mismatch() {
        // (2, 3, 4) @ (3, 4, 2) — batch dims 2 vs 3, not broadcastable
        let a = t(&[1.0; 2 * 3 * 4], &[2, 3, 4]);
        let b = t(&[1.0; 3 * 4 * 2], &[3, 4, 2]);
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_matmul_inner_dim_mismatch() {
        // (2, 3, 4) @ (2, 5, 2) — inner dims 4 vs 5
        let a = t(&[1.0; 2 * 3 * 4], &[2, 3, 4]);
        let b = t(&[1.0; 2 * 5 * 2], &[2, 5, 2]);
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_mm_shape_mismatch() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(mm(&a, &b).is_err());
    }

    #[test]
    fn test_transpose() {
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = transpose(&a).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.data().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // -------------------------------------------------------------------
    // bmm tests
    // -------------------------------------------------------------------

    #[test]
    fn test_bmm_forward_shape() {
        // [2, 3, 4] @ [2, 4, 5] -> [2, 3, 5]
        let a = t(&[1.0; 2 * 3 * 4], &[2, 3, 4]);
        let b = t(&[1.0; 2 * 4 * 5], &[2, 4, 5]);
        let c = bmm(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3, 5]);
    }

    #[test]
    fn test_bmm_forward_correctness() {
        // Batch 0: [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        // Batch 1: [[1, 0], [0, 1]] @ [[9, 10], [11, 12]] = [[9, 10], [11, 12]]
        #[rustfmt::skip]
        let a_data: Vec<f32> = vec![
            // batch 0
            1.0, 2.0, 3.0, 4.0,
            // batch 1 (identity)
            1.0, 0.0, 0.0, 1.0,
        ];
        #[rustfmt::skip]
        let b_data: Vec<f32> = vec![
            // batch 0
            5.0, 6.0, 7.0, 8.0,
            // batch 1
            9.0, 10.0, 11.0, 12.0,
        ];
        let a = t(&a_data, &[2, 2, 2]);
        let b = t(&b_data, &[2, 2, 2]);
        let c = bmm(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);

        let d = c.data().unwrap();
        // batch 0
        assert!((d[0] - 19.0).abs() < 1e-6);
        assert!((d[1] - 22.0).abs() < 1e-6);
        assert!((d[2] - 43.0).abs() < 1e-6);
        assert!((d[3] - 50.0).abs() < 1e-6);
        // batch 1 (identity @ B = B)
        assert!((d[4] - 9.0).abs() < 1e-6);
        assert!((d[5] - 10.0).abs() < 1e-6);
        assert!((d[6] - 11.0).abs() < 1e-6);
        assert!((d[7] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_bmm_batch_size_1() {
        // Single batch should behave like mm.
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[1, 2, 2]);
        let c = bmm(&a, &b).unwrap();
        assert_eq!(c.shape(), &[1, 2, 2]);

        let d = c.data().unwrap();
        // Same result as mm: [[19, 22], [43, 50]]
        assert!((d[0] - 19.0).abs() < 1e-6);
        assert!((d[1] - 22.0).abs() < 1e-6);
        assert!((d[2] - 43.0).abs() < 1e-6);
        assert!((d[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_bmm_shape_mismatch() {
        // Batch dimension mismatch.
        let a = t(&[1.0; 2 * 2 * 2], &[2, 2, 2]);
        let b = t(&[1.0; 3 * 2 * 2], &[3, 2, 2]);
        assert!(bmm(&a, &b).is_err());

        // Inner dimension mismatch.
        let a = t(&[1.0; 2 * 2 * 3], &[2, 2, 3]);
        let b = t(&[1.0; 2 * 4 * 2], &[2, 4, 2]);
        assert!(bmm(&a, &b).is_err());

        // Wrong ndim.
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[1.0; 2 * 2], &[1, 2, 2]);
        assert!(bmm(&a, &b).is_err());
    }
}
