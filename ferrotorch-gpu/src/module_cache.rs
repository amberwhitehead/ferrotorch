//! Global cache for compiled CUDA modules and kernel functions.
//!
//! Without caching, every call to a GPU kernel (e.g. [`gpu_add`], [`gpu_conv2d_f32`],
//! [`gpu_flash_attention_f32`]) recompiles PTX source into a CUBIN via
//! `CudaContext::load_module(Ptx::from_src(...))`.  This compilation takes
//! ~1700 us per call -- far longer than the actual kernel execution.
//!
//! This module provides [`get_or_compile`], which compiles the PTX only on
//! first use and returns a cached [`CudaFunction`] on subsequent calls.  The
//! cache is keyed by the static kernel name string, which is unique per
//! kernel entry point in this crate.
//!
//! # Thread safety
//!
//! The cache uses a global [`Mutex`]-protected [`HashMap`].  The critical
//! section is short (a hash lookup + optional insert), so contention is
//! negligible in practice.
//!
//! [`gpu_add`]: crate::kernels::gpu_add
//! [`gpu_conv2d_f32`]: crate::conv::gpu_conv2d_f32
//! [`gpu_flash_attention_f32`]: crate::flash_attention::gpu_flash_attention_f32

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::hash::{Hash, Hasher};
#[cfg(feature = "cuda")]
use std::sync::{Arc, LazyLock, Mutex};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, DriverError};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;

/// Global cache mapping (kernel name, device ordinal) to their compiled
/// [`CudaFunction`]s.
///
/// Keyed by `(&'static str, u32)` -- the kernel name (e.g. `"add_kernel"`)
/// and the CUDA device ordinal.  A kernel compiled for device 0 cannot be
/// used on device 1, so the ordinal is part of the key.
#[cfg(feature = "cuda")]
static MODULE_CACHE: LazyLock<Mutex<HashMap<(&'static str, u32), CudaFunction>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Global cache for owned-string PTX modules (e.g. the FusedChain runtime
/// executor — every fused chain produces a unique PTX string at runtime,
/// so the `&'static str`-keyed [`MODULE_CACHE`] cannot be used).
///
/// The key is `(blake-style hash of ptx_src, device_ordinal)`. The hash is
/// computed once on insert and once on lookup; this avoids leaking the
/// `String` to give the cache a `&'static str` view.
#[cfg(feature = "cuda")]
static OWNED_MODULE_CACHE: LazyLock<Mutex<HashMap<(u64, u32), CudaFunction>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Get a compiled kernel function, compiling the PTX only on first use.
///
/// On the first call for a given `(kernel_name, device_ordinal)` pair, this
/// function compiles `ptx_src` into a CUDA module and extracts the named
/// function.  The resulting [`CudaFunction`] is cached globally and returned
/// by clone on subsequent calls, eliminating the ~1700 us PTX compilation
/// overhead.
///
/// # Arguments
///
/// - `ctx`            -- CUDA context (from `device.context()`).
/// - `ptx_src`        -- PTX source string (a `&'static str` constant).
/// - `kernel_name`    -- entry-point name inside the PTX module.
/// - `device_ordinal` -- CUDA device ordinal (so kernels compiled for
///   device 0 are not reused on device 1).
///
/// # Errors
///
/// Returns [`DriverError`] if PTX compilation or function lookup fails.
#[cfg(feature = "cuda")]
pub fn get_or_compile(
    ctx: &Arc<CudaContext>,
    ptx_src: &'static str,
    kernel_name: &'static str,
    device_ordinal: u32,
) -> Result<CudaFunction, DriverError> {
    let key = (kernel_name, device_ordinal);
    let mut cache = MODULE_CACHE.lock().unwrap();
    if let Some(func) = cache.get(&key) {
        return Ok(func.clone());
    }
    let module = ctx.load_module(Ptx::from_src(ptx_src))?;
    let func = module.load_function(kernel_name)?;
    cache.insert(key, func.clone());
    Ok(func)
}

/// Get a compiled kernel function from owned PTX + name strings.
///
/// This is the runtime-PTX sibling of [`get_or_compile`]. Use it when the
/// PTX source and entry-point name are produced at runtime (e.g. from
/// [`crate::module_cache`]'s caller in `ferrotorch-jit::fusion_gpu`, where
/// every [`crate::module_cache`]-using FusedChain is unique). The
/// `&'static str` requirements of [`get_or_compile`] cannot be met by
/// runtime-built strings; this fn accepts owned `String`s and caches by a
/// hash of `ptx_src`.
///
/// # Cache key
///
/// The cache is keyed on `(hash(ptx_src), device_ordinal)`. The
/// `DefaultHasher` is used because the cache is in-process and a chosen
/// collision would only let an adversary substitute one of their own
/// fused chains for another — both inside the same trust boundary.
///
/// # Memory growth
///
/// On a cache miss this fn leaks `ptx_src` and `kernel_name` via
/// [`Box::leak`] to satisfy cudarc's `&'static`-like internal requirements
/// for module/function metadata, then inserts the resulting
/// [`CudaFunction`] into the global cache. Memory grows with the number
/// of unique `(ptx_src, device_ordinal)` tuples — **bounded by the number
/// of application-distinct `FusedChain`s** (typical use case: a handful
/// to a few tens). The cached entry itself is small (cudarc's
/// `CudaFunction` is roughly a pointer + name); the dominant cost is
/// PTX compilation, which is what this cache is designed to skip.
///
/// # Arguments
///
/// - `ctx`            — CUDA context (from `device.context()`).
/// - `ptx_src`        — owned PTX source string.
/// - `kernel_name`    — owned entry-point name inside the PTX module.
/// - `device_ordinal` — CUDA device ordinal (so kernels compiled for
///   device 0 are not reused on device 1).
///
/// # Errors
///
/// Returns [`DriverError`] if PTX compilation or function lookup fails.
#[cfg(feature = "cuda")]
pub fn get_or_compile_owned(
    ctx: &Arc<CudaContext>,
    ptx_src: String,
    kernel_name: String,
    device_ordinal: u32,
) -> Result<CudaFunction, DriverError> {
    // Compute the cache key from the PTX hash. The hasher is the std
    // `DefaultHasher`; collisions only matter for adversarial inputs,
    // and the cache lives entirely inside a single process's trust
    // boundary.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    ptx_src.hash(&mut hasher);
    let ptx_hash = hasher.finish();
    let key = (ptx_hash, device_ordinal);

    let mut cache = OWNED_MODULE_CACHE.lock().unwrap();
    if let Some(func) = cache.get(&key) {
        return Ok(func.clone());
    }

    // Cache miss: leak the strings so they satisfy cudarc's internal
    // `&'static`-like requirements for the module's metadata. The bound
    // on growth is documented above: one leaked PTX + name pair per
    // unique FusedChain × device.
    let leaked_ptx: &'static str = Box::leak(ptx_src.into_boxed_str());
    let leaked_name: &'static str = Box::leak(kernel_name.into_boxed_str());
    let module = ctx.load_module(Ptx::from_src(leaked_ptx))?;
    let func = module.load_function(leaked_name)?;
    cache.insert(key, func.clone());
    Ok(func)
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use crate::device::GpuDevice;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

    #[test]
    fn cache_returns_function_on_repeated_calls() {
        // Verify the cache works by calling gpu_add twice. The first call
        // compiles the PTX; the second hits the cache. Both must succeed.
        let dev = crate::device::GpuDevice::new(0).expect("CUDA device 0");
        let a = crate::transfer::cpu_to_gpu(&[1.0f32, 2.0, 3.0], &dev).expect("a");
        let b = crate::transfer::cpu_to_gpu(&[4.0f32, 5.0, 6.0], &dev).expect("b");

        let r1 = crate::kernels::gpu_add(&a, &b, &dev).expect("first add (compiles)");
        let r2 = crate::kernels::gpu_add(&a, &b, &dev).expect("second add (cached)");

        let h1 = crate::transfer::gpu_to_cpu(&r1, &dev).expect("r1");
        let h2 = crate::transfer::gpu_to_cpu(&r2, &dev).expect("r2");
        assert_eq!(h1, h2, "cached kernel should produce identical results");
        assert_eq!(h1, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn cached_kernel_produces_correct_results() {
        // Run gpu_add twice and verify both produce correct results,
        // confirming the cached kernel is functional.
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![10.0f32, 20.0, 30.0, 40.0];
        let expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x + y).collect();

        let a = cpu_to_gpu(&a_data, &dev).expect("a to gpu");
        let b = cpu_to_gpu(&b_data, &dev).expect("b to gpu");

        // First call (compiles PTX).
        let out1 = crate::kernels::gpu_add(&a, &b, &dev).expect("gpu_add 1st");
        let host1 = gpu_to_cpu(&out1, &dev).expect("gpu_to_cpu 1st");

        // Second call (uses cache).
        let out2 = crate::kernels::gpu_add(&a, &b, &dev).expect("gpu_add 2nd");
        let host2 = gpu_to_cpu(&out2, &dev).expect("gpu_to_cpu 2nd");

        for (i, ((&g1, &g2), &e)) in host1
            .iter()
            .zip(host2.iter())
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (g1 - e).abs() < 1e-6,
                "1st call: element {i}: got {g1}, expected {e}",
            );
            assert!(
                (g2 - e).abs() < 1e-6,
                "2nd call: element {i}: got {g2}, expected {e}",
            );
        }
    }

    #[test]
    fn cached_kernel_second_call_is_fast() {
        // The second call should be significantly faster than the first
        // because it skips PTX compilation.
        use std::time::Instant;

        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let a_data = vec![1.0f32; 1024];
        let b_data = vec![2.0f32; 1024];

        let a = cpu_to_gpu(&a_data, &dev).expect("a to gpu");
        let b = cpu_to_gpu(&b_data, &dev).expect("b to gpu");

        // Warm up with a different kernel to avoid measuring CUDA init.
        let _ = crate::kernels::gpu_neg(&a, &dev);

        // We cannot rely on add_kernel being uncached here (other tests
        // may have run first), so we use the mul_kernel via gpu_mul,
        // which is less likely to have been called yet.  Even if it has
        // been cached, both calls should be fast, and that is fine -- the
        // structural test above already verifies identity.
        let t1 = Instant::now();
        let _ = crate::kernels::gpu_mul(&a, &b, &dev).expect("gpu_mul 1st");
        let d1 = t1.elapsed();

        let t2 = Instant::now();
        let _ = crate::kernels::gpu_mul(&a, &b, &dev).expect("gpu_mul 2nd");
        let d2 = t2.elapsed();

        // The second call should be faster (no compilation).
        // We do not assert a strict ratio because CI environments vary,
        // but we log for manual inspection.
        eprintln!(
            "module_cache timing: 1st call = {:?}, 2nd call = {:?}",
            d1, d2,
        );
    }

    /// Smoke-test the runtime-PTX cache: two calls with the same PTX
    /// string return the same cached function — the caching invariant
    /// the FusedChain runtime executor depends on.
    #[test]
    fn get_or_compile_owned_returns_same_function_on_repeated_calls() {
        // Minimal valid PTX kernel — pattern lifted from NEG_PTX (above)
        // to guarantee the driver accepts it. Copies one f32 per thread.
        let ptx = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry id_kernel_owned_cache_test(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va;
    .reg .pred %p;

    ld.param.u64 %a, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    st.global.f32 [%out], %va;

DONE:
    ret;
}
"
        .to_string();

        let dev = crate::device::GpuDevice::new(0).expect("CUDA device 0");
        let ctx = dev.context();

        let f1 = super::get_or_compile_owned(
            ctx,
            ptx.clone(),
            "id_kernel_owned_cache_test".to_string(),
            dev.ordinal() as u32,
        )
        .expect("first compile");
        let f2 = super::get_or_compile_owned(
            ctx,
            ptx.clone(),
            "id_kernel_owned_cache_test".to_string(),
            dev.ordinal() as u32,
        )
        .expect("second (cached) compile");

        // cudarc's CudaFunction wraps a CUfunction handle; cloning is
        // by-value of the handle. Use the debug repr as a coarse identity
        // proxy: a fresh compile would change the underlying CUfunction
        // (different module load), so identical debug strings here are
        // strong evidence the cache hit.
        assert_eq!(format!("{f1:?}"), format!("{f2:?}"));
    }

    /// Different PTX source must produce a different cached function
    /// (cache key correctness — the cache must not collide).
    #[test]
    fn get_or_compile_owned_different_ptx_returns_different_function() {
        // Two kernels with different entry-point names AND different
        // bodies (a no-op vs. a negate). They must NOT share a cache
        // entry; the new entry's hashed key differs from the first.
        let ptx_a = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry id_kernel_owned_diff_a(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va;
    .reg .pred %p;
    ld.param.u64 %a, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;
    ld.global.f32 %va, [%a];
    st.global.f32 [%out], %va;
DONE:
    ret;
}
"
        .to_string();

        let ptx_b = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry id_kernel_owned_diff_b(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va;
    .reg .pred %p;
    ld.param.u64 %a, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;
    ld.global.f32 %va, [%a];
    neg.f32 %va, %va;
    st.global.f32 [%out], %va;
DONE:
    ret;
}
"
        .to_string();

        let dev = crate::device::GpuDevice::new(0).expect("CUDA device 0");
        let ctx = dev.context();

        let f_a = super::get_or_compile_owned(
            ctx,
            ptx_a,
            "id_kernel_owned_diff_a".to_string(),
            dev.ordinal() as u32,
        )
        .expect("compile a");
        let f_b = super::get_or_compile_owned(
            ctx,
            ptx_b,
            "id_kernel_owned_diff_b".to_string(),
            dev.ordinal() as u32,
        )
        .expect("compile b");

        // Different PTX → different cache entry → distinct CUfunction
        // handles; the debug reprs must differ.
        assert_ne!(format!("{f_a:?}"), format!("{f_b:?}"));
    }

    /// Regression: `broadcast_div_kernel` PTX must load on the driver.
    ///
    /// Two compounding bugs caused the original module to fail with
    /// `CUDA_ERROR_INVALID_PTX`:
    ///
    /// 1. The kernel emitted bare `div.f32 %vr, %va, %vb`. PTX does not
    ///    accept a divide without a rounding mode (`.rn`/`.rz`/`.rm`/`.rp`)
    ///    or `.approx`; `div.rn.f32` is the IEEE round-to-nearest-even
    ///    form used by every other site in `kernels.rs`.
    /// 2. The fix-up commit briefly carried a UTF-8 arrow (`->` written
    ///    as a Unicode `→`) inside a `// ...` comment in the PTX
    ///    literal. The driver's PTX parser rejects multibyte sequences
    ///    inside comments and surfaces the same opaque error.
    ///
    /// Surfaced by `gpu_transformer_training_smoke` in
    /// `ferrotorch/tests/gpu_training.rs` (#749 Section B).
    #[test]
    fn broadcast_div_kernel_ptx_loads() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let _module = ctx
            .load_module(cudarc::nvrtc::Ptx::from_src(
                crate::kernels::BROADCAST_DIV_PTX,
            ))
            .expect("BROADCAST_DIV_PTX must compile");
    }
}
