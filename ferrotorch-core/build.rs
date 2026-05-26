//! Build script for `ferrotorch-core`.
//!
//! Gated entirely on `CARGO_FEATURE_MKL`. When the `mkl` feature is OFF
//! (the default), this script does nothing — ferrotorch-core falls back
//! to the pure-Rust faer GEMM and there is no system dependency.
//!
//! When `mkl` is ON, the script wires the crate to the **system MKL
//! 2024.2** dynamic library installed at `$HOME/.local/lib/libmkl_rt.so.2`.
//! That is intentionally the same major+minor MKL that PyTorch's CPU
//! wheel (`torch 2.11.0+cu130`) ships, so calling the Fortran `sgemm_`/
//! `dgemm_` symbols with torch's exact dispatch shape produces
//! byte-for-byte identical results — see the dispatcher port in
//! `src/ops/linalg.rs` and the rationale in `Cargo.toml`'s `mkl` feature
//! comment for the full architectural story (closes #1538 + #1348).
//!
//! The vendored `intel-mkl-src` 2020.1 path that preceded this was
//! abandoned because (a) the 2020.1 vs 2024.2 dispatch tables differ,
//! producing 1-5 ULP drift even with `MKL_CBWR=COMPATIBLE` engaged, and
//! (b) the prior `cblas_sgemm` row-major wrapper picked different MKL
//! micro-kernels than torch's raw `sgemm_` calls. System MKL + raw
//! Fortran symbol resolves both.
//!
//! ## Linker glue
//!
//! MKL ships its primary dispatch library as `libmkl_rt.so.2` (the
//! soname includes the major version). A bare `-lmkl_rt` flag would
//! look for `libmkl_rt.so` (no version suffix) and fail. The script
//! solves this with the standard cargo build-script trick of
//! materialising a symlink in `OUT_DIR/libmkl_rt.so` that points at
//! `$HOME/.local/lib/libmkl_rt.so.2`, then emitting
//! `cargo:rustc-link-search=$OUT_DIR` and `cargo:rustc-link-lib=mkl_rt`.
//! Link time: the linker resolves `-lmkl_rt` via the symlink in
//! OUT_DIR. Run time: the loader uses the recorded soname
//! `libmkl_rt.so.2` and needs to find it via `LD_LIBRARY_PATH` or
//! rpath — the consumer is responsible for setting `LD_LIBRARY_PATH=
//! $HOME/.local/lib` (or rpath-injecting at build time).
//!
//! If `$HOME/.local/lib/libmkl_rt.so.2` is absent, the script emits a
//! `cargo:warning=` and aborts the build with an explicit message
//! rather than silently falling through to a missing-symbol link
//! error.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_MKL");

    if std::env::var_os("CARGO_FEATURE_MKL").is_none() {
        return;
    }

    let home = std::env::var_os("HOME").expect("$HOME must be set when building --features mkl");
    let mkl_dir = PathBuf::from(&home).join(".local").join("lib");
    let mkl_so = mkl_dir.join("libmkl_rt.so.2");

    println!("cargo:rerun-if-changed={}", mkl_so.display());

    if !mkl_so.exists() {
        // Hard error: --features mkl without the lib installed is a
        // misconfiguration we surface explicitly rather than letting
        // the linker fail with a cryptic message.
        panic!(
            "ferrotorch-core(mkl): system MKL 2024.2 not found at {}. \
             Install via e.g. `pip install mkl==2024.2.*` and symlink \
             `$(python -c 'import mkl, os; print(os.path.dirname(mkl.__file__))')/../../libmkl_rt.so.2` \
             to `~/.local/lib/libmkl_rt.so.2`, or otherwise place the soname there.",
            mkl_so.display()
        );
    }

    // Materialise OUT_DIR/libmkl_rt.so → libmkl_rt.so.2 so a bare
    // `-lmkl_rt` flag resolves at link-time. The bare-soname-symlink
    // trick is the standard cargo build-script approach for linking
    // against a versioned soname without forcing every consumer to
    // craft custom RUSTFLAGS.
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set by cargo"));
    let link_name = out_dir.join("libmkl_rt.so");
    // Remove a stale symlink before recreating — if a prior build left
    // a link pointing somewhere else, std::os::unix::fs::symlink would
    // fail with EEXIST.
    if link_name.exists() || link_name.symlink_metadata().is_ok() {
        let _ = std::fs::remove_file(&link_name);
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(&mkl_so, &link_name).unwrap_or_else(|e| {
        panic!(
            "ferrotorch-core(mkl): failed to symlink {} -> {}: {e}",
            link_name.display(),
            mkl_so.display()
        );
    });

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // Also publish the original soname-resident dir so the rpath line
    // below can encode it; the runtime loader still needs to find
    // `libmkl_rt.so.2` itself.
    println!("cargo:rustc-link-search=native={}", mkl_dir.display());
    println!("cargo:rustc-link-lib=mkl_rt");
    // Embed an rpath so consumers don't have to set LD_LIBRARY_PATH
    // for `cargo run`/`cargo test` invocations. This is a runtime
    // convenience; setting LD_LIBRARY_PATH still works if rpath is
    // stripped by a downstream packaging step.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", mkl_dir.display());
}
