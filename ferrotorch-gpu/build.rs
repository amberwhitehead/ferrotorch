//! Build script for `ferrotorch-gpu`.
//!
//! Two responsibilities, both opt-in / no-op when their preconditions
//! are absent:
//!
//! 1. When the `cusparselt` feature is enabled, locate `cusparseLt.h` on
//!    the host, run `bindgen` to emit `cusparselt_sys.rs` into `OUT_DIR`,
//!    and instruct cargo to link against `libcusparseLt.so`. When the
//!    feature is **off**, this is a no-op — the default workspace build
//!    does not require libclang or the cuSPARSELt SDK.
//!
//! 2. When the `cuda` feature is enabled on Linux, force the CUDA-12.x
//!    cuSOLVER (`libcusolver.so.11`) to be the one resolved at runtime
//!    via an `OUT_DIR` compat symlink + an emitted rpath. See
//!    [`cuda_cusolver_compat`] for the full rationale; the short version
//!    is that the workspace pins `CUDARC_CUDA_VERSION=12080`, whose
//!    cuSOLVER bindings dlopen legacy untyped symbols
//!    (`cusolverDnGeqrf` et al.) that exist in CUDA 12.x's
//!    `libcusolver.so.11` but were **removed** in CUDA 13.x's
//!    `libcusolver.so.12`. Without this, the default loader path resolves
//!    the bare `libcusolver.so` to the system CUDA 13.x lib and the first
//!    cuSOLVER call panics with `undefined symbol: cusolverDnGeqrf`.
//!
//! Probe order for the cuSPARSELt SDK header:
//!   1. `$CUSPARSELT_INCLUDE_DIR/cusparseLt.h`
//!   2. `$CUDA_PATH/include/cusparseLt.h`
//!   3. `/usr/local/cuda/include/cusparseLt.h`
//!   4. `/usr/local/cuda-12.9/include/cusparseLt.h`
//!   5. `/usr/local/cuda-12.8/include/cusparseLt.h`
//!   6. `/usr/include/cusparseLt.h`
//!   7. `/opt/nvidia/cusparselt/include/cusparseLt.h`
//!
//! NVIDIA distributes cuSPARSELt as a separate SDK from the CUDA
//! toolkit (it ships in its own tarball / RPM); on systems without it
//! installed the build script emits a `cargo::warning=` and aborts so
//! the user sees a clear path to fix.

fn main() {
    // The script runs unconditionally — but every action below is gated
    // on the relevant `CARGO_FEATURE_*` env var, which cargo sets only
    // when that feature is active. Re-run if a gate flips or any probed
    // env var changes.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUSPARSELT");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    println!("cargo:rerun-if-env-changed=CUSPARSELT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CUSPARSELT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDARC_CUDA_VERSION");

    // Emit the `ferrotorch_cuda13` cfg when the resolved cudarc CUDA version
    // is >= 13000. CUDA 13.x reshaped a few driver structs. Kept as a coarse
    // "any CUDA 13" gate for future use.
    println!("cargo::rustc-check-cfg=cfg(ferrotorch_cuda13)");
    if cuda_version_at_least(13000) {
        println!("cargo::rustc-cfg=ferrotorch_cuda13");
    }

    // `CUmemLocation::id` moved into an anonymous union (`__bindgen_anon_1`)
    // in CUDA 13.0.20 — NOT at the 13.0.0 boundary. cudarc's `cuda-13000`
    // and `cuda-13010` bindings still expose `.id` as a direct field; only
    // `cuda-13020` and later wrap it in the union. Gating the anon-union
    // accessor at the coarser `ferrotorch_cuda13` cfg breaks the build when
    // a host pins to the older 13.0 patch level (e.g. lucida's GB10 driver
    // 580.126.09 doesn't export the 13.0.20+ symbols cudarc dlsyms at
    // startup, so `CUDARC_CUDA_VERSION=13000` is the only working choice).
    println!("cargo::rustc-check-cfg=cfg(ferrotorch_cuda_mem_location_anon_union)");
    if cuda_version_at_least(13020) {
        println!("cargo::rustc-cfg=ferrotorch_cuda_mem_location_anon_union");
    }

    if std::env::var_os("CARGO_FEATURE_CUSPARSELT").is_some() {
        #[cfg(feature = "cusparselt")]
        cusparselt::generate();
    }

    // cuSOLVER 12.x compat shim. Gated on the `cuda` feature (cargo sets
    // CARGO_FEATURE_CUDA) and on Linux only — the soname/loader mechanics
    // below are Linux-specific and the bug is specific to a host with a
    // CUDA-13 default toolkit but a 12080-pinned cudarc (WSL2 + RTX 3090).
    // On macOS / Windows / non-cuda builds this whole block is skipped, so
    // it can never break CI or a host without the 12.x toolkit.
    //
    // Also skip it on a deliberate CUDA-13 build: there the shim can never
    // find a `libcusolver.so.11` and would only emit a misleading warning
    // predicting a `cusolverDnGeqrf` panic — that 12.x→13.x soname problem
    // does not apply when cudarc is itself building against 13.x.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some()
        && cfg!(target_os = "linux")
        && !cuda_version_at_least(13000)
    {
        cuda_cusolver_compat::ensure();
    }
}

/// Resolve the CUDA version cudarc will build against and report whether
/// it is `>= min` (using cudarc's `MAJOR<MINOR:02d><PATCH:02d>` encoding,
/// e.g. `13020` for CUDA 13.0.20). Mirrors cudarc's own resolution order:
/// the `CUDARC_CUDA_VERSION` env var wins; otherwise probe `nvcc --version`
/// ("release 13.0, V13.0.88"). Defaults to false when neither is available,
/// so cfgs gated on this are never emitted by accident.
fn cuda_version_at_least(min: u32) -> bool {
    if let Ok(v) = std::env::var("CUDARC_CUDA_VERSION")
        && let Ok(n) = v.trim().parse::<u32>()
    {
        return n >= min;
    }
    // nvcc fallback: parse "release MAJOR.MINOR, VMAJOR.MINOR.PATCH" out of
    // the second line of `nvcc --version`. The V-line is the authoritative
    // patch number; the "release" line only gives MAJOR.MINOR.
    if let Ok(out) = std::process::Command::new("nvcc").arg("--version").output()
        && let Ok(s) = String::from_utf8(out.stdout)
    {
        // Try the V-line first (e.g. "V13.0.88"): gives full M.N.P.
        if let Some(i) = s.find(", V")
            && let Some(rest) = s.get(i + 3..)
            && let Some(end) = rest.find(char::is_whitespace)
            && let Some(version) = rest.get(..end)
        {
            let parts: Vec<&str> = version.split('.').collect();
            if let (Some(maj), Some(min_), Some(pat)) = (parts.first(), parts.get(1), parts.get(2))
                && let (Ok(m), Ok(n_), Ok(p)) =
                    (maj.parse::<u32>(), min_.parse::<u32>(), pat.parse::<u32>())
            {
                return m * 1000 + n_ * 100 + p >= min;
            }
        }
        // Fallback: just MAJOR.MINOR from "release X.Y" line (patch = 0).
        if let Some(i) = s.find("release ")
            && let Some(rest) = s.get(i + 8..)
            && let Some(comma) = rest.find(',')
            && let Some(version) = rest.get(..comma)
        {
            let parts: Vec<&str> = version.split('.').collect();
            if let (Some(maj), Some(min_)) = (parts.first(), parts.get(1))
                && let (Ok(m), Ok(n_)) = (maj.parse::<u32>(), min_.parse::<u32>())
            {
                return m * 1000 + n_ * 100 >= min;
            }
        }
    }
    false
}

/// Force the CUDA-12.x cuSOLVER (`libcusolver.so.11`) to be resolved at
/// runtime for this crate's binaries and tests.
///
/// ## Why this exists (do not "simplify" it away)
///
/// The workspace pins `CUDARC_CUDA_VERSION=12080` in `.cargo/config.toml`.
/// That pin is **required**: the WSL `libcuda` driver shim
/// (`/usr/lib/wsl/lib/libcuda.so.1`) lacks CUDA-13 driver symbols (e.g.
/// `cuCoredumpDeregisterCompleteCallback`), so building cudarc against
/// CUDA-13 bindings would break the whole driver load. We must stay on the
/// 12.x bindings.
///
/// cudarc's CUDA-12.x cuSOLVER bindings `dlopen` the **legacy untyped**
/// symbols `cusolverDnGeqrf` / `Gesvd` / `Getrf` / `Getrs` / `Potrf` /
/// `Potrs` / `Syevd` eagerly at `Lib` init (an `.expect()`), so the first
/// cuSOLVER call panics if those symbols are absent.
///
/// CUDA 12.x cuSOLVER ships as soname `libcusolver.so.11` and *has* those
/// legacy symbols. CUDA 13.x cuSOLVER ships as `libcusolver.so.12` and
/// **removed** them. On this host `/usr/local/cuda` → `cuda-13.1`, so the
/// dynamic loader resolves the bare `libcusolver.so` to 13.1's `.so.12`
/// (no legacy symbols) ⇒ `undefined symbol: cusolverDnGeqrf` ⇒ panic.
///
/// ## The fix
///
/// The CUDA 12.8 toolkit is installed at
/// `/usr/local/cuda-12.8/.../libcusolver.so.11.7.3.90` and has every
/// symbol cudarc-12080 needs. We:
///   1. Locate that `libcusolver.so.11*` (absolute path).
///   2. Symlink `${OUT_DIR}/cuda-compat/libcusolver.so` (and
///      `libcusolver.so.11`) → it.
///   3. Emit an rpath + link-search for that dir. rpath/RUNPATH is
///      searched before the system default CUDA path, so cudarc's first
///      `dlopen` candidate `libcusolver.so` now resolves to the 12.x lib.
///
/// Self-contained: no `/tmp`, no manual env vars, no sudo, no committed
/// machine-specific symlinks (the symlink lives in `OUT_DIR`, which is
/// build output, not tracked). If no 12.x cuSOLVER is found, it emits a
/// `cargo:warning=` and does nothing — builds on CI / macOS / hosts
/// without the 12.x toolkit are unaffected.
mod cuda_cusolver_compat {
    use std::path::{Path, PathBuf};

    pub fn ensure() {
        let Some(lib) = locate_cusolver_so11() else {
            println!(
                "cargo:warning=ferrotorch-gpu(cuda): no CUDA 12.x cuSOLVER (libcusolver.so.11*) \
                 found. The cudarc 12080 pin needs the legacy cusolverDn* symbols that exist only \
                 in libcusolver.so.11 (CUDA 12.x); a CUDA 13.x libcusolver.so.12 lacks them. \
                 cusolver tests (cusolver::*) may panic with 'undefined symbol: cusolverDnGeqrf'. \
                 Install the CUDA 12.8 toolkit or set CUDA_PATH to a CUDA 12.x prefix. Searched \
                 $CUDA_PATH/targets/x86_64-linux/lib and /usr/local/cuda-12*."
            );
            return;
        };

        // Re-run if the located lib changes (e.g. toolkit upgrade).
        println!("cargo:rerun-if-changed={}", lib.display());

        let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set by cargo"));
        let compat_dir = out_dir.join("cuda-compat");
        if let Err(e) = std::fs::create_dir_all(&compat_dir) {
            println!(
                "cargo:warning=ferrotorch-gpu(cuda): failed to create compat dir {}: {e}. \
                 cusolver tests may fail.",
                compat_dir.display()
            );
            return;
        }

        // Symlink both the bare name and the soname so whichever candidate
        // cudarc's dlopen tries (`libcusolver.so` then `libcusolver.so.11`)
        // resolves to the 12.x lib found above. Overwrite if present.
        for name in ["libcusolver.so", "libcusolver.so.11"] {
            let link = compat_dir.join(name);
            let _ = std::fs::remove_file(&link); // ignore: may not exist
            if let Err(e) = std::os::unix::fs::symlink(&lib, &link) {
                println!(
                    "cargo:warning=ferrotorch-gpu(cuda): failed to symlink {} -> {}: {e}. \
                     cusolver tests may fail.",
                    link.display(),
                    lib.display()
                );
                return;
            }
        }

        let compat_dir_str = compat_dir.to_string_lossy();
        // rpath: the loader searches this dir for `libcusolver.so` *before*
        // the system default CUDA path, so the bare-name dlopen resolves to
        // our 12.x symlink. link-search lets the linker itself find the lib
        // at build time too.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{compat_dir_str}");
        println!("cargo:rustc-link-search=native={compat_dir_str}");
    }

    /// Locate a CUDA-12.x `libcusolver.so.11*` (the versioned file like
    /// `libcusolver.so.11.7.3.90`, or the `.so.11` symlink). Returns an
    /// absolute, canonicalized path. Search order:
    ///   1. `$CUDA_PATH/targets/x86_64-linux/lib`
    ///   2. likely CUDA 12.x roots, then a glob over `/usr/local/cuda-12.*`.
    fn locate_cusolver_so11() -> Option<PathBuf> {
        let mut dirs: Vec<PathBuf> = Vec::new();

        // 1. Honor an explicit CUDA_PATH first.
        if let Some(p) = std::env::var_os("CUDA_PATH") {
            dirs.push(PathBuf::from(&p).join("targets/x86_64-linux/lib"));
            dirs.push(PathBuf::from(&p).join("lib64"));
        }

        // 2. Likely fixed CUDA 12.x roots (newest first).
        for root in [
            "/usr/local/cuda-12.9",
            "/usr/local/cuda-12.8",
            "/usr/local/cuda-12",
        ] {
            dirs.push(PathBuf::from(root).join("targets/x86_64-linux/lib"));
            dirs.push(PathBuf::from(root).join("lib64"));
        }

        // 3. Glob /usr/local/cuda-12.*/targets/x86_64-linux/lib for any
        //    other 12.x toolkit installs (no glob crate dependency: read
        //    /usr/local and filter dir names starting with "cuda-12.").
        if let Ok(entries) = std::fs::read_dir("/usr/local") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("cuda-12.") {
                    dirs.push(entry.path().join("targets/x86_64-linux/lib"));
                    dirs.push(entry.path().join("lib64"));
                }
            }
        }

        for dir in dirs {
            if let Some(found) = find_so11_in(&dir) {
                return found.canonicalize().ok().or(Some(found));
            }
        }
        None
    }

    /// Find a file named `libcusolver.so.11*` in `dir`. Prefers the real
    /// versioned file (e.g. `libcusolver.so.11.7.3.90`) but accepts the
    /// bare `.so.11` symlink — both resolve to the same 12.x library.
    fn find_so11_in(dir: &Path) -> Option<PathBuf> {
        let entries = std::fs::read_dir(dir).ok()?;
        let mut best: Option<PathBuf> = None;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("libcusolver.so.11") {
                // Prefer the longer (fully versioned) name so the rpath
                // points at the concrete file, not a symlink chain.
                match &best {
                    Some(prev) if prev.file_name().map(|n| n.len()).unwrap_or(0) >= name.len() => {}
                    _ => best = Some(entry.path()),
                }
            }
        }
        best
    }
}

#[cfg(feature = "cusparselt")]
mod cusparselt {
    use std::path::{Path, PathBuf};

    /// Header probe + bindgen run + link directives.
    pub fn generate() {
        let header = match locate_header() {
            Some(p) => p,
            None => {
                println!(
                    "cargo:warning=cusparselt feature is enabled but `cusparseLt.h` was not found on this host. Set CUSPARSELT_INCLUDE_DIR to the directory containing cusparseLt.h, or install the NVIDIA cuSPARSELt SDK (https://docs.nvidia.com/cuda/cusparselt/getting_started.html). Searched: $CUSPARSELT_INCLUDE_DIR, $CUDA_PATH/include, /usr/local/cuda/include, /usr/local/cuda-12.*/include, /usr/include, /opt/nvidia/cusparselt/include."
                );
                panic!(
                    "ferrotorch-gpu: cusparselt feature requires cusparseLt.h but none of the probed locations contained it. See cargo:warning above for resolution."
                );
            }
        };

        // Tell rustc to link against `libcusparseLt.so`. The library
        // search path defaults to the system loader path; the user can
        // extend it via CUSPARSELT_LIB_DIR for non-default install
        // prefixes (e.g. /opt/nvidia/cusparselt/lib64).
        if let Ok(dir) = std::env::var("CUSPARSELT_LIB_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
        }
        // Common implicit search paths so `LD_LIBRARY_PATH` is not the
        // only way to find the lib at runtime.
        for candidate in [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda-12.9/lib64",
            "/usr/local/cuda-12.8/lib64",
            "/usr/lib64",
            "/opt/nvidia/cusparselt/lib64",
        ] {
            if Path::new(candidate).exists() {
                println!("cargo:rustc-link-search=native={candidate}");
            }
        }
        println!("cargo:rustc-link-lib=cusparseLt");

        // Re-run if the located header changes.
        println!("cargo:rerun-if-changed={}", header.display());

        let header_str = header.to_string_lossy().to_string();
        let mut builder = bindgen::Builder::default()
            .header(header_str.clone())
            .allowlist_function("cusparseLt.*")
            .allowlist_type("cusparseLt.*")
            .allowlist_var("CUSPARSELT_.*")
            .allowlist_var("CUSPARSE_.*")
            .allowlist_type("cudaDataType.*")
            .allowlist_type("cudaStream_t")
            .allowlist_type("cusparseStatus_t")
            .allowlist_type("cusparseOperation_t")
            .allowlist_type("cusparseComputeType.*")
            .allowlist_type("cusparseOrder_t")
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: false,
            })
            .derive_default(true)
            .derive_debug(true)
            .layout_tests(false)
            .generate_comments(false);

        // Add include path containing the header so bindgen finds the
        // CUDA toolkit headers it transitively depends on.
        if let Some(parent) = header.parent() {
            builder = builder.clang_arg(format!("-I{}", parent.display()));
        }
        for path in cuda_include_dirs() {
            builder = builder.clang_arg(format!("-I{}", path.display()));
        }

        let bindings = builder
            .generate()
            .expect("bindgen failed to generate cusparseLt bindings");

        let out_path = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR set by cargo"))
            .join("cusparselt_sys.rs");
        bindings
            .write_to_file(&out_path)
            .expect("failed to write cusparselt_sys.rs");
    }

    fn locate_header() -> Option<PathBuf> {
        let candidates: Vec<PathBuf> = [
            std::env::var_os("CUSPARSELT_INCLUDE_DIR").map(PathBuf::from),
            std::env::var_os("CUDA_PATH").map(|p| PathBuf::from(p).join("include")),
            Some(PathBuf::from("/usr/local/cuda/include")),
            Some(PathBuf::from("/usr/local/cuda-12.9/include")),
            Some(PathBuf::from("/usr/local/cuda-12.8/include")),
            Some(PathBuf::from("/usr/include")),
            Some(PathBuf::from("/opt/nvidia/cusparselt/include")),
        ]
        .into_iter()
        .flatten()
        .map(|d| d.join("cusparseLt.h"))
        .collect();
        candidates.into_iter().find(|p| p.exists())
    }

    fn cuda_include_dirs() -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Some(p) = std::env::var_os("CUDA_PATH") {
            out.push(PathBuf::from(p).join("include"));
        }
        for c in [
            "/usr/local/cuda/include",
            "/usr/local/cuda-12.9/include",
            "/usr/local/cuda-12.8/include",
            "/usr/include",
        ] {
            let p = PathBuf::from(c);
            if p.exists() {
                out.push(p);
            }
        }
        out
    }
}
