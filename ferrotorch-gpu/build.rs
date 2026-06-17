//! Build script for `ferrotorch-gpu`.
//!
//! Two responsibilities, both opt-in / no-op when their preconditions
//! are absent:
//!
//! 1. When the `cusparselt` feature is enabled, instruct cargo to link
//!    against `libcusparseLt.so`. The cuSPARSELt FFI is hand-authored Rust
//!    in `src/cusparselt.rs`; the build no longer runs bindgen, parses
//!    `cusparseLt.h`, or requires a C/C++ toolchain.
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
//! Probe order for the cuSPARSELt shared library:
//!   1. `$CUSPARSELT_LIB_DIR`
//!   2. `/usr/local/cuda*/lib64`, `/usr/lib64`,
//!      `/opt/nvidia/cusparselt/lib64`
//!   3. Python NVIDIA package installs such as
//!      `$HOME/.local/lib/python*/site-packages/nvidia/cusparselt/lib`
//!
//! NVIDIA distributes cuSPARSELt as a separate SDK from the CUDA toolkit
//! (it ships in its own tarball / RPM and in PyTorch's NVIDIA wheel
//! dependencies). A `cusparselt` build still needs the shared library for
//! linking/runtime, but it does not need the header.

fn main() {
    // The script runs unconditionally — but every action below is gated
    // on the relevant `CARGO_FEATURE_*` env var, which cargo sets only
    // when that feature is active. Re-run if a gate flips or any probed
    // env var changes.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUSPARSELT");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
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

/// Resolve the CUDA version cudarc will build against and report whether it is
/// `>= min` using cudarc's `MAJOR<MINOR:02d><PATCH:02d>` encoding, e.g. `13020`
/// for CUDA 13.0.20.
///
/// The workspace supplies a default `CUDARC_CUDA_VERSION` in `.cargo/config.toml`.
/// Keep this build script independent of `nvcc` or any CUDA C/C++ toolchain: if
/// the env var is absent or malformed, fail closed and do not emit CUDA-version
/// cfgs by accident.
fn cuda_version_at_least(min: u32) -> bool {
    if let Ok(v) = std::env::var("CUDARC_CUDA_VERSION")
        && let Ok(n) = v.trim().parse::<u32>()
    {
        return n >= min;
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
    use std::path::PathBuf;

    /// Library probe + link directives.
    pub fn generate() {
        emit_link_directives();
    }

    fn emit_link_directives() {
        let mut dirs = Vec::new();
        if let Ok(dir) = std::env::var("CUSPARSELT_LIB_DIR") {
            dirs.push(PathBuf::from(dir));
        }
        for candidate in [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda-12.9/lib64",
            "/usr/local/cuda-12.8/lib64",
            "/usr/lib64",
            "/opt/nvidia/cusparselt/lib64",
        ] {
            dirs.push(PathBuf::from(candidate));
        }
        dirs.extend(python_cusparselt_dirs("lib"));

        let mut saw_existing_dir = false;
        for dir in dirs {
            if !dir.exists() {
                continue;
            }
            saw_existing_dir = true;
            println!("cargo:rustc-link-search=native={}", dir.display());
            let bare = dir.join("libcusparseLt.so");
            if bare.exists() {
                println!("cargo:rustc-link-lib=cusparseLt");
                return;
            }
            let versioned = dir.join("libcusparseLt.so.0");
            if versioned.exists() {
                if let Some(link_dir) = emit_cusparselt_compat_symlink(&versioned) {
                    println!("cargo:rustc-link-search=native={}", link_dir.display());
                    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", link_dir.display());
                }
                println!("cargo:rustc-link-lib=cusparseLt");
                return;
            }
        }

        if !saw_existing_dir {
            println!(
                "cargo:warning=cuSPARSELt library directory was not found; relying on the system linker path for libcusparseLt"
            );
        }
        println!("cargo:rustc-link-lib=cusparseLt");
    }

    fn emit_cusparselt_compat_symlink(versioned: &PathBuf) -> Option<PathBuf> {
        let out_dir = PathBuf::from(std::env::var_os("OUT_DIR")?);
        let compat_dir = out_dir.join("cusparselt-compat");
        if let Err(e) = std::fs::create_dir_all(&compat_dir) {
            println!(
                "cargo:warning=cuSPARSELt library {} exists, but build.rs failed to create compat dir {}: {e}",
                versioned.display(),
                compat_dir.display()
            );
            return None;
        }

        #[cfg(unix)]
        {
            for name in ["libcusparseLt.so", "libcusparseLt.so.0"] {
                let link = compat_dir.join(name);
                let _ = std::fs::remove_file(&link);
                if let Err(e) = std::os::unix::fs::symlink(versioned, &link) {
                    println!(
                        "cargo:warning=cuSPARSELt library {} exists, but build.rs failed to symlink {}: {e}",
                        versioned.display(),
                        link.display()
                    );
                    return None;
                }
            }
            println!("cargo:rerun-if-changed={}", versioned.display());
            Some(compat_dir)
        }
        #[cfg(not(unix))]
        {
            println!(
                "cargo:warning=cuSPARSELt library {} exists, but this build script can only create a versioned-library compat symlink on Unix hosts",
                versioned.display()
            );
            None
        }
    }

    fn python_cusparselt_dirs(kind: &str) -> impl Iterator<Item = PathBuf> {
        let mut roots = Vec::new();
        if let Some(home) = std::env::var_os("HOME") {
            roots.push(PathBuf::from(home).join(".local/lib"));
        }
        roots.push(PathBuf::from("/usr/local/lib"));
        roots.push(PathBuf::from("/usr/lib"));

        let mut out = Vec::new();
        for root in roots {
            let Ok(entries) = std::fs::read_dir(root) else {
                continue;
            };
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if !name.starts_with("python") {
                    continue;
                }
                let site_packages = entry.path().join("site-packages");
                out.push(site_packages.join("nvidia/cusparselt").join(kind));
                let dist_packages = entry.path().join("dist-packages");
                out.push(dist_packages.join("nvidia/cusparselt").join(kind));
            }
        }
        out.into_iter().filter(|p| p.exists())
    }
}
