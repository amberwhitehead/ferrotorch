//! Systemic guard (#1686): every hand-written PTX kernel const in the crate
//! MUST actually JIT-compile on the live GPU.
//!
//! Motivation: #1684 (Philox) and #1685 (complex transpose) shipped kernels
//! that NEVER ran on real hardware — a `.reg .u32 %tid` shadowed the builtin
//! special register, so `cusolverDnXgeev`/the kernel failed with
//! CUDA_ERROR_INVALID_PTX. The Philox path silently CPU-fell-back (correctness
//! tests couldn't tell), and the transpose path hard-errored but had no GPU
//! test. Both were invisible for a long time.
//!
//! This test walks every `*_PTX` string constant in `src/`, reconstructs the
//! string-literal body, and JIT-compiles it on the device via
//! `module_cache::get_or_compile_owned` — exactly the driver path production
//! uses. A kernel that fails JIT (non-ASCII, register-name shadow, bad
//! instruction, wrong `.target`, …) makes this test fail loudly.
//!
//! SCOPE / KNOWN LIMITATION: this covers STATIC `&str` PTX consts only. It does
//! NOT cover (a) PTX generated at runtime by `get_f64_ptx` (the #1685 f64
//! stride bug lived there — guarded instead by the per-dtype kernel tests), nor
//! (b) silent CPU-fallback where a kernel compiles but is never dispatched.
//! Closing those fully needs per-entry-point on-device exercise; tracked
//! separately. This guard is the cheap, self-maintaining first line.
#![cfg(feature = "cuda")]

use std::fs;
use std::path::Path;

use ferrotorch_gpu::GpuDevice;

/// Parse a Rust string-literal body starting at `bytes[start]` (the byte just
/// after the opening `"`). Returns (decoded, index_after_closing_quote).
///
/// The PTX consts use only two escape forms: `\n` (newline) and a trailing
/// backslash-newline line continuation (`"\<NL>...\n\<NL>..."`). We also handle
/// `\t`, `\"`, `\\` defensively.
fn parse_string_literal(bytes: &[u8], start: usize) -> Option<(String, usize)> {
    let mut out = String::new();
    let mut i = start;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\\' {
            let n = *bytes.get(i + 1)?;
            match n {
                b'n' => {
                    out.push('\n');
                    i += 2;
                }
                b't' => {
                    out.push('\t');
                    i += 2;
                }
                b'"' => {
                    out.push('"');
                    i += 2;
                }
                b'\\' => {
                    out.push('\\');
                    i += 2;
                }
                b'\n' => {
                    // line continuation: skip backslash, newline, leading ws
                    i += 2;
                    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
                        i += 1;
                    }
                }
                _ => {
                    out.push(n as char);
                    i += 2;
                }
            }
        } else if c == b'"' {
            return Some((out, i + 1));
        } else {
            // PTX is ASCII; push raw byte
            out.push(c as char);
            i += 1;
        }
    }
    None
}

/// Extract `(const_name, entry_name, ptx_body)` for every `*_PTX` string const
/// that defines a full module (contains `.entry`).
fn extract_ptx_consts(src: &str) -> Vec<(String, String, String)> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut search = 0usize;
    while let Some(rel) = src[search..].find("_PTX") {
        let name_end = search + rel; // index of '_' in "_PTX" ... we want the const name
        // Walk back to the start of the identifier for a friendlier label.
        let mut s = name_end + 4; // just after "_PTX"
        // Find the assignment's opening quote: `... = "`
        // Look ahead a bounded window for `= "`.
        let window_end = (s + 80).min(bytes.len());
        if let Some(eqrel) = bytes[s..window_end].iter().position(|&b| b == b'=') {
            let after_eq = s + eqrel + 1;
            // find the first `"` after `=`
            let quote_window_end = (after_eq + 40).min(bytes.len());
            if let Some(qrel) = bytes[after_eq..quote_window_end]
                .iter()
                .position(|&b| b == b'"')
            {
                let q = after_eq + qrel;
                if let Some((body, end)) = parse_string_literal(bytes, q + 1) {
                    if body.contains(".entry") {
                        let entry = body
                            .split(".entry")
                            .nth(1)
                            .and_then(|rest| rest.split('(').next())
                            .map(|n| n.trim().to_string())
                            .unwrap_or_default();
                        // const name: scan back from name_end to identifier start
                        let cn_start = src[..name_end]
                            .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
                            .map(|p| p + 1)
                            .unwrap_or(0);
                        let const_name = src[cn_start..name_end + 4].to_string();
                        if !entry.is_empty() {
                            out.push((const_name, entry, body));
                        }
                    }
                    s = end;
                }
            }
        }
        search = s.max(name_end + 4);
    }
    out
}

#[test]
fn every_static_ptx_const_jit_compiles() {
    let dev = GpuDevice::new(0).expect("cuda device 0");
    let ctx = dev.context();

    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut total = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut entries = fs::read_dir(&src_dir)
        .expect("read src dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "rs").unwrap_or(false))
        .collect::<Vec<_>>();
    entries.sort();

    for path in entries {
        let src = fs::read_to_string(&path).expect("read source");
        for (const_name, entry, body) in extract_ptx_consts(&src) {
            total += 1;
            let res =
                ferrotorch_gpu::module_cache::get_or_compile_owned(ctx, body, entry.clone(), 0);
            if let Err(e) = res {
                failures.push(format!(
                    "{}::{const_name} (entry `{entry}`): {e:?}",
                    path.file_name().unwrap().to_string_lossy()
                ));
            }
        }
    }

    assert!(
        total > 50,
        "extractor found only {total} PTX consts — parser likely broke"
    );
    eprintln!("ptx_compile_guard: JIT-validated {total} static PTX modules on the live GPU");
    assert!(
        failures.is_empty(),
        "{} PTX kernel(s) failed to JIT-compile on the live GPU (silent-fallback / hard-error class, cf. #1684/#1685):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
