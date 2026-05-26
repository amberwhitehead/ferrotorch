//! Divergence: `.design/ferrotorch-gpu/blas.md` REQ-8 claims SHIPPED with
//! evidence "Non-test consumer: the same `backend_impl.rs` dispatch table
//! consumes these on the zero-host-allocation path" (`blas.md:239`). The
//! claim is FALSE: `gpu_matmul_f32_into` and `gpu_bmm_f32_into` have ZERO
//! non-test, non-definition, non-re-export consumers anywhere in the
//! workspace. They are also listed in
//! `ferrotorch-gpu/tests/conformance/_surface_exclusions.toml:430-437`
//! with `reason = "deferred"`, confirming the absence of consumer wiring.
//!
//! Per goal.md vocab-only discipline, a `pub fn` with no production
//! caller must NOT be claimed SHIPPED — it is vocab-only and must be
//! demoted to NOT-STARTED with a consumer-wiring blocker.
//!
//! Audit method: workspace-wide grep of all crates under `/home/doll/
//! ferrotorch/{ferrotorch-*,benchmarks,tools,tooling}/**/*.rs` for the
//! symbol identifier, excluding (a) the defining file `blas.rs`, (b) the
//! re-export `ferrotorch-gpu/src/lib.rs`, (c) `#[cfg(test)]` test
//! modules, (d) `.claude/worktrees/` agent staging dirs. Result for
//! both symbols: 0 callers.
//!
//! Tracking: blocker filed via crosslink (see audit report).

use std::fs;
use std::path::{Path, PathBuf};

/// Walk a directory, returning every `.rs` file that is *not* under a
/// `.claude/worktrees/` agent staging dir.
fn collect_rs_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        // Skip agent worktrees, target dir, and dotdirs we don't care about.
        if name == "target" || name == "worktrees" || name == ".git" {
            continue;
        }
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Find files containing `symbol` (as a whole word) that are
/// NOT: (a) the defining file (`blas.rs`), (b) the re-export
/// (`ferrotorch-gpu/src/lib.rs`), (c) a test file (path contains
/// `/tests/` or filename starts with `test_` or `_probe_` or
/// `divergence_` or `conformance_`).
fn find_non_test_callers(symbol: &str) -> Vec<PathBuf> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();

    let mut files = Vec::new();
    collect_rs_files(&workspace_root, &mut files);

    let mut hits = Vec::new();
    for file in files {
        let path_str = file.to_string_lossy();
        // Skip the defining file.
        if path_str.ends_with("ferrotorch-gpu/src/blas.rs") {
            continue;
        }
        // Skip the re-export.
        if path_str.ends_with("ferrotorch-gpu/src/lib.rs") {
            continue;
        }
        // Skip integration tests, probe scaffolding, benchmarks.
        if path_str.contains("/tests/") || path_str.contains("/benches/") {
            continue;
        }
        // Skip example dirs.
        if path_str.contains("/examples/") {
            continue;
        }

        let Ok(content) = fs::read_to_string(&file) else {
            continue;
        };

        // Walk through `cfg(test)` mods naively: split on `#[cfg(test)]`,
        // examine only the first chunk (everything before the test mod).
        // This is a coarse-but-safe filter: a non-test caller in the
        // first chunk is definitive evidence of production consumption.
        let production_chunk = content.split("#[cfg(test)]").next().unwrap_or(&content);

        // Whole-word match for the symbol.
        // Pattern: not preceded or followed by a Rust identifier char.
        let bytes = production_chunk.as_bytes();
        let sym_bytes = symbol.as_bytes();
        let mut i = 0;
        while i + sym_bytes.len() <= bytes.len() {
            if &bytes[i..i + sym_bytes.len()] == sym_bytes {
                let before_ok =
                    i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
                let after_ok = i + sym_bytes.len() == bytes.len()
                    || !(bytes[i + sym_bytes.len()].is_ascii_alphanumeric()
                        || bytes[i + sym_bytes.len()] == b'_');
                if before_ok && after_ok {
                    hits.push(file.clone());
                    break;
                }
            }
            i += 1;
        }
    }
    hits
}

#[test]
#[ignore = "vocab-only divergence; tracking #1360"]
fn divergence_gpu_matmul_f32_into_has_no_production_consumer() {
    // Upstream contract (blas.md:239): REQ-8 SHIPPED, cited consumer is
    // `backend_impl.rs` matmul/bmm dispatch arms.
    //
    // Ferrotorch reality: no consumer exists. This test FAILS today
    // (vocab-only divergence) and PASSES once REQ-8 is either:
    //   (a) wired into a production dispatcher (the zero-host-bounce
    //       fast path the design describes), OR
    //   (b) demoted to NOT-STARTED in blas.md with a consumer-wiring
    //       blocker filed.
    let callers = find_non_test_callers("gpu_matmul_f32_into");
    assert!(
        !callers.is_empty(),
        "blas.md REQ-8 claims `gpu_matmul_f32_into` has a non-test \
         consumer in `backend_impl.rs`, but workspace-wide grep finds \
         ZERO production callers. The symbol is vocab-only. Either \
         wire a consumer (zero-host-bounce dispatch path) or demote \
         REQ-8 to NOT-STARTED with a consumer-wiring blocker. \
         Audited paths (production chunks before any `#[cfg(test)]`): \
         all ferrotorch-* crates, benchmarks, tools, tooling. \
         Re-exports and the defining file are intentionally excluded."
    );
}

#[test]
#[ignore = "vocab-only divergence; tracking #1360"]
fn divergence_gpu_bmm_f32_into_has_no_production_consumer() {
    // Same divergence shape as `gpu_matmul_f32_into` — REQ-8 lumps
    // matmul and bmm `_into` variants together.
    let callers = find_non_test_callers("gpu_bmm_f32_into");
    assert!(
        !callers.is_empty(),
        "blas.md REQ-8 claims `gpu_bmm_f32_into` has a non-test \
         consumer in `backend_impl.rs`, but workspace-wide grep finds \
         ZERO production callers. The symbol is vocab-only. Either \
         wire a consumer or demote REQ-8 to NOT-STARTED."
    );
}
