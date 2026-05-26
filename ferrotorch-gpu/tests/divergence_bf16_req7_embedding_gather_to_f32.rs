//! Divergence: `.design/ferrotorch-gpu/bf16.md` REQ-7 claims SHIPPED for
//! the transformer-primitive bundle (`bf16.md:263`), explicitly listing
//! `gpu_embedding_gather_bf16_to_f32` and asserting "each is called from
//! the bf16 arm of `backend_impl.rs`'s embedding/attention dispatchers".
//!
//! Reality of `gpu_embedding_gather_bf16_to_f32`:
//!   - Defined: `ferrotorch-gpu/src/bf16.rs:840` (pub fn).
//!   - Re-exported: `ferrotorch-gpu/src/lib.rs:195`.
//!   - Non-test consumers anywhere in the workspace: ZERO.
//!   - Never called from `backend_impl.rs` (grep returns 0 hits).
//!   - Never called from `ferrotorch-llama/src/gpu.rs` (the production
//!     bf16 consumer that DOES wire the other six REQ-7 primitives).
//!
//! Other REQ-7 primitives have an honest non-test consumer in
//! `ferrotorch-llama/src/gpu.rs` (rope_half / transpose_to_heads /
//! transpose_from_heads / repeat_kv / causal_mask / embedding_gather);
//! the `_to_f32` variant alone is vocab-only.
//!
//! Per goal.md vocab-only discipline, this must be either consumed in
//! production or demoted from REQ-7's SHIPPED status. The design doc's
//! `backend_impl.rs`-only citation is also wrong for the OTHER six
//! primitives (their real consumer is `ferrotorch-llama/src/gpu.rs`),
//! but that is a citation defect, not a missing-consumer defect, so
//! this test pins only the missing-consumer case.
//!
//! Tracking: blocker filed via crosslink (see audit report).

use std::fs;
use std::path::{Path, PathBuf};

fn collect_rs_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
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
        if path_str.ends_with("ferrotorch-gpu/src/bf16.rs") {
            continue;
        }
        if path_str.ends_with("ferrotorch-gpu/src/lib.rs") {
            continue;
        }
        if path_str.contains("/tests/") || path_str.contains("/benches/") {
            continue;
        }
        if path_str.contains("/examples/") {
            continue;
        }

        let Ok(content) = fs::read_to_string(&file) else {
            continue;
        };

        let production_chunk = content
            .split("#[cfg(test)]")
            .next()
            .unwrap_or(&content);

        let bytes = production_chunk.as_bytes();
        let sym_bytes = symbol.as_bytes();
        let mut i = 0;
        while i + sym_bytes.len() <= bytes.len() {
            if &bytes[i..i + sym_bytes.len()] == sym_bytes {
                let before_ok = i == 0
                    || !(bytes[i - 1].is_ascii_alphanumeric()
                        || bytes[i - 1] == b'_');
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
fn divergence_gpu_embedding_gather_bf16_to_f32_has_no_production_consumer() {
    // bf16.md REQ-7 (line 263) claims `gpu_embedding_gather_bf16_to_f32`
    // is consumed by `backend_impl.rs`'s embedding/attention dispatch.
    // Workspace-wide grep finds ZERO non-test, non-definition,
    // non-re-export callers. Vocab-only.
    let callers = find_non_test_callers("gpu_embedding_gather_bf16_to_f32");
    assert!(
        !callers.is_empty(),
        "bf16.md REQ-7 claims `gpu_embedding_gather_bf16_to_f32` has a \
         non-test consumer in `backend_impl.rs`'s embedding/attention \
         dispatch arm, but workspace-wide grep finds ZERO production \
         callers. The symbol is vocab-only. Either wire a consumer \
         (the obvious candidate is `ferrotorch-llama/src/gpu.rs`'s \
         embedding step, which currently uses the bf16->bf16 variant) \
         or demote REQ-7 from SHIPPED with a consumer-wiring blocker. \
         Note: the OTHER six REQ-7 primitives (rope_half, \
         transpose_to/from_heads, repeat_kv, causal_mask, \
         embedding_gather) DO have a non-test consumer in \
         `ferrotorch-llama/src/gpu.rs`, but bf16.md's citation of \
         `backend_impl.rs` for all seven is wrong — backend_impl has \
         zero hits for any of them."
    );
}
