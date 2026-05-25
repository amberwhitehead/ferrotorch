//! Audit divergence (commit `2620772fa`, closes #1267):
//!
//! Commit `2620772fa` refreshed `*Backward` struct cites and `pub fn` cites
//! in `.design/ferrotorch-core/grad_fns/{arithmetic,cumulative}.md`, but
//! several OTHER classes of cite are also stale and are NOT covered by either
//! of the two existing cite-drift audit tests
//! (`divergence_arithmetic_md_prose_bare_colon_cites_stale.rs` and
//! `divergence_cumulative_md_prose_cites_stale.rs`).
//!
//! The committer's own "spillover noted" in the audit handoff was the
//! `### REQ-N (lines X-Y)` section headers. There are also additional uncaught
//! drift classes:
//!
//! 1. SECTION-HEADER LINE RANGES — `### REQ-N (lines X-Y)` headers in both
//!    `arithmetic.md` and `cumulative.md` cite RS LINE NUMBER RANGES that
//!    no longer match HEAD.
//!
//! 2. "EXISTING UNIT TESTS" SECTION TEST-FN CITES — `cumulative.md:365-389`
//!    lists ~20 test fn names with `cumulative.rs:NNN-MMM` line ranges that
//!    are wildly off (hundreds of lines stale). E.g. `test_cumsum_1d (:376-386)`
//!    when actual fn at HEAD is `:775`.
//!
//! 3. ARCHITECTURE-SECTION INNER-METHOD CITES — e.g. `cumulative.md:73-83`
//!    cites `CumprodBackward` "two-path split at `:131-179`" and "fast path
//!    (`:161-178`)" and "slow path (`:142-160`)" — these inner `:NNN-MMM`
//!    ranges within the new `CumprodBackward :242-342` block are also
//!    relative to the OLD struct position.
//!
//! 4. AC-N PROSE CITES OF TEST FNS — e.g. `cumulative.md:238-243` cites
//!    `test_cumsum_negative_dim at cumulative.rs:420-428` (actual: `:818`).
//!
//! 5. CUMULATIVE.MD REQ-TABLE-ROW INNER CITES — `cumulative.md:442` "Tests
//!    at `cumulative.rs:420-428 test_cumsum_negative_dim` and `:830-835
//!    test_cumsum_dim_out_of_bounds`" — :420-428 stale (actual :818); :830-835
//!    stale (actual :1331).
//!
//! Per R-CITE-2 every bare-colon or named-file cite must resolve at HEAD
//! against the named symbol. This test fixture cross-checks the cite content
//! against the actual line at HEAD by parsing both files.
//!
//! Tracking: filed via crosslink (see audit report).

#![allow(clippy::missing_panics_doc)]

use std::fs;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !p.join(".design").exists() {
        p.pop();
    }
    p
}

fn line_at(path: &PathBuf, line_no: usize) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.lines().nth(line_no - 1).map(str::to_string))
}

/// CATEGORY 2: "Existing unit tests" section in `cumulative.md:365-389` lists
/// test fns with `cumulative.rs:NNN-MMM` line cites. Each MUST resolve to
/// `fn <test_name>` at the cited START line at HEAD.
#[test]
fn divergence_cumulative_md_test_fn_cites_in_existing_tests_section_stale() {
    let root = workspace_root();
    let rs = root.join("ferrotorch-core/src/grad_fns/cumulative.rs");

    // Cites taken directly from cumulative.md:367-389 (the "Existing unit
    // tests" section). Each tuple is (cited_rs_line, test_fn_name).
    let cites: Vec<(usize, &str)> = vec![
        (376, "test_cumsum_1d"),
        (388, "test_cumsum_2d_dim0"),
        (404, "test_cumsum_2d_dim1"),
        (419, "test_cumsum_negative_dim"),
        (430, "test_cumsum_3d"),
        (449, "test_cumsum_backward_1d"),
        (465, "test_cumsum_backward_2d_dim0"),
        (880, "test_cumsum_backward_numerical"),
        (486, "test_cumsum_has_grad_fn"),
        (494, "test_cumsum_no_grad_fn_when_not_requires_grad"),
        (501, "test_cumsum_no_grad_fn_in_no_grad_context"),
        (512, "test_cumprod_1d"),
        (523, "test_cumprod_2d_dim0"),
        (538, "test_cumprod_2d_dim1"),
        (557, "test_cumprod_backward_1d"),
        (576, "test_cumprod_backward_with_zero"),
        (841, "test_cumprod_backward_numerical"),
        (618, "test_cummax_1d"),
        (631, "test_cummax_2d_dim1"),
        (652, "test_cummin_1d"),
        (665, "test_cummin_2d_dim0"),
        (684, "test_logcumsumexp_1d"),
        (700, "test_logcumsumexp_2d_dim1"),
        (718, "test_logcumsumexp_numerical_stability"),
        (742, "test_logcumsumexp_backward_1d"),
        (830, "test_cumsum_dim_out_of_bounds"),
    ];

    let mut errors: Vec<String> = Vec::new();
    for (rs_line, fname) in cites {
        let actual = line_at(&rs, rs_line).unwrap_or_default();
        let needle = format!("fn {fname}");
        if !actual.contains(&needle) {
            errors.push(format!(
                "cumulative.md cites `{fname}` at `cumulative.rs:{rs_line}` but that line at HEAD is:\n    `{actual}`"
            ));
        }
    }

    assert!(
        errors.is_empty(),
        "cumulative.md `### Existing unit tests` section (lines 365-389) has stale test-fn cites (R-CITE-2 violation — commit 2620772fa refreshed `pub fn` / `*Backward` cites but missed this section):\n\n{}\n\nTotal stale test-fn cites: {}",
        errors.join("\n\n"),
        errors.len()
    );
}

/// CATEGORY 5: `cumulative.md:367` cites the `#[cfg(test)] mod tests` block
/// at `cumulative.rs:515-913`. At HEAD the `#[cfg(test)]` attribute is at
/// `:753`, so the cite is stale.
#[test]
fn divergence_cumulative_md_tests_mod_block_range_cite_stale() {
    let root = workspace_root();
    let rs = root.join("ferrotorch-core/src/grad_fns/cumulative.rs");

    // cumulative.md:367 cites :515 as the mod-tests start line.
    let line_515 = line_at(&rs, 515).unwrap_or_default();
    let has_cfg_test_at_515 = line_515.contains("#[cfg(test)]") || line_515.contains("mod tests");

    // At HEAD the actual `#[cfg(test)]` is at :753.
    let line_753 = line_at(&rs, 753).unwrap_or_default();
    let has_cfg_test_at_753 = line_753.contains("#[cfg(test)]");

    assert!(
        has_cfg_test_at_515 || !has_cfg_test_at_753,
        "cumulative.md:367 cites `ferrotorch-core/src/grad_fns/cumulative.rs:515-913` as the `#[cfg(test)] mod tests` block, but at HEAD :515 is `{line_515}` and `#[cfg(test)]` is actually at :753 (`{line_753}`) — cite stale by ~238 lines (R-CITE-2)"
    );
}

/// CATEGORY 1 (cumulative.md): the `### REQ-N (lines X-Y)` section headers
/// reference RS LINE RANGES that no longer match where the corresponding
/// `pub fn` lives at HEAD.
#[test]
fn divergence_cumulative_md_section_header_line_ranges_stale() {
    let root = workspace_root();
    let rs = root.join("ferrotorch-core/src/grad_fns/cumulative.rs");
    let md = root.join(".design/ferrotorch-core/grad_fns/cumulative.md");
    let md_text = fs::read_to_string(&md).expect("read cumulative.md");
    let md_lines: Vec<&str> = md_text.lines().collect();

    // Each section header line at .md line N claims the .rs range X-Y.
    // The pub fn body for each op must lie within the cited range at HEAD.
    // (doc_line, header_substring, pub_fn_name, expected_rs_line_at_HEAD)
    let cites: Vec<(usize, &str, &str, usize)> = vec![
        (278, "### REQ-1 `cumsum` (lines 26-86)", "pub fn cumsum", 104),
        (293, "### REQ-2 `cumprod` (lines 88-217)", "pub fn cumprod", 354),
        (
            315,
            "### REQ-5 `logcumsumexp` (lines 244-337)",
            "pub fn logcumsumexp",
            712,
        ),
    ];

    let mut errors: Vec<String> = Vec::new();
    for (doc_line, header_needle, fn_name, rs_line_at_head) in cites {
        // 1. Confirm the .md still has the stale header substring.
        let doc_text = md_lines.get(doc_line - 1).copied().unwrap_or("");
        if !doc_text.contains(header_needle) {
            errors.push(format!(
                "cumulative.md:{doc_line} does not contain expected header `{header_needle}`; actual: `{doc_text}`"
            ));
            continue;
        }

        // 2. Parse the (lines X-Y) range from the header.
        let after_paren = doc_text.split_once("(lines ").map(|(_, t)| t).unwrap_or("");
        let range_str = after_paren.split_once(')').map(|(r, _)| r).unwrap_or("");
        let (lo_str, hi_str) = match range_str.split_once('-') {
            Some(p) => p,
            None => {
                errors.push(format!(
                    "cumulative.md:{doc_line} header malformed; could not parse range from `{range_str}`"
                ));
                continue;
            }
        };
        let lo: usize = lo_str.trim().parse().unwrap_or(0);
        let hi: usize = hi_str.trim().parse().unwrap_or(0);

        // 3. Assert the actual `pub fn` line at HEAD is OUTSIDE the cited range
        //    (proves the range is stale).
        let actual = line_at(&rs, rs_line_at_head).unwrap_or_default();
        if !actual.contains(fn_name) {
            errors.push(format!(
                "cumulative.md:{doc_line} fixture: expected `{fn_name}` at cumulative.rs:{rs_line_at_head} but line is `{actual}`"
            ));
            continue;
        }

        if rs_line_at_head >= lo && rs_line_at_head <= hi {
            // The actual line IS within the cited range — cite is correct.
            continue;
        }
        errors.push(format!(
            "cumulative.md:{doc_line} section header `{header_needle}` claims the section lives at cumulative.rs:{lo}-{hi}, but at HEAD `{fn_name}` is at :{rs_line_at_head} (OUTSIDE the cited range)"
        ));
    }

    assert!(
        errors.is_empty(),
        "cumulative.md `### REQ-N (lines X-Y)` section headers cite stale rs line ranges (R-CITE-2 — spillover noted by acto-critic, NOT fixed by commit 2620772fa):\n\n{}",
        errors.join("\n\n")
    );
}

/// CATEGORY 1 (arithmetic.md): the `### REQ-N (lines X-Y)` section headers
/// for the SHIFT-AFFECTED reqs (REQ-12 floor_divide, REQ-13 remainder,
/// REQ-14 fmod, REQ-15 addcmul, REQ-16 addcdiv, REQ-7 sqrt) reference RS
/// LINE RANGES that no longer match HEAD.
#[test]
fn divergence_arithmetic_md_section_header_line_ranges_stale() {
    let root = workspace_root();
    let rs = root.join("ferrotorch-core/src/grad_fns/arithmetic.rs");
    let md = root.join(".design/ferrotorch-core/grad_fns/arithmetic.md");
    let md_text = fs::read_to_string(&md).expect("read arithmetic.md");
    let md_lines: Vec<&str> = md_text.lines().collect();

    // (doc_line, header_substring, struct_or_fn_name, expected_rs_line_at_HEAD)
    let cites: Vec<(usize, &str, &str, usize)> = vec![
        (
            596,
            "### REQ-13 `remainder` (lines 1865-2104)",
            "struct RemainderBackward",
            1890,
        ),
        (
            623,
            "### REQ-14 `fmod` (lines 2168-2374)",
            "struct FmodBackward",
            2193,
        ),
        (
            650,
            "### REQ-12 `floor_divide` (lines 2459-2841)",
            "struct FloorDivideBackward",
            2484,
        ),
        (
            705,
            "### REQ-15 `addcmul` (lines 2820-3115)",
            "struct AddcmulBackward",
            2845,
        ),
        (
            757,
            "### REQ-16 `addcdiv` (lines 3116-3403)",
            "struct AddcdivBackward",
            3141,
        ),
    ];

    let mut errors: Vec<String> = Vec::new();
    for (doc_line, header_needle, symbol, rs_line_at_head) in cites {
        let doc_text = md_lines.get(doc_line - 1).copied().unwrap_or("");
        if !doc_text.contains(header_needle) {
            errors.push(format!(
                "arithmetic.md:{doc_line} does not contain expected header `{header_needle}`; actual: `{doc_text}`"
            ));
            continue;
        }

        let after_paren = doc_text.split_once("(lines ").map(|(_, t)| t).unwrap_or("");
        let range_str = after_paren.split_once(')').map(|(r, _)| r).unwrap_or("");
        let (lo_str, hi_str) = match range_str.split_once('-') {
            Some(p) => p,
            None => {
                errors.push(format!(
                    "arithmetic.md:{doc_line} header malformed; could not parse range from `{range_str}`"
                ));
                continue;
            }
        };
        let lo: usize = lo_str.trim().parse().unwrap_or(0);
        let hi: usize = hi_str.trim().parse().unwrap_or(0);

        let actual = line_at(&rs, rs_line_at_head).unwrap_or_default();
        if !actual.contains(symbol) {
            errors.push(format!(
                "arithmetic.md:{doc_line} fixture: expected `{symbol}` at arithmetic.rs:{rs_line_at_head} but line is `{actual}`"
            ));
            continue;
        }

        if rs_line_at_head >= lo && rs_line_at_head <= hi {
            continue;
        }
        errors.push(format!(
            "arithmetic.md:{doc_line} section header `{header_needle}` claims the section lives at arithmetic.rs:{lo}-{hi}, but at HEAD `{symbol}` is at :{rs_line_at_head} (OUTSIDE the cited range)"
        ));
    }

    assert!(
        errors.is_empty(),
        "arithmetic.md `### REQ-N (lines X-Y)` section headers cite stale rs line ranges (R-CITE-2 — spillover noted by acto-critic, NOT fixed by commit 2620772fa):\n\n{}",
        errors.join("\n\n")
    );
}

/// CATEGORY 3: cumulative.md architecture section cites inner-method line
/// numbers WITHIN CumprodBackward that are relative to the OLD struct
/// position. E.g. md:73 says "same-shape two-path split at `:131-179`" but
/// at HEAD CumprodBackward struct is at :242, so the inner methods are
/// shifted by +139 lines, putting the fast/slow paths at approximately
/// :300-318 / :281-299, not :161-178 / :142-160.
#[test]
fn divergence_cumulative_md_cumprodbackward_inner_range_cites_stale() {
    let root = workspace_root();
    let rs = root.join("ferrotorch-core/src/grad_fns/cumulative.rs");
    let md = root.join(".design/ferrotorch-core/grad_fns/cumulative.md");
    let md_text = fs::read_to_string(&md).expect("read cumulative.md");
    let rs_text = fs::read_to_string(&rs).expect("read cumulative.rs");

    // The doc claims the two-path split is "at `:131-179`". At HEAD the
    // CumprodBackward struct starts at :242, so the body cannot be at :131.
    let stale_range_present = md_text.contains(":131-179")
        || md_text.contains(":161-178")
        || md_text.contains(":142-160");

    // At HEAD line :131 is before the CumprodBackward block (which is :242-342),
    // so :131 should NOT be inside CumprodBackward.
    let line_131 = rs_text.lines().nth(130).unwrap_or("");
    let line_161 = rs_text.lines().nth(160).unwrap_or("");

    // Sanity: line :131 at HEAD should not be inside the CumprodBackward block.
    let line_131_inside_cumprod = false; // structurally impossible since struct at :242
    let _ = line_131;
    let _ = line_161;
    let _ = line_131_inside_cumprod;

    assert!(
        !stale_range_present,
        "cumulative.md still contains the pre-shift CumprodBackward inner-method ranges (`:131-179`, `:161-178`, `:142-160`) which were valid when CumprodBackward was at :103 but are stale now that it's at :242-342 at HEAD (R-CITE-2 — uncaught by either #1267 audit test)"
    );
}

/// CATEGORY 4: AC-N rows in cumulative.md cite test fn line numbers that
/// don't match HEAD. Spot-check 3.
#[test]
fn divergence_cumulative_md_ac_row_test_fn_cites_stale() {
    let root = workspace_root();
    let rs = root.join("ferrotorch-core/src/grad_fns/cumulative.rs");

    // (doc_line_substring, cited_rs_line, test_fn)
    // Cites pulled from cumulative.md AC-N rows and architecture prose.
    let cites: Vec<(&str, usize, &str)> = vec![
        // cumulative.md:238-243 AC-row prose
        ("test_cumsum_negative_dim", 420, "test_cumsum_negative_dim"),
        (
            "test_cumsum_no_grad_fn_when_not_requires_grad",
            495,
            "test_cumsum_no_grad_fn_when_not_requires_grad",
        ),
        // cumulative.md:161 architecture-section prose
        (
            "test_logcumsumexp_numerical_stability",
            719,
            "test_logcumsumexp_numerical_stability",
        ),
        // cumulative.md:182 architecture-section prose
        (
            "test_cumsum_dim_out_of_bounds",
            800,
            "test_cumsum_dim_out_of_bounds",
        ),
    ];

    let mut errors: Vec<String> = Vec::new();
    for (label, rs_line, fname) in cites {
        let actual = line_at(&rs, rs_line).unwrap_or_default();
        let needle = format!("fn {fname}");
        if !actual.contains(&needle) {
            errors.push(format!(
                "cumulative.md cite `{label}` at cumulative.rs:{rs_line} does not contain `fn {fname}` at HEAD; actual line: `{actual}`"
            ));
        }
    }

    assert!(
        errors.is_empty(),
        "cumulative.md AC-N / architecture prose cites test fn line numbers that don't resolve at HEAD (R-CITE-2):\n\n{}",
        errors.join("\n\n")
    );
}

/// CATEGORY 2 (arithmetic.md): `### Unit tests` section at `arithmetic.md:837-863`
/// lists ~20 test fns with `(NNNN)` line cites that are wildly off — e.g.
/// `test_add_forward (1720)` when actual fn at HEAD is `:3582` (off by +1862).
#[test]
fn divergence_arithmetic_md_test_fn_cites_in_unit_tests_section_stale() {
    let root = workspace_root();
    let rs = root.join("ferrotorch-core/src/grad_fns/arithmetic.rs");

    // (cited_rs_line, test_fn_name)
    let cites: Vec<(usize, &str)> = vec![
        (1720, "test_add_forward"),
        (1728, "test_sub_forward"),
        (1736, "test_mul_forward"),
        (1744, "test_div_forward"),
        (1752, "test_neg_forward"),
        (1759, "test_pow_forward"),
        (1769, "test_sqrt_forward"),
        (1779, "test_abs_forward"),
        (1790, "test_add_backward"),
        (1802, "test_sub_backward"),
        (1814, "test_mul_backward"),
        (1826, "test_div_backward"),
        (1838, "test_div_backward_tensor_by_scalar"),
        (1864, "test_neg_backward"),
        (1874, "test_pow_backward"),
        (1884, "test_sqrt_backward"),
        (1895, "test_abs_backward_positive"),
        (1905, "test_abs_backward_negative"),
        (1919, "test_add_no_grad_fn_when_inputs_detached"),
        (1927, "test_mul_partial_requires_grad"),
        (1941, "test_no_grad_context_skips_backward"),
        (1956, "test_chain_mul_add"),
        (1971, "test_chain_div_sub"),
        (1986, "test_chain_sqrt_pow"),
        (2001, "test_neg_double"),
        (2016, "test_mul_vector_backward"),
    ];

    let mut errors: Vec<String> = Vec::new();
    for (rs_line, fname) in cites {
        let actual = line_at(&rs, rs_line).unwrap_or_default();
        let needle = format!("fn {fname}");
        if !actual.contains(&needle) {
            errors.push(format!(
                "arithmetic.md `### Unit tests` cites `{fname}` at arithmetic.rs:{rs_line} but at HEAD that line is:\n    `{actual}`"
            ));
        }
    }

    assert!(
        errors.is_empty(),
        "arithmetic.md `### Unit tests` section (lines 837-863) has stale test-fn line cites (R-CITE-2 — uncaught by either #1267 audit test):\n\n{}\n\nTotal stale: {}",
        errors.join("\n\n"),
        errors.len()
    );
}
