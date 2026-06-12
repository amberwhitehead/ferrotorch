//! Audit of `f44e70391` (closes #1268) — the new generic cite-drift test
//! `divergence_cite_drift_generic.rs` claims to be a structural durability
//! contract that catches future drift in arithmetic.md / cumulative.md +
//! their `//!` doc-comments. The commit message states:
//!
//!   "The test surface is now ITSELF the durable contract; future cite
//!    drift in arithmetic.md / cumulative.md / arithmetic.rs //! /
//!    cumulative.rs //! will fail this test BEFORE landing."
//!
//! and:
//!
//!   "Hand-rolled scanner ... Symbol-hint validation gated to `test_*` fns
//!    (where the cite SHOULD point at the fn declaration); skipped for
//!    `*Backward` and helper hints since those cites often point INSIDE
//!    the symbol's body."
//!
//! This audit pins THREE structural-coverage gaps that the generic test
//! does not catch, and ONE concrete refresh-miss that escaped:
//!
//!   GAP A (#1269): a single `*Backward` cite whose line number is moved to
//!   a clearly-wrong line (e.g. one that declares a different `pub fn`) is
//!   NOT caught — symbol-hint validation is hard-coded to skip non-`test_*`
//!   hints. The deleted test
//!   `divergence_arithmetic_md_prose_bare_colon_cites_stale.rs` DID catch
//!   this category by hard-coding the (struct_name, expected_rs_line)
//!   tuples; the generic test claims to subsume that coverage but does not.
//!
//!   GAP B (#1269): a +1 or +2 line shift in arithmetic.rs (e.g. inserting
//!   a single `use` import at the top) is NOT caught — symbol-hint
//!   validation uses a +/-3 line window AND only runs for `test_*` cites,
//!   so a one-line shift slides every cite by 1 but every cite still lands
//!   on a substantive line near the right symbol. The commit message
//!   advertises "durable contract" but the contract only triggers at +3
//!   lines of shift or larger, and only for cites that happen to land on
//!   blank/brace lines.
//!
//!   GAP C (#1269): a typo'd file path (e.g. `arithmatic.rs:1565` or
//!   `gradfns/arithmetic.rs:1565`) is silently skipped — `resolve_cite_path`
//!   returns None for unresolvable paths and `validate_cite` treats None as
//!   success. A doc reviewer mistyping a path produces a cite that
//!   superficially looks resolved but is never actually checked.
//!
//!   REFRESH-MISS (#1270): cumulative.md:446 (REQ-6 status table) still
//!   cites `cumulative.rs:420-428 test_cumsum_negative_dim` (actual fn at
//!   :818; :420-428 is inside `impl GradFn for CummaxBackward`) and
//!   `:830-835 test_cumsum_dim_out_of_bounds` (actual at :1331; :830-835 is
//!   inside a different test). cumulative.md:447 (REQ-7 status table) still
//!   cites `cumulative.rs:449-484` (actual `test_cumsum_backward_*` at
//!   :848+; :449-484 is inside `LogcumsumexpBackward` / helper fn) and
//!   `:742-779` (actual `test_logcumsumexp_backward_1d` at :1146; :742-779
//!   is `fn dim_strides`). The generic test passes because the cited ranges
//!   contain substantive lines and have no `test_*` symbol-hint immediately
//!   preceding the backtick.
//!
//! Per goal.md R-CITE-2 every cite must resolve at HEAD; R-CHAR-3 — every
//! expected value is computed from the actual file contents at test time.

#![allow(clippy::missing_panics_doc)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

/// Serializes the probes that MUTATE shared `.design`/`.rs` files (arithmetic.md,
/// arithmetic.rs) and spawn `cargo test` subprocesses. Without this, the
/// default multi-threaded test harness can run two such probes concurrently —
/// one rewriting arithmetic.md while the other reads it through a subprocess —
/// producing flaky failures. The lock is a process-wide critical section; each
/// mutating probe restores the file before releasing it, so the on-disk state
/// is always restored between probes.
static MUTATING_PROBE_LOCK: Mutex<()> = Mutex::new(());

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !p.join(".design").exists() {
        p.pop();
    }
    p
}

/// REFRESH-MISS: cumulative.md:446-447 still contains stale ranges. We
/// directly verify the .md says `:420-428` / `:830-835` / `:449-484` /
/// `:742-779` and that those rs ranges do NOT contain `fn <expected_test>`.
#[test]
fn divergence_1270_cumulative_md_req_table_rows_446_447_stale_test_cites() {
    let root = workspace_root();
    let md_path = root.join(".design/ferrotorch-core/grad_fns/cumulative.md");
    let rs_path = root.join("ferrotorch-core/src/grad_fns/cumulative.rs");
    let md = fs::read_to_string(&md_path).unwrap();
    let rs = fs::read_to_string(&rs_path).unwrap();
    let rs_lines: Vec<&str> = rs.lines().collect();

    // Expected stale cite -> test function name the doc claims is at that range.
    // (cite_substring_in_md, expected_test_fn_name, md_line_for_msg)
    let cases: &[(&str, &str, usize)] = &[
        // cumulative.md:446 (REQ-6 row)
        (
            "cumulative.rs:420-428 test_cumsum_negative_dim",
            "test_cumsum_negative_dim",
            446,
        ),
        (
            ":830-835 test_cumsum_dim_out_of_bounds",
            "test_cumsum_dim_out_of_bounds",
            446,
        ),
        // cumulative.md:447 (REQ-7 row)
        ("cumulative.rs:449-484", "test_cumsum_backward", 447),
        (":742-779", "test_logcumsumexp_backward", 447),
    ];

    let mut still_stale: Vec<String> = Vec::new();
    for (cite_substr, expected_fn, _md_line) in cases {
        // 1. Confirm the stale cite is still literally present in cumulative.md.
        if !md.contains(cite_substr) {
            // Good — refresh happened. Skip this case.
            continue;
        }
        // 2. Parse the line range from the cite substring (`<lo>-<hi>` or
        // `<lo>` after the last `:`).
        let after_colon = cite_substr.rsplit(':').next().unwrap();
        let nums: String = after_colon
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        let (lo_str, hi_str) = match nums.split_once('-') {
            Some((a, b)) => (a, b),
            None => (nums.as_str(), nums.as_str()),
        };
        let lo: usize = lo_str.parse().unwrap_or(0);
        let hi: usize = hi_str.parse().unwrap_or(lo);
        // 3. Check whether ANY line in [lo..=hi] contains `fn <expected_fn>`.
        let needle = format!("fn {expected_fn}");
        let mut any_hit = false;
        for i in lo..=hi {
            if let Some(line) = rs_lines.get(i.saturating_sub(1))
                && line.contains(&needle)
            {
                any_hit = true;
                break;
            }
        }
        if !any_hit {
            // Find where the fn actually is, for the error message.
            let mut actual_line: Option<usize> = None;
            for (i, line) in rs_lines.iter().enumerate() {
                if line.contains(&needle) {
                    actual_line = Some(i + 1);
                    break;
                }
            }
            still_stale.push(format!(
                "cumulative.md still cites `{cite_substr}` but range :{lo}-{hi} in cumulative.rs does NOT contain `{needle}` (actual fn at line {actual:?})",
                actual = actual_line,
            ));
        }
    }
    assert!(
        still_stale.is_empty(),
        "REFRESH-MISS: f44e70391 left {} cite(s) stale in cumulative.md REQ-6/REQ-7 table rows that the new generic audit does NOT catch (R-CITE-2):\n\n{}",
        still_stale.len(),
        still_stale.join("\n\n")
    );
}

/// GAP A — NOW CLOSED (#1643), CONVERTED TO A POSITIVE PROBE.
///
/// History: the original gap-A probe (closed-source above this commit)
/// demonstrated that the generic walker did NOT catch a `*Backward` cite
/// drifting to a wrong line — symbol-hint validation skipped non-`test_*`
/// hints. The #1633 S3 conversion then replaced every `*Backward:NNN` LINE
/// cite with a line-number-FREE symbol anchor (`` `RsqrtBackward` struct in
/// `grad_fns/arithmetic.rs` ``), so there was no `*Backward:NNN` cite left to
/// corrupt and the probe's premise went stale (it was `#[ignore]`'d tracking
/// #1643).
///
/// #1643 added a struct-symbol-anchor parser + validator to the walker
/// (`parse_struct_anchors` / `validate_struct_anchor`) and a scoped contract
/// test `all_design_docs_s3_struct_anchors_resolve_at_head`. This probe now
/// PROVES that new check works end-to-end through the real test binary:
///   1. Confirm arithmetic.md carries the real S3 anchor
///      `` `RsqrtBackward` struct in `grad_fns/arithmetic.rs` ``.
///   2. Corrupt the FILE half to a file that does not declare RsqrtBackward
///      (`grad_fns/cumulative.rs`) — the AliasTable-style "right struct,
///      wrong file" drift the enhancement is built to catch.
///   3. Run the scoped `all_design_docs_s3_struct_anchors_resolve_at_head`
///      test as a subprocess and assert it now FAILS (the corruption is
///      caught). The scoped test is green at HEAD, so its exit status cleanly
///      isolates THIS corruption (unlike the broader
///      `all_design_docs_cites_resolve_at_head` walker, which carries
///      pre-existing unrelated line-number-cite drift).
///   4. Restore arithmetic.md.
#[test]
fn divergence_1643_walker_catches_corrupted_s3_struct_anchor() {
    let _guard = MUTATING_PROBE_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let root = workspace_root();
    let md_path = root.join(".design/ferrotorch-core/grad_fns/arithmetic.md");
    let rs_arith = root.join("ferrotorch-core/src/grad_fns/arithmetic.rs");
    let rs_cumul = root.join("ferrotorch-core/src/grad_fns/cumulative.rs");

    // Sanity on the structural facts the corruption relies on: RsqrtBackward
    // IS declared in arithmetic.rs and is NOT declared in cumulative.rs.
    let arith = fs::read_to_string(&rs_arith).unwrap();
    let cumul = fs::read_to_string(&rs_cumul).unwrap();
    assert!(
        arith.contains("struct RsqrtBackward"),
        "probe assumption broken: arithmetic.rs no longer declares struct RsqrtBackward at HEAD",
    );
    assert!(
        !cumul.contains("struct RsqrtBackward"),
        "probe assumption broken: cumulative.rs unexpectedly declares struct RsqrtBackward — pick a different decoy file",
    );

    let original_md = fs::read_to_string(&md_path).unwrap();
    let real_anchor = "`RsqrtBackward` struct in `grad_fns/arithmetic.rs`";
    assert!(
        original_md.contains(real_anchor),
        "probe assumption broken: arithmetic.md does not contain the S3 anchor `{real_anchor}` at HEAD",
    );

    // Corrupt the FILE half: re-point the anchor at a file that does NOT
    // declare RsqrtBackward.
    let corrupted = original_md.replacen(
        real_anchor,
        "`RsqrtBackward` struct in `grad_fns/cumulative.rs`",
        1,
    );
    assert_ne!(corrupted, original_md, "edit had no effect");

    // Write, run the SCOPED struct-anchor test, restore.
    fs::write(&md_path, &corrupted).unwrap();
    let result = Command::new("cargo")
        .args([
            "test",
            "-p",
            "ferrotorch-core",
            "--test",
            "divergence_cite_drift_generic",
            "--",
            "all_design_docs_s3_struct_anchors_resolve_at_head",
            "--exact",
        ])
        .current_dir(&root)
        .output();
    fs::write(&md_path, &original_md).unwrap();
    let output = result.expect("cargo test invocation failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}\n{}", String::from_utf8_lossy(&output.stderr));

    // The scoped test MUST now fail — proving the new S3 struct-anchor check
    // catches a right-struct/wrong-file corruption (gap A's S3-era successor).
    assert!(
        !output.status.success(),
        "GAP A SUCCESSOR REGRESSED (#1643): after re-pointing the real `RsqrtBackward` struct anchor in arithmetic.md from `grad_fns/arithmetic.rs` to `grad_fns/cumulative.rs` (which does NOT declare RsqrtBackward), the scoped struct-anchor test `all_design_docs_s3_struct_anchors_resolve_at_head` STILL PASSED. The S3 line-number-free symbol-anchor contract is not actually catching moved/renamed structs.\n\nScoped-test output:\n{combined}",
    );
    assert!(
        combined.contains("STALE") || combined.contains("RsqrtBackward"),
        "scoped test failed but the failure does not mention the corrupted RsqrtBackward anchor — it may have failed for an unrelated reason:\n{combined}",
    );
}

/// GAP B: a +1 line shift in arithmetic.rs is NOT caught. We DEMONSTRATE:
///   1. Back up arithmetic.rs.
///   2. Prepend a single comment line, shifting every line below by +1.
///   3. Run the generic test; assert it STILL PASSES.
///   4. Restore arithmetic.rs.
///
/// IGNORED under #1952: this probe is red-by-design until the generic
/// gate validates symbols on ALL cites (not just test_* hints with a
/// +/-3 window). It previously appeared green only because a leaked
/// sentinel line from this very probe was COMMITTED at the top of
/// arithmetic.rs, misaligning every cite by +1 — the gate then "caught"
/// the probe's second shift by accident. That residue (and the
/// self-mutation race that leaks it under parallel test runs) is
/// exactly #1952's scope; the gate-strengthening retires this ignore.
#[test]
#[ignore = "#1952: generic gate misses +1 shifts; probe also self-mutates under races — retire when the gate validates symbols on all cites"]
fn divergence_1269_gap_b_generic_test_misses_plus_one_line_shift_in_rs() {
    let _guard = MUTATING_PROBE_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let root = workspace_root();
    let rs_path = root.join("ferrotorch-core/src/grad_fns/arithmetic.rs");
    let original_rs = fs::read_to_string(&rs_path).unwrap();
    let corrupted_rs = format!("// DIVERGENCE-1269-GAP-B-PROBE-PLUS-ONE\n{original_rs}");

    fs::write(&rs_path, &corrupted_rs).unwrap();
    let result = Command::new("cargo")
        .args([
            "test",
            "-p",
            "ferrotorch-core",
            "--test",
            "divergence_cite_drift_generic",
            "--",
            "all_design_docs_cites_resolve_at_head",
            "--exact",
        ])
        .current_dir(&root)
        .output();
    fs::write(&rs_path, &original_rs).unwrap();
    let output = result.expect("cargo test invocation failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}\n{}", String::from_utf8_lossy(&output.stderr));

    let generic_passed = output.status.success();
    assert!(
        !generic_passed,
        "GAP B confirmed: the generic cite-drift test PASSED after prepending a single line to arithmetic.rs (every cite below now off by +1). The commit message advertises 'durable contract; future cite drift ... will fail this test BEFORE landing' but a +1 shift is the most basic form of drift (e.g. adding an import). Symbol-hint validation uses a +/-3 line window AND only runs for `test_*` cites, so the contract is silent on small shifts.\n\nGeneric-test output:\n{combined}",
    );
}

/// GAP C: a typo'd file path (`arithmatic.rs:1565` or
/// `gradfns/arithmetic.rs:1565`) is silently skipped.
#[test]
fn divergence_1269_gap_c_generic_test_silently_skips_typo_filepaths() {
    let _guard = MUTATING_PROBE_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let root = workspace_root();
    let md_path = root.join(".design/ferrotorch-core/grad_fns/arithmetic.md");
    let original_md = fs::read_to_string(&md_path).unwrap();
    // Append a synthetic section with two clear typos. Neither resolves.
    let typo_section = "\n\n## DIVERGENCE-1269-GAP-C-PROBE\n\n\
        - typo'd basename: `arithmatic.rs:1565`\n\
        - typo'd directory: `gradfns/arithmetic.rs:1565`\n\
        - typo'd extension: `arithmetic.rss:1565`\n";
    let corrupted = format!("{original_md}{typo_section}");

    fs::write(&md_path, &corrupted).unwrap();
    let result = Command::new("cargo")
        .args([
            "test",
            "-p",
            "ferrotorch-core",
            "--test",
            "divergence_cite_drift_generic",
            "--",
            "all_design_docs_cites_resolve_at_head",
            "--exact",
        ])
        .current_dir(&root)
        .output();
    fs::write(&md_path, &original_md).unwrap();
    let output = result.expect("cargo test invocation failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}\n{}", String::from_utf8_lossy(&output.stderr));

    let generic_passed = output.status.success();
    assert!(
        !generic_passed,
        "GAP C confirmed: the generic cite-drift test PASSED with three typo'd file paths in arithmetic.md (`arithmatic.rs`, `gradfns/arithmetic.rs`, `arithmetic.rss`). `resolve_cite_path` returns None for unresolvable basenames and `validate_cite` treats None as success — typos are invisible. R-CITE-2 violation: a reviewer can introduce a cite that looks like it's resolved but is never checked.\n\nGeneric-test output:\n{combined}",
    );
}
