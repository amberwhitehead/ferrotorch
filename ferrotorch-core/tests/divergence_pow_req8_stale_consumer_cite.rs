//! Divergence test for #1193 audit (REQ-8 design-doc citation).
//!
//! The closing commit 2debdcf9e moved REQ-8 (pow) NOT-STARTED -> SHIPPED in
//! `.design/ferrotorch-core/grad_fns/arithmetic.md:470`. The SHIPPED row cites
//! three "non-test production consumer" sites:
//!
//!   1. `ferrotorch-core/src/methods.rs:35` (Tensor::pow_t)               -- valid
//!   2. `ferrotorch-core/src/autograd/grad_penalty.rs:111,118`            -- valid
//!   3. `ferrotorch-core/src/autograd/graph.rs:876`                       -- STALE
//!
//! The third citation points inside a `#[cfg(test)] mod tests` block
//! (`graph.rs:652` opens the test module; line 876 is inside
//! `test_backward_one_element_through_pow_and_add` at `graph.rs:870`).
//!
//! Per goal.md R-HONEST-2 every SHIPPED REQ citation must be backed by a
//! NON-TEST production consumer. Test-only callers don't count. The design doc
//! row presents `graph.rs:876` as if it were an additional production consumer
//! alongside `methods.rs:35` and `grad_penalty.rs:111,118` -- it is not.
//!
//! Why this test exists at all: REQ-8 SHIPPED still survives on the strength
//! of the methods.rs + grad_penalty.rs cites alone. But citing a test-block
//! line as a "non-test consumer" is the same shape of citation-theater the
//! goal.md anti-drift rules (R-HONEST-2) name as forbidden. Pin it.
//!
//! Tracking: blocker filed via crosslink; on fix, the design-doc row drops
//! the `graph.rs:876` cite (or replaces it with a real non-test caller).

use std::fs;

/// Returns the 0-indexed start-of-mod-tests line in graph.rs, or None if not
/// found. Used to determine whether a referenced line is inside `#[cfg(test)]`.
fn find_cfg_test_mod_start(source: &str) -> Option<usize> {
    let mut prev_was_cfg = false;
    for (i, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if prev_was_cfg && trimmed.starts_with("mod tests") {
            return Some(i + 1); // 1-indexed
        }
        prev_was_cfg = trimmed == "#[cfg(test)]";
    }
    None
}

#[test]
fn divergence_pow_req8_graph_consumer_cite_is_in_test_block() {
    // The design doc claims `graph.rs:876` is a non-test production consumer
    // of `arithmetic::pow`. The actual file places that line inside the test
    // module. This test fails until the doc cite is corrected (or the line
    // is moved out of tests into a real production consumer).

    let graph_src = fs::read_to_string("../ferrotorch-core/src/autograd/graph.rs")
        .or_else(|_| fs::read_to_string("ferrotorch-core/src/autograd/graph.rs"))
        .expect("must locate graph.rs from cargo test cwd (workspace root or crate root)");

    let mod_tests_line = find_cfg_test_mod_start(&graph_src)
        .expect("expected a `#[cfg(test)]\\nmod tests` block in graph.rs");

    let cited_line: usize = 876;

    // The cite is valid only if the cited line precedes the test module.
    assert!(
        cited_line < mod_tests_line,
        "design-doc REQ-8 cite at graph.rs:{cited_line} is INSIDE `#[cfg(test)] mod tests` \
         (test mod starts at line {mod_tests_line}). Per goal.md R-HONEST-2, a SHIPPED-REQ \
         citation must be a NON-TEST production consumer. The cite is citation-theater and \
         must be removed from `.design/ferrotorch-core/grad_fns/arithmetic.md:470` (or \
         replaced with a real non-test caller of `arithmetic::pow`)."
    );
}
