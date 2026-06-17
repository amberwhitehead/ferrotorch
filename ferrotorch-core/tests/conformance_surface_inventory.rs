//! Conformance Phase 2.0 — surface inventory generator.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/759>.
//!
//! Walks `src/lib.rs` and every `mod` it transitively declares, parses each
//! file with `syn`, and emits a sorted JSON inventory of every `pub` item
//! to `tests/conformance/_surface.json`. The committed JSON file is the
//! contract; PRs that change the public surface show up as JSON diffs and
//! the strict coverage gate (`conformance_surface_coverage.rs`) fails if a
//! new item is not referenced by a conformance test or excluded.
//!
//! The walker lives in `tests/common/surface_inventory.rs` and is shared with
//! the coverage gate, so coverage recomputes the current source surface
//! inside its own test binary instead of trusting this producer to have run
//! first.

use std::fs;

#[path = "common/surface_inventory.rs"]
mod surface_inventory;

use surface_inventory::{collect_surface_items, out_path, render_json};

#[test]
fn surface_inventory_writes_json() {
    let unique = collect_surface_items();
    let json = render_json(&unique);
    fs::create_dir_all(out_path().parent().expect("conformance dir")).expect("mkdir conformance");
    fs::write(out_path(), &json).expect("write _surface.json");

    // Sanity: the inventory must contain the 21 creation-module functions
    // covered by Phase 2.0. If any are missing, the walker is broken or
    // the source genuinely lost a `pub` declaration.
    let must_contain = [
        "ferrotorch_core::creation::arange",
        "ferrotorch_core::creation::eye",
        "ferrotorch_core::creation::from_slice",
        "ferrotorch_core::creation::from_vec",
        "ferrotorch_core::creation::full",
        "ferrotorch_core::creation::full_like",
        "ferrotorch_core::creation::full_meta",
        "ferrotorch_core::creation::linspace",
        "ferrotorch_core::creation::meta_like",
        "ferrotorch_core::creation::ones",
        "ferrotorch_core::creation::ones_like",
        "ferrotorch_core::creation::ones_meta",
        "ferrotorch_core::creation::rand",
        "ferrotorch_core::creation::rand_like",
        "ferrotorch_core::creation::randn",
        "ferrotorch_core::creation::randn_like",
        "ferrotorch_core::creation::scalar",
        "ferrotorch_core::creation::tensor",
        "ferrotorch_core::creation::zeros",
        "ferrotorch_core::creation::zeros_like",
        "ferrotorch_core::creation::zeros_meta",
    ];
    let paths: Vec<&str> = unique.iter().map(|i| i.path.as_str()).collect();
    let mut missing: Vec<&str> = Vec::new();
    for needle in must_contain {
        if !paths.contains(&needle) {
            missing.push(needle);
        }
    }
    assert!(
        missing.is_empty(),
        "surface inventory missing {} expected creation items: {missing:?}",
        missing.len()
    );

    // Sanity: ferrotorch-core has 71+ source files; the surface should be
    // substantially larger than ferrotorch-tokenize's ~30 items. A walker
    // that returns <200 items has almost certainly lost a module branch.
    assert!(
        unique.len() >= 200,
        "surface inventory unexpectedly small ({} items); expected hundreds. \
         The module walker likely failed to descend into one of the 71 source \
         files in ferrotorch-core/src/.",
        unique.len()
    );
}
