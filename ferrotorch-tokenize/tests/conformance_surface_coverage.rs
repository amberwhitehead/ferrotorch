//! Conformance Phase 1 — strict coverage gate.
//!
//! Tracking issue: <https://github.com/<owner>/ferrotorch/issues/758>.
//!
//! Loads the surface inventory (`_surface.json`) produced by
//! `conformance_surface_inventory.rs` and scans the conformance test files
//! for references to each `pub` item. Fails CI if any inventory item is
//! neither referenced nor explicitly excluded in `_surface_exclusions.toml`.
//!
//! This is the project's signal: "we added a public API without proving
//! its contract." Adding new `pub` items without a corresponding
//! conformance test breaks the build by design.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Surface {
    items: Vec<SurfaceItem>,
}

#[derive(Debug, Deserialize)]
struct SurfaceItem {
    path: String,
    kind: String,
    #[allow(
        dead_code,
        reason = "field deserialized for forward-compat with future filters"
    )]
    signature: String,
}

#[derive(Debug, Deserialize)]
struct ExclusionsFile {
    #[serde(default, rename = "exclusion")]
    exclusions: Vec<Exclusion>,
}

#[derive(Debug, Deserialize)]
struct Exclusion {
    path: String,
    reason: String,
}

fn conformance_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
}

fn tests_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn read_surface() -> Surface {
    let p = conformance_dir().join("_surface.json");
    let bytes = fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. Run `cargo test -p ferrotorch-tokenize --test \
             conformance_surface_inventory` first to regenerate it.",
            p.display()
        )
    });
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn read_exclusions() -> Vec<Exclusion> {
    let p = conformance_dir().join("_surface_exclusions.toml");
    if !p.exists() {
        return Vec::new();
    }
    let body = fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let parsed: ExclusionsFile =
        toml::from_str(&body).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()));
    parsed.exclusions
}

/// Scan the conformance test directory for each `.rs` file and collect
/// their full text. The coverage check is text-grep based: an item is
/// "covered" iff its short identifier appears anywhere in any conformance
/// test source. Substring grep is intentional — we don't want to demand a
/// specific call shape (the tests sometimes reference a type via `use`,
/// sometimes via a method call, sometimes via a `Debug` print).
fn read_conformance_test_sources() -> String {
    let mut combined = String::new();
    let root = tests_dir();
    let entries =
        fs::read_dir(&root).unwrap_or_else(|e| panic!("read tests dir {}: {e}", root.display()));
    for entry in entries {
        let entry = entry.expect("readdir entry");
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        // We only count *conformance* test sources towards coverage. A
        // generic integration test that happens to mention `encode` is
        // not the same as a conformance proof. The inventory test itself
        // (which only exists to write _surface.json) is also excluded;
        // it would otherwise vacuously cover any item it lists.
        if !name.starts_with("conformance_") {
            continue;
        }
        if name == "conformance_surface_inventory.rs" || name == "conformance_surface_coverage.rs" {
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let body =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        combined.push_str(&body);
        combined.push('\n');
    }
    combined
}

/// Extract the short identifier from a fully-qualified path.
///
/// `ferrotorch_tokenize::ChatMessage::new` -> `new`
/// `ferrotorch_tokenize::encode` -> `encode`
/// `ferrotorch_tokenize::ChatMessage` -> `ChatMessage`
fn short_ident(path: &str) -> &str {
    path.rsplit("::").next().unwrap_or(path)
}

/// For methods, also need the *type* name to be referenced somewhere
/// (otherwise `new` is too generic and matches anything). Returns the
/// `Type::method` segment for methods, the bare ident otherwise.
fn coverage_keys(path: &str) -> Vec<String> {
    let segs: Vec<&str> = path.split("::").collect();
    if segs.len() >= 3
        && segs[segs.len() - 2]
            .chars()
            .next()
            .is_some_and(char::is_uppercase)
    {
        // Likely a method on a type (`...::Foo::bar`). Require both the
        // type name and the method name to appear (in either order, on
        // any line) — typical call sites read `Foo::bar(...)` or
        // `Foo::new(...)`.
        let ty = segs[segs.len() - 2];
        let m = segs[segs.len() - 1];
        vec![format!("{ty}::{m}")]
    } else {
        // Free function / type / re-export: short ident is enough.
        vec![short_ident(path).to_string()]
    }
}

#[test]
fn every_public_item_has_a_conformance_reference() {
    let surface = read_surface();
    let exclusions = read_exclusions();
    let exclusion_set: BTreeMap<String, String> =
        exclusions.into_iter().map(|e| (e.path, e.reason)).collect();

    let test_sources = read_conformance_test_sources();
    assert!(
        !test_sources.is_empty(),
        "no conformance test source files found (expected files matching \
         tests/conformance_*.rs other than the inventory + coverage gates). \
         Add at least one conformance_<topic>.rs file."
    );

    let mut covered: Vec<&str> = Vec::new();
    let mut excluded: Vec<(&str, &str)> = Vec::new();
    let mut uncovered: Vec<&SurfaceItem> = Vec::new();

    for item in &surface.items {
        // Glob re-exports (`pub use foo::*`) are never auto-coverable;
        // require an explicit exclusion entry. The inventory writer
        // stores them with `path` ending in `::*`.
        if item.path.ends_with("::*") {
            if let Some(reason) = exclusion_set.get(&item.path) {
                excluded.push((item.path.as_str(), reason.as_str()));
            } else {
                uncovered.push(item);
            }
            continue;
        }

        if let Some(reason) = exclusion_set.get(&item.path) {
            excluded.push((item.path.as_str(), reason.as_str()));
            continue;
        }

        let keys = coverage_keys(&item.path);
        let referenced = keys.iter().any(|k| test_sources.contains(k.as_str()));
        if referenced {
            covered.push(item.path.as_str());
        } else {
            uncovered.push(item);
        }
    }

    // Stable, alphabetized report regardless of pass/fail.
    eprintln!("--- conformance surface coverage ---");
    eprintln!(
        "covered {}/{} (excluded: {})",
        covered.len(),
        surface.items.len(),
        excluded.len()
    );
    for path in &covered {
        eprintln!("  COVERED   {path}");
    }
    for (path, reason) in &excluded {
        eprintln!("  EXCLUDED  {path}  — {reason}");
    }
    for item in &uncovered {
        eprintln!(
            "  UNCOVERED {}  (kind={}; {})",
            item.path,
            item.kind,
            short_signature(&item.signature)
        );
    }

    assert!(
        uncovered.is_empty(),
        "{} ferrotorch-tokenize public item(s) lack a conformance reference. \
         Either author a test in tests/conformance_*.rs that references the \
         item by name, OR add it to tests/conformance/_surface_exclusions.toml \
         with a written reason. Uncovered: {:?}",
        uncovered.len(),
        uncovered
            .iter()
            .map(|i| i.path.as_str())
            .collect::<Vec<_>>()
    );

    // Stale-exclusion guard: an exclusion for an item that no longer
    // exists in the surface is suspect (probably the item was renamed or
    // removed and the exclusion was forgotten).
    let surface_paths: std::collections::BTreeSet<&str> =
        surface.items.iter().map(|i| i.path.as_str()).collect();
    let stale: Vec<&str> = exclusion_set
        .keys()
        .filter(|k| !surface_paths.contains(k.as_str()))
        .map(String::as_str)
        .collect();
    assert!(
        stale.is_empty(),
        "_surface_exclusions.toml lists items that no longer exist in the \
         surface inventory (stale entries — remove or update): {stale:?}"
    );
}

fn short_signature(sig: &str) -> &str {
    // Cap the printed signature so the failure message stays readable.
    // Note: this is a byte-index slice, fine because `_surface.json`
    // signatures are ASCII (Rust syntax tokens). Truncate at a code-point
    // boundary defensively in case a future inventory entry contains
    // non-ASCII (e.g. an embedded string literal).
    if sig.len() <= 80 {
        return sig;
    }
    let mut end = 80;
    while end > 0 && !sig.is_char_boundary(end) {
        end -= 1;
    }
    &sig[..end]
}
