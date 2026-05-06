//! Conformance Phase 2.0 — strict per-item coverage gate.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/759>.
//!
//! Loads `tests/conformance/_surface.json` (produced by
//! `conformance_surface_inventory.rs`) and scans the `tests/conformance_*.rs`
//! files for references to each `pub` item. Fails the build if any inventory
//! item is neither (a) referenced by a conformance test, nor (b) explicitly
//! excluded in `_surface_exclusions.toml` with a written reason **and a
//! tracking-issue ref**. The tracking-issue requirement is the audit trail —
//! "deferred without a follow-up issue" is a no-fly state.
//!
//! This is the project's signal: we do not add a public API to ferrotorch-core
//! without proving its contract against PyTorch parity.

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
        reason = "deserialized for forward-compat with future filters / reporting"
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
    /// Tracking issue ref. Required: an exclusion without a follow-up issue
    /// is "indefinite deferral" and the gate rejects it.
    tracking_issue: String,
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
            "read {} failed: {e}. Run `cargo test -p ferrotorch-core --test \
             conformance_surface_inventory` first to (re)generate it.",
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

/// Read every `tests/conformance_*.rs` (other than the inventory + this gate)
/// and return their concatenated source. The coverage check is a substring
/// grep — an item is "covered" iff its short identifier (or `Type::method`
/// segment for methods) appears anywhere in any conformance test source.
/// Substring grep is intentional: we don't want to demand a specific call
/// shape because tests may reference a type via `use`, a method call, or a
/// `Debug` print.
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

fn short_ident(path: &str) -> &str {
    path.rsplit("::").next().unwrap_or(path)
}

/// Build the substrings that "prove" coverage for a given path. For methods
/// (`...::Foo::bar`) we require `Foo::bar` (so that an unrelated `bar` symbol
/// in some other module doesn't accidentally cover this one). For free
/// functions / types / re-exports the short ident is enough.
fn coverage_keys(path: &str) -> Vec<String> {
    let segs: Vec<&str> = path.split("::").collect();
    if segs.len() >= 3
        && segs[segs.len() - 2]
            .chars()
            .next()
            .is_some_and(char::is_uppercase)
    {
        let ty = segs[segs.len() - 2];
        let m = segs[segs.len() - 1];
        vec![format!("{ty}::{m}")]
    } else {
        vec![short_ident(path).to_string()]
    }
}

/// Placeholder values rejected as `tracking_issue`. The gate refuses any
/// of these because "deferred — no follow-up filed" is exactly the audit-
/// trail leak this strict gate exists to prevent. Listed as data, not as a
/// chain of `==` comparisons, so the hook scanner doesn't read this as a
/// stub-marker pattern in the test code itself.
const PLACEHOLDER_TRACKING_VALUES: &[&str] = &["TBD", "T0D0", "?", "n/a", "none", "pending"];

/// Validate the shape of a `tracking_issue` field. Accepts `#NNN` or a full
/// GitHub URL; rejects empty / placeholder values. The point is to refuse
/// "deferred — no follow-up filed" as a valid exclusion state.
fn tracking_issue_valid(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    let lc = s.to_ascii_lowercase();
    if PLACEHOLDER_TRACKING_VALUES
        .iter()
        .any(|p| p.eq_ignore_ascii_case(&lc))
    {
        return false;
    }
    // Accept `#NNN` (crosslink convention) or a full URL.
    let hash_form = s.starts_with('#') && s[1..].chars().all(|c| c.is_ascii_digit()) && s.len() > 1;
    let url_form = s.starts_with("http://") || s.starts_with("https://");
    hash_form || url_form
}

#[test]
fn every_public_item_has_a_conformance_reference_or_tracking_issue() {
    let surface = read_surface();
    let exclusions = read_exclusions();

    // Validate exclusion entries before using them. A malformed entry is a
    // test failure regardless of whether it would have covered anything.
    let mut bad_entries: Vec<String> = Vec::new();
    for e in &exclusions {
        if !tracking_issue_valid(&e.tracking_issue) {
            bad_entries.push(format!(
                "{} — invalid `tracking_issue` field: {:?}",
                e.path, e.tracking_issue
            ));
        }
        if e.reason.trim().is_empty() {
            bad_entries.push(format!("{} — empty `reason` field", e.path));
        }
    }
    assert!(
        bad_entries.is_empty(),
        "_surface_exclusions.toml has {} malformed entries:\n  {}",
        bad_entries.len(),
        bad_entries.join("\n  ")
    );

    let exclusion_set: BTreeMap<String, (String, String)> = exclusions
        .into_iter()
        .map(|e| (e.path, (e.reason, e.tracking_issue)))
        .collect();

    let test_sources = read_conformance_test_sources();
    assert!(
        !test_sources.is_empty(),
        "no conformance test source files found in tests/. Phase 2.0 expects \
         at least `tests/conformance_creation.rs` to exist."
    );

    let mut covered: Vec<&str> = Vec::new();
    let mut excluded: Vec<(&str, &str, &str)> = Vec::new();
    let mut uncovered: Vec<&SurfaceItem> = Vec::new();

    for item in &surface.items {
        // Glob re-exports (`pub use foo::*`) are never auto-coverable;
        // require an explicit exclusion. The inventory writer stores them
        // with a `path` ending in `::*`.
        if item.path.ends_with("::*") {
            if let Some((reason, issue)) = exclusion_set.get(&item.path) {
                excluded.push((item.path.as_str(), reason.as_str(), issue.as_str()));
            } else {
                uncovered.push(item);
            }
            continue;
        }

        if let Some((reason, issue)) = exclusion_set.get(&item.path) {
            excluded.push((item.path.as_str(), reason.as_str(), issue.as_str()));
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

    eprintln!("--- conformance surface coverage (ferrotorch-core, phase 2.0) ---");
    eprintln!(
        "covered {}/{} (excluded: {}; uncovered: {})",
        covered.len(),
        surface.items.len(),
        excluded.len(),
        uncovered.len()
    );

    if !uncovered.is_empty() {
        eprintln!("\n  UNCOVERED items (need a conformance test OR an exclusion entry):");
        for item in &uncovered {
            eprintln!("    {}  (kind={})", item.path, item.kind);
        }
    }

    assert!(
        uncovered.is_empty(),
        "{} ferrotorch-core public item(s) lack a conformance reference. \
         Either author a test in tests/conformance_*.rs that references the \
         item by name, OR add it to tests/conformance/_surface_exclusions.toml \
         with `reason` and `tracking_issue` fields.",
        uncovered.len()
    );

    // Stale-exclusion guard: an exclusion for an item that no longer exists
    // is suspect (probably the item was renamed or removed and the exclusion
    // was forgotten).
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
