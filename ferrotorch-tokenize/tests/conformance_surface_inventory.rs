//! Conformance Phase 1 — surface inventory generator.
//!
//! Tracking issue: <https://github.com/<owner>/ferrotorch/issues/758>.
//!
//! Parses `src/lib.rs` (and any `mod` it declares) with `syn` and emits a
//! sorted JSON inventory of every `pub` item to
//! `tests/conformance/_surface.json`. The committed JSON file is the
//! contract; PRs that change the public surface show up as JSON diffs and
//! the coverage gate (`conformance_surface_coverage.rs`) fails if a new
//! item is not referenced by a conformance test.
//!
//! The test is intentionally a write-on-each-run producer so regeneration
//! never falls behind the source. `cargo test -p ferrotorch-tokenize
//! --test conformance_surface_inventory` rewrites the file; CI should
//! diff the result against `git` to catch undocumented surface changes.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use syn::{Item, ItemFn, ItemImpl, ItemMod, ItemStruct, Visibility};

#[derive(Debug)]
struct SurfaceItem {
    /// Fully qualified path, e.g. `ferrotorch_tokenize::encode`.
    path: String,
    /// One of: `fn`, `struct`, `enum`, `trait`, `type`, `const`, `static`,
    /// `method`, `re-export`.
    kind: &'static str,
    /// One-line signature trimmed of body / where-clauses.
    signature: String,
}

const CRATE_NAME: &str = "ferrotorch_tokenize";

fn crate_src_dir() -> PathBuf {
    // `CARGO_MANIFEST_DIR` is set by Cargo when running tests; this is
    // `<workspace>/ferrotorch-tokenize`.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn out_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("_surface.json")
}

/// Pretty-print a `syn` token stream into a single-line signature.
fn fmt_tokens<T: quote::ToTokens>(t: &T) -> String {
    let s = quote::quote!(#t).to_string();
    // Collapse internal whitespace runs to single spaces for stable diffs.
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

fn is_pub(vis: &Visibility) -> bool {
    matches!(vis, Visibility::Public(_))
}

/// Produce a synthetic signature for a function — no body, no where clauses.
fn fn_signature(item: &ItemFn) -> String {
    let sig = &item.sig;
    fmt_tokens(sig)
}

/// Produce a synthetic struct header — no fields.
fn struct_signature(item: &ItemStruct) -> String {
    let attrs: String = item
        .attrs
        .iter()
        .filter(|a| {
            // Keep only `#[non_exhaustive]` and `#[derive(...)]` for diff value.
            let p = a.path();
            p.is_ident("non_exhaustive") || p.is_ident("derive")
        })
        .map(fmt_tokens)
        .collect::<Vec<_>>()
        .join(" ");
    let vis = fmt_tokens(&item.vis);
    let ident = item.ident.to_string();
    let generics = fmt_tokens(&item.generics);
    if attrs.is_empty() {
        format!("{vis} struct {ident}{generics}")
    } else {
        format!("{attrs} {vis} struct {ident}{generics}")
    }
    .trim()
    .to_string()
}

fn collect_methods(impl_block: &ItemImpl, module_path: &str, out: &mut Vec<SurfaceItem>) {
    // Only enumerate inherent impls — trait impls expose nothing new at the
    // public surface beyond what the trait already promises.
    if impl_block.trait_.is_some() {
        return;
    }
    let ty = fmt_tokens(&impl_block.self_ty);
    for item in &impl_block.items {
        if let syn::ImplItem::Fn(method) = item
            && is_pub(&method.vis)
        {
            let sig = fmt_tokens(&method.sig);
            // `module_path` is the path to the module that declares the
            // impl block; include the type name so e.g.
            // `ferrotorch_tokenize::ChatMessage::new` is the path.
            out.push(SurfaceItem {
                path: format!("{module_path}::{ty}::{}", method.sig.ident),
                kind: "method",
                signature: sig,
            });
        }
    }
}

/// Recursively walk a module's items, accumulating public surface.
///
/// `module_path` starts as `ferrotorch_tokenize`; submodules appended.
/// `dir` is the directory the current module's source is in (so an inline
/// `mod foo { ... }` continues there, while `mod foo;` resolves to
/// `dir/foo.rs` or `dir/foo/mod.rs`).
fn walk_items(items: &[Item], module_path: &str, dir: &Path, out: &mut Vec<SurfaceItem>) {
    for item in items {
        match item {
            Item::Fn(f) if is_pub(&f.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", f.sig.ident),
                kind: "fn",
                signature: fn_signature(f),
            }),
            Item::Struct(s) if is_pub(&s.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", s.ident),
                kind: "struct",
                signature: struct_signature(s),
            }),
            Item::Enum(e) if is_pub(&e.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", e.ident),
                kind: "enum",
                signature: format!(
                    "{} enum {}{}",
                    fmt_tokens(&e.vis),
                    e.ident,
                    fmt_tokens(&e.generics)
                ),
            }),
            Item::Trait(t) if is_pub(&t.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", t.ident),
                kind: "trait",
                signature: format!(
                    "{} trait {}{}",
                    fmt_tokens(&t.vis),
                    t.ident,
                    fmt_tokens(&t.generics)
                ),
            }),
            Item::Type(ty) if is_pub(&ty.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", ty.ident),
                kind: "type",
                signature: format!(
                    "{} type {}{} = {}",
                    fmt_tokens(&ty.vis),
                    ty.ident,
                    fmt_tokens(&ty.generics),
                    fmt_tokens(&ty.ty)
                ),
            }),
            Item::Const(c) if is_pub(&c.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", c.ident),
                kind: "const",
                signature: format!(
                    "{} const {}: {}",
                    fmt_tokens(&c.vis),
                    c.ident,
                    fmt_tokens(&c.ty)
                ),
            }),
            Item::Static(s) if is_pub(&s.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", s.ident),
                kind: "static",
                signature: format!(
                    "{} static {}: {}",
                    fmt_tokens(&s.vis),
                    s.ident,
                    fmt_tokens(&s.ty)
                ),
            }),
            Item::Use(u) if is_pub(&u.vis) => {
                // Re-exports via `pub use` widen the surface. Collect each
                // leaf from the use tree.
                let mut leaves = Vec::new();
                collect_use_leaves(&u.tree, &mut Vec::new(), &mut leaves);
                for (segments, alias) in leaves {
                    let display_name =
                        alias.unwrap_or_else(|| segments.last().cloned().unwrap_or_default());
                    if display_name.is_empty() || display_name == "*" {
                        // Glob re-exports (`pub use foo::*;`) hide the items
                        // they expose; record the glob site so the coverage
                        // test surfaces them as needing a manual exclusion.
                        out.push(SurfaceItem {
                            path: format!("{module_path}::*"),
                            kind: "re-export",
                            signature: format!("pub use {};", segments.join("::") + "::*"),
                        });
                    } else {
                        out.push(SurfaceItem {
                            path: format!("{module_path}::{display_name}"),
                            kind: "re-export",
                            signature: format!("pub use {};", segments.join("::")),
                        });
                    }
                }
            }
            Item::Mod(m) => walk_module(m, module_path, dir, out),
            Item::Impl(i) => collect_methods(i, module_path, out),
            _ => {}
        }
    }
}

fn collect_use_leaves(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    out: &mut Vec<(Vec<String>, Option<String>)>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            collect_use_leaves(&p.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Name(n) => {
            let mut segs = prefix.clone();
            segs.push(n.ident.to_string());
            out.push((segs, None));
        }
        syn::UseTree::Rename(r) => {
            let mut segs = prefix.clone();
            segs.push(r.ident.to_string());
            out.push((segs, Some(r.rename.to_string())));
        }
        syn::UseTree::Glob(_) => {
            let mut segs = prefix.clone();
            segs.push("*".to_string());
            out.push((segs, None));
        }
        syn::UseTree::Group(g) => {
            for t in &g.items {
                collect_use_leaves(t, prefix, out);
            }
        }
    }
}

fn walk_module(m: &ItemMod, parent_path: &str, parent_dir: &Path, out: &mut Vec<SurfaceItem>) {
    // Modules with private visibility do not expose their `pub` items
    // outside the crate, so skip them entirely.
    if !is_pub(&m.vis) {
        return;
    }
    let new_path = format!("{parent_path}::{}", m.ident);
    if let Some((_, items)) = &m.content {
        // Inline module: `pub mod foo { ... }` — keep parent dir.
        walk_items(items, &new_path, parent_dir, out);
    } else {
        // External module: resolve `<dir>/<ident>.rs` or `<dir>/<ident>/mod.rs`.
        let ident = m.ident.to_string();
        let candidate_a = parent_dir.join(format!("{ident}.rs"));
        let candidate_b = parent_dir.join(&ident).join("mod.rs");
        let (path, new_dir): (PathBuf, PathBuf) = if candidate_a.exists() {
            (candidate_a.clone(), parent_dir.to_path_buf())
        } else if candidate_b.exists() {
            (candidate_b.clone(), parent_dir.join(&ident))
        } else {
            // Module declared but file missing — surface as an inventory
            // anomaly instead of silently dropping it.
            out.push(SurfaceItem {
                path: new_path.clone(),
                kind: "fn",
                signature: format!("/* UNRESOLVED MODULE: pub mod {ident}; */"),
            });
            return;
        };
        let src =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let file =
            syn::parse_file(&src).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        walk_items(&file.items, &new_path, &new_dir, out);
    }
}

/// Render `items` to a stable JSON document. Hand-rolls the formatting so
/// the output is human-diffable without depending on `serde_json` formatting
/// quirks.
fn render_json(items: &[SurfaceItem]) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"crate\": \"");
    s.push_str(CRATE_NAME);
    s.push_str("\",\n");
    s.push_str("  \"description\": \"Auto-generated by tests/conformance_surface_inventory.rs. Do not edit by hand.\",\n");
    s.push_str("  \"items\": [\n");
    for (i, it) in items.iter().enumerate() {
        s.push_str("    { \"path\": ");
        s.push_str(&json_escape(&it.path));
        s.push_str(", \"kind\": ");
        s.push_str(&json_escape(it.kind));
        s.push_str(", \"signature\": ");
        s.push_str(&json_escape(&it.signature));
        s.push_str(" }");
        if i + 1 < items.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[test]
fn surface_inventory_writes_json() {
    let lib_rs = crate_src_dir().join("lib.rs");
    let src = fs::read_to_string(&lib_rs).expect("read src/lib.rs");
    let file = syn::parse_file(&src).expect("parse src/lib.rs");

    let mut items = Vec::new();
    walk_items(&file.items, CRATE_NAME, &crate_src_dir(), &mut items);

    // Sort by path, with kind as tiebreaker so the `Foo` struct lists
    // before `Foo::new` deterministically.
    items.sort_by(|a, b| a.path.cmp(&b.path).then(a.kind.cmp(b.kind)));

    // Dedupe (a `Foo::new` defined in two impl blocks would otherwise
    // appear twice).
    let mut seen = BTreeMap::new();
    let mut unique = Vec::new();
    for it in items {
        let key = format!("{}|{}", it.path, it.kind);
        if seen.insert(key, ()).is_none() {
            unique.push(it);
        }
    }

    let json = render_json(&unique);
    fs::create_dir_all(out_path().parent().expect("conformance dir")).expect("mkdir conformance");
    fs::write(out_path(), &json).expect("write _surface.json");

    // Sanity: ferrotorch-tokenize must export at least the items the audit
    // mandate enumerated. If any are missing, the inventory walker is
    // broken (or the source genuinely lost a `pub`).
    let must_contain = [
        "ferrotorch_tokenize::load_tokenizer",
        "ferrotorch_tokenize::encode",
        "ferrotorch_tokenize::encode_batch",
        "ferrotorch_tokenize::decode",
        "ferrotorch_tokenize::vocab_size",
        "ferrotorch_tokenize::token_to_id",
        "ferrotorch_tokenize::id_to_token",
        "ferrotorch_tokenize::apply_chat_template",
        "ferrotorch_tokenize::apply_chat_template_to_ids",
        "ferrotorch_tokenize::load_chat_template",
        "ferrotorch_tokenize::ChatMessage",
        "ferrotorch_tokenize::ChatMessage::new",
    ];
    let paths: Vec<&str> = unique.iter().map(|i| i.path.as_str()).collect();
    for needle in must_contain {
        assert!(
            paths.contains(&needle),
            "surface inventory missing expected item {needle}; got: {paths:?}"
        );
    }
}
