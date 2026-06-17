#![allow(dead_code)]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use syn::{Item, ItemFn, ItemImpl, ItemMod, ItemStruct, Visibility};

/// One public item discovered from `ferrotorch-core/src`.
#[derive(Debug, Clone, Deserialize)]
pub struct SurfaceItem {
    /// Fully qualified path, e.g. `ferrotorch_core::creation::zeros`.
    pub path: String,
    /// One of: `fn`, `struct`, `enum`, `trait`, `type`, `const`, `static`,
    /// `method`, `re-export`.
    pub kind: String,
    /// One-line signature trimmed of body / where-clauses.
    pub signature: String,
}

#[derive(Debug, Deserialize)]
pub struct Surface {
    pub items: Vec<SurfaceItem>,
}

const CRATE_NAME: &str = "ferrotorch_core";

pub fn crate_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

pub fn conformance_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
}

pub fn out_path() -> PathBuf {
    conformance_dir().join("_surface.json")
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

fn fn_signature(item: &ItemFn) -> String {
    fmt_tokens(&item.sig)
}

fn struct_signature(item: &ItemStruct) -> String {
    let attrs: String = item
        .attrs
        .iter()
        .filter(|a| {
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
    // Inherent impls: every `pub` method becomes a surface item. Trait impls
    // are dictated by the trait, so they do not widen public surface.
    if impl_block.trait_.is_some() {
        return;
    }
    let ty = fmt_tokens(&impl_block.self_ty);
    let ty_clean = strip_generic_whitespace(&ty);
    for item in &impl_block.items {
        if let syn::ImplItem::Fn(method) = item
            && is_pub(&method.vis)
        {
            let sig = fmt_tokens(&method.sig);
            out.push(SurfaceItem {
                path: format!("{module_path}::{ty_clean}::{}", method.sig.ident),
                kind: "method".to_string(),
                signature: sig,
            });
        }
    }
}

/// Strip whitespace inside `<...>` generic arguments so paths read cleanly.
/// `Tensor < T >` becomes `Tensor<T>`; `Foo` is unchanged.
fn strip_generic_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0i32;
    for ch in s.chars() {
        match ch {
            '<' => {
                depth += 1;
                out.push(ch);
            }
            '>' => {
                depth -= 1;
                out.push(ch);
            }
            ' ' if depth > 0 => {}
            c => out.push(c),
        }
    }
    out
}

/// Recursively walk a module's items, accumulating public surface.
fn walk_items(items: &[Item], module_path: &str, dir: &Path, out: &mut Vec<SurfaceItem>) {
    for item in items {
        match item {
            Item::Fn(f) if is_pub(&f.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", f.sig.ident),
                kind: "fn".to_string(),
                signature: fn_signature(f),
            }),
            Item::Struct(s) if is_pub(&s.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", s.ident),
                kind: "struct".to_string(),
                signature: struct_signature(s),
            }),
            Item::Enum(e) if is_pub(&e.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", e.ident),
                kind: "enum".to_string(),
                signature: format!(
                    "{} enum {}{}",
                    fmt_tokens(&e.vis),
                    e.ident,
                    fmt_tokens(&e.generics)
                ),
            }),
            Item::Trait(t) if is_pub(&t.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", t.ident),
                kind: "trait".to_string(),
                signature: format!(
                    "{} trait {}{}",
                    fmt_tokens(&t.vis),
                    t.ident,
                    fmt_tokens(&t.generics)
                ),
            }),
            Item::Type(ty) if is_pub(&ty.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", ty.ident),
                kind: "type".to_string(),
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
                kind: "const".to_string(),
                signature: format!(
                    "{} const {}: {}",
                    fmt_tokens(&c.vis),
                    c.ident,
                    fmt_tokens(&c.ty)
                ),
            }),
            Item::Static(s) if is_pub(&s.vis) => out.push(SurfaceItem {
                path: format!("{module_path}::{}", s.ident),
                kind: "static".to_string(),
                signature: format!(
                    "{} static {}: {}",
                    fmt_tokens(&s.vis),
                    s.ident,
                    fmt_tokens(&s.ty)
                ),
            }),
            Item::Use(u) if is_pub(&u.vis) => {
                let mut leaves = Vec::new();
                collect_use_leaves(&u.tree, &mut Vec::new(), &mut leaves);
                for (segments, alias) in leaves {
                    let display_name =
                        alias.unwrap_or_else(|| segments.last().cloned().unwrap_or_default());
                    if display_name.is_empty() || display_name == "*" {
                        out.push(SurfaceItem {
                            path: format!("{module_path}::*"),
                            kind: "re-export".to_string(),
                            signature: format!("pub use {};", segments.join("::") + "::*"),
                        });
                    } else {
                        out.push(SurfaceItem {
                            path: format!("{module_path}::{display_name}"),
                            kind: "re-export".to_string(),
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
    // Modules with non-public visibility do not expose their `pub` items
    // outside the crate, so skip them entirely.
    if !is_pub(&m.vis) {
        return;
    }
    let new_path = format!("{parent_path}::{}", m.ident);
    if let Some((_, items)) = &m.content {
        walk_items(items, &new_path, parent_dir, out);
    } else {
        let ident = m.ident.to_string();
        let candidate_a = parent_dir.join(format!("{ident}.rs"));
        let candidate_b = parent_dir.join(&ident).join("mod.rs");
        let (path, new_dir): (PathBuf, PathBuf) = if candidate_a.exists() {
            (candidate_a, parent_dir.to_path_buf())
        } else if candidate_b.exists() {
            (candidate_b, parent_dir.join(&ident))
        } else {
            out.push(SurfaceItem {
                path: new_path.clone(),
                kind: "fn".to_string(),
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

pub fn collect_surface_items() -> Vec<SurfaceItem> {
    let lib_rs = crate_src_dir().join("lib.rs");
    let src = fs::read_to_string(&lib_rs).expect("read src/lib.rs");
    let file = syn::parse_file(&src).expect("parse src/lib.rs");

    let mut items = Vec::new();
    walk_items(&file.items, CRATE_NAME, &crate_src_dir(), &mut items);

    items.sort_by(|a, b| a.path.cmp(&b.path).then(a.kind.cmp(&b.kind)));

    let mut seen = BTreeSet::new();
    let mut unique = Vec::new();
    for it in items {
        let key = format!("{}|{}|{}", it.path, it.kind, it.signature);
        if seen.insert(key) {
            unique.push(it);
        }
    }
    unique
}

/// Render `items` to a stable JSON document. Hand-rolled so the output is
/// human-diffable without depending on `serde_json`'s formatting quirks.
pub fn render_json(items: &[SurfaceItem]) -> String {
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
        s.push_str(&json_escape(&it.kind));
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

pub fn read_surface_json(text: &str, label: &str) -> Surface {
    serde_json::from_str(text).unwrap_or_else(|e| panic!("parse {label}: {e}"))
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
