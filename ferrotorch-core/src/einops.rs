//! Einops-style tensor rearrangement operations.
//!
//! Provides `rearrange`, `repeat`, and `reduce` with readable string patterns
//! for expressing tensor shape transformations declaratively.
//!
//! # Pattern syntax
//!
//! A pattern has the form `"left -> right"` where `left` and `right` are
//! space-separated axis names. Parenthesized groups denote merged/split
//! dimensions:
//!
//! - `"b c h w -> b (c h w)"` merges `c`, `h`, `w` into one axis
//! - `"b (c h) w -> b c h w"` splits a dimension (requires `axes_lengths`)
//! - `"b h w c -> b c h w"` transposes (reorders) axes
//!
//! Axes present on the left but absent on the right are reduced (for `reduce`)
//! or must be size-1 (for `rearrange`). Axes present on the right but absent
//! on the left are new axes (for `repeat`).
//!
//! ## REQ status (per `.design/ferrotorch-core/einops.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `rearrange` at `einops.rs:393`; consumer: re-export at `lib.rs:143` |
//! | REQ-2 | SHIPPED | `rearrange_with` at `einops.rs:424`; consumer: re-export at `lib.rs:143` |
//! | REQ-3 | SHIPPED | `repeat` at `einops.rs:514`; consumer: re-export at `lib.rs:143` |
//! | REQ-4 | SHIPPED | `reduce` + `EinopsReduction` at `einops.rs:614,43`; consumer: re-export at `lib.rs:143` |
//! | REQ-5 | SHIPPED | `parse_pattern`/`parse_side`/`read_axis_name` at `einops.rs:89-195`; consumer: every public API invokes `parse_pattern` first |
//! | REQ-6 | SHIPPED | `resolve_sizes` at `einops.rs:221`; consumer: every public API after `parse_pattern` |

use std::collections::HashMap;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Public API — Reduction enum
// ---------------------------------------------------------------------------

/// Reduction operation for [`reduce`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EinopsReduction {
    /// Arithmetic mean along reduced axes.
    Mean,
    /// Sum along reduced axes.
    Sum,
    /// Element-wise maximum along reduced axes.
    Max,
    /// Element-wise minimum along reduced axes.
    Min,
}

// ---------------------------------------------------------------------------
// Pattern parser
// ---------------------------------------------------------------------------

/// A single axis on one side of the pattern. Either a bare name or a
/// parenthesized group of names (representing a merged/split dimension).
#[derive(Debug, Clone, PartialEq)]
enum AxisSpec {
    /// A single named axis, e.g. `b`.
    Single(String),
    /// A parenthesized group of axes, e.g. `(c h w)`.
    Group(Vec<String>),
}

/// Parsed einops pattern.
#[derive(Debug)]
struct ParsedPattern {
    left: Vec<AxisSpec>,
    right: Vec<AxisSpec>,
}

/// Flatten an `AxisSpec` list into individual axis names in order.
fn flatten_axes(specs: &[AxisSpec]) -> Vec<String> {
    let mut out = Vec::new();
    for spec in specs {
        match spec {
            AxisSpec::Single(name) => out.push(name.clone()),
            AxisSpec::Group(names) => out.extend(names.iter().cloned()),
        }
    }
    out
}

/// Parse one side of the pattern (e.g. `"b (c h) w"`) into a list of
/// `AxisSpec` entries.
fn parse_side(s: &str) -> FerrotorchResult<Vec<AxisSpec>> {
    let s = s.trim();
    let mut specs = Vec::new();
    let mut chars = s.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }

        if c == '(' {
            // Consume the opening paren.
            chars.next();
            let mut group = Vec::new();
            loop {
                // Skip whitespace inside parens.
                while let Some(&c2) = chars.peek() {
                    if c2.is_whitespace() {
                        chars.next();
                    } else {
                        break;
                    }
                }
                match chars.peek() {
                    None => {
                        return Err(FerrotorchError::InvalidArgument {
                            message: "einops: unmatched '(' in pattern".into(),
                        });
                    }
                    Some(&')') => {
                        chars.next();
                        break;
                    }
                    _ => {}
                }
                // Read an axis name.
                let name = read_axis_name(&mut chars)?;
                if name.is_empty() {
                    return Err(FerrotorchError::InvalidArgument {
                        message: "einops: empty axis name inside parentheses".into(),
                    });
                }
                group.push(name);
            }
            if group.is_empty() {
                return Err(FerrotorchError::InvalidArgument {
                    message: "einops: empty parenthesized group".into(),
                });
            }
            specs.push(AxisSpec::Group(group));
        } else if c.is_ascii_alphanumeric() || c == '_' {
            let name = read_axis_name(&mut chars)?;
            specs.push(AxisSpec::Single(name));
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("einops: unexpected character '{c}' in pattern"),
            });
        }
    }

    Ok(specs)
}

/// Read an axis name: a run of alphanumeric / underscore characters.
#[allow(clippy::unnecessary_wraps)] // reason: keeps signature uniform with sibling parser helpers (read_int, read_group) that DO fail
fn read_axis_name(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> FerrotorchResult<String> {
    let mut name = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_alphanumeric() || c == '_' {
            name.push(c);
            chars.next();
        } else {
            break;
        }
    }
    Ok(name)
}

/// Parse a full pattern like `"b c h w -> b (c h) w"`.
fn parse_pattern(pattern: &str) -> FerrotorchResult<ParsedPattern> {
    let pattern = pattern.trim();
    let (left_str, right_str) =
        pattern
            .split_once("->")
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("einops: pattern must contain '->', got: \"{pattern}\""),
            })?;

    let left = parse_side(left_str)?;
    let right = parse_side(right_str)?;

    // Validate: no duplicate axis names within a side.
    let left_names = flatten_axes(&left);
    let right_names = flatten_axes(&right);

    let mut seen = HashMap::new();
    for name in &left_names {
        if seen.insert(name.as_str(), "left").is_some() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("einops: duplicate axis name '{name}' on left side of pattern"),
            });
        }
    }
    seen.clear();
    for name in &right_names {
        if seen.insert(name.as_str(), "right").is_some() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("einops: duplicate axis name '{name}' on right side of pattern"),
            });
        }
    }

    Ok(ParsedPattern { left, right })
}

// ---------------------------------------------------------------------------
// Axis-size resolution
// ---------------------------------------------------------------------------

/// Resolve the size of every named axis. Returns a map from axis name to
/// its size.
///
/// - Axes that appear as `Single` on the left get their size from the
///   corresponding input dimension.
/// - Axes inside a `Group` on the left come from splitting an input dim.
///   If there are N sub-axes and all but one have known sizes (from
///   `axes_lengths`), the remaining one is inferred.
/// - Axes that only appear on the right (new axes) must have their size
///   supplied in `axes_lengths`.
fn resolve_sizes(
    pattern: &ParsedPattern,
    input_shape: &[usize],
    axes_lengths: &[(&str, usize)],
) -> FerrotorchResult<HashMap<String, usize>> {
    let left_flat = flatten_axes(&pattern.left);
    let right_flat = flatten_axes(&pattern.right);

    // Count how many input dimensions the left side represents.
    let left_dim_count = pattern.left.len();
    if left_dim_count != input_shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "einops: left side of pattern has {} axes but input tensor has {} dimensions",
                left_dim_count,
                input_shape.len()
            ),
        });
    }

    let user_sizes: HashMap<&str, usize> = axes_lengths.iter().copied().collect();
    let mut sizes: HashMap<String, usize> = HashMap::new();

    // First pass: assign sizes from the left side.
    for (dim_idx, spec) in pattern.left.iter().enumerate() {
        let dim_size = input_shape[dim_idx];
        match spec {
            AxisSpec::Single(name) => {
                sizes.insert(name.clone(), dim_size);
            }
            AxisSpec::Group(names) => {
                // This is a split: one input dim is being decomposed into
                // multiple named axes. We need axes_lengths for all but
                // (at most) one of them.
                let mut unknown_idx: Option<usize> = None;
                let mut known_product: usize = 1;

                for (i, name) in names.iter().enumerate() {
                    if let Some(&sz) = user_sizes.get(name.as_str()) {
                        sizes.insert(name.clone(), sz);
                        known_product *= sz;
                    } else if let Some(&sz) = sizes.get(name) {
                        // Already known from a previous occurrence (shouldn't happen
                        // since we check duplicates, but be defensive).
                        known_product *= sz;
                    } else {
                        if unknown_idx.is_some() {
                            return Err(FerrotorchError::InvalidArgument {
                                message: format!(
                                    "einops: cannot infer sizes for split '({})' — \
                                     provide sizes for all but one sub-axis via axes_lengths",
                                    names.join(" ")
                                ),
                            });
                        }
                        unknown_idx = Some(i);
                    }
                }

                if let Some(ui) = unknown_idx {
                    if known_product == 0 || !dim_size.is_multiple_of(known_product) {
                        return Err(FerrotorchError::InvalidArgument {
                            message: format!(
                                "einops: dimension {} (size {}) is not divisible by \
                                 known product {} for split '({})'",
                                dim_idx,
                                dim_size,
                                known_product,
                                names.join(" ")
                            ),
                        });
                    }
                    sizes.insert(names[ui].clone(), dim_size / known_product);
                } else {
                    // All sub-axes are known; verify the product matches.
                    if known_product != dim_size {
                        return Err(FerrotorchError::ShapeMismatch {
                            message: format!(
                                "einops: split '({})' product {} does not match dimension {} size {}",
                                names.join(" "),
                                known_product,
                                dim_idx,
                                dim_size
                            ),
                        });
                    }
                }
            }
        }
    }

    // Second pass: axes that only appear on the right (new axes) must come
    // from axes_lengths.
    for name in &right_flat {
        if !sizes.contains_key(name) {
            if let Some(&sz) = user_sizes.get(name.as_str()) {
                sizes.insert(name.clone(), sz);
            } else if !left_flat.contains(name) {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "einops: axis '{name}' appears on the right but not the left \
                         and has no size in axes_lengths"
                    ),
                });
            }
        }
    }

    Ok(sizes)
}

// ---------------------------------------------------------------------------
// Core implementation helpers
// ---------------------------------------------------------------------------

/// Compute the output shape from the right side of the pattern and the
/// resolved axis sizes.
fn output_shape(
    right: &[AxisSpec],
    sizes: &HashMap<String, usize>,
) -> FerrotorchResult<Vec<usize>> {
    right
        .iter()
        .map(|spec| match spec {
            AxisSpec::Single(name) => Ok(*sizes.get(name).unwrap()),
            AxisSpec::Group(names) => {
                let dims: Vec<usize> = names.iter().map(|n| *sizes.get(n).unwrap()).collect();
                crate::shape::checked_numel(&dims, "einops output_shape")
            }
        })
        .collect()
}

/// Build the "elementary" shape from a pattern side: each `AxisSpec::Group`
/// is expanded into its individual sub-axis sizes.
fn elementary_shape(specs: &[AxisSpec], sizes: &HashMap<String, usize>) -> Vec<usize> {
    let mut shape = Vec::new();
    for spec in specs {
        match spec {
            AxisSpec::Single(name) => shape.push(*sizes.get(name).unwrap()),
            AxisSpec::Group(names) => {
                for n in names {
                    shape.push(*sizes.get(n).unwrap());
                }
            }
        }
    }
    shape
}

/// Differentiable reshape shim: converts a `usize` shape and delegates to
/// [`crate::grad_fns::shape::reshape`], which attaches a `ReshapeBackward`
/// node when the input tracks gradients and degrades to a zero-copy
/// `view_reshape` when it does not. Non-contiguous inputs are materialized
/// device-aware via `contiguous()` (GPU `strided_copy_*` for f32/f64, host
/// round-trip otherwise) — see `tensor.rs` `view_operation` (#1705).
fn reshape_diff<T: Float>(input: &Tensor<T>, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let shape_isize: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
    crate::grad_fns::shape::reshape(input, &shape_isize)
}

// ---------------------------------------------------------------------------
// Public API — rearrange
// ---------------------------------------------------------------------------

/// Rearrange tensor dimensions using an einops-style pattern.
///
/// # Examples
/// ```ignore
/// // Flatten spatial dims: [B, C, H, W] -> [B, C*H*W]
/// rearrange(&t, "b c h w -> b (c h w)")?;
///
/// // Transpose: [B, H, W, C] -> [B, C, H, W]
/// rearrange(&t, "b h w c -> b c h w")?;
///
/// // Merge dims: [B, H, W, C] -> [B, H*W, C]
/// rearrange(&t, "b h w c -> b (h w) c")?;
/// ```
pub fn rearrange<T: Float>(input: &Tensor<T>, pattern: &str) -> FerrotorchResult<Tensor<T>> {
    rearrange_with(input, pattern, &[])
}

/// Rearrange with explicit axis sizes for ambiguous splits.
///
/// # Examples
/// ```ignore
/// // Split a dimension: [B, C*H, W] -> [B, C, H, W] with C=3
/// rearrange_with(&t, "b (c h) w -> b c h w", &[("c", 3)])?;
/// ```
///
/// # Device and autograd behavior (CORE-061 / #1755)
///
/// Built entirely from differentiable tensor operations: when `input` tracks
/// gradients the output carries a real backward chain
/// (`ReshapeBackward` → `PermuteBackward` → `ReshapeBackward`/`ContiguousBackward`)
/// reaching the original leaf; when it does not, the same composition takes
/// the zero-copy no-grad fast paths.
///
/// When the operation reduces to a pure flatten/unflatten (no axis reordering
/// at the elementary level — e.g. `"b c h w -> b (c h w)"` or
/// `"b (c h) w -> b c h w"`), this is a single (zero-copy) reshape and runs
/// on any device with no data movement.
///
/// When the operation requires actual axis reordering (e.g.
/// `"b h w c -> b c h w"`), it goes through `reshape → permute → reshape`.
/// The `permute` is a zero-copy stride view; the final reshape materializes
/// the permuted layout via `contiguous()`, which stays on-device for CUDA
/// f32/f64 (`strided_copy_*` kernels) and round-trips through the host for
/// other dtypes (explicit per R-LOUD-2; see `methods.rs::contiguous_t`).
pub fn rearrange_with<T: Float>(
    input: &Tensor<T>,
    pattern: &str,
    axes_lengths: &[(&str, usize)],
) -> FerrotorchResult<Tensor<T>> {
    let parsed = parse_pattern(pattern)?;
    let sizes = resolve_sizes(&parsed, input.shape(), axes_lengths)?;

    let left_names = flatten_axes(&parsed.left);
    let right_names = flatten_axes(&parsed.right);

    // For rearrange, left and right must name exactly the same set of axes.
    let mut left_sorted = left_names.clone();
    left_sorted.sort();
    let mut right_sorted = right_names.clone();
    right_sorted.sort();
    if left_sorted != right_sorted {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "einops rearrange: left axes {left_names:?} and right axes {right_names:?} must name \
                 the same set of axes (use `repeat` for new axes, `reduce` for removed axes)"
            ),
        });
    }

    let out_shape = output_shape(&parsed.right, &sizes)?;
    let left_elem_shape = elementary_shape(&parsed.left, &sizes);

    // Compute the permutation from left elementary order to right elementary
    // order. Since left and right name the same axes, every right axis has a
    // unique position in left.
    let perm: Vec<usize> = right_names
        .iter()
        .map(|name| {
            left_names
                .iter()
                .position(|n| n == name)
                .expect("axis sets validated to match above")
        })
        .collect();

    // Fast path: identity permutation. This covers the common cases of pure
    // flatten (`"b c h w -> b (c h w)"`), pure unflatten
    // (`"b (c h) w -> b c h w"`), and grouping rearrangements where the
    // elementary axis order matches between left and right. A single
    // differentiable reshape — zero-copy on any device for contiguous
    // inputs, with `ReshapeBackward` attached when gradients are tracked.
    let is_identity_perm = perm.iter().enumerate().all(|(i, &p)| i == p);
    if is_identity_perm {
        return reshape_diff(input, &out_shape);
    }

    // General path: reshape to elementary form, permute (zero-copy stride
    // view), reshape to the merged output (materializes the permuted layout
    // device-aware). Every step participates in autograd (CORE-061 / #1755);
    // gradient behavior does not depend on layout or device.
    let elem = reshape_diff(input, &left_elem_shape)?;
    let permuted = elem.permute(&perm)?;
    reshape_diff(&permuted, &out_shape)
}

// ---------------------------------------------------------------------------
// Public API — repeat
// ---------------------------------------------------------------------------

/// Repeat tensor elements along new or existing axes.
///
/// Axes on the right that do not appear on the left are new dimensions and
/// must have their size specified in `axes_lengths`.
///
/// # Examples
/// ```ignore
/// // Add a batch dim by repeating: [H, W] -> [B, H, W]
/// repeat(&t, "h w -> b h w", &[("b", 4)])?;
///
/// // Tile: [C] -> [C, 3]
/// repeat(&t, "c -> c n", &[("n", 3)])?;
/// ```
///
/// # Device and autograd behavior (CORE-061 / #1755)
///
/// Built from differentiable operations
/// (`reshape → permute → reshape → expand → reshape`): tracked inputs get a
/// real backward chain whose `ExpandBackward` sums gradients over the
/// repeated axes (the einops repeat VJP), reaching the original leaf.
/// Untracked inputs take the no-grad fast paths.
///
/// On CUDA, `expand` materializes on-device for f32/f64; other dtypes return
/// a structured `Err(NotImplementedOnCuda)` rather than silently demoting to
/// host (R-LOUD-1).
pub fn repeat<T: Float>(
    input: &Tensor<T>,
    pattern: &str,
    axes_lengths: &[(&str, usize)],
) -> FerrotorchResult<Tensor<T>> {
    let parsed = parse_pattern(pattern)?;
    let sizes = resolve_sizes(&parsed, input.shape(), axes_lengths)?;

    let left_names = flatten_axes(&parsed.left);
    let right_names = flatten_axes(&parsed.right);

    // Every left axis must appear on the right.
    for name in &left_names {
        if !right_names.contains(name) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "einops repeat: left axis '{name}' does not appear on the right — \
                     use `reduce` to remove axes"
                ),
            });
        }
    }

    // Build the elementary shapes and the output shape.
    let left_elem_shape = elementary_shape(&parsed.left, &sizes);
    let right_elem_shape = elementary_shape(&parsed.right, &sizes);
    let out_shape = output_shape(&parsed.right, &sizes)?;

    // Differentiable composition (CORE-061 / #1755). Coordinate mapping is BY
    // AXIS NAME (CORE-062 / #1756): the kept-axis permutation is derived from
    // each left axis's position among the right names, so reordered patterns
    // read the correct elements by construction.
    //
    // 1. Split: reshape to the left elementary form.
    let elem = reshape_diff(input, &left_elem_shape)?;

    // 2. Reorder: bring the kept axes into their right-side relative order.
    //    (Every left axis appears exactly once on the right — validated above.)
    let kept_perm: Vec<usize> = right_names
        .iter()
        .filter_map(|name| left_names.iter().position(|n| n == name))
        .collect();
    let is_identity_perm = kept_perm.iter().enumerate().all(|(i, &p)| i == p);
    let permuted = if is_identity_perm {
        elem
    } else {
        elem.permute(&kept_perm)?
    };

    // 3. Seed: insert size-1 dims at the new-axis positions (right
    //    elementary order). Pure metadata — numel is unchanged.
    let pre_expand_shape: Vec<usize> = right_names
        .iter()
        .map(|name| {
            if left_names.contains(name) {
                *sizes.get(name).expect("left axes sized in resolve_sizes")
            } else {
                1
            }
        })
        .collect();
    let seeded = reshape_diff(&permuted, &pre_expand_shape)?;

    // 4. Broadcast the new axes to their requested sizes. `ExpandBackward`
    //    sum-reduces gradients over these axes — exactly the repeat VJP.
    let expanded = if pre_expand_shape == right_elem_shape {
        seeded // no new axes (pure reorder through the repeat API)
    } else {
        crate::grad_fns::shape::expand(&seeded, &right_elem_shape)?
    };

    // 5. Merge groups into the final output shape.
    reshape_diff(&expanded, &out_shape)
}

// ---------------------------------------------------------------------------
// Public API — reduce
// ---------------------------------------------------------------------------

/// Reduce along axes that appear on the left but not the right.
///
/// # Examples
/// ```ignore
/// // Global average pool: [B, C, H, W] -> [B, C]
/// reduce(&t, "b c h w -> b c", EinopsReduction::Mean)?;
///
/// // Sum over batch: [B, C] -> [C]
/// reduce(&t, "b c -> c", EinopsReduction::Sum)?;
/// ```
///
/// # Device and autograd behavior (CORE-061 / #1755)
///
/// One differentiable composition for every pattern (see the body comment):
/// tracked inputs receive a backward chain reaching the original leaf for
/// Sum/Mean (any device) and Max/Min (CPU). Max/Min backward on CUDA
/// surfaces `CummaxBackward`/`CumminBackward`'s structured
/// `Err(NotImplementedOnCuda)` — loud, never a silent detach (tracked
/// follow-up: #1962). Max/Min gradient ties follow cummax/cummin (full
/// gradient to the recorded occurrence), diverging from torch `amax`/`amin`
/// even-split ONLY on exact ties (tracked follow-up: #1963).
pub fn reduce<T: Float>(
    input: &Tensor<T>,
    pattern: &str,
    reduction: EinopsReduction,
) -> FerrotorchResult<Tensor<T>> {
    let parsed = parse_pattern(pattern)?;
    let sizes = resolve_sizes(&parsed, input.shape(), &[])?;

    let left_names = flatten_axes(&parsed.left);
    let right_names = flatten_axes(&parsed.right);

    // Every right axis must appear on the left.
    for name in &right_names {
        if !left_names.contains(name) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "einops reduce: right axis '{name}' does not appear on the left — \
                     use `repeat` to add new axes"
                ),
            });
        }
    }

    // Identify reduced axes (on left but not right).
    let reduced_axes: Vec<&str> = left_names
        .iter()
        .filter(|n| !right_names.contains(n))
        .map(String::as_str)
        .collect();

    if reduced_axes.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "einops reduce: no axes are being reduced — use `rearrange` instead".into(),
        });
    }

    // Build the elementary shapes and the output shape.
    let left_elem_shape = elementary_shape(&parsed.left, &sizes);
    let right_elem_shape = elementary_shape(&parsed.right, &sizes);
    let out_shape = output_shape(&parsed.right, &sizes)?;

    // ----------------------------------------------------------------------
    // Single differentiable composition (CORE-061 / #1755) — one path for
    // every pattern, layout, and device, so gradient behavior never depends
    // on which internal decomposition happened to fire:
    //
    //   reshape(left elementary)                       — split
    //   → permute(kept-in-RIGHT-order ++ reduced)      — reorder by axis name
    //   → reshape([right_elem..., reduce_count])       — collapse reduced run
    //   → sum_dim / sum_dim·(1/N) / cummax / cummin    — reduce trailing dim
    //   → reshape(out_shape)                           — merge groups
    //
    // Coordinate mapping is BY AXIS NAME (CORE-062 / #1756): the permutation
    // places each kept axis at its right-side position before any
    // flattening, so reordered kept axes land correctly by construction.
    //
    // All steps run on the input's native device. For Mean, sum_dim is
    // followed by a scalar multiply by 1/reduce_count (`mul` is GPU-aware;
    // `mean_dim` is not). For Max/Min, the running cummax/cummin's last slice
    // is the global extremum; its backward (`CummaxBackward`/`CumminBackward`)
    // routes gradients to the recorded extremum positions on CPU and returns
    // a structured `Err(NotImplementedOnCuda)` on CUDA — loud, never a
    // silent detach (R-LOUD-1).
    // ----------------------------------------------------------------------
    let reduced_left_positions: Vec<usize> = left_names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            if right_names.contains(name) {
                None
            } else {
                Some(i)
            }
        })
        .collect();
    let reduce_count: usize = reduced_left_positions
        .iter()
        .map(|&i| left_elem_shape[i])
        .product();
    if reduce_count == 0 && matches!(reduction, EinopsReduction::Max | EinopsReduction::Min) {
        // Mirrors torch: `amax`/`amin` over an empty slice has no identity.
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "einops reduce: cannot compute {reduction:?} over an empty reduction axis \
                 (reduced extent is 0)"
            ),
        });
    }

    // 1. Split: reshape to the left elementary form.
    let elem = reshape_diff(input, &left_elem_shape)?;

    // 2. Reorder: kept axes (in right-side order, by name) first, reduced
    //    axes last.
    let mut perm: Vec<usize> = right_names
        .iter()
        .map(|name| {
            left_names
                .iter()
                .position(|n| n == name)
                .expect("validated above: every right axis appears on the left")
        })
        .collect();
    perm.extend(reduced_left_positions.iter().copied());
    let is_identity_perm = perm.iter().enumerate().all(|(i, &p)| i == p);
    let permuted = if is_identity_perm {
        elem
    } else {
        elem.permute(&perm)?
    };

    // 3. Collapse the (now trailing) reduced axes into a single dimension.
    let mut grouped_shape = right_elem_shape.clone();
    grouped_shape.push(reduce_count);
    let grouped = reshape_diff(&permuted, &grouped_shape)?;
    let last_dim = right_elem_shape.len() as i64;

    // 4. Reduce the trailing dimension.
    let reduced_t = match reduction {
        EinopsReduction::Sum => crate::grad_fns::reduction::sum_dim(&grouped, last_dim, false)?,
        EinopsReduction::Mean => {
            let summed = crate::grad_fns::reduction::sum_dim(&grouped, last_dim, false)?;
            let n_recip = <T as num_traits::One>::one() / T::from(reduce_count).unwrap();
            let scale_t = crate::creation::scalar(n_recip)?.to(input.device())?;
            crate::grad_fns::arithmetic::mul(&summed, &scale_t)?
        }
        EinopsReduction::Max => {
            // The running max ends with the global max along that axis.
            let cmax = crate::grad_fns::cumulative::cummax(&grouped, last_dim)?;
            cmax.values
                .narrow(right_elem_shape.len(), reduce_count - 1, 1)?
                .squeeze_t(last_dim as isize)?
        }
        EinopsReduction::Min => {
            let cmin = crate::grad_fns::cumulative::cummin(&grouped, last_dim)?;
            cmin.values
                .narrow(right_elem_shape.len(), reduce_count - 1, 1)?
                .squeeze_t(last_dim as isize)?
        }
    };

    // 5. Merge groups into the final output shape.
    reshape_diff(&reduced_t, &out_shape)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    /// Helper: create a leaf tensor.
    fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    // -----------------------------------------------------------------------
    // rearrange tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rearrange_identity() {
        // "b c h w -> b c h w" should be a no-op.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 3, 2, 2]);
        let r = rearrange(&t, "b c h w -> b c h w").unwrap();
        assert_eq!(r.shape(), &[2, 3, 2, 2]);
        assert_eq!(r.data().unwrap(), data.as_slice());
    }

    #[test]
    fn test_rearrange_flatten() {
        // "b c h w -> b (c h w)" merges c, h, w.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 3, 2, 2]); // B=2, C=3, H=2, W=2
        let r = rearrange(&t, "b c h w -> b (c h w)").unwrap();
        assert_eq!(r.shape(), &[2, 12]);
        assert_eq!(r.data().unwrap(), data.as_slice());
    }

    #[test]
    // reason: rearrange ("b h w c -> b c h w") is pure axis permutation —
    // each output slot holds the exact bit pattern of an input slot, so
    // bit-exact equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_rearrange_transpose_nhwc_to_nchw() {
        // "b h w c -> b c h w" transposes.
        // Input shape: [1, 2, 2, 3] (B=1, H=2, W=2, C=3)
        // Output shape: [1, 3, 2, 2]
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = leaf(&data, &[1, 2, 2, 3]);
        let r = rearrange(&t, "b h w c -> b c h w").unwrap();
        assert_eq!(r.shape(), &[1, 3, 2, 2]);

        // Verify specific elements.
        // Input[0,0,0,:] = [0,1,2], Input[0,0,1,:] = [3,4,5]
        // Input[0,1,0,:] = [6,7,8], Input[0,1,1,:] = [9,10,11]
        // Output[0,c,h,w] = Input[0,h,w,c]
        // Output[0,0,0,0] = Input[0,0,0,0] = 0
        // Output[0,0,0,1] = Input[0,0,1,0] = 3
        // Output[0,0,1,0] = Input[0,1,0,0] = 6
        // Output[0,0,1,1] = Input[0,1,1,0] = 9
        // Output[0,1,0,0] = Input[0,0,0,1] = 1
        // etc.
        let out = r.data().unwrap();
        assert_eq!(out[0], 0.0); // [0,0,0,0]
        assert_eq!(out[1], 3.0); // [0,0,0,1]
        assert_eq!(out[2], 6.0); // [0,0,1,0]
        assert_eq!(out[3], 9.0); // [0,0,1,1]
        assert_eq!(out[4], 1.0); // [0,1,0,0]
        assert_eq!(out[5], 4.0); // [0,1,0,1]
    }

    #[test]
    fn test_rearrange_split_with_axes_lengths() {
        // "b (c h) w -> b c h w" with c=3 splits dimension 1.
        // Input: [2, 6, 4] -> Output: [2, 3, 2, 4]
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 6, 4]);
        let r = rearrange_with(&t, "b (c h) w -> b c h w", &[("c", 3)]).unwrap();
        assert_eq!(r.shape(), &[2, 3, 2, 4]);

        // The data should be the same since (c h) is already in order and
        // we're just splitting.
        assert_eq!(r.data().unwrap(), data.as_slice());
    }

    #[test]
    fn test_rearrange_merge_dims() {
        // "b h w c -> b (h w) c" merges h and w.
        // Input: [1, 2, 3, 4] -> Output: [1, 6, 4]
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[1, 2, 3, 4]);
        let r = rearrange(&t, "b h w c -> b (h w) c").unwrap();
        assert_eq!(r.shape(), &[1, 6, 4]);
        // Data stays the same since h and w are adjacent and in order.
        assert_eq!(r.data().unwrap(), data.as_slice());
    }

    // -----------------------------------------------------------------------
    // repeat tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_repeat_new_batch_dim() {
        // "h w -> b h w" adds a batch dimension.
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = leaf(&data, &[2, 2]);
        let r = repeat(&t, "h w -> b h w", &[("b", 3)]).unwrap();
        assert_eq!(r.shape(), &[3, 2, 2]);

        let out = r.data().unwrap();
        // Each batch should be a copy of the original.
        assert_eq!(&out[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&out[4..8], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&out[8..12], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_repeat_tile() {
        // "c -> c n" tiles a 1-D tensor.
        let data = vec![10.0f32, 20.0, 30.0];
        let t = leaf(&data, &[3]);
        let r = repeat(&t, "c -> c n", &[("n", 2)]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);

        let out = r.data().unwrap();
        assert_eq!(out, &[10.0, 10.0, 20.0, 20.0, 30.0, 30.0]);
    }

    // -----------------------------------------------------------------------
    // reduce tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_mean_spatial() {
        // "b c h w -> b c" — global average pool.
        // B=1, C=2, H=2, W=2
        // Channel 0: [1, 2, 3, 4] mean = 2.5
        // Channel 1: [5, 6, 7, 8] mean = 6.5
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let t = leaf(&data, &[1, 2, 2, 2]);
        let r = reduce(&t, "b c h w -> b c", EinopsReduction::Mean).unwrap();
        assert_eq!(r.shape(), &[1, 2]);
        let out = r.data().unwrap();
        assert!((out[0] - 2.5).abs() < 1e-6, "expected 2.5, got {}", out[0]);
        assert!((out[1] - 6.5).abs() < 1e-6, "expected 6.5, got {}", out[1]);
    }

    #[test]
    fn test_reduce_sum_batch() {
        // "b c -> c" — sum over batch.
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = leaf(&data, &[3, 2]); // B=3, C=2
        let r = reduce(&t, "b c -> c", EinopsReduction::Sum).unwrap();
        assert_eq!(r.shape(), &[2]);
        let out = r.data().unwrap();
        // c=0: 1 + 3 + 5 = 9
        // c=1: 2 + 4 + 6 = 12
        assert!((out[0] - 9.0).abs() < 1e-6);
        assert!((out[1] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_max() {
        // "b c -> c" — max over batch.
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0, 6.0];
        let t = leaf(&data, &[3, 2]);
        let r = reduce(&t, "b c -> c", EinopsReduction::Max).unwrap();
        assert_eq!(r.shape(), &[2]);
        let out = r.data().unwrap();
        assert!((out[0] - 4.0).abs() < 1e-6); // max(1, 3, 4)
        assert!((out[1] - 6.0).abs() < 1e-6); // max(5, 2, 6)
    }

    #[test]
    fn test_reduce_min() {
        // "b c -> c" — min over batch.
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0, 6.0];
        let t = leaf(&data, &[3, 2]);
        let r = reduce(&t, "b c -> c", EinopsReduction::Min).unwrap();
        assert_eq!(r.shape(), &[2]);
        let out = r.data().unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6); // min(1, 3, 4)
        assert!((out[1] - 2.0).abs() < 1e-6); // min(5, 2, 6)
    }

    // -----------------------------------------------------------------------
    // Error tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_pattern_no_arrow() {
        let t = leaf(&[1.0, 2.0, 3.0], &[3]);
        assert!(rearrange(&t, "a b c").is_err());
    }

    #[test]
    fn test_mismatched_axis_count() {
        let t = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        // Left side has 3 axes but tensor has 2 dims.
        assert!(rearrange(&t, "a b c -> a b c").is_err());
    }

    #[test]
    fn test_rearrange_missing_axis_on_right() {
        // "b c h w -> b c" would be a reduce, not a rearrange.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 3, 2, 2]);
        assert!(rearrange(&t, "b c h w -> b c").is_err());
    }

    #[test]
    fn test_rearrange_extra_axis_on_right() {
        // "b c -> b c n" would be a repeat, not a rearrange.
        let t = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(rearrange(&t, "b c -> b c n").is_err());
    }

    #[test]
    fn test_repeat_missing_new_axis_size() {
        let t = leaf(&[1.0, 2.0], &[2]);
        // "c -> c n" but no size given for n.
        assert!(repeat(&t, "c -> c n", &[]).is_err());
    }

    #[test]
    fn test_reduce_no_reduction() {
        // "b c -> b c" reduces nothing — should error.
        let t = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(reduce(&t, "b c -> b c", EinopsReduction::Sum).is_err());
    }

    #[test]
    fn test_unmatched_paren() {
        let t = leaf(&[1.0, 2.0], &[2]);
        assert!(rearrange(&t, "(a -> a").is_err());
    }

    #[test]
    fn test_duplicate_axis_name() {
        let t = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(rearrange(&t, "a a -> a a").is_err());
    }

    // -----------------------------------------------------------------------
    // Parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_simple() {
        let p = parse_pattern("b c h w -> b c h w").unwrap();
        assert_eq!(flatten_axes(&p.left), vec!["b", "c", "h", "w"]);
        assert_eq!(flatten_axes(&p.right), vec!["b", "c", "h", "w"]);
    }

    #[test]
    fn test_parse_groups() {
        let p = parse_pattern("b c h w -> b (c h w)").unwrap();
        assert_eq!(p.right.len(), 2); // b, (c h w)
        match &p.right[1] {
            AxisSpec::Group(names) => assert_eq!(names, &["c", "h", "w"]),
            _ => panic!("expected Group"),
        }
    }

    #[test]
    fn test_parse_left_group() {
        let p = parse_pattern("b (c h) w -> b c h w").unwrap();
        assert_eq!(p.left.len(), 3); // b, (c h), w
        match &p.left[1] {
            AxisSpec::Group(names) => assert_eq!(names, &["c", "h"]),
            _ => panic!("expected Group"),
        }
    }

    // -----------------------------------------------------------------------
    // GPU-aware fast-path tests (run on CPU but exercise the same code path
    // that is taken on CUDA — verifies the view_reshape / sum_dim / cummax
    // compositions are correct).
    // -----------------------------------------------------------------------

    #[test]
    fn test_rearrange_identity_perm_is_view() {
        // Pure flatten — should hit the identity-permutation fast path and
        // share storage with the input (zero-copy view).
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 3, 2, 2]);
        let r = rearrange(&t, "b c h w -> b (c h w)").unwrap();
        assert_eq!(r.shape(), &[2, 12]);
        // Same buffer pointer indicates a zero-copy view (no materialization).
        assert!(
            std::ptr::eq(r.data().unwrap().as_ptr(), t.data().unwrap().as_ptr()),
            "expected view_reshape fast path to share storage with input"
        );
    }

    #[test]
    fn test_rearrange_pure_unflatten_is_view() {
        // Split a dim — also identity perm, also zero-copy.
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 6, 4]);
        let r = rearrange_with(&t, "b (c h) w -> b c h w", &[("c", 3)]).unwrap();
        assert_eq!(r.shape(), &[2, 3, 2, 4]);
        assert!(
            std::ptr::eq(r.data().unwrap().as_ptr(), t.data().unwrap().as_ptr()),
            "expected view_reshape fast path to share storage with input"
        );
    }

    #[test]
    fn test_reduce_sum_axis_aligned_fast_path() {
        // "b c h w -> b c" with reduced axes (h,w) contiguous at the end —
        // hits the sum_dim fast path. Verify correctness against PyTorch
        // semantics: sum over h*w within each (b,c).
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[1, 2, 3, 4]); // B=1, C=2, H=3, W=4
        let r = reduce(&t, "b c h w -> b c", EinopsReduction::Sum).unwrap();
        assert_eq!(r.shape(), &[1, 2]);
        let out = r.data().unwrap();
        // Channel 0: sum(0..12) = 66
        // Channel 1: sum(12..24) = 210
        assert!((out[0] - 66.0).abs() < 1e-5);
        assert!((out[1] - 210.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_mean_axis_aligned_fast_path() {
        // Same shape as above; verify mean = sum / N for the fast path.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[1, 2, 3, 4]);
        let r = reduce(&t, "b c h w -> b c", EinopsReduction::Mean).unwrap();
        assert_eq!(r.shape(), &[1, 2]);
        let out = r.data().unwrap();
        // Channel 0: 66 / 12 = 5.5
        // Channel 1: 210 / 12 = 17.5
        assert!((out[0] - 5.5).abs() < 1e-5);
        assert!((out[1] - 17.5).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_max_axis_aligned_fast_path() {
        // Reduced axis is contiguous — hits cummax fast path.
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0, 6.0];
        let t = leaf(&data, &[3, 2]);
        let r = reduce(&t, "b c -> c", EinopsReduction::Max).unwrap();
        assert_eq!(r.shape(), &[2]);
        let out = r.data().unwrap();
        // Reduced axis is `b`, which is left position 0, so the kept axis
        // (c) appears AFTER the reduced axis. That means kept axes are not
        // a leading prefix; depending on interpretation this may take the
        // fallback. Either way the answer must be correct.
        assert!((out[0] - 4.0).abs() < 1e-6);
        assert!((out[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_min_axis_aligned_fast_path() {
        let data = vec![1.0f32, 5.0, 3.0, 2.0, 4.0, 6.0];
        let t = leaf(&data, &[3, 2]);
        let r = reduce(&t, "b c -> c", EinopsReduction::Min).unwrap();
        assert_eq!(r.shape(), &[2]);
        let out = r.data().unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_sum_trailing_reduce_full_pool() {
        // "b c h w -> b" reduces c, h, w (all contiguous trailing axes).
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = leaf(&data, &[2, 2, 2, 3]);
        let r = reduce(&t, "b c h w -> b", EinopsReduction::Sum).unwrap();
        assert_eq!(r.shape(), &[2]);
        let out = r.data().unwrap();
        // First batch: sum 0..12 = 66; second batch: sum 12..24 = 210.
        assert!((out[0] - 66.0).abs() < 1e-5);
        assert!((out[1] - 210.0).abs() < 1e-5);
    }
}
