//! Einstein summation (`einsum`) for ferrotorch tensors.
//!
//! Supports both explicit (`"ij,jk->ik"`) and implicit (`"ij,jk"`) notation,
//! ASCII uppercase/lowercase labels, ellipsis, PyTorch-style shared-label
//! broadcasting, and one or more operands. Handles single-input operations
//! (trace, transpose, axis-sum), optimized two-input contractions via TTGT
//! (transpose-transpose-GEMM-transpose), and n-ary contractions through
//! left-to-right pairwise contraction.
//!
//! ## Device dispatch (#803)
//!
//! For CUDA inputs, the forward pass is decomposed into GPU-aware
//! sub-primitives instead of falling silently to CPU:
//!
//! * Pure permutation (e.g. `"ij->ji"`, `"abc->bca"`): zero-copy
//!   `permute_t` + on-device `contiguous_t` (uses the backend
//!   `strided_copy_*` kernel).
//! * Axis sum / projection (e.g. `"ij->i"`, `"ijk->ij"`): repeated
//!   `sum_dim` along the dropped axes.
//! * Full reduction (e.g. `"ij->"`): `grad_fns::reduction::sum`.
//! * Two-input matmul (`"ij,jk->ik"`): `grad_fns::linalg::matmul_differentiable`.
//! * Two-input batched matmul (`"bij,bjk->bik"`): `grad_fns::linalg::bmm`.
//!
//! Equations whose optimized two-input structure does not map onto the existing
//! matmul/bmm primitives fall back to a resident GPU composite over
//! `permute_t`/`reshape`/`expand`/`mul`/`sum_dim`; they do not silently
//! materialise the operands on CPU. Per `rust-gpu-discipline` §3, no silent CPU
//! detour is permitted in a non-autograd path.
//!
//! ## Repeated-index extension (#821)
//!
//! Single-input equations with repeated indices (`"ii->"` trace, `"ii->i"`
//! diagonal, `"iij->j"` mixed repeated/free projections, `"ii"` implicit
//! trace) are decomposed on-device by building a strided view that walks each
//! repeated-label diagonal while preserving free axes, then materialising it
//! through the existing `strided_copy_f{32,64}` GPU kernels (via
//! `as_strided_copy`). Reductions and permutations after diagonalisation are
//! handled by the usual resident GPU composite path. No new GPU primitive is
//! introduced; the existing CL-496 strided_copy surface is the on-device
//! decomposition target.
//!
//! ## Multi-axis 2-input extension (#822)
//!
//! Two-input contractions with multiple contracting axes or permuted
//! operand layouts (e.g. `"ijk,jkl->il"`, `"bijk,bjkl->bil"`) are handled
//! by a general permute+reshape+matmul/bmm decomposition: each operand
//! is permuted into `[batch_dims, free_dims, contract_dims]` (A) /
//! `[batch_dims, contract_dims, free_dims]` (B), reshaped to a 3-D
//! `[batch, M, K]` / `[batch, K, N]` form, contracted via `bmm`, then
//! reshaped + permuted back to the requested output layout. Equations
//! whose decomposition cannot be expressed through this route (e.g.
//! diagonal+contract combos with repeated input indices on a 2-input
//! equation) still return `Err(NotImplementedOnCuda)`.
//!
//! ## REQ status (per `.design/ferrotorch-core/einsum.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `einsum` at `einsum.rs:1517`; consumer: `Tensor::einsum` at `methods.rs:638` invokes `einsum_differentiable` |
//! | REQ-2 | SHIPPED | `parse_equation` at `einsum.rs:72`; consumer: every `einsum` call |
//! | REQ-3 | SHIPPED | `einsum_single` referenced at `einsum.rs:1531`; consumer: `Tensor::einsum` at `methods.rs:638` |
//! | REQ-4 | SHIPPED | `einsum_two` / n-ary pairwise contraction in this module; consumer: `Tensor::einsum` at `methods.rs:638` |
//! | REQ-5 | SHIPPED | `einsum_differentiable` with single/two/n-ary backward nodes; consumer: `Tensor::einsum` at `methods.rs:641` |
//! | REQ-6 | SHIPPED | `build_dim_map` at `einsum.rs:149`; consumer: every `einsum` call |
//! | REQ-7 | SHIPPED | documented in `//!` at `einsum.rs:8-24`; parity-sweep runner gap tracked by #1532 |

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::autograd::autocast_ops::autocast_guard;
use crate::autograd::no_grad::{is_grad_enabled, no_grad};
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// Equation parser
// ---------------------------------------------------------------------------

/// Parsed einsum equation.
#[derive(Debug, Clone)]
struct ParsedEquation {
    input_subscripts: Vec<Vec<char>>,
    output_subscripts: Vec<char>,
}

const ELLIPSIS_LABEL_BASE: u32 = 0xE000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubscriptToken {
    Label(char),
    Ellipsis,
}

fn ellipsis_label(index: usize) -> char {
    char::from_u32(ELLIPSIS_LABEL_BASE + index as u32)
        .expect("private-use ellipsis label must be valid Unicode")
}

fn is_ellipsis_label(c: char) -> bool {
    let code = c as u32;
    (ELLIPSIS_LABEL_BASE..ELLIPSIS_LABEL_BASE + 1024).contains(&code)
}

fn is_named_label(c: char) -> bool {
    c.is_ascii_alphabetic() || is_ellipsis_label(c)
}

fn parse_subscript_tokens(part: &str, role: &str) -> FerrotorchResult<Vec<SubscriptToken>> {
    let chars: Vec<char> = part.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '.' {
            if i + 2 < chars.len() && chars[i + 1] == '.' && chars[i + 2] == '.' {
                out.push(SubscriptToken::Ellipsis);
                i += 3;
                continue;
            }
            return Err(FerrotorchError::InvalidArgument {
                message: format!("einsum: found '.' in {role} that is not part of an ellipsis"),
            });
        }
        if !is_named_label(c) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("einsum: invalid character '{c}' in {role} subscripts"),
            });
        }
        out.push(SubscriptToken::Label(c));
        i += 1;
    }
    Ok(out)
}

fn count_explicit_labels(tokens: &[SubscriptToken]) -> usize {
    tokens
        .iter()
        .filter(|&&token| matches!(token, SubscriptToken::Label(_)))
        .count()
}

fn has_ellipsis(tokens: &[SubscriptToken]) -> bool {
    tokens
        .iter()
        .any(|&token| matches!(token, SubscriptToken::Ellipsis))
}

/// Parse an einsum equation string like `"ij,jk->ik"` or `"ij,jk"`.
fn parse_equation(equation: &str, input_shapes: &[&[usize]]) -> FerrotorchResult<ParsedEquation> {
    let n_inputs = input_shapes.len();

    let (lhs, rhs) = equation
        .split_once("->")
        .map_or((equation, None), |(l, r)| (l, Some(r)));

    // Parse input subscripts.
    let input_parts: Vec<&str> = lhs.split(',').collect();
    if input_parts.len() != n_inputs {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "einsum: equation has {} input subscripts but {} tensors were provided",
                input_parts.len(),
                n_inputs
            ),
        });
    }

    let input_tokens: Vec<Vec<SubscriptToken>> = input_parts
        .iter()
        .map(|part| parse_subscript_tokens(part, "input"))
        .collect::<FerrotorchResult<Vec<_>>>()?;

    let mut label_counts: BTreeMap<char, usize> = BTreeMap::new();
    let mut ellipsis_rank = 0usize;
    for (operand, (tokens, shape)) in input_tokens.iter().zip(input_shapes.iter()).enumerate() {
        if tokens
            .iter()
            .filter(|&&token| matches!(token, SubscriptToken::Ellipsis))
            .count()
            > 1
        {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("einsum: operand {operand} has more than one ellipsis"),
            });
        }

        let explicit = count_explicit_labels(tokens);
        let has_ell = has_ellipsis(tokens);
        if has_ell {
            if explicit > shape.len() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "einsum: input {operand} has {explicit} explicit subscripts but tensor has {} dimensions",
                        shape.len()
                    ),
                });
            }
            ellipsis_rank = ellipsis_rank.max(shape.len() - explicit);
        } else if explicit != shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "einsum: input {operand} has {explicit} subscripts but tensor has {} dimensions",
                    shape.len()
                ),
            });
        }

        for &token in tokens {
            if let SubscriptToken::Label(c) = token {
                *label_counts.entry(c).or_insert(0) += 1;
            }
        }
    }

    let input_subscripts: Vec<Vec<char>> = input_tokens
        .iter()
        .zip(input_shapes.iter())
        .map(|(tokens, shape)| {
            let explicit = count_explicit_labels(tokens);
            let covered = if has_ellipsis(tokens) {
                shape.len() - explicit
            } else {
                0
            };
            let ellipsis_start = ellipsis_rank - covered;
            let mut labels = Vec::with_capacity(shape.len());
            for &token in tokens {
                match token {
                    SubscriptToken::Label(c) => labels.push(c),
                    SubscriptToken::Ellipsis => {
                        labels.extend((ellipsis_start..ellipsis_rank).map(ellipsis_label));
                    }
                }
            }
            labels
        })
        .collect();

    let output_subscripts = if let Some(rhs) = rhs {
        let output_tokens = parse_subscript_tokens(rhs, "output")?;
        if output_tokens
            .iter()
            .filter(|&&token| matches!(token, SubscriptToken::Ellipsis))
            .count()
            > 1
        {
            return Err(FerrotorchError::InvalidArgument {
                message: "einsum: output has more than one ellipsis".into(),
            });
        }
        let mut out = Vec::new();
        for token in output_tokens {
            match token {
                SubscriptToken::Label(c) => {
                    if !label_counts.contains_key(&c) {
                        return Err(FerrotorchError::InvalidArgument {
                            message: format!(
                                "einsum: output index '{c}' does not appear in any input subscripts"
                            ),
                        });
                    }
                    out.push(c);
                }
                SubscriptToken::Ellipsis => {
                    out.extend((0..ellipsis_rank).map(ellipsis_label));
                }
            }
        }
        let mut seen = std::collections::HashSet::new();
        for &c in &out {
            if !seen.insert(c) {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "einsum: output subscript '{c}' appears more than once in the output"
                    ),
                });
            }
        }
        out
    } else {
        let mut out: Vec<char> = (0..ellipsis_rank).map(ellipsis_label).collect();
        out.extend(
            label_counts
                .into_iter()
                .filter(|&(_, count)| count == 1)
                .map(|(c, _)| c),
        );
        out
    };

    Ok(ParsedEquation {
        input_subscripts,
        output_subscripts,
    })
}

/// Re-serialize a [`ParsedEquation`] into canonical explicit form:
/// whitespace-free, comma-joined input subscripts, and an explicit
/// `->output` (for implicit-mode equations the output is the
/// forward-computed sorted once-occurring labels).
///
/// `einsum_differentiable` stores THIS string in the grad-fns rather than
/// the user's raw equation, so the backward passes never re-parse
/// whitespace (CORE-163 / #1857) and never mistake an implicit output for
/// an empty one (CORE-164 / #1858).
fn canonical_equation(parsed: &ParsedEquation) -> String {
    let lhs: Vec<String> = parsed
        .input_subscripts
        .iter()
        .map(|subs| subs.iter().collect())
        .collect();
    let rhs: String = parsed.output_subscripts.iter().collect();
    format!("{}->{rhs}", lhs.join(","))
}

// ---------------------------------------------------------------------------
// Dimension map: index char -> size
// ---------------------------------------------------------------------------

/// Build a map from index character to its dimension size, validating consistency.
fn build_dim_map<T: Float>(
    parsed: &ParsedEquation,
    inputs: &[&Tensor<T>],
) -> FerrotorchResult<BTreeMap<char, usize>> {
    let mut dim_map: BTreeMap<char, usize> = BTreeMap::new();

    for (i, (subs, tensor)) in parsed
        .input_subscripts
        .iter()
        .zip(inputs.iter())
        .enumerate()
    {
        if subs.len() != tensor.ndim() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "einsum: input {} has {} subscripts but tensor has {} dimensions",
                    i,
                    subs.len(),
                    tensor.ndim()
                ),
            });
        }
        let mut local_sizes: BTreeMap<char, usize> = BTreeMap::new();
        for (axis, &c) in subs.iter().enumerate() {
            let size = tensor.shape()[axis];
            if let Some(&existing) = local_sizes.get(&c) {
                if existing != size {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "einsum: repeated index '{c}' in operand {i} has inconsistent sizes: {existing} vs {size}"
                        ),
                    });
                }
                continue;
            }
            local_sizes.insert(c, size);
        }

        for (c, size) in local_sizes {
            match dim_map.get(&c).copied() {
                Some(existing) if existing == size => {}
                Some(existing) if size == 1 => {
                    dim_map.insert(c, existing);
                }
                Some(1) | None => {
                    dim_map.insert(c, size);
                }
                Some(existing) => {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "einsum: index '{c}' has size {size} for operand {i} which does not broadcast with previously seen size {existing}"
                        ),
                    });
                }
            }
        }
    }

    // Validate output subscripts reference known indices.
    for &c in &parsed.output_subscripts {
        if !dim_map.contains_key(&c) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "einsum: output index '{c}' does not appear in any input subscripts"
                ),
            });
        }
    }

    Ok(dim_map)
}

fn validate_same_device<T: Float>(op: &str, inputs: &[&Tensor<T>]) -> FerrotorchResult<()> {
    let Some(first) = inputs.first() else {
        return Ok(());
    };
    let expected = first.device();
    for tensor in inputs.iter().skip(1) {
        if tensor.device() != expected {
            return Err(FerrotorchError::DeviceMismatch {
                expected,
                got: tensor.device(),
            });
        }
    }
    if expected.is_xpu() || expected.is_mps() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: device {expected} is not implemented for einsum"),
        });
    }
    Ok(())
}

fn needs_operand_broadcast<T: Float>(
    parsed: &ParsedEquation,
    inputs: &[&Tensor<T>],
    dim_map: &BTreeMap<char, usize>,
) -> bool {
    parsed
        .input_subscripts
        .iter()
        .zip(inputs.iter())
        .any(|(subs, tensor)| {
            subs.iter()
                .enumerate()
                .any(|(axis, &c)| tensor.shape()[axis] != dim_map[&c])
        })
}

fn chars_to_string(chars: &[char]) -> String {
    chars.iter().collect()
}

fn first_occurrence_union(lhs: &[char], rhs: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(lhs.len() + rhs.len());
    for &c in lhs.iter().chain(rhs.iter()) {
        if !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

fn nary_pair_output_subscripts(
    lhs: &[char],
    rhs: &[char],
    remaining: &[Vec<char>],
    final_output: &[char],
) -> Vec<char> {
    first_occurrence_union(lhs, rhs)
        .into_iter()
        .filter(|c| final_output.contains(c) || remaining.iter().any(|subs| subs.contains(c)))
        .collect()
}

fn full_product_order(parsed: &ParsedEquation, dim_map: &BTreeMap<char, usize>) -> Vec<char> {
    let mut order = parsed.output_subscripts.clone();
    for &c in dim_map.keys() {
        if !order.contains(&c) {
            order.push(c);
        }
    }
    order
}

fn align_operand_for_product<T: Float>(
    subs: &[char],
    tensor: &Tensor<T>,
    full_order: &[char],
    full_shape: &[usize],
) -> FerrotorchResult<Tensor<T>> {
    let (dedup_subs, diagonalised) = diagonalize_repeats_gpu(subs, tensor)?;
    let present: Vec<char> = full_order
        .iter()
        .copied()
        .filter(|c| dedup_subs.contains(c))
        .collect();

    let ordered = if present == dedup_subs || present.is_empty() {
        diagonalised
    } else {
        let perm: Vec<usize> = present
            .iter()
            .map(|c| {
                dedup_subs
                    .iter()
                    .position(|dc| dc == c)
                    .ok_or_else(|| FerrotorchError::Internal {
                        message: format!("einsum product align: missing operand label '{c}'"),
                    })
            })
            .collect::<FerrotorchResult<Vec<_>>>()?;
        let view = crate::methods::permute_t(&diagonalised, &perm)?;
        crate::methods::contiguous_t(&view)?
    };

    let mut reshape_shape = Vec::with_capacity(full_order.len());
    for &c in full_order {
        if let Some(axis) = present.iter().position(|&pc| pc == c) {
            reshape_shape.push(ordered.shape()[axis] as isize);
        } else {
            reshape_shape.push(1);
        }
    }
    let reshaped = crate::grad_fns::shape::reshape(&ordered, &reshape_shape)?;
    if reshaped.shape() == full_shape {
        Ok(reshaped)
    } else {
        crate::grad_fns::shape::expand(&reshaped, full_shape)
    }
}

fn einsum_product_composite<T: Float>(
    parsed: &ParsedEquation,
    inputs: &[&Tensor<T>],
    dim_map: &BTreeMap<char, usize>,
) -> FerrotorchResult<Tensor<T>> {
    validate_same_device("einsum", inputs)?;
    no_grad(|| {
        let full_order = full_product_order(parsed, dim_map);
        let full_shape: Vec<usize> = full_order.iter().map(|c| dim_map[c]).collect();
        let mut aligned = Vec::with_capacity(inputs.len());
        for (subs, tensor) in parsed.input_subscripts.iter().zip(inputs.iter()) {
            aligned.push(align_operand_for_product(
                subs,
                tensor,
                &full_order,
                &full_shape,
            )?);
        }

        let mut product =
            aligned
                .first()
                .cloned()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: "einsum: expected at least one input tensor".into(),
                })?;
        for operand in aligned.iter().skip(1) {
            product = crate::grad_fns::arithmetic::mul(&product, operand)?;
        }

        let mut reduced = product;
        for axis in (parsed.output_subscripts.len()..full_order.len()).rev() {
            reduced = crate::grad_fns::reduction::sum_dim(&reduced, axis as i64, false)?;
        }
        crate::methods::contiguous_t(&reduced)
    })
}

// ---------------------------------------------------------------------------
// Single-input einsum (trace, transpose, axis-sum, diagonal)
// ---------------------------------------------------------------------------

/// GPU dispatch for single-input einsum (#803, extended in #821).
///
/// Decomposes the equation into GPU-aware primitives:
///
/// * Pure permutation (set(in)==set(out), no repeated input indices):
///   `permute_t` (zero-copy stride view) + `contiguous_t` (on-device
///   strided_copy kernel).
/// * Axis sum / projection (set(out) ⊊ set(in), no repeated input
///   indices): repeated `sum_dim` along the dropped axes; if there is
///   also a permutation among the kept axes, a final `permute_t`.
/// * Full reduction (out empty, no repeats): `grad_fns::reduction::sum`.
/// * Repeated input indices ("ii->", "ii->i", "ii"): on-device diagonal
///   extraction via `as_strided_copy` (shape `[N]`, stride `[N+1]`),
///   then `sum_dim` for the trace case. Implemented via the existing
///   `strided_copy_f{32,64}` kernel — no new primitive surface.
fn einsum_single_gpu<T: Float>(
    parsed: &ParsedEquation,
    input: &Tensor<T>,
    dim_map: &BTreeMap<char, usize>,
) -> FerrotorchResult<Tensor<T>> {
    let in_subs = &parsed.input_subscripts[0];
    let out_subs = &parsed.output_subscripts;

    // Repeated input indices: extract the diagonal on-device via the
    // existing strided_copy kernel, then optionally reduce. Implemented
    // as a composite over existing primitives — no new GPU surface.
    if has_duplicate_chars(in_subs) {
        return einsum_single_repeated_gpu(in_subs, out_subs, input, dim_map);
    }

    // Output indices must all appear in the input (caller has already
    // validated this in `build_dim_map`, so this is just a safety check).
    for &c in out_subs {
        if !in_subs.contains(&c) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "einsum: output index '{c}' does not appear in any input subscripts"
                ),
            });
        }
    }

    // Disable autograd inside the composite — `einsum_differentiable`
    // attaches the autograd node *outside* the forward call. The
    // sub-ops we call here (sum_dim, matmul_differentiable, etc.) would
    // otherwise build their own grad_fn chain that double-counts the
    // gradient.
    no_grad(|| {
        // Step 1: sum out any axes whose chars do not appear in the
        // output. We sum from the highest dim downward so each removal
        // shifts only the dims after it (already removed) — it does not
        // shift the index of any dim still queued for removal.
        let mut keep_chars: Vec<char> = in_subs.clone();
        let mut current = input.clone();
        let mut axis = in_subs.len();
        for &c in in_subs.iter().rev() {
            axis -= 1;
            if !out_subs.contains(&c) {
                current = crate::grad_fns::reduction::sum_dim(&current, axis as i64, false)?;
                keep_chars.remove(axis);
            }
        }

        // Step 2: if the remaining chars are not in the same order as
        // out_subs, permute. After step 1, `keep_chars` and `out_subs`
        // are the same set of distinct chars (since in_subs has no
        // duplicates and we kept exactly those that appear in out_subs).
        if keep_chars == *out_subs {
            // Already in the right order — make sure the result is
            // contiguous (sum_dim returns contiguous; pure no-op
            // single-axis cases need no permute).
            let out_shape: Vec<usize> = out_subs.iter().map(|c| dim_map[c]).collect();
            // Verify the shape is what we expect; if not, that's an
            // internal-bug condition.
            if current.shape() != out_shape.as_slice() {
                return Err(FerrotorchError::Internal {
                    message: format!(
                        "einsum_single_gpu: shape mismatch after reduction: got {:?} expected {:?}",
                        current.shape(),
                        out_shape
                    ),
                });
            }
            return Ok(current);
        }

        // Compute the permutation: for each output position, find the
        // current axis position of that char in keep_chars.
        let perm: Vec<usize> = out_subs
            .iter()
            .map(|c| {
                keep_chars
                    .iter()
                    .position(|kc| kc == c)
                    .expect("out_subs char must exist in keep_chars (validated above)")
            })
            .collect();
        // permute_t produces a stride-view; materialise it via
        // contiguous_t so the caller gets a fresh on-device buffer
        // (same semantics as the CPU path which always returns a
        // freshly allocated row-major result).
        let permuted = crate::methods::permute_t(&current, &perm)?;
        let materialised = crate::methods::contiguous_t(&permuted)?;
        Ok(materialised)
    })
}

/// GPU implementation of repeated-index single-input einsum (#821, #824).
///
/// Handles the patterns where one or more input indices repeat — i.e.
/// `"ii->"` (trace), `"ii->i"` (diagonal extraction), `"iij->j"` and
/// `"ijki->jk"` (mixed repeated/free projections), `"ii"` (implicit trace).
/// The decomposition is purely composite over existing GPU primitives:
///
/// 1. Construct a strided view that selects only the positions where the
///    repeated indices coincide. For `"ii"` over an `[N, N]` tensor this
///    is shape `[N]` with stride `[N+1]` over the underlying contiguous
///    storage. Generalises to `iii…` of rank `r` over an `[N, N, ..., N]`
///    tensor as shape `[N]` with stride `[1 + N + N^2 + … + N^{r-1}]`.
/// 2. Materialise the view via `as_strided_copy`, which dispatches to the
///    existing `strided_copy_f{32,64}` GPU kernel (CL-496) on CUDA — no
///    host bounce, no new kernel.
/// 3. If the output is empty (`"ii->"` or implicit `"ii"`), reduce the
///    `[N]` diagonal vector with `sum_dim`. Otherwise return the diagonal
///    directly (e.g. `"ii->i"` produces an `[N]` vector).
///
/// Restrictions: repeated output labels remain invalid (`"ii->ii"` is not
/// a legal PyTorch einsum). Mixed repeated/free inputs are handled by the
/// same diagonalisation helper; unsupported structures must error rather
/// than silently producing CPU results.
fn einsum_single_repeated_gpu<T: Float>(
    in_subs: &[char],
    out_subs: &[char],
    input: &Tensor<T>,
    dim_map: &BTreeMap<char, usize>,
) -> FerrotorchResult<Tensor<T>> {
    if in_subs.len() < 2 {
        // A single index can't repeat with itself within rank 1 — shouldn't reach here.
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "einsum_repeated_index",
        });
    }

    // Build the diagonalised tensor + new (deduped) subscript list. After
    // this every char in `new_subs` appears exactly once and `diag` is a
    // freshly materialised on-device tensor whose layout corresponds to
    // those subs in row-major order. Implements both the homogeneous
    // diagonal/trace patterns from #821 (`"ii"`, `"iii"`, `"iii->i"`) and
    // the mixed repeated/free patterns added in #824 (`"iij->j"`,
    // `"iji->j"`, `"iijk->jk"`, `"iij->ij"`, etc.) through the same
    // `as_strided_copy` (GPU `strided_copy_f{32,64}`) primitive.
    let (new_subs, diag) = diagonalize_repeats_gpu(in_subs, input)?;

    // Output may not introduce chars absent from the deduped input subs.
    // `build_dim_map` already validates this upstream, but we recheck
    // so this branch is independently safe against direct callers.
    for &c in out_subs {
        if !new_subs.contains(&c) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "einsum: output index '{c}' does not appear in any input subscripts"
                ),
            });
        }
    }

    // Recurse into the standard single-input GPU dispatch with the
    // diagonalised tensor + deduped subs. `einsum_single_gpu` will see
    // no repeats and route through the sum-axes-then-permute path —
    // every step on-device.
    let new_parsed = ParsedEquation {
        input_subscripts: vec![new_subs],
        output_subscripts: out_subs.to_vec(),
    };
    einsum_single_gpu(&new_parsed, &diag, dim_map)
}

/// Diagonalise repeated input chars on a GPU (or CPU) tensor via a single
/// `as_strided_copy` (#821 / #824 / #825 shared machinery).
///
/// Given an input tensor and its subscript list (which may contain
/// repeated chars, e.g. `"iij"` or `"iji"`), this builds a strided view
/// that walks the diagonal across each repeat-class while preserving free
/// axes, then materialises that view through the existing
/// `strided_copy_f{32,64}` kernel on CUDA (or the CPU walker on host) —
/// no host bounce, no new primitive surface.
///
/// The returned subscript list contains each character exactly once, in
/// the order of its first appearance in `in_subs`. For each output axis
/// the corresponding stride is the *sum* of the original strides of all
/// input axes carrying that char — stepping along the new axis advances
/// every original axis with that char in lock-step, which is the
/// definition of a generalised diagonal.
///
/// Examples (assume row-major input strides):
///
/// * `"ii"` over `[N, N]` (strides `[N, 1]`) → subs `"i"`, view shape
///   `[N]`, stride `[N + 1]`. Standard 2-D diagonal.
/// * `"iij"` over `[N, N, M]` (strides `[N*M, M, 1]`) → subs `"ij"`,
///   view shape `[N, M]`, strides `[N*M + M, 1]`.
/// * `"iji"` over `[N, M, N]` (strides `[M*N, N, 1]`) → subs `"ij"`,
///   view shape `[N, M]`, strides `[M*N + 1, N]`. Free index between
///   the two repeats — its stride is preserved.
/// * `"iii"` over `[N, N, N]` (strides `[N*N, N, 1]`) → subs `"i"`,
///   view shape `[N]`, stride `[N*N + N + 1]`. Homogeneous case from
///   #821 — covered by the same code path.
///
/// If `in_subs` contains no repeated chars the input is returned
/// unchanged (with a clone of the subs vector) so callers can use this
/// as an unconditional pre-pass.
fn diagonalize_repeats_gpu<T: Float>(
    in_subs: &[char],
    input: &Tensor<T>,
) -> FerrotorchResult<(Vec<char>, Tensor<T>)> {
    if !has_duplicate_chars(in_subs) {
        return Ok((in_subs.to_vec(), input.clone()));
    }

    // Use the input's existing strides directly — `as_strided` views the
    // underlying storage at `storage_offset` with whatever strides we
    // hand it, so we don't need to materialise a contiguous copy first.
    let in_strides = input.strides();
    let in_shape = input.shape();
    if in_strides.len() != in_subs.len() || in_shape.len() != in_subs.len() {
        return Err(FerrotorchError::Internal {
            message: format!(
                "diagonalize_repeats_gpu: subs/shape/strides length mismatch: \
                 {} vs {} vs {}",
                in_subs.len(),
                in_shape.len(),
                in_strides.len()
            ),
        });
    }

    // Walk `in_subs` in order, collecting unique chars (preserving first-
    // occurrence order) and accumulating the collapsed stride per char.
    let mut new_subs: Vec<char> = Vec::with_capacity(in_subs.len());
    let mut new_sizes: Vec<usize> = Vec::with_capacity(in_subs.len());
    let mut new_strides: Vec<isize> = Vec::with_capacity(in_subs.len());
    for (axis, &c) in in_subs.iter().enumerate() {
        if let Some(pos) = new_subs.iter().position(|&nc| nc == c) {
            // Repeat: validate consistent size, accumulate stride.
            if new_sizes[pos] != in_shape[axis] {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "einsum: repeated index '{c}' addresses incompatible sizes \
                         {} vs {}",
                        new_sizes[pos], in_shape[axis]
                    ),
                });
            }
            let add = in_strides[axis];
            new_strides[pos] = new_strides[pos].checked_add(add).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: "einsum diagonalisation: stride sum overflowed".into(),
                }
            })?;
        } else {
            // First sighting: introduce as new axis.
            new_subs.push(c);
            new_sizes.push(in_shape[axis]);
            new_strides.push(in_strides[axis]);
        }
    }

    // `as_strided` is metadata-only on every device. `as_strided_copy`
    // materialises through the existing GPU `strided_copy_f{32,64}`
    // kernel for CUDA tensors and the CPU walker for host tensors —
    // both already on-device-correct. We pass `None` for the storage
    // offset so the new view inherits `input`'s offset.
    let view = input.as_strided(&new_sizes, &new_strides, None)?;
    let materialised = view.as_strided_copy(&new_sizes, &new_strides, None)?;
    Ok((new_subs, materialised))
}

/// Returns `true` if `chars` contains any character more than once.
fn has_duplicate_chars(chars: &[char]) -> bool {
    let mut seen = std::collections::HashSet::new();
    for &c in chars {
        if !seen.insert(c) {
            return true;
        }
    }
    false
}

fn einsum_single<T: Float>(
    parsed: &ParsedEquation,
    input: &Tensor<T>,
    dim_map: &BTreeMap<char, usize>,
) -> FerrotorchResult<Tensor<T>> {
    // GPU-aware dispatch (#803): decompose into GPU primitives where
    // possible instead of falling to CPU.
    if input.is_cuda() {
        return einsum_single_gpu(parsed, input, dim_map);
    }

    let in_subs = &parsed.input_subscripts[0];
    let out_subs = &parsed.output_subscripts;

    // Compute output shape.
    let out_shape: Vec<usize> = out_subs.iter().map(|c| dim_map[c]).collect();
    let out_numel: usize = if out_shape.is_empty() {
        1
    } else {
        crate::shape::numel(&out_shape)
    };

    let data = input.data_vec()?;
    let in_shape = input.shape();

    // General approach: iterate over all output index combinations plus all
    // summed-over index combinations. For each, accumulate the product.
    //
    // Summed indices: indices in input but not in output.
    let summed_indices: Vec<char> = in_subs
        .iter()
        .filter(|c| !out_subs.contains(c))
        .copied()
        .collect::<Vec<_>>();
    // Deduplicate (a repeated index like "ii" means diagonal/trace).
    let summed_unique: Vec<char> = {
        let mut v = summed_indices.clone();
        v.sort_unstable();
        v.dedup();
        // But we need to include only indices not in output.
        v.into_iter().filter(|c| !out_subs.contains(c)).collect()
    };

    // Compute strides for the input tensor (row-major).
    let in_strides: Vec<usize> = {
        let mut strides = vec![1usize; in_shape.len()];
        for i in (0..in_shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * in_shape[i + 1];
        }
        strides
    };

    // Compute ranges for summed indices.
    let summed_sizes: Vec<usize> = summed_unique.iter().map(|c| dim_map[c]).collect();
    let summed_numel: usize = if summed_sizes.is_empty() {
        1
    } else {
        crate::shape::numel(&summed_sizes)
    };

    let mut result = vec![<T as num_traits::Zero>::zero(); out_numel];

    // For each output element...
    for (out_idx, result_elem) in result.iter_mut().enumerate() {
        // Decode output multi-index.
        let mut out_multi = vec![0usize; out_subs.len()];
        {
            let mut remainder = out_idx;
            for i in (0..out_subs.len()).rev() {
                let size = dim_map[&out_subs[i]];
                out_multi[i] = remainder % size;
                remainder /= size;
            }
        }

        // Build a map from char -> value for the output indices.
        let mut idx_vals: BTreeMap<char, usize> = BTreeMap::new();
        for (i, &c) in out_subs.iter().enumerate() {
            idx_vals.insert(c, out_multi[i]);
        }

        let mut acc = <T as num_traits::Zero>::zero();

        // Iterate over summed indices.
        for s_idx in 0..summed_numel {
            let mut remainder = s_idx;
            let mut valid = true;
            for i in (0..summed_unique.len()).rev() {
                let val = remainder % summed_sizes[i];
                remainder /= summed_sizes[i];
                idx_vals.insert(summed_unique[i], val);
            }

            // Check consistency for repeated indices (e.g., "ii"):
            // If a char appears more than once in input subscripts, all
            // corresponding axis values must match.
            // For repeated input indices, enforce equality.
            let mut first_occurrence: BTreeMap<char, Option<usize>> = BTreeMap::new();
            for &c in in_subs {
                let val = idx_vals[&c];
                match first_occurrence.get(&c) {
                    Some(Some(prev_val)) => {
                        if *prev_val != val {
                            valid = false;
                            break;
                        }
                    }
                    _ => {
                        first_occurrence.insert(c, Some(val));
                    }
                }
            }

            if !valid {
                continue;
            }

            // Compute flat index into input.
            let mut flat_idx = 0usize;
            for (axis, &c) in in_subs.iter().enumerate() {
                flat_idx += idx_vals[&c] * in_strides[axis];
            }

            acc += data[flat_idx];
        }

        *result_elem = acc;
    }

    Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)
}

// ---------------------------------------------------------------------------
// Two-input einsum via TTGT
// ---------------------------------------------------------------------------

/// GPU dispatch for two-input einsum (#803, extended in #822).
///
/// Maps contraction patterns onto existing GPU primitives:
///
/// * 2D matmul `"ij,jk->ik"` (and any equivalent re-letter):
///   `grad_fns::linalg::matmul_differentiable` (forward-only via `no_grad`).
/// * Batched matmul `"bij,bjk->bik"` (and any equivalent re-letter):
///   `grad_fns::linalg::bmm`.
/// * Vector / Hadamard / outer / matrix-vector special cases (1-D operands).
/// * **General multi-axis decomposition (#822):** any equation whose
///   indices partition cleanly into `batch / free_a / free_b / contract`
///   sets — including multi-axis contractions like `"ijk,jkl->il"` and
///   permuted operand or output layouts. Each operand is permuted into
///   `[batch, free, contract]` (A) / `[batch, contract, free]` (B),
///   reshaped to 3-D, contracted with `bmm`, then reshaped + permuted
///   back to the requested output layout.
///
/// Operands with repeated input chars (e.g. `"ii,j->j"`, `"ij,jj->i"`,
/// `"ii,jk->jk"`) are handled by a pre-pass that diagonalises each
/// offending operand on-device via [`diagonalize_repeats_gpu`] —
/// replacing `"ii"` with `"i"`, `"jj"` with `"j"`, etc. — before falling
/// into the general permute+matmul+reshape decomposition (#825).
fn einsum_two_gpu<T: Float>(
    parsed: &ParsedEquation,
    a: &Tensor<T>,
    b: &Tensor<T>,
    dim_map: &BTreeMap<char, usize>,
) -> FerrotorchResult<Tensor<T>> {
    let a_subs_orig = &parsed.input_subscripts[0];
    let b_subs_orig = &parsed.input_subscripts[1];
    let out_subs = &parsed.output_subscripts;

    // Pre-pass for #825: if either operand carries repeated input chars
    // (e.g. `"ii,j->j"` or `"ij,jj->i"`), diagonalise that operand on-
    // device first so every remaining char is distinct within each
    // operand. The diagonalisation reuses the same `as_strided_copy`
    // (GPU `strided_copy_f{32,64}`) machinery introduced for #821/#824 —
    // no new primitive surface, no host bounce. After the pre-pass,
    // every downstream branch (matmul, bmm, vector/Hadamard/outer
    // shortcuts, the general permute+bmm decomposition) sees operands
    // with no repeated chars and behaves exactly as it did pre-#825.
    let (a_subs_owned, a_diagonalised) = diagonalize_repeats_gpu(a_subs_orig, a)?;
    let (b_subs_owned, b_diagonalised) = diagonalize_repeats_gpu(b_subs_orig, b)?;
    let a_subs = &a_subs_owned;
    let b_subs = &b_subs_owned;
    let a = &a_diagonalised;
    let b = &b_diagonalised;

    // Safety net: after the pre-pass neither operand should still carry
    // repeats. If that ever changes we want a structured error rather
    // than wrong values.
    if has_duplicate_chars(a_subs) || has_duplicate_chars(b_subs) {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "einsum_repeated_index",
        });
    }

    no_grad(|| {
        // Generalised 2D x 2D contraction with a single contracted
        // index. Covers `"ij,jk->ik"` (the canonical matmul) AND its
        // backward-derived siblings `"ik,jk->ij"` (= A @ B^T) and
        // `"ij,ik->jk"` (= A^T @ B), which `EinsumBackwardTwo`
        // generates from a forward `"ij,jk->ik"` matmul. The dispatch
        // identifies the contracted char, transposes operands as
        // needed (zero-copy via `permute_t`+`contiguous_t`), calls
        // `matmul_differentiable` (GPU 2D x 2D), then permutes the
        // result to match `out_subs` order.
        if a_subs.len() == 2
            && b_subs.len() == 2
            && out_subs.len() == 2
            && a_subs[0] != a_subs[1]
            && b_subs[0] != b_subs[1]
            && out_subs[0] != out_subs[1]
        {
            // Find the contracted char: in both a_subs and b_subs but
            // not in out_subs.
            let contracted: Option<char> = a_subs
                .iter()
                .copied()
                .find(|c| b_subs.contains(c) && !out_subs.contains(c));
            if let Some(c) = contracted {
                // The other chars: one from A (= a_other), one from B
                // (= b_other). Both must appear in out_subs.
                let a_other = if a_subs[0] == c { a_subs[1] } else { a_subs[0] };
                let b_other = if b_subs[0] == c { b_subs[1] } else { b_subs[0] };
                if a_other != b_other && out_subs.contains(&a_other) && out_subs.contains(&b_other)
                {
                    // Position the contracted dim: A wants c at axis 1, B at axis 0.
                    let a_oriented = if a_subs[1] == c {
                        a.clone()
                    } else {
                        let permuted = crate::methods::permute_t(a, &[1, 0])?;
                        crate::methods::contiguous_t(&permuted)?
                    };
                    let b_oriented = if b_subs[0] == c {
                        b.clone()
                    } else {
                        let permuted = crate::methods::permute_t(b, &[1, 0])?;
                        crate::methods::contiguous_t(&permuted)?
                    };
                    let mm =
                        crate::grad_fns::linalg::matmul_differentiable(&a_oriented, &b_oriented)?;
                    // mm has shape [a_other_size, b_other_size]; permute
                    // if out_subs order doesn't match.
                    if out_subs[0] == a_other && out_subs[1] == b_other {
                        return Ok(mm);
                    }
                    let permuted = crate::methods::permute_t(&mm, &[1, 0])?;
                    return crate::methods::contiguous_t(&permuted);
                }
            }
        }

        // Generalised 3D batched-matmul pattern: a has 3 distinct
        // chars [bat, p, q], b has 3 distinct chars [bat, p2, q2], one
        // of which equals bat (the batch char shared with a), and one
        // of the other two is the contracted index. out has [bat, X, Y]
        // where X and Y are the non-batch, non-contracted chars from
        // a and b respectively (in some order). Covers the canonical
        // `"bij,bjk->bik"` AND its backward siblings
        // `"bik,bjk->bij"` (bmm of A and B^T) and
        // `"bij,bik->bjk"` (bmm of A^T and B), which
        // `EinsumBackwardTwo` generates from a forward bmm.
        if a_subs.len() == 3
            && b_subs.len() == 3
            && out_subs.len() == 3
            && a_subs[0] == b_subs[0]
            && a_subs[0] == out_subs[0]
        {
            let bat = a_subs[0];
            // Distinct chars within each operand, batch char only at leading position.
            let a_uniq = a_subs[0] != a_subs[1] && a_subs[1] != a_subs[2] && a_subs[0] != a_subs[2];
            let b_uniq = b_subs[0] != b_subs[1] && b_subs[1] != b_subs[2] && b_subs[0] != b_subs[2];
            if a_uniq
                && b_uniq
                && bat != out_subs[1]
                && bat != out_subs[2]
                && out_subs[1] != out_subs[2]
            {
                // Find the contracted char: in a (excluding bat) and b
                // (excluding bat) but not in out.
                let a_non_batch = [a_subs[1], a_subs[2]];
                let b_non_batch = [b_subs[1], b_subs[2]];
                let contracted: Option<char> = a_non_batch
                    .iter()
                    .copied()
                    .find(|c| b_non_batch.contains(c) && !out_subs.contains(c));
                if let Some(c) = contracted {
                    let a_other = if a_subs[1] == c { a_subs[2] } else { a_subs[1] };
                    let b_other = if b_subs[1] == c { b_subs[2] } else { b_subs[1] };
                    if a_other != b_other
                        && out_subs.contains(&a_other)
                        && out_subs.contains(&b_other)
                    {
                        // Want A oriented as [bat, a_other, c] (contracted
                        // dim at axis 2). If A is [bat, c, a_other], swap
                        // axes 1 and 2.
                        let a_oriented = if a_subs[2] == c {
                            a.clone()
                        } else {
                            let permuted = crate::methods::permute_t(a, &[0, 2, 1])?;
                            crate::methods::contiguous_t(&permuted)?
                        };
                        // Want B oriented as [bat, c, b_other] (contracted
                        // dim at axis 1).
                        let b_oriented = if b_subs[1] == c {
                            b.clone()
                        } else {
                            let permuted = crate::methods::permute_t(b, &[0, 2, 1])?;
                            crate::methods::contiguous_t(&permuted)?
                        };
                        let result = crate::grad_fns::linalg::bmm(&a_oriented, &b_oriented)?;
                        // result shape: [bat, a_other, b_other]. Permute
                        // if out_subs has them in the opposite order.
                        if out_subs[1] == a_other && out_subs[2] == b_other {
                            return Ok(result);
                        }
                        let permuted = crate::methods::permute_t(&result, &[0, 2, 1])?;
                        return crate::methods::contiguous_t(&permuted);
                    }
                }
            }
        }

        // Hadamard / elementwise pattern: a_subs == b_subs == out_subs
        // (e.g. "ij,ij->ij"). On the algebra level this is just a *
        // b. `mul` is GPU-aware (broadcast_mul kernel).
        if a_subs == b_subs && b_subs.as_slice() == out_subs.as_slice() {
            return crate::grad_fns::arithmetic::mul(a, b);
        }

        // Dot product pattern: a_subs == b_subs (both 1D, same single
        // char) and out_subs is empty. e.g. "i,i->" or implicit "i,i".
        if a_subs.len() == 1 && b_subs.as_slice() == a_subs.as_slice() && out_subs.is_empty() {
            let prod = crate::grad_fns::arithmetic::mul(a, b)?;
            return crate::grad_fns::reduction::sum(&prod);
        }

        // Outer product pattern: a_subs and b_subs are both 1D with
        // distinct chars, out_subs is `a_subs ++ b_subs`. e.g. "i,j->ij".
        if a_subs.len() == 1
            && b_subs.len() == 1
            && a_subs[0] != b_subs[0]
            && out_subs.len() == 2
            && out_subs[0] == a_subs[0]
            && out_subs[1] == b_subs[0]
        {
            // a: [m] -> [m, 1]; b: [n] -> [1, n]; broadcast_mul -> [m, n].
            let a_unsq = crate::grad_fns::shape::unsqueeze(a, 1)?;
            let b_unsq = crate::grad_fns::shape::unsqueeze(b, 0)?;
            return crate::grad_fns::arithmetic::mul(&a_unsq, &b_unsq);
        }

        // Scalar-broadcast vector pattern: a is empty subs (scalar)
        // and b is 1D, out matches b. e.g. ",i->i". Generated by
        // EinsumBackwardTwo for grad_a of dot ("i,i->" backward).
        if a_subs.is_empty() && b_subs.len() == 1 && out_subs.as_slice() == b_subs.as_slice() {
            // mul broadcasts scalar a against vector b.
            return crate::grad_fns::arithmetic::mul(a, b);
        }
        // Symmetric: vector × scalar. e.g. "i,->i".
        if b_subs.is_empty() && a_subs.len() == 1 && out_subs.as_slice() == a_subs.as_slice() {
            return crate::grad_fns::arithmetic::mul(a, b);
        }

        // Matrix-vector pattern: a is 2D [I,J], b is 1D [J], out is
        // 1D [I]. e.g. "ij,j->i". Generated by EinsumBackwardTwo for
        // grad_a of outer ("i,j->ij" backward). Implemented via
        // matmul_differentiable on (a, b.unsqueeze(1)) followed by
        // squeeze — matmul is GPU-aware for 2D x 2D, and the unsqueeze/
        // squeeze are zero-copy stride views.
        if a_subs.len() == 2
            && b_subs.len() == 1
            && out_subs.len() == 1
            && a_subs[1] == b_subs[0]
            && a_subs[0] == out_subs[0]
            && a_subs[0] != a_subs[1]
        {
            let b_unsq = crate::grad_fns::shape::unsqueeze(b, 1)?; // [J] -> [J,1]
            let mm_result = crate::grad_fns::linalg::matmul_differentiable(a, &b_unsq)?; // [I,1]
            return crate::grad_fns::shape::squeeze(&mm_result, 1);
        }

        // Vector-matrix pattern: a is 1D [I], b is 2D [I,J], out is
        // 1D [J]. e.g. "i,ij->j". Generated by EinsumBackwardTwo for
        // grad_b of outer.
        if a_subs.len() == 1
            && b_subs.len() == 2
            && out_subs.len() == 1
            && a_subs[0] == b_subs[0]
            && b_subs[1] == out_subs[0]
            && b_subs[0] != b_subs[1]
        {
            let a_unsq = crate::grad_fns::shape::unsqueeze(a, 0)?; // [I] -> [1,I]
            let mm_result = crate::grad_fns::linalg::matmul_differentiable(&a_unsq, b)?; // [1,J]
            return crate::grad_fns::shape::squeeze(&mm_result, 0);
        }

        // General permute+reshape+bmm decomposition (#822).
        // Falls back to NotImplementedOnCuda if the equation can't be
        // expressed through this route (e.g. mixed input repeats or chars
        // missing from the dim_map).
        einsum_two_gpu_general(a_subs, b_subs, out_subs, a, b, dim_map)
    })
}

/// General GPU decomposition for 2-input einsum (#822).
///
/// Strategy: classify each unique char into one of four buckets —
/// `batch` (in A, B, and out), `free_a` (in A and out, not in B),
/// `free_b` (in B and out, not in A), `contract` (in A and B, not in out).
/// Then:
///
/// 1. Permute A so its axes are `[batch..., free_a..., contract...]`.
/// 2. Permute B so its axes are `[batch..., contract..., free_b...]`.
/// 3. Reshape both to 3-D `[batch_total, M, K]` and `[batch_total, K, N]`.
/// 4. Call `bmm` → `[batch_total, M, N]`.
/// 5. Reshape back to `[batch..., free_a..., free_b...]`.
/// 6. Permute to match `out_subs` order.
///
/// Equations with chars that appear only in one input but not in the
/// output (lone-summed indices, e.g. `"ijk,kl->jl"` where `i` is only in
/// A) are handled by an `axis_sum` pre-pass on the offending operand
/// before the four-way classification.
///
/// If any operand has repeated input chars (e.g. `"iij,j->j"`) this path
/// declines with `Err(NotImplementedOnCuda { op: "einsum_general" })` —
/// these cases need diagonal extraction on the operand first, which is
/// out of scope here and tracked as a sub-cascade.
fn einsum_two_gpu_general<T: Float>(
    a_subs: &[char],
    b_subs: &[char],
    out_subs: &[char],
    a: &Tensor<T>,
    b: &Tensor<T>,
    dim_map: &BTreeMap<char, usize>,
) -> FerrotorchResult<Tensor<T>> {
    // We already filter out repeated-index operands at the entry to
    // einsum_two_gpu, but keep the guard here as a safety net so this
    // function is independently safe.
    if has_duplicate_chars(a_subs) || has_duplicate_chars(b_subs) {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "einsum_general",
        });
    }

    // Sum out lone-A indices (chars only in A and not in B or out) and
    // lone-B indices (only in B and not in A or out) up front. After this
    // every char in each operand is either batch / free / contract.
    let a_only_lone: Vec<char> = a_subs
        .iter()
        .copied()
        .filter(|c| !b_subs.contains(c) && !out_subs.contains(c))
        .collect();
    let b_only_lone: Vec<char> = b_subs
        .iter()
        .copied()
        .filter(|c| !a_subs.contains(c) && !out_subs.contains(c))
        .collect();

    let (a_reduced_subs, a_reduced) = reduce_lone_axes(a_subs, &a_only_lone, a)?;
    let (b_reduced_subs, b_reduced) = reduce_lone_axes(b_subs, &b_only_lone, b)?;

    // Classify chars after lone-axis reduction.
    let mut batch_chars: Vec<char> = Vec::new();
    let mut free_a_chars: Vec<char> = Vec::new();
    let mut free_b_chars: Vec<char> = Vec::new();
    let mut contract_chars: Vec<char> = Vec::new();

    for &c in &a_reduced_subs {
        let in_b = b_reduced_subs.contains(&c);
        let in_out = out_subs.contains(&c);
        match (in_b, in_out) {
            (true, true) => {
                if !batch_chars.contains(&c) {
                    batch_chars.push(c);
                }
            }
            (true, false) => {
                if !contract_chars.contains(&c) {
                    contract_chars.push(c);
                }
            }
            (false, true) => {
                if !free_a_chars.contains(&c) {
                    free_a_chars.push(c);
                }
            }
            (false, false) => {
                // Should already be summed out above; defensive Err.
                return Err(FerrotorchError::Internal {
                    message: format!(
                        "einsum_two_gpu_general: lone-A char '{c}' survived reduction"
                    ),
                });
            }
        }
    }
    for &c in &b_reduced_subs {
        if !a_reduced_subs.contains(&c) && out_subs.contains(&c) && !free_b_chars.contains(&c) {
            free_b_chars.push(c);
        }
    }

    // Build the source-axis lookup for each operand. Since we filtered
    // out repeats, each char appears at exactly one axis in each operand.
    let a_axis_of = |c: char| -> Option<usize> { a_reduced_subs.iter().position(|&x| x == c) };
    let b_axis_of = |c: char| -> Option<usize> { b_reduced_subs.iter().position(|&x| x == c) };

    // Build A permutation: [batch..., free_a..., contract...]
    let mut a_perm: Vec<usize> = Vec::with_capacity(a_reduced_subs.len());
    for &c in &batch_chars {
        a_perm.push(a_axis_of(c).ok_or_else(|| FerrotorchError::Internal {
            message: format!("einsum_two_gpu_general: batch char '{c}' missing from A"),
        })?);
    }
    for &c in &free_a_chars {
        a_perm.push(a_axis_of(c).ok_or_else(|| FerrotorchError::Internal {
            message: format!("einsum_two_gpu_general: free-A char '{c}' missing from A"),
        })?);
    }
    for &c in &contract_chars {
        a_perm.push(a_axis_of(c).ok_or_else(|| FerrotorchError::Internal {
            message: format!("einsum_two_gpu_general: contract char '{c}' missing from A"),
        })?);
    }
    if a_perm.len() != a_reduced_subs.len() {
        return Err(FerrotorchError::Internal {
            message: format!(
                "einsum_two_gpu_general: A permutation has {} axes, expected {}",
                a_perm.len(),
                a_reduced_subs.len()
            ),
        });
    }

    // Build B permutation: [batch..., contract..., free_b...]
    let mut b_perm: Vec<usize> = Vec::with_capacity(b_reduced_subs.len());
    for &c in &batch_chars {
        b_perm.push(b_axis_of(c).ok_or_else(|| FerrotorchError::Internal {
            message: format!("einsum_two_gpu_general: batch char '{c}' missing from B"),
        })?);
    }
    for &c in &contract_chars {
        b_perm.push(b_axis_of(c).ok_or_else(|| FerrotorchError::Internal {
            message: format!("einsum_two_gpu_general: contract char '{c}' missing from B"),
        })?);
    }
    for &c in &free_b_chars {
        b_perm.push(b_axis_of(c).ok_or_else(|| FerrotorchError::Internal {
            message: format!("einsum_two_gpu_general: free-B char '{c}' missing from B"),
        })?);
    }
    if b_perm.len() != b_reduced_subs.len() {
        return Err(FerrotorchError::Internal {
            message: format!(
                "einsum_two_gpu_general: B permutation has {} axes, expected {}",
                b_perm.len(),
                b_reduced_subs.len()
            ),
        });
    }

    // Apply permutations on-device (zero-copy stride view + strided_copy).
    let a_perm_view = crate::methods::permute_t(&a_reduced, &a_perm)?;
    let a_permuted = crate::methods::contiguous_t(&a_perm_view)?;
    let b_perm_view = crate::methods::permute_t(&b_reduced, &b_perm)?;
    let b_permuted = crate::methods::contiguous_t(&b_perm_view)?;

    // Compute group sizes.
    let batch_sizes: Vec<usize> = batch_chars.iter().map(|c| dim_map[c]).collect();
    let free_a_sizes: Vec<usize> = free_a_chars.iter().map(|c| dim_map[c]).collect();
    let free_b_sizes: Vec<usize> = free_b_chars.iter().map(|c| dim_map[c]).collect();
    let contract_sizes: Vec<usize> = contract_chars.iter().map(|c| dim_map[c]).collect();

    let batch_total: usize = crate::shape::numel(&batch_sizes).max(1);
    let free_a_total: usize = crate::shape::numel(&free_a_sizes).max(1);
    let free_b_total: usize = crate::shape::numel(&free_b_sizes).max(1);
    let contract_total: usize = crate::shape::numel(&contract_sizes).max(1);

    // Reshape A to [batch_total, free_a_total, contract_total]. Use raw
    // usize shapes so reshape's no-op fast path works on every device.
    let a_3d = crate::grad_fns::shape::reshape(
        &a_permuted,
        &[
            batch_total as isize,
            free_a_total as isize,
            contract_total as isize,
        ],
    )?;
    // Reshape B to [batch_total, contract_total, free_b_total].
    let b_3d = crate::grad_fns::shape::reshape(
        &b_permuted,
        &[
            batch_total as isize,
            contract_total as isize,
            free_b_total as isize,
        ],
    )?;

    // bmm requires 3-D; we always feed it 3-D here.
    let bmm_result = crate::grad_fns::linalg::bmm(&a_3d, &b_3d)?;
    // bmm_result shape: [batch_total, free_a_total, free_b_total].

    // Reshape back to [batch..., free_a..., free_b...].
    let mut intermediate_shape: Vec<isize> =
        Vec::with_capacity(batch_sizes.len() + free_a_sizes.len() + free_b_sizes.len());
    intermediate_shape.extend(batch_sizes.iter().map(|&n| n as isize));
    intermediate_shape.extend(free_a_sizes.iter().map(|&n| n as isize));
    intermediate_shape.extend(free_b_sizes.iter().map(|&n| n as isize));

    // Handle the corner case where every group is empty (rare — would
    // only arise from a fully-reduced equation): keep at least a scalar.
    let intermediate = if intermediate_shape.is_empty() {
        // 0-D scalar: bmm_result is [1, 1, 1]; reshape to [].
        crate::grad_fns::shape::reshape(&bmm_result, &[])?
    } else {
        crate::grad_fns::shape::reshape(&bmm_result, &intermediate_shape)?
    };

    // Build the intermediate's char order.
    let intermediate_chars: Vec<char> = batch_chars
        .iter()
        .chain(free_a_chars.iter())
        .chain(free_b_chars.iter())
        .copied()
        .collect();

    // If the intermediate already matches out_subs, we're done.
    if intermediate_chars == *out_subs {
        return Ok(intermediate);
    }

    // Otherwise build a permutation to reorder.
    if intermediate_chars.len() != out_subs.len() {
        return Err(FerrotorchError::Internal {
            message: format!(
                "einsum_two_gpu_general: intermediate has {} axes, output has {}",
                intermediate_chars.len(),
                out_subs.len()
            ),
        });
    }
    let out_perm: Vec<usize> = out_subs
        .iter()
        .map(|c| {
            intermediate_chars
                .iter()
                .position(|ic| ic == c)
                .ok_or_else(|| FerrotorchError::Internal {
                    message: format!(
                        "einsum_two_gpu_general: out char '{c}' missing from intermediate"
                    ),
                })
        })
        .collect::<FerrotorchResult<Vec<_>>>()?;

    let permuted_view = crate::methods::permute_t(&intermediate, &out_perm)?;
    crate::methods::contiguous_t(&permuted_view)
}

/// Sum out the listed `lone_chars` axes from `tensor`, returning the new
/// subscript list (with those chars removed) and the reduced tensor.
///
/// Used as a pre-pass for [`einsum_two_gpu_general`] so the four-way
/// classification (`batch`/`free_a`/`free_b`/`contract`) is exhaustive.
fn reduce_lone_axes<T: Float>(
    subs: &[char],
    lone_chars: &[char],
    tensor: &Tensor<T>,
) -> FerrotorchResult<(Vec<char>, Tensor<T>)> {
    if lone_chars.is_empty() {
        return Ok((subs.to_vec(), tensor.clone()));
    }
    let mut current_subs = subs.to_vec();
    let mut current = tensor.clone();
    // Sum from the highest dim downward so removals don't shift indices
    // we still need to remove.
    for axis in (0..subs.len()).rev() {
        if lone_chars.contains(&subs[axis]) {
            current = crate::grad_fns::reduction::sum_dim(&current, axis as i64, false)?;
            current_subs.remove(axis);
        }
    }
    Ok((current_subs, current))
}

fn einsum_two<T: Float>(
    parsed: &ParsedEquation,
    a: &Tensor<T>,
    b: &Tensor<T>,
    dim_map: &BTreeMap<char, usize>,
) -> FerrotorchResult<Tensor<T>> {
    // GPU-aware dispatch (#803): map the common contraction patterns
    // onto the existing GPU primitives instead of falling to CPU.
    if a.is_cuda() || b.is_cuda() {
        if a.device() != b.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: a.device(),
                got: b.device(),
            });
        }
        return match einsum_two_gpu(parsed, a, b, dim_map) {
            Err(FerrotorchError::NotImplementedOnCuda { .. }) => {
                einsum_product_composite(parsed, &[a, b], dim_map)
            }
            other => other,
        };
    }

    let a_subs = &parsed.input_subscripts[0];
    let b_subs = &parsed.input_subscripts[1];
    let out_subs = &parsed.output_subscripts;

    // Classify indices.
    // batch:    in A, in B, in output
    // free_a:   in A, NOT in B, in output
    // free_b:   in B, NOT in A, in output
    // contract: NOT in output (shared by A and B, or lone to either)
    //
    // A LONE index (present in exactly one operand and absent from the
    // output) is summed over (CORE-161 / #1855): it joins the contract
    // group. The inner contraction loop below skips a char that is absent
    // from an operand's `char_to_axis` map, so accumulating over the lone
    // char's range performs exactly the implicit summation — matching the
    // GPU path's `reduce_lone_axes` pre-pass and torch's semantics
    // (`torch.einsum("ij,j->j", A, B)[j] == sum_i A[i,j] * B[j]`).
    let mut batch_chars: Vec<char> = Vec::new();
    let mut free_a_chars: Vec<char> = Vec::new();
    let mut free_b_chars: Vec<char> = Vec::new();
    let mut contract_chars: Vec<char> = Vec::new();

    // Collect unique chars from A.
    let a_unique: Vec<char> = {
        let mut v = a_subs.clone();
        v.sort_unstable();
        v.dedup();
        v
    };
    let b_unique: Vec<char> = {
        let mut v = b_subs.clone();
        v.sort_unstable();
        v.dedup();
        v
    };

    for &c in &a_unique {
        let in_b = b_unique.contains(&c);
        let in_out = out_subs.contains(&c);
        match (in_b, in_out) {
            (true, true) => batch_chars.push(c),
            (true, false) => contract_chars.push(c),
            (false, true) => free_a_chars.push(c),
            (false, false) => {
                // Lone-A index: summed over in A only. Putting it in the
                // contract group makes the inner loop iterate its range;
                // `compute_b_flat` ignores it (not in `b_char_to_axis`),
                // so the accumulation sums A over this axis (#1855).
                contract_chars.push(c);
            }
        }
    }
    for &c in &b_unique {
        if !a_unique.contains(&c) {
            if out_subs.contains(&c) {
                free_b_chars.push(c);
            } else {
                // Lone-B index: summed over in B only — same contraction
                // treatment as lone-A above; `compute_a_flat` ignores it
                // (#1855). Previously these chars were dropped from every
                // group, silently reading B at index 0 along the axis.
                contract_chars.push(c);
            }
        }
    }

    // Compute sizes.
    let batch_sizes: Vec<usize> = batch_chars.iter().map(|c| dim_map[c]).collect();
    let free_a_sizes: Vec<usize> = free_a_chars.iter().map(|c| dim_map[c]).collect();
    let free_b_sizes: Vec<usize> = free_b_chars.iter().map(|c| dim_map[c]).collect();
    let contract_sizes: Vec<usize> = contract_chars.iter().map(|c| dim_map[c]).collect();

    // `.max(1)` collapses an EMPTY group (no chars → product of [] == 1) to a
    // single scalar slot. It must NOT mask a genuine zero-size dim: if any
    // group's product is 0, that dim is empty. torch (`at::native::einsum`
    // lowers to `at::bmm` over reshaped operands, `aten/src/ATen/native/
    // Linear.cpp:261-264`) propagates a zero-size dim into the output: an
    // einsum whose OUTPUT carries a zero-size axis returns a correctly-shaped
    // EMPTY tensor (numel 0), no panic. Short-circuit here so the
    // `decode_multi` index arithmetic below never divides by a zero size when
    // the output is empty. e.g. `F.bilinear(zeros(0,3), zeros(0,2), W, b)` ->
    // `einsum("bi,oij->boj")` with b=0 -> `[0, o, j]` (#1605).
    let out_shape_empty: Vec<usize> = out_subs.iter().map(|c| dim_map[c]).collect();
    if out_shape_empty.contains(&0) {
        return Tensor::from_storage(TensorStorage::cpu(Vec::new()), out_shape_empty, false);
    }

    let batch_total: usize = crate::shape::numel(&batch_sizes).max(1);
    let free_a_total: usize = crate::shape::numel(&free_a_sizes).max(1);
    let free_b_total: usize = crate::shape::numel(&free_b_sizes).max(1);
    // `.max(1)` collapses an EMPTY contraction group (no contract chars ->
    // product of [] == 1, the "single dot-product term" slot) to 1, but a
    // GENUINE zero-size contracted dim must give 0 terms: the sum over an empty
    // contraction is the zero init, matching torch's `at::bmm` over a zero K
    // (`aten/src/ATen/native/Linear.cpp:261-264`). Without the `is_empty()`
    // guard the inner loop would run `ci=0` and index into the empty operand
    // storage (#1605: `einsum("ij,jk->ik")` with j=0 -> zero-filled [i,k]).
    let contract_total: usize = if contract_sizes.is_empty() {
        1
    } else {
        crate::shape::numel(&contract_sizes)
    };

    let a_data = a.data_vec()?;
    let b_data = b.data_vec()?;
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Compute input strides.
    let a_strides = row_major_strides(a_shape);
    let b_strides = row_major_strides(b_shape);

    // Step 1-2: Build permuted + reshaped 3D views.
    // A target layout: [batch..., free_a..., contract...]
    // B target layout: [batch..., contract..., free_b...]
    //
    // Rather than physically transposing, we use indirect indexing.
    // For the GEMM: C[batch, fa, fb] = sum_c A[batch, fa, c] * B[batch, c, fb]

    // Precompute multi-index decoders for each group.
    // For each flat index in a group, compute the contribution to the input flat index.

    // A: for a given (batch_flat, free_a_flat, contract_flat), compute flat index into A.
    // B: for a given (batch_flat, contract_flat, free_b_flat), compute flat index into B.

    // Build lookup: for each char, which axis in A (or B) does it correspond to?
    let a_char_to_axis: BTreeMap<char, Vec<usize>> = {
        let mut m: BTreeMap<char, Vec<usize>> = BTreeMap::new();
        for (axis, &c) in a_subs.iter().enumerate() {
            m.entry(c).or_default().push(axis);
        }
        m
    };
    let b_char_to_axis: BTreeMap<char, Vec<usize>> = {
        let mut m: BTreeMap<char, Vec<usize>> = BTreeMap::new();
        for (axis, &c) in b_subs.iter().enumerate() {
            m.entry(c).or_default().push(axis);
        }
        m
    };

    // Helper: decode a flat index for a group of chars into per-char values.
    // Callers guarantee `sizes` carries no zero entry: the empty-output case is
    // short-circuited before any `decode_multi` call, and a zero contracted dim
    // makes `contract_total == 0` so the contraction loop never iterates (#1605).
    fn decode_multi(flat: usize, sizes: &[usize]) -> Vec<usize> {
        let mut result = vec![0usize; sizes.len()];
        let mut remainder = flat;
        for i in (0..sizes.len()).rev() {
            result[i] = remainder % sizes[i];
            remainder /= sizes[i];
        }
        result
    }

    // Compute A flat index from (batch_vals, free_a_vals, contract_vals).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn compute_a_flat(
        batch_chars: &[char],
        batch_vals: &[usize],
        free_a_chars: &[char],
        free_a_vals: &[usize],
        contract_chars: &[char],
        contract_vals: &[usize],
        a_char_to_axis: &BTreeMap<char, Vec<usize>>,
        a_shape: &[usize],
        a_strides: &[usize],
    ) -> usize {
        let mut flat = 0usize;
        for (i, &c) in batch_chars.iter().enumerate() {
            if let Some(axes) = a_char_to_axis.get(&c) {
                for &ax in axes {
                    let v = if a_shape[ax] == 1 { 0 } else { batch_vals[i] };
                    flat += v * a_strides[ax];
                }
            }
        }
        for (i, &c) in free_a_chars.iter().enumerate() {
            if let Some(axes) = a_char_to_axis.get(&c) {
                for &ax in axes {
                    let v = if a_shape[ax] == 1 { 0 } else { free_a_vals[i] };
                    flat += v * a_strides[ax];
                }
            }
        }
        for (i, &c) in contract_chars.iter().enumerate() {
            if let Some(axes) = a_char_to_axis.get(&c) {
                for &ax in axes {
                    let v = if a_shape[ax] == 1 {
                        0
                    } else {
                        contract_vals[i]
                    };
                    flat += v * a_strides[ax];
                }
            }
        }
        flat
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn compute_b_flat(
        batch_chars: &[char],
        batch_vals: &[usize],
        free_b_chars: &[char],
        free_b_vals: &[usize],
        contract_chars: &[char],
        contract_vals: &[usize],
        b_char_to_axis: &BTreeMap<char, Vec<usize>>,
        b_shape: &[usize],
        b_strides: &[usize],
    ) -> usize {
        let mut flat = 0usize;
        for (i, &c) in batch_chars.iter().enumerate() {
            if let Some(axes) = b_char_to_axis.get(&c) {
                for &ax in axes {
                    let v = if b_shape[ax] == 1 { 0 } else { batch_vals[i] };
                    flat += v * b_strides[ax];
                }
            }
        }
        for (i, &c) in contract_chars.iter().enumerate() {
            if let Some(axes) = b_char_to_axis.get(&c) {
                for &ax in axes {
                    let v = if b_shape[ax] == 1 {
                        0
                    } else {
                        contract_vals[i]
                    };
                    flat += v * b_strides[ax];
                }
            }
        }
        for (i, &c) in free_b_chars.iter().enumerate() {
            if let Some(axes) = b_char_to_axis.get(&c) {
                for &ax in axes {
                    let v = if b_shape[ax] == 1 { 0 } else { free_b_vals[i] };
                    flat += v * b_strides[ax];
                }
            }
        }
        flat
    }

    // Step 6: GEMM — C[batch, free_a, free_b] = sum_contract A[...] * B[...]
    // Result is [batch_total, free_a_total, free_b_total] in row-major.
    let gemm_size = batch_total * free_a_total * free_b_total;
    let mut gemm_result = vec![<T as num_traits::Zero>::zero(); gemm_size];

    for bi in 0..batch_total {
        let batch_vals = decode_multi(bi, &batch_sizes);
        for fa in 0..free_a_total {
            let free_a_vals = decode_multi(fa, &free_a_sizes);
            for fb in 0..free_b_total {
                let free_b_vals = decode_multi(fb, &free_b_sizes);
                let mut acc = <T as num_traits::Zero>::zero();
                for ci in 0..contract_total {
                    let contract_vals = decode_multi(ci, &contract_sizes);
                    let a_flat = compute_a_flat(
                        &batch_chars,
                        &batch_vals,
                        &free_a_chars,
                        &free_a_vals,
                        &contract_chars,
                        &contract_vals,
                        &a_char_to_axis,
                        a_shape,
                        &a_strides,
                    );
                    let b_flat = compute_b_flat(
                        &batch_chars,
                        &batch_vals,
                        &free_b_chars,
                        &free_b_vals,
                        &contract_chars,
                        &contract_vals,
                        &b_char_to_axis,
                        b_shape,
                        &b_strides,
                    );
                    acc += a_data[a_flat] * b_data[b_flat];
                }
                gemm_result[bi * (free_a_total * free_b_total) + fa * free_b_total + fb] = acc;
            }
        }
    }

    // Step 7: Reshape + permute to output shape.
    // The gemm_result is laid out as [batch..., free_a..., free_b...].
    // We need to permute to match the output subscripts order.
    let intermediate_chars: Vec<char> = batch_chars
        .iter()
        .chain(free_a_chars.iter())
        .chain(free_b_chars.iter())
        .copied()
        .collect();
    let intermediate_sizes: Vec<usize> = batch_sizes
        .iter()
        .chain(free_a_sizes.iter())
        .chain(free_b_sizes.iter())
        .copied()
        .collect();

    // If output subscript order matches intermediate, we're done.
    if intermediate_chars == *out_subs {
        let out_shape: Vec<usize> = out_subs.iter().map(|c| dim_map[c]).collect();
        return Tensor::from_storage(TensorStorage::cpu(gemm_result), out_shape, false);
    }

    // Otherwise, permute.
    let out_shape: Vec<usize> = out_subs.iter().map(|c| dim_map[c]).collect();
    let out_numel: usize = if out_shape.is_empty() {
        1
    } else {
        crate::shape::numel(&out_shape)
    };

    // Build permutation: for each output axis, find which intermediate axis it corresponds to.
    let perm: Vec<usize> = out_subs
        .iter()
        .map(|c| {
            intermediate_chars
                .iter()
                .position(|ic| ic == c)
                .expect("output char must exist in intermediate")
        })
        .collect();

    let inter_strides = row_major_strides(&intermediate_sizes);

    let mut result = vec![<T as num_traits::Zero>::zero(); out_numel];
    for (out_flat, result_elem) in result.iter_mut().enumerate() {
        // Decode output multi-index.
        let out_multi = decode_multi(out_flat, &out_shape);
        // Map to intermediate multi-index.
        let mut inter_flat = 0usize;
        for (out_axis, &inter_axis) in perm.iter().enumerate() {
            inter_flat += out_multi[out_axis] * inter_strides[inter_axis];
        }
        *result_elem = gemm_result[inter_flat];
    }

    Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)
}

/// Compute row-major strides for a shape.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn einsum_nary_pairwise<T: Float>(
    parsed: &ParsedEquation,
    inputs: &[&Tensor<T>],
) -> FerrotorchResult<Tensor<T>> {
    validate_same_device("einsum", inputs)?;
    let mut current = inputs[0].clone();
    let mut current_subs = parsed.input_subscripts[0].clone();

    for operand_idx in 1..inputs.len() {
        let rhs_subs = &parsed.input_subscripts[operand_idx];
        let remaining = &parsed.input_subscripts[operand_idx + 1..];
        let pair_out = nary_pair_output_subscripts(
            &current_subs,
            rhs_subs,
            remaining,
            &parsed.output_subscripts,
        );
        let equation = format!(
            "{},{}->{}",
            chars_to_string(&current_subs),
            chars_to_string(rhs_subs),
            chars_to_string(&pair_out)
        );
        let next = einsum_forward(&equation, &[&current, inputs[operand_idx]])?;
        current = next;
        current_subs = pair_out;
    }

    if current_subs == parsed.output_subscripts {
        return Ok(current);
    }

    let equation = format!(
        "{}->{}",
        chars_to_string(&current_subs),
        chars_to_string(&parsed.output_subscripts)
    );
    einsum_forward(&equation, &[&current])
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Raw forward implementation. Public callers use `einsum`, which mirrors
// `torch.einsum` by attaching autograd when inputs require gradients.
fn einsum_forward<T: Float>(equation: &str, inputs: &[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
    if inputs.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "einsum: expected at least one input tensor, got 0".into(),
        });
    }

    let input_shapes: Vec<&[usize]> = inputs.iter().map(|tensor| tensor.shape()).collect();
    let parsed = parse_equation(equation, &input_shapes)?;
    let dim_map = build_dim_map(&parsed, inputs)?;

    let result = if inputs.len() > 2 {
        einsum_nary_pairwise(&parsed, inputs)?
    } else if inputs.iter().any(|tensor| tensor.is_cuda())
        && needs_operand_broadcast(&parsed, inputs, &dim_map)
    {
        einsum_product_composite(&parsed, inputs, &dim_map)?
    } else {
        match inputs.len() {
            1 => einsum_single(&parsed, inputs[0], &dim_map)?,
            2 => einsum_two(&parsed, inputs[0], inputs[1], &dim_map)?,
            _ => unreachable!(),
        }
    };

    Ok(result)
}

/// Einstein summation.
///
/// Evaluates the contraction specified by `equation` on the given `inputs`.
/// Like `torch.einsum`, this public entry point participates in autograd when
/// any input requires gradients.
///
/// # Examples
///
/// ```ignore
/// // Matrix multiply: (M,K) @ (K,N) -> (M,N)
/// let c = einsum("ij,jk->ik", &[&a, &b])?;
///
/// // Batched matrix multiply
/// let c = einsum("bij,bjk->bik", &[&a, &b])?;
///
/// // Trace
/// let t = einsum("ii->", &[&a])?;
///
/// // Outer product
/// let o = einsum("i,j->ij", &[&a, &b])?;
///
/// // Transpose
/// let t = einsum("ij->ji", &[&a])?;
/// ```
pub fn einsum<T: Float>(equation: &str, inputs: &[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
    einsum_differentiable(equation, inputs)
}

/// Differentiable Einstein summation. If any input requires grad and grad
/// is enabled, attaches [`EinsumBackward`].
///
/// Participates in autocast: classified as `ReducedPrecision` (`"einsum"`).
pub fn einsum_differentiable<T: Float>(
    equation: &str,
    inputs: &[&Tensor<T>],
) -> FerrotorchResult<Tensor<T>> {
    autocast_guard("einsum");

    let result = einsum_forward(equation, inputs)?;

    let any_requires_grad = inputs.iter().any(|t| t.requires_grad());

    if is_grad_enabled() && any_requires_grad {
        // Store the CANONICAL equation (whitespace-free, explicit output)
        // in the grad-fns, never the raw user string: the backward passes
        // re-parse it by hand and previously panicked on torch-legal
        // spaced equations ("ii -> i", CORE-163 / #1857) and treated
        // implicit-mode outputs as empty (CORE-164 / #1858). The
        // re-parse here cannot fail: `einsum_forward` above already parsed the
        // same string for the same input count.
        let input_shapes: Vec<&[usize]> = inputs.iter().map(|tensor| tensor.shape()).collect();
        let canonical = canonical_equation(&parse_equation(equation, &input_shapes)?);
        let wrapped = match inputs.len() {
            1 => {
                let grad_fn = Arc::new(EinsumBackwardSingle {
                    equation: canonical,
                    input: inputs[0].clone(),
                });
                // Reuse the result's storage as-is. For CUDA inputs the
                // forward path now produces a device tensor (#803), and
                // calling `data_vec()` here would yank it back to CPU
                // — re-introducing the silent-detour the dispatch
                // closes. `into_storage_and_shape` keeps the storage
                // bound to whichever device the forward produced.
                let (storage, shape) = result.into_storage_and_shape()?;
                Tensor::from_operation(storage, shape, grad_fn)
            }
            2 => {
                let grad_fn = Arc::new(EinsumBackwardTwo {
                    equation: canonical,
                    a: inputs[0].clone(),
                    b: inputs[1].clone(),
                });
                let (storage, shape) = result.into_storage_and_shape()?;
                Tensor::from_operation(storage, shape, grad_fn)
            }
            _ => {
                let grad_fn = Arc::new(EinsumBackwardN {
                    equation: canonical,
                    inputs: inputs.iter().map(|tensor| (*tensor).clone()).collect(),
                });
                let (storage, shape) = result.into_storage_and_shape()?;
                Tensor::from_operation(storage, shape, grad_fn)
            }
        }?;
        Ok(wrapped)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Backward: single-input
// ---------------------------------------------------------------------------

/// Backward for single-input einsum: `C = einsum(eq, [A])`.
///
/// For a single-input einsum like `"ij->ji"` (transpose) or `"ii->"` (trace),
/// the gradient is computed by reversing the equation:
/// `grad_A = einsum(reverse_eq, [grad_C])`.
#[derive(Debug)]
struct EinsumBackwardSingle<T: Float> {
    equation: String,
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for EinsumBackwardSingle<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let (lhs, rhs) = self
            .equation
            .split_once("->")
            .unwrap_or((&self.equation, ""));

        let in_subs: Vec<char> = lhs.chars().collect();
        let out_subs: Vec<char> = rhs.chars().collect();

        // Repeated input indices (e.g. "ii->" trace, "iij->j" mixed
        // diagonal projection) are the VJP of PyTorch's
        // `op.diagonal(...).movedim(...)` forward decomposition: first
        // compute the VJP for the deduped diagonalized tensor, then embed it
        // back into the input by zeroing off-diagonal positions.
        if has_duplicate_chars(&in_subs) {
            return self.backward_repeated_index(grad_output, &in_subs, &out_subs);
        }

        let in_shape = self.input.shape();
        let grad_input = single_input_distinct_backward(grad_output, &in_subs, &out_subs, |c| {
            in_subs
                .iter()
                .position(|&ic| ic == c)
                .map(|axis| in_shape[axis])
        })?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "EinsumBackward"
    }
}

impl<T: Float> EinsumBackwardSingle<T> {
    /// Backward path for repeated-input-index cases (`"ii->"` trace,
    /// `"ii->i"` diagonal, `"iij->j"` mixed repeats). PyTorch implements the
    /// forward by taking diagonal views for repeated labels; the VJP is the
    /// corresponding diagonal embed. This implementation keeps the dynamic
    /// gradient math on the tensor device: compute the deduped VJP with
    /// reshape/expand/permute, then mask off non-diagonal coordinates on the
    /// original input device.
    fn backward_repeated_index(
        &self,
        grad_output: &Tensor<T>,
        in_subs: &[char],
        out_subs: &[char],
    ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let in_shape = self.input.shape();
        let x_dedup = dedup_first_occurrence(in_subs);

        let g = single_input_distinct_backward(grad_output, &x_dedup, out_subs, |c| {
            in_subs
                .iter()
                .position(|&ic| ic == c)
                .map(|axis| in_shape[axis])
        })?;

        let view_shape: Vec<isize> = {
            let mut seen = Vec::new();
            in_subs
                .iter()
                .map(|&c| {
                    if seen.contains(&c) {
                        1
                    } else {
                        seen.push(c);
                        in_subs
                            .iter()
                            .position(|&ic| ic == c)
                            .map(|axis| in_shape[axis] as isize)
                            .expect("deduped repeated index must exist")
                    }
                })
                .collect()
        };
        let g_contig = crate::methods::contiguous_t(&g)?;
        let g_view = crate::grad_fns::shape::reshape(&g_contig, &view_shape)?;
        let g_full = if g_view.shape() == in_shape {
            g_view
        } else {
            crate::grad_fns::shape::expand(&g_view, in_shape)?
        };
        let mask = diagonal_mask::<T>(in_subs, in_shape)?;
        let mask = if self.input.is_cuda() {
            mask.to(self.input.device())?
        } else {
            mask
        };
        let grad_tensor = crate::grad_fns::arithmetic::mul(&g_full, &mask)?;
        Ok(vec![Some(grad_tensor)])
    }
}

fn dedup_first_occurrence(subs: &[char]) -> Vec<char> {
    let mut dedup = Vec::with_capacity(subs.len());
    for &c in subs {
        if !dedup.contains(&c) {
            dedup.push(c);
        }
    }
    dedup
}

fn single_input_distinct_backward<T, F>(
    grad_output: &Tensor<T>,
    in_subs: &[char],
    out_subs: &[char],
    size_of: F,
) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(char) -> Option<usize>,
{
    if has_duplicate_chars(in_subs) {
        return Err(FerrotorchError::Internal {
            message: "einsum backward: distinct-input VJP received repeated subscripts".to_string(),
        });
    }
    for &c in out_subs {
        if !in_subs.contains(&c) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "einsum backward: output index '{c}' does not appear in input subscripts"
                ),
            });
        }
    }

    // Projection / axis-sum / full-reduce / pure permutation (#791):
    // forward over distinct labels is permute-to-output-order followed by
    // reductions over dropped axes. The VJP inserts singleton dropped axes,
    // expands them, then permutes back to input order.
    let dropped_chars: Vec<char> = in_subs
        .iter()
        .filter(|c| !out_subs.contains(c))
        .copied()
        .collect();
    let intermediate_chars: Vec<char> = out_subs
        .iter()
        .chain(dropped_chars.iter())
        .copied()
        .collect();
    let intermediate_shape: Vec<usize> = intermediate_chars
        .iter()
        .map(|&c| {
            size_of(c).ok_or_else(|| FerrotorchError::Internal {
                message: format!("einsum backward: no saved size for label '{c}'"),
            })
        })
        .collect::<FerrotorchResult<Vec<_>>>()?;

    let unsqueezed_shape: Vec<usize> = (0..intermediate_chars.len())
        .map(|i| {
            if i < out_subs.len() {
                intermediate_shape[i]
            } else {
                1
            }
        })
        .collect();

    let grad_unsq = if grad_output.shape() == unsqueezed_shape.as_slice() {
        grad_output.clone()
    } else if grad_output.is_contiguous() {
        grad_output.view_reshape(unsqueezed_shape.clone())?
    } else {
        grad_output
            .contiguous()?
            .view_reshape(unsqueezed_shape.clone())?
    };

    let grad_expanded =
        if intermediate_shape.is_empty() || grad_unsq.shape() == intermediate_shape.as_slice() {
            grad_unsq
        } else {
            crate::grad_fns::shape::expand(&grad_unsq, &intermediate_shape)?
        };

    if intermediate_chars == in_subs {
        return crate::methods::contiguous_t(&grad_expanded);
    }
    let perm: Vec<usize> = in_subs
        .iter()
        .map(|c| {
            intermediate_chars
                .iter()
                .position(|ic| ic == c)
                .expect("in_subs char must exist in intermediate_chars")
        })
        .collect();
    let permuted = crate::methods::permute_t(&grad_expanded, &perm)?;
    crate::methods::contiguous_t(&permuted)
}

fn reduce_and_expand_gradient_to_target<T, F>(
    g: Tensor<T>,
    present: &[char],
    target_dedup: &[char],
    size_of: F,
) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(char) -> usize,
{
    if g.ndim() != present.len() {
        return Err(FerrotorchError::Internal {
            message: format!(
                "einsum backward: derived gradient has {} dims but {} present labels",
                g.ndim(),
                present.len()
            ),
        });
    }

    let mut reduced = g;
    for axis in (0..present.len()).rev() {
        let c = present[axis];
        let current = reduced.shape()[axis];
        let target = size_of(c);
        if current == target {
            continue;
        }
        if target == 1 {
            reduced = crate::grad_fns::reduction::sum_dim(&reduced, axis as i64, true)?;
        } else if current != 1 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "einsum backward: label '{c}' gradient size {current} cannot be reduced or expanded to target size {target}"
                ),
            });
        }
    }

    let mut unsq_shape = Vec::with_capacity(target_dedup.len());
    for &c in target_dedup {
        if let Some(axis) = present.iter().position(|&pc| pc == c) {
            unsq_shape.push(reduced.shape()[axis] as isize);
        } else {
            unsq_shape.push(1);
        }
    }
    let full_shape: Vec<usize> = target_dedup.iter().map(|&c| size_of(c)).collect();
    let reshaped = crate::grad_fns::shape::reshape(&reduced, &unsq_shape)?;
    if reshaped.shape() == full_shape.as_slice() {
        crate::methods::contiguous_t(&reshaped)
    } else {
        let expanded = crate::grad_fns::shape::expand(&reshaped, &full_shape)?;
        crate::methods::contiguous_t(&expanded)
    }
}

// ---------------------------------------------------------------------------
// Backward: two-input
// ---------------------------------------------------------------------------

/// Backward for two-input einsum: `C = einsum(eq, [A, B])`.
///
/// For the classical case `"ij,jk->ik"` (distinct, non-lone subscripts):
/// - `grad_A = einsum("ik,jk->ij", [grad_C, B])` (swap output with A-input)
/// - `grad_B = einsum("ij,ik->jk", [A, grad_C])` (swap output with B-input)
///
/// The general scheme implemented by [`Self::grad_for_target`] extends the
/// swap to repeated and lone subscripts (CORE-162 / #1856); the stored
/// `equation` is the canonical form produced by [`canonical_equation`]
/// (whitespace-free, explicit output — CORE-163/164, #1857/#1858).
#[derive(Debug)]
struct EinsumBackwardTwo<T: Float> {
    equation: String,
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> EinsumBackwardTwo<T> {
    /// Gradient w.r.t. operand `target` (0 = A, 1 = B).
    ///
    /// General scheme (CORE-162 / #1856):
    /// 1. Dedup the target's subscripts in first-occurrence order.
    /// 2. `present` = deduped chars that appear in the output or in the
    ///    other operand. These are recoverable by a plain einsum:
    ///    `g = einsum("{out},{other}->{present}", [grad_C, other])`.
    ///    (The naive textual swap `->{target}` instead produced repeated
    ///    OUTPUT subscripts for repeated-index operands — e.g. grad of
    ///    `"ii,j->ij"` w.r.t. A became `"ij,j->ii"` — which no einsum can
    ///    express and which panicked the CPU permute step.)
    /// 3. Lone chars (absent from both) were summed over in the forward;
    ///    the gradient broadcasts back along them: unsqueeze + `expand`.
    /// 4. Repeated subscripts: the forward read only the generalized
    ///    diagonal, so the gradient is a diagonal-embed scatter — the
    ///    expanded `g` multiplied by a 0/1 diagonal mask (zero off the
    ///    diagonal). Matches `torch.einsum("ii,j->ij").backward()`:
    ///    `A.grad == diag_embed(einsum("ij,j->i", [grad_C, B]))`.
    ///
    /// For an operand with distinct, non-lone subscripts steps 3–4 are
    /// no-ops and this reduces exactly to the classical swap equation.
    fn grad_for_target(
        &self,
        target: usize,
        grad_output: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        let (lhs, rhs) =
            self.equation
                .split_once("->")
                .ok_or_else(|| FerrotorchError::Internal {
                    message: format!(
                        "EinsumBackwardTwo: stored equation '{}' is not canonical (no '->')",
                        self.equation
                    ),
                })?;
        let parts: Vec<&str> = lhs.split(',').collect();
        if parts.len() != 2 {
            return Err(FerrotorchError::Internal {
                message: format!(
                    "EinsumBackwardTwo: stored equation '{}' does not have two inputs",
                    self.equation
                ),
            });
        }

        let x_str = parts[target];
        let o_str = parts[1 - target];
        let x_subs: Vec<char> = x_str.chars().collect();
        let o_subs: Vec<char> = o_str.chars().collect();
        let out_subs: Vec<char> = rhs.chars().collect();
        let (x, other) = if target == 0 {
            (&self.a, &self.b)
        } else {
            (&self.b, &self.a)
        };
        let x_shape = x.shape();
        // Every char of x_subs maps to (at least) one axis of x; the
        // forward's `build_dim_map` validated subscript/rank agreement
        // and repeated-index size consistency.
        let size_of = |c: char| -> usize {
            x_subs
                .iter()
                .position(|&xc| xc == c)
                .map_or(1, |ax| x_shape[ax])
        };

        // Step 1: dedup target subscripts, first-occurrence order.
        let mut x_dedup: Vec<char> = Vec::with_capacity(x_subs.len());
        for &c in &x_subs {
            if !x_dedup.contains(&c) {
                x_dedup.push(c);
            }
        }

        // Step 2: contract grad_C with the other operand onto the
        // recoverable chars. Operand order mirrors the forward equation
        // so the derived equation stays human-auditable.
        let present: Vec<char> = x_dedup
            .iter()
            .copied()
            .filter(|c| out_subs.contains(c) || o_subs.contains(c))
            .collect();
        let present_str: String = present.iter().collect();
        let g = if target == 0 {
            einsum_forward(
                &format!("{rhs},{o_str}->{present_str}"),
                &[grad_output, other],
            )?
        } else {
            einsum_forward(
                &format!("{o_str},{rhs}->{present_str}"),
                &[other, grad_output],
            )?
        };

        // Step 3: reduce or expand every recoverable char back to the target
        // operand's deduped shape. This covers both lone chars (insert a
        // size-1 axis then expand) and shared-label broadcasting (sum when the
        // target carried size 1, expand when the other operand carried size 1).
        let g = reduce_and_expand_gradient_to_target(g, &present, &x_dedup, size_of)?;

        // Step 4: diagonal-embed for repeated subscripts.
        if !has_duplicate_chars(&x_subs) {
            return Ok(g);
        }
        // View `g` (shape per x_dedup) over the operand's full rank with
        // size-1 axes at every non-first occurrence of a repeated char,
        // expand to the operand shape, then zero everything off the
        // generalized diagonal with a 0/1 mask. The mask is host-built
        // constant data (like `creation::eye`) uploaded to the operand's
        // device; the masking multiply itself runs on-device (R-LOUD-2:
        // explicit, result-preserving host constant — not a compute detour).
        let view_shape: Vec<isize> = {
            let mut seen: Vec<char> = Vec::new();
            x_subs
                .iter()
                .map(|&c| {
                    if seen.contains(&c) {
                        1
                    } else {
                        seen.push(c);
                        size_of(c) as isize
                    }
                })
                .collect()
        };
        let g_contig = crate::methods::contiguous_t(&g)?;
        let g_view = crate::grad_fns::shape::reshape(&g_contig, &view_shape)?;
        let g_full = crate::grad_fns::shape::expand(&g_view, x_shape)?;
        let mask = diagonal_mask::<T>(&x_subs, x_shape)?;
        let mask = if x.is_cuda() {
            mask.to(x.device())?
        } else {
            mask
        };
        crate::grad_fns::arithmetic::mul(&g_full, &mask)
    }
}

/// Build the 0/1 generalized-diagonal mask for an operand whose subscripts
/// contain repeats: `mask[idx] == 1` iff every repeated char carries the
/// same coordinate on all of its axes (CORE-162 / #1856). Host-built
/// constant; the caller uploads it when the operand lives on CUDA.
fn diagonal_mask<T: Float>(subs: &[char], shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let numel: usize = crate::shape::numel(shape);
    let mut data = vec![<T as num_traits::Zero>::zero(); numel];
    for (flat, slot) in data.iter_mut().enumerate() {
        let mut coords = vec![0usize; shape.len()];
        let mut rem = flat;
        for i in (0..shape.len()).rev() {
            coords[i] = rem % shape[i];
            rem /= shape[i];
        }
        let mut first_val: BTreeMap<char, usize> = BTreeMap::new();
        let mut on_diag = true;
        for (axis, &c) in subs.iter().enumerate() {
            match first_val.get(&c) {
                Some(&prev) if prev != coords[axis] => {
                    on_diag = false;
                    break;
                }
                _ => {
                    first_val.insert(c, coords[axis]);
                }
            }
        }
        if on_diag {
            *slot = <T as num_traits::One>::one();
        }
    }
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

impl<T: Float> GradFn<T> for EinsumBackwardTwo<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_a = if self.a.requires_grad() {
            Some(self.grad_for_target(0, grad_output)?)
        } else {
            None
        };

        let grad_b = if self.b.requires_grad() {
            Some(self.grad_for_target(1, grad_output)?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "EinsumBackward"
    }
}

// ---------------------------------------------------------------------------
// Backward: n-input
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct EinsumBackwardN<T: Float> {
    equation: String,
    inputs: Vec<Tensor<T>>,
}

impl<T: Float> EinsumBackwardN<T> {
    fn grad_for_target(
        &self,
        target: usize,
        grad_output: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        let (lhs, rhs) =
            self.equation
                .split_once("->")
                .ok_or_else(|| FerrotorchError::Internal {
                    message: format!(
                        "EinsumBackwardN: stored equation '{}' is not canonical (no '->')",
                        self.equation
                    ),
                })?;
        let parts: Vec<&str> = lhs.split(',').collect();
        if parts.len() != self.inputs.len() {
            return Err(FerrotorchError::Internal {
                message: format!(
                    "EinsumBackwardN: stored equation '{}' has {} inputs but backward stored {} tensors",
                    self.equation,
                    parts.len(),
                    self.inputs.len()
                ),
            });
        }
        if target >= self.inputs.len() {
            return Err(FerrotorchError::Internal {
                message: format!(
                    "EinsumBackwardN: target {target} out of range for {} tensors",
                    self.inputs.len()
                ),
            });
        }

        let x_str = parts[target];
        let x_subs: Vec<char> = x_str.chars().collect();
        let out_subs: Vec<char> = rhs.chars().collect();
        let x = &self.inputs[target];
        let x_shape = x.shape();
        let size_of = |c: char| -> usize {
            x_subs
                .iter()
                .position(|&xc| xc == c)
                .map_or(1, |axis| x_shape[axis])
        };

        let mut x_dedup: Vec<char> = Vec::with_capacity(x_subs.len());
        for &c in &x_subs {
            if !x_dedup.contains(&c) {
                x_dedup.push(c);
            }
        }

        let present: Vec<char> = x_dedup
            .iter()
            .copied()
            .filter(|c| {
                out_subs.contains(c)
                    || parts
                        .iter()
                        .enumerate()
                        .any(|(idx, part)| idx != target && part.chars().any(|oc| oc == *c))
            })
            .collect();
        let present_str = chars_to_string(&present);

        let mut lhs_parts = Vec::with_capacity(parts.len());
        lhs_parts.push(rhs.to_string());
        for (idx, part) in parts.iter().enumerate() {
            if idx != target {
                lhs_parts.push((*part).to_string());
            }
        }
        let derived_equation = format!("{}->{}", lhs_parts.join(","), present_str);

        let mut operands: Vec<&Tensor<T>> = Vec::with_capacity(parts.len());
        operands.push(grad_output);
        for (idx, input) in self.inputs.iter().enumerate() {
            if idx != target {
                operands.push(input);
            }
        }

        let g = einsum_forward(&derived_equation, &operands)?;
        let g = reduce_and_expand_gradient_to_target(g, &present, &x_dedup, size_of)?;

        if !has_duplicate_chars(&x_subs) {
            return Ok(g);
        }

        let view_shape: Vec<isize> = {
            let mut seen: Vec<char> = Vec::new();
            x_subs
                .iter()
                .map(|&c| {
                    if seen.contains(&c) {
                        1
                    } else {
                        seen.push(c);
                        size_of(c) as isize
                    }
                })
                .collect()
        };
        let g_contig = crate::methods::contiguous_t(&g)?;
        let g_view = crate::grad_fns::shape::reshape(&g_contig, &view_shape)?;
        let g_full = crate::grad_fns::shape::expand(&g_view, x_shape)?;
        let mask = diagonal_mask::<T>(&x_subs, x_shape)?;
        let mask = if x.is_cuda() {
            mask.to(x.device())?
        } else {
            mask
        };
        crate::grad_fns::arithmetic::mul(&g_full, &mask)
    }
}

impl<T: Float> GradFn<T> for EinsumBackwardN<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let mut grads = Vec::with_capacity(self.inputs.len());
        for idx in 0..self.inputs.len() {
            if self.inputs[idx].requires_grad() {
                grads.push(Some(self.grad_for_target(idx, grad_output)?));
            } else {
                grads.push(None);
            }
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        self.inputs.iter().collect()
    }

    fn name(&self) -> &'static str {
        "EinsumBackward"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: {a} vs {e} (diff {})",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Zero-size dims (#1605): must not panic with `% 0` in `decode_multi`.
    // torch lowers einsum to `at::bmm` over reshaped operands
    // (`aten/src/ATen/native/Linear.cpp:261-264`), so a zero-size dim
    // propagates into the output as an empty tensor, and a zero CONTRACTED
    // dim yields a zero-filled output of the non-contracted shape.
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_zero_batch_bilinear_eq() {
        // The exact equation `Bilinear::forward_pair` issues, with a zero
        // batch axis. torch `F.bilinear(zeros(0,3), zeros(0,2), W, b)` -> [0,4].
        // Here the einsum core: "bi,oij->boj" with b=0, o=4, i=3, j=2 -> [0,4,2].
        let x1 = t(&[], &[0, 3]); // [b=0, i=3]
        // weight [o=4, i=3, j=2]
        let w = t(
            &(0..(4 * 3 * 2)).map(|i| i as f32).collect::<Vec<_>>(),
            &[4, 3, 2],
        );
        let out = einsum("bi,oij->boj", &[&x1, &w]).unwrap();
        // b=0 in the output -> empty tensor, shape [0, 4, 2], numel 0, no panic.
        assert_eq!(out.shape(), &[0, 4, 2]);
        assert_eq!(out.data().unwrap().len(), 0);
    }

    #[test]
    fn test_einsum_zero_contracted_dim_zero_filled() {
        // Contracted dim j=0, output dims nonzero: matmul "ij,jk->ik" with
        // i=2, j=0, k=3. torch returns a zero-filled [2,3] (sum over empty
        // contraction). Must not panic; must be all zeros.
        let a = t(&[], &[2, 0]); // [i=2, j=0]
        let b = t(&[], &[0, 3]); // [j=0, k=3]
        let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_close(c.data().unwrap(), &[0.0; 6], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Matrix multiply: "ij,jk->ik"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_mm() {
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_close(c.data().unwrap(), &[19.0, 22.0, 43.0, 50.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Batched matrix multiply: "bij,bjk->bik"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_bmm() {
        // Batch 0: [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        // Batch 1: [[1, 0], [0, 1]] @ [[9, 10], [11, 12]] = [[9, 10], [11, 12]]
        #[rustfmt::skip]
        let a_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            1.0, 0.0, 0.0, 1.0,
        ];
        #[rustfmt::skip]
        let b_data: Vec<f32> = vec![
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        let a = t(&a_data, &[2, 2, 2]);
        let b = t(&b_data, &[2, 2, 2]);
        let c = einsum("bij,bjk->bik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);

        let d = c.data().unwrap();
        // batch 0
        assert_close(&d[0..4], &[19.0, 22.0, 43.0, 50.0], 1e-6);
        // batch 1
        assert_close(&d[4..8], &[9.0, 10.0, 11.0, 12.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Trace: "ii->"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_trace() {
        // [[1, 2], [3, 4]] -> trace = 1 + 4 = 5
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = einsum("ii->", &[&a]).unwrap();
        assert!(c.is_scalar());
        assert!((c.item().unwrap() - 5.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Outer product: "i,j->ij"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_outer_product() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[4.0, 5.0], &[2]);
        let c = einsum("i,j->ij", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        // [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]]
        assert_close(c.data().unwrap(), &[4.0, 5.0, 8.0, 10.0, 12.0, 15.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Transpose: "ij->ji"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_transpose() {
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let c = einsum("ij->ji", &[&a]).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        assert_close(c.data().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Sum all: "ij->"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_sum_all() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let c = einsum("ij->", &[&a]).unwrap();
        assert!(c.is_scalar());
        assert!((c.item().unwrap() - 21.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Sum over axis: "ij->i" (sum over j)
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_sum_axis() {
        // [[1, 2, 3], [4, 5, 6]] -> [6, 15]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let c = einsum("ij->i", &[&a]).unwrap();
        assert_eq!(c.shape(), &[2]);
        assert_close(c.data().unwrap(), &[6.0, 15.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Implicit mode: "ij,jk" (no ->)
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_implicit_mm() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        // j appears twice -> contracted. i,k appear once -> output "ik"
        let c = einsum("ij,jk", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_close(c.data().unwrap(), &[19.0, 22.0, 43.0, 50.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Backward: matrix multiply
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_backward_mm() {
        // Same as MmBackward test:
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = A @ B = [[19, 22], [43, 50]]
        // L = sum(C) = 134
        // dL/dA = ones @ B^T = [[11, 15], [11, 15]]
        // dL/dB = A^T @ ones = [[4, 4], [6, 6]]
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = leaf(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let c = einsum_differentiable("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2]);

        // Build sum for scalar.
        let c_data = c.data().unwrap();
        let loss_val: f32 = c_data.iter().sum();

        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let g = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(g)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![loss_val]),
            vec![],
            Arc::new(SumBackward { input: c }),
        )
        .unwrap();

        loss.backward().unwrap();

        let a_grad = a.grad().unwrap().expect("a should have grad");
        let b_grad = b.grad().unwrap().expect("b should have grad");

        assert_eq!(a_grad.shape(), &[2, 2]);
        assert_eq!(b_grad.shape(), &[2, 2]);

        // dL/dA = [[11, 15], [11, 15]]
        assert_close(a_grad.data().unwrap(), &[11.0, 15.0, 11.0, 15.0], 1e-5);
        // dL/dB = [[4, 4], [6, 6]]
        assert_close(b_grad.data().unwrap(), &[4.0, 4.0, 6.0, 6.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Invalid equation
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_invalid_equation() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        // Wrong number of inputs for the equation.
        assert!(einsum("ij,jk,kl->il", &[&a, &b]).is_err());

        // Three-input contractions are valid and contract left-to-right.
        let eye = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let three = einsum("ij,jk,kl->il", &[&a, &b, &eye]).unwrap();
        assert_close(three.data().unwrap(), &[19.0, 22.0, 43.0, 50.0], 1e-6);

        // Subscript count mismatch with tensor dims.
        assert!(einsum("ijk,jk->ik", &[&a, &b]).is_err());

        // Inconsistent dimension sizes.
        let c = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert!(einsum("ij,jk->ik", &[&c, &a]).is_err()); // c is 2x3, a is 2x2; j=3 vs j=2

        // Invalid character.
        assert!(einsum("i1,1j->ij", &[&a, &b]).is_err());
    }

    // -----------------------------------------------------------------------
    // Diagonal extraction: "ii->i"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_diagonal() {
        // [[1, 2], [3, 4]] -> diagonal = [1, 4]
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = einsum("ii->i", &[&a]).unwrap();
        assert_eq!(c.shape(), &[2]);
        assert_close(c.data().unwrap(), &[1.0, 4.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Dot product via einsum: "i,i->"
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_dot() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[4.0, 5.0, 6.0], &[3]);
        let c = einsum("i,i->", &[&a, &b]).unwrap();
        assert!(c.is_scalar());
        assert!((c.item().unwrap() - 32.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Non-square matrix multiply
    // -----------------------------------------------------------------------

    #[test]
    fn test_einsum_non_square_mm() {
        // (2,3) @ (3,4) -> (2,4)
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        );
        let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 4]);
        // Row 0: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
        //       = [38, 44, 50, 56]
        // Row 1: [4*1+5*5+6*9, 4*2+5*6+6*10, 4*3+5*7+6*11, 4*4+5*8+6*12]
        //       = [83, 98, 113, 128]
        assert_close(
            c.data().unwrap(),
            &[38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0],
            1e-5,
        );
    }

    // -------------------------------------------------------------------
    // autocast_guard integration
    // -------------------------------------------------------------------

    #[test]
    fn test_einsum_differentiable_fires_autocast_guard() {
        use crate::autograd::autocast::{AutocastDtype, autocast, set_autocast_debug};
        use crate::autograd::autocast_ops::{AutocastCategory, drain_autocast_events};

        set_autocast_debug(true);
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        // Outside autocast: no events.
        drain_autocast_events();
        let _ = einsum_differentiable("ij,jk->ik", &[&a, &b]).unwrap();
        assert!(drain_autocast_events().is_empty());

        // Inside autocast: records "einsum" as ReducedPrecision.
        autocast(AutocastDtype::F16, || {
            drain_autocast_events();
            let _ = einsum_differentiable("ij,jk->ik", &[&a, &b]).unwrap();
            let events = drain_autocast_events();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].op, "einsum");
            assert_eq!(events[0].category, AutocastCategory::ReducedPrecision);
        });
    }
}
