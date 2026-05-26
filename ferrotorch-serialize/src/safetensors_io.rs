//! Save and load `StateDict<T>` using the
//! [SafeTensors](https://huggingface.co/docs/safetensors/) format.
//!
//! Files produced by this module are fully compatible with Python's
//! `safetensors` library, enabling seamless model exchange between Rust and
//! the `HuggingFace` ecosystem.
//!
//! # Sharded checkpoints
//!
//! Large models (Llama 3 8B, Mistral, etc.) are shipped as multiple
//! `model-00001-of-NNNNN.safetensors` files alongside a
//! `model.safetensors.index.json` that maps tensor names to shards.
//! [`load_safetensors_sharded`] loads all shards into a single
//! [`StateDict`]; [`load_safetensors_auto`] detects whether the given
//! path is a single file or an index and dispatches accordingly.
//!
//! ## REQ status (per `.design/ferrotorch-serialize/safetensors_io.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream cites)
//! live in the design doc; this synopsis is a one-line summary per REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`save_safetensors` sorted-key write) | SHIPPED | `pub fn save_safetensors<T: Float>` in `safetensors_io.rs` building the `Vec<(String, TensorView)>` and delegating to `safetensors::serialize_to_file`; consumer: `pub use safetensors_io::save_safetensors` in `lib.rs`; training checkpoint export paths call this entry. Wire format is HuggingFace's spec per R-DEV-3. |
//! | REQ-2 (`load_safetensors` single-file decode + bf16/f16 upcast) | SHIPPED | `pub fn load_safetensors<T: Float>` in `safetensors_io.rs` + `fn decode_tensor_list` + `fn decode_view`; consumer: `pub use safetensors_io::load_safetensors` in `lib.rs`; downstream model loaders call this for single-file HuggingFace checkpoints. |
//! | REQ-3 (`load_safetensors_mmap` zero-copy load) | SHIPPED | `pub fn load_safetensors_mmap<T: Float>` in `safetensors_io.rs` using `memmap2::Mmap::map`; consumer: `pub use safetensors_io::load_safetensors_mmap` in `lib.rs`; production inference servers prefer this on large checkpoints. |
//! | REQ-4 (`SafeTensorsIndex` HuggingFace sharded index) | SHIPPED | `#[non_exhaustive] pub struct SafeTensorsIndex` + `SafeTensorsIndexMetadata` + `impl SafeTensorsIndex::{from_file, shard_files, group_by_shard}` in `safetensors_io.rs`; consumer: `load_safetensors_sharded` / `_mmap` / `_with_progress` / `_filtered` all construct an index via `SafeTensorsIndex::from_file`. |
//! | REQ-5 (`load_safetensors_sharded` rayon parallel) | SHIPPED | `pub fn load_safetensors_sharded<T: Float>` in `safetensors_io.rs` with `par_iter` over sorted shard files + serial merge + missing-key check; consumer: `pub use safetensors_io::load_safetensors_sharded` in `lib.rs`; `load_safetensors_auto` dispatches here for `*.index.json`. |
//! | REQ-6 (`load_safetensors_sharded_mmap` rayon + mmap) | SHIPPED | `pub fn load_safetensors_sharded_mmap<T: Float>` + `fn load_one_shard_owned_mmap` in `safetensors_io.rs`; consumer: `pub use safetensors_io::load_safetensors_sharded_mmap` in `lib.rs`; production loaders prefer this on disk-bound multi-shard checkpoints. |
//! | REQ-7 (`load_safetensors_auto` suffix dispatch) | SHIPPED | `pub fn load_safetensors_auto<T: Float>` in `safetensors_io.rs` dispatching on `.index.json` suffix; consumer: `pub use safetensors_io::load_safetensors_auto` in `lib.rs`; the meta-crate glob makes `ferrotorch::load_safetensors_auto` the primary user-facing entry. |
//! | REQ-8 (`ShardProgress` + `_with_progress` callback) | SHIPPED | `#[non_exhaustive] pub struct ShardProgress<'a>` + `pub fn load_safetensors_sharded_with_progress<T, F>` in `safetensors_io.rs`; consumer: `pub use safetensors_io::{ShardProgress, load_safetensors_sharded_with_progress}` in `lib.rs`; CLI / TUI tools loading 70B+ checkpoints wire progress bars through this entry. |
//! | REQ-9 (`load_safetensors_sharded_filtered` predicate-based load) | SHIPPED | `pub fn load_safetensors_sharded_filtered<T, F>` in `safetensors_io.rs` filtering per-shard expected keys before opening; consumer: `pub use safetensors_io::load_safetensors_sharded_filtered` in `lib.rs`; LoRA / adapter training loops use this to skip base-model weights. |
//! | REQ-10 (vectorized f16/bf16 → f32 upcast) | SHIPPED | `fn half_to_f32` / `fn bf16_to_f32` + `bytemuck::cast_slice` half-buffer reinterpret + `T == f32` fast-path specialization in `decode_view` in `safetensors_io.rs`; consumer: every bf16/f16 safetensors load (production HuggingFace transformer paths) hits this code path. |
//! | REQ-11 (serial-vs-parallel decode dispatch + env override) | SHIPPED | `fn decode_tensor_list` in `safetensors_io.rs` with the bf16/f16 → rayon, f32/f64 → serial dispatch + `FERROTORCH_FORCE_SERIAL_LOAD` env override; consumer: both `load_safetensors` and `load_safetensors_mmap` route through `decode_tensor_list`. |
//! | REQ-12 (LE endianness contract) | SHIPPED | `fn as_le_bytes` in `safetensors_io.rs` with SAFETY block citing the LE invariant; consumer: `save_safetensors` calls `as_le_bytes` for every tensor; the crate-root LE invariant (mirrored from `state_dict.rs`'s `compile_error!`) governs platform support. |

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use half::{bf16, f16};
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::serialize_to_file;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::Deserialize;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::StateDict;

/// Convert IEEE 754 half-precision (f16) bits to f32.
///
/// CL-1127 part B: thin wrapper over `half::f16::from_bits().to_f32()` so
/// we delegate the subnormal / Inf / NaN edge cases to the well-tested
/// `half` crate, which lowers to a vectorizable scalar op (`cvtph2ps` when
/// F16C is available on x86-64, table lookup otherwise).
#[inline]
fn half_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

/// Convert `bfloat16` bits to f32 by left-shifting into the upper 16 bits.
///
/// `bf16` shares its 8-bit exponent layout with `f32`, so the conversion
/// is bit-exact and reduces to an `unpcklwd` + `pslld` pair on x86-64 once
/// LLVM auto-vectorizes the loop body. (CL-1127 part B.)
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

/// Return the `safetensors::Dtype` that corresponds to the concrete `Float`
/// type `T`.
fn st_dtype<T: Float>() -> FerrotorchResult<Dtype> {
    let size = std::mem::size_of::<T>();
    match size {
        4 => Ok(Dtype::F32),
        8 => Ok(Dtype::F64),
        _ => Err(FerrotorchError::InvalidArgument {
            message: format!("unsupported element size {size} for safetensors serialization"),
        }),
    }
}

/// Return the expected `safetensors::Dtype` for the concrete `Float` type `T`,
/// used during loading to validate the file contents.
fn expected_dtype<T: Float>() -> FerrotorchResult<Dtype> {
    st_dtype::<T>()
}

/// Convert a slice of `T` to its raw little-endian byte representation.
///
/// # Safety
///
/// This reinterprets the memory of a `&[T]` as `&[u8]`. This is safe for
/// `f32` and `f64` on little-endian platforms (x86, ARM), which is the same
/// assumption that the `SafeTensors` format makes.
fn as_le_bytes<T: Float>(data: &[T]) -> &[u8] {
    // SAFETY: `data: &[T]` where `T: Float` is one of f32/f64/bf16 — every
    // such `T` is `Copy`, has a stable bit-level representation with no
    // padding, and has no `Drop` semantics, so reinterpreting the same
    // memory as `&[u8]` is sound. The byte length `size_of_val(data)`
    // equals `data.len() * size_of::<T>()`, the exact extent the source
    // pointer is valid for. The returned `&[u8]` reborrows `data` and is
    // tied to its lifetime, so it cannot dangle. Big-endian targets would
    // produce wrong bytes; the SafeTensors specification mandates LE.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

/// Internal helper used by [`save_safetensors`] to retain owned shape
/// vectors and byte-slice borrows while we build the `TensorView` list.
struct TensorEntry<'a> {
    name: &'a str,
    shape: Vec<usize>,
    data: &'a [u8],
}

/// Save a state dict using the `SafeTensors` format (`HuggingFace` standard).
///
/// The tensors are sorted by name for deterministic output. The resulting
/// file can be loaded by Python's `safetensors` library:
///
/// ```python
/// from safetensors import safe_open
/// with safe_open("model.safetensors", framework="numpy") as f:
///     weight = f.get_tensor("weight")
/// ```
pub fn save_safetensors<T: Float>(
    state: &StateDict<T>,
    path: impl AsRef<Path>,
) -> FerrotorchResult<()> {
    let path = path.as_ref();
    let dtype = st_dtype::<T>()?;

    // Collect tensor data so the byte slices live long enough.
    // We need to hold onto the data references while building TensorViews.
    let mut keys: Vec<&String> = state.keys().collect();
    keys.sort();

    // Build (name, TensorView) pairs. We need the tensor data to outlive the
    // TensorView, so we collect data slices first.
    let mut entries: Vec<TensorEntry<'_>> = Vec::with_capacity(keys.len());
    for key in &keys {
        let tensor = &state[*key];
        let data_slice = tensor
            .data()
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("failed to read tensor \"{key}\": {e}"),
            })?;
        let byte_data = as_le_bytes(data_slice);
        entries.push(TensorEntry {
            name: key.as_str(),
            shape: tensor.shape().to_vec(),
            data: byte_data,
        });
    }

    // Build TensorView objects. The safetensors crate requires Vec<(String, TensorView)>
    // or any IntoIterator<Item = (S, V)> where V: View.
    let views: Vec<(String, TensorView<'_>)> = entries
        .iter()
        .map(|entry| {
            let view = TensorView::new(dtype, entry.shape.clone(), entry.data).map_err(|e| {
                FerrotorchError::InvalidArgument {
                    message: format!("failed to create TensorView for \"{}\": {e}", entry.name),
                }
            });
            view.map(|v| (entry.name.to_string(), v))
        })
        .collect::<FerrotorchResult<Vec<_>>>()?;

    serialize_to_file(views, &None, path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to write safetensors file {}: {e}", path.display()),
    })?;

    Ok(())
}

/// Decode a single `TensorView` into a `Tensor<T>`, handling the bf16/f16
/// upcast and the element-size / dtype validation shared by both the
/// single-file and sharded load paths.
fn decode_view<T: Float>(name: &str, view: &TensorView<'_>) -> FerrotorchResult<Tensor<T>> {
    let shape: Vec<usize> = view.shape().to_vec();
    let byte_data = view.data();
    let numel: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };

    // Auto-cast f16/bf16 to the target type (f32 or f64).
    if view.dtype() == Dtype::F16 || view.dtype() == Dtype::BF16 {
        let is_bf16 = view.dtype() == Dtype::BF16;
        if byte_data.len() != numel * 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "tensor '{name}': expected {} bytes for {:?} with {numel} elements, got {}",
                    numel * 2,
                    view.dtype(),
                    byte_data.len()
                ),
            });
        }

        // CL-1127 part B (vectorized fast path): reinterpret the byte buffer
        // as `&[u16]` of LE-encoded half-floats in one shot, then delegate
        // the conversion to `half::{bf16,f16}::to_f32`. The `half` crate
        // exposes that as a `#[inline]` scalar function that LLVM
        // auto-vectorizes (`unpcklwd` + `pslld` for bf16 on x86-64,
        // `cvtph2ps` for f16 when F16C is available). The previous scalar
        // loop did byte-by-byte hi/lo assembly *plus* a fallible
        // `cast::<f32, T>` on every element, which inhibited SIMD.
        let raw_u16: &[u16] = bytemuck::cast_slice(byte_data);
        debug_assert_eq!(raw_u16.len(), numel);

        // Specialize the trivial `T == f32` path: the saturating cast helper
        // is a no-op for f32→f32 (every f32 maps to itself), so the loop
        // collapses to a pure `half::*_to_f32` map — exactly the shape the
        // audit (CL-1127) wanted so LLVM SIMD fires.
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let f32_data: Vec<f32> = if is_bf16 {
                raw_u16.iter().map(|&b| bf16_to_f32(b)).collect()
            } else {
                raw_u16.iter().map(|&b| half_to_f32(b)).collect()
            };
            // SAFETY: TypeId::of::<T>() == TypeId::of::<f32>() above implies
            // `Vec<f32>` and `Vec<T>` have identical layout (same element
            // size, alignment, drop semantics — both are `Copy` floats with
            // no Drop). The transmute is over the `Vec` itself (allocation +
            // length + capacity), not its contents, so no bit pattern is
            // re-interpreted and no aliasing arises. The `Vec<f32>` is moved
            // into `ManuallyDrop` before `from_raw_parts`, so the original
            // allocation is transferred wholesale exactly once.
            let typed: Vec<T> = unsafe {
                let mut md = std::mem::ManuallyDrop::new(f32_data);
                Vec::from_raw_parts(md.as_mut_ptr().cast::<T>(), md.len(), md.capacity())
            };
            return Tensor::from_storage(TensorStorage::cpu(typed), shape, false);
        }

        // Generic path (T == f64 today, or any narrower Float added later):
        // route every element through the saturating `cast::<f32, T>` to
        // keep the same error semantics as the previous scalar loop. The
        // half→f32 stage still vectorizes; only the per-element cast is
        // sequential.
        let mut float_data: Vec<T> = Vec::with_capacity(numel);
        for &bits in raw_u16 {
            let f32_val = if is_bf16 {
                bf16_to_f32(bits)
            } else {
                half_to_f32(bits)
            };
            float_data.push(cast::<f32, T>(f32_val)?);
        }
        return Tensor::from_storage(TensorStorage::cpu(float_data), shape, false);
    }

    // Auto-downcast F32 file to a smaller target (e.g. bf16).
    if view.dtype() == Dtype::F32 && std::mem::size_of::<T>() < 4 {
        if byte_data.len() != numel * 4 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "tensor '{name}': expected {} bytes for F32 with {numel} elements, got {}",
                    numel * 4,
                    byte_data.len()
                ),
            });
        }
        // CL-1127 part B: reinterpret the byte buffer as `&[f32]` in one
        // shot rather than walking 4-byte chunks with `try_into` +
        // `from_le_bytes` on every element. Every f32 bit pattern is valid
        // (including NaN payloads), so `bytemuck::try_cast_slice` either
        // succeeds (aligned input from a real safetensors header) or we
        // fall back to the scalar `from_le_bytes` loop (misaligned mmap).
        let mut data: Vec<T> = Vec::with_capacity(numel);
        match bytemuck::try_cast_slice::<u8, f32>(byte_data) {
            Ok(raw_f32) => {
                debug_assert_eq!(raw_f32.len(), numel);
                for &v in raw_f32 {
                    data.push(cast::<f32, T>(v)?);
                }
            }
            Err(_) => {
                for chunk in byte_data.chunks_exact(4) {
                    let arr: [u8; 4] =
                        chunk
                            .try_into()
                            .map_err(|e| FerrotorchError::InvalidArgument {
                                message: format!(
                                    "malformed safetensors chunk for tensor '{name}': {e}"
                                ),
                            })?;
                    let f32_val = f32::from_le_bytes(arr);
                    data.push(cast::<f32, T>(f32_val)?);
                }
            }
        }
        return Tensor::from_storage(TensorStorage::cpu(data), shape, false);
    }

    // Validate dtype for matching-size types.
    let expected = expected_dtype::<T>()?;
    let elem_size = std::mem::size_of::<T>();
    if view.dtype() != expected {
        return Err(FerrotorchError::DtypeMismatch {
            expected: format!("{expected:?}"),
            got: format!("{:?}", view.dtype()),
        });
    }

    // Validate byte length.
    let expected_bytes = numel * elem_size;
    if byte_data.len() != expected_bytes {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "tensor \"{name}\" has {} bytes but shape {:?} with dtype {:?} requires {} bytes",
                byte_data.len(),
                shape,
                expected,
                expected_bytes,
            ),
        });
    }

    // Reinterpret bytes as T values (little-endian assumption, same as
    // the safetensors specification).
    //
    // CL-1127 part B: fast path uses `bytemuck::try_cast_slice` to
    // reinterpret the byte buffer as a `&[f32]` / `&[f64]` slice (matching
    // `elem_size`), then transmutes the resulting `Vec` to `Vec<T>` via a
    // TypeId guard. This replaces the previous scalar `read_unaligned`
    // chunk-by-chunk loop with a single contiguous `to_vec` that LLVM
    // lowers to `memcpy` — material on Llama-class checkpoints where this
    // path dominates the F32 single-file load. If the source buffer is
    // misaligned (`cast_slice` fails), fall back to the safe scalar loop.
    let data: Vec<T> =
        if elem_size == 4 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            match bytemuck::try_cast_slice::<u8, f32>(byte_data) {
                Ok(slice) => {
                    let v_f32 = slice.to_vec();
                    // SAFETY: TypeId::of::<T>() == TypeId::of::<f32>() above
                    // implies T and f32 have identical layout. We transmute the
                    // `Vec` (allocation + length + capacity) wholesale; no
                    // element is reinterpreted, and `Vec::from_raw_parts` takes
                    // ownership of the original allocation exactly once.
                    unsafe {
                        let mut md = std::mem::ManuallyDrop::new(v_f32);
                        Vec::from_raw_parts(md.as_mut_ptr().cast::<T>(), md.len(), md.capacity())
                    }
                }
                Err(_) => byte_data
                    .chunks_exact(elem_size)
                    .map(|chunk| {
                        let arr: [u8; 4] = chunk.try_into().expect("chunks_exact 4");
                        let v = f32::from_le_bytes(arr);
                        // SAFETY: T == f32 here (TypeId match above), every f32
                        // bit pattern is valid (including NaN payloads), so the
                        // reinterpret-as-T is sound.
                        unsafe { std::mem::transmute_copy::<f32, T>(&v) }
                    })
                    .collect(),
            }
        } else if elem_size == 8 && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            match bytemuck::try_cast_slice::<u8, f64>(byte_data) {
                Ok(slice) => {
                    let v_f64 = slice.to_vec();
                    // SAFETY: see f32 branch above — identical layout argument.
                    unsafe {
                        let mut md = std::mem::ManuallyDrop::new(v_f64);
                        Vec::from_raw_parts(md.as_mut_ptr().cast::<T>(), md.len(), md.capacity())
                    }
                }
                Err(_) => byte_data
                    .chunks_exact(elem_size)
                    .map(|chunk| {
                        let arr: [u8; 8] = chunk.try_into().expect("chunks_exact 8");
                        let v = f64::from_le_bytes(arr);
                        // SAFETY: T == f64 here, every f64 bit pattern is valid.
                        unsafe { std::mem::transmute_copy::<f64, T>(&v) }
                    })
                    .collect(),
            }
        } else {
            // Generic fallback: a future `Float` type that's neither f32 nor
            // f64 lands here. `expected = st_dtype::<T>()` only succeeds for
            // size 4 or 8 today, so this branch is unreachable in practice.
            byte_data
                .chunks_exact(elem_size)
                .map(|chunk| {
                    let mut bytes = [0u8; 8];
                    bytes[..elem_size].copy_from_slice(chunk);
                    // SAFETY: `expected = st_dtype::<T>()` succeeded above only
                    // for `T: Float` with `size_of::<T>() == 4` (f32) or `8`
                    // (f64), and `elem_size = size_of::<T>()` is set from that
                    // match — so `elem_size` here is always 4 or 8, and
                    // `chunks_exact` hands us exactly that many bytes which we
                    // copy fully into `bytes`. `bytes` is an 8-byte stack
                    // array; we only read `elem_size` of its bytes, all of
                    // which are initialized by the `copy_from_slice`.
                    // `read_unaligned` requires no alignment, so the cast from
                    // `*const u8` to `*const T` at any address is sound. Every
                    // bit pattern is a valid `f32`/`f64` (NaNs included), so
                    // the read cannot produce an invalid value.
                    unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<T>()) }
                })
                .collect()
        };

    Tensor::from_storage(TensorStorage::cpu(data), shape, false)
}

/// Load a state dict from a `SafeTensors` file.
///
/// The dtype stored in the file must match the requested type `T`. For
/// example, loading an `F32` file into `StateDict<f64>` produces an error.
/// `bf16` and `f16` tensors are automatically up-cast to the target
/// `Float` type.
pub fn load_safetensors<T: Float>(path: impl AsRef<Path>) -> FerrotorchResult<StateDict<T>> {
    let path = path.as_ref();

    let file_data = std::fs::read(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to read safetensors file {}: {e}", path.display()),
    })?;

    let st =
        SafeTensors::deserialize(&file_data).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to parse safetensors file {}: {e}", path.display()),
        })?;

    let tensor_list = st.tensors();
    let state: StateDict<T> = decode_tensor_list::<T>(&tensor_list)?;
    Ok(state)
}

/// Decode every `(name, TensorView)` pair into an owned [`StateDict<T>`],
/// choosing serial vs rayon-parallel iteration based on the file's
/// dtype.
///
/// Rationale (ferrotorch#1178): a rayon `par_iter` over `decode_view`
/// is a clear win on `BF16` / `F16` checkpoints — each element
/// carries a `half::*_to_f32` upcast that LLVM auto-vectorizes and
/// scales linearly with worker count. The same `par_iter` is a clear
/// *loss* on native-`F32` / `F64` files (SD-1.5 `UNet`: 315 s serial
/// → 379 s parallel on a 16-core RTX 3090 box), where
/// `decode_view` is a pure mmap → `Vec<T>` memcpy bound by memory
/// bandwidth: extra workers only add page-fault and allocator
/// contention. The dispatcher therefore peeks at the first tensor's
/// dtype and picks the strategy that wins for that family. Files
/// are nearly always homogeneous in practice (one dtype per
/// checkpoint), so the single peek is sufficient.
///
/// An operator escape hatch (`FERROTORCH_FORCE_SERIAL_LOAD=1`) is
/// honored as a diagnostic flag — useful when chasing a regression
/// or running under tooling that mis-attributes rayon-thread
/// activity (e.g. `valgrind`, `perf record` without
/// `--call-graph=dwarf`).
fn decode_tensor_list<T: Float>(
    tensor_list: &[(String, TensorView<'_>)],
) -> FerrotorchResult<StateDict<T>> {
    let parallel_dtype = tensor_list
        .first()
        .is_some_and(|(_, v)| matches!(v.dtype(), Dtype::BF16 | Dtype::F16));
    let force_serial = std::env::var_os("FERROTORCH_FORCE_SERIAL_LOAD").is_some();
    if parallel_dtype && !force_serial {
        let decoded: FerrotorchResult<Vec<(String, Tensor<T>)>> = tensor_list
            .par_iter()
            .map(|(name, view)| decode_view::<T>(name, view).map(|t| (name.clone(), t)))
            .collect();
        Ok(decoded?.into_iter().collect())
    } else {
        let mut state: StateDict<T> = HashMap::with_capacity(tensor_list.len());
        for (name, view) in tensor_list {
            let tensor = decode_view::<T>(name, view)?;
            state.insert(name.clone(), tensor);
        }
        Ok(state)
    }
}

// ---------------------------------------------------------------------------
// Sharded loading (HuggingFace `model.safetensors.index.json`)
// ---------------------------------------------------------------------------

/// `HuggingFace` safetensors index file (`model.safetensors.index.json`).
///
/// Maps each tensor name to the shard file that contains it, plus a
/// total-size metadata field used for progress reporting.
///
/// `#[non_exhaustive]` so we can extend the index format (e.g. tensor
/// hashes, per-shard byte ranges) without forcing every reader to update
/// pattern matches; field access keeps working.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct SafeTensorsIndex {
    pub metadata: SafeTensorsIndexMetadata,
    pub weight_map: HashMap<String, String>,
}

/// Metadata section of the safetensors index.
///
/// `#[non_exhaustive]` for the same reason as [`SafeTensorsIndex`]:
/// `HuggingFace` already adds optional metadata fields between safetensors
/// versions and we want to absorb those without semver breaks.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct SafeTensorsIndexMetadata {
    /// Sum of the sizes (in bytes) of every tensor referenced by the index.
    pub total_size: u64,
}

impl SafeTensorsIndex {
    /// Parse an index file from disk.
    pub fn from_file(path: impl AsRef<Path>) -> FerrotorchResult<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to read index file {}: {e}", path.display()),
        })?;
        serde_json::from_slice::<SafeTensorsIndex>(&bytes).map_err(|e| {
            FerrotorchError::InvalidArgument {
                message: format!("failed to parse index file {}: {e}", path.display()),
            }
        })
    }

    /// Unique shard filenames, sorted lexicographically for deterministic
    /// load order.
    pub fn shard_files(&self) -> Vec<String> {
        let mut files: HashSet<&String> = self.weight_map.values().collect();
        let mut sorted: Vec<String> = files.drain().cloned().collect();
        sorted.sort();
        sorted
    }

    /// Tensor names grouped by the shard file that contains them.
    pub fn group_by_shard(&self) -> HashMap<String, Vec<String>> {
        let mut grouped: HashMap<String, Vec<String>> = HashMap::new();
        for (key, shard) in &self.weight_map {
            grouped.entry(shard.clone()).or_default().push(key.clone());
        }
        for keys in grouped.values_mut() {
            keys.sort();
        }
        grouped
    }
}

/// Load a sharded `HuggingFace` safetensors checkpoint from its
/// `model.safetensors.index.json`.
///
/// Shards are loaded one at a time (sorted by filename) and each tensor
/// the index maps to that shard is decoded into the returned
/// [`StateDict`]. Shards not referenced by the index are ignored; tensor
/// names that the index claims but the shard does not contain produce an
/// error.
///
/// Memory usage peaks at ≈ (one shard size) + (decoded shard size)
/// during each shard's decode — 8-10 GB for a 4 GB Llama 3 8B shard
/// at bf16→f32 upcast. Sequential loading keeps total RSS bounded
/// regardless of checkpoint size.
pub fn load_safetensors_sharded<T: Float>(
    index_path: impl AsRef<Path>,
) -> FerrotorchResult<StateDict<T>> {
    let index_path = index_path.as_ref();
    let dir: PathBuf = index_path
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    let index = SafeTensorsIndex::from_file(index_path)?;

    let grouped = index.group_by_shard();

    // Deterministic shard order for reproducible loads. Even with parallel
    // shard decoding the *final* HashMap insertion order is irrelevant, but
    // we still sort the work units so the rayon partitioner sees a stable
    // workload and so that any error message ("shard X is missing tensor
    // Y") names the same shard run-to-run for the same inputs.
    let mut shard_files: Vec<&String> = grouped.keys().collect();
    shard_files.sort();

    // CL-1127 part A: decode every shard in parallel. Shards live in
    // independent files, the safetensors header lookups don't share state,
    // and `decode_view` only touches its own borrowed bytes plus per-call
    // `Vec<T>` allocations. We collect each shard's tensors into its own
    // `StateDict<T>` and merge serially at the end (the final merge is
    // O(total tensors) hash inserts, dwarfed by the per-shard byte
    // traversal we just parallelized). The previous loop was strictly
    // sequential even though each shard load is I/O- and CPU-bound and
    // embarrassingly parallel across files.
    let per_shard: Vec<StateDict<T>> = shard_files
        .par_iter()
        .map(|shard_file| {
            let shard_path = dir.join(shard_file.as_str());
            let expected_keys = grouped
                .get(*shard_file)
                .expect("grouped map built from shard_files keys");
            load_one_shard_owned::<T>(&shard_path, expected_keys)
        })
        .collect::<FerrotorchResult<Vec<_>>>()?;

    let mut state: StateDict<T> = HashMap::with_capacity(index.weight_map.len());
    for shard_state in per_shard {
        state.extend(shard_state);
    }

    // Cross-check: every key declared in the index must now be present.
    for key in index.weight_map.keys() {
        if !state.contains_key(key) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("safetensors index declares \"{key}\" but no shard provided it"),
            });
        }
    }

    Ok(state)
}

/// Read one safetensors shard from disk and return the decoded tensors as
/// an owned [`StateDict`], independent of any caller-side aggregate state.
///
/// Used by [`load_safetensors_sharded`] (CL-1127 part A) so that each
/// shard can be decoded on its own rayon worker thread without
/// synchronizing on a shared map. Any tensor present in the shard but not
/// in `expected_keys` is skipped (the HF index is authoritative). Any
/// `expected_keys` entry that is not in the shard produces a
/// [`FerrotorchError::InvalidArgument`] so a corrupt index is caught early.
fn load_one_shard_owned<T: Float>(
    shard_path: &Path,
    expected_keys: &[String],
) -> FerrotorchResult<StateDict<T>> {
    let file_data = std::fs::read(shard_path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to read shard {}: {e}", shard_path.display()),
    })?;
    let st =
        SafeTensors::deserialize(&file_data).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to parse shard {}: {e}", shard_path.display()),
        })?;

    decode_shard_tensors::<T>(shard_path, expected_keys, &st)
}

/// Decode the subset of `st.tensors()` named in `expected_keys` into a
/// fresh `StateDict<T>`. Shared between the read-into-memory and mmap
/// shard paths so the dtype handling and missing-key validation stay in
/// one place.
fn decode_shard_tensors<T: Float>(
    shard_path: &Path,
    expected_keys: &[String],
    st: &SafeTensors<'_>,
) -> FerrotorchResult<StateDict<T>> {
    let expected_set: HashSet<&str> = expected_keys
        .iter()
        .map(std::string::String::as_str)
        .collect();
    let mut state: StateDict<T> = HashMap::with_capacity(expected_keys.len());
    let mut found: HashSet<String> = HashSet::with_capacity(expected_keys.len());

    let tensors = st.tensors();
    for (name, view) in &tensors {
        if !expected_set.contains(name.as_str()) {
            continue;
        }
        let tensor = decode_view::<T>(name, view)?;
        state.insert(name.clone(), tensor);
        found.insert(name.clone());
    }

    // Report missing keys — the index said this shard has them, but it didn't.
    for key in expected_keys {
        if !found.contains(key) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "safetensors shard {} is missing tensor \"{key}\" declared in the index",
                    shard_path.display()
                ),
            });
        }
    }

    Ok(state)
}

/// Read one safetensors shard from disk and merge its tensors into `state`.
///
/// Thin sequential wrapper around [`load_one_shard_owned`] retained for
/// the progress-callback and filtered-load paths, which still load
/// shard-by-shard sequentially to preserve their per-shard semantics (one
/// callback fire per shard / predicate evaluated under the caller's
/// thread).
fn load_one_shard_into<T: Float>(
    shard_path: &Path,
    expected_keys: &[String],
    state: &mut StateDict<T>,
) -> FerrotorchResult<()> {
    let shard_state = load_one_shard_owned::<T>(shard_path, expected_keys)?;
    state.extend(shard_state);
    Ok(())
}

/// Auto-detect whether `path` is a single safetensors file or a
/// `model.safetensors.index.json` and dispatch to the correct loader.
///
/// Detection rule:
/// - `*.index.json` → [`load_safetensors_sharded`]
/// - anything else → [`load_safetensors`] (single-file)
pub fn load_safetensors_auto<T: Float>(path: impl AsRef<Path>) -> FerrotorchResult<StateDict<T>> {
    let p = path.as_ref();
    let filename = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if filename.ends_with(".index.json") {
        load_safetensors_sharded(p)
    } else {
        load_safetensors(p)
    }
}

/// Progress information passed to a [`load_safetensors_sharded_with_progress`]
/// callback before each shard is opened.
///
/// `#[non_exhaustive]` because additional progress signals (bytes
/// loaded, ETA, current tensor name) may be added in future versions
/// without breaking callbacks that pattern-match on existing fields.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ShardProgress<'a> {
    /// 0-based index of the shard about to be loaded.
    pub shard_index: usize,
    /// Total number of shards declared in the index.
    pub shard_count: usize,
    /// File name of the shard (no directory prefix).
    pub shard_file: &'a str,
    /// Number of tensors expected from this shard.
    pub tensors_in_shard: usize,
    /// Cumulative number of tensors loaded so far (across previous shards).
    pub tensors_loaded_so_far: usize,
    /// Total tensors across all shards (`weight_map.len()` from the index).
    pub total_tensors: usize,
}

/// Sharded loader that calls `progress` once before each shard is opened.
///
/// Useful for progress bars / logging on huge models (Llama 3 70B + ships
/// ~140 GB across many shards). The callback is purely advisory — its
/// return value is ignored. (#586)
pub fn load_safetensors_sharded_with_progress<T, F>(
    index_path: impl AsRef<Path>,
    mut progress: F,
) -> FerrotorchResult<StateDict<T>>
where
    T: Float,
    F: FnMut(ShardProgress<'_>),
{
    let index_path = index_path.as_ref();
    let dir: PathBuf = index_path
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    let index = SafeTensorsIndex::from_file(index_path)?;

    let grouped = index.group_by_shard();
    let total_tensors = index.weight_map.len();
    let mut state: StateDict<T> = HashMap::with_capacity(total_tensors);

    let mut shard_files: Vec<&String> = grouped.keys().collect();
    shard_files.sort();
    let shard_count = shard_files.len();

    for (shard_index, shard_file) in shard_files.iter().enumerate() {
        let expected_keys = grouped
            .get(*shard_file)
            .expect("grouped map built from shard_files keys");
        progress(ShardProgress {
            shard_index,
            shard_count,
            shard_file: shard_file.as_str(),
            tensors_in_shard: expected_keys.len(),
            tensors_loaded_so_far: state.len(),
            total_tensors,
        });
        let shard_path = dir.join(shard_file);
        load_one_shard_into::<T>(&shard_path, expected_keys, &mut state)?;
    }

    for key in index.weight_map.keys() {
        if !state.contains_key(key) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("safetensors index declares \"{key}\" but no shard provided it"),
            });
        }
    }

    Ok(state)
}

/// Memory-map a safetensors file and decode its tensors. (#587)
///
/// Behavior identical to [`load_safetensors`] except the on-disk bytes are
/// memory-mapped instead of read into a heap `Vec<u8>`. This halves peak
/// RSS during the decode phase: the file pages are owned by the OS page
/// cache, and the only Rust-side allocations are the decoded `Tensor<T>`
/// buffers.
///
/// # Safety / concurrency
///
/// The returned `StateDict` does not borrow from the mmap — `decode_view`
/// copies each tensor into a fresh `Vec<T>`. The mmap is dropped before
/// this function returns, so external mutation of the file after the call
/// returns can't corrupt the result.
///
/// While the mmap is live, callers must not modify the underlying file
/// (this matches the `safetensors` Python library's contract).
pub fn load_safetensors_mmap<T: Float>(path: impl AsRef<Path>) -> FerrotorchResult<StateDict<T>> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to open safetensors file {}: {e}", path.display()),
    })?;
    // SAFETY: the mmap is dropped before this function returns. We do not
    // expose the borrowed bytes to the caller — `decode_view` copies into
    // owned `Tensor<T>` buffers below. The file must not be mutated while
    // the mmap is live, which matches the safetensors library contract.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to mmap safetensors file {}: {e}", path.display()),
    })?;

    let st = SafeTensors::deserialize(&mmap[..]).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to parse safetensors file {}: {e}", path.display()),
    })?;

    let tensor_list = st.tensors();
    let mut state: StateDict<T> = HashMap::with_capacity(tensor_list.len());
    for (name, view) in &tensor_list {
        let tensor = decode_view::<T>(name, view)?;
        state.insert(name.clone(), tensor);
    }
    Ok(state)
}

/// Memory-mapped sharded loader. Identical contract to
/// [`load_safetensors_sharded`] but each shard is mmap'd instead of read
/// into a heap `Vec<u8>`. Useful for huge HF transformer checkpoints
/// where the doubled buffer (raw bytes + decoded tensors) would otherwise
/// peak at 2× the model's on-disk size. (#587)
pub fn load_safetensors_sharded_mmap<T: Float>(
    index_path: impl AsRef<Path>,
) -> FerrotorchResult<StateDict<T>> {
    let index_path = index_path.as_ref();
    let dir: PathBuf = index_path
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    let index = SafeTensorsIndex::from_file(index_path)?;

    let grouped = index.group_by_shard();

    let mut shard_files: Vec<&String> = grouped.keys().collect();
    shard_files.sort();

    // CL-1127 part A: mirror of [`load_safetensors_sharded`] — each shard's
    // mmap is opened on a rayon worker, decoded into an owned
    // `StateDict<T>`, and merged serially at the end. mmap doesn't share
    // state between shards, so this is sound; the parallel mapping cost vs.
    // sequential is bounded by the file system's page cache and is dwarfed
    // by the decode-time savings.
    let per_shard: Vec<StateDict<T>> = shard_files
        .par_iter()
        .map(|shard_file| {
            let shard_path = dir.join(shard_file.as_str());
            let expected_keys = grouped
                .get(*shard_file)
                .expect("grouped map built from shard_files keys");
            load_one_shard_owned_mmap::<T>(&shard_path, expected_keys)
        })
        .collect::<FerrotorchResult<Vec<_>>>()?;

    let mut state: StateDict<T> = HashMap::with_capacity(index.weight_map.len());
    for shard_state in per_shard {
        state.extend(shard_state);
    }

    for key in index.weight_map.keys() {
        if !state.contains_key(key) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("safetensors index declares \"{key}\" but no shard provided it"),
            });
        }
    }

    Ok(state)
}

/// mmap counterpart of [`load_one_shard_owned`]. The mmap is dropped at
/// the end of this function — owned `Tensor<T>` buffers in the returned
/// `StateDict<T>` are independent of it. (CL-1127 part A.)
fn load_one_shard_owned_mmap<T: Float>(
    shard_path: &Path,
    expected_keys: &[String],
) -> FerrotorchResult<StateDict<T>> {
    let file = File::open(shard_path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to open shard {}: {e}", shard_path.display()),
    })?;
    // SAFETY: see `load_safetensors_mmap`. mmap dropped before return; no
    // borrow escapes.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to mmap shard {}: {e}", shard_path.display()),
    })?;
    let st = SafeTensors::deserialize(&mmap[..]).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to parse shard {}: {e}", shard_path.display()),
    })?;

    decode_shard_tensors::<T>(shard_path, expected_keys, &st)
}

/// Sharded loader that returns only tensors whose name is accepted by
/// `predicate`. Matches the typical "load only encoder weights" or
/// "load layer 12 only" patterns used by inference servers and by
/// adapter / `LoRA` training that wants to skip the base model. (#586)
///
/// Shards that contain no accepted tensors are still opened to validate
/// the index, but their tensor data is dropped after the predicate check.
pub fn load_safetensors_sharded_filtered<T, F>(
    index_path: impl AsRef<Path>,
    predicate: F,
) -> FerrotorchResult<StateDict<T>>
where
    T: Float,
    F: Fn(&str) -> bool,
{
    let index_path = index_path.as_ref();
    let dir: PathBuf = index_path
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    let index = SafeTensorsIndex::from_file(index_path)?;

    let grouped = index.group_by_shard();
    let mut state: StateDict<T> = HashMap::new();

    let mut shard_files: Vec<&String> = grouped.keys().collect();
    shard_files.sort();

    for shard_file in shard_files {
        let expected_keys = grouped
            .get(shard_file)
            .expect("grouped map built from shard_files keys");
        let filtered: Vec<String> = expected_keys
            .iter()
            .filter(|k| predicate(k.as_str()))
            .cloned()
            .collect();
        if filtered.is_empty() {
            continue;
        }
        let shard_path = dir.join(shard_file);
        load_one_shard_into::<T>(&shard_path, &filtered, &mut state)?;
    }

    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;
    use std::collections::HashMap;

    fn make_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        let storage = TensorStorage::cpu(data);
        Tensor::from_storage(storage, shape, false).unwrap()
    }

    fn make_tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        let storage = TensorStorage::cpu(data);
        Tensor::from_storage(storage, shape, false).unwrap()
    }

    #[test]
    fn test_save_load_roundtrip_f32() {
        let mut state: StateDict<f32> = HashMap::new();
        state.insert(
            "weight".to_string(),
            make_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        );
        state.insert(
            "bias".to_string(),
            make_tensor_f32(vec![0.1, 0.2, 0.3], vec![3]),
        );

        let dir = std::env::temp_dir().join("ferrotorch_test_st_f32");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();
        let loaded: StateDict<f32> = load_safetensors(&path).unwrap();

        assert_eq!(loaded.len(), 2);

        let w = &loaded["weight"];
        assert_eq!(w.shape(), &[2, 3]);
        let w_data = w.data().unwrap();
        assert_eq!(w_data, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let b = &loaded["bias"];
        assert_eq!(b.shape(), &[3]);
        let b_data = b.data().unwrap();
        assert_eq!(b_data, &[0.1f32, 0.2, 0.3]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is an arbitrary round-trip value, not π.
    fn test_save_load_roundtrip_f64() {
        let mut state: StateDict<f64> = HashMap::new();
        state.insert(
            "layer.weight".to_string(),
            make_tensor_f64(vec![1.0, -2.5, 3.14, 0.0, 99.9, -0.001], vec![3, 2]),
        );
        state.insert(
            "layer.bias".to_string(),
            make_tensor_f64(vec![0.5, -0.5, 1.0], vec![3]),
        );

        let dir = std::env::temp_dir().join("ferrotorch_test_st_f64");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();
        let loaded: StateDict<f64> = load_safetensors(&path).unwrap();

        assert_eq!(loaded.len(), 2);

        let w = &loaded["layer.weight"];
        assert_eq!(w.shape(), &[3, 2]);
        let w_data = w.data().unwrap();
        assert_eq!(w_data, &[1.0f64, -2.5, 3.14, 0.0, 99.9, -0.001]);

        let b = &loaded["layer.bias"];
        assert_eq!(b.shape(), &[3]);
        let b_data = b.data().unwrap();
        assert_eq!(b_data, &[0.5f64, -0.5, 1.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_correct_tensor_names_and_shapes() {
        let mut state: StateDict<f32> = HashMap::new();
        state.insert(
            "encoder.0.weight".to_string(),
            make_tensor_f32(vec![1.0; 12], vec![3, 4]),
        );
        state.insert(
            "encoder.0.bias".to_string(),
            make_tensor_f32(vec![0.0; 3], vec![3]),
        );
        state.insert(
            "decoder.weight".to_string(),
            make_tensor_f32(vec![2.0; 8], vec![4, 2]),
        );

        let dir = std::env::temp_dir().join("ferrotorch_test_st_names");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();

        // Read back and verify via the raw safetensors crate that names/shapes
        // are correct (independent of our load function).
        let file_data = std::fs::read(&path).unwrap();
        let st = SafeTensors::deserialize(&file_data).unwrap();

        let mut names: Vec<String> = st.names().iter().map(|s| (*s).clone()).collect();
        names.sort();
        assert_eq!(
            names,
            vec!["decoder.weight", "encoder.0.bias", "encoder.0.weight"],
        );

        let enc_w = st.tensor("encoder.0.weight").unwrap();
        assert_eq!(enc_w.shape(), &[3, 4]);
        assert_eq!(enc_w.dtype(), Dtype::F32);

        let enc_b = st.tensor("encoder.0.bias").unwrap();
        assert_eq!(enc_b.shape(), &[3]);

        let dec_w = st.tensor("decoder.weight").unwrap();
        assert_eq!(dec_w.shape(), &[4, 2]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_file_readable_by_safetensors_crate() {
        // Verify the file we produce is valid safetensors by deserializing it
        // directly with the safetensors crate (not our wrapper).
        let mut state: StateDict<f32> = HashMap::new();
        state.insert(
            "x".to_string(),
            make_tensor_f32(vec![1.0, 2.0, 3.0], vec![3]),
        );

        let dir = std::env::temp_dir().join("ferrotorch_test_st_valid");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();

        let file_data = std::fs::read(&path).unwrap();
        let st = SafeTensors::deserialize(&file_data).unwrap();

        assert_eq!(st.len(), 1);
        let tv = st.tensor("x").unwrap();
        assert_eq!(tv.dtype(), Dtype::F32);
        assert_eq!(tv.shape(), &[3]);
        // Verify the raw bytes decode correctly.
        let bytes = tv.data();
        assert_eq!(bytes.len(), 3 * 4); // 3 elements * 4 bytes each
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0f32, 2.0, 3.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_missing_file() {
        let result = load_safetensors::<f32>("/nonexistent/path/model.safetensors");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("failed to read"));
    }

    #[test]
    fn test_dtype_mismatch() {
        // Save as f32, try to load as f64.
        let mut state: StateDict<f32> = HashMap::new();
        state.insert("x".to_string(), make_tensor_f32(vec![1.0, 2.0], vec![2]));

        let dir = std::env::temp_dir().join("ferrotorch_test_st_dtype_mm");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();

        let result = load_safetensors::<f64>(&path);
        assert!(result.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_empty_state_dict() {
        let state: StateDict<f32> = HashMap::new();

        let dir = std::env::temp_dir().join("ferrotorch_test_st_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();
        let loaded: StateDict<f32> = load_safetensors(&path).unwrap();
        assert!(loaded.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_high_rank_tensor() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let mut state: StateDict<f32> = HashMap::new();
        state.insert(
            "conv.weight".to_string(),
            make_tensor_f32(data.clone(), vec![2, 3, 2, 2]),
        );

        let dir = std::env::temp_dir().join("ferrotorch_test_st_4d");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();
        let loaded: StateDict<f32> = load_safetensors(&path).unwrap();

        let t = &loaded["conv.weight"];
        assert_eq!(t.shape(), &[2, 3, 2, 2]);
        let loaded_data = t.data().unwrap();
        assert_eq!(loaded_data, data.as_slice());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_1d_tensor() {
        let mut state: StateDict<f32> = HashMap::new();
        state.insert("vec".to_string(), make_tensor_f32(vec![42.0], vec![1]));

        let dir = std::env::temp_dir().join("ferrotorch_test_st_1d");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");

        save_safetensors(&state, &path).unwrap();
        let loaded: StateDict<f32> = load_safetensors(&path).unwrap();

        let v = &loaded["vec"];
        assert_eq!(v.shape(), &[1]);
        assert_eq!(v.data().unwrap(), &[42.0f32]);

        std::fs::remove_dir_all(&dir).ok();
    }

    // -- Sharded loader tests (#507) ----------------------------------------

    /// Write `state` to `<dir>/<filename>` as a single safetensors file.
    fn write_shard(dir: &Path, filename: &str, state: &StateDict<f32>) -> std::path::PathBuf {
        let path = dir.join(filename);
        save_safetensors(state, &path).unwrap();
        path
    }

    /// Write a `model.safetensors.index.json` file describing the union
    /// of the given shards.
    fn write_index(
        dir: &Path,
        filename: &str,
        shards: &[(&str, &StateDict<f32>)],
    ) -> std::path::PathBuf {
        use std::fmt::Write as _;

        let mut weight_map: Vec<(String, String)> = Vec::new();
        let mut total_size: u64 = 0;
        for (shard_file, sd) in shards {
            for (key, tensor) in *sd {
                weight_map.push((key.clone(), shard_file.to_string()));
                total_size += (tensor.numel() * std::mem::size_of::<f32>()) as u64;
            }
        }
        // Build minimal JSON manually to avoid pulling serde_json as a
        // dev-only dependency everywhere it's not already available.
        let mut json = String::from("{\"metadata\":{\"total_size\":");
        json.push_str(&total_size.to_string());
        json.push_str("},\"weight_map\":{");
        for (i, (k, v)) in weight_map.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            write!(json, "\"{k}\":\"{v}\"").unwrap();
        }
        json.push_str("}}");

        let path = dir.join(filename);
        std::fs::write(&path, json).unwrap();
        path
    }

    #[test]
    fn sharded_loader_merges_all_shards() {
        let tmp = tempfile::tempdir().unwrap();

        let mut shard_a: StateDict<f32> = HashMap::new();
        shard_a.insert(
            "model.layers.0.q_proj.weight".to_string(),
            make_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        );
        shard_a.insert(
            "model.embed_tokens.weight".to_string(),
            make_tensor_f32(vec![0.1; 12], vec![4, 3]),
        );

        let mut shard_b: StateDict<f32> = HashMap::new();
        shard_b.insert(
            "model.layers.1.q_proj.weight".to_string(),
            make_tensor_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
        );
        shard_b.insert(
            "lm_head.weight".to_string(),
            make_tensor_f32(vec![0.5; 6], vec![2, 3]),
        );

        write_shard(tmp.path(), "model-00001-of-00002.safetensors", &shard_a);
        write_shard(tmp.path(), "model-00002-of-00002.safetensors", &shard_b);
        let index_path = write_index(
            tmp.path(),
            "model.safetensors.index.json",
            &[
                ("model-00001-of-00002.safetensors", &shard_a),
                ("model-00002-of-00002.safetensors", &shard_b),
            ],
        );

        let merged: StateDict<f32> = load_safetensors_sharded(&index_path).unwrap();
        assert_eq!(merged.len(), 4);

        assert_eq!(
            merged["model.layers.0.q_proj.weight"].data().unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            merged["model.layers.1.q_proj.weight"].data().unwrap(),
            &[5.0f32, 6.0, 7.0, 8.0]
        );
        assert_eq!(merged["model.embed_tokens.weight"].shape(), &[4, 3]);
        assert_eq!(merged["lm_head.weight"].shape(), &[2, 3]);
    }

    #[test]
    fn sharded_loader_respects_index_over_shard_contents() {
        // If a shard physically contains tensors the index does not
        // list for that shard, they must be ignored.
        let tmp = tempfile::tempdir().unwrap();
        let mut shard_a: StateDict<f32> = HashMap::new();
        shard_a.insert("a".to_string(), make_tensor_f32(vec![1.0], vec![1]));
        shard_a.insert("stray".to_string(), make_tensor_f32(vec![9.9], vec![1]));
        write_shard(tmp.path(), "a.safetensors", &shard_a);

        // Index only lists "a".
        let json = r#"{"metadata":{"total_size":4},"weight_map":{"a":"a.safetensors"}}"#;
        let idx = tmp.path().join("model.safetensors.index.json");
        std::fs::write(&idx, json).unwrap();

        let merged: StateDict<f32> = load_safetensors_sharded(&idx).unwrap();
        assert_eq!(merged.len(), 1);
        assert!(merged.contains_key("a"));
        assert!(!merged.contains_key("stray"));
    }

    #[test]
    fn sharded_loader_rejects_index_with_missing_tensor() {
        let tmp = tempfile::tempdir().unwrap();
        let mut shard: StateDict<f32> = HashMap::new();
        shard.insert("present".to_string(), make_tensor_f32(vec![1.0], vec![1]));
        write_shard(tmp.path(), "s.safetensors", &shard);

        // Index claims "missing" is in s.safetensors, but it isn't.
        let json = r#"{
            "metadata":{"total_size":4},
            "weight_map":{"present":"s.safetensors","missing":"s.safetensors"}
        }"#;
        let idx = tmp.path().join("model.safetensors.index.json");
        std::fs::write(&idx, json).unwrap();

        let err = load_safetensors_sharded::<f32>(&idx).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("missing"), "error message was: {msg}");
    }

    #[test]
    fn sharded_loader_rejects_missing_shard_file() {
        let tmp = tempfile::tempdir().unwrap();
        let json = r#"{
            "metadata":{"total_size":4},
            "weight_map":{"a":"nonexistent.safetensors"}
        }"#;
        let idx = tmp.path().join("model.safetensors.index.json");
        std::fs::write(&idx, json).unwrap();
        assert!(load_safetensors_sharded::<f32>(&idx).is_err());
    }

    #[test]
    fn sharded_loader_rejects_malformed_index_json() {
        let tmp = tempfile::tempdir().unwrap();
        let idx = tmp.path().join("model.safetensors.index.json");
        std::fs::write(&idx, "{ this is not valid").unwrap();
        assert!(load_safetensors_sharded::<f32>(&idx).is_err());
    }

    #[test]
    fn safe_tensors_index_exposes_shard_files_and_groups() {
        let tmp = tempfile::tempdir().unwrap();
        let mut shard_a: StateDict<f32> = HashMap::new();
        shard_a.insert("k1".to_string(), make_tensor_f32(vec![0.0], vec![1]));
        shard_a.insert("k2".to_string(), make_tensor_f32(vec![0.0], vec![1]));
        let mut shard_b: StateDict<f32> = HashMap::new();
        shard_b.insert("k3".to_string(), make_tensor_f32(vec![0.0], vec![1]));

        let idx_path = write_index(
            tmp.path(),
            "model.safetensors.index.json",
            &[("a.safetensors", &shard_a), ("b.safetensors", &shard_b)],
        );
        let idx = SafeTensorsIndex::from_file(&idx_path).unwrap();

        assert_eq!(idx.shard_files(), vec!["a.safetensors", "b.safetensors"]);
        let grouped = idx.group_by_shard();
        assert_eq!(
            grouped["a.safetensors"],
            vec!["k1".to_string(), "k2".to_string()]
        );
        assert_eq!(grouped["b.safetensors"], vec!["k3".to_string()]);
    }

    #[test]
    fn load_safetensors_auto_dispatches_on_filename() {
        let tmp = tempfile::tempdir().unwrap();

        // Single-file case: .safetensors → load_safetensors.
        let mut sd: StateDict<f32> = HashMap::new();
        sd.insert("w".to_string(), make_tensor_f32(vec![1.0, 2.0], vec![2]));
        let single_path = tmp.path().join("model.safetensors");
        save_safetensors(&sd, &single_path).unwrap();
        let loaded: StateDict<f32> = load_safetensors_auto(&single_path).unwrap();
        assert_eq!(loaded.len(), 1);

        // Sharded case: .index.json → load_safetensors_sharded.
        let mut shard: StateDict<f32> = HashMap::new();
        shard.insert("x".to_string(), make_tensor_f32(vec![3.0], vec![1]));
        let shard_subdir = tmp.path().join("sharded");
        std::fs::create_dir_all(&shard_subdir).unwrap();
        write_shard(&shard_subdir, "m-1.safetensors", &shard);
        let idx_path = write_index(
            &shard_subdir,
            "model.safetensors.index.json",
            &[("m-1.safetensors", &shard)],
        );
        let loaded_sharded: StateDict<f32> = load_safetensors_auto(&idx_path).unwrap();
        assert_eq!(loaded_sharded.len(), 1);
        assert!(loaded_sharded.contains_key("x"));
    }

    /// End-to-end load of the downloaded Meta-Llama-3-8B checkpoint.
    /// Ignored by default (pulls 16 GB from disk and needs 30+ GB RAM
    /// for the bf16→f32 upcast). Run with:
    ///   `cargo test --release -p ferrotorch-serialize -- --ignored llama3_8b_sharded_load`
    #[test]
    #[ignore = "requires Meta-Llama-3-8B weights in the HF cache and ~30GB RAM"]
    fn llama3_8b_sharded_load() {
        // Resolve the snapshot directory dynamically so we don't hard-code
        // the commit hash.
        let base = dirs_home()
            .join(".cache")
            .join("huggingface")
            .join("hub")
            .join("models--meta-llama--Meta-Llama-3-8B")
            .join("snapshots");
        let snapshot = std::fs::read_dir(&base)
            .expect("HF cache snapshots dir missing")
            .next()
            .expect("no snapshot in HF cache")
            .unwrap()
            .path();
        let idx = snapshot.join("model.safetensors.index.json");
        assert!(idx.exists(), "index.json missing at {}", idx.display());

        let state: StateDict<f32> = load_safetensors_sharded(&idx).unwrap();
        // Llama 3 8B has:
        //   1 embed_tokens + 1 norm + 1 lm_head + 32 * (
        //     2 layernorms + q_proj + k_proj + v_proj + o_proj +
        //     gate_proj + up_proj + down_proj
        //   ) = 3 + 32 * 9 = 291 tensors.
        assert_eq!(state.len(), 291);

        // Spot-check key shapes.
        assert_eq!(state["model.embed_tokens.weight"].shape(), &[128_256, 4096]);
        assert_eq!(state["lm_head.weight"].shape(), &[128_256, 4096]);
        assert_eq!(
            state["model.layers.0.self_attn.q_proj.weight"].shape(),
            &[4096, 4096]
        );
        // GQA: K/V have num_kv_heads=8 * head_dim=128 = 1024 output rows.
        assert_eq!(
            state["model.layers.0.self_attn.k_proj.weight"].shape(),
            &[1024, 4096]
        );
        assert_eq!(
            state["model.layers.0.self_attn.v_proj.weight"].shape(),
            &[1024, 4096]
        );
    }

    /// Helper for the ignored real-weights test: HOME directory lookup
    /// without a dirs crate dep.
    fn dirs_home() -> std::path::PathBuf {
        std::env::var_os("HOME")
            .map(std::path::PathBuf::from)
            .expect("$HOME not set")
    }

    // -----------------------------------------------------------------------
    // Sharded loader: progress callback + filtered variants (#586)
    // -----------------------------------------------------------------------

    #[test]
    fn sharded_loader_progress_callback_fires_per_shard() {
        let tmp = tempfile::tempdir().unwrap();

        let mut shard_a: StateDict<f32> = HashMap::new();
        shard_a.insert(
            "model.layers.0.q_proj.weight".to_string(),
            make_tensor_f32(vec![1.0, 2.0], vec![2]),
        );
        let mut shard_b: StateDict<f32> = HashMap::new();
        shard_b.insert(
            "model.layers.1.q_proj.weight".to_string(),
            make_tensor_f32(vec![3.0, 4.0], vec![2]),
        );

        write_shard(tmp.path(), "model-00001-of-00002.safetensors", &shard_a);
        write_shard(tmp.path(), "model-00002-of-00002.safetensors", &shard_b);
        let index_path = write_index(
            tmp.path(),
            "model.safetensors.index.json",
            &[
                ("model-00001-of-00002.safetensors", &shard_a),
                ("model-00002-of-00002.safetensors", &shard_b),
            ],
        );

        let mut events: Vec<(usize, usize, String, usize, usize, usize)> = Vec::new();
        let _state: StateDict<f32> = load_safetensors_sharded_with_progress(&index_path, |p| {
            events.push((
                p.shard_index,
                p.shard_count,
                p.shard_file.to_string(),
                p.tensors_in_shard,
                p.tensors_loaded_so_far,
                p.total_tensors,
            ));
        })
        .unwrap();

        assert_eq!(events.len(), 2);
        // First fires before any tensor is loaded.
        assert_eq!(events[0].0, 0);
        assert_eq!(events[0].1, 2);
        assert_eq!(events[0].4, 0);
        assert_eq!(events[0].5, 2);
        // Second fires after first shard is done.
        assert_eq!(events[1].0, 1);
        assert_eq!(events[1].4, 1);
    }

    #[test]
    fn sharded_loader_filter_keeps_only_matching_keys() {
        let tmp = tempfile::tempdir().unwrap();
        let mut shard_a: StateDict<f32> = HashMap::new();
        shard_a.insert(
            "model.embed_tokens.weight".to_string(),
            make_tensor_f32(vec![0.1; 6], vec![2, 3]),
        );
        shard_a.insert(
            "model.layers.0.q_proj.weight".to_string(),
            make_tensor_f32(vec![1.0; 4], vec![2, 2]),
        );
        let mut shard_b: StateDict<f32> = HashMap::new();
        shard_b.insert(
            "model.layers.5.q_proj.weight".to_string(),
            make_tensor_f32(vec![5.0; 4], vec![2, 2]),
        );
        shard_b.insert(
            "lm_head.weight".to_string(),
            make_tensor_f32(vec![0.2; 6], vec![2, 3]),
        );

        write_shard(tmp.path(), "model-00001-of-00002.safetensors", &shard_a);
        write_shard(tmp.path(), "model-00002-of-00002.safetensors", &shard_b);
        let index_path = write_index(
            tmp.path(),
            "model.safetensors.index.json",
            &[
                ("model-00001-of-00002.safetensors", &shard_a),
                ("model-00002-of-00002.safetensors", &shard_b),
            ],
        );

        // Only load tensors whose name contains ".q_proj.".
        let filtered: StateDict<f32> =
            load_safetensors_sharded_filtered(&index_path, |k| k.contains(".q_proj.")).unwrap();

        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains_key("model.layers.0.q_proj.weight"));
        assert!(filtered.contains_key("model.layers.5.q_proj.weight"));
        assert!(!filtered.contains_key("model.embed_tokens.weight"));
        assert!(!filtered.contains_key("lm_head.weight"));
    }

    // -----------------------------------------------------------------------
    // mmap loaders (#587)
    // -----------------------------------------------------------------------

    #[test]
    fn mmap_loader_matches_read_loader_for_single_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");

        let mut sd: StateDict<f32> = HashMap::new();
        sd.insert(
            "weight".to_string(),
            make_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
        );
        sd.insert("bias".to_string(), make_tensor_f32(vec![0.5; 3], vec![3]));
        save_safetensors::<f32>(&sd, &path).unwrap();

        let from_read = load_safetensors::<f32>(&path).unwrap();
        let from_mmap = load_safetensors_mmap::<f32>(&path).unwrap();
        assert_eq!(from_read.len(), from_mmap.len());
        for (k, v) in &from_read {
            let m = from_mmap.get(k).expect("mmap loader missing key");
            assert_eq!(v.shape(), m.shape());
            assert_eq!(v.data().unwrap(), m.data().unwrap());
        }
    }

    #[test]
    fn mmap_sharded_loader_matches_read_loader() {
        let tmp = tempfile::tempdir().unwrap();

        let mut shard_a: StateDict<f32> = HashMap::new();
        shard_a.insert(
            "model.layers.0.q_proj.weight".to_string(),
            make_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        );
        let mut shard_b: StateDict<f32> = HashMap::new();
        shard_b.insert(
            "model.layers.1.q_proj.weight".to_string(),
            make_tensor_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
        );

        write_shard(tmp.path(), "model-00001-of-00002.safetensors", &shard_a);
        write_shard(tmp.path(), "model-00002-of-00002.safetensors", &shard_b);
        let index_path = write_index(
            tmp.path(),
            "model.safetensors.index.json",
            &[
                ("model-00001-of-00002.safetensors", &shard_a),
                ("model-00002-of-00002.safetensors", &shard_b),
            ],
        );

        let from_read: StateDict<f32> = load_safetensors_sharded(&index_path).unwrap();
        let from_mmap: StateDict<f32> = load_safetensors_sharded_mmap(&index_path).unwrap();
        assert_eq!(from_read.len(), from_mmap.len());
        for (k, v) in &from_read {
            let m = from_mmap.get(k).expect("mmap sharded loader missing key");
            assert_eq!(v.data().unwrap(), m.data().unwrap());
        }
    }

    #[test]
    fn mmap_loader_returns_owned_data_after_file_drop() {
        // The mmap is dropped before the loader returns. Confirm the
        // resulting Tensors hold owned heap data by mutating-and-rereading
        // the underlying file: the loaded tensor must keep its values.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");

        let mut sd: StateDict<f32> = HashMap::new();
        sd.insert(
            "w".to_string(),
            make_tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        );
        save_safetensors::<f32>(&sd, &path).unwrap();
        let loaded = load_safetensors_mmap::<f32>(&path).unwrap();

        // Overwrite the file with garbage. Loaded tensor must still be valid.
        std::fs::write(&path, b"garbage that is not safetensors").unwrap();
        let v = loaded["w"].data().unwrap().to_vec();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn sharded_loader_filter_skips_shard_with_no_matches() {
        let tmp = tempfile::tempdir().unwrap();
        let mut shard_a: StateDict<f32> = HashMap::new();
        shard_a.insert(
            "model.embed_tokens.weight".to_string(),
            make_tensor_f32(vec![0.0; 4], vec![2, 2]),
        );
        let mut shard_b: StateDict<f32> = HashMap::new();
        shard_b.insert(
            "lm_head.weight".to_string(),
            make_tensor_f32(vec![0.0; 4], vec![2, 2]),
        );

        write_shard(tmp.path(), "model-00001-of-00002.safetensors", &shard_a);
        write_shard(tmp.path(), "model-00002-of-00002.safetensors", &shard_b);
        let index_path = write_index(
            tmp.path(),
            "model.safetensors.index.json",
            &[
                ("model-00001-of-00002.safetensors", &shard_a),
                ("model-00002-of-00002.safetensors", &shard_b),
            ],
        );

        // No tensors match — empty result, no error.
        let result: StateDict<f32> =
            load_safetensors_sharded_filtered(&index_path, |k| k.contains("nonexistent")).unwrap();
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // CL-1127 Part C: vectorized bf16/f16 upcast — bit-exact verification.
    //
    // The optimisation in `decode_view` swaps the scalar byte-by-byte hi/lo
    // assembly + per-element `cast::<f32, T>` loop for a `bytemuck::cast_slice`
    // + `half::*::to_f32` map. The cast must remain *byte-exact* with the
    // pre-optimisation behaviour, because the audit framing in #1127 is
    // explicitly "pure perf — bit-exact must hold". These tests build small
    // bf16 / f16 safetensors files via the upstream `safetensors` crate
    // (byte-identical to what HF Python emits) and verify the decoded f32
    // buffer matches the reference `half::*::to_f32` mapping applied to
    // the same raw bits.
    // -----------------------------------------------------------------------

    /// Write a tiny safetensors file holding a single tensor at the given
    /// dtype with the given raw little-endian bytes. We bypass our own
    /// `save_safetensors` (which only emits F32/F64) and call the upstream
    /// crate directly, the same way `HuggingFace` does.
    fn write_typed_st(path: &Path, name: &str, dtype: Dtype, shape: &[usize], bytes: &[u8]) {
        let view = TensorView::new(dtype, shape.to_vec(), bytes).unwrap();
        safetensors::serialize_to_file([(name.to_string(), view)], &None, path).unwrap();
    }

    #[test]
    fn bf16_upcast_is_bit_exact() {
        // Build a bf16 tensor whose bits exercise every interesting region:
        //   ±0, ±1.0, ±small, ±large, ±Inf, NaN, denormal.
        let bf16_bits: Vec<u16> = vec![
            0x0000, // +0.0
            0x8000, // -0.0
            0x3F80, // +1.0
            0xBF80, // -1.0
            0x4049, // +π upper bits (~3.140625)
            0x7F80, // +Inf
            0xFF80, // -Inf
            0x7FC0, // qNaN
            0x0001, // smallest positive subnormal
            0x4080, // +4.0
            0xC080, // -4.0
            0x4248, // +50.0
        ];
        let raw: Vec<u8> = bf16_bits.iter().flat_map(|b| b.to_le_bytes()).collect();

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bf16.safetensors");
        write_typed_st(&path, "w", Dtype::BF16, &[bf16_bits.len()], &raw);

        let loaded: StateDict<f32> = load_safetensors(&path).unwrap();
        let got = loaded["w"].data().unwrap();

        // Reference: each bf16 bit pattern routed through `bf16::to_f32` is
        // the bit-exact contract we are claiming to preserve.
        let expected: Vec<f32> = bf16_bits
            .iter()
            .map(|&b| half::bf16::from_bits(b).to_f32())
            .collect();
        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            // `to_bits` so NaN compares equal when bit patterns match.
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "bf16 element {i}: got bits 0x{:08X}, want 0x{:08X}",
                g.to_bits(),
                e.to_bits()
            );
        }
    }

    #[test]
    fn f16_upcast_is_bit_exact() {
        // Same coverage as the bf16 test but with IEEE 754 binary16 bits.
        let f16_bits: Vec<u16> = vec![
            0x0000, // +0.0
            0x8000, // -0.0
            0x3C00, // +1.0
            0xBC00, // -1.0
            0x4248, // +π in f16 (~3.140625)
            0x7C00, // +Inf
            0xFC00, // -Inf
            0x7E00, // qNaN
            0x0001, // smallest positive subnormal
            0x7BFF, // max finite (~65504)
            0xFBFF, // min finite (-65504)
            0x4900, // +10.0
        ];
        let raw: Vec<u8> = f16_bits.iter().flat_map(|b| b.to_le_bytes()).collect();

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("f16.safetensors");
        write_typed_st(&path, "w", Dtype::F16, &[f16_bits.len()], &raw);

        let loaded: StateDict<f32> = load_safetensors(&path).unwrap();
        let got = loaded["w"].data().unwrap();

        let expected: Vec<f32> = f16_bits
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect();
        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "f16 element {i}: got bits 0x{:08X}, want 0x{:08X}",
                g.to_bits(),
                e.to_bits()
            );
        }
    }

    #[test]
    fn bf16_upcast_to_f64_is_bit_exact() {
        // The bf16/f16→T path has a generic branch when T != f32 (e.g.
        // T == f64). Exercise it explicitly so the saturating-cast branch
        // is also bit-exact.
        let bf16_bits: Vec<u16> = vec![
            0x0000, 0x3F80, 0xBF80, 0x4049, 0x7F80, 0xFF80, 0x4080, 0xC080,
        ];
        let raw: Vec<u8> = bf16_bits.iter().flat_map(|b| b.to_le_bytes()).collect();

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bf16_to_f64.safetensors");
        write_typed_st(&path, "w", Dtype::BF16, &[bf16_bits.len()], &raw);

        let loaded: StateDict<f64> = load_safetensors(&path).unwrap();
        let got = loaded["w"].data().unwrap();

        // Reference: f64 cast must agree with the f32 widening for every
        // bit pattern produced by the bf16→f32 stage. The `cast<f32,f64>`
        // widening is bit-exact for all finite values because every f32
        // is representable as f64; Inf passes through.
        let expected: Vec<f64> = bf16_bits
            .iter()
            .map(|&b| f64::from(half::bf16::from_bits(b).to_f32()))
            .collect();
        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "bf16→f64 element {i}: got bits 0x{:016X}, want 0x{:016X}",
                g.to_bits(),
                e.to_bits()
            );
        }
    }

    #[test]
    fn dispatcher_multi_tensor_bf16_round_trips_through_parallel_path() {
        // Multi-tensor BF16 file — first-tensor dtype check sends this
        // to the rayon `par_iter` branch of the dispatcher. Each
        // tensor is small enough that rayon's threadpool startup cost
        // dominates wall-clock here; the point is *correctness*
        // (parallel decode produces identical output to serial). The
        // production speedup comes at SD-1.5 / Llama scale, not in
        // tests. (ferrotorch#1178)
        let bits_a: Vec<u16> = vec![0x3F80, 0x4000, 0x4040, 0x4080]; // 1.0, 2.0, 3.0, 4.0
        let bits_b: Vec<u16> = vec![0xBF80, 0xC000, 0x4248, 0x4049]; // -1.0, -2.0, 50.0, ~π
        let bits_c: Vec<u16> = vec![0x0000, 0x8000, 0x7F80, 0xFF80]; // +0, -0, +Inf, -Inf
        let raw_a: Vec<u8> = bits_a.iter().flat_map(|b| b.to_le_bytes()).collect();
        let raw_b: Vec<u8> = bits_b.iter().flat_map(|b| b.to_le_bytes()).collect();
        let raw_c: Vec<u8> = bits_c.iter().flat_map(|b| b.to_le_bytes()).collect();

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("multi_bf16.safetensors");

        let view_a = TensorView::new(Dtype::BF16, vec![bits_a.len()], &raw_a).unwrap();
        let view_b = TensorView::new(Dtype::BF16, vec![bits_b.len()], &raw_b).unwrap();
        let view_c = TensorView::new(Dtype::BF16, vec![bits_c.len()], &raw_c).unwrap();
        safetensors::serialize_to_file(
            [
                ("a".to_string(), view_a),
                ("b".to_string(), view_b),
                ("c".to_string(), view_c),
            ],
            &None,
            &path,
        )
        .unwrap();

        let loaded: StateDict<f32> = load_safetensors(&path).unwrap();
        assert_eq!(loaded.len(), 3);
        for (name, bits) in [("a", &bits_a), ("b", &bits_b), ("c", &bits_c)] {
            let got = loaded[name].data().unwrap();
            let expected: Vec<f32> = bits
                .iter()
                .map(|&b| half::bf16::from_bits(b).to_f32())
                .collect();
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    e.to_bits(),
                    "bf16 tensor {name} element {i}: got bits 0x{:08X}, want 0x{:08X}",
                    g.to_bits(),
                    e.to_bits()
                );
            }
        }
    }

    #[test]
    fn dispatcher_force_serial_env_var_matches_default_for_bf16() {
        // The `FERROTORCH_FORCE_SERIAL_LOAD` escape hatch must produce
        // bit-identical output to the default parallel path on a
        // BF16 file (the only dtype family the dispatcher routes
        // through `par_iter`). Verifies the two code paths agree on
        // every element. (ferrotorch#1178)
        let bf16_bits: Vec<u16> = (0..64u16).map(|i| 0x3F80 + i).collect();
        let raw: Vec<u8> = bf16_bits.iter().flat_map(|b| b.to_le_bytes()).collect();

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("force_serial.safetensors");
        write_typed_st(&path, "w", Dtype::BF16, &[bf16_bits.len()], &raw);

        // Default load: dispatcher takes the parallel branch.
        let parallel: StateDict<f32> = load_safetensors(&path).unwrap();
        let parallel_data = parallel["w"].data().unwrap().to_vec();

        // SAFETY: tests run single-threaded in the rust test harness
        // unless `--test-threads` is set; in CI that flag is unset.
        // `FERROTORCH_FORCE_SERIAL_LOAD` is read by `decode_tensor_list`
        // inside the same process; toggling it here is local to this
        // test's scope (cleared in the `defer`-shaped block below).
        let key = "FERROTORCH_FORCE_SERIAL_LOAD";
        unsafe {
            std::env::set_var(key, "1");
        }
        let serial: StateDict<f32> = load_safetensors(&path).unwrap();
        unsafe {
            std::env::remove_var(key);
        }
        let serial_data = serial["w"].data().unwrap().to_vec();

        assert_eq!(parallel_data.len(), serial_data.len());
        for (i, (p, s)) in parallel_data.iter().zip(serial_data.iter()).enumerate() {
            assert_eq!(
                p.to_bits(),
                s.to_bits(),
                "force-serial vs parallel disagree at element {i}: \
                 parallel=0x{:08X}, serial=0x{:08X}",
                p.to_bits(),
                s.to_bits(),
            );
        }
    }

    #[test]
    fn f32_reinterpret_is_bit_exact() {
        // The same-size T path uses bytemuck::try_cast_slice + Vec transmute.
        // Verify it preserves every f32 bit pattern, including NaN payloads
        // and ±0 — these are exactly the patterns the previous
        // `read_unaligned` loop preserved, so the new code must match.
        let f32_bits: Vec<u32> = vec![
            0x0000_0000, // +0.0
            0x8000_0000, // -0.0
            0x3F80_0000, // 1.0
            0xBF80_0000, // -1.0
            0x4049_0FDB, // π
            0x7F80_0000, // +Inf
            0xFF80_0000, // -Inf
            0x7FC0_0000, // canonical qNaN
            0x7FC0_DEAD, // qNaN with payload
            0x0080_0000, // smallest positive normal
            0x007F_FFFF, // largest positive subnormal
        ];
        let raw: Vec<u8> = f32_bits.iter().flat_map(|b| b.to_le_bytes()).collect();

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("f32_bits.safetensors");
        write_typed_st(&path, "w", Dtype::F32, &[f32_bits.len()], &raw);

        let loaded: StateDict<f32> = load_safetensors(&path).unwrap();
        let got = loaded["w"].data().unwrap();

        assert_eq!(got.len(), f32_bits.len());
        for (i, (g, e)) in got.iter().zip(f32_bits.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                *e,
                "f32 element {i}: got 0x{:08X}, want 0x{e:08X}",
                g.to_bits()
            );
        }
    }

    #[test]
    fn sharded_parallel_loader_merges_many_shards() {
        // CL-1127 part A coverage: spin up 8 shards in parallel and verify
        // the result matches a sequentially-built reference. Any race or
        // partial merge would fail this test.
        let tmp = tempfile::tempdir().unwrap();
        let mut shards: Vec<(String, StateDict<f32>)> = Vec::new();
        for shard_idx in 0..8u32 {
            let mut sd: StateDict<f32> = HashMap::new();
            for tensor_idx in 0..4u32 {
                let key = format!("shard_{shard_idx}.tensor_{tensor_idx}.weight");
                let base = (shard_idx * 4 + tensor_idx) as f32;
                let data: Vec<f32> = (0..16).map(|j| base + j as f32 * 0.125).collect();
                sd.insert(key, make_tensor_f32(data, vec![4, 4]));
            }
            let fname = format!("model-{:05}-of-00008.safetensors", shard_idx + 1);
            write_shard(tmp.path(), &fname, &sd);
            shards.push((fname, sd));
        }

        let shard_refs: Vec<(&str, &StateDict<f32>)> =
            shards.iter().map(|(f, sd)| (f.as_str(), sd)).collect();
        let index_path = write_index(tmp.path(), "model.safetensors.index.json", &shard_refs);

        let parallel: StateDict<f32> = load_safetensors_sharded(&index_path).unwrap();
        assert_eq!(parallel.len(), 8 * 4);

        // Reference: every (key, data) pair must survive the parallel load
        // bit-exactly.
        for (_, sd) in &shards {
            for (key, tensor) in sd {
                let loaded = parallel.get(key).expect("missing key after parallel load");
                assert_eq!(loaded.shape(), tensor.shape());
                assert_eq!(loaded.data().unwrap(), tensor.data().unwrap());
            }
        }
    }

    #[test]
    fn sharded_parallel_mmap_loader_merges_many_shards() {
        // Same scenario but through the mmap path so the rayon parallelism
        // on Mmap-backed SafeTensors deserialisation is exercised.
        let tmp = tempfile::tempdir().unwrap();
        let mut shards: Vec<(String, StateDict<f32>)> = Vec::new();
        for shard_idx in 0..6u32 {
            let mut sd: StateDict<f32> = HashMap::new();
            for tensor_idx in 0..3u32 {
                let key = format!("mmap_shard_{shard_idx}.t_{tensor_idx}");
                let base = (shard_idx * 3 + tensor_idx) as f32;
                let data: Vec<f32> = (0..9).map(|j| base + j as f32).collect();
                sd.insert(key, make_tensor_f32(data, vec![3, 3]));
            }
            let fname = format!("mmap-{:05}-of-00006.safetensors", shard_idx + 1);
            write_shard(tmp.path(), &fname, &sd);
            shards.push((fname, sd));
        }

        let shard_refs: Vec<(&str, &StateDict<f32>)> =
            shards.iter().map(|(f, sd)| (f.as_str(), sd)).collect();
        let index_path = write_index(tmp.path(), "model.safetensors.index.json", &shard_refs);

        let parallel: StateDict<f32> = load_safetensors_sharded_mmap(&index_path).unwrap();
        assert_eq!(parallel.len(), 6 * 3);

        for (_, sd) in &shards {
            for (key, tensor) in sd {
                let loaded = parallel
                    .get(key)
                    .expect("missing key after parallel mmap load");
                assert_eq!(loaded.shape(), tensor.shape());
                assert_eq!(loaded.data().unwrap(), tensor.data().unwrap());
            }
        }
    }
}
