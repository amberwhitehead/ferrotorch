//! GPU-dequantized GGUF loading for [`LlamaGpuInferencer`].
//!
//! This module bridges two GPU runtimes that, in the current cubecl
//! release, do **not** share a CUDA context or expose pointer handoff:
//!
//! * **ferrotorch-cubecl** owns the GGUF dequantization kernels
//!   (`dequantize_q*_to_gpu`): given the host-split block buffers it
//!   JIT-compiles a `#[cube]` kernel and produces an `f32`
//!   [`cubecl::server::Handle`] that lives on the cubecl `CudaRuntime`
//!   device.
//! * **ferrotorch-llama**'s GPU inferencer stores every weight as a
//!   `cudarc::driver::CudaSlice<u16>` (bf16 bit layout) bound to a
//!   `ferrotorch_gpu::GpuDevice`'s cudarc context.
//!
//! ## The bridge is a host bounce (documented limitation)
//!
//! cubecl's `Handle` and cudarc's `CudaSlice` are backed by **separate
//! CUDA contexts** with no published API to alias a device pointer
//! across them. The only correct cross-runtime transfer today is:
//!
//! ```text
//! cubecl dequant kernel  →  f32 Handle (cubecl device)
//!   →  client.read_one(handle)  →  Vec<u8>  →  &[f32]  (host)
//!   →  bf16::from_f32 narrowing →  Vec<u16>  (host bf16 bits)
//!   →  device.stream().clone_htod →  CudaSlice<u16>  (cudarc device)
//! ```
//!
//! The host bounce is the honest current cost: the dequant runs on the
//! GPU (real kernel, real device memory), but the result is pulled to
//! host and re-uploaded into the cudarc context. The
//! cubecl-output-stays-on-device optimization is **blocked** until
//! cubecl exposes context sharing / device-pointer handoff with cudarc.
//! When that lands, [`gpu_dequantize_to_bf16_cudarc`] is the single
//! function to change.
//!
//! ## ComputeClient context duplication (documented limitation)
//!
//! [`cubecl_cuda_client`] constructs a **fresh** `CudaRuntime::client`
//! for the requested ordinal. There is no published API to reuse the
//! `GpuDevice`'s existing cudarc `CudaContext` inside cubecl, so the
//! cubecl client opens its own context on the same physical device. On
//! a single-GPU box this is one extra context's worth of bookkeeping
//! (and the JIT module cache), not duplicated weights — the weights
//! only ever exist once on each side of a single tensor's bounce. This
//! is the hardest sub-problem of the bridge and is intentionally scoped
//! as an MVP.
//!
//! ## REQ status
//!
//! See `.design/ferrotorch-llama/gpu_gguf_loader.md` for the full REQ
//! table with impl + production-consumer anchors.

#![cfg(feature = "cuda")]

use std::path::Path;

use cubecl::prelude::*;
use cubecl_cuda::{CudaDevice, CudaRuntime};
use cudarc::driver::CudaSlice;
use ferrotorch_core::{FerrotorchError, FerrotorchResult};
use ferrotorch_cubecl::{
    dequantize_q4_0_to_gpu, dequantize_q4_1_to_gpu, dequantize_q5_0_to_gpu, dequantize_q5_1_to_gpu,
    dequantize_q8_0_to_gpu, dequantize_q8_1_to_gpu, split_q4_0_blocks, split_q4_1_blocks,
    split_q5_0_blocks, split_q5_1_blocks, split_q8_0_blocks, split_q8_1_blocks,
};
use ferrotorch_gpu::GpuDevice;
use ferrotorch_serialize::gguf::{GgmlType, GgufFile, dequantize_gguf_tensor, load_gguf_mmap};
use half::bf16;

use crate::config::LlamaConfig;
use crate::gguf_remap::gguf_key_to_hf;
use crate::gpu::LlamaGpuInferencer;

/// Build a cubecl `CudaRuntime` compute client for the given device
/// ordinal.
///
/// MVP: opens a **fresh** cubecl CUDA context on the physical device at
/// `device_ordinal`. There is no published cubecl API to adopt an
/// existing cudarc `CudaContext`, so the cubecl client and the
/// `ferrotorch_gpu::GpuDevice` hold independent contexts on the same
/// GPU. See the module docs for the context-duplication limitation.
#[must_use]
pub fn cubecl_cuda_client(device_ordinal: usize) -> ComputeClient<CudaRuntime> {
    let device = CudaDevice {
        index: device_ordinal,
    };
    CudaRuntime::client(&device)
}

/// Map a [`ferrotorch_serialize`] `GgmlType` to the matching cubecl
/// dequant routine and run the **GPU bridge** for one tensor's raw
/// quantized bytes, returning a `cudarc` `CudaSlice<u16>` of bf16 bits.
///
/// For quantized types (`Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`Q8_0`/`Q8_1`) this
/// is the cross-runtime bridge:
///
/// 1. host-split the raw block stream (`split_q*_blocks`),
/// 2. dispatch the cubecl dequant kernel (`dequantize_q*_to_gpu`) →
///    f32 [`cubecl::server::Handle`] on the cubecl device,
/// 3. `client.read_one` the handle back to host f32,
/// 4. narrow each f32 to bf16 bits ([`bf16::from_f32`]),
/// 5. `clone_htod` the `Vec<u16>` into a cudarc `CudaSlice<u16>` on
///    `device`'s context.
///
/// For the scalar types (`F32`, `F16`) there is no quant kernel; the
/// caller should use the host fallback (`gpu_dequantize_scalar_to_bf16_cudarc`).
/// Passing a scalar type here returns [`FerrotorchError::InvalidArgument`].
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if `num_elements` is not
/// a multiple of the 32-element GGUF block size, if `ggml_type` is a
/// scalar (non-quantized) type, or if `client.read_one` fails. Returns
/// [`FerrotorchError::Gpu`] for any cudarc driver error during the
/// host-to-device upload.
pub fn gpu_dequantize_to_bf16_cudarc(
    client: &ComputeClient<CudaRuntime>,
    raw_quant_bytes: &[u8],
    ggml_type: GgmlType,
    num_elements: usize,
    device: &GpuDevice,
) -> FerrotorchResult<CudaSlice<u16>> {
    const BLOCK: usize = 32;
    if num_elements % BLOCK != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "gpu_dequantize_to_bf16_cudarc: num_elements ({num_elements}) must be a \
                 multiple of the GGUF block size {BLOCK} for quantized type {ggml_type:?}"
            ),
        });
    }
    let num_blocks = num_elements / BLOCK;

    // Phase 1+2: host-split + cubecl GPU dequant → f32 Handle on device.
    let handle = match ggml_type {
        GgmlType::Q4_0 => {
            let (scales, nibbles) = split_q4_0_blocks(raw_quant_bytes, num_blocks);
            dequantize_q4_0_to_gpu::<CudaRuntime>(client, &scales, &nibbles, num_elements)
        }
        GgmlType::Q4_1 => {
            let (scales, mins, nibbles) = split_q4_1_blocks(raw_quant_bytes, num_blocks);
            dequantize_q4_1_to_gpu::<CudaRuntime>(client, &scales, &mins, &nibbles, num_elements)
        }
        GgmlType::Q5_0 => {
            let (scales, qh, nibbles) = split_q5_0_blocks(raw_quant_bytes, num_blocks);
            dequantize_q5_0_to_gpu::<CudaRuntime>(client, &scales, &qh, &nibbles, num_elements)
        }
        GgmlType::Q5_1 => {
            let (scales, mins, qh, nibbles) = split_q5_1_blocks(raw_quant_bytes, num_blocks);
            dequantize_q5_1_to_gpu::<CudaRuntime>(
                client,
                &scales,
                &mins,
                &qh,
                &nibbles,
                num_elements,
            )
        }
        GgmlType::Q8_0 => {
            let (scales, bytes) = split_q8_0_blocks(raw_quant_bytes, num_blocks);
            dequantize_q8_0_to_gpu::<CudaRuntime>(client, &scales, &bytes, num_elements)
        }
        GgmlType::Q8_1 => {
            let (scales, mins, bytes) = split_q8_1_blocks(raw_quant_bytes, num_blocks);
            dequantize_q8_1_to_gpu::<CudaRuntime>(client, &scales, &mins, &bytes, num_elements)
        }
        GgmlType::F32 | GgmlType::F16 => {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "gpu_dequantize_to_bf16_cudarc: {ggml_type:?} is a scalar (non-quantized) \
                     type with no cubecl dequant kernel; use the host scalar fallback"
                ),
            });
        }
    };

    // Phase 3: read the f32 Handle back to host (the documented host bounce).
    let bytes = client
        .read_one(handle)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("cubecl read_one (GGUF dequant handle) failed: {e}"),
        })?;
    let f32_host: &[f32] = f32::from_bytes(&bytes);
    debug_assert_eq!(f32_host.len(), num_elements);

    // Phase 4: narrow f32 → bf16 bits on host.
    let bf16_bits: Vec<u16> = f32_host
        .iter()
        .take(num_elements)
        .map(|&x| bf16::from_f32(x).to_bits())
        .collect();

    // Phase 5: upload into the cudarc context as CudaSlice<u16>.
    device
        .stream()
        .clone_htod(&bf16_bits)
        .map_err(|e| FerrotorchError::Gpu {
            source: Box::new(e),
        })
}

/// Host-dequantize a non-quantized (`F32` / `F16`) GGUF tensor straight
/// to a cudarc `CudaSlice<u16>` of bf16 bits.
///
/// Quantized tensors go through the GPU bridge
/// ([`gpu_dequantize_to_bf16_cudarc`]); the scalar formats have no
/// cubecl kernel, so we dequant on host via the reference
/// `ferrotorch_serialize::gguf::dequantize_gguf_tensor` path (same
/// numerical oracle the bridge is validated against) and upload.
///
/// # Errors
///
/// Forwards [`dequantize_gguf_tensor`] errors and returns
/// [`FerrotorchError::Gpu`] for any cudarc upload error.
fn gpu_dequantize_scalar_to_bf16_cudarc(
    file: &GgufFile,
    tensor_name: &str,
    device: &GpuDevice,
) -> FerrotorchResult<CudaSlice<u16>> {
    let tensor = dequantize_gguf_tensor(file, tensor_name)?;
    let data = tensor.data()?;
    let bf16_bits: Vec<u16> = data.iter().map(|&x| bf16::from_f32(x).to_bits()).collect();
    device
        .stream()
        .clone_htod(&bf16_bits)
        .map_err(|e| FerrotorchError::Gpu {
            source: Box::new(e),
        })
}

/// Dequantize one GGUF tensor (any supported type) to a cudarc
/// `CudaSlice<u16>` of bf16 bits, choosing the GPU bridge for quantized
/// types and the host fallback for scalar types.
///
/// This is the per-tensor dispatch used by [`bf16_state_dict_from_gguf_gpu`].
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if `tensor_name` is not
/// present in `file`, plus the error surfaces of the underlying bridge
/// / host paths.
fn gguf_tensor_to_bf16_cudarc(
    client: &ComputeClient<CudaRuntime>,
    file: &GgufFile,
    tensor_name: &str,
    device: &GpuDevice,
) -> FerrotorchResult<CudaSlice<u16>> {
    let info = file
        .tensors
        .iter()
        .find(|t| t.name == tensor_name)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("GGUF tensor {tensor_name:?} not found"),
        })?;
    let num_elements: usize = if info.dims.is_empty() {
        1
    } else {
        info.dims.iter().map(|&d| d as usize).product()
    };

    match info.ggml_type {
        GgmlType::F32 | GgmlType::F16 => {
            gpu_dequantize_scalar_to_bf16_cudarc(file, tensor_name, device)
        }
        quantized => {
            // Slice the raw on-disk bytes for this tensor out of the data
            // section, identical to `dequantize_gguf_tensor`'s range math.
            let block_bytes = quantized_block_bytes(quantized);
            let num_blocks = num_elements / 32;
            let byte_len = num_blocks * block_bytes;
            let offset = info.offset as usize;
            let raw = file.data().get(offset..offset + byte_len).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!(
                        "GGUF tensor {tensor_name:?} needs bytes [{offset}..{}] but data \
                         section is too small",
                        offset + byte_len
                    ),
                }
            })?;
            gpu_dequantize_to_bf16_cudarc(client, raw, quantized, num_elements, device)
        }
    }
}

/// On-disk block byte size for a quantized GGML type (mirrors the
/// private `GgmlType::block_bytes` in ferrotorch-serialize, which is not
/// exported). Only the quantized variants are reachable here.
fn quantized_block_bytes(ty: GgmlType) -> usize {
    match ty {
        GgmlType::Q4_0 => 18,
        GgmlType::Q4_1 => 20,
        GgmlType::Q5_0 => 22,
        GgmlType::Q5_1 => 24,
        GgmlType::Q8_0 => 34,
        GgmlType::Q8_1 => 40,
        // Scalar types never reach this helper (the caller branches first).
        GgmlType::F32 => 4,
        GgmlType::F16 => 2,
    }
}

impl LlamaGpuInferencer {
    /// Load a GGUF (llama.cpp) quantized checkpoint, dequantizing every
    /// quantized tensor on the GPU via the cubecl→cudarc bridge, and
    /// assemble a GPU-resident inferencer.
    ///
    /// `config` must describe the architecture the GGUF stores (vocab,
    /// hidden size, layer count, head counts). Deriving a full
    /// [`LlamaConfig`] from GGUF metadata alone is intentionally NOT done
    /// here: the existing host path (`gguf_to_hf_state_dict` →
    /// `LlamaForCausalLM::load_hf_state_dict`) also takes an explicit
    /// config, because GGUF's metadata→config mapping (rope scaling,
    /// activation, tie flag) is architecture-specific and lossy. Pass the
    /// matching `LlamaConfig::llama3_8b()` / `llama2_7b()` / etc.
    ///
    /// Tensor naming: GGUF uses `token_embd.weight` / `blk.{i}.attn_q.weight`;
    /// these are remapped to the HF names [`LlamaGpuInferencer::new`]
    /// consumes via [`gguf_key_to_hf`].
    ///
    /// Each weight is dequantized to bf16 bits and handed to
    /// [`LlamaGpuInferencer::new`], which validates shapes and owns the
    /// final upload. Quantized tensors run the GPU dequant kernel (real
    /// device compute); the bf16 host narrowing + re-upload is the
    /// documented host bounce.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if the GGUF cannot be
    /// opened/parsed, a required tensor is missing, or `config` fails
    /// validation. Returns [`FerrotorchError::Gpu`] for any CUDA driver
    /// error, and [`FerrotorchError::ShapeMismatch`] if a GGUF tensor's
    /// shape disagrees with the config-derived expectation.
    pub fn from_gguf_gpu_dequant(
        gguf_path: &Path,
        config: LlamaConfig,
        device: GpuDevice,
    ) -> FerrotorchResult<Self> {
        config.validate()?;
        let file = load_gguf_mmap(gguf_path)?;
        let client = cubecl_cuda_client(device.ordinal());

        let state = bf16_state_dict_from_gguf_gpu(&client, &file, &device)?;
        // `new` re-uploads the bf16 host bits; this keeps the single,
        // shape-validated assembly path (tied-embeddings handling, rope
        // caches, per-layer plumbing) in one place. The substantive GPU
        // work — the dequant kernels — has already run inside
        // `bf16_state_dict_from_gguf_gpu`.
        LlamaGpuInferencer::new(config, state, device)
    }
}

/// Dequantize every GGUF tensor on the GPU bridge, narrow to bf16, and
/// assemble an HF-keyed `StateDict<bf16>` ready for
/// [`LlamaGpuInferencer::new`].
///
/// The on-device dequant + host readback happens per tensor inside
/// [`gguf_tensor_to_bf16_cudarc`]; here we re-download each
/// `CudaSlice<u16>` to host bf16 so it can flow through the existing
/// `StateDict<bf16>` → `new` assembly path. (The re-download is part of
/// the same documented host-bounce envelope: the substantive
/// cross-runtime correctness — cubecl kernel output == host reference —
/// is the load-bearing property, verified in
/// `tests/divergence_gpu_gguf_bridge.rs`.)
///
/// # Errors
///
/// Forwards every per-tensor error from [`gguf_tensor_to_bf16_cudarc`].
fn bf16_state_dict_from_gguf_gpu(
    client: &ComputeClient<CudaRuntime>,
    file: &GgufFile,
    device: &GpuDevice,
) -> FerrotorchResult<ferrotorch_nn::StateDict<bf16>> {
    use ferrotorch_core::{Tensor, TensorStorage};

    let mut state: ferrotorch_nn::StateDict<bf16> =
        std::collections::HashMap::with_capacity(file.tensors.len());

    let infos: Vec<(String, Vec<usize>)> = file
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.dims.iter().map(|&d| d as usize).collect()))
        .collect();

    for (gguf_name, shape) in &infos {
        let Some(hf_name) = gguf_key_to_hf(gguf_name) else {
            // Drop tensors the Llama model does not consume (rope_freqs,
            // tokenizer scores, etc.) — matches `gguf_to_hf_state_dict`.
            continue;
        };
        // Run the GPU bridge for this tensor, then re-download to host
        // bf16 to fit the StateDict<bf16> assembly contract.
        let cuda_bits = gguf_tensor_to_bf16_cudarc(client, file, gguf_name, device)?;
        let host_bits: Vec<u16> =
            device
                .stream()
                .clone_dtoh(&cuda_bits)
                .map_err(|e| FerrotorchError::Gpu {
                    source: Box::new(e),
                })?;
        let bf16_data: Vec<bf16> = host_bits.into_iter().map(bf16::from_bits).collect();
        // GGUF stores 2-D weights as [in, out] dims; the model expects the
        // shape its config derives. We pass the GGUF dims through verbatim;
        // `LlamaGpuInferencer::new` validates against the config-derived
        // shape and surfaces any mismatch as `ShapeMismatch`.
        let tensor = Tensor::from_storage(TensorStorage::cpu(bf16_data), shape.clone(), false)?;
        state.insert(hf_name, tensor);
    }

    Ok(state)
}
