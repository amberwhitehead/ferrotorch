# ferrotorch-llama — GPU-dequantized GGUF loader (`gpu_gguf.rs`)

Tracks #1350. Bridges the ferrotorch-cubecl GGUF dequant kernels into
the ferrotorch-llama GPU path (`LlamaGpuInferencer`) via a real
cubecl-`Handle` → cudarc-`CudaSlice<u16>` transfer.

All anchors are symbolic (S3 style: `symbol in file`), no line numbers.

## Architecture summary

ferrotorch-cubecl owns the dequant kernels (`dequantize_q*_to_gpu`):
host-split block buffers → `#[cube]` kernel → f32 `cubecl::server::Handle`
on the cubecl `CudaRuntime` device. ferrotorch-llama's inferencer stores
weights as `cudarc::driver::CudaSlice<u16>` (bf16 bits) on a
`ferrotorch_gpu::GpuDevice`. The two runtimes hold **independent CUDA
contexts** with no pointer-handoff API, so the bridge is a host bounce.

## REQ table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (cubecl client constructor) | SHIPPED | `pub fn cubecl_cuda_client in gpu_gguf.rs` builds `CudaRuntime::client(&CudaDevice{index})`; non-test consumer: `LlamaGpuInferencer::from_gguf_gpu_dequant in gpu_gguf.rs` calls it with `device.ordinal()`. |
| REQ-2 (cross-runtime bridge) | SHIPPED | `pub fn gpu_dequantize_to_bf16_cudarc in gpu_gguf.rs` — split → cubecl dequant kernel → `read_one` → `bf16::from_f32` → `clone_htod` → `CudaSlice<u16>`; non-test consumer: `fn gguf_tensor_to_bf16_cudarc in gpu_gguf.rs` (the per-tensor dispatch the loader runs). |
| REQ-3 (per-tensor dispatch) | SHIPPED | `fn gguf_tensor_to_bf16_cudarc in gpu_gguf.rs` routes quantized→bridge, scalar→host fallback; consumer: `fn bf16_state_dict_from_gguf_gpu in gpu_gguf.rs`. |
| REQ-4 (scalar host fallback) | SHIPPED | `fn gpu_dequantize_scalar_to_bf16_cudarc in gpu_gguf.rs` uses `ferrotorch_serialize::gguf::dequantize_gguf_tensor` (the bridge's correctness oracle) for F32/F16; consumer: `gguf_tensor_to_bf16_cudarc`. |
| REQ-5 (whole-file loader) | SHIPPED | `pub fn LlamaGpuInferencer::from_gguf_gpu_dequant in gpu_gguf.rs`: `load_gguf_mmap` → GPU-dequant every tensor → `gguf_key_to_hf` remap → `LlamaGpuInferencer::new`; consumer: examples / harness GGUF load path. |
| REQ-6 (host-bounce limitation) | DOCUMENTED-LIMITATION | module docs in `gpu_gguf.rs` ("The bridge is a host bounce") — cubecl exposes no device-pointer handoff to cudarc; output is `read_one`'d to host and re-uploaded. The single function to change when cubecl ships context sharing is `gpu_dequantize_to_bf16_cudarc`. |
| REQ-7 (ComputeClient-context limitation) | DOCUMENTED-LIMITATION | module docs in `gpu_gguf.rs` ("ComputeClient context duplication") — `cubecl_cuda_client` opens a fresh cubecl context on the same physical device; no API to adopt `GpuDevice`'s cudarc `CudaContext`. One extra context's bookkeeping on a single-GPU box, not duplicated weights. |
| REQ-8 (`apply_token_mask` phantom) | NOT-WIRED (phantom-consumer follow-up) | `ferrotorch_cubecl::apply_token_mask_to_gpu` has zero production call sites (doc-comments + cubecl test only). ferrotorch-llama's GPU logit path (`forward_from_ids` / `GraphedDecoder::decode_step`) downloads logits to host as `Vec<f32>` and applies sampling/constrained decoding on host (`generation.rs`, `ferrotorch-grammar`); there is no on-device logit-masking entry point to wire a `CudaSlice` consumer into without inventing a GPU constrained-generation path. Filed as a separate follow-up, NOT fabricated here. |

## Correctness contract (R-CHAR-3)

The bridge is validated against the **existing host reference**
`ferrotorch_serialize::gguf::dequantize_gguf_tensor` (REQ-9 of
`gguf.md`), narrowed to bf16. For a synthetic Q4_0 and Q8_0 block of
known bytes: GPU-bridge output (`CudaSlice<u16>` → host bf16 → f32) must
equal the host-reference output narrowed to bf16, bit-for-bit (both
sides land in bf16, so the comparison is exact at bf16 precision, not a
tolerance fudge). Verified live on the RTX 3090 in
`tests/divergence_gpu_gguf_bridge.rs`.

## Why the bridge requires a `GgufFile` raw-bytes accessor

`gpu_dequantize_to_bf16_cudarc` takes `raw_quant_bytes: &[u8]`. The
whole-file loader feeds it each quantized tensor's on-disk block bytes,
sliced from the GGUF data section. The data section is currently a
private field on `GgufFile` (`ferrotorch-serialize`); a public
`GgufFile::data()` accessor is required for the loader to be a genuine
GPU-bridge consumer rather than re-parsing the file. See the dispatch
note in #1350.
