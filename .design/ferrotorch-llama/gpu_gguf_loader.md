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
| REQ-8 (`apply_token_mask` GPU grammar masking) | SHIPPED | impl: `pub fn apply_grammar_mask_gpu in gpu_gguf.rs` calls `ferrotorch_cubecl::quant::apply_token_mask_to_gpu(&cubecl_cuda_client(ordinal), logits, allow_mask)` — real cubecl `CudaRuntime` JIT + on-device `kernel_apply_token_mask` dispatch (one thread/token; `mask!=0` → bit-exact passthrough, `mask==0` → `f32::MIN`) — then `read_one`s the masked logits back to host. The autoregressive decode lives in the model-agnostic `pub fn masked_decode_loop in gpu_gguf.rs`, which carries the **grammar-completion guard** (`if proc.is_complete() { break }` at the top of the loop body, #1667): a completed grammar emits an all-deny mask, so without the guard the next step forces every logit to `f32::MIN`, `argmax_f32` returns the forbidden index 0, and `step_token(0)` errors `AlreadyComplete` — the guard stops at completion and returns the valid sequence instead. Non-test production consumer: `pub fn LlamaGpuInferencer::generate_masked in gpu_gguf.rs` is a thin wrapper delegating to `masked_decode_loop(\|ids\| self.forward_from_ids(ids), grammar, prompt, max_new_tokens, ordinal)`, so the cubecl mask kernel + the completion guard run once per generated token under an active grammar. Re-exported `pub use gpu_gguf::{apply_grammar_mask_gpu, masked_decode_loop} in lib.rs` (cuda-gated). Live-GPU coverage: `tests/divergence_apply_token_mask_gpu.rs` (allowed==orig bit-exact, disallowed==`f32::MIN`, forbidden max-logit token never wins masked argmax) + `tests/divergence_apply_token_mask_gpu_reaudit.rs` (#1667: drives the real `masked_decode_loop` with an injected synthetic-logits closure — stops at grammar completion not error, every emitted token grammar-allowed, forbidden max-logit token never emitted; the guard is load-bearing — removing it fails both) — verified on RTX 3090. Closes the #1350 phantom-consumer follow-up + the #1667 completion-guard divergence. |

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
