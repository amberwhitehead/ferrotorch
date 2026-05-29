//! Live-GPU verification of the cubecl→cudarc GGUF dequant bridge (#1350).
//!
//! The substantive #1350 deliverable is the cross-runtime bridge
//! `ferrotorch_llama::gpu_dequantize_to_bf16_cudarc`: it runs a
//! ferrotorch-cubecl dequant kernel on the GPU, reads the f32 handle
//! back, narrows to bf16, and uploads into a `cudarc` `CudaSlice<u16>`
//! on a separate context. This test proves the bridge is **numerically
//! correct** by comparing its output against the EXISTING host
//! reference `ferrotorch_serialize::gguf::dequantize_gguf_tensor`
//! (R-CHAR-3 oracle), for synthetic Q4_0 and Q8_0 blocks of known bytes.
//!
//! Because the GPU path dequants in f32 then narrows to bf16, the
//! reference is also narrowed to bf16; the two then compare bit-for-bit
//! exactly (no tolerance fudge — both land on the same bf16 grid).
//!
//! Also asserts the result is a genuine on-device `CudaSlice<u16>`
//! (residency: round-trips through `clone_dtoh`), and that
//! `cubecl_cuda_client` constructs + JIT-compiles + runs on the live
//! 3090.
//!
//! Run with `cargo test -p ferrotorch-llama --features cuda --test divergence_gpu_gguf_bridge`.

#![cfg(feature = "cuda")]

use ferrotorch_gpu::GpuDevice;
use ferrotorch_llama::{cubecl_cuda_client, gpu_dequantize_to_bf16_cudarc};
use ferrotorch_serialize::gguf::{GgmlType, parse_gguf_bytes};
use half::bf16;

const GGUF_MAGIC: u32 = 0x4655_4747;
const DEFAULT_ALIGNMENT: usize = 32;

/// Build a minimal valid GGUF file holding a single tensor of the given
/// ggml type, so the host reference can dequant it via the public
/// `parse_gguf_bytes` + `dequantize_gguf_tensor` path.
fn build_single_tensor_gguf(name: &str, dims: &[u64], ggml_type: u32, data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes()); // version
    buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

    // tensor info: name, n_dims, dims, ggml_type, offset
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&ggml_type.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // data offset

    // align to data section
    let rem = buf.len() % DEFAULT_ALIGNMENT;
    if rem != 0 {
        buf.resize(buf.len() + (DEFAULT_ALIGNMENT - rem), 0);
    }
    buf.extend_from_slice(data);
    buf
}

/// Host reference: dequant a tensor via the existing serialize path and
/// narrow to bf16 bits — the exact target the GPU bridge must hit.
fn host_reference_bf16_bits(name: &str, dims: &[u64], ggml_type: u32, raw: &[u8]) -> Vec<u16> {
    let bytes = build_single_tensor_gguf(name, dims, ggml_type, raw);
    let file = parse_gguf_bytes(&bytes).expect("synthetic GGUF must parse");
    let tensor = ferrotorch_serialize::gguf::dequantize_gguf_tensor(&file, name)
        .expect("host reference dequant must succeed");
    tensor
        .data()
        .expect("cpu tensor data")
        .iter()
        .map(|&x| bf16::from_f32(x).to_bits())
        .collect()
}

/// Construct a deterministic Q4_0 block stream of `num_blocks` blocks and
/// return its raw on-disk bytes (18 bytes/block).
fn synth_q4_0(num_blocks: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
        state
    };
    let mut raw = Vec::with_capacity(num_blocks * 18);
    for _ in 0..num_blocks {
        let s = half::f16::from_f32((next() as f32 / u32::MAX as f32) * 4.0 - 2.0);
        raw.extend_from_slice(&s.to_bits().to_le_bytes());
        for _ in 0..16 {
            let lo = (next() & 0xF) as u8;
            let hi = (next() & 0xF) as u8;
            raw.push((hi << 4) | lo);
        }
    }
    raw
}

/// Construct a deterministic Q8_0 block stream of `num_blocks` blocks and
/// return its raw on-disk bytes (34 bytes/block).
fn synth_q8_0(num_blocks: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(214_013).wrapping_add(2_531_011);
        state
    };
    let mut raw = Vec::with_capacity(num_blocks * 34);
    for _ in 0..num_blocks {
        let s = half::f16::from_f32((next() as f32 / u32::MAX as f32) * 0.5);
        raw.extend_from_slice(&s.to_bits().to_le_bytes());
        for _ in 0..32 {
            raw.push((next() & 0xFF) as u8); // full -128..127 range
        }
    }
    raw
}

#[test]
fn q4_0_bridge_matches_host_reference_at_bf16() {
    let device = GpuDevice::new(0).expect("CUDA device 0 must initialize on the 3090");
    let client = cubecl_cuda_client(device.ordinal());

    let num_blocks = 4;
    let num_elements = num_blocks * 32;
    let raw = synth_q4_0(num_blocks, 0xCAFE_BABE);

    // GPU bridge: cubecl dequant kernel → CudaSlice<u16>.
    let cuda_bits =
        gpu_dequantize_to_bf16_cudarc(&client, &raw, GgmlType::Q4_0, num_elements, &device)
            .expect("Q4_0 GPU bridge must run on the 3090");

    // Residency: it is a genuine on-device CudaSlice<u16>.
    assert_eq!(cuda_bits.len(), num_elements, "CudaSlice element count");
    let got_bits: Vec<u16> = device
        .stream()
        .clone_dtoh(&cuda_bits)
        .expect("download bf16 bits from device");

    // Host reference: existing serialize dequant → bf16 bits.
    let expected_bits = host_reference_bf16_bits("w", &[num_elements as u64], 2, &raw);

    assert_eq!(got_bits.len(), expected_bits.len());
    assert_eq!(
        got_bits, expected_bits,
        "Q4_0 GPU bridge must equal host reference bit-for-bit at bf16 precision"
    );
}

#[test]
fn q8_0_bridge_matches_host_reference_at_bf16() {
    let device = GpuDevice::new(0).expect("CUDA device 0 must initialize on the 3090");
    let client = cubecl_cuda_client(device.ordinal());

    let num_blocks = 5;
    let num_elements = num_blocks * 32;
    let raw = synth_q8_0(num_blocks, 0xDEAD_BEEF);

    let cuda_bits =
        gpu_dequantize_to_bf16_cudarc(&client, &raw, GgmlType::Q8_0, num_elements, &device)
            .expect("Q8_0 GPU bridge must run on the 3090");

    assert_eq!(cuda_bits.len(), num_elements, "CudaSlice element count");
    let got_bits: Vec<u16> = device
        .stream()
        .clone_dtoh(&cuda_bits)
        .expect("download bf16 bits from device");

    let expected_bits = host_reference_bf16_bits("w", &[num_elements as u64], 8, &raw);

    assert_eq!(got_bits.len(), expected_bits.len());
    assert_eq!(
        got_bits, expected_bits,
        "Q8_0 GPU bridge must equal host reference bit-for-bit at bf16 precision"
    );
}

#[test]
fn bridge_rejects_non_block_multiple_element_count() {
    let device = GpuDevice::new(0).expect("CUDA device 0 must initialize on the 3090");
    let client = cubecl_cuda_client(device.ordinal());
    let raw = synth_q4_0(1, 1);
    // 33 is not a multiple of the 32-element block size.
    let err = gpu_dequantize_to_bf16_cudarc(&client, &raw, GgmlType::Q4_0, 33, &device)
        .expect_err("non-block-multiple element count must be rejected");
    let msg = format!("{err}");
    assert!(msg.contains("multiple"), "got: {msg}");
}
