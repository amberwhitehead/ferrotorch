//! ADVERSARIAL RE-AUDIT of the cubecl->cudarc GGUF GPU-dequant bridge (#1350).
//!
//! The prior bridge tests (`divergence_gpu_gguf_bridge.rs`) only verified
//! `Q4_0` and `Q8_0` bit-exact vs the host reference. This file closes the
//! coverage gap by driving EVERY quant format the bridge dispatches
//! (`Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`Q8_0`/`Q8_1`) through
//! `gpu_dequantize_to_bf16_cudarc` on the live RTX 3090 and comparing,
//! bit-for-bit at bf16 precision, against the INDEPENDENT host reference
//! `ferrotorch_serialize::gguf::dequantize_gguf_tensor` (R-CHAR-3 oracle:
//! the expected value is produced by a code path that shares no dequant
//! logic with the cubecl kernels).
//!
//! Coverage added beyond the prior file:
//!   * `Q4_1`, `Q5_0`, `Q5_1`, `Q8_1` GPU-vs-host bit-exact (NEVER directly
//!     verified before — the formats with the most room for a block-layout,
//!     scale/min, or high-bit (qh) dispatch error).
//!   * a multi-block (2048-element, 64-block) `Q5_0` tensor crossing many
//!     blocks, to catch a per-block-index/qh-word miscompute that a 1-block
//!     test would mask.
//!   * the GPU bridge's non-block-multiple rejection AGREEING with the host
//!     reference's rejection (both reject the same odd element count).
//!   * an explicit guard that `bf16::from_f32` is round-to-nearest (not
//!     truncation), pinned to an IEEE-derived value, so the bf16-narrowing
//!     contract the bridge and host path share cannot silently drift to
//!     truncation on either side.
//!
//! Run: `cargo test -p ferrotorch-llama --features cuda --test divergence_gpu_gguf_bridge_reaudit`.

#![cfg(feature = "cuda")]

use ferrotorch_gpu::GpuDevice;
use ferrotorch_llama::{cubecl_cuda_client, gpu_dequantize_to_bf16_cudarc};
use ferrotorch_serialize::gguf::{GgmlType, dequantize_gguf_tensor, parse_gguf_bytes};
use half::{bf16, f16};

const GGUF_MAGIC: u32 = 0x4655_4747;
const DEFAULT_ALIGNMENT: usize = 32;

// ggml_type discriminants (matches GgmlType::from_u32 in ferrotorch-serialize).
const T_Q4_0: u32 = 2;
const T_Q4_1: u32 = 3;
const T_Q5_0: u32 = 6;
const T_Q5_1: u32 = 7;
const T_Q8_0: u32 = 8;
const T_Q8_1: u32 = 9;

/// Build a minimal valid GGUF file holding a single tensor of the given
/// ggml type so the host reference can dequant it via the public
/// `parse_gguf_bytes` + `dequantize_gguf_tensor` path. Data offset = 0,
/// aligned to the 32-byte data section.
fn build_single_tensor_gguf(name: &str, dims: &[u64], ggml_type: u32, data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes()); // version
    buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&ggml_type.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // data offset

    let rem = buf.len() % DEFAULT_ALIGNMENT;
    if rem != 0 {
        buf.resize(buf.len() + (DEFAULT_ALIGNMENT - rem), 0);
    }
    buf.extend_from_slice(data);
    buf
}

/// Host reference: dequant a tensor via the existing serialize path and
/// narrow to bf16 bits — the exact target the GPU bridge must hit.
fn host_reference_bf16_bits(dims: &[u64], ggml_type: u32, raw: &[u8]) -> Vec<u16> {
    let bytes = build_single_tensor_gguf("w", dims, ggml_type, raw);
    let file = parse_gguf_bytes(&bytes).expect("synthetic GGUF must parse");
    let tensor =
        dequantize_gguf_tensor(&file, "w").expect("host reference dequant must succeed");
    tensor
        .data()
        .expect("cpu tensor data")
        .iter()
        .map(|&x| bf16::from_f32(x).to_bits())
        .collect()
}

/// Run the GPU bridge for a single tensor and download its bf16 bits.
fn gpu_bridge_bf16_bits(raw: &[u8], ty: GgmlType, num_elements: usize) -> Vec<u16> {
    let device = GpuDevice::new(0).expect("CUDA device 0 must initialize on the 3090");
    let client = cubecl_cuda_client(device.ordinal());
    let cuda_bits = gpu_dequantize_to_bf16_cudarc(&client, raw, ty, num_elements, &device)
        .expect("GPU bridge must run on the 3090");
    assert_eq!(
        cuda_bits.len(),
        num_elements,
        "CudaSlice element count must equal num_elements (on-device residency)"
    );
    device
        .stream()
        .clone_dtoh(&cuda_bits)
        .expect("download bf16 bits from device")
}

// --- deterministic synthetic block builders, one per format -----------------
// Each builds raw on-disk bytes in the exact layout the host reference parses,
// using a small LCG so the bytes are reproducible and exercise the full quant
// range (nibbles 0..16, high bits 0/1, signed i8 -128..127, varied scales/mins).

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12345);
        self.0
    }
    fn unit(&mut self) -> f32 {
        self.next() as f32 / u32::MAX as f32
    }
}

fn synth_q4_1(num_blocks: usize, seed: u32) -> Vec<u8> {
    let mut r = Lcg(seed);
    let mut raw = Vec::with_capacity(num_blocks * 20);
    for _ in 0..num_blocks {
        let s = f16::from_f32(r.unit() * 2.0);
        let m = f16::from_f32(r.unit() * 4.0 - 2.0);
        raw.extend_from_slice(&s.to_bits().to_le_bytes());
        raw.extend_from_slice(&m.to_bits().to_le_bytes());
        for _ in 0..16 {
            let lo = (r.next() & 0xF) as u8;
            let hi = (r.next() & 0xF) as u8;
            raw.push((hi << 4) | lo);
        }
    }
    raw
}

fn synth_q5_0(num_blocks: usize, seed: u32) -> Vec<u8> {
    let mut r = Lcg(seed);
    let mut raw = Vec::with_capacity(num_blocks * 22);
    for _ in 0..num_blocks {
        let s = f16::from_f32(r.unit() * 4.0 - 2.0);
        raw.extend_from_slice(&s.to_bits().to_le_bytes());
        let qh = r.next(); // 32 high bits, one per element
        raw.extend_from_slice(&qh.to_le_bytes());
        for _ in 0..16 {
            let lo = (r.next() & 0xF) as u8;
            let hi = (r.next() & 0xF) as u8;
            raw.push((hi << 4) | lo);
        }
    }
    raw
}

fn synth_q5_1(num_blocks: usize, seed: u32) -> Vec<u8> {
    let mut r = Lcg(seed);
    let mut raw = Vec::with_capacity(num_blocks * 24);
    for _ in 0..num_blocks {
        let s = f16::from_f32(r.unit() * 2.0);
        let m = f16::from_f32(r.unit() * 4.0 - 2.0);
        raw.extend_from_slice(&s.to_bits().to_le_bytes());
        raw.extend_from_slice(&m.to_bits().to_le_bytes());
        let qh = r.next();
        raw.extend_from_slice(&qh.to_le_bytes());
        for _ in 0..16 {
            let lo = (r.next() & 0xF) as u8;
            let hi = (r.next() & 0xF) as u8;
            raw.push((hi << 4) | lo);
        }
    }
    raw
}

fn synth_q8_1(num_blocks: usize, seed: u32) -> Vec<u8> {
    let mut r = Lcg(seed);
    let mut raw = Vec::with_capacity(num_blocks * 40);
    for _ in 0..num_blocks {
        // Q8_1 scale + min are f32 on disk (NOT f16).
        let s = r.unit() * 0.5;
        let m = r.unit() * 4.0 - 2.0;
        raw.extend_from_slice(&s.to_le_bytes());
        raw.extend_from_slice(&m.to_le_bytes());
        for _ in 0..32 {
            raw.push((r.next() & 0xFF) as u8); // full -128..127 signed range
        }
    }
    raw
}

// --- per-format GPU-vs-host bit-exact tests ---------------------------------

#[test]
fn q4_1_bridge_matches_host_reference_at_bf16() {
    let num_blocks = 4;
    let num_elements = num_blocks * 32;
    let raw = synth_q4_1(num_blocks, 0x1111_2222);
    let got = gpu_bridge_bf16_bits(&raw, GgmlType::Q4_1, num_elements);
    let expected = host_reference_bf16_bits(&[num_elements as u64], T_Q4_1, &raw);
    assert_eq!(got.len(), expected.len());
    assert_eq!(
        got, expected,
        "Q4_1 GPU bridge must equal host reference bit-for-bit at bf16 (scale*nibble + min)"
    );
}

#[test]
fn q5_0_bridge_matches_host_reference_at_bf16() {
    let num_blocks = 4;
    let num_elements = num_blocks * 32;
    let raw = synth_q5_0(num_blocks, 0x3333_4444);
    let got = gpu_bridge_bf16_bits(&raw, GgmlType::Q5_0, num_elements);
    let expected = host_reference_bf16_bits(&[num_elements as u64], T_Q5_0, &raw);
    assert_eq!(got.len(), expected.len());
    assert_eq!(
        got, expected,
        "Q5_0 GPU bridge must equal host reference bit-for-bit at bf16 (qh high-bit, -16 centering)"
    );
}

#[test]
fn q5_1_bridge_matches_host_reference_at_bf16() {
    let num_blocks = 4;
    let num_elements = num_blocks * 32;
    let raw = synth_q5_1(num_blocks, 0x5555_6666);
    let got = gpu_bridge_bf16_bits(&raw, GgmlType::Q5_1, num_elements);
    let expected = host_reference_bf16_bits(&[num_elements as u64], T_Q5_1, &raw);
    assert_eq!(got.len(), expected.len());
    assert_eq!(
        got, expected,
        "Q5_1 GPU bridge must equal host reference bit-for-bit at bf16 (qh high-bit, scale*val + min)"
    );
}

#[test]
fn q8_1_bridge_matches_host_reference_at_bf16() {
    let num_blocks = 4;
    let num_elements = num_blocks * 32;
    let raw = synth_q8_1(num_blocks, 0x7777_8888);
    let got = gpu_bridge_bf16_bits(&raw, GgmlType::Q8_1, num_elements);
    let expected = host_reference_bf16_bits(&[num_elements as u64], T_Q8_1, &raw);
    assert_eq!(got.len(), expected.len());
    assert_eq!(
        got, expected,
        "Q8_1 GPU bridge must equal host reference bit-for-bit at bf16 (f32 scale/min, signed i8)"
    );
}

// --- re-pin Q4_0 / Q8_0 in this file too, for a single regression surface ---

fn synth_q4_0(num_blocks: usize, seed: u32) -> Vec<u8> {
    let mut r = Lcg(seed);
    let mut raw = Vec::with_capacity(num_blocks * 18);
    for _ in 0..num_blocks {
        let s = f16::from_f32(r.unit() * 4.0 - 2.0);
        raw.extend_from_slice(&s.to_bits().to_le_bytes());
        for _ in 0..16 {
            let lo = (r.next() & 0xF) as u8;
            let hi = (r.next() & 0xF) as u8;
            raw.push((hi << 4) | lo);
        }
    }
    raw
}

fn synth_q8_0(num_blocks: usize, seed: u32) -> Vec<u8> {
    let mut r = Lcg(seed);
    let mut raw = Vec::with_capacity(num_blocks * 34);
    for _ in 0..num_blocks {
        let s = f16::from_f32(r.unit() * 0.5);
        raw.extend_from_slice(&s.to_bits().to_le_bytes());
        for _ in 0..32 {
            raw.push((r.next() & 0xFF) as u8);
        }
    }
    raw
}

#[test]
fn q4_0_bridge_matches_host_reference_at_bf16_reaudit() {
    let num_blocks = 3;
    let num_elements = num_blocks * 32;
    let raw = synth_q4_0(num_blocks, 0x9999_AAAA);
    let got = gpu_bridge_bf16_bits(&raw, GgmlType::Q4_0, num_elements);
    let expected = host_reference_bf16_bits(&[num_elements as u64], T_Q4_0, &raw);
    assert_eq!(got, expected, "Q4_0 GPU bridge bit-exact vs host reference");
}

#[test]
fn q8_0_bridge_matches_host_reference_at_bf16_reaudit() {
    let num_blocks = 3;
    let num_elements = num_blocks * 32;
    let raw = synth_q8_0(num_blocks, 0xBBBB_CCCC);
    let got = gpu_bridge_bf16_bits(&raw, GgmlType::Q8_0, num_elements);
    let expected = host_reference_bf16_bits(&[num_elements as u64], T_Q8_0, &raw);
    assert_eq!(got, expected, "Q8_0 GPU bridge bit-exact vs host reference");
}

// --- multi-block crossing test (hunt item 3): 64 blocks = 2048 elements ------

#[test]
fn q5_0_large_multiblock_bridge_matches_host_reference() {
    let num_blocks = 64;
    let num_elements = num_blocks * 32; // 2048
    let raw = synth_q5_0(num_blocks, 0xFACE_F00D);
    let got = gpu_bridge_bf16_bits(&raw, GgmlType::Q5_0, num_elements);
    let expected = host_reference_bf16_bits(&[num_elements as u64], T_Q5_0, &raw);
    assert_eq!(got.len(), num_elements);
    assert_eq!(
        got, expected,
        "Q5_0 2048-elem (64-block) GPU bridge must stay bit-exact across all blocks/qh words"
    );
}

// --- block-multiple rejection AGREEMENT (hunt item 3) ------------------------

#[test]
fn bridge_and_host_agree_on_rejecting_non_block_multiple() {
    // Host reference: a tensor whose declared element count is not a multiple
    // of 32 but whose data section is short relative to div_ceil block bytes.
    // We give exactly 1 block of bytes but declare 33 elements => host needs
    // ceil(33/32)=2 blocks of bytes and must reject.
    let raw = synth_q4_0(1, 0xDEAD_0001);
    let host_bytes = build_single_tensor_gguf("w", &[33u64], T_Q4_0, &raw);
    let host_file = parse_gguf_bytes(&host_bytes).expect("parse");
    let host_result = dequantize_gguf_tensor(&host_file, "w");
    assert!(
        host_result.is_err(),
        "host reference must reject 33-elem Q4_0 with only 1 block of data"
    );

    // GPU bridge: the SAME odd element count is rejected (block-multiple guard).
    let device = GpuDevice::new(0).expect("CUDA device 0 must initialize on the 3090");
    let client = cubecl_cuda_client(device.ordinal());
    let bridge_result = gpu_dequantize_to_bf16_cudarc(&client, &raw, GgmlType::Q4_0, 33, &device);
    assert!(
        bridge_result.is_err(),
        "GPU bridge must reject 33-elem Q4_0 (non-block-multiple); host also rejects"
    );
}

// --- bf16 narrowing contract: round-to-nearest, NOT truncation ---------------

/// `bf16` keeps the f32 sign + exponent + top 7 mantissa bits. A value whose
/// 8th mantissa bit is set (with lower bits) must ROUND UP, not truncate.
///
/// Pick `x` = f32 with mantissa `0x008001` (bit 15 set => the bit just below
/// the bf16 cutoff is 1 and a lower bit is 1, forcing round-half-up). Under
/// round-to-nearest-even the bf16 mantissa increments; under truncation it
/// would not. The bridge and the host path both rely on `bf16::from_f32`
/// being round-to-nearest so their narrowed outputs coincide.
#[test]
fn bf16_from_f32_is_round_to_nearest_not_truncation() {
    // 1.0 + 257/65536 : mantissa bits beyond the 7 bf16 keeps are 1_0000_0001,
    // i.e. > half an ULP => must round up to the next bf16 representable value.
    let x = f32::from_bits(0x3F80_8100); // 1.0 with mantissa 0x008100
    let rounded = bf16::from_f32(x);
    // Truncation would yield bits 0x3F80 (mantissa 0x00 of bf16, value 1.0).
    // Round-to-nearest yields 0x3F81 (mantissa 0x01, value 1.0078125).
    assert_eq!(
        rounded.to_bits(),
        0x3F81,
        "bf16::from_f32 must round to nearest (got truncation?): bits={:#06X}",
        rounded.to_bits()
    );
    assert_ne!(
        rounded.to_bits(),
        0x3F80,
        "bf16::from_f32 truncated instead of rounding — bridge/host narrowing would drift 1 ULP"
    );
}
