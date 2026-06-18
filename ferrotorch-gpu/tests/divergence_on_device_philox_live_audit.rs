//! Adversarial re-audit of the LIVE on-device Philox PTX kernels (#1684).
//!
//! Commit 2240fa9e2 fixed the original static uniform/normal PTX kernels so
//! they JIT and execute on the GPU for the first time (the `%tid` register
//! shadow bug).
//! The prior audit (310a0c545) only verified the CPU fallback. These probes
//! exercise the *on-device* kernel output directly on a live RTX 3090.
//!
//! Reference discipline (R-CHAR-3): `torch.rand(..., device="cuda")` uses
//! PyTorch's CUDA distribution kernel (`DistributionTemplates.h`) which launches
//! one cuRAND Philox state per logical CUDA thread:
//! `curand_init(seed, thread_linear_idx, offset, &state)`. Each thread writes a
//! `float4`/`double2` across grid-stride lanes. These probes compute that layout
//! directly instead of comparing to Ferrotorch's CPU `PhiloxGenerator`, whose
//! contiguous-counter stream was the bug tracked by #1683.
//!
//! Normal byte parity is checked against a live torch CUDA oracle because
//! PyTorch delegates float/half/bfloat to `curand_normal4` and double to
//! `curand_normal2_double` (`DistributionTemplates.h:443-453`), whose log/sincos
//! math is libdevice-specific rather than host-libm equivalent.
//!
//! SERIALIZATION: the CUDA RNG manager is a process-global singleton. Tests that
//! seed-then-sample (and especially the multi-call stream-continuity probe) hold
//! `SEED_LOCK` across their seed+sample window so a concurrent test's
//! `manual_seed` cannot perturb the shared per-device Philox counter between
//! calls. Without this guard the stream-continuity probe is flaky under the
//! default multi-threaded test runner (the failure is a harness race, not a
//! kernel divergence — verified: it passes in isolation / single-threaded).

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, manual_seed, rand_on_device, randn_on_device};
use ferrotorch_gpu::{GpuDevice, init_cuda_backend};
use half::{bf16, f16};
use std::process::Command;
use std::sync::Mutex;

static SEED_LOCK: Mutex<()> = Mutex::new(());

fn ensure_init() {
    if !ferrotorch_core::gpu_dispatch::has_gpu_backend() {
        init_cuda_backend().expect("init_cuda_backend");
    }
}

fn to_host(t: &ferrotorch_core::Tensor<f32>) -> Vec<f32> {
    let cpu = t.to(Device::Cpu).expect("tensor.to(Cpu)");
    cpu.data().expect("cpu data").to_vec()
}

fn to_host_f64(t: &ferrotorch_core::Tensor<f64>) -> Vec<f64> {
    let cpu = t.to(Device::Cpu).expect("tensor.to(Cpu)");
    cpu.data().expect("cpu data").to_vec()
}

fn to_host_f16_bits(t: &ferrotorch_core::Tensor<f16>) -> Vec<u64> {
    let cpu = t.to(Device::Cpu).expect("f16 tensor.to(Cpu)");
    cpu.data()
        .expect("cpu f16 data")
        .iter()
        .map(|x| x.to_bits() as u64)
        .collect()
}

fn to_host_bf16_bits(t: &ferrotorch_core::Tensor<bf16>) -> Vec<u64> {
    let cpu = t.to(Device::Cpu).expect("bf16 tensor.to(Cpu)");
    cpu.data()
        .expect("cpu bf16 data")
        .iter()
        .map(|x| x.to_bits() as u64)
        .collect()
}

fn f32_bits(xs: &[f32]) -> Vec<u64> {
    xs.iter().map(|x| x.to_bits() as u64).collect()
}

fn f64_bits(xs: &[f64]) -> Vec<u64> {
    xs.iter().map(|x| x.to_bits()).collect()
}

fn select_bits(bits: &[u64], indices: &[usize]) -> Vec<u64> {
    indices.iter().map(|&i| bits[i]).collect()
}

fn grid_boundary_indices(n: usize, stride: u64, lanes: u64) -> Vec<usize> {
    let mut indices = Vec::new();
    for lane in 0..lanes {
        let base = stride * lane;
        for raw in [base.saturating_sub(1), base, base + 1] {
            if raw < n as u64 {
                indices.push(raw as usize);
            }
        }
    }
    indices.extend([0usize, 1, n - 1]);
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn torch_cuda_randn_bits(
    dtype: &str,
    n: usize,
    seed: u64,
    skip_calls: usize,
    indices: &[usize],
) -> Vec<u64> {
    let index_arg = indices
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let script = r#"
import sys
import torch

dtype_name = sys.argv[1]
n = int(sys.argv[2])
seed = int(sys.argv[3])
skip_calls = int(sys.argv[4])
indices = [int(x) for x in sys.argv[5].split(",") if x]

if not torch.cuda.is_available():
    raise SystemExit("torch CUDA oracle unavailable")

dtypes = {
    "f32": torch.float32,
    "f64": torch.float64,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}
dtype = dtypes[dtype_name]
torch.cuda.manual_seed_all(seed)
for _ in range(skip_calls):
    torch.randn((n,), device="cuda", dtype=dtype)
x = torch.randn((n,), device="cuda", dtype=dtype).cpu()
torch.cuda.synchronize()
if dtype_name == "bf16":
    arr = x.view(torch.int16).numpy().view("uint16")
elif dtype_name == "f16":
    arr = x.numpy().view("uint16")
elif dtype_name == "f32":
    arr = x.numpy().view("uint32")
else:
    arr = x.numpy().view("uint64")
print(" ".join(format(int(arr[i]), "x") for i in indices))
"#;
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(dtype)
        .arg(n.to_string())
        .arg(seed.to_string())
        .arg(skip_calls.to_string())
        .arg(index_arg)
        .output()
        .expect("launch python3 torch oracle");
    assert!(
        output.status.success(),
        "torch CUDA randn oracle failed: status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("torch oracle stdout utf8")
        .split_whitespace()
        .map(|hex| u64::from_str_radix(hex, 16).expect("hex oracle bit pattern"))
        .collect()
}

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

fn philox_round(c: [u32; 4], k0: u32, k1: u32) -> [u32; 4] {
    let prod0 = (PHILOX_M0 as u64) * (c[0] as u64);
    let prod1 = (PHILOX_M1 as u64) * (c[2] as u64);
    [
        ((prod1 >> 32) as u32) ^ c[1] ^ k0,
        prod1 as u32,
        ((prod0 >> 32) as u32) ^ c[3] ^ k1,
        prod0 as u32,
    ]
}

fn philox4(seed: u64, counter: u64, subsequence: u64) -> [u32; 4] {
    let mut c = [
        counter as u32,
        (counter >> 32) as u32,
        subsequence as u32,
        (subsequence >> 32) as u32,
    ];
    let mut k0 = seed as u32;
    let mut k1 = (seed >> 32) as u32;
    for round in 0..10 {
        c = philox_round(c, k0, k1);
        if round != 9 {
            k0 = k0.wrapping_add(PHILOX_W0);
            k1 = k1.wrapping_add(PHILOX_W1);
        }
    }
    c
}

fn curand_uniform_f32_to_torch_rand(word: u32) -> f32 {
    let v = (word as f32).mul_add(2.328_306_4e-10, 1.164_153_2e-10);
    if v == 1.0 { 0.0 } else { v }
}

fn curand_uniform_f64_to_torch_rand(x: u32, y: u32) -> f64 {
    let z = (x as u64) ^ ((y as u64) << 21);
    let v = (z as f64).mul_add(1.110_223_024_625_156_5e-16, 5.551_115_123_125_783e-17);
    if v == 1.0 { 0.0 } else { v }
}

fn torch_distribution_stride(n: usize, unroll_factor: u64) -> u64 {
    let device = GpuDevice::new(0).expect("GpuDevice::new");
    let max_threads_per_sm = device
        .context()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        )
        .expect("max threads per SM") as u64;
    let sm_count = device
        .context()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .expect("SM count") as u64;
    let block = 256u64;
    let mut grid = (n as u64).div_ceil(block);
    let blocks_per_sm = (max_threads_per_sm / block).max(1);
    grid = grid.min((sm_count * blocks_per_sm).max(1)).max(1);
    let stride = grid * block;
    assert!(
        (n as u64).div_ceil(stride * unroll_factor) > 0,
        "non-empty policy must do at least one curand call per thread"
    );
    stride
}

fn torch_cuda_uniform_f32_reference(n: usize, seed: u64, base_counter: u64) -> Vec<f32> {
    let stride = torch_distribution_stride(n, 4);
    let mut out = vec![0.0; n];
    for tid in 0..stride {
        let mut linear = tid;
        let mut counter = base_counter;
        while linear < n as u64 {
            let r = philox4(seed, counter, tid);
            for (lane, word) in r.into_iter().enumerate() {
                let idx = linear + stride * lane as u64;
                if idx < n as u64 {
                    out[idx as usize] = curand_uniform_f32_to_torch_rand(word);
                }
            }
            linear += stride * 4;
            counter += 1;
        }
    }
    out
}

fn torch_cuda_uniform_f64_reference(n: usize, seed: u64, base_counter: u64) -> Vec<f64> {
    let stride = torch_distribution_stride(n, 2);
    let mut out = vec![0.0; n];
    for tid in 0..stride {
        let mut linear = tid;
        let mut counter = base_counter;
        while linear < n as u64 {
            let r = philox4(seed, counter, tid);
            let lanes = [
                curand_uniform_f64_to_torch_rand(r[0], r[1]),
                curand_uniform_f64_to_torch_rand(r[2], r[3]),
            ];
            for (lane, value) in lanes.into_iter().enumerate() {
                let idx = linear + stride * lane as u64;
                if idx < n as u64 {
                    out[idx as usize] = value;
                }
            }
            linear += stride * 2;
            counter += 1;
        }
    }
    out
}

fn calls_per_thread(n: usize, unroll_factor: u64) -> u64 {
    let stride = torch_distribution_stride(n, unroll_factor);
    ((n as u64 - 1) / (stride * unroll_factor)) + 1
}

#[test]
fn normal_f32_gpu_bit_exact_with_torch_cuda_small_and_tail_lengths() {
    ensure_init();
    let seed = 123u64;
    for &n in &[1usize, 2, 3, 7, 4097] {
        let indices: Vec<usize> = (0..n).collect();
        let expected = torch_cuda_randn_bits("f32", n, seed, 0, &indices);
        let got = {
            let _g = SEED_LOCK.lock().unwrap();
            manual_seed(seed).unwrap();
            let t = randn_on_device::<f32>(&[n], Device::Cuda(0)).expect("gpu f32 normal");
            assert_eq!(t.device(), Device::Cuda(0), "f32 normal must stay CUDA");
            f32_bits(&to_host(&t))
        };
        assert_eq!(
            got, expected,
            "f32 normal bit pattern diverges from torch.cuda at n={n}"
        );
    }
}

#[test]
fn normal_f64_gpu_bit_exact_with_torch_cuda_small_and_tail_lengths() {
    ensure_init();
    let seed = 124u64;
    for &n in &[1usize, 2, 3, 7, 4097] {
        let indices: Vec<usize> = (0..n).collect();
        let expected = torch_cuda_randn_bits("f64", n, seed, 0, &indices);
        let got = {
            let _g = SEED_LOCK.lock().unwrap();
            manual_seed(seed).unwrap();
            let t = randn_on_device::<f64>(&[n], Device::Cuda(0)).expect("gpu f64 normal");
            assert_eq!(t.device(), Device::Cuda(0), "f64 normal must stay CUDA");
            f64_bits(&to_host_f64(&t))
        };
        assert_eq!(
            got, expected,
            "f64 normal bit pattern diverges from torch.cuda at n={n}"
        );
    }
}

#[test]
fn normal_half_and_bfloat_gpu_bit_exact_with_torch_cuda_small_and_tail_lengths() {
    ensure_init();
    let seed = 125u64;
    for &n in &[1usize, 2, 3, 7, 4097] {
        let indices: Vec<usize> = (0..n).collect();

        let expected_f16 = torch_cuda_randn_bits("f16", n, seed, 0, &indices);
        let got_f16 = {
            let _g = SEED_LOCK.lock().unwrap();
            manual_seed(seed).unwrap();
            let t = randn_on_device::<f16>(&[n], Device::Cuda(0)).expect("gpu f16 normal");
            assert_eq!(t.device(), Device::Cuda(0), "f16 normal must stay CUDA");
            to_host_f16_bits(&t)
        };
        assert_eq!(
            got_f16, expected_f16,
            "f16 normal bit pattern diverges from torch.cuda at n={n}"
        );

        let expected_bf16 = torch_cuda_randn_bits("bf16", n, seed, 0, &indices);
        let got_bf16 = {
            let _g = SEED_LOCK.lock().unwrap();
            manual_seed(seed).unwrap();
            let t = randn_on_device::<bf16>(&[n], Device::Cuda(0)).expect("gpu bf16 normal");
            assert_eq!(t.device(), Device::Cuda(0), "bf16 normal must stay CUDA");
            to_host_bf16_bits(&t)
        };
        assert_eq!(
            got_bf16, expected_bf16,
            "bf16 normal bit pattern diverges from torch.cuda at n={n}"
        );
    }
}

#[test]
fn normal_gpu_bit_exact_with_torch_cuda_consecutive_call_offsets() {
    ensure_init();
    let n = 7usize;
    let seed = 77u64;
    let indices: Vec<usize> = (0..n).collect();

    let expected_f32_second = torch_cuda_randn_bits("f32", n, seed, 1, &indices);
    let expected_f64_second = torch_cuda_randn_bits("f64", n, seed, 1, &indices);
    let expected_f16_second = torch_cuda_randn_bits("f16", n, seed, 1, &indices);
    let expected_bf16_second = torch_cuda_randn_bits("bf16", n, seed, 1, &indices);

    let (got_f32_second, got_f64_second, got_f16_second, got_bf16_second) = {
        let _g = SEED_LOCK.lock().unwrap();

        manual_seed(seed).unwrap();
        let _ = randn_on_device::<f32>(&[n], Device::Cuda(0)).expect("first f32 normal");
        let got_f32_second = f32_bits(&to_host(
            &randn_on_device::<f32>(&[n], Device::Cuda(0)).expect("second f32 normal"),
        ));

        manual_seed(seed).unwrap();
        let _ = randn_on_device::<f64>(&[n], Device::Cuda(0)).expect("first f64 normal");
        let got_f64_second = f64_bits(&to_host_f64(
            &randn_on_device::<f64>(&[n], Device::Cuda(0)).expect("second f64 normal"),
        ));

        manual_seed(seed).unwrap();
        let _ = randn_on_device::<f16>(&[n], Device::Cuda(0)).expect("first f16 normal");
        let got_f16_second = to_host_f16_bits(
            &randn_on_device::<f16>(&[n], Device::Cuda(0)).expect("second f16 normal"),
        );

        manual_seed(seed).unwrap();
        let _ = randn_on_device::<bf16>(&[n], Device::Cuda(0)).expect("first bf16 normal");
        let got_bf16_second = to_host_bf16_bits(
            &randn_on_device::<bf16>(&[n], Device::Cuda(0)).expect("second bf16 normal"),
        );

        (
            got_f32_second,
            got_f64_second,
            got_f16_second,
            got_bf16_second,
        )
    };

    assert_eq!(
        got_f32_second, expected_f32_second,
        "f32 second normal call did not continue torch.cuda Philox offset"
    );
    assert_eq!(
        got_f64_second, expected_f64_second,
        "f64 second normal call did not continue torch.cuda Philox offset"
    );
    assert_eq!(
        got_f16_second, expected_f16_second,
        "f16 second normal call did not continue torch.cuda Philox offset"
    );
    assert_eq!(
        got_bf16_second, expected_bf16_second,
        "bf16 second normal call did not continue torch.cuda Philox offset"
    );
}

#[test]
fn normal_gpu_bit_exact_with_torch_cuda_grid_stride_lanes() {
    ensure_init();

    let seed = 2028u64;
    let f32_stride = torch_distribution_stride(1_000_000, 4);
    let f32_n = (f32_stride * 4 + 17) as usize;
    let f32_indices = grid_boundary_indices(f32_n, f32_stride, 4);
    let expected_f32 = torch_cuda_randn_bits("f32", f32_n, seed, 0, &f32_indices);
    let expected_f16 = torch_cuda_randn_bits("f16", f32_n, seed, 0, &f32_indices);
    let expected_bf16 = torch_cuda_randn_bits("bf16", f32_n, seed, 0, &f32_indices);

    let f64_stride = torch_distribution_stride(1_000_000, 2);
    let f64_n = (f64_stride * 2 + 11) as usize;
    let f64_indices = grid_boundary_indices(f64_n, f64_stride, 2);
    let expected_f64 = torch_cuda_randn_bits("f64", f64_n, seed, 0, &f64_indices);

    let (got_f32, got_f16, got_bf16, got_f64) = {
        let _g = SEED_LOCK.lock().unwrap();

        manual_seed(seed).unwrap();
        let got_f32_all = f32_bits(&to_host(
            &randn_on_device::<f32>(&[f32_n], Device::Cuda(0)).expect("grid f32 normal"),
        ));
        let got_f32 = select_bits(&got_f32_all, &f32_indices);

        manual_seed(seed).unwrap();
        let got_f16_all = to_host_f16_bits(
            &randn_on_device::<f16>(&[f32_n], Device::Cuda(0)).expect("grid f16 normal"),
        );
        let got_f16 = select_bits(&got_f16_all, &f32_indices);

        manual_seed(seed).unwrap();
        let got_bf16_all = to_host_bf16_bits(
            &randn_on_device::<bf16>(&[f32_n], Device::Cuda(0)).expect("grid bf16 normal"),
        );
        let got_bf16 = select_bits(&got_bf16_all, &f32_indices);

        manual_seed(seed).unwrap();
        let got_f64_all = f64_bits(&to_host_f64(
            &randn_on_device::<f64>(&[f64_n], Device::Cuda(0)).expect("grid f64 normal"),
        ));
        let got_f64 = select_bits(&got_f64_all, &f64_indices);

        (got_f32, got_f16, got_bf16, got_f64)
    };

    assert_eq!(
        got_f32, expected_f32,
        "f32 normal grid-stride lane boundaries diverge from torch.cuda"
    );
    assert_eq!(
        got_f16, expected_f16,
        "f16 normal grid-stride lane boundaries diverge from torch.cuda"
    );
    assert_eq!(
        got_bf16, expected_bf16,
        "bf16 normal grid-stride lane boundaries diverge from torch.cuda"
    );
    assert_eq!(
        got_f64, expected_f64,
        "f64 normal grid-stride lane boundaries diverge from torch.cuda"
    );
}

/// PROBE 1+3 — UNIFORM on-device, boundary lengths (n not divisible by 4).
/// Reference: PyTorch CUDA distribution layout. A correct kernel is
/// bit-identical at awkward lane boundaries n = 5, 7, 4097.
#[test]
fn uniform_gpu_bit_exact_with_torch_cuda_layout_boundaries() {
    ensure_init();
    for &n in &[1usize, 4, 5, 7, 8, 4096, 4097] {
        let seed = 2024u64;
        let _g = SEED_LOCK.lock().unwrap();
        manual_seed(seed).unwrap();
        let gpu = to_host(&rand_on_device::<f32>(&[n], Device::Cuda(0)).expect("gpu uniform"));
        let expected = torch_cuda_uniform_f32_reference(n, seed, 0);
        assert_eq!(gpu.len(), n, "gpu uniform wrong length at n={n}");
        assert_eq!(
            gpu, expected,
            "on-device uniform kernel diverges from torch.cuda Philox layout at n={n}"
        );
    }
}

/// PROBE 1+3 — force PyTorch's grid-stride unroll lanes. For tensors larger
/// than the thread-grid cap, output positions consume x/y/z/w from each
/// thread's Philox state before the per-thread counter advances.
#[test]
fn uniform_gpu_bit_exact_with_torch_cuda_layout_grid_stride_lanes() {
    ensure_init();
    let seed = 2025u64;
    let stride = torch_distribution_stride(1_000_000, 4);
    let n = (stride * 4 + 17) as usize;
    let _g = SEED_LOCK.lock().unwrap();
    manual_seed(seed).unwrap();
    let gpu = to_host(&rand_on_device::<f32>(&[n], Device::Cuda(0)).expect("gpu uniform"));
    let expected = torch_cuda_uniform_f32_reference(n, seed, 0);
    assert_eq!(gpu, expected, "grid-stride f32 uniform layout mismatch");
}

/// PROBE 1 — UNIFORM range invariant: every on-device value strictly in [0,1).
#[test]
fn uniform_gpu_strict_unit_interval() {
    ensure_init();
    let v = {
        let _g = SEED_LOCK.lock().unwrap();
        manual_seed(11).unwrap();
        to_host(&rand_on_device::<f32>(&[1_000_000], Device::Cuda(0)).expect("gpu uniform"))
    };
    for &x in &v {
        assert!(
            (0.0..1.0).contains(&x),
            "on-device uniform produced {x} outside [0,1)"
        );
    }
}

/// PROBE 6 — consecutive on-device UNIFORM calls continue PyTorch's CUDA
/// generator offset. After a call of n, PyTorch advances by
/// calls_per_thread * 4 curand elements, i.e. calls_per_thread Philox counters
/// in this manager's counter units.
#[test]
fn uniform_gpu_consecutive_calls_continue_torch_cuda_offset() {
    ensure_init();
    let stride = torch_distribution_stride(1_000_000, 4);
    let n = (stride * 4 + 17) as usize;
    let seed = 77u64;
    let (a, b) = {
        let _g = SEED_LOCK.lock().unwrap();
        manual_seed(seed).unwrap();
        let a = to_host(&rand_on_device::<f32>(&[n], Device::Cuda(0)).expect("a"));
        let b = to_host(&rand_on_device::<f32>(&[n], Device::Cuda(0)).expect("b"));
        (a, b)
    };
    assert_eq!(
        a,
        torch_cuda_uniform_f32_reference(n, seed, 0),
        "first call mismatch"
    );
    assert_eq!(
        b,
        torch_cuda_uniform_f32_reference(n, seed, calls_per_thread(n, 4)),
        "second on-device call does not continue torch.cuda Philox offset"
    );
}

/// PROBE 4 — NORMAL on-device, ODD n: each thread writes grid-stride lanes with
/// per-lane bounds guards. For odd n exactly n finite values must be written.
#[test]
fn normal_gpu_odd_length_finite_count() {
    ensure_init();
    for &n in &[1usize, 7, 4097] {
        let gpu = {
            let _g = SEED_LOCK.lock().unwrap();
            manual_seed(5).unwrap();
            to_host(&randn_on_device::<f32>(&[n], Device::Cuda(0)).expect("gpu normal"))
        };
        assert_eq!(gpu.len(), n, "gpu normal wrong length at n={n}");
        for (i, &x) in gpu.iter().enumerate() {
            assert!(
                x.is_finite(),
                "on-device normal value[{i}]={x} not finite at n={n}"
            );
        }
    }
}

/// PROBE 2 — NORMAL distribution moments from the on-device kernel. The
/// bit-exact tests above prove PyTorch CUDA parity; this keeps a broad
/// standard-normal sanity check over a larger sample.
#[test]
fn normal_gpu_moments_standard_normal() {
    ensure_init();
    let n = 1_000_000usize;
    let v = {
        let _g = SEED_LOCK.lock().unwrap();
        manual_seed(13).unwrap();
        to_host(&randn_on_device::<f32>(&[n], Device::Cuda(0)).expect("gpu normal"))
    };
    assert_eq!(v.len(), n);

    let nf = n as f64;
    let mut s1 = 0.0f64;
    for &x in &v {
        assert!(x.is_finite(), "normal value not finite: {x}");
        s1 += x as f64;
    }
    let mean = s1 / nf;
    let (mut m2, mut m3, mut m4) = (0.0f64, 0.0f64, 0.0f64);
    for &x in &v {
        let d = x as f64 - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= nf;
    m3 /= nf;
    m4 /= nf;
    let std = m2.sqrt();
    let skew = m3 / m2.powf(1.5);
    let kurt = m4 / (m2 * m2); // standard normal == 3.0

    assert!((mean).abs() < 0.01, "on-device normal mean {mean} != ~0");
    assert!((std - 1.0).abs() < 0.01, "on-device normal std {std} != ~1");
    assert!(skew.abs() < 0.05, "on-device normal skew {skew} != ~0");
    assert!(
        (kurt - 3.0).abs() < 0.1,
        "on-device normal kurtosis {kurt} != ~3"
    );
}

/// F64 UNIFORM uses Rust-generated PTX, not CUDA C/NVRTC/libdevice. Exercise
/// odd lengths so both lanes of the two-u32 -> f64 packing and the last-lane
/// guard run through the public backend API.
#[test]
fn uniform_f64_gpu_bit_exact_with_torch_cuda_layout() {
    ensure_init();
    for &n in &[1usize, 2, 3, 7, 4097] {
        let seed = 2026u64;
        let t = {
            let _g = SEED_LOCK.lock().unwrap();
            manual_seed(seed).unwrap();
            rand_on_device::<f64>(&[n], Device::Cuda(0)).expect("gpu f64 uniform")
        };
        assert_eq!(t.device(), Device::Cuda(0), "f64 uniform must stay CUDA");
        let v = to_host_f64(&t);
        assert_eq!(v.len(), n, "f64 uniform wrong length at n={n}");
        assert_eq!(
            v,
            torch_cuda_uniform_f64_reference(n, seed, 0),
            "f64 uniform torch.cuda layout mismatch at n={n}"
        );
    }
}

#[test]
fn uniform_f64_gpu_grid_stride_and_consecutive_calls_match_torch_cuda_layout() {
    ensure_init();
    let seed = 2027u64;
    let stride = torch_distribution_stride(1_000_000, 2);
    let n = (stride * 2 + 11) as usize;
    let (a, b) = {
        let _g = SEED_LOCK.lock().unwrap();
        manual_seed(seed).unwrap();
        let a = to_host_f64(&rand_on_device::<f64>(&[n], Device::Cuda(0)).expect("a"));
        let b = to_host_f64(&rand_on_device::<f64>(&[n], Device::Cuda(0)).expect("b"));
        (a, b)
    };
    assert_eq!(
        a,
        torch_cuda_uniform_f64_reference(n, seed, 0),
        "f64 first grid-stride call mismatch"
    );
    assert_eq!(
        b,
        torch_cuda_uniform_f64_reference(n, seed, calls_per_thread(n, 2)),
        "f64 second grid-stride call did not continue torch.cuda offset"
    );
}

/// F64 NORMAL mirrors PyTorch's double Box-Muller structure: two 32-bit Philox
/// lanes per uniform, double log/sqrt/sincos math, two outputs per thread with
/// a guarded odd tail. The math is emitted as Rust-owned PTX.
#[test]
fn normal_f64_gpu_odd_tail_and_moments() {
    ensure_init();
    for &n in &[1usize, 7, 4097] {
        let t = {
            let _g = SEED_LOCK.lock().unwrap();
            manual_seed(41).unwrap();
            randn_on_device::<f64>(&[n], Device::Cuda(0)).expect("gpu f64 normal")
        };
        assert_eq!(t.device(), Device::Cuda(0), "f64 normal must stay CUDA");
        let v = to_host_f64(&t);
        assert_eq!(v.len(), n, "f64 normal wrong length at n={n}");
        for (i, &x) in v.iter().enumerate() {
            assert!(
                x.is_finite(),
                "f64 normal value[{i}]={x} not finite at n={n}"
            );
        }
    }

    let n = 200_000usize;
    let t = {
        let _g = SEED_LOCK.lock().unwrap();
        manual_seed(43).unwrap();
        randn_on_device::<f64>(&[n], Device::Cuda(0)).expect("gpu f64 normal moments")
    };
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "f64 normal moments must stay CUDA"
    );
    let v = to_host_f64(&t);
    assert_eq!(v.len(), n);

    let nf = n as f64;
    let mean = v
        .iter()
        .inspect(|&&x| assert!(x.is_finite(), "f64 normal value not finite: {x}"))
        .sum::<f64>()
        / nf;
    let variance = v
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / nf;
    let std = variance.sqrt();

    assert!(mean.abs() < 0.015, "f64 normal mean {mean} != ~0");
    assert!((std - 1.0).abs() < 0.015, "f64 normal std {std} != ~1");
}
