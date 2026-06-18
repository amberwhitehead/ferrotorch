//! CUDA dropout Philox parity probes for crosslink #2018.
//!
//! The oracle is derived from `/home/doll/pytorch/aten/src/ATen/native/cuda/Dropout.cu`:
//! one Philox/cuRAND state per logical CUDA thread, `curand_uniform4` lane
//! consumption, PyTorch's contiguous vector-size choice, and generator advance
//! by `((nelem - 1) / (block * grid * UNROLL) + 1)` 4x32 counter groups.

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, manual_seed};
use ferrotorch_gpu::init_cuda_backend;
use ferrotorch_nn::{Dropout, Module};
use half::{bf16, f16};
use std::sync::Mutex;

static SEED_LOCK: Mutex<()> = Mutex::new(());

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

fn ensure_init() {
    if !ferrotorch_core::gpu_dispatch::has_gpu_backend() {
        init_cuda_backend().expect("init_cuda_backend");
    }
}

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

fn curand_uniform_f32(word: u32) -> f32 {
    (word as f32).mul_add(2.328_306_4e-10, 1.164_153_2e-10)
}

fn torch_dropout_stride(n: usize) -> u64 {
    let block = 256u64;
    let grid = (n as u64).div_ceil(block).max(1);
    // All probes in this file deliberately use n <= 8, so PyTorch's SM-count
    // cap cannot reduce the one-block grid selected by Dropout.cu.
    assert!(
        grid == 1,
        "test oracle only covers one-block small-n probes"
    );
    block
}

fn torch_dropout_vector_size(n: usize, element_size: usize) -> u32 {
    let mut vector_size = 16 / element_size as u32;
    vector_size = vector_size.min(8);
    while vector_size > 1 && !n.is_multiple_of(vector_size as usize) {
        vector_size /= 2;
    }
    vector_size
}

fn calls_per_thread(n: usize) -> u64 {
    let stride = torch_dropout_stride(n);
    ((n as u64 - 1) / (stride * 4)) + 1
}

fn torch_cuda_dropout_mask(
    n: usize,
    element_size: usize,
    drop_probability: f64,
    seed: u64,
    base_counter: u64,
) -> Vec<bool> {
    let keep_probability = (1.0 - drop_probability) as f32;
    let stride = torch_dropout_stride(n);
    let vector_size = torch_dropout_vector_size(n, element_size);
    let mut keep = vec![false; n];

    if vector_size == 1 {
        for tid in 0..stride {
            let mut linear = tid;
            let mut counter = base_counter;
            while linear < n as u64 {
                let words = philox4(seed, counter, tid);
                for (lane, word) in words.into_iter().enumerate() {
                    let idx = linear + stride * lane as u64;
                    if idx < n as u64 {
                        keep[idx as usize] = curand_uniform_f32(word) < keep_probability;
                    }
                }
                linear += stride * 4;
                counter += 1;
            }
        }
    } else {
        let groups = vector_size.div_ceil(4);
        for tid in 0..stride {
            let mut linear = tid * vector_size as u64;
            let mut counter = base_counter;
            while linear < n as u64 {
                for group in 0..groups {
                    let words = philox4(seed, counter + group as u64, tid);
                    let first_lane = (group * 4) as usize;
                    let last_lane = (first_lane + 4).min(vector_size as usize);
                    for lane in first_lane..last_lane {
                        let idx = linear + lane as u64;
                        if idx < n as u64 {
                            keep[idx as usize] =
                                curand_uniform_f32(words[lane - first_lane]) < keep_probability;
                        }
                    }
                }
                linear += stride * vector_size as u64;
                counter += groups as u64;
            }
        }
    }

    keep
}

fn tensor_f32(data: Vec<f32>, requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.clone()),
        vec![data.len()],
        requires_grad,
    )
    .unwrap()
}

fn to_host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.to(Device::Cpu)
        .expect("to CPU")
        .data()
        .expect("cpu data")
        .to_vec()
}

fn to_host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.to(Device::Cpu)
        .expect("to CPU")
        .data()
        .expect("cpu data")
        .to_vec()
}

#[test]
fn cuda_dropout_f32_matches_torch_philox_vectorized_and_scalar_layouts() {
    ensure_init();
    let _guard = SEED_LOCK.lock().unwrap();
    let seed = 3018u64;
    let drop_probability = 0.25;
    let layer = Dropout::<f32>::new(drop_probability).unwrap();

    for &n in &[2usize, 4, 5] {
        let input: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let keep = torch_cuda_dropout_mask(n, 4, drop_probability, seed, 0);
        let scale = 1.0f32 / (1.0 - drop_probability as f32);
        let expected: Vec<f32> = input
            .iter()
            .zip(keep.iter())
            .map(|(&x, &k)| if k { x * scale } else { 0.0 })
            .collect();

        manual_seed(seed).unwrap();
        let x = tensor_f32(input, false)
            .to(Device::Cuda(0))
            .expect("to cuda");
        let y = layer.forward(&x).expect("dropout forward");
        assert_eq!(y.device(), Device::Cuda(0), "dropout output must stay CUDA");
        assert_eq!(
            to_host_f32(&y),
            expected,
            "f32 CUDA dropout mask diverged from PyTorch source layout at n={n}"
        );
    }
}

#[test]
fn cuda_dropout_consecutive_calls_advance_like_torch_source() {
    ensure_init();
    let _guard = SEED_LOCK.lock().unwrap();
    let seed = 4018u64;
    let n = 4usize;
    let drop_probability = 0.5;
    let layer = Dropout::<f32>::new(drop_probability).unwrap();
    let input = vec![1.0f32; n];
    let scale = 1.0f32 / (1.0 - drop_probability as f32);

    manual_seed(seed).unwrap();
    let x = tensor_f32(input, false)
        .to(Device::Cuda(0))
        .expect("to cuda");
    let y1 = layer.forward(&x).expect("dropout forward 1");
    let y2 = layer.forward(&x).expect("dropout forward 2");

    let expected1: Vec<f32> = torch_cuda_dropout_mask(n, 4, drop_probability, seed, 0)
        .into_iter()
        .map(|k| if k { scale } else { 0.0 })
        .collect();
    let expected2: Vec<f32> =
        torch_cuda_dropout_mask(n, 4, drop_probability, seed, calls_per_thread(n))
            .into_iter()
            .map(|k| if k { scale } else { 0.0 })
            .collect();

    assert_eq!(to_host_f32(&y1), expected1, "first dropout call mismatch");
    assert_eq!(
        to_host_f32(&y2),
        expected2,
        "second dropout call did not continue the PyTorch Philox stream"
    );
}

#[test]
fn cuda_dropout_backward_uses_resident_forward_mask() {
    ensure_init();
    let _guard = SEED_LOCK.lock().unwrap();
    let seed = 5018u64;
    let n = 5usize;
    let drop_probability = 0.5;
    let layer = Dropout::<f32>::new(drop_probability).unwrap();
    let scale = 1.0f32 / (1.0 - drop_probability as f32);
    let expected_grad: Vec<f32> = torch_cuda_dropout_mask(n, 4, drop_probability, seed, 0)
        .into_iter()
        .map(|k| if k { scale } else { 0.0 })
        .collect();

    manual_seed(seed).unwrap();
    let x = tensor_f32(vec![1.0; n], true)
        .to(Device::Cuda(0))
        .expect("to cuda");
    let y = layer.forward(&x).expect("dropout forward");
    let grad_output = tensor_f32(vec![1.0; n], false)
        .to(Device::Cuda(0))
        .expect("grad to cuda");
    let grads = y
        .grad_fn()
        .expect("dropout grad fn")
        .backward(&grad_output)
        .expect("dropout backward");
    let dx = grads[0].as_ref().expect("input grad");

    assert_eq!(dx.device(), Device::Cuda(0), "dropout grad must stay CUDA");
    assert_eq!(
        to_host_f32(dx),
        expected_grad,
        "dropout backward did not use the forward Philox mask"
    );
}

#[test]
fn cuda_dropout_f64_uses_f64_backend_and_torch_mask_layout() {
    ensure_init();
    let _guard = SEED_LOCK.lock().unwrap();
    let seed = 6018u64;
    let drop_probability = 0.5;
    for &n in &[2usize, 3] {
        let input: Vec<f64> = (0..n).map(|i| i as f64 + 1.25).collect();
        let keep = torch_cuda_dropout_mask(n, 8, drop_probability, seed, 0);
        let scale = 1.0 / (1.0 - drop_probability);
        let expected: Vec<f64> = input
            .iter()
            .zip(keep.iter())
            .map(|(&x, &k)| if k { x * scale } else { 0.0 })
            .collect();

        manual_seed(seed).unwrap();
        let x = Tensor::from_storage(TensorStorage::cpu(input), vec![n], false)
            .unwrap()
            .to(Device::Cuda(0))
            .expect("to cuda");
        let y = Dropout::<f64>::new(drop_probability)
            .unwrap()
            .forward(&x)
            .expect("dropout f64 forward");

        assert_eq!(y.device(), Device::Cuda(0), "f64 dropout must stay CUDA");
        assert_eq!(
            to_host_f64(&y),
            expected,
            "f64 dropout source-layout mismatch at n={n}"
        );
    }
}

#[test]
fn cuda_dropout_half_and_bfloat_keep_dtype_and_mask_layout() {
    ensure_init();
    let _guard = SEED_LOCK.lock().unwrap();
    let seed = 7018u64;
    let drop_probability = 0.5;
    let scale = 1.0f32 / (1.0 - drop_probability as f32);

    for &n in &[2usize, 3, 4, 8] {
        let keep = torch_cuda_dropout_mask(n, 2, drop_probability, seed, 0);

        manual_seed(seed).unwrap();
        let x_f16 = Tensor::from_storage(
            TensorStorage::cpu(vec![f16::from_f32(1.0); n]),
            vec![n],
            false,
        )
        .unwrap()
        .to(Device::Cuda(0))
        .expect("f16 to cuda");
        let y_f16 = Dropout::<f16>::new(drop_probability)
            .unwrap()
            .forward(&x_f16)
            .expect("dropout f16 forward");
        assert_eq!(
            y_f16.device(),
            Device::Cuda(0),
            "f16 dropout must stay CUDA"
        );
        let got_f16: Vec<f32> = y_f16
            .to(Device::Cpu)
            .expect("f16 to cpu")
            .data()
            .expect("f16 cpu data")
            .iter()
            .map(|x| x.to_f32())
            .collect();
        let expected_f16: Vec<f32> = keep
            .iter()
            .map(|&k| {
                if k {
                    f16::from_f32(scale).to_f32()
                } else {
                    0.0
                }
            })
            .collect();
        assert_eq!(
            got_f16, expected_f16,
            "f16 dropout mask/layout mismatch at n={n}"
        );

        manual_seed(seed).unwrap();
        let x_bf16 = Tensor::from_storage(
            TensorStorage::cpu(vec![bf16::from_f32(1.0); n]),
            vec![n],
            false,
        )
        .unwrap()
        .to(Device::Cuda(0))
        .expect("bf16 to cuda");
        let y_bf16 = Dropout::<bf16>::new(drop_probability)
            .unwrap()
            .forward(&x_bf16)
            .expect("dropout bf16 forward");
        assert_eq!(
            y_bf16.device(),
            Device::Cuda(0),
            "bf16 dropout must stay CUDA"
        );
        let got_bf16: Vec<f32> = y_bf16
            .to(Device::Cpu)
            .expect("bf16 to cpu")
            .data()
            .expect("bf16 cpu data")
            .iter()
            .map(|x| x.to_f32())
            .collect();
        let expected_bf16: Vec<f32> = keep
            .iter()
            .map(|&k| {
                if k {
                    bf16::from_f32(scale).to_f32()
                } else {
                    0.0
                }
            })
            .collect();
        assert_eq!(
            got_bf16, expected_bf16,
            "bf16 dropout mask/layout mismatch at n={n}"
        );
    }
}
