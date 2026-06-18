#![cfg(feature = "cuda")]

use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::grad_fns::indexing::{index_add, index_copy};
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};
use std::process::Command;
use std::sync::Once;

static INIT: Once = Once::new();

fn ensure_cuda() {
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu f32")
}

fn cpu_f64(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu f64")
}

fn cpu_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(f16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu f16")
}

fn cpu_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(bf16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu bf16")
}

fn cuda_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    cpu_f32(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(requires_grad)
}

fn cuda_f64(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    cpu_f64(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(requires_grad)
}

fn cuda_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    cpu_f16(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda f16")
        .requires_grad_(requires_grad)
}

fn cuda_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    cpu_bf16(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda bf16")
        .requires_grad_(requires_grad)
}

fn cuda_idx(data: &[i64], shape: &[usize]) -> IntTensor<i64> {
    IntTensor::from_vec(data.to_vec(), shape.to_vec())
        .expect("index")
        .to(Device::Cuda(0))
        .expect("index to cuda")
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu").data_vec().expect("data").to_vec()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("cpu").data_vec().expect("data").to_vec()
}

fn host_f16_bits(t: &Tensor<f16>) -> Vec<u16> {
    t.cpu()
        .expect("cpu f16")
        .data_vec()
        .expect("f16 data")
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn host_bf16_bits(t: &Tensor<bf16>) -> Vec<u16> {
    t.cpu()
        .expect("cpu bf16")
        .data_vec()
        .expect("bf16 data")
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

#[derive(Debug)]
struct TorchBits {
    out: Vec<u16>,
    grad_input: Vec<u16>,
    grad_source: Vec<u16>,
}

fn parse_bits_line(line: &str, label: &str) -> Option<Vec<u16>> {
    let rest = line.strip_prefix(label)?.trim();
    Some(
        rest.split_whitespace()
            .map(|hex| u16::from_str_radix(hex, 16).expect("oracle hex u16"))
            .collect(),
    )
}

fn torch_indexing_bits(dtype: &str, case_name: &str) -> TorchBits {
    let script = r#"
import sys
import torch

dtype_name, case_name = sys.argv[1], sys.argv[2]
if not torch.cuda.is_available():
    raise SystemExit("torch CUDA oracle unavailable")

dtypes = {"f16": torch.float16, "bf16": torch.bfloat16}
dtype = dtypes[dtype_name]

def bits(t):
    a = t.detach().cpu().contiguous().view(torch.int16).numpy().view("uint16").reshape(-1)
    return " ".join(format(int(x), "04x") for x in a)

if case_name == "index_add_duplicate_alpha":
    x = torch.tensor([1., 2., 3., 4., 5., 6.], device="cuda", dtype=dtype).reshape(2, 3).clone().detach().requires_grad_(True)
    src = torch.tensor([10., 20., 30., 40.], device="cuda", dtype=dtype).reshape(2, 2).clone().detach().requires_grad_(True)
    index = torch.tensor([1, 1], device="cuda", dtype=torch.long)
    out = x.index_add(1, index, src, alpha=1.5)
elif case_name == "index_copy_scalar_source":
    x = torch.tensor([1., 2., 3., 4.], device="cuda", dtype=dtype).clone().detach().requires_grad_(True)
    src = torch.tensor(9., device="cuda", dtype=dtype).clone().detach().requires_grad_(True)
    index = torch.tensor([1], device="cuda", dtype=torch.long)
    out = x.index_copy(0, index, src)
else:
    raise SystemExit(f"unknown case {case_name}")

out.sum().backward()
torch.cuda.synchronize()
print("out", bits(out))
print("grad_input", bits(x.grad))
print("grad_source", bits(src.grad))
"#;
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(dtype)
        .arg(case_name)
        .output()
        .expect("launch torch indexing oracle");
    assert!(
        output.status.success(),
        "torch CUDA indexing oracle failed: status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("oracle stdout utf8");
    let mut out = None;
    let mut grad_input = None;
    let mut grad_source = None;
    for line in stdout.lines() {
        out = out.or_else(|| parse_bits_line(line, "out"));
        grad_input = grad_input.or_else(|| parse_bits_line(line, "grad_input"));
        grad_source = grad_source.or_else(|| parse_bits_line(line, "grad_source"));
    }
    TorchBits {
        out: out.expect("oracle out bits"),
        grad_input: grad_input.expect("oracle input grad bits"),
        grad_source: grad_source.expect("oracle source grad bits"),
    }
}

#[test]
fn cuda_index_add_f32_forward_backward_alpha_stays_resident() {
    ensure_cuda();
    let input = cuda_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let source = cuda_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2], true);
    let index = cuda_idx(&[2, 0], &[2]);

    let out = index_add(&input, 1, &index, &source, 2.5).expect("index_add cuda");
    assert!(out.is_cuda(), "forward must stay CUDA-resident");
    assert_eq!(host_f32(&out), vec![51.0, 2.0, 28.0, 104.0, 5.0, 81.0]);

    backward(&out.sum_all().expect("sum")).expect("index_add backward");
    let gi = input.grad().expect("grad access").expect("input grad");
    let gs = source.grad().expect("grad access").expect("source grad");
    assert!(
        gi.is_cuda() && gs.is_cuda(),
        "grads must stay CUDA-resident"
    );
    assert_eq!(host_f32(&gi), vec![1.0; 6]);
    assert_eq!(host_f32(&gs), vec![2.5; 4]);
}

#[test]
fn cuda_index_add_f16_bf16_duplicate_alpha_matches_torch_bits_and_stays_resident() {
    ensure_cuda();
    let index = cuda_idx(&[1, 1], &[2]);

    let expected_f16 = torch_indexing_bits("f16", "index_add_duplicate_alpha");
    let input_f16 = cuda_f16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let source_f16 = cuda_f16(&[10.0, 20.0, 30.0, 40.0], &[2, 2], true);
    let out_f16 = index_add(&input_f16, 1, &index, &source_f16, 1.5).expect("f16 index_add cuda");
    assert!(out_f16.is_cuda(), "f16 forward must stay CUDA-resident");
    backward(&out_f16.sum_all().expect("f16 sum")).expect("f16 index_add backward");
    let gi_f16 = input_f16
        .grad()
        .expect("f16 grad access")
        .expect("input grad");
    let gs_f16 = source_f16
        .grad()
        .expect("f16 source grad access")
        .expect("source grad");
    assert!(gi_f16.is_cuda() && gs_f16.is_cuda());
    assert_eq!(host_f16_bits(&out_f16), expected_f16.out);
    assert_eq!(host_f16_bits(&gi_f16), expected_f16.grad_input);
    assert_eq!(host_f16_bits(&gs_f16), expected_f16.grad_source);

    let expected_bf16 = torch_indexing_bits("bf16", "index_add_duplicate_alpha");
    let input_bf16 = cuda_bf16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let source_bf16 = cuda_bf16(&[10.0, 20.0, 30.0, 40.0], &[2, 2], true);
    let out_bf16 =
        index_add(&input_bf16, 1, &index, &source_bf16, 1.5).expect("bf16 index_add cuda");
    assert!(out_bf16.is_cuda(), "bf16 forward must stay CUDA-resident");
    backward(&out_bf16.sum_all().expect("bf16 sum")).expect("bf16 index_add backward");
    let gi_bf16 = input_bf16
        .grad()
        .expect("bf16 grad access")
        .expect("input grad");
    let gs_bf16 = source_bf16
        .grad()
        .expect("bf16 source grad access")
        .expect("source grad");
    assert!(gi_bf16.is_cuda() && gs_bf16.is_cuda());
    assert_eq!(host_bf16_bits(&out_bf16), expected_bf16.out);
    assert_eq!(host_bf16_bits(&gi_bf16), expected_bf16.grad_input);
    assert_eq!(host_bf16_bits(&gs_bf16), expected_bf16.grad_source);
}

#[test]
fn cuda_index_copy_f64_forward_backward_stays_resident() {
    ensure_cuda();
    let input = cuda_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let source = cuda_f64(&[10.0, 20.0, 30.0, 40.0], &[2, 2], true);
    let index = cuda_idx(&[2, 0], &[2]);

    let out = index_copy(&input, 1, &index, &source).expect("index_copy cuda");
    assert!(out.is_cuda(), "forward must stay CUDA-resident");
    assert_eq!(host_f64(&out), vec![20.0, 2.0, 10.0, 40.0, 5.0, 30.0]);

    backward(&out.sum_all().expect("sum")).expect("index_copy backward");
    let gi = input.grad().expect("grad access").expect("input grad");
    let gs = source.grad().expect("grad access").expect("source grad");
    assert!(
        gi.is_cuda() && gs.is_cuda(),
        "grads must stay CUDA-resident"
    );
    assert_eq!(host_f64(&gi), vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    assert_eq!(host_f64(&gs), vec![1.0; 4]);
}

#[test]
fn cuda_index_copy_f16_bf16_scalar_source_matches_torch_bits_and_stays_resident() {
    ensure_cuda();
    let index = cuda_idx(&[1], &[1]);

    let expected_f16 = torch_indexing_bits("f16", "index_copy_scalar_source");
    let input_f16 = cuda_f16(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let source_f16 = cuda_f16(&[9.0], &[], true);
    let out_f16 = index_copy(&input_f16, 0, &index, &source_f16).expect("f16 index_copy cuda");
    assert!(out_f16.is_cuda(), "f16 forward must stay CUDA-resident");
    backward(&out_f16.sum_all().expect("f16 sum")).expect("f16 index_copy backward");
    let gi_f16 = input_f16
        .grad()
        .expect("f16 grad access")
        .expect("input grad");
    let gs_f16 = source_f16
        .grad()
        .expect("f16 source grad access")
        .expect("source grad");
    assert!(gi_f16.is_cuda() && gs_f16.is_cuda());
    assert_eq!(host_f16_bits(&out_f16), expected_f16.out);
    assert_eq!(host_f16_bits(&gi_f16), expected_f16.grad_input);
    assert_eq!(host_f16_bits(&gs_f16), expected_f16.grad_source);

    let expected_bf16 = torch_indexing_bits("bf16", "index_copy_scalar_source");
    let input_bf16 = cuda_bf16(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let source_bf16 = cuda_bf16(&[9.0], &[], true);
    let out_bf16 = index_copy(&input_bf16, 0, &index, &source_bf16).expect("bf16 index_copy cuda");
    assert!(out_bf16.is_cuda(), "bf16 forward must stay CUDA-resident");
    backward(&out_bf16.sum_all().expect("bf16 sum")).expect("bf16 index_copy backward");
    let gi_bf16 = input_bf16
        .grad()
        .expect("bf16 grad access")
        .expect("input grad");
    let gs_bf16 = source_bf16
        .grad()
        .expect("bf16 source grad access")
        .expect("source grad");
    assert!(gi_bf16.is_cuda() && gs_bf16.is_cuda());
    assert_eq!(host_bf16_bits(&out_bf16), expected_bf16.out);
    assert_eq!(host_bf16_bits(&gi_bf16), expected_bf16.grad_input);
    assert_eq!(host_bf16_bits(&gs_bf16), expected_bf16.grad_source);
}

#[test]
fn cuda_index_add_duplicate_indices_accumulate_and_empty_index_clones() {
    ensure_cuda();
    let input = cuda_f32(&[1.0, 2.0, 3.0], &[1, 3], false);
    let source = cuda_f32(&[10.0, 20.0], &[1, 2], false);
    let index = cuda_idx(&[1, 1], &[2]);

    let out = index_add(&input, 1, &index, &source, 1.0).expect("duplicate add");
    assert!(out.is_cuda());
    assert_eq!(host_f32(&out), vec![1.0, 32.0, 3.0]);

    let empty_index = cuda_idx(&[], &[0]);
    let empty_source = cuda_f32(&[], &[1, 0], false);
    let cloned = index_copy(&input, 1, &empty_index, &empty_source).expect("empty copy");
    assert!(cloned.is_cuda(), "empty index must not demote to CPU");
    assert_eq!(host_f32(&cloned), vec![1.0, 2.0, 3.0]);
}

#[test]
fn cuda_index_add_view_input_uses_logical_values_on_device() {
    ensure_cuda();
    let base = cuda_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let input = base.transpose(0, 1).expect("transpose"); // [[1,4],[2,5],[3,6]]
    assert!(input.is_cuda());
    assert!(!input.is_contiguous(), "probe must use a strided CUDA view");
    let source = cuda_f32(&[10.0, 20.0, 30.0], &[3, 1], false);
    let index = cuda_idx(&[1], &[1]);

    let out = index_add(&input, 1, &index, &source, 2.0).expect("view index_add");
    assert!(out.is_cuda());
    assert_eq!(host_f32(&out), vec![1.0, 24.0, 2.0, 45.0, 3.0, 66.0]);
}
