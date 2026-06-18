#![cfg(feature = "cuda")]

//! CUDA `scatter_reduce` must not run as a CPU fold plus re-upload.
//! These cases mirror live torch 2.11.0+cu130 for larger `src`, duplicate
//! destinations, `include_self=false`, and all shipped reduce modes.

use ferrotorch_core::grad_fns::indexing::{ScatterReduce, scatter_reduce};
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

fn cuda_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
}

fn cuda_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
}

fn cuda_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(f16::from_f32).collect()),
        shape.to_vec(),
        false,
    )
    .unwrap()
    .to(Device::Cuda(0))
    .unwrap()
    .requires_grad_(requires_grad)
}

fn cuda_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(bf16::from_f32).collect()),
        shape.to_vec(),
        false,
    )
    .unwrap()
    .to(Device::Cuda(0))
    .unwrap()
    .requires_grad_(requires_grad)
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().unwrap().data_vec().unwrap()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().unwrap().data_vec().unwrap()
}

fn host_f16_bits(t: &Tensor<f16>) -> Vec<u16> {
    t.cpu()
        .unwrap()
        .data_vec()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn host_bf16_bits(t: &Tensor<bf16>) -> Vec<u16> {
    t.cpu()
        .unwrap()
        .data_vec()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

#[derive(Debug)]
struct TorchScatterBits {
    out: Vec<u16>,
    grad_input: Option<Vec<u16>>,
    grad_src: Option<Vec<u16>>,
}

fn parse_bits_line(line: &str, label: &str) -> Option<Vec<u16>> {
    let rest = line.strip_prefix(label)?.trim();
    Some(
        rest.split_whitespace()
            .map(|hex| u16::from_str_radix(hex, 16).expect("oracle hex u16"))
            .collect(),
    )
}

fn torch_reduce_name(reduce: ScatterReduce) -> &'static str {
    match reduce {
        ScatterReduce::Sum => "sum",
        ScatterReduce::Mean => "mean",
        ScatterReduce::Prod => "prod",
        ScatterReduce::Amax => "amax",
        ScatterReduce::Amin => "amin",
    }
}

fn torch_scatter_reduce_bits(
    dtype: &str,
    reduce: ScatterReduce,
    include_self: bool,
    backward: bool,
) -> TorchScatterBits {
    let script = r#"
import sys
import torch

dtype_name, reduce, include_self_raw, backward_raw = sys.argv[1:5]
if not torch.cuda.is_available():
    raise SystemExit("torch CUDA oracle unavailable")

dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
include_self = include_self_raw == "1"
backward = backward_raw == "1"

def bits(t):
    a = t.detach().cpu().contiguous().view(torch.int16).numpy().view("uint16").reshape(-1)
    return " ".join(format(int(x), "04x") for x in a)

x = torch.tensor([1., 2., 3., 4., 5., 6.], device="cuda", dtype=dtype).reshape(2, 3).clone().detach()
src = torch.tensor([10., 20., 40., 50.], device="cuda", dtype=dtype).reshape(2, 2).clone().detach()
if backward:
    x.requires_grad_(True)
    src.requires_grad_(True)
index = torch.tensor([[0, 1], [1, 0]], device="cuda", dtype=torch.long)
out = x.scatter_reduce(0, index, src, reduce=reduce, include_self=include_self)
if backward:
    out.sum().backward()
torch.cuda.synchronize()
print("out", bits(out))
if backward:
    print("grad_input", bits(x.grad))
    print("grad_src", bits(src.grad))
"#;
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(dtype)
        .arg(torch_reduce_name(reduce))
        .arg(if include_self { "1" } else { "0" })
        .arg(if backward { "1" } else { "0" })
        .output()
        .expect("launch torch scatter_reduce oracle");
    assert!(
        output.status.success(),
        "torch CUDA scatter_reduce oracle failed: status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("oracle stdout utf8");
    let mut out = None;
    let mut grad_input = None;
    let mut grad_src = None;
    for line in stdout.lines() {
        out = out.or_else(|| parse_bits_line(line, "out"));
        grad_input = grad_input.or_else(|| parse_bits_line(line, "grad_input"));
        grad_src = grad_src.or_else(|| parse_bits_line(line, "grad_src"));
    }
    TorchScatterBits {
        out: out.expect("oracle output bits"),
        grad_input,
        grad_src,
    }
}

#[test]
fn cuda_scatter_reduce_all_modes_match_torch_and_stay_resident() {
    ensure_cuda();
    let input = cuda_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let src = cuda_f32(&[10.0, 20.0, 99.0, 40.0, 50.0, 99.0], &[2, 3]);
    let index = [0, 1, 1, 0];
    let index_shape = [2, 2];

    let cases = [
        (
            ScatterReduce::Sum,
            true,
            vec![11.0, 52.0, 3.0, 44.0, 25.0, 6.0],
        ),
        (
            ScatterReduce::Sum,
            false,
            vec![10.0, 50.0, 3.0, 40.0, 20.0, 6.0],
        ),
        (
            ScatterReduce::Prod,
            true,
            vec![10.0, 100.0, 3.0, 160.0, 100.0, 6.0],
        ),
        (
            ScatterReduce::Prod,
            false,
            vec![10.0, 50.0, 3.0, 40.0, 20.0, 6.0],
        ),
        (
            ScatterReduce::Amax,
            true,
            vec![10.0, 50.0, 3.0, 40.0, 20.0, 6.0],
        ),
        (
            ScatterReduce::Amin,
            true,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        (
            ScatterReduce::Amin,
            false,
            vec![10.0, 50.0, 3.0, 40.0, 20.0, 6.0],
        ),
    ];

    for (reduce, include_self, expected) in cases {
        let out =
            scatter_reduce(&input, 0, &index, &index_shape, &src, reduce, include_self).unwrap();
        assert!(
            out.is_cuda(),
            "{reduce:?} include_self={include_self} output must stay CUDA"
        );
        assert_eq!(
            host_f32(&out),
            expected,
            "{reduce:?} include_self={include_self}"
        );
    }
}

#[test]
fn cuda_scatter_reduce_f16_bf16_all_modes_match_torch_bits_and_stay_resident() {
    ensure_cuda();
    let index = [0usize, 1, 1, 0];
    let index_shape = [2, 2];
    let cases = [
        (ScatterReduce::Sum, true),
        (ScatterReduce::Sum, false),
        (ScatterReduce::Mean, true),
        (ScatterReduce::Mean, false),
        (ScatterReduce::Prod, true),
        (ScatterReduce::Prod, false),
        (ScatterReduce::Amax, true),
        (ScatterReduce::Amax, false),
        (ScatterReduce::Amin, true),
        (ScatterReduce::Amin, false),
    ];

    for (reduce, include_self) in cases {
        let expected_f16 = torch_scatter_reduce_bits("f16", reduce, include_self, false);
        let input_f16 = cuda_f16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let src_f16 = cuda_f16(&[10.0, 20.0, 40.0, 50.0], &[2, 2], false);
        let out_f16 = scatter_reduce(
            &input_f16,
            0,
            &index,
            &index_shape,
            &src_f16,
            reduce,
            include_self,
        )
        .expect("f16 scatter_reduce");
        assert!(
            out_f16.is_cuda(),
            "f16 {reduce:?} include_self={include_self} must stay CUDA"
        );
        assert_eq!(host_f16_bits(&out_f16), expected_f16.out);

        let expected_bf16 = torch_scatter_reduce_bits("bf16", reduce, include_self, false);
        let input_bf16 = cuda_bf16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let src_bf16 = cuda_bf16(&[10.0, 20.0, 40.0, 50.0], &[2, 2], false);
        let out_bf16 = scatter_reduce(
            &input_bf16,
            0,
            &index,
            &index_shape,
            &src_bf16,
            reduce,
            include_self,
        )
        .expect("bf16 scatter_reduce");
        assert!(
            out_bf16.is_cuda(),
            "bf16 {reduce:?} include_self={include_self} must stay CUDA"
        );
        assert_eq!(host_bf16_bits(&out_bf16), expected_bf16.out);
    }
}

#[test]
fn cuda_scatter_reduce_f16_bf16_sum_backward_matches_torch_bits_and_stays_resident() {
    ensure_cuda();
    let index = [0usize, 1, 1, 0];
    let index_shape = [2, 2];

    let expected_f16 = torch_scatter_reduce_bits("f16", ScatterReduce::Sum, false, true);
    let input_f16 = cuda_f16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let src_f16 = cuda_f16(&[10.0, 20.0, 40.0, 50.0], &[2, 2], true);
    let out_f16 = scatter_reduce(
        &input_f16,
        0,
        &index,
        &index_shape,
        &src_f16,
        ScatterReduce::Sum,
        false,
    )
    .expect("f16 scatter_reduce sum");
    out_f16
        .sum_all()
        .expect("f16 sum")
        .backward()
        .expect("f16 scatter_reduce backward");
    let gi_f16 = input_f16.grad().unwrap().expect("f16 input grad");
    let gs_f16 = src_f16.grad().unwrap().expect("f16 src grad");
    assert!(gi_f16.is_cuda() && gs_f16.is_cuda());
    assert_eq!(host_f16_bits(&out_f16), expected_f16.out);
    assert_eq!(
        host_f16_bits(&gi_f16),
        expected_f16.grad_input.expect("torch f16 grad input")
    );
    assert_eq!(
        host_f16_bits(&gs_f16),
        expected_f16.grad_src.expect("torch f16 grad src")
    );

    let expected_bf16 = torch_scatter_reduce_bits("bf16", ScatterReduce::Sum, false, true);
    let input_bf16 = cuda_bf16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let src_bf16 = cuda_bf16(&[10.0, 20.0, 40.0, 50.0], &[2, 2], true);
    let out_bf16 = scatter_reduce(
        &input_bf16,
        0,
        &index,
        &index_shape,
        &src_bf16,
        ScatterReduce::Sum,
        false,
    )
    .expect("bf16 scatter_reduce sum");
    out_bf16
        .sum_all()
        .expect("bf16 sum")
        .backward()
        .expect("bf16 scatter_reduce backward");
    let gi_bf16 = input_bf16.grad().unwrap().expect("bf16 input grad");
    let gs_bf16 = src_bf16.grad().unwrap().expect("bf16 src grad");
    assert!(gi_bf16.is_cuda() && gs_bf16.is_cuda());
    assert_eq!(host_bf16_bits(&out_bf16), expected_bf16.out);
    assert_eq!(
        host_bf16_bits(&gi_bf16),
        expected_bf16.grad_input.expect("torch bf16 grad input")
    );
    assert_eq!(
        host_bf16_bits(&gs_bf16),
        expected_bf16.grad_src.expect("torch bf16 grad src")
    );
}

#[test]
fn cuda_scatter_reduce_include_self_false_keeps_untouched_slots_f64() {
    ensure_cuda();
    let input = cuda_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let src = cuda_f64(&[10.0, 20.0], &[1, 2]);
    let index = [0, 0];
    let index_shape = [1, 2];

    for reduce in [
        ScatterReduce::Sum,
        ScatterReduce::Prod,
        ScatterReduce::Amax,
        ScatterReduce::Amin,
    ] {
        let out = scatter_reduce(&input, 0, &index, &index_shape, &src, reduce, false).unwrap();
        assert!(out.is_cuda(), "{reduce:?} output must stay CUDA");
        assert_eq!(host_f64(&out), vec![10.0, 20.0, 3.0, 4.0]);
    }
}

#[test]
fn cuda_scatter_reduce_nan_extrema_match_torch_ordered_comparison() {
    ensure_cuda();
    let input = cuda_f32(&[1.0, f32::NAN, 3.0], &[3]);
    let src = cuda_f32(&[2.0, 4.0], &[2]);
    let index = [1, 1];
    let index_shape = [2];

    for reduce in [ScatterReduce::Amax, ScatterReduce::Amin] {
        let out = scatter_reduce(&input, 0, &index, &index_shape, &src, reduce, true).unwrap();
        let got = host_f32(&out);
        assert_eq!(got[0], 1.0);
        assert!(
            got[1].is_nan(),
            "{reduce:?} must keep self NaN at touched slot"
        );
        assert_eq!(got[2], 3.0);
    }
}
