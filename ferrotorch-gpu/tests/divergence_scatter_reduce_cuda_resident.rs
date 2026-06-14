#![cfg(feature = "cuda")]

//! CUDA `scatter_reduce` must not run as a CPU fold plus re-upload.
//! These cases mirror live torch 2.11.0+cu130 for larger `src`, duplicate
//! destinations, `include_self=false`, and all shipped reduce modes.

use ferrotorch_core::grad_fns::indexing::{ScatterReduce, scatter_reduce};
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;
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

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().unwrap().data_vec().unwrap()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().unwrap().data_vec().unwrap()
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
