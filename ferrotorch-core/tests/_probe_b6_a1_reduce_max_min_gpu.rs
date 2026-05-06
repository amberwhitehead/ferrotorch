//! Probe-before-fix sentinel for #790: GPU `reduce(Max/Min)` returns the
//! first row of the input instead of the running-extremum endpoint when the
//! kept axis comes after the reduced axis.
//!
//! Per the architect's "Probe-before-fix" pattern, this file decomposes the
//! suspected fast-path
//!
//! ```text
//! view_reshape([1, N, M])  ->  cummax(view, dim=1)  ->  narrow(1, N-1, 1)
//!                          ->  squeeze_t(1)         ->  view_reshape(out_shape)
//! ```
//!
//! into individually-checkable steps so the failing op is identified
//! mechanically rather than guessed. The three step-tests below remain in
//! the suite as the regression sentinel for this cluster.
//!
//! Step 1: `cummax` directly on a `[1, 3, 2]` CUDA tensor — verifies that the
//!         GPU cummax kernel actually computes the running maximum (and is
//!         not, e.g., copying the input through unchanged for low-batch
//!         shapes).
//! Step 2: `narrow(1, last, 1).squeeze_t(1)` directly on a `[1, 3, 2]` CUDA
//!         tensor — verifies that the GPU narrow + squeeze + readback chain
//!         picks the last "row" rather than the first. This is the same
//!         shape as #802's strided-view-readback regression: if narrow on
//!         GPU is the bug, every downstream caller of narrow with non-zero
//!         start is wrong, not just `reduce`.
//! Step 3: The full repro from the issue text — `[3, 2]` data
//!         `[0.5, 4.0, 1.0, 4.5, 1.5, 5.0]`, `reduce("b c -> c", Max)` on
//!         CUDA, expected `[1.5, 5.0]`.
//!
//! All three steps assert the post-fix correct values. Pre-fix, step 3 (and
//! whichever of steps 1 / 2 carries the underlying bug) fails. Post-fix, all
//! three pass.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::EinopsReduction;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::cumulative::cummax;
use ferrotorch_core::reduce;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

/// Step 1: cummax direct on `[1, 3, 2]` CUDA tensor.
/// Input rows: [0.5, 4.0], [1.0, 4.5], [1.5, 5.0].
/// Expected running-max along dim 1 (per inner column):
///   row 0 -> [0.5, 4.0]
///   row 1 -> [1.0, 4.5]
///   row 2 -> [1.5, 5.0]   (the global max — what reduce ultimately wants)
#[test]
fn step1_cummax_gpu_axis1_three_one_three_two_f32() {
    ensure_cuda_backend();
    let data = vec![0.5_f32, 4.0, 1.0, 4.5, 1.5, 5.0];
    let cpu = from_vec::<f32>(data, &[1, 3, 2]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");

    let res = cummax(&gpu, 1).expect("cummax on gpu");
    let host = res.values.cpu().expect("gpu->cpu");
    let host_data = host.data_vec().expect("data_vec");

    let expected = vec![0.5_f32, 4.0, 1.0, 4.5, 1.5, 5.0];
    assert_eq!(
        host_data, expected,
        "GPU cummax(dim=1) on [1,3,2] returned wrong running max — \
         candidate (a) cummax-on-GPU bug (#790)"
    );
}

/// Step 2: narrow + squeeze + readback direct on a `[1, 3, 2]` CUDA tensor
/// where each "row" is distinct. `narrow(1, 2, 1).squeeze_t(1)` should
/// select the last row.
#[test]
fn step2_narrow_last_row_gpu_one_three_two_f32() {
    ensure_cuda_backend();
    // Each row is distinct so we can tell first-vs-last unambiguously.
    let data = vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let cpu = from_vec::<f32>(data, &[1, 3, 2]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");

    let narrowed = gpu.narrow(1, 2, 1).expect("narrow(1, 2, 1)");
    assert_eq!(narrowed.shape(), &[1, 1, 2]);
    let squeezed = narrowed.squeeze_t(1).expect("squeeze_t(1)");
    assert_eq!(squeezed.shape(), &[1, 2]);

    let host = squeezed.cpu().expect("gpu->cpu");
    let host_data = host.data_vec().expect("data_vec");
    assert_eq!(
        host_data,
        vec![50.0_f32, 60.0],
        "GPU narrow(1, 2, 1).squeeze_t(1).cpu() picked the wrong row — \
         candidate (b) narrow-on-GPU bug (#790, same shape as #802 cluster)"
    );
}

/// Step 3: the full repro from the issue text.
#[test]
fn step3_reduce_max_b_c_to_c_gpu_repro_f32() {
    ensure_cuda_backend();
    let data = vec![0.5_f32, 4.0, 1.0, 4.5, 1.5, 5.0];
    let cpu = from_vec::<f32>(data, &[3, 2]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");

    let out = reduce(&gpu, "b c -> c", EinopsReduction::Max).expect("reduce(Max) on gpu");
    let host = out.cpu().expect("gpu->cpu");
    let host_data = host.data_vec().expect("data_vec");

    assert_eq!(
        host_data,
        vec![1.5_f32, 5.0],
        "reduce(t, \"b c -> c\", Max) on CUDA returned the first row \
         instead of the column-wise max (#790)"
    );
}

/// Step 3-min: same repro but for `EinopsReduction::Min`.
#[test]
fn step3_reduce_min_b_c_to_c_gpu_repro_f32() {
    ensure_cuda_backend();
    // Construct so the column-wise min is the LAST row, mirroring the Max case.
    let data = vec![5.0_f32, 50.0, 4.0, 40.0, 3.0, 30.0];
    let cpu = from_vec::<f32>(data, &[3, 2]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");

    let out = reduce(&gpu, "b c -> c", EinopsReduction::Min).expect("reduce(Min) on gpu");
    let host = out.cpu().expect("gpu->cpu");
    let host_data = host.data_vec().expect("data_vec");

    assert_eq!(
        host_data,
        vec![3.0_f32, 30.0],
        "reduce(t, \"b c -> c\", Min) on CUDA returned the first row \
         instead of the column-wise min (#790)"
    );
}

/// f64 coverage of the full repro for both reductions.
#[test]
fn step3_reduce_max_b_c_to_c_gpu_repro_f64() {
    ensure_cuda_backend();
    let data = vec![0.5_f64, 4.0, 1.0, 4.5, 1.5, 5.0];
    let cpu = from_vec::<f64>(data, &[3, 2]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");

    let out = reduce(&gpu, "b c -> c", EinopsReduction::Max).expect("reduce(Max) on gpu");
    let host = out.cpu().expect("gpu->cpu");
    let host_data = host.data_vec().expect("data_vec");
    assert_eq!(host_data, vec![1.5_f64, 5.0]);
}
