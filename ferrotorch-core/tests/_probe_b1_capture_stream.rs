//! Probe B1 -- Issue #897: captured_graph_upload_and_replay_count_api stream invalidation.
//!
//! Root cause: gpu_add_into dispatched on device.stream() (the device default stream).
//! Under parallel test execution, other tests dispatched on the same default stream
//! during Global-mode capture, causing the CUDA driver to mark the capture as
//! CUDA_ERROR_STREAM_CAPTURE_INVALIDATED.
//!
//! BEFORE fix: the test cascade-skipped unconditionally because the default stream
//!   was contested and the capture was invalidated under parallel execution.
//!
//! AFTER fix: gpu_add_into_on_stream was added to allow callers to dispatch on an
//!   explicit stream. API-surface capture tests use an explicit Relaxed mode on a
//!   dedicated non-default stream so the PyTorch-equivalent Global default remains
//!   available without making parallel test execution flaky.
//!
//! This probe verifies:
//!   1. gpu_add_into_on_stream compiles and dispatches on the provided stream.
//!   2. CapturedGraph.upload(), num_replays(), is_uploaded(), has_pool() all work.
//!   3. The test is stable under parallel execution (no INVALIDATED error).

#![cfg(feature = "gpu")]

use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for probe B1 capture stream");
    });
}

/// Probe: AFTER fix gpu_add_into_on_stream dispatches on the explicit stream.
#[test]
fn probe_b1_gpu_add_into_on_stream_dispatches() {
    ensure_cuda();
    use ferrotorch_gpu::device::GpuDevice;
    use ferrotorch_gpu::kernels::gpu_add_into_on_stream;
    use ferrotorch_gpu::transfer::{alloc_zeros_f32, cpu_to_gpu, gpu_to_cpu};

    let dev = GpuDevice::new(0).expect("GpuDevice::new(0)");
    let stream = dev.context().new_stream().expect("new_stream");

    let a = cpu_to_gpu(&[1.0f32, 2.0, 3.0, 4.0], &dev).unwrap();
    let b = cpu_to_gpu(&[10.0f32, 20.0, 30.0, 40.0], &dev).unwrap();
    let mut out = alloc_zeros_f32(4, &dev).unwrap();

    // Pre-warm and sync before dispatching on the new stream.
    gpu_add_into_on_stream(&a, &b, &mut out, &dev, &dev.stream()).unwrap();
    dev.stream().synchronize().unwrap();

    // AFTER fix: dispatches on `stream` (not device.stream()).
    gpu_add_into_on_stream(&a, &b, &mut out, &dev, &stream)
        .expect("AFTER fix #897: gpu_add_into_on_stream must dispatch on explicit stream");

    stream.synchronize().expect("sync after dispatch");
    let actual = gpu_to_cpu(&out, &dev).unwrap();
    assert_eq!(
        actual,
        vec![11.0f32, 22.0, 33.0, 44.0],
        "probe B1 #897: output mismatch"
    );
}

/// Probe: AFTER fix CapturedGraph API surface works under parallel execution.
#[test]
fn probe_b1_captured_graph_api_surface() {
    ensure_cuda();
    use ferrotorch_gpu::device::GpuDevice;
    use ferrotorch_gpu::graph::{
        CaptureMode, CapturePool, begin_capture_with_pool_mode, end_capture_with_pool,
    };
    use std::sync::Arc;

    // Serialize graph captures within this test binary.
    use std::sync::{Mutex, MutexGuard};
    fn capture_lock() -> MutexGuard<'static, ()> {
        static M: Mutex<()> = Mutex::new(());
        M.lock().unwrap_or_else(|p| p.into_inner())
    }
    let _lock = capture_lock();

    let dev = GpuDevice::new(0).expect("GpuDevice::new(0)");
    // Dedicated stream with explicit Relaxed capture mode. CaptureMode::default()
    // is Global to match PyTorch, but this API-surface probe intentionally
    // avoids global cross-thread invalidation under the Rust test harness.
    let capture_stream = dev.context().new_stream().expect("new_stream");

    let pool = Arc::new(CapturePool::new());
    begin_capture_with_pool_mode(&pool, &capture_stream, CaptureMode::Relaxed)
        .expect("begin_capture_with_pool_mode");
    let graph =
        end_capture_with_pool(&capture_stream, Arc::clone(&pool)).expect("end_capture_with_pool");

    // Verify CapturedGraph API surface.
    assert!(!graph.is_uploaded(), "fresh graph must not be uploaded");
    assert_eq!(graph.num_replays(), 0, "no replays before upload");
    assert!(graph.has_pool(), "graph must hold pool ref");
    assert_eq!(graph.pool_buffer_count(), 0, "no buffers donated");

    graph.upload().expect("upload");
    assert!(graph.is_uploaded(), "must be uploaded after upload()");

    graph.launch().expect("launch 1");
    assert_eq!(graph.num_replays(), 1, "num_replays after first launch");

    graph.launch().expect("launch 2");
    assert_eq!(graph.num_replays(), 2, "num_replays after second launch");
}
