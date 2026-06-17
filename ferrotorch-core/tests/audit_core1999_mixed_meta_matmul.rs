//! CORE-1999: mixed meta/real `matmul` follows PyTorch's composite
//! decomposition instead of blanket-rejecting every mixed-device pair.
//!
//! PyTorch source anchors:
//! - `/home/doll/pytorch/aten/src/ATen/native/LinearAlgebra.cpp::_matmul_impl`
//! - `/home/doll/pytorch/aten/src/ATen/native/LinearAlgebra.cpp::should_fold`
//!
//! Live PyTorch 2.11 oracle used here:
//! - dot and mv mixed meta/real pairs reject.
//! - mm output uses the left operand's options.
//! - bmm output uses the right operand's options.
//! - `should_fold` can turn `ND @ 1D` into mv (mixed pairs reject), and a
//!   grad-tracked small operand forces that fold.
//! - the 3D/3D broadcast-autograd optimization recursively squeezes a
//!   grad-tracked batch-1 operand, changing which lower-level op owns the
//!   output device.

use ferrotorch_core::creation::{zeros, zeros_meta};
use ferrotorch_core::device::Device;
use ferrotorch_core::ops::linalg::matmul as ops_matmul;
use ferrotorch_core::tensor::Tensor;

fn cpu(shape: &[usize]) -> Tensor<f32> {
    zeros(shape).expect("cpu zeros")
}

fn meta(shape: &[usize]) -> Tensor<f32> {
    zeros_meta(shape).expect("meta zeros")
}

fn tensor(shape: &[usize], device: Device, requires_grad: bool) -> Tensor<f32> {
    let t = match device {
        Device::Cpu => cpu(shape),
        Device::Meta => meta(shape),
        other => panic!("unsupported test device {other}"),
    };
    t.requires_grad_(requires_grad)
}

struct ForwardCase<'a> {
    label: String,
    a_shape: &'a [usize],
    b_shape: &'a [usize],
    a_device: Device,
    b_device: Device,
    a_requires_grad: bool,
    b_requires_grad: bool,
    expected_shape: &'a [usize],
    expected_device: Device,
}

struct RejectCase<'a> {
    label: String,
    a_shape: &'a [usize],
    b_shape: &'a [usize],
    a_device: Device,
    b_device: Device,
    a_requires_grad: bool,
    b_requires_grad: bool,
}

fn assert_forward(case: &ForwardCase<'_>) {
    let raw_a = tensor(case.a_shape, case.a_device, case.a_requires_grad);
    let raw_b = tensor(case.b_shape, case.b_device, case.b_requires_grad);
    let raw = ops_matmul(&raw_a, &raw_b)
        .unwrap_or_else(|err| panic!("{}: raw matmul errored: {err}", case.label));
    assert_eq!(
        raw.shape(),
        case.expected_shape,
        "{}: raw shape",
        case.label
    );
    assert_eq!(
        raw.device(),
        case.expected_device,
        "{}: raw device",
        case.label
    );
    assert!(
        !raw.requires_grad(),
        "{}: raw ops::linalg::matmul must not attach autograd",
        case.label
    );

    let tracked_a = tensor(case.a_shape, case.a_device, case.a_requires_grad);
    let tracked_b = tensor(case.b_shape, case.b_device, case.b_requires_grad);
    let tracked = tracked_a
        .matmul(&tracked_b)
        .unwrap_or_else(|err| panic!("{}: Tensor::matmul errored: {err}", case.label));
    assert_eq!(
        tracked.shape(),
        case.expected_shape,
        "{}: tracked shape",
        case.label
    );
    assert_eq!(
        tracked.device(),
        case.expected_device,
        "{}: tracked device",
        case.label
    );
    assert_eq!(
        tracked.requires_grad(),
        case.a_requires_grad || case.b_requires_grad,
        "{}: tracked requires_grad",
        case.label
    );
    assert_eq!(
        tracked.grad_fn().is_some(),
        case.a_requires_grad || case.b_requires_grad,
        "{}: tracked grad_fn",
        case.label
    );
}

fn assert_mixed_pair(
    label: &str,
    a_shape: &[usize],
    b_shape: &[usize],
    expected_shape: &[usize],
    meta_cpu_device: Device,
    cpu_meta_device: Device,
) {
    assert_forward(&ForwardCase {
        label: format!("{label} meta@cpu"),
        a_shape,
        b_shape,
        a_device: Device::Meta,
        b_device: Device::Cpu,
        a_requires_grad: false,
        b_requires_grad: false,
        expected_shape,
        expected_device: meta_cpu_device,
    });
    assert_forward(&ForwardCase {
        label: format!("{label} cpu@meta"),
        a_shape,
        b_shape,
        a_device: Device::Cpu,
        b_device: Device::Meta,
        a_requires_grad: false,
        b_requires_grad: false,
        expected_shape,
        expected_device: cpu_meta_device,
    });
}

fn assert_rejects(case: &RejectCase<'_>) {
    let raw_a = tensor(case.a_shape, case.a_device, case.a_requires_grad);
    let raw_b = tensor(case.b_shape, case.b_device, case.b_requires_grad);
    assert!(
        ops_matmul(&raw_a, &raw_b).is_err(),
        "{}: raw matmul must reject",
        case.label
    );

    let tracked_a = tensor(case.a_shape, case.a_device, case.a_requires_grad);
    let tracked_b = tensor(case.b_shape, case.b_device, case.b_requires_grad);
    assert!(
        tracked_a.matmul(&tracked_b).is_err(),
        "{}: Tensor::matmul must reject",
        case.label
    );
}

#[test]
fn cpu_meta_success_cases_match_pytorch_output_device_owner() {
    assert_mixed_pair(
        "1d x 2d uses mm left options",
        &[5],
        &[5, 4],
        &[4],
        Device::Meta,
        Device::Cpu,
    );
    assert_mixed_pair(
        "2d x 2d uses mm left options",
        &[3, 5],
        &[5, 4],
        &[3, 4],
        Device::Meta,
        Device::Cpu,
    );
    assert_mixed_pair(
        "3d x 2d folded mm uses left options",
        &[2, 3, 5],
        &[5, 4],
        &[2, 3, 4],
        Device::Meta,
        Device::Cpu,
    );
    assert_mixed_pair(
        "2d x 3d folded mm uses right options",
        &[3, 5],
        &[2, 5, 4],
        &[2, 3, 4],
        Device::Cpu,
        Device::Meta,
    );
    assert_mixed_pair(
        "1d x 3d expanded bmm uses right options",
        &[5],
        &[2, 5, 4],
        &[2, 4],
        Device::Cpu,
        Device::Meta,
    );
    assert_mixed_pair(
        "3d x 3d expanded bmm uses right options",
        &[2, 3, 5],
        &[2, 5, 4],
        &[2, 3, 4],
        Device::Cpu,
        Device::Meta,
    );
    assert_mixed_pair(
        "broadcast bmm uses right options",
        &[2, 1, 3, 5],
        &[1, 4, 5, 6],
        &[2, 4, 3, 6],
        Device::Cpu,
        Device::Meta,
    );
    assert_mixed_pair(
        "zero batch folded mm uses left options",
        &[0, 3, 5],
        &[5, 4],
        &[0, 3, 4],
        Device::Meta,
        Device::Cpu,
    );
    assert_mixed_pair(
        "zero contraction 2d x 2d uses left options",
        &[3, 0],
        &[0, 4],
        &[3, 4],
        Device::Meta,
        Device::Cpu,
    );
}

#[test]
fn cpu_meta_mixed_dot_mv_and_folded_mv_reject_like_pytorch() {
    for (label, a_shape, b_shape) in [
        ("dot", &[5][..], &[5][..]),
        ("mv", &[3, 5][..], &[5][..]),
        ("3d x 1d folded mv", &[2, 3, 5][..], &[5][..]),
        ("1d x 3d folded mv after mT", &[1][..], &[2, 1, 4][..]),
    ] {
        assert_rejects(&RejectCase {
            label: format!("{label} meta@cpu"),
            a_shape,
            b_shape,
            a_device: Device::Meta,
            b_device: Device::Cpu,
            a_requires_grad: false,
            b_requires_grad: false,
        });
        assert_rejects(&RejectCase {
            label: format!("{label} cpu@meta"),
            a_shape,
            b_shape,
            a_device: Device::Cpu,
            b_device: Device::Meta,
            a_requires_grad: false,
            b_requires_grad: false,
        });
    }
}

#[test]
fn cpu_meta_requires_grad_small_vector_forces_folded_mv_rejection() {
    assert_rejects(&RejectCase {
        label: "meta 1d requires_grad forces fold before 1d x 3d".into(),
        a_shape: &[5],
        b_shape: &[2, 5, 4],
        a_device: Device::Meta,
        b_device: Device::Cpu,
        a_requires_grad: true,
        b_requires_grad: false,
    });
    assert_rejects(&RejectCase {
        label: "cpu 1d requires_grad forces fold before 1d x 3d".into(),
        a_shape: &[5],
        b_shape: &[2, 5, 4],
        a_device: Device::Cpu,
        b_device: Device::Meta,
        a_requires_grad: true,
        b_requires_grad: false,
    });

    assert_forward(&ForwardCase {
        label: "rhs requires_grad keeps expanded bmm meta@cpu".into(),
        a_shape: &[5],
        b_shape: &[2, 5, 4],
        a_device: Device::Meta,
        b_device: Device::Cpu,
        a_requires_grad: false,
        b_requires_grad: true,
        expected_shape: &[2, 4],
        expected_device: Device::Cpu,
    });
    assert_forward(&ForwardCase {
        label: "rhs requires_grad keeps expanded bmm cpu@meta".into(),
        a_shape: &[5],
        b_shape: &[2, 5, 4],
        a_device: Device::Cpu,
        b_device: Device::Meta,
        a_requires_grad: false,
        b_requires_grad: true,
        expected_shape: &[2, 4],
        expected_device: Device::Meta,
    });
}

#[test]
fn cpu_meta_rank3_broadcast_requires_grad_matches_recursive_pytorch_path() {
    assert_forward(&ForwardCase {
        label: "left broadcast batch requires_grad recurses to 2d x 3d meta@cpu".into(),
        a_shape: &[1, 3, 5],
        b_shape: &[2, 5, 4],
        a_device: Device::Meta,
        b_device: Device::Cpu,
        a_requires_grad: true,
        b_requires_grad: false,
        expected_shape: &[2, 3, 4],
        expected_device: Device::Cpu,
    });
    assert_forward(&ForwardCase {
        label: "left broadcast batch requires_grad recurses to 2d x 3d cpu@meta".into(),
        a_shape: &[1, 3, 5],
        b_shape: &[2, 5, 4],
        a_device: Device::Cpu,
        b_device: Device::Meta,
        a_requires_grad: true,
        b_requires_grad: false,
        expected_shape: &[2, 3, 4],
        expected_device: Device::Meta,
    });
    assert_forward(&ForwardCase {
        label: "right broadcast batch requires_grad recurses to 3d x 2d meta@cpu".into(),
        a_shape: &[2, 3, 5],
        b_shape: &[1, 5, 4],
        a_device: Device::Meta,
        b_device: Device::Cpu,
        a_requires_grad: false,
        b_requires_grad: true,
        expected_shape: &[2, 3, 4],
        expected_device: Device::Meta,
    });
    assert_forward(&ForwardCase {
        label: "right broadcast batch requires_grad recurses to 3d x 2d cpu@meta".into(),
        a_shape: &[2, 3, 5],
        b_shape: &[1, 5, 4],
        a_device: Device::Cpu,
        b_device: Device::Meta,
        a_requires_grad: false,
        b_requires_grad: true,
        expected_shape: &[2, 3, 4],
        expected_device: Device::Cpu,
    });
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::*;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-1999 CUDA/meta parity tests");
        });
    }

    fn cuda(shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        ensure_cuda_backend();
        cpu(shape)
            .to(Device::Cuda(0))
            .expect("upload to cuda")
            .requires_grad_(requires_grad)
    }

    fn tensor_cuda_or_meta(shape: &[usize], device: Device, requires_grad: bool) -> Tensor<f32> {
        match device {
            Device::Cuda(0) => cuda(shape, requires_grad),
            Device::Meta => meta(shape).requires_grad_(requires_grad),
            other => panic!("unsupported CUDA test device {other}"),
        }
    }

    fn assert_cuda_forward(case: &ForwardCase<'_>) {
        let raw_a = tensor_cuda_or_meta(case.a_shape, case.a_device, case.a_requires_grad);
        let raw_b = tensor_cuda_or_meta(case.b_shape, case.b_device, case.b_requires_grad);
        let raw = ops_matmul(&raw_a, &raw_b)
            .unwrap_or_else(|err| panic!("{}: raw matmul errored: {err}", case.label));
        assert_eq!(
            raw.shape(),
            case.expected_shape,
            "{}: raw shape",
            case.label
        );
        assert_eq!(
            raw.device(),
            case.expected_device,
            "{}: raw device",
            case.label
        );
        if case.expected_device.is_cuda() {
            assert!(
                raw.gpu_handle().is_ok(),
                "{}: CUDA output must remain resident",
                case.label
            );
        }

        let tracked_a = tensor_cuda_or_meta(case.a_shape, case.a_device, case.a_requires_grad);
        let tracked_b = tensor_cuda_or_meta(case.b_shape, case.b_device, case.b_requires_grad);
        let tracked = tracked_a
            .matmul(&tracked_b)
            .unwrap_or_else(|err| panic!("{}: Tensor::matmul errored: {err}", case.label));
        assert_eq!(
            tracked.shape(),
            case.expected_shape,
            "{}: tracked shape",
            case.label
        );
        assert_eq!(
            tracked.device(),
            case.expected_device,
            "{}: tracked device",
            case.label
        );
        if case.expected_device.is_cuda() {
            assert!(
                tracked.gpu_handle().is_ok(),
                "{}: tracked CUDA output must remain resident",
                case.label
            );
        }
    }

    fn assert_cuda_rejects(case: &RejectCase<'_>) {
        let raw_a = tensor_cuda_or_meta(case.a_shape, case.a_device, case.a_requires_grad);
        let raw_b = tensor_cuda_or_meta(case.b_shape, case.b_device, case.b_requires_grad);
        assert!(
            ops_matmul(&raw_a, &raw_b).is_err(),
            "{}: raw matmul must reject",
            case.label
        );

        let tracked_a = tensor_cuda_or_meta(case.a_shape, case.a_device, case.a_requires_grad);
        let tracked_b = tensor_cuda_or_meta(case.b_shape, case.b_device, case.b_requires_grad);
        assert!(
            tracked_a.matmul(&tracked_b).is_err(),
            "{}: Tensor::matmul must reject",
            case.label
        );
    }

    #[test]
    fn cuda_meta_success_cases_match_pytorch_output_device_owner() {
        for case in [
            ForwardCase {
                label: "meta@cuda mm left options".into(),
                a_shape: &[3, 5],
                b_shape: &[5, 4],
                a_device: Device::Meta,
                b_device: Device::Cuda(0),
                a_requires_grad: false,
                b_requires_grad: false,
                expected_shape: &[3, 4],
                expected_device: Device::Meta,
            },
            ForwardCase {
                label: "cuda@meta mm left options".into(),
                a_shape: &[3, 5],
                b_shape: &[5, 4],
                a_device: Device::Cuda(0),
                b_device: Device::Meta,
                a_requires_grad: false,
                b_requires_grad: false,
                expected_shape: &[3, 4],
                expected_device: Device::Cuda(0),
            },
            ForwardCase {
                label: "meta@cuda 2d x 3d uses right options".into(),
                a_shape: &[3, 5],
                b_shape: &[2, 5, 4],
                a_device: Device::Meta,
                b_device: Device::Cuda(0),
                a_requires_grad: false,
                b_requires_grad: false,
                expected_shape: &[2, 3, 4],
                expected_device: Device::Cuda(0),
            },
            ForwardCase {
                label: "cuda@meta 2d x 3d uses right options".into(),
                a_shape: &[3, 5],
                b_shape: &[2, 5, 4],
                a_device: Device::Cuda(0),
                b_device: Device::Meta,
                a_requires_grad: false,
                b_requires_grad: false,
                expected_shape: &[2, 3, 4],
                expected_device: Device::Meta,
            },
            ForwardCase {
                label: "meta@cuda 3d x 3d uses right options".into(),
                a_shape: &[2, 3, 5],
                b_shape: &[2, 5, 4],
                a_device: Device::Meta,
                b_device: Device::Cuda(0),
                a_requires_grad: false,
                b_requires_grad: false,
                expected_shape: &[2, 3, 4],
                expected_device: Device::Cuda(0),
            },
            ForwardCase {
                label: "cuda@meta 3d x 3d uses right options".into(),
                a_shape: &[2, 3, 5],
                b_shape: &[2, 5, 4],
                a_device: Device::Cuda(0),
                b_device: Device::Meta,
                a_requires_grad: false,
                b_requires_grad: false,
                expected_shape: &[2, 3, 4],
                expected_device: Device::Meta,
            },
            ForwardCase {
                label: "meta@cuda right broadcast batch requires_grad uses left options".into(),
                a_shape: &[2, 3, 5],
                b_shape: &[1, 5, 4],
                a_device: Device::Meta,
                b_device: Device::Cuda(0),
                a_requires_grad: false,
                b_requires_grad: true,
                expected_shape: &[2, 3, 4],
                expected_device: Device::Meta,
            },
            ForwardCase {
                label: "cuda@meta right broadcast batch requires_grad uses left options".into(),
                a_shape: &[2, 3, 5],
                b_shape: &[1, 5, 4],
                a_device: Device::Cuda(0),
                b_device: Device::Meta,
                a_requires_grad: false,
                b_requires_grad: true,
                expected_shape: &[2, 3, 4],
                expected_device: Device::Cuda(0),
            },
        ] {
            assert_cuda_forward(&case);
        }
    }

    #[test]
    fn cuda_meta_folded_mv_rejects_like_pytorch() {
        for case in [
            RejectCase {
                label: "meta@cuda mv".into(),
                a_shape: &[3, 5],
                b_shape: &[5],
                a_device: Device::Meta,
                b_device: Device::Cuda(0),
                a_requires_grad: false,
                b_requires_grad: false,
            },
            RejectCase {
                label: "cuda@meta mv".into(),
                a_shape: &[3, 5],
                b_shape: &[5],
                a_device: Device::Cuda(0),
                b_device: Device::Meta,
                a_requires_grad: false,
                b_requires_grad: false,
            },
            RejectCase {
                label: "meta 1d requires_grad forces fold before 1d x cuda 3d".into(),
                a_shape: &[5],
                b_shape: &[2, 5, 4],
                a_device: Device::Meta,
                b_device: Device::Cuda(0),
                a_requires_grad: true,
                b_requires_grad: false,
            },
            RejectCase {
                label: "cuda 1d requires_grad forces fold before 1d x meta 3d".into(),
                a_shape: &[5],
                b_shape: &[2, 5, 4],
                a_device: Device::Cuda(0),
                b_device: Device::Meta,
                a_requires_grad: true,
                b_requires_grad: false,
            },
        ] {
            assert_cuda_rejects(&case);
        }
    }
}
