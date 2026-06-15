//! Regression tests for CORE-003 / crosslink #1697.
//!
//! PyTorch treats floating dtype conversion as differentiable:
//!
//! ```text
//! >>> x = torch.tensor([1., 2., 3.], dtype=torch.float32, requires_grad=True)
//! >>> y = x.to(torch.float64)
//! >>> y.requires_grad, y.is_leaf, type(y.grad_fn).__name__
//! (True, False, 'ToCopyBackward0')
//! >>> (y * y).sum().backward()
//! >>> x.grad, x.grad.dtype
//! (tensor([2., 4., 6.]), torch.float32)
//! ```

use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;

#[test]
fn cpu_to_dtype_leaf_backward_reaches_source() {
    let x =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]), vec![3], true).unwrap();
    let y = x.to_dtype::<f64>().unwrap();
    assert!(y.requires_grad(), "torch: cast output tracks gradients");
    assert!(
        !y.is_leaf(),
        "torch: cast output is a non-leaf ToCopyBackward0"
    );

    sum(&mul(&y, &y).expect("mul cast output"))
        .expect("sum cast output")
        .backward()
        .expect("backward through dtype cast");

    let grad = x
        .grad()
        .expect("grad lookup")
        .expect("cast backward must reach the original f32 leaf");
    assert_eq!(grad.data().expect("grad data"), &[2.0, 4.0, 6.0]);
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use ferrotorch_core::dtype::Float;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-003 gpu lane");
        });
    }

    fn cuda_tensor<T: Float>(values: Vec<T>) -> Tensor<T> {
        Tensor::from_storage(
            TensorStorage::on_device(values, Device::Cuda(0)).expect("upload cuda tensor"),
            vec![3],
            true,
        )
        .expect("cuda tensor")
    }

    fn assert_f32_values(t: &Tensor<f32>, expected: &[f32]) {
        assert_eq!(t.device(), Device::Cuda(0));
        let cpu = t.to(Device::Cpu).expect("download f32");
        assert_eq!(cpu.data().expect("f32 data"), expected);
    }

    fn assert_f64_values(t: &Tensor<f64>, expected: &[f64]) {
        assert_eq!(t.device(), Device::Cuda(0));
        let cpu = t.to(Device::Cpu).expect("download f64");
        assert_eq!(cpu.data().expect("f64 data"), expected);
    }

    fn assert_f16_values(t: &Tensor<half::f16>, expected: &[f32]) {
        assert_eq!(t.device(), Device::Cuda(0));
        let cpu = t.to(Device::Cpu).expect("download f16");
        let got: Vec<f32> = cpu
            .data()
            .expect("f16 data")
            .iter()
            .map(|v| v.to_f32())
            .collect();
        assert_eq!(got, expected);
    }

    fn assert_bf16_values(t: &Tensor<half::bf16>, expected: &[f32]) {
        assert_eq!(t.device(), Device::Cuda(0));
        let cpu = t.to(Device::Cpu).expect("download bf16");
        let got: Vec<f32> = cpu
            .data()
            .expect("bf16 data")
            .iter()
            .map(|v| v.to_f32())
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn cuda_f32_to_bf16_backward_reaches_cuda_leaf() {
        ensure_cuda_backend();
        let storage = TensorStorage::on_device(vec![1.0f32, 2.0, 3.0], Device::Cuda(0)).unwrap();
        let x = Tensor::from_storage(storage, vec![3], true).unwrap();
        let y = x.to_dtype::<half::bf16>().unwrap();
        assert_eq!(y.device(), Device::Cuda(0));
        assert!(y.requires_grad(), "torch: cast output tracks gradients");
        assert!(
            !y.is_leaf(),
            "torch: cast output is a non-leaf ToCopyBackward0"
        );

        sum(&y)
            .expect("bf16 cuda sum")
            .backward()
            .expect("backward through cuda dtype cast");

        let grad = x
            .grad()
            .expect("grad lookup")
            .expect("cast backward must reach the original CUDA leaf");
        assert_eq!(grad.device(), Device::Cuda(0));
        assert_eq!(
            grad.to(Device::Cpu).unwrap().data().unwrap(),
            &[1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn cuda_f32_f64_cast_backward_reaches_cuda_leafs() {
        ensure_cuda_backend();

        let x32 = cuda_tensor(vec![1.0f32, 2.0, 3.0]);
        let y64 = x32.to_dtype::<f64>().expect("cuda f32 -> f64");
        assert_eq!(y64.device(), Device::Cuda(0));
        sum(&mul(&y64, &y64).expect("mul f64 cast"))
            .expect("sum f64 cast")
            .backward()
            .expect("backward f32 -> f64 cast");
        let g32 = x32
            .grad()
            .expect("f32 grad lookup")
            .expect("f32 source receives grad");
        assert_f32_values(&g32, &[2.0, 4.0, 6.0]);

        let x64 = cuda_tensor(vec![1.0f64, 2.0, 3.0]);
        let y32 = x64.to_dtype::<f32>().expect("cuda f64 -> f32");
        assert_eq!(y32.device(), Device::Cuda(0));
        sum(&mul(&y32, &y32).expect("mul f32 cast"))
            .expect("sum f32 cast")
            .backward()
            .expect("backward f64 -> f32 cast");
        let g64 = x64
            .grad()
            .expect("f64 grad lookup")
            .expect("f64 source receives grad");
        assert_f64_values(&g64, &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn cuda_float_to_dtype_pair_matrix_is_resident() {
        ensure_cuda_backend();
        let values = [1.0_f32, -2.0, 0.5];

        let f32_x = cuda_tensor(values.to_vec());
        assert_f64_values(
            &f32_x.to_dtype::<f64>().expect("f32 -> f64"),
            &[1.0, -2.0, 0.5],
        );
        assert_f16_values(&f32_x.to_dtype::<half::f16>().expect("f32 -> f16"), &values);
        assert_bf16_values(
            &f32_x.to_dtype::<half::bf16>().expect("f32 -> bf16"),
            &values,
        );

        let f64_x = cuda_tensor(vec![1.0_f64, -2.0, 0.5]);
        assert_f32_values(&f64_x.to_dtype::<f32>().expect("f64 -> f32"), &values);
        assert_f16_values(&f64_x.to_dtype::<half::f16>().expect("f64 -> f16"), &values);
        assert_bf16_values(
            &f64_x.to_dtype::<half::bf16>().expect("f64 -> bf16"),
            &values,
        );

        let f16_values: Vec<half::f16> = values.iter().map(|&v| half::f16::from_f32(v)).collect();
        let f16_x = cuda_tensor(f16_values);
        assert_f32_values(&f16_x.to_dtype::<f32>().expect("f16 -> f32"), &values);
        assert_f64_values(
            &f16_x.to_dtype::<f64>().expect("f16 -> f64"),
            &[1.0, -2.0, 0.5],
        );
        assert_bf16_values(
            &f16_x.to_dtype::<half::bf16>().expect("f16 -> bf16"),
            &values,
        );

        let bf16_values: Vec<half::bf16> =
            values.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let bf16_x = cuda_tensor(bf16_values);
        assert_f32_values(&bf16_x.to_dtype::<f32>().expect("bf16 -> f32"), &values);
        assert_f64_values(
            &bf16_x.to_dtype::<f64>().expect("bf16 -> f64"),
            &[1.0, -2.0, 0.5],
        );
        assert_f16_values(
            &bf16_x.to_dtype::<half::f16>().expect("bf16 -> f16"),
            &values,
        );
    }
}
