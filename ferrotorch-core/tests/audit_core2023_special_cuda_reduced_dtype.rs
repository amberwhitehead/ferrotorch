#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::{
    Device, Tensor, TensorStorage, beta, digamma, erf, erfc, erfinv, gammainc, gammaincc, gammaln,
    gammaln_sign, log_beta, multigammaln, xlogy,
};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-2023 reduced dtype tests");
    });
}

fn cuda_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::f16> {
    let values: Vec<half::f16> = data.iter().copied().map(half::f16::from_f32).collect();
    Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), false)
        .expect("cpu f16 tensor")
        .to(Device::Cuda(0))
        .expect("upload f16")
        .requires_grad_(requires_grad)
}

fn cuda_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::bf16> {
    let values: Vec<half::bf16> = data.iter().copied().map(half::bf16::from_f32).collect();
    Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), false)
        .expect("cpu bf16 tensor")
        .to(Device::Cuda(0))
        .expect("upload bf16")
        .requires_grad_(requires_grad)
}

fn read_cuda_f16(t: &Tensor<half::f16>, label: &str) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "{label}: expected CUDA-resident f16 tensor, got {:?}",
        t.device()
    );
    t.to(Device::Cpu)
        .expect("download f16")
        .data_vec()
        .expect("logical f16 data")
        .into_iter()
        .map(|v| v.to_f32())
        .collect()
}

fn read_cuda_bf16(t: &Tensor<half::bf16>, label: &str) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "{label}: expected CUDA-resident bf16 tensor, got {:?}",
        t.device()
    );
    t.to(Device::Cpu)
        .expect("download bf16")
        .data_vec()
        .expect("logical bf16 data")
        .into_iter()
        .map(|v| v.to_f32())
        .collect()
}

fn assert_close_or_special(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a:?}");
        } else if e.is_infinite() {
            assert_eq!(a, e, "{label}[{i}]: expected {e:?}, got {a:?}");
        } else {
            let diff = (a - e).abs();
            assert!(
                diff <= tol,
                "{label}[{i}]: expected {e:?}, got {a:?}, diff={diff:?}"
            );
        }
    }
}

fn assert_not_implemented_on_cuda<T>(result: Result<T, FerrotorchError>, op: &'static str) {
    match result {
        Err(FerrotorchError::NotImplementedOnCuda { op: got }) => assert_eq!(got, op),
        Err(other) => panic!("expected NotImplementedOnCuda({op}), got {other:?}"),
        Ok(_) => panic!("expected NotImplementedOnCuda({op}), got Ok"),
    }
}

fn assert_f16_unary_grad<F>(label: &'static str, data: &[f32], expected: &[f32], tol: f32, f: F)
where
    F: Fn(&Tensor<half::f16>) -> Result<Tensor<half::f16>, FerrotorchError>,
{
    let x = cuda_f16(data, &[data.len()], true);
    let loss = reduce_sum(&f(&x).expect(label)).expect("sum f16");
    loss.backward().expect(label);
    let grad = x.grad().expect("grad lookup").expect(label);
    assert_close_or_special(&read_cuda_f16(&grad, label), expected, tol, label);
}

fn assert_bf16_unary_grad<F>(label: &'static str, data: &[f32], expected: &[f32], tol: f32, f: F)
where
    F: Fn(&Tensor<half::bf16>) -> Result<Tensor<half::bf16>, FerrotorchError>,
{
    let x = cuda_bf16(data, &[data.len()], true);
    let loss = reduce_sum(&f(&x).expect(label)).expect("sum bf16");
    loss.backward().expect(label);
    let grad = x.grad().expect("grad lookup").expect(label);
    assert_close_or_special(&read_cuda_bf16(&grad, label), expected, tol, label);
}

#[test]
fn cuda_half_special_unaries_match_pytorch_f32_opmath() {
    ensure_cuda_backend();

    // PyTorch 2.11 CUDA source contract, pinned 2026-06-18:
    // UnarySpecialOpsKernel.cu / UnaryGammaKernels.cu dispatch Half through
    // f32 opmath and store Half output. Runtime Jiterator is blocked locally
    // by missing libnvrtc-builtins.so.13.0 for several ops, so these constants
    // come from torch CPU f32 opmath rounded to float16.
    let x = cuda_f16(&[-0.75, -0.25, 0.0, 0.5, 1.0], &[5], false);

    assert_close_or_special(
        &read_cuda_f16(&erf(&x).expect("erf f16"), "erf f16"),
        &[-0.710_937_5, -0.276_367_2, 0.0, 0.520_507_8, 0.842_773_44],
        2e-3,
        "erf f16",
    );
    assert_close_or_special(
        &read_cuda_f16(&erfc(&x).expect("erfc f16"), "erfc f16"),
        &[1.710_937_5, 1.276_367_2, 1.0, 0.479_492_2, 0.157_348_63],
        2e-3,
        "erfc f16",
    );
    assert_close_or_special(
        &read_cuda_f16(&erfinv(&x).expect("erfinv f16"), "erfinv f16"),
        &[
            -0.813_476_56,
            -0.225_341_8,
            0.0,
            0.477_050_78,
            f32::INFINITY,
        ],
        4e-3,
        "erfinv f16",
    );
    assert_close_or_special(
        &read_cuda_f16(&gammaln(&x).expect("gammaln f16"), "gammaln f16"),
        &[1.576_171_9, 1.589_843_8, f32::INFINITY, 0.572_265_6, 0.0],
        8e-3,
        "gammaln f16",
    );
    assert_close_or_special(
        &read_cuda_f16(&digamma(&x).expect("digamma f16"), "digamma f16"),
        &[
            -2.894_531_3,
            2.914_062_5,
            f32::NEG_INFINITY,
            -1.963_867_2,
            -0.577_148_44,
        ],
        2e-2,
        "digamma f16",
    );
}

#[test]
fn cuda_bfloat_special_unaries_match_pytorch_f32_opmath() {
    ensure_cuda_backend();

    let x = cuda_bf16(&[-0.75, -0.25, 0.0, 0.5, 1.0], &[5], false);

    assert_close_or_special(
        &read_cuda_bf16(&erf(&x).expect("erf bf16"), "erf bf16"),
        &[-0.710_937_5, -0.275_390_63, 0.0, 0.519_531_25, 0.843_75],
        8e-3,
        "erf bf16",
    );
    assert_close_or_special(
        &read_cuda_bf16(&erfc(&x).expect("erfc bf16"), "erfc bf16"),
        &[1.710_937_5, 1.273_437_5, 1.0, 0.480_468_75, 0.157_226_56],
        8e-3,
        "erfc bf16",
    );
    assert_close_or_special(
        &read_cuda_bf16(&erfinv(&x).expect("erfinv bf16"), "erfinv bf16"),
        &[-0.812_5, -0.225_585_94, 0.0, 0.476_562_5, f32::INFINITY],
        2e-2,
        "erfinv bf16",
    );
    assert_close_or_special(
        &read_cuda_bf16(&gammaln(&x).expect("gammaln bf16"), "gammaln bf16"),
        &[1.578_125, 1.585_937_5, f32::INFINITY, 0.574_218_75, 0.0],
        2e-2,
        "gammaln bf16",
    );
    assert_close_or_special(
        &read_cuda_bf16(&digamma(&x).expect("digamma bf16"), "digamma bf16"),
        &[
            -2.890_625,
            2.921_875,
            f32::NEG_INFINITY,
            -1.960_937_5,
            -0.578_125,
        ],
        6e-2,
        "digamma bf16",
    );
}

#[test]
fn cuda_reduced_xlogy_broadcast_matches_torch_branch_order() {
    ensure_cuda_backend();

    let xh = cuda_f16(&[0.0, 2.0], &[2, 1], false);
    let yh = cuda_f16(&[f32::NAN, 0.5, 4.0], &[1, 3], false);
    let out = xlogy(&xh, &yh).expect("xlogy f16");
    assert_eq!(out.shape(), &[2, 3]);
    assert_close_or_special(
        &read_cuda_f16(&out, "xlogy f16"),
        &[f32::NAN, 0.0, 0.0, f32::NAN, -1.386_718_8, 2.773_437_5],
        3e-3,
        "xlogy f16",
    );

    let xb = cuda_bf16(&[0.0, 2.0], &[2, 1], false);
    let yb = cuda_bf16(&[f32::NAN, 0.5, 4.0], &[1, 3], false);
    let out = xlogy(&xb, &yb).expect("xlogy bf16");
    assert_eq!(out.shape(), &[2, 3]);
    assert_close_or_special(
        &read_cuda_bf16(&out, "xlogy bf16"),
        &[f32::NAN, 0.0, 0.0, f32::NAN, -1.382_812_5, 2.765_625],
        2e-2,
        "xlogy bf16",
    );
}

#[test]
fn cuda_reduced_extension_specials_stay_resident_and_match_reference_contracts() {
    ensure_cuda_backend();

    let xh = cuda_f16(&[2.0, 3.5], &[2], false);
    assert_close_or_special(
        &read_cuda_f16(
            &multigammaln(&xh, 3).expect("multigammaln f16"),
            "multigammaln f16",
        ),
        &[1.596_679_7, 3.896_484_4],
        2e-2,
        "multigammaln f16",
    );

    let ah = cuda_f16(&[0.5, 2.0, -0.5], &[3], false);
    let bh = cuda_f16(&[2.0, 0.5, 3.0], &[3], false);
    assert_close_or_special(
        &read_cuda_f16(&log_beta(&ah, &bh).expect("log_beta f16"), "log_beta f16"),
        &[0.287_597_66, 0.287_597_66, 1.673_828_1],
        2e-2,
        "log_beta f16",
    );
    assert_close_or_special(
        &read_cuda_f16(&beta(&ah, &bh).expect("beta f16"), "beta f16"),
        &[1.333_007_8, 1.333_007_8, -5.332_031_3],
        3e-2,
        "beta f16",
    );

    let sb = cuda_bf16(
        &[
            -2.0,
            -1.5,
            -0.0,
            0.0,
            2.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
        ],
        &[8],
        false,
    );
    assert_close_or_special(
        &read_cuda_bf16(
            &gammaln_sign(&sb).expect("gammaln_sign bf16"),
            "gammaln_sign bf16",
        ),
        &[f32::NAN, 1.0, -1.0, 1.0, 1.0, f32::NAN, 1.0, f32::NAN],
        0.0,
        "gammaln_sign bf16",
    );
}

#[test]
fn cuda_reduced_unary_backward_stays_resident_and_uses_f32_opmath() {
    ensure_cuda_backend();

    assert_f16_unary_grad(
        "erf f16 grad",
        &[-0.5, 0.0, 0.5],
        &[0.878_906_25, 1.127_929_7, 0.878_906_25],
        4e-3,
        erf,
    );
    assert_f16_unary_grad(
        "erfc f16 grad",
        &[-0.5, 0.0, 0.5],
        &[-0.878_906_25, -1.127_929_7, -0.878_906_25],
        4e-3,
        erfc,
    );
    assert_f16_unary_grad(
        "erfinv f16 grad",
        &[-0.5, 0.0, 0.5],
        &[1.112_304_7, 0.886_230_47, 1.112_304_7],
        6e-3,
        erfinv,
    );
    assert_f16_unary_grad(
        "lgamma f16 grad",
        &[0.5, 1.0, 2.0],
        &[-1.963_867_2, -0.577_148_44, 0.422_851_56],
        2e-2,
        gammaln,
    );
    assert_f16_unary_grad(
        "digamma f16 grad",
        &[1.0, 2.0, 4.0],
        &[1.644_531_3, 0.645_019_53, 0.283_935_55],
        4e-2,
        digamma,
    );
    assert_f16_unary_grad(
        "multigammaln f16 grad",
        &[2.0, 3.5],
        &[-0.117_919_92, 2.728_515_6],
        5e-2,
        |x| multigammaln(x, 3),
    );

    assert_bf16_unary_grad(
        "erf bf16 grad",
        &[-0.5, 0.0, 0.5],
        &[0.878_906_25, 1.125, 0.878_906_25],
        2e-2,
        erf,
    );
    assert_bf16_unary_grad(
        "erfc bf16 grad",
        &[-0.5, 0.0, 0.5],
        &[-0.878_906_25, -1.125, -0.878_906_25],
        2e-2,
        erfc,
    );
    assert_bf16_unary_grad(
        "erfinv bf16 grad",
        &[-0.5, 0.0, 0.5],
        &[1.109_375, 0.886_718_75, 1.109_375],
        2e-2,
        erfinv,
    );
    assert_bf16_unary_grad(
        "lgamma bf16 grad",
        &[0.5, 1.0, 2.0],
        &[-1.960_937_5, -0.578_125, 0.421_875],
        6e-2,
        gammaln,
    );
    assert_bf16_unary_grad(
        "digamma bf16 grad",
        &[1.0, 2.0, 4.0],
        &[1.648_437_5, 0.644_531_25, 0.283_203_13],
        8e-2,
        digamma,
    );
    assert_bf16_unary_grad(
        "multigammaln bf16 grad",
        &[2.0, 3.5],
        &[-0.118_164_06, 2.734_375],
        8e-2,
        |x| multigammaln(x, 3),
    );
}

#[test]
fn cuda_reduced_binary_backward_stays_resident_and_reduces_broadcast_axes() {
    ensure_cuda_backend();

    let xh = cuda_f16(&[0.5, 2.0], &[2, 1], true);
    let yh = cuda_f16(&[2.0, 4.0], &[1, 2], true);
    reduce_sum(&xlogy(&xh, &yh).expect("xlogy f16 forward"))
        .expect("xlogy f16 sum")
        .backward()
        .expect("xlogy f16 backward");
    assert_close_or_special(
        &read_cuda_f16(
            &xh.grad().expect("xh grad").expect("xlogy f16 dx"),
            "xlogy f16 dx",
        ),
        &[2.080_078_1, 2.080_078_1],
        5e-3,
        "xlogy f16 dx",
    );
    assert_close_or_special(
        &read_cuda_f16(
            &yh.grad().expect("yh grad").expect("xlogy f16 dy"),
            "xlogy f16 dy",
        ),
        &[1.25, 0.625],
        0.0,
        "xlogy f16 dy",
    );

    let xb = cuda_bf16(&[0.5, 2.0], &[2, 1], true);
    let yb = cuda_bf16(&[2.0, 4.0], &[1, 2], true);
    reduce_sum(&xlogy(&xb, &yb).expect("xlogy bf16 forward"))
        .expect("xlogy bf16 sum")
        .backward()
        .expect("xlogy bf16 backward");
    assert_close_or_special(
        &read_cuda_bf16(
            &xb.grad().expect("xb grad").expect("xlogy bf16 dx"),
            "xlogy bf16 dx",
        ),
        &[2.078_125, 2.078_125],
        2e-2,
        "xlogy bf16 dx",
    );
    assert_close_or_special(
        &read_cuda_bf16(
            &yb.grad().expect("yb grad").expect("xlogy bf16 dy"),
            "xlogy bf16 dy",
        ),
        &[1.25, 0.625],
        0.0,
        "xlogy bf16 dy",
    );

    let ah = cuda_f16(&[2.0, 3.0], &[2], true);
    let bh = cuda_f16(&[0.5, 1.5], &[2], true);
    reduce_sum(&log_beta(&ah, &bh).expect("log_beta f16 forward"))
        .expect("log_beta f16 sum")
        .backward()
        .expect("log_beta f16 backward");
    assert_close_or_special(
        &read_cuda_f16(
            &ah.grad().expect("ah grad").expect("log_beta f16 da"),
            "log_beta f16 da",
        ),
        &[-0.280_273_44, -0.466_064_45],
        3e-2,
        "log_beta f16 da",
    );
    assert_close_or_special(
        &read_cuda_f16(
            &bh.grad().expect("bh grad").expect("log_beta f16 db"),
            "log_beta f16 db",
        ),
        &[-2.666_015_6, -1.352_539_1],
        5e-2,
        "log_beta f16 db",
    );

    let ab = cuda_bf16(&[2.0, 3.0], &[2], true);
    let bb = cuda_bf16(&[0.5, 1.5], &[2], true);
    reduce_sum(&beta(&ab, &bb).expect("beta bf16 forward"))
        .expect("beta bf16 sum")
        .backward()
        .expect("beta bf16 backward");
    assert_close_or_special(
        &read_cuda_bf16(
            &ab.grad().expect("ab grad").expect("beta bf16 da"),
            "beta bf16 da",
        ),
        &[-0.373_046_88, -0.070_800_78],
        5e-2,
        "beta bf16 da",
    );
    assert_close_or_special(
        &read_cuda_bf16(
            &bb.grad().expect("bb grad").expect("beta bf16 db"),
            "beta bf16 db",
        ),
        &[-3.562_5, -0.206_054_69],
        8e-2,
        "beta bf16 db",
    );
}

#[test]
fn cuda_gammainc_reduced_dtype_rejection_remains_pytorch_parity() {
    ensure_cuda_backend();

    // PyTorch CUDA IGammaKernel.cu dispatches AT_DISPATCH_FLOATING_TYPES only.
    // Half/BFloat16 must still reject cleanly instead of using the f32-opmath
    // widening route that is correct for erf/lgamma/digamma/xlogy.
    let ah = cuda_f16(&[0.5, 2.0], &[2], false);
    let xh = cuda_f16(&[0.5, 1.5], &[2], false);
    assert_not_implemented_on_cuda(gammainc(&ah, &xh), "gammainc");
    assert_not_implemented_on_cuda(gammaincc(&ah, &xh), "gammaincc");

    let ab = cuda_bf16(&[0.5, 2.0], &[2], false);
    let xb = cuda_bf16(&[0.5, 1.5], &[2], false);
    assert_not_implemented_on_cuda(gammainc(&ab, &xb), "gammainc");
    assert_not_implemented_on_cuda(gammaincc(&ab, &xb), "gammaincc");
}
