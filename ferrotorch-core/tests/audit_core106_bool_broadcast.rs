//! CORE-106 (#1800, CLASS-V Medium) regression battery: the `BoolTensor`
//! comparison constructors (float `gt`/`lt`/`ge`/…, integer `gt_int`/…) and
//! logical binary ops (`and`/`or`/`xor`) must broadcast compatible operands
//! the way `torch.gt` / `torch.logical_and` do (right-aligned NumPy rules),
//! on CPU and on CUDA, and keep returning a structured `ShapeMismatch` for
//! genuinely incompatible shapes.
//!
//! Pre-fix observed behavior (R-AHON-1 probe at HEAD, red run pasted in
//! #1800): every broadcast-compatible case below returned
//! `Err(ShapeMismatch)` — the constructors required exactly equal shapes.
//!
//! All expectations are pasted from a LIVE torch==2.11.0+cu130 session
//! (RTX 3090) — snippets quoted per test (R-ORACLE-1(b)). Bool outputs are
//! exact; comparisons are pure, so assertions are `==` with no tolerance.

use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::{BoolTensor, Tensor, TensorStorage};

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Singleton-free trailing-dim broadcast: `[2,3]` vs `[3]`.
///
/// Live torch 2.11.0+cu130:
/// ```text
/// >>> a = torch.tensor([[1.,2.,3.],[4.,5.,6.]]); b = torch.tensor([2.,5.,3.])
/// >>> torch.gt(a, b)
/// tensor([[False, False, False],
///         [ True, False,  True]])
/// >>> torch.lt(a, b)
/// tensor([[ True,  True, False],
///         [False, False, False]])
/// ```
#[test]
fn float_compare_broadcasts_trailing_dim() {
    let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = t(&[2.0, 5.0, 3.0], &[3]);
    let gt = BoolTensor::gt(&a, &b).expect("gt [2,3]x[3] broadcasts in torch");
    assert_eq!(gt.shape(), &[2, 3]);
    assert_eq!(
        gt.data().unwrap(),
        &[false, false, false, true, false, true]
    );
    let lt = BoolTensor::lt(&a, &b).expect("lt [2,3]x[3] broadcasts in torch");
    assert_eq!(
        lt.data().unwrap(),
        &[true, true, false, false, false, false]
    );
}

/// 0-d scalar broadcast: `[2,3]` vs `[]`.
///
/// Live torch 2.11.0+cu130:
/// ```text
/// >>> torch.gt(a, torch.tensor(3.0))
/// tensor([[False, False, False],
///         [ True,  True,  True]])
/// ```
#[test]
fn float_compare_broadcasts_zero_dim_scalar() {
    let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let s = t(&[3.0], &[]);
    let gt = BoolTensor::gt(&a, &s).expect("gt [2,3]x0-d broadcasts in torch");
    assert_eq!(gt.shape(), &[2, 3]);
    assert_eq!(gt.data().unwrap(), &[false, false, false, true, true, true]);
}

/// Multi-axis broadcast: `[2,1,4]` vs `[3,1]` → `[2,3,4]` (both operands
/// expand on different axes).
///
/// Live torch 2.11.0+cu130:
/// ```text
/// >>> a3 = torch.arange(8.).reshape(2,1,4); b3 = torch.tensor([[1.],[5.],[6.]])
/// >>> r3 = torch.ge(a3, b3); r3.shape, r3.flatten().tolist()
/// (torch.Size([2, 3, 4]), [False, True, True, True, False, False, False,
///  False, False, False, False, False, True, True, True, True, False, True,
///  True, True, False, False, True, True])
/// ```
#[test]
fn float_compare_broadcasts_multi_axis() {
    let a = t(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 1, 4]);
    let b = t(&[1.0, 5.0, 6.0], &[3, 1]);
    let ge = BoolTensor::ge(&a, &b).expect("ge [2,1,4]x[3,1] broadcasts in torch");
    assert_eq!(ge.shape(), &[2, 3, 4]);
    assert_eq!(
        ge.data().unwrap(),
        &[
            false, true, true, true, false, false, false, false, false, false, false, false, true,
            true, true, true, false, true, true, true, false, false, true, true
        ]
    );
}

/// Integer comparisons broadcast too: `[2,2]` vs `[1]` and `[2,2]` vs `[2,1]`.
///
/// Live torch 2.11.0+cu130:
/// ```text
/// >>> ai = torch.tensor([[1,2],[3,4]]); bi = torch.tensor([2])
/// >>> torch.gt(ai, bi)
/// tensor([[False, False],
///         [ True,  True]])
/// >>> torch.le(ai, torch.tensor([[2],[4]]))
/// tensor([[True, True],
///         [True, True]])
/// ```
#[test]
fn int_compare_broadcasts() {
    let a = IntTensor::<i32>::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
    let b = IntTensor::<i32>::from_vec(vec![2], vec![1]).unwrap();
    let gt = BoolTensor::gt_int(&a, &b).expect("gt_int [2,2]x[1] broadcasts in torch");
    assert_eq!(gt.shape(), &[2, 2]);
    assert_eq!(gt.data().unwrap(), &[false, false, true, true]);

    let b2 = IntTensor::<i32>::from_vec(vec![2, 4], vec![2, 1]).unwrap();
    let le = BoolTensor::le_int(&a, &b2).expect("le_int [2,2]x[2,1] broadcasts in torch");
    assert_eq!(le.shape(), &[2, 2]);
    assert_eq!(le.data().unwrap(), &[true, true, true, true]);

    let a16 = IntTensor::<i16>::from_vec(vec![-3, 0, 2, i16::MAX], vec![2, 2]).unwrap();
    let b16 = IntTensor::<i16>::from_vec(vec![-3, 7], vec![2, 1]).unwrap();
    assert_eq!(
        BoolTensor::eq_int(&a16, &b16).unwrap().data().unwrap(),
        &[true, false, false, false]
    );
    assert_eq!(
        BoolTensor::ne_int(&a16, &b16).unwrap().data().unwrap(),
        &[false, true, true, true]
    );
    assert_eq!(
        BoolTensor::lt_int(&a16, &b16).unwrap().data().unwrap(),
        &[false, false, true, false]
    );
    assert_eq!(
        BoolTensor::le_int(&a16, &b16).unwrap().data().unwrap(),
        &[true, false, true, false]
    );
    assert_eq!(
        BoolTensor::gt_int(&a16, &b16).unwrap().data().unwrap(),
        &[false, true, false, true]
    );
    assert_eq!(
        BoolTensor::ge_int(&a16, &b16).unwrap().data().unwrap(),
        &[true, true, false, true]
    );
}

/// Logical binary ops broadcast: `[2,2]` vs `[2]` and `[2,2]` vs 0-d.
///
/// Live torch 2.11.0+cu130:
/// ```text
/// >>> m1 = torch.tensor([[True,False],[True,True]]); m2 = torch.tensor([True,False])
/// >>> torch.logical_and(m1, m2)
/// tensor([[ True, False],
///         [ True, False]])
/// >>> torch.logical_or(m1, m2)
/// tensor([[ True, False],
///         [ True,  True]])
/// >>> torch.logical_xor(m1, m2)
/// tensor([[False, False],
///         [False,  True]])
/// >>> torch.logical_and(m1, torch.tensor(True))
/// tensor([[ True, False],
///         [ True,  True]])
/// ```
#[test]
fn logical_ops_broadcast() {
    let m1 = BoolTensor::from_vec(vec![true, false, true, true], vec![2, 2]).unwrap();
    let m2 = BoolTensor::from_vec(vec![true, false], vec![2]).unwrap();
    let and = m1
        .and(&m2)
        .expect("logical_and [2,2]x[2] broadcasts in torch");
    assert_eq!(and.shape(), &[2, 2]);
    assert_eq!(and.data().unwrap(), &[true, false, true, false]);
    assert_eq!(
        m1.or(&m2).unwrap().data().unwrap(),
        &[true, false, true, true]
    );
    assert_eq!(
        m1.xor(&m2).unwrap().data().unwrap(),
        &[false, false, false, true]
    );

    let m0 = BoolTensor::from_vec(vec![true], vec![]).unwrap();
    let and0 = m1
        .and(&m0)
        .expect("logical_and [2,2]x0-d broadcasts in torch");
    assert_eq!(and0.shape(), &[2, 2]);
    assert_eq!(and0.data().unwrap(), &[true, false, true, true]);
}

/// Genuinely incompatible shapes keep the structured error.
///
/// Live torch 2.11.0+cu130:
/// ```text
/// >>> torch.gt(torch.zeros(2,3), torch.zeros(4))
/// RuntimeError: The size of tensor a (3) must match the size of tensor b (4)
/// at non-singleton dimension 1
/// ```
#[test]
fn incompatible_shapes_still_error() {
    let a = t(&[0.0; 6], &[2, 3]);
    let b = t(&[0.0; 4], &[4]);
    assert!(matches!(
        BoolTensor::gt(&a, &b),
        Err(FerrotorchError::ShapeMismatch { .. })
    ));
    let m1 = BoolTensor::zeros(&[2, 3]);
    let m2 = BoolTensor::zeros(&[4]);
    assert!(matches!(
        m1.and(&m2),
        Err(FerrotorchError::ShapeMismatch { .. })
    ));
}

// ---------------------------------------------------------------------------
// CUDA cases (gpu feature + hardware) — every result asserts is_cuda()
// (R-ORACLE-3) before values are read back for comparison.
// ---------------------------------------------------------------------------
#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();
    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the CORE-106 GPU pins");
        });
    }

    /// Float compare broadcast on CUDA, fully resident.
    ///
    /// Live torch 2.11.0+cu130 (RTX 3090):
    /// ```text
    /// >>> torch.gt(a.cuda(), b.cuda())
    /// tensor([[False, False, False],
    ///         [ True, False,  True]], device='cuda:0')
    /// ```
    #[test]
    fn float_compare_broadcasts_on_cuda() {
        ensure_cuda_backend();
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .to(Device::Cuda(0))
            .unwrap();
        let b = t(&[2.0, 5.0, 3.0], &[3]).to(Device::Cuda(0)).unwrap();
        let gt = BoolTensor::gt(&a, &b).expect("cuda gt [2,3]x[3] broadcasts in torch");
        assert!(
            gt.is_cuda(),
            "broadcast gt on CUDA operands must stay resident (got {:?})",
            gt.device()
        );
        assert_eq!(gt.shape(), &[2, 3]);
        assert_eq!(
            gt.to(Device::Cpu).unwrap().data().unwrap(),
            &[false, false, false, true, false, true]
        );
    }

    /// Integer compare broadcast on CUDA. The values are torch-exact and the
    /// mask is returned on the operands' device; broadcasted integer operands
    /// use the resident rank-general compare kernel, with no host value round
    /// trip.
    ///
    /// Live torch 2.11.0+cu130 (RTX 3090):
    /// ```text
    /// >>> torch.gt(ai.cuda(), bi.cuda())
    /// tensor([[False, False],
    ///         [ True,  True]], device='cuda:0')
    /// ```
    #[test]
    fn int_compare_broadcasts_on_cuda() {
        ensure_cuda_backend();
        let a = IntTensor::<i32>::from_vec(vec![1, 2, 3, 4], vec![2, 2])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let b = IntTensor::<i32>::from_vec(vec![2], vec![1])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let gt = BoolTensor::gt_int(&a, &b).expect("cuda gt_int [2,2]x[1] broadcasts in torch");
        assert!(
            gt.is_cuda(),
            "broadcast gt_int on CUDA operands must return a CUDA mask (got {:?})",
            gt.device()
        );
        assert_eq!(gt.shape(), &[2, 2]);
        assert_eq!(
            gt.to(Device::Cpu).unwrap().data().unwrap(),
            &[false, false, true, true]
        );
    }

    /// Signed i16 comparisons are a first-class CUDA TensorIterator case in
    /// PyTorch, both for equal-shape operands and broadcast operands.
    ///
    /// Live torch 2.11.0+cu130 (RTX 3090):
    /// ```text
    /// >>> a = torch.tensor([[-3, 0], [2, 32767]], dtype=torch.int16, device='cuda')
    /// >>> b = torch.tensor([[-3], [7]], dtype=torch.int16, device='cuda')
    /// >>> [op(a, b).cpu().tolist() for op in (torch.eq, torch.ne, torch.lt, torch.le, torch.gt, torch.ge)]
    /// [[[True, False], [False, False]], [[False, True], [True, True]],
    ///  [[False, False], [True, False]], [[True, False], [True, False]],
    ///  [[False, True], [False, True]], [[True, True], [False, True]]]
    /// ```
    #[test]
    fn int16_compare_same_shape_and_broadcast_on_cuda() {
        ensure_cuda_backend();
        let same_a = IntTensor::<i16>::from_vec(vec![i16::MIN, -1, 0, 9], vec![4])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let same_b = IntTensor::<i16>::from_vec(vec![i16::MIN, 0, 0, 5], vec![4])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let same_gt = BoolTensor::gt_int(&same_a, &same_b).expect("same-shape i16 gt runs on CUDA");
        assert!(same_gt.is_cuda());
        assert_eq!(
            same_gt.to(Device::Cpu).unwrap().data().unwrap(),
            &[false, false, false, true]
        );
        let same_ne = BoolTensor::ne_int(&same_a, &same_b).expect("same-shape i16 ne runs on CUDA");
        assert!(same_ne.is_cuda());
        assert_eq!(
            same_ne.to(Device::Cpu).unwrap().data().unwrap(),
            &[false, true, false, true]
        );

        let a = IntTensor::<i16>::from_vec(vec![-3, 0, 2, i16::MAX], vec![2, 2])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let b = IntTensor::<i16>::from_vec(vec![-3, 7], vec![2, 1])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();

        let checks: [(&str, BoolTensor, &[bool]); 6] = [
            (
                "eq",
                BoolTensor::eq_int(&a, &b).unwrap(),
                &[true, false, false, false],
            ),
            (
                "ne",
                BoolTensor::ne_int(&a, &b).unwrap(),
                &[false, true, true, true],
            ),
            (
                "lt",
                BoolTensor::lt_int(&a, &b).unwrap(),
                &[false, false, true, false],
            ),
            (
                "le",
                BoolTensor::le_int(&a, &b).unwrap(),
                &[true, false, true, false],
            ),
            (
                "gt",
                BoolTensor::gt_int(&a, &b).unwrap(),
                &[false, true, false, true],
            ),
            (
                "ge",
                BoolTensor::ge_int(&a, &b).unwrap(),
                &[true, true, false, true],
            ),
        ];

        for (name, mask, expected) in checks {
            assert!(mask.is_cuda(), "i16 {name} broadcast result left CUDA");
            assert_eq!(mask.shape(), &[2, 2], "i16 {name} broadcast shape");
            assert_eq!(
                mask.to(Device::Cpu).unwrap().data().unwrap(),
                expected,
                "i16 {name} broadcast values"
            );
        }
    }

    /// Covers the pre-existing i64 rank-general path with 0-d scalar broadcast.
    ///
    /// Live torch 2.11.0+cu130:
    /// ```text
    /// >>> torch.le(torch.tensor([[1,2],[3,4]], dtype=torch.int64, device='cuda'),
    /// ...          torch.tensor(3, dtype=torch.int64, device='cuda'))
    /// tensor([[ True,  True],
    ///         [ True, False]], device='cuda:0')
    /// ```
    #[test]
    fn int64_compare_zero_dim_broadcast_on_cuda() {
        ensure_cuda_backend();
        let a = IntTensor::<i64>::from_vec(vec![1, 2, 3, 4], vec![2, 2])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let scalar = IntTensor::<i64>::from_vec(vec![3], vec![])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let le =
            BoolTensor::le_int(&a, &scalar).expect("cuda le_int [2,2]x0-d broadcasts in torch");
        assert!(le.is_cuda());
        assert_eq!(le.shape(), &[2, 2]);
        assert_eq!(
            le.to(Device::Cpu).unwrap().data().unwrap(),
            &[true, true, true, false]
        );
    }

    /// Logical op broadcast on CUDA, fully resident via the broadcast_bool
    /// kernel (#1663) + the bool_and kernel.
    ///
    /// Live torch 2.11.0+cu130 (RTX 3090):
    /// ```text
    /// >>> torch.logical_and(m1.cuda(), m2.cuda())
    /// tensor([[ True, False],
    ///         [ True, False]], device='cuda:0')
    /// ```
    #[test]
    fn logical_and_broadcasts_on_cuda() {
        ensure_cuda_backend();
        let m1 = BoolTensor::from_vec(vec![true, false, true, true], vec![2, 2])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let m2 = BoolTensor::from_vec(vec![true, false], vec![2])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let and = m1
            .and(&m2)
            .expect("cuda logical_and [2,2]x[2] broadcasts in torch");
        assert!(
            and.is_cuda(),
            "broadcast and on CUDA operands must stay resident (got {:?})",
            and.device()
        );
        assert_eq!(and.shape(), &[2, 2]);
        assert_eq!(
            and.to(Device::Cpu).unwrap().data().unwrap(),
            &[true, false, true, false]
        );
    }

    /// Same-shape CUDA compares are untouched by the broadcast change:
    /// still resident, still exact.
    ///
    /// Live torch 2.11.0+cu130 (RTX 3090):
    /// ```text
    /// >>> torch.gt(torch.tensor([1.,5.], device='cuda'),
    /// ...          torch.tensor([2.,4.], device='cuda'))
    /// tensor([False,  True], device='cuda:0')
    /// ```
    #[test]
    fn same_shape_cuda_compare_unchanged() {
        ensure_cuda_backend();
        let a = t(&[1.0, 5.0], &[2]).to(Device::Cuda(0)).unwrap();
        let b = t(&[2.0, 4.0], &[2]).to(Device::Cuda(0)).unwrap();
        let gt = BoolTensor::gt(&a, &b).unwrap();
        assert!(gt.is_cuda());
        assert_eq!(gt.to(Device::Cpu).unwrap().data().unwrap(), &[false, true]);
    }
}
