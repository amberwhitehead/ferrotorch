//! ACToR discriminator re-audit of commit `f7ed52a09` (#1441 extended
//! nn-layer parity runner arms for linear / bilinear / conv{1,2,3}d).
//!
//! Every EXPECTED value in this file is the LIVE torch 2.11.0+cu130 output of
//! the matching `torch.nn.functional.*` call (computed via the parity-sweep
//! oracle's torch backend, same op_db callable the `execute` cmd routes to),
//! NOT copied from the ferrotorch side (R-CHAR-3). The torch driver script is
//! reproduced inline above each constant block so the value is regenerable.
//!
//! These tests drive the EXACT production constructors the runner arms use
//! (`Conv2d::new_full` + `Parameter::set_data` + `Module::forward`;
//! `Linear::new` + `set_data` + `forward`; `Bilinear::new` + `set_data` +
//! `forward_pair`). The pre-existing `conv2d_groups_dilation.rs` only checks
//! the production path against a NAIVE Rust reimplementation in the same crate
//! — a shared channel-partition convention bug would pass that test while
//! diverging from torch. This file closes that gap with a torch oracle ref.
//!
//! RESULT (verified against committed f7ed52a09 via /tmp/ft-audit-1441 clean
//! worktree, since the live working tree carries an unrelated non-compiling
//! in-progress conv1d change):
//!   - conv2d groups=2 / dilation=2, linear 3-D, bilinear 2-D: ALL PASS — the
//!     extended arms are REAL and the grouped/dilated conv2d genuinely matches
//!     torch (NOT atol-masked: a mis-partitioned grouped conv diverges by
//!     max ~21.0 / mean ~4.6 absolute on a randn case, 6 orders of magnitude
//!     above the runner's atol=1e-5).
//!   - bilinear zero-batch: FAILS with a `% 0` panic at einsum.rs:1310 — the
//!     real #1605 production crash the runner arm hides behind a skip guard.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::Bilinear;
use ferrotorch_nn::Conv2d;
use ferrotorch_nn::Linear;
use ferrotorch_nn::module::Module;

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Element-wise close check with the runner's conv envelope (rtol=1e-4,
/// atol=1e-5). A correct grouped/dilated conv matches torch to ~6 sig figs;
/// a mis-partitioned grouped conv diverges by O(1) >> this envelope.
fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-5_f32 + 1e-4_f32 * w.abs();
        assert!(
            (g - w).abs() <= tol,
            "{ctx}: element {i} ferrotorch={g} torch={w} |diff|={} > tol={tol}",
            (g - w).abs()
        );
    }
}

/// Drive `Conv2d::new_full` exactly as the runner's conv2d arm does and assert
/// the grouped (groups=2) output matches `torch.nn.functional.conv2d`.
///
/// torch driver (oracle execute / direct torch.nn.functional.conv2d):
///   inp = torch.arange(1*4*3*3).float().reshape(1,4,3,3)
///   w   = (torch.arange(4*2*2*2).float()*0.1 - 1.0).reshape(4,2,2,2)
///   b   = torch.tensor([0.5,-0.5,1.0,-1.0])
///   torch.nn.functional.conv2d(inp,w,b,stride=1,padding=0,dilation=1,groups=2)
///
/// `aten/src/ATen/native/Convolution.cpp` grouped conv: group g of the OUTPUT
/// channels convolves ONLY group g of the INPUT channels. A correct partition
/// must yield torch's values below; a swapped/mis-sliced partition will not.
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "oracle-derived f32 expected values copied verbatim from live torch 2.11 — full precision is intentional"
)]
fn divergence_1441_conv2d_groups2_matches_torch() {
    let in_c = 4;
    let out_c = 4;
    let groups = 2;
    let (kh, kw) = (2usize, 2usize);

    let input: Vec<f32> = (0..(4 * 3 * 3)).map(|i| i as f32).collect();
    let weight: Vec<f32> = (0..(4 * 2 * 2 * 2)).map(|i| i as f32 * 0.1 - 1.0).collect();
    let bias = vec![0.5f32, -0.5, 1.0, -1.0];

    // EXPECTED: live torch.nn.functional.conv2d output, [1,4,2,2] flattened.
    let torch_out: [f32; 16] = [
        -24.700_000_762_939_453,
        -29.899_999_618_530_273,
        -40.299_999_237_060_55,
        -45.5,
        15.900_001_525_878_906,
        17.100_000_381_469_727,
        19.5,
        20.700_000_762_939_453,
        195.800_003_051_757_8,
        203.400_009_155_273_44,
        218.600_006_103_515_6,
        226.200_012_207_031_25,
        350.600_006_103_515_6,
        364.600_036_621_093_75,
        392.599_975_585_937_5,
        406.600_006_103_515_6,
    ];

    let mut conv =
        Conv2d::<f32>::new_full(in_c, out_c, (kh, kw), (1, 1), (0, 0), (1, 1), groups, true)
            .unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        // weight shape [out, in/groups, kH, kW] = [4,2,2,2].
        params[0].set_data(t(&weight, &[4, 2, 2, 2]));
        params[1].set_data(t(&bias, &[4]));
    }
    let y = Module::<f32>::forward(&conv, &t(&input, &[1, 4, 3, 3])).unwrap();
    assert_eq!(y.shape(), &[1, 4, 2, 2]);
    assert_close(
        y.data().unwrap(),
        &torch_out,
        "conv2d groups=2 vs torch.nn.functional.conv2d",
    );
}

/// Dilated dense conv2d (dilation=2, groups=1) through the production path.
///
/// torch driver:
///   inp = torch.arange(1*2*5*5).float().reshape(1,2,5,5)
///   w   = (torch.arange(3*2*2*2).float()*0.05 - 0.3).reshape(3,2,2,2)
///   torch.nn.functional.conv2d(inp,w,None,stride=1,padding=0,dilation=2,groups=1)
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "oracle-derived f32 expected values copied verbatim from live torch 2.11 — full precision is intentional"
)]
fn divergence_1441_conv2d_dilation2_matches_torch() {
    let mut conv = Conv2d::<f32>::new_full(2, 3, (2, 2), (1, 1), (0, 0), (2, 2), 1, false).unwrap();
    let input: Vec<f32> = (0..(2 * 5 * 5)).map(|i| i as f32).collect();
    let weight: Vec<f32> = (0..(3 * 2 * 2 * 2))
        .map(|i| i as f32 * 0.05 - 0.3)
        .collect();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&weight, &[3, 2, 2, 2]));
    }
    let y = Module::<f32>::forward(&conv, &t(&input, &[1, 2, 5, 5])).unwrap();

    // EXPECTED: live torch.nn.functional.conv2d output, [1,3,3,3] flattened.
    let torch_out: [f32; 27] = [
        -6.300_001_621_246_338,
        -7.300_001_144_409_18,
        -8.300_001_144_409_18,
        -11.300_002_098_083_496,
        -12.300_002_098_083_496,
        -13.300_001_144_409_18,
        -16.300_003_051_757_812,
        -17.300_001_144_409_18,
        -18.300_003_051_757_812,
        52.900_001_525_878_906,
        55.099_998_474_121_094,
        57.299_999_237_060_55,
        63.899_997_711_181_64,
        66.099_998_474_121_09,
        68.300_003_051_757_81,
        74.900_001_525_878_9,
        77.099_998_474_121_09,
        79.300_003_051_757_81,
        112.099_998_474_121_09,
        117.5,
        122.899_993_896_484_38,
        139.100_006_103_515_62,
        144.5,
        149.899_993_896_484_38,
        166.100_006_103_515_62,
        171.5,
        176.899_993_896_484_38,
    ];
    assert_eq!(y.shape(), &[1, 3, 3, 3]);
    assert_close(
        y.data().unwrap(),
        &torch_out,
        "conv2d dilation=2 vs torch.nn.functional.conv2d",
    );
}

/// N-D linear (3-D input) through the production `Linear` module path the
/// runner's linear arm uses. Pins the flatten/reshape (*, in) -> (*, out).
///
/// torch driver:
///   inp = torch.arange(2*3*4).float().reshape(2,3,4)*0.1
///   w   = (torch.arange(5*4).float()*0.05 - 0.4).reshape(5,4)
///   b   = torch.tensor([0.1,0.2,0.3,0.4,0.5])
///   torch.nn.functional.linear(inp, w, b)
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "oracle-derived f32 expected values copied verbatim from live torch 2.11 — full precision is intentional"
)]
fn divergence_1441_linear_3d_matches_torch() {
    let mut ln = Linear::<f32>::new(4, 5, true).unwrap();
    let input: Vec<f32> = (0..(2 * 3 * 4)).map(|i| i as f32 * 0.1).collect();
    let weight: Vec<f32> = (0..(5 * 4)).map(|i| i as f32 * 0.05 - 0.4).collect();
    let bias = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
    {
        let mut params = Module::<f32>::parameters_mut(&mut ln);
        params[0].set_data(t(&weight, &[5, 4]));
        params[1].set_data(t(&bias, &[5]));
    }
    let y = Module::<f32>::forward(&ln, &t(&input, &[2, 3, 4])).unwrap();
    assert_eq!(y.shape(), &[2, 3, 5]);

    // EXPECTED: live torch.nn.functional.linear output, [2,3,5] flattened.
    let torch_out: [f32; 30] = [
        -0.070_000_000_298_023_22,
        0.150_000_005_960_464_48,
        0.370_000_004_768_371_6,
        0.590_000_033_378_601_1,
        0.810_000_002_384_185_8,
        -0.590_000_033_378_601_1,
        -0.049_999_997_019_767_76,
        0.490_000_009_536_743_16,
        1.029_999_971_389_770_5,
        1.569_999_933_242_797_9,
        -1.110_000_014_305_114_7,
        -0.249_999_985_098_838_8,
        0.610_000_014_305_114_7,
        1.469_999_909_400_94,
        2.329_999_923_706_054_7,
        -1.629_999_995_231_628_4,
        -0.450_000_047_683_715_8,
        0.730_000_019_073_486_3,
        1.909_999_966_621_399,
        3.090_000_152_587_890_6,
        -2.150_000_095_367_431_6,
        -0.650_000_035_762_786_9,
        0.850_000_023_841_857_9,
        2.350_000_143_051_147_5,
        3.849_999_904_632_568_4,
        -2.670_000_076_293_945_3,
        -0.850_000_083_446_502_7,
        0.970_000_088_214_874_3,
        2.790_000_200_271_606_4,
        4.610_000_133_514_404,
    ];
    assert_close(
        y.data().unwrap(),
        &torch_out,
        "linear 3-D vs torch.nn.functional.linear",
    );
}

/// Bilinear (2-D batched) through the production `Bilinear::forward_pair`.
///
/// torch driver:
///   x1 = (torch.arange(2*3).float()*0.2 - 0.5).reshape(2,3)
///   x2 = (torch.arange(2*2).float()*0.3 + 0.1).reshape(2,2)
///   W  = (torch.arange(4*3*2).float()*0.1 - 0.6).reshape(4,3,2)
///   b  = torch.tensor([0.05,-0.05,0.15,-0.15])
///   torch.nn.functional.bilinear(x1, x2, W, b)
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "oracle-derived f32 expected values copied verbatim from live torch 2.11 — full precision is intentional"
)]
fn divergence_1441_bilinear_2d_matches_torch() {
    let mut bl = Bilinear::<f32>::new(3, 2, 4, true).unwrap();
    let x1: Vec<f32> = (0..(2 * 3)).map(|i| i as f32 * 0.2 - 0.5).collect();
    let x2: Vec<f32> = (0..(2 * 2)).map(|i| i as f32 * 0.3 + 0.1).collect();
    let weight: Vec<f32> = (0..(4 * 3 * 2)).map(|i| i as f32 * 0.1 - 0.6).collect();
    let bias = vec![0.05f32, -0.05, 0.15, -0.15];
    {
        let mut params = Module::<f32>::parameters_mut(&mut bl);
        params[0].set_data(t(&weight, &[4, 3, 2]));
        params[1].set_data(t(&bias, &[4]));
    }
    let y = bl.forward_pair(&t(&x1, &[2, 3]), &t(&x2, &[2, 2])).unwrap();
    assert_eq!(y.shape(), &[2, 4]);

    // EXPECTED: live torch.nn.functional.bilinear output, [2,4] flattened.
    let torch_out: [f32; 8] = [
        0.233_999_997_377_395_63,
        -0.135_999_992_489_814_76,
        -0.206_000_030_040_740_97,
        -0.776_000_022_888_183_6,
        -0.336_000_055_074_691_8,
        0.481_999_993_324_279_8,
        1.600_000_023_841_858,
        2.218_000_173_568_725_6,
    ];
    assert_close(
        y.data().unwrap(),
        &torch_out,
        "bilinear 2-D vs torch.nn.functional.bilinear",
    );
}

/// DIVERGENCE (#1605): production einsum panics with `% 0` on a zero-size
/// batch axis. torch `F.bilinear([0,in1],[0,in2],W,b)` returns a clean
/// `[0, out]`. ferrotorch's `Bilinear::forward_pair` routes through
/// `einsum_differentiable("bi,oij->boj")` whose `decode_multi`
/// (`ferrotorch-core/src/einsum.rs:1310` — `result[i] = remainder % sizes[i];`)
/// divides by the zero batch size -> PANIC. The #1441 runner arm SKIPS these
/// op_db samples (so the conv/bilinear sweep shows 0-failed) but the underlying
/// PRODUCTION op still panics — a crash that violates R-CODE-2 (no panics in
/// production). VERIFIED FAILING against committed f7ed52a09:
///   thread '...' panicked at ferrotorch-core/src/einsum.rs:1310:25:
///   attempt to calculate the remainder with a divisor of zero
///
/// Permanent regression coverage as of #1605: `einsum_two`'s general CPU
/// fallback short-circuits a zero-size OUTPUT dim to an empty tensor and gives
/// 0 contraction terms for a zero shared dim (matching torch's `at::bmm`
/// lowering, `aten/src/ATen/native/Linear.cpp:261-264`) instead of the prior
/// `% 0` panic in `decode_multi`.
///
/// torch driver: torch.nn.functional.bilinear(torch.zeros(0,3), torch.zeros(0,2), W, b)
///   -> shape [0, 4] (no panic).
#[test]
fn divergence_1441_bilinear_zero_batch_panics_1605() {
    let bl = Bilinear::<f32>::new(3, 2, 4, true).unwrap();
    let x1 = t(&[], &[0, 3]);
    let x2 = t(&[], &[0, 2]);
    // torch returns shape [0, 4]; ferrotorch must not panic.
    let y = bl
        .forward_pair(&x1, &x2)
        .expect("forward_pair on zero-batch must not error/panic (torch returns [0,4])");
    assert_eq!(
        y.shape(),
        &[0, 4],
        "zero-batch bilinear must be shape [0, out] like torch"
    );
}
