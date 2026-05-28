//! Divergence audit for commit 94e08dbd0 (REQ-26, #1345):
//! `linalg::householder_product` forward [m,k] shape fix +
//! `HouseholderProductBackward` reflector-recursion VJP.
//!
//! All reference values below are LIVE `torch.linalg.householder_product`
//! float64 (torch 2.11.0) outputs (R-CHAR-3 (a)); reproduce with:
//!   import torch; torch.set_default_dtype(torch.float64)
//!   V = torch.tensor(<v>).reshape(<shape>).requires_grad_(True)
//!   tau = torch.tensor(<tau>).requires_grad_(True)
//!   Q = torch.linalg.householder_product(V, tau)         # shape [m, k]
//!   Q.backward(torch.tensor(<g>).reshape(Q.shape))
//!   Q.detach().reshape(-1); V.grad.reshape(-1); tau.grad.reshape(-1)
//!
//! The generator's in-lib tests only assert grads for V matrices whose
//! upper-triangle/diagonal are ALREADY clean (1s on the diagonal, 0s above) and
//! never asserted the forward Q VALUES against torch. These probes attack:
//!  (1) forward Q values (shape + element values) for square/tall/truncated,
//!  (2) implicit-unit-diagonal / strict-lower invariance: a "dirty" V whose
//!      diagonal and upper-triangle hold garbage must produce the SAME Q and
//!      grads as the clean V (torch does `tril(-1)` + `diag.fill_(1)`),
//!  (3) tau_j == 0 reflector (identity reflector, sigma_j = 0).

use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::linalg::householder_product;
use ferrotorch_core::{Tensor, TensorStorage};

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}
fn nograd(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn close(a: &[f64], b: &[f64], tol: f64, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: len {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "{label}[{i}]: ferrotorch={x}, torch={y}, diff={}",
            (x - y).abs()
        );
    }
}

/// Forward Q + (V.grad, tau.grad) via sum(Q*g) loss.
fn fwd_and_grad(
    v: &[f64],
    shape: &[usize],
    tau: &[f64],
    g: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = shape[0];
    let k = shape[1];
    let vt = leaf(v, shape);
    let taut = leaf(tau, &[k]);
    let q = householder_product(&vt, &taut).expect("householder_product");
    assert_eq!(q.shape(), &[m, k], "torch returns leading k columns [m,k]");
    let q_vals = q.data().unwrap().to_vec();
    let gt = nograd(g, &[m, k]);
    let loss = sum(&mul(&q, &gt).unwrap()).unwrap();
    loss.backward().unwrap();
    let gv = vt.grad().unwrap().unwrap().data().unwrap().to_vec();
    let gtau = taut.grad().unwrap().unwrap().data().unwrap().to_vec();
    (q_vals, gv, gtau)
}

/// Divergence probe: forward Q VALUES for square 3x3 vs torch.
/// Upstream `torch.linalg.householder_product` returns the listed [3,3].
/// (The generator's lib tests assert grads but never the forward values.)
#[test]
fn divergence_hh_forward_values_square_3x3() {
    let v = [1.0, 0.2, 0.3, 0.5, 1.0, 0.1, 0.3, 0.15, 1.0];
    let tau = [0.4, 0.5, 0.6];
    let g = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
    let (q, _, _) = fwd_and_grad(&v, &[3, 3], &tau, &g);
    let torch_q = [
        0.6,
        -0.091_000_000_000_000_01,
        -0.041_460_000_000_000_004,
        -0.2,
        0.454_5,
        -0.050_73,
        -0.12,
        -0.102_3,
        0.383_062,
    ];
    close(&q, &torch_q, 1e-9, "hh fwd square 3x3 Q vs torch");
}

/// Divergence probe: forward Q VALUES for tall 4x2 (truncated k<m) vs torch.
#[test]
fn divergence_hh_forward_values_tall_4x2() {
    let v = [1.0, 0.2, 0.5, 1.0, 0.3, 0.15, 0.6, 0.35];
    let tau = [0.4, 0.5];
    let g = [0.2, -0.5, 0.3, 0.1, -0.6, 0.8, 0.4, -0.2];
    let (q, _, _) = fwd_and_grad(&v, &[4, 2], &tau, &g);
    let torch_q = [0.6, -0.049, -0.2, 0.475_5, -0.12, -0.0897, -0.24, -0.204_4];
    close(&q, &torch_q, 1e-9, "hh fwd tall 4x2 Q vs torch");
}

/// Divergence probe: implicit-unit-diagonal / strict-lower invariance.
/// A "dirty" V (diagonal=999, upper-triangle=888) must yield IDENTICAL Q,
/// V.grad, tau.grad to the clean V with the same strict-lower part — torch
/// does `input.tril(-1); input.diagonal().fill_(1)` before everything
/// (FunctionsManual.cpp:5564-5565) and the forward ignores diag+upper.
/// Reference = LIVE torch on the dirty V (matches the clean-V oracle exactly).
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "literals are verbatim LIVE torch float64 repr() oracle values; \
              trailing digits beyond f64 precision are kept for provenance"
)]
fn divergence_hh_dirty_input_invariance_square_3x3() {
    let v_dirty = [999.0, 888.0, 888.0, 0.5, 999.0, 888.0, 0.3, 0.15, 999.0];
    let tau = [0.4, 0.5, 0.6];
    let g = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
    let (q, gv, gtau) = fwd_and_grad(&v_dirty, &[3, 3], &tau, &g);
    // LIVE torch on the DIRTY V (identical to clean V — diag/upper ignored).
    let torch_q = [
        0.6,
        -0.091_000_000_000_000_01,
        -0.041_460_000_000_000_004,
        -0.2,
        0.454_5,
        -0.050_73,
        -0.12,
        -0.102_3,
        0.383_062,
    ];
    let torch_gv = [
        0.0,
        0.0,
        0.0,
        -0.063_616_000_000_000_034,
        0.0,
        0.0,
        0.059_570_000_000_000_04,
        -0.320_46,
        0.0,
    ];
    let torch_gtau = [
        -0.181_823_749_999_999_98,
        -0.236_509_000_000_000_02,
        -0.217_588_749_999_999_94,
    ];
    close(&q, &torch_q, 1e-9, "hh dirty square 3x3 Q vs torch");
    close(&gv, &torch_gv, 1e-9, "hh dirty square 3x3 V.grad vs torch");
    close(
        &gtau,
        &torch_gtau,
        1e-9,
        "hh dirty square 3x3 tau.grad vs torch",
    );
}

/// Divergence probe: dirty tall 4x3 (diag=999, upper=888) forward + grads.
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "literals are verbatim LIVE torch float64 repr() oracle values; \
              trailing digits beyond f64 precision are kept for provenance"
)]
fn divergence_hh_dirty_input_invariance_tall_4x3() {
    let v_dirty = [
        999.0, 888.0, 888.0, 0.5, 999.0, 888.0, 0.3, 0.15, 999.0, 0.6, 0.35, 0.4,
    ];
    let tau = [0.4, 0.5, 0.6];
    let g = [
        0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25, 0.4, -0.2, 0.9,
    ];
    let (q, gv, gtau) = fwd_and_grad(&v_dirty, &[4, 3], &tau, &g);
    let torch_q = [
        0.6,
        -0.049,
        0.005_975_999_999_999_995,
        -0.2,
        0.475_5,
        0.014_987_999_999_999_994,
        -0.12,
        -0.0897,
        0.403_592_800_000_000_03,
        -0.24,
        -0.204_4,
        -0.232_214_4,
    ];
    let torch_gv = [
        0.0,
        0.0,
        0.0,
        -0.066_642_400_000_000_046,
        0.0,
        0.0,
        0.013_191_200_000_000_153,
        -0.341_559_599_999_999_63,
        0.0,
        -0.062_754_800_000_000_027,
        0.021_881_199_999_999_948,
        -0.419_784_15,
    ];
    let torch_gtau = [
        -0.352_916_900_000_000_03,
        -0.258_881_520_000_000_09,
        -0.424_873_349_999_999_48,
    ];
    close(&q, &torch_q, 1e-9, "hh dirty tall 4x3 Q vs torch");
    close(&gv, &torch_gv, 1e-9, "hh dirty tall 4x3 V.grad vs torch");
    close(
        &gtau,
        &torch_gtau,
        1e-9,
        "hh dirty tall 4x3 tau.grad vs torch",
    );
}

/// Divergence probe: tau_j == 0 (identity reflector i=1). torch: sigma_1 =
/// 0/(0-1) = 0 so the i=1 reflector is the identity; forward + grads stay
/// finite. Reference = LIVE torch with tau = [0.4, 0.0, 0.6].
#[test]
fn divergence_hh_tau_zero_middle_square_3x3() {
    let v = [1.0, 0.2, 0.3, 0.5, 1.0, 0.1, 0.3, 0.15, 1.0];
    let tau = [0.4, 0.0, 0.6];
    let g = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
    let (q, gv, gtau) = fwd_and_grad(&v, &[3, 3], &tau, &g);
    let torch_q = [0.6, -0.2, -0.048, -0.2, 0.9, -0.024, -0.12, -0.06, 0.385_6];
    let torch_gv = [0.0, 0.0, 0.0, -0.036_8, 0.0, 0.0, -0.024, 0.0, 0.0];
    let torch_gtau = [-0.134, -0.236_509, -0.181];
    close(&q, &torch_q, 1e-9, "hh tau=0 mid Q vs torch");
    close(&gv, &torch_gv, 1e-9, "hh tau=0 mid V.grad vs torch");
    close(&gtau, &torch_gtau, 1e-9, "hh tau=0 mid tau.grad vs torch");
}

// ---------------------------------------------------------------------------
// Randomized oracle probes (torch float64, seed 20260527). The generator only
// tested 3-reflector hand-picked matrices; these stress the accumulation order
// with 5 reflectors (5x5), a deep truncated chain (6x3), and the degenerate
// single-reflector (4x1) case.
// ---------------------------------------------------------------------------

/// Divergence probe: dense 5x5, k=5 (5 reflectors — deepest accumulation chain).
#[test]
fn divergence_hh_rand_5x5() {
    let v = [
        -0.860_390_640_804_349_6,
        -1.181_005_741_904_507_2,
        -1.680_987_726_719_389_3,
        0.960_002_161_877_625_3,
        0.018_228_467_839_749_095,
        -0.278_899_433_927_946_97,
        -1.339_932_146_025_129_4,
        -1.417_412_470_191_54,
        0.224_441_626_257_53,
        1.363_906_974_488_316,
        1.951_482_731_695_170_8,
        0.409_553_589_611_978_85,
        0.574_057_938_306_661_7,
        0.508_033_589_081_896_7,
        0.347_581_312_597_432_5,
        -0.472_282_338_255_401_8,
        0.075_799_203_805_963_78,
        -0.022_789_808_353_076_077,
        -0.760_861_119_377_913_8,
        0.260_648_932_661_293_35,
        0.216_204_081_454_930_7,
        -1.011_502_888_720_027_5,
        0.656_681_623_449_037,
        -0.518_551_000_366_473_4,
        -0.621_968_978_157_899_4,
    ];
    let tau = [
        1.490_138_736_996_093,
        0.911_296_095_914_049_4,
        0.581_667_376_873_338_5,
        0.467_240_347_297_260_94,
        0.970_835_731_958_463_8,
    ];
    let g = [
        1.968_238_144_058_202_6,
        -0.924_514_990_102_285_2,
        1.049_209_101_325_479,
        0.072_336_483_711_922_53,
        -0.333_122_429_111_028_46,
        -1.352_916_992_046_643_5,
        0.468_962_705_252_267_46,
        -0.622_979_474_871_547_3,
        0.310_583_715_025_425_37,
        -1.052_967_304_774_175,
        -1.293_869_940_665_41,
        0.411_112_524_963_505_2,
        -0.220_158_210_683_962_07,
        -0.235_619_586_462_071_48,
        -0.854_655_891_137_596_8,
        -0.160_166_333_613_520_6,
        -0.462_348_812_606_211_9,
        -0.326_621_789_163_831_6,
        0.197_118_393_170_628_5,
        1.024_068_457_408_067_8,
        -0.114_141_116_224_816_67,
        1.619_197_989_768_093_7,
        -0.013_575_407_986_493_837,
        0.518_795_305_115_307_1,
        0.094_618_520_576_180_12,
    ];
    let (q, gv, gtau) = fwd_and_grad(&v, &[5, 5], &tau, &g);
    let torch_q = [
        -0.490_138_736_996_093,
        0.776_608_684_768_225_4,
        -0.882_417_157_320_780_3,
        0.498_940_254_504_192_94,
        0.018_779_845_293_460_246,
        0.415_598_850_222_316_2,
        -0.127_891_818_479_434_84,
        -0.263_034_086_396_733_2,
        0.027_406_342_627_386_834,
        0.015_525_595_583_732_295,
        -2.907_980_013_077_927_4,
        1.142_313_850_328_709_4,
        -1.512_209_226_416_376_2,
        0.956_404_253_909_009_3,
        0.035_505_607_198_351_04,
        0.703_766_207_033_466,
        -0.435_854_084_053_557_86,
        0.391_413_740_104_056,
        0.311_692_310_789_353_1,
        -9.553_679_349_431_083e-6,
        -0.322_174_076_872_650_8,
        1.089_684_600_836_580_1,
        -0.057_756_158_568_478_6,
        0.125_548_353_577_790_98,
        0.002_223_512_030_028_107,
    ];
    let torch_gv = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.920_242_665_654_383,
        0.0,
        0.0,
        0.0,
        0.0,
        2.172_646_708_158_507_7,
        2.671_155_508_264_565_7,
        0.0,
        0.0,
        0.0,
        0.753_307_968_459_044_7,
        0.089_176_082_065_379_87,
        -0.715_542_367_803_553_1,
        0.0,
        0.0,
        1.085_726_796_700_387_4,
        -1.446_736_722_863_233_7,
        0.425_050_008_301_903_5,
        -0.015_835_557_816_613_97,
        0.0,
    ];
    let torch_gtau = [
        -0.446_288_064_779_222_6,
        1.740_258_875_536_158,
        3.122_638_421_335_055_5,
        0.097_439_227_978_207_84,
        1.808_665_546_057_054_7,
    ];
    close(&q, &torch_q, 1e-8, "hh rand 5x5 Q vs torch");
    close(&gv, &torch_gv, 1e-8, "hh rand 5x5 V.grad vs torch");
    close(&gtau, &torch_gtau, 1e-8, "hh rand 5x5 tau.grad vs torch");
}

/// Divergence probe: tall 6x3, k=3 < m (deep truncated chain).
#[test]
fn divergence_hh_rand_6x3() {
    let v = [
        0.780_601_875_127_541_3,
        -0.559_122_948_002_774_3,
        -0.818_972_575_423_840_4,
        -0.244_708_168_315_322_58,
        0.590_362_212_286_774_3,
        0.448_093_195_893_070_23,
        0.297_617_106_457_348_73,
        -0.830_347_047_118_230_5,
        -0.123_759_767_081_031_95,
        -0.991_618_270_032_897_9,
        0.375_416_122_417_001_6,
        0.885_123_398_590_677_8,
        0.611_387_148_958_021_9,
        0.638_607_103_574_068_2,
        1.156_603_827_996_155_5,
        -0.023_780_722_463_915_653,
        -0.981_813_828_931_864,
        -0.291_501_138_658_310_45,
    ];
    let tau = [
        1.224_381_797_312_343_4,
        1.507_734_086_759_036_8,
        0.661_015_280_187_052_5,
    ];
    let g = [
        0.212_131_318_057_197_26,
        -0.772_001_659_968_248_4,
        0.074_128_038_974_970_58,
        0.497_255_842_607_240_9,
        -0.464_187_050_814_999_4,
        0.121_735_635_422_202_59,
        -0.300_908_466_878_723_25,
        1.376_759_698_476_546_2,
        -0.307_822_104_699_414_8,
        0.540_117_185_933_277_7,
        0.408_560_353_506_244_1,
        -0.830_250_590_160_043_9,
        1.165_672_098_127_892_7,
        0.236_400_558_587_156_9,
        -0.183_302_248_327_371_96,
        -0.130_949_758_041_322_83,
        -0.375_411_448_994_41,
        -0.319_330_550_126_807_17,
    ];
    let (q, gv, gtau) = fwd_and_grad(&v, &[6, 3], &tau, &g);
    let torch_q = [
        -0.224_381_797_312_343_42,
        -0.531_691_122_705_459_5,
        0.723_766_276_970_441_5,
        0.299_616_226_938_926_1,
        -0.377_624_926_012_266_3,
        1.599_816_445_628_210_7,
        -0.364_396_967_715_147_7,
        1.093_702_173_311_210_2,
        -0.921_076_944_223_309_9,
        1.214_119_359_710_636_3,
        -0.038_793_053_197_979_69,
        -0.635_692_548_154_481_4,
        -0.748_571_296_294_892_3,
        -1.287_918_817_742_262,
        0.812_727_418_502_856_3,
        0.029_116_683_711_755_07,
        1.492_958_175_757_563_2,
        -1.569_137_427_713_034_5,
    ];
    let torch_gv = [
        0.0,
        0.0,
        0.0,
        -2.045_993_128_864_658_7,
        0.0,
        0.0,
        1.115_016_795_546_537_5,
        -3.655_692_535_442_800_5,
        0.0,
        -1.893_626_055_915_418_3,
        0.831_243_169_092_628_4,
        0.246_465_521_134_247_72,
        -2.579_845_293_116_728_4,
        -1.411_163_548_030_3,
        0.855_935_328_219_577_5,
        2.311_696_673_947_808_6,
        -0.128_414_562_117_491_4,
        -0.420_626_226_393_931_7,
    ];
    let torch_gtau = [
        0.433_147_187_789_925_67,
        1.955_591_019_975_493_8,
        1.774_016_216_652_828_6,
    ];
    close(&q, &torch_q, 1e-8, "hh rand 6x3 Q vs torch");
    close(&gv, &torch_gv, 1e-8, "hh rand 6x3 V.grad vs torch");
    close(&gtau, &torch_gtau, 1e-8, "hh rand 6x3 tau.grad vs torch");
}

/// Divergence probe: single reflector 4x1, k=1 (degenerate chain — no advance).
#[test]
fn divergence_hh_rand_4x1() {
    let v = [
        -0.186_807_320_058_418_55,
        -1.228_576_951_838_881_8,
        1.403_126_446_403_931,
        -0.335_381_171_786_733_5,
    ];
    let tau = [1.399_655_588_900_360_8];
    let g = [
        0.760_625_196_695_654_3,
        0.488_764_269_757_200_8,
        0.281_807_243_980_54,
        -1.103_336_391_351_525_3,
    ];
    let (q, gv, gtau) = fwd_and_grad(&v, &[4, 1], &tau, &g);
    let torch_q = [
        -0.399_655_588_900_360_8,
        1.719_584_597_035_460_4,
        -1.963_893_772_643_164_7,
        0.469_418_131_503_253_5,
    ];
    let torch_gv = [
        0.0,
        -0.684_101_641_820_470_5,
        -0.394_433_084_029_971_46,
        1.544_290_946_592_319_4,
    ];
    let torch_gtau = [-0.925_590_128_613_317_6];
    close(&q, &torch_q, 1e-9, "hh rand 4x1 Q vs torch");
    close(&gv, &torch_gv, 1e-9, "hh rand 4x1 V.grad vs torch");
    close(&gtau, &torch_gtau, 1e-9, "hh rand 4x1 tau.grad vs torch");
}
