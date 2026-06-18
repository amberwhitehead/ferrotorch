//! Adversarial re-audit of commit a21490bb6 (Bilinear / Conv2d padding_mode /
//! Embedding extras / FeatureAlphaDropout / functional dropout RNG) and
//! 21e019daf (Conv*d/Linear bias init U(-bound, bound)).
//!
//! These commits landed on builder self-report + orchestrator smoke. This
//! file is the INDEPENDENT critic pass requested under #1542: "do not trust
//! weak signals like compilability". Every expected value here is derived
//! from a math identity, a PyTorch source `file:line`, or a hand-computed
//! contraction — NEVER copied from the ferrotorch side (R-CHAR-3).

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_nn::padding::PaddingMode;
use ferrotorch_nn::{Bilinear, Conv2d, Embedding, FeatureAlphaDropout, Linear, Module, Parameter};

fn t2d(d: &[f32], r: usize, c: usize) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(d.to_vec()), vec![r, c], false).unwrap()
}

// ===========================================================================
// #1442 Bilinear — hand-computed x1^T W x2 + b against an ASYMMETRIC weight
// (the existing wave-I test uses all-ones weight, which can't catch index
// transposition bugs in the einsum decomposition).
// ===========================================================================

/// Bilinear with a non-symmetric weight pins the exact contraction
/// y[o] = sum_i sum_j x1[i] * W[o,i,j] * x2[j] + b[o].
/// W[o,i,j] distinct per index so any (i,j) swap shows.
/// x1 = [1, 2], x2 = [3, 5, 7]. out=1, in1=2, in2=3.
///
/// Hand computation for o=0, W[0] = [[0,1,2],[10,11,12]]:
///   i=0 (x1=1): 1*(0*3 + 1*5 + 2*7) = 19
///   i=1 (x1=2): 2*(10*3 + 11*5 + 12*7) = 2*169 = 338
///   total = 357. With bias b[0] = 1000 -> 1357.
#[test]
fn bilinear_asymmetric_weight_hand_computed() {
    let mut layer = Bilinear::<f32>::new(2, 3, 1, true).unwrap();
    let w: Vec<f32> = vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
    layer.weight = Parameter::from_slice(&w, &[1, 2, 3]).unwrap();
    layer.bias = Some(Parameter::from_slice(&[1000.0f32], &[1]).unwrap());

    let x1 = t2d(&[1.0, 2.0], 1, 2);
    let x2 = t2d(&[3.0, 5.0, 7.0], 1, 3);
    let y = layer.forward_pair(&x1, &x2).unwrap();
    assert_eq!(y.shape(), &[1, 1]);
    let got = y.data().unwrap()[0];
    assert!(
        (got - 1357.0).abs() < 1e-3,
        "Bilinear contraction wrong: got {got}, expected 1357.0 (357 + bias 1000)"
    );
}

/// Two output features with distinct weights — catches a stub that returns
/// the same value for all out channels or swaps the i/j contraction order.
#[test]
fn bilinear_two_outputs_distinct() {
    let mut layer = Bilinear::<f32>::new(2, 2, 2, false).unwrap();
    // W[o,i,j]: out 0 = [[1,0],[0,1]], out 1 = [[0,1],[1,0]]
    let w: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    layer.weight = Parameter::from_slice(&w, &[2, 2, 2]).unwrap();
    let x1 = t2d(&[2.0, 3.0], 1, 2);
    let x2 = t2d(&[5.0, 7.0], 1, 2);
    let y = layer.forward_pair(&x1, &x2).unwrap();
    let d = y.data().unwrap();
    // o=0: x1[0]*x2[0] + x1[1]*x2[1] = 2*5 + 3*7 = 31
    // o=1: x1[0]*x2[1] + x1[1]*x2[0] = 2*7 + 3*5 = 29
    assert!((d[0] - 31.0).abs() < 1e-3, "out0 got {} want 31", d[0]);
    assert!((d[1] - 29.0).abs() < 1e-3, "out1 got {} want 29", d[1]);
}

// ===========================================================================
// #1443 Conv2d padding_mode — the padding mode must CHANGE the output.
// The existing wave-I test uses an identity kernel (output == input
// regardless of pad mode) so it cannot prove reflect != zeros. This uses a
// box kernel and checks each non-zero mode differs from zero-padding.
// ===========================================================================

fn box_conv(mode: Option<PaddingMode>) -> Vec<f32> {
    let mut conv = Conv2d::<f32>::new(1, 1, (3, 3), (1, 1), (1, 1), false).unwrap();
    if let Some(m) = mode {
        conv = conv.with_padding_mode(m);
    }
    let kernel = vec![1.0f32; 9];
    conv.set_weight(Parameter::from_slice(&kernel, &[1, 1, 3, 3]).unwrap())
        .unwrap();
    let input: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let x = Tensor::from_storage(TensorStorage::cpu(input), vec![1, 1, 4, 4], false).unwrap();
    conv.forward(&x).unwrap().data().unwrap().to_vec()
}

/// Reflect padding must produce a DIFFERENT result than zero padding at the
/// border. Upstream `_ConvNd._conv_forward` routes non-zero modes through
/// `F.pad(..., mode='reflect')` (torch/nn/modules/conv.py).
#[test]
fn conv2d_reflect_differs_from_zero_pad() {
    let zeros = box_conv(None);
    let reflect = box_conv(Some(PaddingMode::Reflect));
    assert!(
        (zeros[0] - reflect[0]).abs() > 1e-4,
        "reflect padding did not change border output: zero={}, reflect={} (padding_mode is a no-op / stub)",
        zeros[0],
        reflect[0]
    );
}

/// Replicate padding must differ from zeros at the border.
#[test]
fn conv2d_replicate_differs_from_zero_pad() {
    let zeros = box_conv(None);
    let repl = box_conv(Some(PaddingMode::Replicate));
    assert!(
        (zeros[0] - repl[0]).abs() > 1e-4,
        "replicate padding is a no-op: zero={}, replicate={}",
        zeros[0],
        repl[0]
    );
}

/// Circular padding must differ from zeros at the border.
#[test]
fn conv2d_circular_differs_from_zero_pad() {
    let zeros = box_conv(None);
    let circ = box_conv(Some(PaddingMode::Circular));
    assert!(
        (zeros[0] - circ[0]).abs() > 1e-4,
        "circular padding is a no-op: zero={}, circular={}",
        zeros[0],
        circ[0]
    );
}

// ===========================================================================
// #1445 Embedding max_norm — gathered rows clipped so L2 norm <= max_norm.
// torch/nn/functional.py:_no_grad_embedding_renorm_.
// ===========================================================================

/// Row0 = [3,4] has L2 norm 5; max_norm = 1.0. After forward the gathered
/// row must have norm <= 1.0 (clipped) and stay direction-preserving (not 0).
#[test]
fn embedding_max_norm_clips_output_rows() {
    let weight = Tensor::from_storage(
        TensorStorage::cpu(vec![3.0f32, 4.0, 0.0, 0.0]),
        vec![2, 2],
        false,
    )
    .unwrap();
    let emb = Embedding::<f32>::from_pretrained(weight, None)
        .unwrap()
        .with_max_norm(1.0);
    let idx = Tensor::from_storage(TensorStorage::cpu(vec![0.0f32]), vec![1], false).unwrap();
    let out = emb.forward(&idx).unwrap();
    let d = out.data().unwrap();
    let norm = (d[0] * d[0] + d[1] * d[1]).sqrt();
    assert!(
        norm <= 1.0 + 1e-4,
        "max_norm=1.0 not enforced: gathered row norm = {norm} (row {d:?})"
    );
    assert!(
        norm > 0.5,
        "renorm collapsed row to ~0 ({norm}); expected ~1.0 direction-preserving clip"
    );
}

// ===========================================================================
// #1448 FeatureAlphaDropout — entire channels dropped as a unit.
// torch/nn/modules/dropout.py FeatureAlphaDropout.
// ===========================================================================

/// For every (b,c) channel: all spatial positions share the SAME mask
/// decision. Dropped channel -> uniform constant; kept channel -> affine
/// a*x+b with a single (a,b). Per-element dropout would break this.
#[test]
fn feature_alpha_dropout_drops_whole_channels() {
    ferrotorch_core::manual_seed(7).unwrap();
    let c = 8usize;
    let hw = 4usize;
    let mut data = Vec::with_capacity(c * hw);
    for ch in 0..c {
        for s in 0..hw {
            data.push((ch * 10 + s) as f32 + 1.0);
        }
    }
    let x =
        Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![1, c, 2, 2], false).unwrap();
    let mut layer = FeatureAlphaDropout::<f32>::new(0.5).unwrap();
    layer.train();
    let out = layer.forward(&x).unwrap();
    let od = out.data().unwrap();

    for ch in 0..c {
        let base = ch * hw;
        let first = od[base];
        let all_equal = (0..hw).all(|s| (od[base + s] - first).abs() < 1e-4);
        let dx = data[base + 1] - data[base];
        let slope0 = (od[base + 1] - od[base]) / dx;
        let kept_affine = (0..hw - 1).all(|s| {
            let ddx = data[base + s + 1] - data[base + s];
            let slope = (od[base + s + 1] - od[base + s]) / ddx;
            (slope - slope0).abs() < 1e-3
        });
        assert!(
            all_equal || kept_affine,
            "channel {ch} is neither fully-dropped (uniform) nor uniformly-affine: \
             per-element dropout detected. outputs={:?}",
            &od[base..base + hw]
        );
    }
}

// ===========================================================================
// #1452 functional::dropout global RNG determinism under manual_seed.
// ===========================================================================

/// Two dropout calls under the SAME manual_seed produce identical masks
/// (reads the global thread-local generator). Distinct seeds differ.
#[test]
fn functional_dropout_deterministic_under_manual_seed() {
    let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 64]), vec![64], false).unwrap();
    ferrotorch_core::manual_seed(123).unwrap();
    let a = ferrotorch_nn::functional::dropout(&x, 0.5, true)
        .unwrap()
        .data()
        .unwrap()
        .to_vec();
    ferrotorch_core::manual_seed(123).unwrap();
    let b = ferrotorch_nn::functional::dropout(&x, 0.5, true)
        .unwrap()
        .data()
        .unwrap()
        .to_vec();
    assert_eq!(
        a, b,
        "functional::dropout not deterministic under manual_seed"
    );
    ferrotorch_core::manual_seed(999).unwrap();
    let c = ferrotorch_nn::functional::dropout(&x, 0.5, true)
        .unwrap()
        .data()
        .unwrap()
        .to_vec();
    assert_ne!(
        a, c,
        "distinct seeds produced identical masks (RNG not seeded)"
    );
}

// ===========================================================================
// 21e019daf — bias init U(-bound, bound), bound = 1/sqrt(fan_in).
// torch/nn/modules/linear.py:124-128, conv.py:198-201.
// ===========================================================================

/// Linear(100, 10): bound = 1/sqrt(100) = 0.1. Every bias element in
/// [-0.1, 0.1] and NOT all zero (pre-fix it was identically 0).
#[test]
fn linear_bias_init_bounded_and_nonzero() {
    let lin = Linear::<f32>::new(100, 10, true).unwrap();
    let b = lin.bias.as_ref().expect("bias present").data().unwrap();
    let bound = 1.0f32 / 100.0f32.sqrt();
    let mut nonzero = 0;
    for &v in b.iter() {
        assert!(
            v.abs() <= bound + 1e-6,
            "Linear bias {v} exceeds bound {bound} (1/sqrt(fan_in=100))"
        );
        if v.abs() > 1e-9 {
            nonzero += 1;
        }
    }
    assert!(
        nonzero >= b.len() - 1,
        "Linear bias is (nearly) all zeros — uniform init did not take effect: {b:?}"
    );
}

/// Conv2d(in=4, out=6, k=3x3): fan_in = 4*3*3 = 36, bound = 1/6.
#[test]
fn conv2d_bias_init_bounded_and_nonzero() {
    let conv = Conv2d::<f32>::new(4, 6, (3, 3), (1, 1), (0, 0), true).unwrap();
    let named = conv.named_parameters();
    let bias_param = named
        .iter()
        .find(|(n, _)| n == "bias")
        .map(|(_, p)| *p)
        .expect("bias present");
    let b = bias_param.data().unwrap();
    let fan_in = 4.0f32 * 3.0 * 3.0;
    let bound = 1.0f32 / fan_in.sqrt();
    let mut nonzero = 0;
    for &v in b.iter() {
        assert!(
            v.abs() <= bound + 1e-6,
            "Conv2d bias {v} exceeds bound {bound} (1/sqrt(fan_in=36))"
        );
        if v.abs() > 1e-9 {
            nonzero += 1;
        }
    }
    assert!(
        nonzero >= b.len() - 1,
        "Conv2d bias is (nearly) all zeros — uniform init did not take effect: {b:?}"
    );
}

// ===========================================================================
// ADVERSARIAL: Conv2d non-zero padding_mode autograd severance.
// `crate::padding::functional_pad_2d` returns a tensor with
// requires_grad=false (padding.rs:654 `Tensor::from_storage(..., false)`),
// so a Conv2d with padding_mode != Zeros breaks the gradient path back to a
// requires_grad input. Upstream `F.pad` is differentiable
// (torch/nn/modules/conv.py `_conv_forward` keeps the graph intact), so
// `loss.backward()` populates `input.grad`. This test asserts the upstream
// contract: a reflect-padded Conv2d in a training graph yields a NON-None
// gradient on the input.
// ===========================================================================

#[test]
fn conv2d_reflect_padding_preserves_input_autograd() {
    use ferrotorch_core::Tensor as T;
    let mut conv = Conv2d::<f32>::new(1, 1, (3, 3), (1, 1), (1, 1), false)
        .unwrap()
        .with_padding_mode(PaddingMode::Reflect);
    let kernel = vec![1.0f32; 9];
    conv.set_weight(Parameter::from_slice(&kernel, &[1, 1, 3, 3]).unwrap())
        .unwrap();
    let input_data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    // requires_grad=true input.
    let x = T::from_storage(TensorStorage::cpu(input_data), vec![1, 1, 4, 4], true).unwrap();
    let out = conv.forward(&x).unwrap();
    let loss = out.sum_all().unwrap();
    loss.backward().unwrap();
    // Upstream: input.grad is populated (F.pad is differentiable).
    let g = x.grad().unwrap();
    assert!(
        g.is_some(),
        "Conv2d reflect-pad severed the autograd graph: input.grad is None.          functional_pad_2d returns requires_grad=false, so gradients cannot flow          to the input (diverges from torch F.pad which is differentiable)."
    );
}

/// CONTROL: the SAME Conv2d with default (Zeros) padding DOES populate
/// input.grad — proving the severance above is specific to the non-zero
/// padding_mode pre-pad path, not a general Conv2d autograd defect.
#[test]
fn conv2d_zero_padding_preserves_input_autograd_control() {
    use ferrotorch_core::Tensor as T;
    let mut conv = Conv2d::<f32>::new(1, 1, (3, 3), (1, 1), (1, 1), false).unwrap();
    let kernel = vec![1.0f32; 9];
    conv.set_weight(Parameter::from_slice(&kernel, &[1, 1, 3, 3]).unwrap())
        .unwrap();
    let input_data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let x = T::from_storage(TensorStorage::cpu(input_data), vec![1, 1, 4, 4], true).unwrap();
    let out = conv.forward(&x).unwrap();
    out.sum_all().unwrap().backward().unwrap();
    assert!(
        x.grad().unwrap().is_some(),
        "zero-pad Conv2d should populate input.grad (control)"
    );
}

// ---------------------------------------------------------------------------
// Pad adjoint VALUE checks (not just graph connectivity). For loss = sum(pad(x))
// the gradient d(loss)/d(x[s]) equals the multiplicity of source index `s` in
// the pad's gather map (each output element contributes a 1 to its source).
// These are hand-computed from the gather index map, never from ferrotorch.
// ---------------------------------------------------------------------------

/// Reflect pad of a 1x1x1x4 row [1,2,3,4] with left=right=1 maps output cols
/// to source cols [1,0,1,2,3,2] -> multiplicities [1,2,2,1]. Loss=sum(pad)
/// so input.grad must be exactly [[1,2,2,1]] (the reflect adjoint folds the
/// out-of-bounds grad back onto the mirrored interior columns).
#[test]
fn pad2d_reflect_adjoint_values() {
    use ferrotorch_core::Tensor as T;
    use ferrotorch_nn::padding::functional_pad_2d;
    let x = T::from_storage(
        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
        vec![1, 1, 1, 4],
        true,
    )
    .unwrap();
    let padded = functional_pad_2d(&x, 1, 1, 0, 0, PaddingMode::Reflect, 0.0).unwrap();
    padded.sum_all().unwrap().backward().unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("reflect pad severed grad")
        .data()
        .unwrap()
        .to_vec();
    assert_eq!(
        g,
        vec![1.0, 2.0, 2.0, 1.0],
        "reflect adjoint multiplicity wrong: {g:?}"
    );
}

/// Replicate pad left=2,right=1 of [1,2,3,4] maps output cols to sources
/// [0,0,0,1,2,3,3] -> multiplicities [3,1,1,2]. Replicate adjoint sums the
/// replicated edge columns into the boundary element.
#[test]
fn pad2d_replicate_adjoint_values() {
    use ferrotorch_core::Tensor as T;
    use ferrotorch_nn::padding::functional_pad_2d;
    let x = T::from_storage(
        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
        vec![1, 1, 1, 4],
        true,
    )
    .unwrap();
    let padded = functional_pad_2d(&x, 2, 1, 0, 0, PaddingMode::Replicate, 0.0).unwrap();
    padded.sum_all().unwrap().backward().unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("replicate pad severed grad")
        .data()
        .unwrap()
        .to_vec();
    assert_eq!(
        g,
        vec![3.0, 1.0, 1.0, 2.0],
        "replicate adjoint edge-sum wrong: {g:?}"
    );
}

/// Circular pad left=1,right=2 of [1,2,3,4] maps output cols to sources
/// [3,0,1,2,3,0,1] -> multiplicities [2,2,1,2]. Circular adjoint wraps the
/// out-of-bounds grad around to the opposite edge.
#[test]
fn pad2d_circular_adjoint_values() {
    use ferrotorch_core::Tensor as T;
    use ferrotorch_nn::padding::functional_pad_2d;
    let x = T::from_storage(
        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
        vec![1, 1, 1, 4],
        true,
    )
    .unwrap();
    let padded = functional_pad_2d(&x, 1, 2, 0, 0, PaddingMode::Circular, 0.0).unwrap();
    padded.sum_all().unwrap().backward().unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("circular pad severed grad")
        .data()
        .unwrap()
        .to_vec();
    assert_eq!(
        g,
        vec![2.0, 2.0, 1.0, 2.0],
        "circular adjoint wrap wrong: {g:?}"
    );
}

/// Zeros pad adjoint is a pure interior crop: padded columns have no source,
/// so loss=sum(pad) gives input.grad of all-ones (each interior element
/// appears exactly once).
#[test]
fn pad2d_zeros_adjoint_is_crop() {
    use ferrotorch_core::Tensor as T;
    use ferrotorch_nn::padding::functional_pad_2d;
    let x = T::from_storage(
        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
        vec![1, 1, 1, 4],
        true,
    )
    .unwrap();
    let padded = functional_pad_2d(&x, 2, 2, 0, 0, PaddingMode::Zeros, 0.0).unwrap();
    padded.sum_all().unwrap().backward().unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("zeros pad severed grad")
        .data()
        .unwrap()
        .to_vec();
    assert_eq!(
        g,
        vec![1.0, 1.0, 1.0, 1.0],
        "zeros adjoint crop wrong: {g:?}"
    );
}
