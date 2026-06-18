//! #1690 — RNN/GRU/LSTM sequence forward must BATCH the input-to-hidden
//! projection into ONE GEMM across all timesteps (instead of `seq_len`
//! separate `[batch, in]@[in, k*hs]` GEMMs) WITHOUT changing forward values
//! or gradients.
//!
//! The input projection `x_t @ W_ih^T` has no time dependency, so the
//! per-timestep small GEMMs are replaced by a single
//! `[seq_len*batch, in]@[in, k*hs]` GEMM (the stacked sequence projection).
//! Only the recurrent `h @ W_hh^T` (depends on the previous hidden state)
//! stays inside the per-step loop. This mirrors upstream's
//! `FullLayer::operator()` CPU path at `aten/src/ATen/native/RNN.cpp:863-869`
//! (`params.linear_ih(inputs)` over the stacked sequence + per-step
//! `pre_compute_input=true`).
//!
//! The load-bearing concern is autograd correctness: the gradient to
//! `weight_ih` now accumulates through ONE big matmul node whose upstream
//! grad is the concatenation of the per-step grads — it must equal the sum
//! the per-step version produced across `seq_len` separate matmul nodes,
//! NOT be doubled or dropped.
//!
//! Strategy (R-CHAR-3 compliant — no tautologies): the reference re-derives
//! the OLD per-step-projection forward INLINE from the cell gate equations
//! (transpose hoisted as in #1680, but the input projection computed per
//! step — the pre-#1690 code shape). Equivalence in BOTH forward values AND
//! every gradient proves the batched GEMM is value/grad-preserving. The live
//! `torch.nn.{LSTM,GRU,RNN}` parity gate is the sibling test
//! `divergence_rnn_hoist_autograd_reaudit.rs` (torch-2.11 oracle constants),
//! which this build must keep green; this file is the self-consistency probe
//! pinning the batched-vs-per-step reassociation directly.

use ferrotorch_core::grad_fns::activation::{relu, sigmoid, tanh};
use ferrotorch_core::grad_fns::arithmetic::{add, mul, sub};
use ferrotorch_core::grad_fns::linalg::mm_differentiable as mm;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::grad_fns::shape::{cat, expand, reshape, transpose_2d};
use ferrotorch_core::{Tensor, from_slice};
use ferrotorch_nn::{GRU, LSTM, Module, RNN, RNNNonlinearity};
use std::collections::HashMap;

const EPS: f32 = 1e-5;

fn assert_close(a: &[f32], b: &[f32], ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= EPS + EPS * x.abs().max(y.abs()),
            "{ctx}: mismatch at {i}: {x} vs {y} (|d|={})",
            (x - y).abs()
        );
    }
}

fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i as f32) * 0.137 + offset).sin()) * scale)
        .collect()
}

fn rg(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    from_slice(data, shape).unwrap().requires_grad_(true)
}

// ---- per-timestep reference forwards (pre-#1690 shape) ---------------------

/// Reference LSTM forward computing the input projection PER STEP (the
/// pre-#1690 shape; transpose still hoisted, matching #1680). Single layer.
#[allow(clippy::too_many_arguments)]
fn lstm_ref_perstep(
    input: &Tensor<f32>,
    wih: &Tensor<f32>,
    whh: &Tensor<f32>,
    bih: &Tensor<f32>,
    bhh: &Tensor<f32>,
    h0: &Tensor<f32>,
    c0: &Tensor<f32>,
    batch: usize,
    seq_len: usize,
    hs: usize,
) -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
    let wih_t = transpose_2d(wih).unwrap().contiguous().unwrap();
    let whh_t = transpose_2d(whh).unwrap().contiguous().unwrap();
    let bih2 = expand(&bih.unsqueeze_t(0).unwrap(), &[batch, 4 * hs]).unwrap();
    let bhh2 = expand(&bhh.unsqueeze_t(0).unwrap(), &[batch, 4 * hs]).unwrap();
    let mut h = h0.clone();
    let mut c = c0.clone();
    let mut outs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let x_t = input.narrow(1, t, 1).unwrap().squeeze_t(1).unwrap();
        let xw = mm(&x_t, &wih_t).unwrap(); // per-step projection
        let hw = mm(&h, &whh_t).unwrap();
        let gates = add(&add(&add(&xw, &bih2).unwrap(), &hw).unwrap(), &bhh2).unwrap();
        let ch = gates.chunk(4, 1).unwrap();
        let i = sigmoid(&ch[0]).unwrap();
        let f = sigmoid(&ch[1]).unwrap();
        let g = tanh(&ch[2]).unwrap();
        let o = sigmoid(&ch[3]).unwrap();
        let c_new = add(&mul(&f, &c).unwrap(), &mul(&i, &g).unwrap()).unwrap();
        let h_new = mul(&o, &tanh(&c_new).unwrap()).unwrap();
        outs.push(h_new.clone());
        h = h_new;
        c = c_new;
    }
    let output = reshape(
        &cat(&outs, 1).unwrap(),
        &[batch as isize, seq_len as isize, hs as isize],
    )
    .unwrap();
    let h_n = reshape(&h, &[1, batch as isize, hs as isize]).unwrap();
    let c_n = reshape(&c, &[1, batch as isize, hs as isize]).unwrap();
    (output, h_n, c_n)
}

/// Reference GRU forward computing the input projection PER STEP.
#[allow(clippy::too_many_arguments)]
fn gru_ref_perstep(
    input: &Tensor<f32>,
    wih: &Tensor<f32>,
    whh: &Tensor<f32>,
    bih: &Tensor<f32>,
    bhh: &Tensor<f32>,
    h0: &Tensor<f32>,
    batch: usize,
    seq_len: usize,
    hs: usize,
) -> (Tensor<f32>, Tensor<f32>) {
    let wih_t = transpose_2d(wih).unwrap().contiguous().unwrap();
    let whh_t = transpose_2d(whh).unwrap().contiguous().unwrap();
    let bih2 = expand(&bih.unsqueeze_t(0).unwrap(), &[batch, 3 * hs]).unwrap();
    let bhh2 = expand(&bhh.unsqueeze_t(0).unwrap(), &[batch, 3 * hs]).unwrap();
    let mut h = h0.clone();
    let mut outs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let x_t = input.narrow(1, t, 1).unwrap().squeeze_t(1).unwrap();
        let xw = mm(&x_t, &wih_t).unwrap();
        let hw = mm(&h, &whh_t).unwrap();
        let xwb = add(&xw, &bih2).unwrap();
        let hwb = add(&hw, &bhh2).unwrap();
        let xc = xwb.chunk(3, 1).unwrap();
        let hc = hwb.chunk(3, 1).unwrap();
        let r = sigmoid(&add(&xc[0], &hc[0]).unwrap()).unwrap();
        let z = sigmoid(&add(&xc[1], &hc[1]).unwrap()).unwrap();
        let n = tanh(&add(&xc[2], &mul(&r, &hc[2]).unwrap()).unwrap()).unwrap();
        let h_new = add(&n, &mul(&z, &sub(&h, &n).unwrap()).unwrap()).unwrap();
        outs.push(h_new.clone());
        h = h_new;
    }
    let output = reshape(
        &cat(&outs, 1).unwrap(),
        &[batch as isize, seq_len as isize, hs as isize],
    )
    .unwrap();
    let h_n = reshape(&h, &[1, batch as isize, hs as isize]).unwrap();
    (output, h_n)
}

/// Reference RNN forward computing the input projection PER STEP.
#[allow(clippy::too_many_arguments)]
fn rnn_ref_perstep(
    input: &Tensor<f32>,
    wih: &Tensor<f32>,
    whh: &Tensor<f32>,
    bih: &Tensor<f32>,
    bhh: &Tensor<f32>,
    h0: &Tensor<f32>,
    nonlin: RNNNonlinearity,
    batch: usize,
    seq_len: usize,
    hs: usize,
) -> (Tensor<f32>, Tensor<f32>) {
    let wih_t = transpose_2d(wih).unwrap().contiguous().unwrap();
    let whh_t = transpose_2d(whh).unwrap().contiguous().unwrap();
    let bih2 = expand(&bih.unsqueeze_t(0).unwrap(), &[batch, hs]).unwrap();
    let bhh2 = expand(&bhh.unsqueeze_t(0).unwrap(), &[batch, hs]).unwrap();
    let mut h = h0.clone();
    let mut outs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let x_t = input.narrow(1, t, 1).unwrap().squeeze_t(1).unwrap();
        let xw = mm(&x_t, &wih_t).unwrap();
        let hw = mm(&h, &whh_t).unwrap();
        let pre = add(&add(&add(&xw, &bih2).unwrap(), &hw).unwrap(), &bhh2).unwrap();
        let h_new = match nonlin {
            RNNNonlinearity::Tanh => tanh(&pre).unwrap(),
            RNNNonlinearity::ReLU => relu(&pre).unwrap(),
        };
        outs.push(h_new.clone());
        h = h_new;
    }
    let output = reshape(
        &cat(&outs, 1).unwrap(),
        &[batch as isize, seq_len as isize, hs as isize],
    )
    .unwrap();
    let h_n = reshape(&h, &[1, batch as isize, hs as isize]).unwrap();
    (output, h_n)
}

fn sd1(prefix: &[(&str, Tensor<f32>)]) -> HashMap<String, Tensor<f32>> {
    prefix
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

// ---- LSTM: batched-projection production == per-step reference -------------

#[test]
fn lstm_batched_projection_matches_perstep_fwd_bwd() {
    let (batch, seq_len, in_size, hs) = (3usize, 6usize, 4usize, 5usize);
    let wih = ramp(4 * hs * in_size, 0.3, 0.0);
    let whh = ramp(4 * hs * hs, 0.3, 1.7);
    let bih = ramp(4 * hs, 0.1, 0.5);
    let bhh = ramp(4 * hs, 0.1, 2.3);
    let xdata = ramp(batch * seq_len * in_size, 0.5, 0.9);
    let h0d = ramp(batch * hs, 0.2, 3.1);
    let c0d = ramp(batch * hs, 0.2, 4.4);

    let mut lstm = LSTM::<f32>::new(in_size, hs, 1).unwrap();
    lstm.load_state_dict(
        &sd1(&[
            (
                "layers.0.weight_ih",
                from_slice(&wih, &[4 * hs, in_size]).unwrap(),
            ),
            (
                "layers.0.weight_hh",
                from_slice(&whh, &[4 * hs, hs]).unwrap(),
            ),
            ("layers.0.bias_ih", from_slice(&bih, &[4 * hs]).unwrap()),
            ("layers.0.bias_hh", from_slice(&bhh, &[4 * hs]).unwrap()),
        ]),
        true,
    )
    .unwrap();

    let xp = rg(&xdata, &[batch, seq_len, in_size]);
    let h0p = from_slice(&h0d, &[1, batch, hs]).unwrap();
    let c0p = from_slice(&c0d, &[1, batch, hs]).unwrap();
    let (op, (hp, cp)) = lstm.forward_with_state(&xp, Some((&h0p, &c0p))).unwrap();

    let wr = rg(&wih, &[4 * hs, in_size]);
    let hr = rg(&whh, &[4 * hs, hs]);
    let br = rg(&bih, &[4 * hs]);
    let bhr = rg(&bhh, &[4 * hs]);
    let xr = rg(&xdata, &[batch, seq_len, in_size]);
    let h0r = from_slice(&h0d, &[batch, hs]).unwrap();
    let c0r = from_slice(&c0d, &[batch, hs]).unwrap();
    let (or_, hr_, cr_) =
        lstm_ref_perstep(&xr, &wr, &hr, &br, &bhr, &h0r, &c0r, batch, seq_len, hs);

    assert_close(op.data().unwrap(), or_.data().unwrap(), "LSTM out");
    assert_close(hp.data().unwrap(), hr_.data().unwrap(), "LSTM h_n");
    assert_close(cp.data().unwrap(), cr_.data().unwrap(), "LSTM c_n");

    sum(&op).unwrap().backward().unwrap();
    sum(&or_).unwrap().backward().unwrap();
    let p = lstm.parameters();
    assert_close(
        p[0].tensor()
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        wr.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        "LSTM wih.grad",
    );
    assert_close(
        p[1].tensor()
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        hr.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        "LSTM whh.grad",
    );
    assert_close(
        p[2].tensor()
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        br.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        "LSTM bih.grad",
    );
    assert_close(
        p[3].tensor()
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        bhr.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        "LSTM bhh.grad",
    );
    assert_close(
        xp.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        xr.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        "LSTM input.grad",
    );
}

// ---- GRU: batched-projection production == per-step reference --------------

#[test]
fn gru_batched_projection_matches_perstep_fwd_bwd() {
    let (batch, seq_len, in_size, hs) = (2usize, 7usize, 4usize, 5usize);
    let wih = ramp(3 * hs * in_size, 0.3, 0.0);
    let whh = ramp(3 * hs * hs, 0.3, 1.7);
    let bih = ramp(3 * hs, 0.1, 0.5);
    let bhh = ramp(3 * hs, 0.1, 2.3);
    let xdata = ramp(batch * seq_len * in_size, 0.5, 0.9);
    let h0d = ramp(batch * hs, 0.2, 3.1);

    let mut gru = GRU::<f32>::new(in_size, hs).unwrap();
    gru.load_state_dict(
        &sd1(&[
            (
                "layers.0.weight_ih",
                from_slice(&wih, &[3 * hs, in_size]).unwrap(),
            ),
            (
                "layers.0.weight_hh",
                from_slice(&whh, &[3 * hs, hs]).unwrap(),
            ),
            ("layers.0.bias_ih", from_slice(&bih, &[3 * hs]).unwrap()),
            ("layers.0.bias_hh", from_slice(&bhh, &[3 * hs]).unwrap()),
        ]),
        true,
    )
    .unwrap();

    let xp = rg(&xdata, &[batch, seq_len, in_size]);
    let h0p = from_slice(&h0d, &[1, batch, hs]).unwrap();
    let (op, hp) = gru.forward(&xp, Some(&h0p)).unwrap();

    let wr = rg(&wih, &[3 * hs, in_size]);
    let hr = rg(&whh, &[3 * hs, hs]);
    let br = rg(&bih, &[3 * hs]);
    let bhr = rg(&bhh, &[3 * hs]);
    let xr = rg(&xdata, &[batch, seq_len, in_size]);
    let h0r = from_slice(&h0d, &[batch, hs]).unwrap();
    let (or_, hr_) = gru_ref_perstep(&xr, &wr, &hr, &br, &bhr, &h0r, batch, seq_len, hs);

    assert_close(op.data().unwrap(), or_.data().unwrap(), "GRU out");
    assert_close(hp.data().unwrap(), hr_.data().unwrap(), "GRU h_n");

    sum(&op).unwrap().backward().unwrap();
    sum(&or_).unwrap().backward().unwrap();
    let p = gru.parameters();
    assert_close(
        p[0].tensor()
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        wr.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        "GRU wih.grad",
    );
    assert_close(
        p[1].tensor()
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        hr.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        "GRU whh.grad",
    );
    assert_close(
        p[2].tensor()
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        br.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        "GRU bih.grad",
    );
    assert_close(
        p[3].tensor()
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        bhr.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        "GRU bhh.grad",
    );
    assert_close(
        xp.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        xr.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        "GRU input.grad",
    );
}

// ---- RNN (tanh + relu): batched-projection production == per-step ----------

fn run_rnn_batched_eq(nonlin: RNNNonlinearity, name: &str) {
    let (batch, seq_len, in_size, hs) = (2usize, 7usize, 4usize, 5usize);
    let wih = ramp(hs * in_size, 0.3, 0.0);
    let whh = ramp(hs * hs, 0.3, 1.7);
    let bih = ramp(hs, 0.1, 0.5);
    let bhh = ramp(hs, 0.1, 2.3);
    let xdata = ramp(batch * seq_len * in_size, 0.5, 0.9);
    let h0d = ramp(batch * hs, 0.2, 3.1);

    let mut rnn = RNN::<f32>::with_options(in_size, hs, 1, nonlin).unwrap();
    rnn.load_state_dict(
        &sd1(&[
            (
                "layers.0.weight_ih",
                from_slice(&wih, &[hs, in_size]).unwrap(),
            ),
            ("layers.0.weight_hh", from_slice(&whh, &[hs, hs]).unwrap()),
            ("layers.0.bias_ih", from_slice(&bih, &[hs]).unwrap()),
            ("layers.0.bias_hh", from_slice(&bhh, &[hs]).unwrap()),
        ]),
        true,
    )
    .unwrap();

    let xp = rg(&xdata, &[batch, seq_len, in_size]);
    let h0p = from_slice(&h0d, &[1, batch, hs]).unwrap();
    let (op, hp) = rnn.forward_with_state(&xp, Some(&h0p)).unwrap();

    let wr = rg(&wih, &[hs, in_size]);
    let hr = rg(&whh, &[hs, hs]);
    let br = rg(&bih, &[hs]);
    let bhr = rg(&bhh, &[hs]);
    let xr = rg(&xdata, &[batch, seq_len, in_size]);
    let h0r = from_slice(&h0d, &[batch, hs]).unwrap();
    let (or_, hr_) = rnn_ref_perstep(&xr, &wr, &hr, &br, &bhr, &h0r, nonlin, batch, seq_len, hs);

    assert_close(
        op.data().unwrap(),
        or_.data().unwrap(),
        &format!("{name} out"),
    );
    assert_close(
        hp.data().unwrap(),
        hr_.data().unwrap(),
        &format!("{name} h_n"),
    );

    sum(&op).unwrap().backward().unwrap();
    sum(&or_).unwrap().backward().unwrap();
    let p = rnn.parameters();
    assert_close(
        p[0].tensor()
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        wr.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        &format!("{name} wih.grad"),
    );
    assert_close(
        p[1].tensor()
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        hr.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        &format!("{name} whh.grad"),
    );
    assert_close(
        p[2].tensor()
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        br.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        &format!("{name} bih.grad"),
    );
    assert_close(
        p[3].tensor()
            .grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        bhr.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        &format!("{name} bhh.grad"),
    );
    assert_close(
        xp.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        xr.grad()
            .unwrap()
            .expect("gradient should be populated")
            .data()
            .unwrap(),
        &format!("{name} input.grad"),
    );
}

#[test]
fn rnn_tanh_batched_projection_matches_perstep_fwd_bwd() {
    run_rnn_batched_eq(RNNNonlinearity::Tanh, "RNN(tanh)");
}

#[test]
fn rnn_relu_batched_projection_matches_perstep_fwd_bwd() {
    run_rnn_batched_eq(RNNNonlinearity::ReLU, "RNN(relu)");
}

/// Timing probe (ignored): batched-input-projection production GRU vs the
/// per-step-projection reference at the #1690 target shape
/// (in=128, hidden=256, seq=32, B=16). Both transpose-hoisted; the ONLY
/// difference is per-step vs batched input projection, so this isolates the
/// #1690 win. Run with `--ignored --nocapture` in release.
#[test]
#[ignore = "perf measurement; run with --ignored --nocapture in release"]
fn gru_batched_projection_timing_probe() {
    use std::time::Instant;
    let (batch, seq_len, in_size, hs) = (16usize, 32usize, 128usize, 256usize);
    let wih = ramp(3 * hs * in_size, 0.02, 0.0);
    let whh = ramp(3 * hs * hs, 0.02, 1.0);
    let bih = ramp(3 * hs, 0.01, 0.5);
    let bhh = ramp(3 * hs, 0.01, 2.0);
    let xdata = ramp(batch * seq_len * in_size, 0.05, 0.3);

    let mut gru = GRU::<f32>::new(in_size, hs).unwrap();
    gru.load_state_dict(
        &sd1(&[
            (
                "layers.0.weight_ih",
                from_slice(&wih, &[3 * hs, in_size]).unwrap(),
            ),
            (
                "layers.0.weight_hh",
                from_slice(&whh, &[3 * hs, hs]).unwrap(),
            ),
            ("layers.0.bias_ih", from_slice(&bih, &[3 * hs]).unwrap()),
            ("layers.0.bias_hh", from_slice(&bhh, &[3 * hs]).unwrap()),
        ]),
        true,
    )
    .unwrap();
    let x = from_slice(&xdata, &[batch, seq_len, in_size]).unwrap();

    let wr = from_slice(&wih, &[3 * hs, in_size]).unwrap();
    let hr = from_slice(&whh, &[3 * hs, hs]).unwrap();
    let br = from_slice(&bih, &[3 * hs]).unwrap();
    let bhr = from_slice(&bhh, &[3 * hs]).unwrap();
    let xr = from_slice(&xdata, &[batch, seq_len, in_size]).unwrap();
    let h0r = from_slice(&vec![0.0f32; batch * hs], &[batch, hs]).unwrap();

    let iters = 20;
    let _ = gru.forward(&x, None).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = gru.forward(&x, None).unwrap();
    }
    let batched = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;

    let _ = gru_ref_perstep(&xr, &wr, &hr, &br, &bhr, &h0r, batch, seq_len, hs);
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = gru_ref_perstep(&xr, &wr, &hr, &br, &bhr, &h0r, batch, seq_len, hs);
    }
    let perstep = t1.elapsed().as_secs_f64() / iters as f64 * 1e6;

    println!(
        "GRU [{in_size}->{hs}, seq={seq_len}, B={batch}] forward: batched-projection={batched:.0}us  per-step-projection={perstep:.0}us  (speedup {:.2}x)",
        perstep / batched
    );
}
