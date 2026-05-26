//! Critic audit of the misc-small-blockers Dispatch B (#1453, #1464-#1468,
//! #1481, #1482, #1494-#1496; tracking #1542).
//!
//! The dispatch claims (in working-tree diffs):
//!
//! - #1453 Kaiming fan_out mode (ferrotorch-nn/src/init.rs)
//! - #1464 Muon non-2D fallback / strict-2D rejection (muon.rs)
//! - #1465 Muon custom Newton-Schulz coefficients (muon.rs)
//! - #1466 Muon decoupled weight decay (muon.rs)
//! - #1468 Rprop CUDA fallback path (rprop.rs)
//!
//! The blockers #1481, #1482, #1494, #1495, #1496 had no diff and are
//! NOT-TOUCHED — this audit file does not probe them.
//!
//! Each probe is the smallest observable test that distinguishes
//! GENUINELY-WIRED from VOCAB-ONLY. Tests run against the CPU path; CUDA
//! probes are gated under `--features cuda`.

use ferrotorch_nn::init::{
    self, kaiming_normal_with_fan_mode, kaiming_uniform_with_fan_mode, FanMode, NonLinearity,
};
use ferrotorch_nn::Parameter;
use ferrotorch_optim::muon::{Muon, MuonConfig};
use ferrotorch_optim::optimizer::Optimizer;
use ferrotorch_optim::rprop::{Rprop, RpropConfig};

// ---------------------------------------------------------------------------
// #1453 Kaiming fan_out
// Observable: kaiming_normal_with_fan_mode(FanOut) on [out=64, in=128] must
// produce variance approximately gain^2 / fan_out = 2 / 64 = 0.03125, NOT
// 2 / 128 = 0.015625 (the fan_in baseline).
// ---------------------------------------------------------------------------

#[test]
fn audit_1453_kaiming_normal_fan_out_variance() {
    let mut p = Parameter::<f64>::zeros(&[64, 128]).unwrap();
    kaiming_normal_with_fan_mode(&mut p, NonLinearity::ReLU, FanMode::FanOut).unwrap();
    let data = p.data().unwrap();
    let n = data.len() as f64;
    let mean: f64 = data.iter().sum::<f64>() / n;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

    let expected_var = 2.0 / 64.0;
    let alt_fan_in_var = 2.0 / 128.0;
    assert!(
        (var - expected_var).abs() / expected_var < 0.25,
        "#1453 fan_out variance must be ≈ 2/fan_out = {expected_var:.6}; got {var:.6} \
         (fan_in baseline would be {alt_fan_in_var:.6})"
    );
    assert!(
        (var - expected_var).abs() < (var - alt_fan_in_var).abs(),
        "#1453 variance must be closer to fan_out value {expected_var} than \
         fan_in value {alt_fan_in_var}; got {var}"
    );
}

#[test]
fn audit_1453_kaiming_uniform_fan_out_bound() {
    let mut p = Parameter::<f64>::zeros(&[64, 128]).unwrap();
    kaiming_uniform_with_fan_mode(&mut p, NonLinearity::ReLU, FanMode::FanOut).unwrap();
    let data = p.data().unwrap();
    let limit_fan_out = (6.0_f64 / 64.0).sqrt();
    let max_abs: f64 = data.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    assert!(
        max_abs <= limit_fan_out + 1e-6,
        "#1453 fan_out uniform must respect bound {limit_fan_out}; got max |x| = {max_abs}"
    );
}

// ---------------------------------------------------------------------------
// #1464 Muon non-2D fallback (default) + strict-2D rejection (opt-in).
// ---------------------------------------------------------------------------

#[test]
fn audit_1464_muon_default_1d_falls_back_to_sgd() {
    let p = Parameter::from_slice(&[10.0_f64, 10.0], &[2]).unwrap();
    let grad = ferrotorch_core::Tensor::from_storage(
        ferrotorch_core::TensorStorage::cpu(vec![1.0_f64, 1.0]),
        vec![2],
        false,
    )
    .unwrap();
    p.set_grad(Some(grad)).unwrap();

    let cfg = MuonConfig::new(0.1).momentum(0.0).nesterov(false);
    let mut muon = Muon::new(vec![p], cfg);
    let result = muon.step();
    assert!(
        result.is_ok(),
        "#1464 default Muon::step on 1D param must succeed (SGD fallback); got {:?}",
        result.err()
    );
    let data = muon.param_groups()[0].params()[0].data().unwrap().to_vec();
    // SGD with lr=0.1, grad=1.0: 10.0 - 0.1*1.0 = 9.9.
    assert!(
        (data[0] - 9.9).abs() < 1e-6,
        "#1464 SGD fallback must produce 9.9; got {}",
        data[0]
    );
}

#[test]
fn audit_1464_muon_strict_2d_rejects_1d() {
    let p = Parameter::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    let cfg = MuonConfig::new(0.02);
    let result = Muon::<f64>::new_strict_2d(vec![p], cfg);
    assert!(
        result.is_err(),
        "#1464 Muon::new_strict_2d must reject 1D parameter"
    );
    let msg = format!("{}", result.err().unwrap());
    assert!(
        msg.contains("Muon only supports 2D parameters"),
        "#1464 strict_2d error message must name upstream constraint; got {msg}"
    );
}

// ---------------------------------------------------------------------------
// #1465 Muon custom Newton-Schulz coefficients.
// ---------------------------------------------------------------------------

#[test]
fn audit_1465_quintic_config_stored() {
    let cfg = MuonConfig::new(0.01).with_ns_coefficients((3.4445, -4.7750, 2.0315));
    assert_eq!(
        cfg.ns_coefficients,
        (3.4445, -4.7750, 2.0315),
        "#1465 quintic coefficients must round-trip through MuonConfig"
    );
}

#[test]
fn audit_1465_quintic_differs_from_cubic_step() {
    let make_param = || {
        let p = Parameter::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let g = ferrotorch_core::Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![2.0_f64, 0.5, 0.5, 2.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        p.set_grad(Some(g)).unwrap();
        p
    };

    let p_cubic = make_param();
    let cfg_cubic = MuonConfig::new(0.1).momentum(0.0).nesterov(false).ns_steps(5);
    let mut m_cubic = Muon::new(vec![p_cubic], cfg_cubic);
    m_cubic.step().unwrap();
    let cubic_out = m_cubic.param_groups()[0].params()[0]
        .data()
        .unwrap()
        .to_vec();

    let p_quint = make_param();
    let cfg_quint = MuonConfig::new(0.1)
        .momentum(0.0)
        .nesterov(false)
        .ns_steps(5)
        .with_ns_coefficients((3.4445, -4.7750, 2.0315));
    let mut m_quint = Muon::new(vec![p_quint], cfg_quint);
    m_quint.step().unwrap();
    let quint_out = m_quint.param_groups()[0].params()[0]
        .data()
        .unwrap()
        .to_vec();

    let any_diff = cubic_out
        .iter()
        .zip(quint_out.iter())
        .any(|(c, q)| (c - q).abs() > 1e-4);
    assert!(
        any_diff,
        "#1465 quintic NS must produce post-step output distinct from cubic; \
         cubic={:?} quint={:?}",
        cubic_out, quint_out
    );
}

// ---------------------------------------------------------------------------
// #1466 Muon decoupled weight decay.
// ---------------------------------------------------------------------------

#[test]
fn audit_1466_decoupled_wd_differs_from_l2_under_nonzero_grad() {
    let make_param = || {
        let p = Parameter::from_slice(&[3.0_f64, 1.0, 1.0, 3.0], &[2, 2]).unwrap();
        let g = ferrotorch_core::Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![2.0_f64, 0.5, 0.5, 2.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        p.set_grad(Some(g)).unwrap();
        p
    };

    let cfg_l2 = MuonConfig::new(0.1)
        .momentum(0.0)
        .nesterov(false)
        .ns_steps(5)
        .with_weight_decay(0.5);
    let mut m_l2 = Muon::new(vec![make_param()], cfg_l2);
    m_l2.step().unwrap();
    let l2_out = m_l2.param_groups()[0].params()[0].data().unwrap().to_vec();

    let cfg_dec = MuonConfig::new(0.1)
        .momentum(0.0)
        .nesterov(false)
        .ns_steps(5)
        .with_weight_decay(0.5)
        .with_decoupled_weight_decay(true);
    let mut m_dec = Muon::new(vec![make_param()], cfg_dec);
    m_dec.step().unwrap();
    let dec_out = m_dec.param_groups()[0].params()[0].data().unwrap().to_vec();

    let any_diff = l2_out
        .iter()
        .zip(dec_out.iter())
        .any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(
        any_diff,
        "#1466 decoupled-WD must produce post-step output distinct from L2 \
         (2-D + non-zero grad); l2={:?} dec={:?}",
        l2_out, dec_out
    );
}

/// #1466: with decoupled=true, grad=0, the param must shrink by exactly
/// (1 - lr*wd).
#[test]
fn audit_1466_decoupled_zero_grad_shrinks_by_factor() {
    let p = Parameter::from_slice(&[10.0_f64], &[1]).unwrap();
    let g = ferrotorch_core::Tensor::from_storage(
        ferrotorch_core::TensorStorage::cpu(vec![0.0_f64]),
        vec![1],
        false,
    )
    .unwrap();
    p.set_grad(Some(g)).unwrap();

    let cfg = MuonConfig::new(0.1)
        .momentum(0.0)
        .nesterov(false)
        .with_weight_decay(0.5)
        .with_decoupled_weight_decay(true);
    let mut muon = Muon::new(vec![p], cfg);
    muon.step().unwrap();

    let data = muon.param_groups()[0].params()[0].data().unwrap().to_vec();
    // Expected: 10 * (1 - 0.1 * 0.5) = 9.5.
    assert!(
        (data[0] - 9.5).abs() < 1e-9,
        "#1466 decoupled wd zero-grad must yield 9.5; got {}",
        data[0]
    );
}

// ---------------------------------------------------------------------------
// #1468 Rprop CUDA fallback path.
//
// Without a CUDA device we can't exercise the actual fallback, but we CAN
// observe that the `step()` body consults the `cpu_fallback` flag. The
// current source has an UNCONDITIONAL `if tensor.is_cuda() { return Err(...) }`
// guard with NO `if self.config.cpu_fallback` branch — the new field is
// parsed by the builder but never read in the step body. That makes #1468
// VOCAB-ONLY.
// ---------------------------------------------------------------------------

#[test]
fn audit_1468_cpu_fallback_config_stores_true() {
    let cfg = RpropConfig::default().with_cpu_fallback(true);
    assert!(cfg.cpu_fallback);
}

/// #1468: the dispatch claims the cpu_fallback path is wired. Probe the
/// source: `rprop.rs::step` must reference `cpu_fallback` somewhere in its
/// step body (gating the fail-fast). If it doesn't, the field is parsed
/// but unused — vocab-only.
#[test]
fn audit_1468_cpu_fallback_consulted_in_step() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/rprop.rs"),
    )
    .expect("rprop.rs must be readable");

    let step_start = src.find("fn step(").expect("rprop.rs must have fn step");
    let step_end = src[step_start..]
        .find("fn zero_grad")
        .map(|o| step_start + o)
        .unwrap_or(src.len());
    let step_body = &src[step_start..step_end];

    assert!(
        step_body.contains("cpu_fallback"),
        "#1468 Rprop::step body does not consult self.config.cpu_fallback — \
         the field is parsed but the fallback branch is never executed. \
         step body excerpt:\n{}",
        &step_body[..step_body.len().min(800)]
    );
}

/// #1468: a CPU-only Rprop step with cpu_fallback=true must succeed
/// (sanity — the flag should not break the CPU happy path).
#[test]
fn audit_1468_cpu_fallback_true_does_not_break_cpu_path() {
    let p = Parameter::from_slice(&[5.0_f64], &[1]).unwrap();
    let g = ferrotorch_core::Tensor::from_storage(
        ferrotorch_core::TensorStorage::cpu(vec![1.0_f64]),
        vec![1],
        false,
    )
    .unwrap();
    p.set_grad(Some(g)).unwrap();

    let cfg = RpropConfig::default().with_cpu_fallback(true);
    let mut opt = Rprop::new(vec![p], cfg);
    let result = opt.step();
    assert!(
        result.is_ok(),
        "#1468 CPU rprop step with cpu_fallback=true must succeed; got {:?}",
        result.err()
    );
}

// ---------------------------------------------------------------------------
// Surface sanity
// ---------------------------------------------------------------------------

#[test]
fn audit_surface_kaiming_fan_mode_default() {
    assert_eq!(FanMode::default(), FanMode::FanIn);
}

#[test]
fn audit_surface_init_module_exports() {
    let _ = init::NonLinearity::ReLU;
}
