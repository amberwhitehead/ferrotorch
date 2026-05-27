//! Wave-E audit (#1542): #1468 Rprop CUDA fallback opt-in.

#![allow(clippy::approx_constant)]

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_nn::Parameter;
use ferrotorch_optim::{Optimizer, Rprop, RpropConfig};

/// Default config must keep `cpu_fallback = false` so R-CODE-4 holds: no
/// silent CUDA → CPU demotion.
#[test]
fn audit_1468_default_does_not_silently_fall_back() {
    let cfg = RpropConfig::default();
    assert!(!cfg.cpu_fallback, "default must be fail-fast on CUDA");
}

/// Builder method `with_cpu_fallback(true)` flips the flag.
#[test]
fn audit_1468_with_cpu_fallback_builder_flips_field() {
    let cfg = RpropConfig::default().with_cpu_fallback(true);
    assert!(cfg.cpu_fallback);
}

fn make_param() -> Parameter<f64> {
    let t =
        Tensor::<f64>::from_storage(TensorStorage::cpu(vec![5.0_f64]), vec![1], true).unwrap();
    Parameter::new(t)
}

fn set_grad(p: &Parameter<f64>, val: f64) {
    let g = Tensor::<f64>::from_storage(TensorStorage::cpu(vec![val]), vec![1], false).unwrap();
    p.tensor().set_grad(Some(g)).unwrap();
}

/// CPU step with `cpu_fallback=true` must succeed. The flag is a no-op on
/// CPU tensors; this exercises the new branch in `step()` to ensure the
/// builder doesn't accidentally break the CPU path.
#[test]
fn audit_1468_cpu_step_with_fallback_enabled_succeeds() {
    let p = make_param();
    set_grad(&p, 1.0);
    let mut opt = Rprop::new(vec![p], RpropConfig::default().with_cpu_fallback(true));
    opt.step().expect("CPU step with cpu_fallback=true must succeed");
}

/// `cpu_fallback=false` (default) and a CPU param also succeeds.
#[test]
fn audit_1468_cpu_step_with_fallback_disabled_still_works() {
    let p = make_param();
    set_grad(&p, 1.0);
    let mut opt = Rprop::new(vec![p], RpropConfig::default());
    opt.step()
        .expect("CPU step must succeed regardless of cpu_fallback flag");
}

/// Read the production source for `Rprop::step` and confirm the field is
/// actually consulted (not just defined and never read). A vocab-only
/// "added a field, never read it" implementation would leave the source
/// without a `self.config.cpu_fallback` reference in the step body.
#[test]
fn audit_1468_step_reads_cpu_fallback_field() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/rprop.rs"))
        .expect("read rprop.rs");
    assert!(
        src.contains("self.config.cpu_fallback"),
        "rprop.rs must reference `self.config.cpu_fallback` in `step()` — \
         otherwise the field is vocab-only and the CUDA branch never \
         consults it (#1468)"
    );
    assert!(
        src.contains("tensor.is_cuda() && !self.config.cpu_fallback")
            || src.contains("!self.config.cpu_fallback && tensor.is_cuda()"),
        "the cuda fail-fast gate must consult cpu_fallback"
    );
}
