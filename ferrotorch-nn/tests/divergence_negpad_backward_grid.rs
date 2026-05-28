//! DEFINITIVE FINAL close-audit of the negative-pad chain BACKWARD
//! (#1611/#1620/#1621/#1623/#1624/#1625/#1626/#1627/#1628/#1629/#1631) in
//! `ferrotorch-nn/src/padding.rs`, targeting commit `bf998db9f`
//! (`circular_slicecopy_backward_block` — the autograd transpose of the #1629
//! forward slice-copy).
//!
//! The FORWARD chain is already proven byte-perfect vs live torch by
//! `divergence_negpad_det_reaudit.rs::definitive_negpad_det_grid_all_modes`
//! (deterministic cold-fork + additive-shift gather oracle, 0 across all 4
//! modes). THIS harness independently re-proves the BACKWARD over the SAME
//! deterministic grid.
//!
//! METHOD (R-CHAR-3): the exhaustive reference grid — torch `x.grad` for every
//! ACCEPTED, DEFINED forward case — is produced by LIVE torch 2.11.0+cu130 via
//! `tests/fixtures_pad_grid_backward_gen.py`, spawned at test time. For each
//! defined case we run, under `catch_unwind` (a backward PANIC is an automatic
//! FAIL, R-CODE-2):
//!   (1) `sum(pad(x)).backward()` and compare `x.grad` element-by-element to
//!       torch's all-ones-seed VJP grad;
//!   (2) a NON-UNIFORM seeded grad_output VJP (`backward_with_grad(y, seed)`
//!       with the oracle's distinct-per-cell ramp seed) and compare `x.grad`
//!       to torch's ramp-seed VJP grad. The non-uniform seed makes a
//!       mis-weighted / mis-routed scatter detectable — under the all-ones seed
//!       two distinct sources read the same number of times are
//!       indistinguishable; under a distinct-per-cell ramp the exact set of
//!       output cells routing into each input cell is pinned (the corner-
//!       written-more-than-once multi-write accumulation of the circular
//!       transpose, #1631).
//!
//! Only DEFINED cases are asserted on. Circular over-crop / net-zero reads of
//! `new_empty` uninitialized memory (`PadNd.cpp:148`) carry `garbage_det=True`
//! from the cold-fork + additive-shift classifier and are skipped (ferro Errs
//! the forward on them, so backward is never reached) — but a PANIC on ANY case
//! (defined or not) is still a hard FAIL.
//!
//! Upstream contract mirrored: torch autograd's transpose of `_pad_circular`
//! (`aten/src/ATen/native/PadNd.cpp:148-187`): `new_empty` + `slice` + `copy_`
//! composition; the `copy_`-overwrites-destination adjoint for the LIVE-`out`
//! wrap reads (`:176-179`). reflect/replicate/constant backward is the gather
//! adjoint scatter-add (`PaddingKernel.cpp` index map transpose).
//!
//! VERDICT GATES (all must hold for the negative-pad chain BACKWARD to close):
//!   - grad_sum_mismatch  == 0   (defined cases: ferro sum() grad == torch)
//!   - grad_seed_mismatch == 0   (defined cases: ferro ramp-seed grad == torch)
//!   - grad_panic         == 0   (R-CODE-2: never a panic on any signed pad)
//!   - grad_err           == 0   (defined cases: ferro must produce a grad)

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::Command;

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::padding::{PaddingMode, functional_pad_1d_signed, functional_pad_2d_signed};

use serde_json::Value;

const TOL: f64 = 1e-9;

fn mode_of(s: &str) -> PaddingMode {
    match s {
        "constant" => PaddingMode::Zeros,
        "reflect" => PaddingMode::Reflect,
        "replicate" => PaddingMode::Replicate,
        "circular" => PaddingMode::Circular,
        other => panic!("unknown mode {other}"),
    }
}

fn tensor(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn vnums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}
fn vusize(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}
fn vsigned(v: &Value) -> Vec<isize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as isize)
        .collect()
}

/// One backward outcome, with a PANIC caught as a distinct outcome so it is
/// reported (R-CODE-2) rather than aborting the binary.
enum GradOut {
    Grad(Vec<f64>),
    Err,
    Panic,
}

fn forward(
    rank: usize,
    x: &Tensor<f64>,
    pads: &[isize],
    mode: PaddingMode,
) -> Result<Tensor<f64>, ()> {
    match rank {
        1 => functional_pad_1d_signed(x, pads[0], pads[1], mode, 0.0).map_err(|_| ()),
        2 => functional_pad_2d_signed(x, pads[0], pads[1], pads[2], pads[3], mode, 0.0)
            .map_err(|_| ()),
        other => panic!("unsupported rank {other}"),
    }
}

/// ferro `x.grad` from `sum(pad(x)).backward()` (all-ones seed VJP).
fn grad_sum(
    rank: usize,
    in_data: &[f64],
    in_shape: &[usize],
    pads: &[isize],
    mode: PaddingMode,
) -> GradOut {
    let res = catch_unwind(AssertUnwindSafe(|| {
        let x = tensor(in_data, in_shape, true);
        let y = match forward(rank, &x, pads, mode) {
            Ok(y) => y,
            Err(()) => return None,
        };
        if y.data().unwrap().is_empty() {
            return Some(vec![0.0; in_data.len()]);
        }
        let s = ferrotorch_core::grad_fns::reduction::sum(&y).ok()?;
        ferrotorch_core::backward(&s).ok()?;
        let g = x.grad().ok()??;
        Some(g.data().unwrap().to_vec())
    }));
    match res {
        Ok(Some(g)) => GradOut::Grad(g),
        Ok(None) => GradOut::Err,
        Err(_) => GradOut::Panic,
    }
}

/// ferro `x.grad` from a NON-UNIFORM seeded grad_output VJP
/// (`backward_with_grad(y, seed)`). `seed` is the oracle's distinct-per-cell
/// ramp on the OUTPUT shape — its length must equal `y.numel()`.
fn grad_seeded(
    rank: usize,
    in_data: &[f64],
    in_shape: &[usize],
    pads: &[isize],
    mode: PaddingMode,
    seed: &[f64],
    out_shape: &[usize],
) -> GradOut {
    let res = catch_unwind(AssertUnwindSafe(|| {
        let x = tensor(in_data, in_shape, true);
        let y = match forward(rank, &x, pads, mode) {
            Ok(y) => y,
            Err(()) => return None,
        };
        if y.data().unwrap().is_empty() {
            return Some(vec![0.0; in_data.len()]);
        }
        let seed_t = tensor(seed, out_shape, false);
        ferrotorch_core::backward_with_grad(&y, Some(&seed_t)).ok()?;
        let g = x.grad().ok()??;
        Some(g.data().unwrap().to_vec())
    }));
    match res {
        Ok(Some(g)) => GradOut::Grad(g),
        Ok(None) => GradOut::Err,
        Err(_) => GradOut::Panic,
    }
}

#[derive(Default, Debug, Clone)]
struct ModeCounts {
    defined: usize,
    grad_sum_mismatch: usize,
    grad_seed_mismatch: usize,
    grad_err: usize,
    grad_panic: usize,
    garbage: usize,
}

#[derive(Default)]
struct Report {
    constant: ModeCounts,
    reflect: ModeCounts,
    replicate: ModeCounts,
    circular: ModeCounts,
    examples: Vec<String>,
}
impl Report {
    fn bucket(&mut self, mode: &str) -> &mut ModeCounts {
        match mode {
            "constant" => &mut self.constant,
            "reflect" => &mut self.reflect,
            "replicate" => &mut self.replicate,
            "circular" => &mut self.circular,
            _ => unreachable!(),
        }
    }
}

fn run_backward_oracle() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let script = format!("{manifest}/tests/fixtures_pad_grid_backward_gen.py");
    let output = Command::new("python3")
        .arg(&script)
        .output()
        .expect("failed to spawn python3 backward oracle generator");
    assert!(
        output.status.success(),
        "backward oracle generator failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("oracle stdout not utf8")
}

fn run_backward_grid() -> Report {
    let text = run_backward_oracle();
    let mut rep = Report::default();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let rec: Value = serde_json::from_str(line).expect("bad oracle json line");
        if !rec["ok"].as_bool().unwrap() {
            continue;
        }
        let rank = rec["rank"].as_u64().unwrap() as usize;
        let mode_s = rec["mode"].as_str().unwrap();
        let in_shape = vusize(&rec["in_shape"]);
        let in_data = vnums(&rec["in_data"]);
        let pads = vsigned(&rec["pads"]);
        let garbage = rec
            .get("garbage_det")
            .and_then(|g| g.as_bool())
            .unwrap_or(false);
        let mode = mode_of(mode_s);

        if garbage {
            // No defined backward contract. ferro Errs the forward, so backward
            // is never reached — but a PANIC on the forward/backward path is
            // STILL a hard fail. Run both calls and only flag a panic.
            let r1 = grad_sum(rank, &in_data, &in_shape, &pads, mode);
            if let GradOut::Panic = r1 {
                let c = rep.bucket(mode_s);
                c.grad_panic += 1;
                if rep.examples.len() < 80 {
                    rep.examples.push(format!(
                        "PANIC(garbage) {mode_s} rank{rank} in{in_shape:?} pads{pads:?}"
                    ));
                }
            }
            rep.bucket(mode_s).garbage += 1;
            continue;
        }

        rep.bucket(mode_s).defined += 1;
        let out_shape = vusize(&rec["out_shape"]);
        let want_sum = vnums(&rec["grad_sum"]);
        let seed = vnums(&rec["seed"]);
        let want_seed = vnums(&rec["grad_seed"]);

        // (1) sum() / all-ones seed VJP.
        match grad_sum(rank, &in_data, &in_shape, &pads, mode) {
            GradOut::Grad(g) => {
                if g.len() != want_sum.len()
                    || g.iter()
                        .zip(want_sum.iter())
                        .any(|(a, b)| b.is_nan() || (a - b).abs() > TOL)
                {
                    let c = rep.bucket(mode_s);
                    c.grad_sum_mismatch += 1;
                    if rep.examples.len() < 80 {
                        rep.examples.push(format!(
                            "GRAD_SUM {mode_s} rank{rank} in{in_shape:?} pads{pads:?}: torch{want_sum:?} ferro{g:?}"
                        ));
                    }
                }
            }
            GradOut::Err => {
                let c = rep.bucket(mode_s);
                c.grad_err += 1;
                if rep.examples.len() < 80 {
                    rep.examples.push(format!(
                        "GRAD_ERR(sum) {mode_s} rank{rank} in{in_shape:?} pads{pads:?}: torch DEFINED grad{want_sum:?}, ferro backward Err"
                    ));
                }
            }
            GradOut::Panic => {
                let c = rep.bucket(mode_s);
                c.grad_panic += 1;
                if rep.examples.len() < 80 {
                    rep.examples.push(format!(
                        "GRAD_PANIC(sum) {mode_s} rank{rank} in{in_shape:?} pads{pads:?}: torch DEFINED grad{want_sum:?} (R-CODE-2)"
                    ));
                }
            }
        }

        // (2) NON-UNIFORM seeded VJP (multi-write accumulation discriminator).
        match grad_seeded(rank, &in_data, &in_shape, &pads, mode, &seed, &out_shape) {
            GradOut::Grad(g) => {
                if g.len() != want_seed.len()
                    || g.iter()
                        .zip(want_seed.iter())
                        .any(|(a, b)| b.is_nan() || (a - b).abs() > TOL)
                {
                    let c = rep.bucket(mode_s);
                    c.grad_seed_mismatch += 1;
                    if rep.examples.len() < 80 {
                        rep.examples.push(format!(
                            "GRAD_SEED {mode_s} rank{rank} in{in_shape:?} pads{pads:?}: torch{want_seed:?} ferro{g:?} (non-uniform ramp seed)"
                        ));
                    }
                }
            }
            GradOut::Err => {
                let c = rep.bucket(mode_s);
                c.grad_err += 1;
                if rep.examples.len() < 80 {
                    rep.examples.push(format!(
                        "GRAD_ERR(seed) {mode_s} rank{rank} in{in_shape:?} pads{pads:?}: torch DEFINED grad{want_seed:?}, ferro Err"
                    ));
                }
            }
            GradOut::Panic => {
                let c = rep.bucket(mode_s);
                c.grad_panic += 1;
                if rep.examples.len() < 80 {
                    rep.examples.push(format!(
                        "GRAD_PANIC(seed) {mode_s} rank{rank} in{in_shape:?} pads{pads:?}: torch DEFINED grad{want_seed:?} (R-CODE-2)"
                    ));
                }
            }
        }
    }
    rep
}

fn assert_mode_backward_clean(name: &str, c: &ModeCounts, examples: &[String]) {
    let hard = c.grad_sum_mismatch + c.grad_seed_mismatch + c.grad_err + c.grad_panic;
    if hard != 0 {
        let ex: Vec<&String> = examples
            .iter()
            .filter(|e| e.to_lowercase().contains(name))
            .take(20)
            .collect();
        panic!(
            "MODE {name} BACKWARD DIVERGES from torch over the DETERMINISTIC grid \
             (defined {def}): grad_sum_mismatch={gsm} grad_seed_mismatch={gse} \
             grad_err={ge} grad_panic={gp} (garbage={g} skipped). First examples: {ex:#?}",
            def = c.defined,
            gsm = c.grad_sum_mismatch,
            gse = c.grad_seed_mismatch,
            ge = c.grad_err,
            gp = c.grad_panic,
            g = c.garbage,
        );
    }
}

/// THE definitive DETERMINISTIC BACKWARD grid assertion. For ALL four modes,
/// over the same grid the forward audit covers, every DEFINED torch `x.grad`
/// (both the all-ones `sum()` seed AND a distinct-per-cell non-uniform ramp
/// seed) is reproduced by ferrotorch, with 0 backward panics and 0 backward
/// errs on defined cases.
///
/// NOT `#[ignore]`d: if this passes, the negative-pad chain's BACKWARD path is
/// provably clean against live torch with a deterministic garbage oracle, the
/// twin of the already-clean forward grid. If it FAILS, the failure pins a
/// real defined-case backward divergence (the test failing IS the block).
#[test]
fn definitive_negpad_det_backward_grid_all_modes() {
    let rep = run_backward_grid();

    eprintln!(
        "=== DETERMINISTIC NEG-PAD BACKWARD GRID (live torch 2.11.0+cu130, cold-fork oracle; sum() + non-uniform ramp seed VJP) ==="
    );
    for (name, c) in [
        ("constant", &rep.constant),
        ("reflect", &rep.reflect),
        ("replicate", &rep.replicate),
        ("circular", &rep.circular),
    ] {
        eprintln!(
            "{name:>9}: defined={:>6} grad_sum_mm={} grad_seed_mm={} grad_err={} grad_panic={} | garbage_skipped={}",
            c.defined,
            c.grad_sum_mismatch,
            c.grad_seed_mismatch,
            c.grad_err,
            c.grad_panic,
            c.garbage,
        );
    }
    if !rep.examples.is_empty() {
        eprintln!("--- first backward divergence examples ---");
        for e in rep.examples.iter().take(40) {
            eprintln!("  {e}");
        }
    }

    assert_mode_backward_clean("constant", &rep.constant, &rep.examples);
    assert_mode_backward_clean("reflect", &rep.reflect, &rep.examples);
    assert_mode_backward_clean("replicate", &rep.replicate, &rep.examples);
    assert_mode_backward_clean("circular", &rep.circular, &rep.examples);
}
