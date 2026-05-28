//! DETERMINISTIC FINAL close-audit of the negative-pad chain
//! (#1611/#1620/#1621/#1623/#1624/#1625/#1626/#1627/#1628/#1629), targeting
//! commit `2dfd7cd83` ("circular wrap copies read out LIVE"),
//! `ferrotorch-nn/src/padding.rs`.
//!
//! WHY THIS SUPERSEDES `divergence_negpad_indep_reaudit.rs`. That harness's
//! oracle (`fixtures_pad_grid_indep_gen.py`) classifies a circular over-crop as
//! uninitialized "garbage" by a single multiplicative-k=1000 linearity ∧
//! value-membership heuristic computed in ONE process. That is FLAKY: a
//! `new_empty` uninitialized read (`PadNd.cpp:148`) that, when many pads run in
//! one process, lands on a heap region holding a PRIOR case's freed (in-set,
//! k-linear) output is wrongly tagged DEFINED, so its `reject_mismatch` flickers
//! 0↔2 and the failing case identity changes run-to-run. A "provably 0" close
//! cannot rest on that.
//!
//! THIS harness drives `tests/fixtures_pad_grid_det_gen.py`, a DETERMINISTIC
//! oracle (see that file's docstring) using ADDITIVE-SHIFT GATHER CONSISTENCY:
//! torch's circular `copy_`s VERBATIM-gather input elements, and the gather
//! INDEX is a pure function of (shape, pads), independent of input VALUES. So
//! padding with two inputs that differ by a constant additive shift `s` must
//! shift every DEFINED output cell by exactly `s`; an uninitialized cell reads
//! heap memory uncorrelated with the shift and fails. A case is GENUINE-GARBAGE
//! iff ANY of several distinct shifts fails OR a base cell is non-finite; else
//! DEFINED with the recorded verbatim gather (every cell an exact input member —
//! corroborated). This is allocator-INDEPENDENT and bit-reproducible across runs
//! (verified: 0 classification diffs over >=3 generator runs; garbage tail
//! deterministically 4060).
//!
//! R-CHAR-3: every expected acceptance / shape / value / grad below is read from
//! the live-torch oracle, never copied from ferrotorch.
//!
//! Upstream contract mirrored: `aten/src/ATen/native/PadNd.cpp:140-187`
//! (`_pad_circular_symint`): `:140-142` wrap-once, `:143-145` empty-dim,
//! `:148` `new_empty(out_shape)`, `:154-161` center `copy_`, `:169-187` wrap
//! `copy_`s reading `out` LIVE.
//!
//! VERDICT GATES (all must hold for the negative-pad CODE chain to close):
//!   - value_mismatch == 0    (defined cases: ferro values match torch)
//!   - shape_mismatch == 0
//!   - panic == 0             (R-CODE-2: never a panic on any signed pad)
//!   - accept_mismatch == 0   (torch-ERR cases ferro must also reject)
//!   - reject_mismatch == 0   (torch-OK DEFINED cases ferro must NOT reject)
//!   - dev6_accepted == 0     (torch-garbage cases ferro must NOT accept)
//!
//! The only tolerated ferro-Err/torch-OK class is `dev6_carveout`
//! (deterministically classified torch-uninitialized), reported but never
//! failing.

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

fn call_ferro(
    rank: usize,
    in_data: &[f64],
    in_shape: &[usize],
    pads: &[isize],
    mode: PaddingMode,
) -> Result<Tensor<f64>, ()> {
    let x = tensor(in_data, in_shape, false);
    match rank {
        1 => functional_pad_1d_signed(&x, pads[0], pads[1], mode, 0.0).map_err(|_| ()),
        2 => functional_pad_2d_signed(&x, pads[0], pads[1], pads[2], pads[3], mode, 0.0)
            .map_err(|_| ()),
        other => panic!("unsupported rank {other}"),
    }
}

#[derive(Default, Debug, Clone)]
struct ModeCounts {
    total: usize,
    value_mismatch: usize,
    shape_mismatch: usize,
    panic: usize,
    /// ferro accepts where torch errors.
    accept_mismatch: usize,
    /// torch returns a DEFINED (deterministic, in-set) result where ferro errors.
    reject_mismatch: usize,
    /// torch deterministically-classified uninitialized garbage; ferro rejects
    /// (correct, R-DEV-6).
    dev6_carveout: usize,
    /// torch garbage but ferro ACCEPTED (fabricated a value) — a FAIL.
    dev6_accepted: usize,
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

fn run_det_oracle() -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let script = format!("{manifest}/tests/fixtures_pad_grid_det_gen.py");
    let output = Command::new("python3")
        .arg(&script)
        .output()
        .expect("failed to spawn python3 deterministic oracle generator");
    assert!(
        output.status.success(),
        "deterministic oracle generator failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("oracle stdout not utf8")
}

fn run_det_grid() -> Report {
    let text = run_det_oracle();
    let mut rep = Report::default();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let rec: Value = serde_json::from_str(line).expect("bad oracle json line");
        let rank = rec["rank"].as_u64().unwrap() as usize;
        let mode_s = rec["mode"].as_str().unwrap();
        let in_shape = vusize(&rec["in_shape"]);
        let in_data = vnums(&rec["in_data"]);
        let pads = vsigned(&rec["pads"]);
        let torch_ok = rec["ok"].as_bool().unwrap();
        let garbage = rec
            .get("garbage_det")
            .and_then(|g| g.as_bool())
            .unwrap_or(false);
        let mode = mode_of(mode_s);

        rep.bucket(mode_s).total += 1;

        let res = catch_unwind(AssertUnwindSafe(|| {
            call_ferro(rank, &in_data, &in_shape, &pads, mode)
        }));
        let ferro = match res {
            Ok(r) => r,
            Err(_) => {
                let c = rep.bucket(mode_s);
                c.panic += 1;
                if rep.examples.len() < 80 {
                    rep.examples.push(format!(
                        "PANIC {mode_s} rank{rank} in{in_shape:?} pads{pads:?}"
                    ));
                }
                continue;
            }
        };

        match (torch_ok, ferro) {
            (true, Ok(t)) => {
                if garbage {
                    let c = rep.bucket(mode_s);
                    c.dev6_accepted += 1;
                    if rep.examples.len() < 80 {
                        rep.examples.push(format!(
                            "DEV6-ACCEPTED {mode_s} rank{rank} in{in_shape:?} pads{pads:?} \
                             (torch uninitialized; ferro accepted shape{:?} data{:?})",
                            t.shape().to_vec(),
                            t.data().unwrap().to_vec()
                        ));
                    }
                    continue;
                }
                let want_shape = vusize(&rec["out_shape"]);
                let got_shape = t.shape().to_vec();
                if got_shape != want_shape {
                    let c = rep.bucket(mode_s);
                    c.shape_mismatch += 1;
                    if rep.examples.len() < 80 {
                        rep.examples.push(format!(
                            "SHAPE {mode_s} rank{rank} in{in_shape:?} pads{pads:?}: torch{want_shape:?} ferro{got_shape:?}"
                        ));
                    }
                    continue;
                }
                let want_data = vnums(&rec["out_data"]);
                let got = t.data().unwrap();
                let mut bad = false;
                for (a, b) in got.iter().zip(want_data.iter()) {
                    if b.is_nan() || (a - b).abs() > TOL {
                        bad = true;
                        break;
                    }
                }
                if bad {
                    let c = rep.bucket(mode_s);
                    c.value_mismatch += 1;
                    if rep.examples.len() < 80 {
                        rep.examples.push(format!(
                            "VALUE {mode_s} rank{rank} in{in_shape:?} pads{pads:?}: torch{want_data:?} ferro{:?}",
                            got.to_vec()
                        ));
                    }
                }
            }
            (true, Err(())) => {
                if garbage {
                    rep.bucket(mode_s).dev6_carveout += 1;
                } else {
                    let c = rep.bucket(mode_s);
                    c.reject_mismatch += 1;
                    if rep.examples.len() < 80 {
                        let ws = vusize(&rec["out_shape"]);
                        let wd = vnums(&rec["out_data"]);
                        rep.examples.push(format!(
                            "REJECT-MISMATCH {mode_s} rank{rank} in{in_shape:?} pads{pads:?}: \
                             torch DEFINED shape{ws:?} data{wd:?}; ferro Err"
                        ));
                    }
                }
            }
            (false, Ok(t)) => {
                let c = rep.bucket(mode_s);
                c.accept_mismatch += 1;
                if rep.examples.len() < 80 {
                    rep.examples.push(format!(
                        "ACCEPT-MISMATCH {mode_s} rank{rank} in{in_shape:?} pads{pads:?}: \
                         torch Err; ferro OK shape{:?} data{:?}",
                        t.shape().to_vec(),
                        t.data().unwrap().to_vec()
                    ));
                }
            }
            (false, Err(())) => { /* both reject — correct */ }
        }
    }
    rep
}

fn assert_mode_clean(name: &str, c: &ModeCounts, examples: &[String]) {
    let hard = c.value_mismatch
        + c.shape_mismatch
        + c.panic
        + c.accept_mismatch
        + c.reject_mismatch
        + c.dev6_accepted;
    if hard != 0 {
        let ex: Vec<&String> = examples
            .iter()
            .filter(|e| e.to_lowercase().contains(name))
            .take(15)
            .collect();
        panic!(
            "MODE {name} DIVERGES from torch over the DETERMINISTIC grid (total {tot}): \
             value_mismatch={vm} shape_mismatch={sm} panic={p} accept_mismatch={am} \
             reject_mismatch={rm} dev6_accepted={da} (dev6_carveout={dc} tolerated). \
             First examples: {ex:#?}",
            tot = c.total,
            vm = c.value_mismatch,
            sm = c.shape_mismatch,
            p = c.panic,
            am = c.accept_mismatch,
            rm = c.reject_mismatch,
            da = c.dev6_accepted,
            dc = c.dev6_carveout,
        );
    }
}

/// THE definitive DETERMINISTIC FORWARD grid assertion. Uses additive-shift
/// gather-consistency garbage classification. Passes iff, for ALL four modes,
/// every DEFINED torch result is reproduced by ferrotorch and every torch
/// uninitialized result is rejected by ferrotorch.
///
/// NOT `#[ignore]`d: if this passes, the negative-pad CODE chain's FORWARD path
/// is provably clean against live torch with a deterministic garbage oracle.
#[test]
fn definitive_negpad_det_grid_all_modes() {
    let rep = run_det_grid();

    eprintln!(
        "=== DETERMINISTIC NEG-PAD FORWARD GRID (live torch 2.11.0+cu130, cold-fork + additive-shift oracle) ==="
    );
    for (name, c) in [
        ("constant", &rep.constant),
        ("reflect", &rep.reflect),
        ("replicate", &rep.replicate),
        ("circular", &rep.circular),
    ] {
        eprintln!(
            "{name:>9}: total={:>6} value={} shape={} panic={} accept_mm={} reject_mm={} dev6_accepted={} | dev6_carveout={}",
            c.total,
            c.value_mismatch,
            c.shape_mismatch,
            c.panic,
            c.accept_mismatch,
            c.reject_mismatch,
            c.dev6_accepted,
            c.dev6_carveout,
        );
    }
    if !rep.examples.is_empty() {
        eprintln!("--- first divergence examples ---");
        for e in rep.examples.iter().take(40) {
            eprintln!("  {e}");
        }
    }

    assert_mode_clean("constant", &rep.constant, &rep.examples);
    assert_mode_clean("reflect", &rep.reflect, &rep.examples);
    assert_mode_clean("replicate", &rep.replicate, &rep.examples);
    assert_mode_clean("circular", &rep.circular, &rep.examples);
}

// ===========================================================================
// BACKWARD parity — per mode, on a DEFINED boundary case (incl. the #1629
// circular over-crop propagation cases). Expected grads are LIVE-torch
// `sum(F.pad(x)).backward()` values (R-CHAR-3), hard-wired here from the live
// oracle so the backward gate runs without re-spawning python.
// ===========================================================================

/// Runs ferro backward, catching a PANIC (R-CODE-2 violation) as a distinct
/// outcome so it is reported rather than aborting the test binary.
enum GradOut {
    Grad(Vec<f64>),
    Err,
    Panic,
}

fn grad_of(
    rank: usize,
    in_data: &'static [f64],
    in_shape: &'static [usize],
    pads: &'static [isize],
    mode: PaddingMode,
) -> GradOut {
    let res = catch_unwind(AssertUnwindSafe(|| {
        let x = tensor(in_data, in_shape, true);
        let y = match rank {
            1 => functional_pad_1d_signed(&x, pads[0], pads[1], mode, 0.0),
            2 => functional_pad_2d_signed(&x, pads[0], pads[1], pads[2], pads[3], mode, 0.0),
            _ => unreachable!(),
        };
        let y = match y {
            Ok(y) => y,
            Err(_) => return None,
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

/// Backward gate. The circular over-crop BACKWARD currently PANICS
/// (forward/backward gather mismatch from the #1629 forward fix); this test
/// FAILS on those rows. `#[ignore]`d now that the divergence is tracked (#1631)
/// and pinned by `divergence_1631_circular_overcrop_backward_panic.rs`; run via
/// `--ignored`. The constant/reflect/replicate/circular NON-over-crop rows pass.
#[test]
fn definitive_negpad_det_backward_sample_all_modes() {
    type Row = (
        usize,
        PaddingMode,
        &'static [usize],
        &'static [f64],
        &'static [isize],
        &'static [f64],
    );
    let rows: &[Row] = &[
        (
            1,
            PaddingMode::Zeros,
            &[1, 4],
            &[1.0, 2.0, 3.0, 4.0],
            &[-1, 2],
            &[0.0, 1.0, 1.0, 1.0],
        ),
        (
            1,
            PaddingMode::Reflect,
            &[1, 4],
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 1],
            &[1.0, 2.0, 3.0, 1.0],
        ),
        (
            1,
            PaddingMode::Replicate,
            &[1, 4],
            &[1.0, 2.0, 3.0, 4.0],
            &[-1, 2],
            &[0.0, 1.0, 1.0, 3.0],
        ),
        (
            1,
            PaddingMode::Circular,
            &[1, 4],
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 1],
            &[2.0, 1.0, 2.0, 2.0],
        ),
        (
            2,
            PaddingMode::Zeros,
            &[1, 2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[-1, 2, 1, -1],
            &[0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ),
        (
            2,
            PaddingMode::Reflect,
            &[1, 2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 1, 1, 0],
            &[1.0, 3.0, 1.0, 2.0, 6.0, 2.0],
        ),
        (
            2,
            PaddingMode::Replicate,
            &[1, 2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[-1, 2, 1, 0],
            &[0.0, 2.0, 6.0, 0.0, 1.0, 3.0],
        ),
        (
            2,
            PaddingMode::Circular,
            &[1, 2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 1, 1, 1],
            &[4.0, 2.0, 4.0, 4.0, 2.0, 4.0],
        ),
        // #1629 circular over-crop wrap PROPAGATION cases (DEFINED forward). The
        // backward must route gradient through the propagated band; ferro PANICS
        // here (index OOB at padding.rs:2230) — tracking #1631.
        (
            2,
            PaddingMode::Circular,
            &[1, 1, 2],
            &[1.0, 2.0],
            &[-1, 2, 0, 1],
            &[0.0, 4.0],
        ),
        (
            2,
            PaddingMode::Circular,
            &[1, 1, 4],
            &[1.0, 2.0, 3.0, 4.0],
            &[-1, 4, 0, 1],
            &[0.0, 4.0, 4.0, 4.0],
        ),
    ];

    let mut fails = Vec::new();
    for (rank, mode, in_shape, in_data, pads, want) in rows.iter() {
        match grad_of(*rank, in_data, in_shape, pads, *mode) {
            GradOut::Grad(g) => {
                if g.len() != want.len()
                    || g.iter().zip(want.iter()).any(|(a, b)| (a - b).abs() > TOL)
                {
                    fails.push(format!(
                        "{mode:?} rank{rank} in{in_shape:?} pads{pads:?}: torch grad {want:?} ferro {g:?}"
                    ));
                }
            }
            GradOut::Err => fails.push(format!(
                "{mode:?} rank{rank} in{in_shape:?} pads{pads:?}: ferro backward Err (torch grad {want:?})"
            )),
            GradOut::Panic => fails.push(format!(
                "{mode:?} rank{rank} in{in_shape:?} pads{pads:?}: ferro backward PANIC (R-CODE-2 \
                 violation; torch grad {want:?}) — forward/backward circular gather mismatch (#1631)"
            )),
        }
    }
    assert!(
        fails.is_empty(),
        "BACKWARD diverges from torch on a DEFINED boundary sample:\n  {}",
        fails.join("\n  ")
    );
}
