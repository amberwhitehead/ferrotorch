//! INDEPENDENT acto-critic re-audit of the negative-pad chain
//! (#1611/#1620/#1621/#1623/#1624/#1625/#1626/#1627/#1628), targeting commit
//! `fb19ca311` ("holistic circular allocate-then-copy"),
//! `ferrotorch-nn/src/padding.rs`.
//!
//! This file does NOT trust the chain-close harness's
//! `fixtures_pad_grid_gen.py` value-set `garbage` heuristic (which under-flags
//! any uninitialized read whose heap garbage coincidentally rounds to an input
//! value). Instead it drives `tests/fixtures_pad_grid_indep_gen.py`, an
//! independent live-torch 2.11 oracle that classifies each circular over-crop
//! "torch-OK" record as garbage by DETERMINISM: it re-runs torch on the SAME
//! case under several allocator/heap-pollution states and flags `garbage_indep`
//! iff the output VARIES across runs (or is ever non-finite). A single stable
//! finite value-tuple across all polluted runs == a DEFINED torch result that
//! ferrotorch MUST reproduce; otherwise it is uninitialized memory (R-DEV-6) and
//! ferrotorch MUST reject (never fabricate a value).
//!
//! R-CHAR-3: every expected acceptance / shape / value below is read from the
//! live-torch oracle, never copied from ferrotorch.
//!
//! Upstream contract mirrored: `aten/src/ATen/native/PadNd.cpp:140-187`
//! (`_pad_circular_symint`): `:140-142` wrap-once, `:143-145` empty-dim,
//! `:148` `new_empty(out_shape)`, `:154-161` center `copy_`, `:169-187` wrap
//! `copy_`s.
//!
//! VERDICT GATES (all four must hold for the negative-pad CODE chain to close):
//!   - value_mismatch == 0  (defined cases: ferro values match torch)
//!   - shape_mismatch == 0
//!   - panic == 0           (R-CODE-2: never a panic on any signed pad)
//!   - accept_mismatch == 0 (torch-ERR cases ferro must also reject)
//!   - reject_mismatch == 0 (torch-OK DEFINED cases ferro must NOT reject)
//!   - dev6_accepted == 0   (torch-garbage cases ferro must NOT accept)
//! The only tolerated ferro-Err/torch-OK class is `dev6_carveout` (independently
//! confirmed nondeterministic / uninitialized), reported but never failing.

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

fn tensor(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
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
    let x = tensor(in_data, in_shape);
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
    /// torch returns a DEFINED (deterministic, finite) result where ferro errors.
    reject_mismatch: usize,
    /// torch returns NONDETERMINISTIC/uninitialized garbage; ferro rejects (OK, R-DEV-6).
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

fn run_indep_grid() -> Report {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let script = format!("{manifest}/tests/fixtures_pad_grid_indep_gen.py");
    let output = Command::new("python3")
        .arg(&script)
        .output()
        .expect("failed to spawn python3 independent oracle generator");
    assert!(
        output.status.success(),
        "independent oracle generator failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).expect("oracle stdout not utf8");

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
            .get("garbage_indep")
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
                    // torch nondeterministic/uninitialized but ferro accepted: ferro
                    // fabricated a value with no contract. FAIL.
                    let c = rep.bucket(mode_s);
                    c.dev6_accepted += 1;
                    if rep.examples.len() < 80 {
                        rep.examples.push(format!(
                            "DEV6-ACCEPTED {mode_s} rank{rank} in{in_shape:?} pads{pads:?} \
                             (torch nondeterministic; ferro accepted shape{:?} data{:?})",
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
                        let nd = rec.get("n_distinct").and_then(|x| x.as_u64()).unwrap_or(0);
                        rep.examples.push(format!(
                            "REJECT-MISMATCH {mode_s} rank{rank} in{in_shape:?} pads{pads:?}: \
                             torch DEFINED (n_distinct={nd}) shape{ws:?} data{wd:?}; ferro Err"
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
            "MODE {name} DIVERGES from torch over the INDEPENDENT grid (total {tot}): \
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

/// THE definitive INDEPENDENT grid assertion. Uses determinism-based garbage
/// classification (not the value-set heuristic). Passes iff, for ALL four modes,
/// every DEFINED torch result is reproduced by ferrotorch and every torch
/// nondeterministic/uninitialized result is rejected by ferrotorch.
///
/// NOT `#[ignore]`d: if this passes, the negative-pad CODE chain
/// (#1611...#1628) is provably clean against live torch with an INDEPENDENT
/// garbage oracle and can close.
#[test]
fn definitive_negpad_indep_grid_all_modes() {
    let rep = run_indep_grid();

    eprintln!(
        "=== INDEPENDENT NEG-PAD GRID (live torch 2.11.0+cu130, determinism garbage oracle) ==="
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
