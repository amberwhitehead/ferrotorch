//! Coverage-gap pin for #1653 (torch.special transcendental runner-arm scope).
//!
//! The torch.special transcendental family landed SHIPPED in
//! `ferrotorch-core/src/special.rs` under umbrella #1651 (CPU end-to-end,
//! live-torch-verified): `special.entr`, `special.ndtr`, `special.ndtri`,
//! `i0` (PLAIN — op_db registers it without the `special.` prefix),
//! `special.i0e`, `special.i1`, `special.i1e`, `special.zeta` (binary Hurwitz
//! zeta), `special.airy_ai`, `special.spherical_bessel_j0`,
//! `special.modified_bessel_k0`, `special.modified_bessel_k1`,
//! `special.scaled_modified_bessel_k0`, `special.scaled_modified_bessel_k1`.
//!
//! But the parity-sweep RUNNER (`tools/parity-sweep/runner/src/main.rs`) had
//! NO `match op` dispatch arm for any of them, and `dispatch_ops()` did not
//! list them. So sweeping any of these ops yielded "unknown op — no runner
//! arm" (grep count 0): the sweep never exercised the SHIPPED implementations
//! against the live torch oracle. This was a COVERAGE gap, not a value
//! divergence — hence these tests pin the ADAPTER (dispatch presence), not a
//! wrong value.
//!
//! RESOLVED 2026-05-29 (#1653, impl side): the runner now carries a dispatch
//! arm for each op (`tools/parity-sweep/runner/src/main.rs`, near the
//! transcendental unary cluster) routing through the matching
//! `ferrotorch_core::special::<fn>`, and `dispatch_ops()` lists all 14 names.
//! These tests are un-ignored, serving as permanent regression coverage for
//! the adapters.
//!
//! The op-name keys are the EXACT `op_db` registration names (verified present
//! in the live `torch.testing._internal.common_methods_invocations.op_db` on
//! 2026-05-29). The runner passes them through `oracle_name()` unchanged (the
//! default fall-through arm), so the oracle resolves them via `op_db`.
//!
//! Tracking: crosslink #1653 (impl side; parent #1651).
//!
//! Run with:
//!   LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH" \
//!     cargo test -p parity-sweep-runner \
//!     --test divergence_special_family_coverage -- --nocapture

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

/// The full set of torch.special transcendental op_db names the runner arms of
/// #1653 must dispatch. These are the keys used BOTH as the `dispatch_f32`
/// match arms AND in `dispatch_ops()`; they are the exact `op_db` names.
const SPECIAL_FAMILY: &[&str] = &[
    "special.entr",
    "special.ndtr",
    "special.ndtri",
    "i0",
    "special.i0e",
    "special.i1",
    "special.i1e",
    "special.airy_ai",
    "special.spherical_bessel_j0",
    "special.modified_bessel_k0",
    "special.modified_bessel_k1",
    "special.scaled_modified_bessel_k0",
    "special.scaled_modified_bessel_k1",
    "special.zeta",
];

struct OracleProc {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl OracleProc {
    fn spawn() -> Option<Self> {
        let mut child = Command::new("python3")
            .arg("../oracle.py")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let stdin = child.stdin.take()?;
        let stdout = BufReader::new(child.stdout.take()?);
        let mut p = OracleProc {
            child,
            stdin,
            stdout,
        };
        let ready = p.request(json!({"cmd": "ready"}))?;
        if ready.get("ok").and_then(Value::as_bool) != Some(true) {
            return None;
        }
        Some(p)
    }

    fn request(&mut self, req: Value) -> Option<Value> {
        let line = serde_json::to_string(&req).ok()?;
        self.stdin.write_all(line.as_bytes()).ok()?;
        self.stdin.write_all(b"\n").ok()?;
        self.stdin.flush().ok()?;
        let mut resp = String::new();
        self.stdout.read_line(&mut resp).ok()?;
        serde_json::from_str(&resp).ok()
    }
}

impl Drop for OracleProc {
    fn drop(&mut self) {
        let _ = self.request(json!({"cmd": "shutdown"}));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The runner binary's `dispatch_ops()` list (the `&'static [&'static str]`
/// the sweep enumerates) MUST include every torch.special transcendental op
/// the family shipped under #1651. Probed via `parity-sweep dispatch`, which
/// prints the `dispatch_ops()` set (one op per line, no oracle needed). Before
/// #1653 none of the `special.*` / `i0` names appeared — proving the
/// runner-arm scope was incomplete.
///
/// This test needs NO torch (it shells the runner binary's `dispatch`
/// command, which just enumerates the in-binary `dispatch_ops()` static), so
/// it runs even in CI without the oracle.
#[test]
fn special_family_is_listed_in_dispatch_ops() {
    let out = Command::new(env!("CARGO_BIN_EXE_parity-sweep"))
        .arg("dispatch")
        .output()
        .expect("run parity-sweep dispatch");
    assert!(
        out.status.success(),
        "parity-sweep dispatch exited non-zero: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let listed = String::from_utf8_lossy(&out.stdout);
    let names: std::collections::HashSet<&str> = listed.lines().map(str::trim).collect();
    for op in SPECIAL_FAMILY {
        assert!(
            names.contains(op),
            "torch.special op `{op}` is MISSING from `dispatch_ops()` — the \
             #1653 runner-arm scope is INCOMPLETE (the op is SHIPPED in \
             `ferrotorch_core::special` under #1651 but the parity-sweep never \
             exercises it). Tracking: crosslink #1653."
        );
    }
}

/// Each torch.special op must DISPATCH through the live torch oracle without
/// "unknown op" — i.e. the oracle resolves the name via `op_db` (the runner
/// passes it through `oracle_name()` unchanged). We ask the oracle to `sample`
/// each op at seed 0 / index 0; a successful sample proves op_db carries the
/// name (the precondition for the runner arm to receive inputs). A failing
/// sample with "unknown op" would prove the name key is wrong.
#[test]
fn special_family_samples_from_op_db_without_unknown_op() {
    let Some(mut oracle) = OracleProc::spawn() else {
        eprintln!("SKIP: torch oracle unavailable; cannot probe special-family coverage");
        return;
    };

    for op in SPECIAL_FAMILY {
        let resp = oracle
            .request(json!({
                "cmd": "sample",
                "op": op,
                "seed": 0,
                "index": 0
            }))
            .unwrap_or_else(|| panic!("oracle request for {op}"));
        let ok = resp.get("ok").and_then(Value::as_bool).unwrap_or(false);
        let err = resp.get("err").and_then(Value::as_str).unwrap_or("");
        assert!(
            ok,
            "torch oracle could not sample `{op}` (the op-name key the runner \
             arm uses must match an `op_db` registration). oracle err: {err:?}. \
             If this is `unknown op`, the dispatch-arm key is wrong. Tracking: \
             crosslink #1653."
        );
        // A sample must carry at least one positional arg (the input tensor) so
        // the runner's `unary`/`binary` helper has something to dispatch.
        let args = resp.get("args").and_then(Value::as_array);
        assert!(
            args.map(|a| !a.is_empty()).unwrap_or(false),
            "`{op}` sample produced no positional args; the runner arm would \
             have no input to dispatch. oracle response: {resp:?}"
        );
    }
}
