//! Coverage-gap pin for commit `381fad746` (#1344 runner-arm scope).
//!
//! The commit claims: "ALL torch-public grad_fns/linalg.rs ops now have runner
//! arms (only dot/mv/mm_bt lack bare arms — dot/mv exercised via matmul
//! rank-dispatch, mm_bt crate-private with no torch counterpart)."
//!
//! That claim is INCOMPLETE. `torch.linalg.vector_norm` is a distinct
//! torch-PUBLIC differentiable linalg op (REQ-23 in
//! `.design/ferrotorch-core/grad_fns/linalg.md`): it has a production forward
//! `pub fn vector_norm(input, ord)` at `ferrotorch-core/src/linalg.rs:1518`
//! and a `vector_norm_differentiable` backward in
//! `ferrotorch-core/src/grad_fns/linalg.rs`, backed by
//! `ferray_linalg::vector_norm`, with an `ord` parameter and DISTINCT
//! semantics from `matrix_norm` (the matrix Frobenius/operator norm).
//!
//! The runner `norm` arm (`tools/parity-sweep/runner/src/main.rs:3750`) calls
//! `ferrotorch_core::linalg::matrix_norm` and the oracle's `_norm_torch_call`
//! (`tools/parity-sweep/oracle.py:905`) calls `torch.linalg.matrix_norm` — i.e.
//! the `norm` key covers ONLY `matrix_norm`. There is NO `vector_norm` dispatch
//! arm and NO `vector_norm` entry in the oracle's `_CUSTOM_OPS`. Sweeping it
//! yields `oracle: unknown op: vector_norm`.
//!
//! This test pins the gap: it asks the live torch oracle to `execute`
//! `vector_norm` (the same `_CUSTOM_OPS` path the decomposition/final arms use)
//! and asserts it succeeds. It FAILS today because no adapter is registered —
//! the failure IS the proof the runner-arm scope is not yet complete. NOTE:
//! `vector_norm` is behaviorally CORRECT on spot inputs (ord=2 -> 13.0, ord=1
//! -> 19.0, ord=inf -> 12.0, all matching torch); this is a COVERAGE gap, not
//! a value divergence — hence the test pins the missing ADAPTER, not a wrong
//! value.
//!
//! Tracking: crosslink #1599 (blocker, parent #1344).
//!
//! Run with:
//!   LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH" \
//!     cargo test -p parity-sweep-runner \
//!     --test divergence_linalg_vector_norm_coverage -- --ignored --nocapture

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use base64::{Engine, engine::general_purpose::STANDARD as B64};
use serde_json::{Value, json};

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

fn tensor_arg(data: &[f32], shape: &[usize]) -> Value {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &x in data {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    json!({
        "__tensor__": {
            "shape": shape,
            "dtype": "float32",
            "data_b64": B64.encode(&bytes),
        }
    })
}

/// PINS THE COVERAGE GAP: the oracle must expose a `vector_norm` custom-op
/// adapter (the same `_CUSTOM_OPS` mechanism `solve`/`qr`/`cholesky`/`inv`/
/// `det`/`slogdet`/`cross` use) so the runner can parity-check
/// `torch.linalg.vector_norm` against ferrotorch's `pub fn vector_norm`.
///
/// FAILS today: `execute("vector_norm", ...)` returns
/// `{"ok": false, "err": "unknown op: vector_norm"}` because no adapter is
/// registered — proving the #1344 runner-arm scope is NOT complete (the commit
/// claims only dot/mv/mm_bt lack arms, omitting vector_norm).
#[test]
#[ignore = "divergence: vector_norm (REQ-23) has no parity-sweep runner arm/oracle adapter; #1344 scope incomplete; tracking #1599"]
fn vector_norm_has_a_runner_arm_or_oracle_adapter() {
    let Some(mut oracle) = OracleProc::spawn() else {
        eprintln!("SKIP: torch oracle unavailable; cannot probe vector_norm coverage");
        return;
    };

    // A [3] vector whose Euclidean norm is exactly 13.0 (3-4-12 Pythagorean
    // quadruple) so the expected value is a NAMED symbolic constant, NOT a
    // ferrotorch-derived value (R-CHAR-3 (b)).
    let v = [3.0f32, -4.0, 12.0];
    let resp = oracle
        .request(json!({
            "cmd": "execute",
            "op": "vector_norm",
            "args": [tensor_arg(&v, &[3])],
            "kwargs": {}
        }))
        .expect("oracle request");

    let ok = resp.get("ok").and_then(Value::as_bool).unwrap_or(false);
    assert!(
        ok,
        "torch.linalg.vector_norm has NO parity-sweep oracle adapter — the #1344 \
         runner-arm scope is INCOMPLETE (commit 381fad746 claims only dot/mv/mm_bt \
         lack arms, but vector_norm (REQ-23) is also unarmed). oracle response: {resp:?}. \
         Tracking: crosslink #1599."
    );

    // If/when the adapter lands, the value must be torch's vector 2-norm = 13.0.
    let out = resp
        .get("output")
        .expect("output present once adapter exists");
    let inner = out
        .get("__tensor__")
        .and_then(Value::as_object)
        .expect("tensor envelope");
    let b64 = inner.get("data_b64").and_then(Value::as_str).unwrap();
    let bytes = B64.decode(b64).unwrap();
    let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    // EXPECTED: ||[3,-4,12]||_2 = sqrt(9+16+144) = sqrt(169) = 13.0 (a named
    // Pythagorean constant, traceable to the vector-2-norm definition, NOT
    // copied from the ferrotorch side).
    const TORCH_VECTOR_2NORM_3_4_12: f32 = 13.0;
    assert!(
        (val - TORCH_VECTOR_2NORM_3_4_12).abs() < 1e-4,
        "torch vector_norm([3,-4,12], ord=2) should be 13.0, got {val}"
    );
}
