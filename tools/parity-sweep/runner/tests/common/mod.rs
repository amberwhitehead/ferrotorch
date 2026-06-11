//! Shared live-torch oracle subprocess helper for the parity-sweep runner's
//! divergence/coverage test suites.
//!
//! CORE-206 (crosslink #1900): each test file used to carry its own copy of
//! `OracleProc` whose `spawn()` silently returned `None` when python3/torch
//! was unavailable — every oracle-backed test then passed GREEN vacuously on
//! any machine without torch. This module is the single fail-closed gate:
//!
//!   - `PARITY_ORACLE_REQUIRED=1`  → oracle-unavailable is a PANIC with full
//!     diagnostics (python binary tried, oracle path, spawn/handshake error).
//!     Set in CI (nightly parity-smoke step) so a missing oracle is RED, not
//!     green.
//!   - unset                       → soft skip is preserved for local dev,
//!     but prints an unmissable single-line `VACUOUS-PASS:` marker.
//!   - `PARITY_PYTHON=<path>`      → overrides the python interpreter
//!     (default `python3` from PATH). Also the sabotage knob for the
//!     R-RED-2 gate proof.

// Each integration-test crate links this module separately and uses a subset
// of the helpers (e.g. the coverage pins never call `execute`), so unused
// items here are expected, not dead weight.
#![allow(dead_code)]

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use base64::{Engine, engine::general_purpose::STANDARD as B64};
use serde_json::{Value, json};

/// When set to `1`, an unavailable torch oracle PANICS instead of skipping.
pub const ORACLE_REQUIRED_ENV: &str = "PARITY_ORACLE_REQUIRED";
/// Overrides the python interpreter used to spawn `oracle.py`.
pub const ORACLE_PYTHON_ENV: &str = "PARITY_PYTHON";

pub struct OracleProc {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl OracleProc {
    /// Spawn the persistent python oracle (`tools/parity-sweep/oracle.py`)
    /// and complete the `ready` handshake (which imports torch + op_db).
    ///
    /// Fail-closed gate (CORE-206 / #1900):
    ///   - oracle available            → `Some(proc)`
    ///   - unavailable, REQUIRED set   → panic with diagnostics
    ///   - unavailable, REQUIRED unset → single-line `VACUOUS-PASS:` marker
    ///     on stderr, then `None` (caller `return`s, test is vacuous).
    pub fn spawn() -> Option<Self> {
        match Self::try_spawn() {
            Ok(p) => Some(p),
            Err(why) => {
                let required = std::env::var(ORACLE_REQUIRED_ENV)
                    .map(|v| v == "1")
                    .unwrap_or(false);
                if required {
                    panic!(
                        "{ORACLE_REQUIRED_ENV}=1 but the torch oracle is UNAVAILABLE — \
                         failing closed instead of skipping.\n  {why}"
                    );
                }
                eprintln!(
                    "VACUOUS-PASS: torch oracle unavailable — set {ORACLE_REQUIRED_ENV}=1 \
                     to fail closed ({why})"
                );
                None
            }
        }
    }

    /// The actual spawn + handshake, with every failure mode mapped to a
    /// human-legible diagnostic string (what was tried, what failed).
    fn try_spawn() -> Result<Self, String> {
        let python = std::env::var(ORACLE_PYTHON_ENV).unwrap_or_else(|_| "python3".to_string());
        // CARGO_MANIFEST_DIR = tools/parity-sweep/runner; oracle.py lives one
        // level up. Resolved at compile time, so the tests do not depend on
        // the harness CWD.
        let oracle_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("runner crate dir has a parent")
            .join("oracle.py");
        if !oracle_path.is_file() {
            return Err(format!(
                "oracle script not found at {}",
                oracle_path.display()
            ));
        }
        let mut child = Command::new(&python)
            .arg(&oracle_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Inherit stderr so a torch import traceback lands in the test
            // output right next to the panic/marker.
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                format!(
                    "failed to spawn `{python} {}`: {e} \
                     (python tried: `{python}`{}; override with {ORACLE_PYTHON_ENV})",
                    oracle_path.display(),
                    if std::env::var(ORACLE_PYTHON_ENV).is_ok() {
                        format!(" from {ORACLE_PYTHON_ENV}")
                    } else {
                        " from PATH".to_string()
                    }
                )
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "oracle child stdin missing".to_string())?;
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| "oracle child stdout missing".to_string())?,
        );
        let mut p = OracleProc {
            child,
            stdin,
            stdout,
        };
        // Handshake: confirm torch is importable (oracle.py imports torch +
        // op_db at module load; an ImportError kills it before responding).
        match p.request(json!({"cmd": "ready"})) {
            Some(ready) if ready.get("ok").and_then(Value::as_bool) == Some(true) => Ok(p),
            Some(ready) => Err(format!(
                "oracle `ready` handshake returned not-ok: {ready} \
                 (python: `{python}`, oracle: {})",
                oracle_path.display()
            )),
            None => Err(format!(
                "oracle process exited before answering the `ready` handshake — \
                 most likely `import torch` failed (traceback on stderr above). \
                 python: `{python}`, oracle: {}",
                oracle_path.display()
            )),
        }
    }

    /// One JSONL request/response round trip.
    pub fn request(&mut self, req: Value) -> Option<Value> {
        let line = serde_json::to_string(&req).ok()?;
        self.stdin.write_all(line.as_bytes()).ok()?;
        self.stdin.write_all(b"\n").ok()?;
        self.stdin.flush().ok()?;
        let mut resp = String::new();
        self.stdout.read_line(&mut resp).ok()?;
        serde_json::from_str(&resp).ok()
    }

    /// Live-torch `execute` of a custom op on N f32 tensor args. Returns the
    /// flattened f32 output + its shape (the gauge-invariant / unique derived
    /// quantity the oracle adapter emits). Args ride the oracle's
    /// `{"__tensor__": {shape, dtype, data_b64}}` wire envelope.
    pub fn execute(
        &mut self,
        op: &str,
        args: &[(&[f32], &[usize])],
    ) -> Option<(Vec<f32>, Vec<usize>)> {
        let json_args: Vec<Value> = args
            .iter()
            .map(|(data, shape)| tensor_arg(data, shape))
            .collect();
        let resp =
            self.request(json!({"cmd": "execute", "op": op, "args": json_args, "kwargs": {}}))?;
        if resp.get("ok").and_then(Value::as_bool) != Some(true) {
            eprintln!("oracle execute {op} err: {resp:?}");
            return None;
        }
        decode_tensor(resp.get("output")?)
    }

    /// `execute` specialised to a single 2-D f32 matrix arg (the
    /// decomposition-gauge suites' shape).
    pub fn execute_mat(
        &mut self,
        op: &str,
        data: &[f32],
        shape: &[usize],
    ) -> Option<(Vec<f32>, Vec<usize>)> {
        self.execute(op, &[(data, shape)])
    }
}

impl Drop for OracleProc {
    fn drop(&mut self) {
        let _ = self.request(json!({"cmd": "shutdown"}));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Encode a flat f32 buffer + shape as the oracle's
/// `{"__tensor__": {shape, dtype, data_b64}}` wire envelope.
pub fn tensor_arg(data: &[f32], shape: &[usize]) -> Value {
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

/// Decode the oracle's `{"__tensor__": {shape, dtype, data_b64}}` envelope to
/// (flattened f32, shape). The custom adapters emit f32 (int64 widened to f32
/// for index-style outputs).
pub fn decode_tensor(v: &Value) -> Option<(Vec<f32>, Vec<usize>)> {
    let obj = v.as_object()?;
    let inner = obj.get("__tensor__").and_then(Value::as_object)?;
    let shape: Vec<usize> = inner
        .get("shape")?
        .as_array()?
        .iter()
        .map(|x| x.as_u64().unwrap_or(0) as usize)
        .collect();
    let b64 = inner.get("data_b64").and_then(Value::as_str)?;
    let bytes = B64.decode(b64).ok()?;
    let dtype = inner
        .get("dtype")
        .and_then(Value::as_str)
        .unwrap_or("float32");
    let data: Vec<f32> = match dtype {
        "float32" => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "int64" => bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        other => {
            eprintln!("decode_tensor: unexpected dtype {other}");
            return None;
        }
    };
    Some((data, shape))
}
