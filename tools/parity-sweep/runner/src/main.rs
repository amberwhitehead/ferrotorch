//! ferrotorch ↔ PyTorch parity sweep runner.
//!
//! Spawns the persistent torch oracle (`tools/parity-sweep/oracle.py`), asks
//! it for sampled inputs from `torch.testing._internal.op_db`, runs the same
//! inputs through ferrotorch via the local dispatch table, and diffs the
//! outputs under per-dtype tolerances. No fixtures are stored on disk — every
//! sweep regenerates inputs from a fresh seed so the input space is *swept*,
//! not snapshotted.
//!
//! CLI:
//!
//!   parity-sweep list-ops
//!   parity-sweep sweep --op add [--seeds 32] [--samples-per-seed all]
//!   parity-sweep dispatch          # list ops the Rust dispatch table covers
//!
//! Exit 0 if all probed inputs matched; exit 1 on any divergence.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use base64::{Engine, engine::general_purpose::STANDARD as B64};
use ferrotorch_core::{Tensor, from_vec, grad_fns};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Wire format (matches oracle.py)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WireTensor {
    shape: Vec<usize>,
    dtype: String,
    data_b64: String,
}

impl WireTensor {
    fn to_f32(&self) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
        if self.dtype != "float32" {
            return Err(format!("dispatch supports float32 only, got {}", self.dtype).into());
        }
        let bytes = B64.decode(&self.data_b64)?;
        let expected = self.shape.iter().product::<usize>() * 4;
        if bytes.len() != expected {
            return Err(format!(
                "byte length {} does not match shape {:?} (expected {})",
                bytes.len(),
                self.shape,
                expected
            )
            .into());
        }
        let mut data = Vec::with_capacity(self.shape.iter().product());
        for chunk in bytes.chunks_exact(4) {
            data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(from_vec(data, &self.shape)?)
    }
}

/// An arg returned by the oracle — either a tensor envelope or a JSON scalar.
fn unwrap_tensor_arg(v: &Value) -> Option<WireTensor> {
    let envelope = v.as_object()?.get("__tensor__")?;
    serde_json::from_value(envelope.clone()).ok()
}

// ---------------------------------------------------------------------------
// Oracle subprocess (persistent python + JSONL stdio)
// ---------------------------------------------------------------------------

struct Oracle {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Oracle {
    fn spawn() -> Result<Self, Box<dyn std::error::Error>> {
        // CARGO_MANIFEST_DIR = tools/parity-sweep/runner
        let oracle_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("runner crate has no parent dir")?
            .join("oracle.py");
        if !oracle_path.is_file() {
            return Err(format!("oracle not found at {}", oracle_path.display()).into());
        }
        let mut child = Command::new("python3")
            .arg(&oracle_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Forward stderr to ours so torch import errors are visible.
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child.stdin.take().ok_or("oracle stdin missing")?;
        let stdout = BufReader::new(child.stdout.take().ok_or("oracle stdout missing")?);
        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    fn call(&mut self, req: Value) -> Result<Value, Box<dyn std::error::Error>> {
        self.stdin.write_all(req.to_string().as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let mut line = String::new();
        let n = self.stdout.read_line(&mut line)?;
        if n == 0 {
            return Err("oracle EOF before response".into());
        }
        let resp: Value = serde_json::from_str(&line)?;
        if resp.get("ok").and_then(Value::as_bool) != Some(true) {
            let err = resp
                .get("err")
                .and_then(Value::as_str)
                .unwrap_or("(no err)");
            return Err(format!("oracle: {err}").into());
        }
        Ok(resp)
    }

    fn ready(&mut self) -> Result<(String, usize), Box<dyn std::error::Error>> {
        let r = self.call(json!({"cmd": "ready"}))?;
        let ver = r
            .get("torch")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();
        let n = r.get("ops").and_then(Value::as_u64).unwrap_or(0) as usize;
        Ok((ver, n))
    }

    fn list_ops(&mut self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let r = self.call(json!({"cmd": "list_ops"}))?;
        Ok(r.get("ops")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default())
    }

    fn sample(
        &mut self,
        op: &str,
        seed: u64,
        i: usize,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        self.call(json!({"cmd": "sample", "op": op, "seed": seed, "i": i}))
    }

    /// Execute an adversarial probe (discriminator pass) on the torch side.
    /// Returns the full response — caller inspects `output`, `grads`, or `ok`.
    fn probe(&mut self, op: &str, spec: &Value) -> Result<Value, Box<dyn std::error::Error>> {
        // Bypass `Self::call`'s err-throwing wrapper so we can also report
        // expected-error probes (e.g. dtype mismatches that torch rejects).
        self.stdin.write_all(
            json!({"cmd": "probe", "op": op, "spec": spec})
                .to_string()
                .as_bytes(),
        )?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let mut line = String::new();
        let n = self.stdout.read_line(&mut line)?;
        if n == 0 {
            return Err("oracle EOF before probe response".into());
        }
        Ok(serde_json::from_str(&line)?)
    }

    fn shutdown(mut self) {
        let _ = self.call(json!({"cmd": "shutdown"}));
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------------------
// ferrotorch dispatch table (op name -> closure)
// ---------------------------------------------------------------------------

/// Run ferrotorch's implementation of `op` on the given args/kwargs from the
/// oracle. Returns `Ok(Some(tensor))` on a successful f32 dispatch, `Ok(None)`
/// when the op is not (yet) covered by the Rust dispatch table, and `Err` for
/// runtime failures inside ferrotorch.
fn dispatch_f32(
    op: &str,
    args: &[Value],
    kwargs: &serde_json::Map<String, Value>,
) -> Result<Option<Tensor<f32>>, Box<dyn std::error::Error>> {
    // Helper: extract a 2-tensor binary op's inputs.
    let binary = |name: &str| -> Result<(Tensor<f32>, Tensor<f32>), Box<dyn std::error::Error>> {
        if args.len() < 2 {
            return Err(format!("{name} expects 2 args, got {}", args.len()).into());
        }
        let a = unwrap_tensor_arg(&args[0])
            .ok_or_else(|| format!("{name} arg 0 not a tensor"))?
            .to_f32()?;
        let b = unwrap_tensor_arg(&args[1])
            .ok_or_else(|| format!("{name} arg 1 not a tensor"))?
            .to_f32()?;
        Ok((a, b))
    };
    let unary = |name: &str| -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
        if args.is_empty() {
            return Err(format!("{name} expects 1 arg, got 0").into());
        }
        let t = unwrap_tensor_arg(&args[0])
            .ok_or_else(|| format!("{name} arg 0 not a tensor"))?
            .to_f32()?;
        Ok(t)
    };
    // PyTorch's `torch.add(input, other, *, alpha=1)` (and friends) ships
    // `alpha` as a JSON number in the kwargs envelope. op_db only emits a
    // scalar here, so a Number is sufficient — defaults to 1.0 when absent.
    let alpha_kwarg = |name: &str| -> Result<f64, Box<dyn std::error::Error>> {
        match kwargs.get("alpha") {
            None => Ok(1.0),
            Some(v) => v
                .as_f64()
                .ok_or_else(|| format!("{name}: alpha kwarg is not a JSON number: {v}").into()),
        }
    };

    match op {
        // Binary arithmetic. `torch.add(input, other, *, alpha=1)` is routed
        // through `arithmetic::add_scaled`; the alpha==1 case forwards to the
        // existing `add` path with no extra allocation. `torch.sub(input,
        // other, *, alpha=1)` routes through `arithmetic::sub_scaled` which
        // delegates to `add_scaled(a, b, -alpha)` (matches PyTorch's own
        // `TORCH_IMPL_FUNC(sub_out)` at `aten/src/ATen/native/BinaryOps.cpp:434`).
        "add" => Ok(Some({
            let (a, b) = binary("add")?;
            let alpha = alpha_kwarg("add")?;
            grad_fns::arithmetic::add_scaled(&a, &b, alpha)?
        })),
        "sub" => Ok(Some({
            let (a, b) = binary("sub")?;
            let alpha = alpha_kwarg("sub")?;
            grad_fns::arithmetic::sub_scaled(&a, &b, alpha)?
        })),
        // `torch.rsub(input, other, *, alpha=1)` — operand-swap delegation
        // to sub per upstream `aten/src/ATen/native/BinaryOps.cpp:1169`:
        // `Tensor rsub(self, other, alpha) { return at::sub(other, self,
        // alpha); }`. ferrotorch's `arithmetic::rsub` mirrors this via
        // `sub_scaled(b, a, alpha)` (R-DEV-1 byte-for-byte). op_db emits
        // `rsub` (and `__rsub__`) samples; the `__rsub__` magic-method
        // dispatch falls through here too because the wire args are
        // identical (`args = [input, other]`, `kwargs = {alpha?}`).
        "rsub" => Ok(Some({
            let (a, b) = binary("rsub")?;
            let alpha = alpha_kwarg("rsub")?;
            grad_fns::arithmetic::rsub(&a, &b, alpha)?
        })),
        "mul" => Ok(Some({
            let (a, b) = binary("mul")?;
            grad_fns::arithmetic::mul(&a, &b)?
        })),
        "div" => Ok(Some({
            let (a, b) = binary("div")?;
            grad_fns::arithmetic::div(&a, &b)?
        })),
        // Unary
        "neg" => Ok(Some(grad_fns::arithmetic::neg(&unary("neg")?)?)),
        "abs" => Ok(Some(grad_fns::arithmetic::abs(&unary("abs")?)?)),
        "sqrt" => Ok(Some(grad_fns::arithmetic::sqrt(&unary("sqrt")?)?)),
        // `torch.rsqrt(input, *, out=None)` — `_torch_docs.py:9656`.
        // ferrotorch's `arithmetic::rsqrt<T: Float>(a)` mirrors the upstream
        // unary at `aten/src/ATen/native/UnaryOps.cpp:346
        // CREATE_UNARY_TORCH_IMPL_FUNC(rsqrt_out, rsqrt_stub)` with
        // backward `-0.5 * grad * c^3` per `tools/autograd/derivatives.yaml:1505`.
        // Closes blocker #1195.
        "rsqrt" => Ok(Some(grad_fns::arithmetic::rsqrt(&unary("rsqrt")?)?)),
        // `torch.reciprocal(input, *, out=None)` — `_torch_docs.py:2584`.
        // ferrotorch's `arithmetic::reciprocal<T: Float>(a)` mirrors the
        // upstream unary at `aten/src/ATen/native/UnaryOps.cpp:345
        // CREATE_UNARY_TORCH_IMPL_FUNC(reciprocal_out, reciprocal_stub)`
        // with backward `-grad * c^2` per
        // `tools/autograd/derivatives.yaml:1447-1449
        // self: -grad * (result * result).conj()`. Closes blocker #1196.
        "reciprocal" => Ok(Some(grad_fns::arithmetic::reciprocal(&unary(
            "reciprocal",
        )?)?)),
        // `torch.pow(input, exponent, *, out=None)` — `_torch_docs.py:8672`.
        // ferrotorch's `arithmetic::pow<T: Float>(a, exp: f64)` mirrors the
        // scalar-exponent overload at `aten/src/ATen/native/Pow.cpp:51
        // TORCH_IMPL_FUNC(pow_Tensor_Scalar_out)`. op_db emits pow samples
        // where args[1] is *always* a tensor envelope — but a 0-d tensor
        // models the scalar exponent path (shape == []). We dispatch the
        // 0-d-exp case by extracting the single float and forwarding to
        // `arithmetic::pow(&base, exp as f64)`; non-0-d exp tensors are a
        // legitimate skip (the tensor-exponent overload corresponds to
        // `pow_Tensor_Tensor_out` at `Pow.cpp:47`, which ferrotorch has not
        // implemented). A scalar-base sample (args[0] not a tensor) is
        // likewise skipped — `arithmetic::pow` takes `&Tensor<T>` as base.
        "pow" => {
            if args.len() < 2 {
                return Err(format!("pow expects 2 args, got {}", args.len()).into());
            }
            let base = match unwrap_tensor_arg(&args[0]) {
                Some(t) => t.to_f32()?,
                None => return Ok(None),
            };
            let exp_wire = match unwrap_tensor_arg(&args[1]) {
                Some(w) => w,
                None => return Ok(None),
            };
            // Only 0-d exponent tensors (shape == []) collapse to the
            // scalar-exp dispatch. Any other shape is the tensor-exponent
            // overload (broadcasting between base and exp), which is out of
            // scope for the scalar-exp `arithmetic::pow`.
            if !exp_wire.shape.is_empty() {
                return Ok(None);
            }
            let exp_tensor = exp_wire.to_f32()?;
            let exp_data = exp_tensor.data_vec()?;
            let exp_scalar = match exp_data.first() {
                Some(&v) => v as f64,
                None => return Err("pow: 0-d exponent tensor decoded to empty data".into()),
            };
            Ok(Some(grad_fns::arithmetic::pow(&base, exp_scalar)?))
        }
        _ => Ok(None),
    }
}

fn dispatch_ops() -> &'static [&'static str] {
    &[
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "sqrt",
        "pow",
        "rsub",
        "rsqrt",
        "reciprocal",
    ]
}

// ---------------------------------------------------------------------------
// Adversarial probe materialization (discriminator pass)
// ---------------------------------------------------------------------------
//
// The probe spec language mirrors the oracle's `_materialize_tensor`:
//   {"kind":"tensor","shape":[...],"dtype":"float32","data":[...],"fill":<n>,
//    "transform":"none|transpose|expand|slice_step","transform_args":{...}}
//
// Special float tokens in `data` / `fill` (since JSON has no NaN/Inf literals):
//   "NaN", "+Inf", "-Inf", "+0", "-0", "DENORM" (= f32::MIN_POSITIVE / 2).

fn resolve_scalar_token(v: &Value) -> Result<f32, Box<dyn std::error::Error>> {
    if let Some(s) = v.as_str() {
        Ok(match s {
            "NaN" => f32::NAN,
            "+Inf" => f32::INFINITY,
            "-Inf" => f32::NEG_INFINITY,
            "+0" => 0.0,
            "-0" => -0.0,
            // f32::MIN_POSITIVE / 2 — a true subnormal.
            "DENORM" => f32::MIN_POSITIVE / 2.0,
            // Exact f32::MAX as a token: round-trip-safe across JSON, vs.
            // hand-written 3.4028234663852886e38 which serde re-emits with
            // truncated precision and ends up slightly *above* f32::MAX.
            "F32_MAX" => f32::MAX,
            "-F32_MAX" => f32::MIN, // f32::MIN == -f32::MAX (the most negative finite)
            other => return Err(format!("unknown float token: {other}").into()),
        })
    } else if let Some(n) = v.as_f64() {
        Ok(n as f32)
    } else if let Some(n) = v.as_i64() {
        Ok(n as f32)
    } else {
        Err(format!("scalar not a number or token: {v}").into())
    }
}

/// Materialize a ferrotorch f32 tensor from a probe spec. Returns Ok(None) if
/// the spec asks for a dtype the dispatch table cannot consume — the probe
/// then becomes an "expected unsupported" finding rather than a divergence.
fn materialize_ferrotorch_tensor(
    spec: &Value,
) -> Result<Option<Tensor<f32>>, Box<dyn std::error::Error>> {
    let obj = spec.as_object().ok_or("tensor spec not an object")?;
    let dtype = obj
        .get("dtype")
        .and_then(Value::as_str)
        .unwrap_or("float32");
    if dtype != "float32" {
        // Non-f32 inputs (f64, int, bool) are out of dispatch_f32's domain.
        // Treat as "ferrotorch dispatch declines" rather than a crash.
        return Ok(None);
    }
    let shape: Vec<usize> = obj
        .get("shape")
        .and_then(Value::as_array)
        .ok_or("tensor spec missing shape")?
        .iter()
        .map(|v| v.as_u64().map(|u| u as usize).ok_or("shape dim not u64"))
        .collect::<Result<_, _>>()?;
    let mut numel = 1usize;
    for d in &shape {
        numel = numel.saturating_mul(*d);
    }

    let data_vals: Vec<f32> = if let Some(arr) = obj.get("data").and_then(Value::as_array) {
        if numel != arr.len() && numel != 0 {
            return Err(format!(
                "data len {} != shape numel {} ({:?})",
                arr.len(),
                numel,
                shape
            )
            .into());
        }
        arr.iter()
            .map(resolve_scalar_token)
            .collect::<Result<_, _>>()?
    } else if let Some(f) = obj.get("fill") {
        if f.is_null() {
            vec![0.0; numel]
        } else {
            let v = resolve_scalar_token(f)?;
            vec![v; numel]
        }
    } else {
        vec![0.0; numel]
    };

    // For empty tensors with numel == 0 the vector is empty but shape can still
    // be e.g. [0, 5]; from_vec handles this fine.
    let base = from_vec(data_vals, &shape)?;

    // Apply transform.
    let transform = obj
        .get("transform")
        .and_then(Value::as_str)
        .unwrap_or("none");
    let targs = obj.get("transform_args").and_then(Value::as_object);
    let transformed = match transform {
        "none" => base,
        "transpose" => {
            let ta = targs.ok_or("transpose needs transform_args")?;
            let dim0 = ta.get("dim0").and_then(Value::as_u64).ok_or("dim0")? as usize;
            let dim1 = ta.get("dim1").and_then(Value::as_u64).ok_or("dim1")? as usize;
            base.transpose(dim0, dim1)?
        }
        "expand" => {
            let ta = targs.ok_or("expand needs transform_args")?;
            let new_shape: Vec<usize> = ta
                .get("shape")
                .and_then(Value::as_array)
                .ok_or("expand.shape missing")?
                .iter()
                .map(|v| v.as_u64().map(|u| u as usize).ok_or("expand dim not u64"))
                .collect::<Result<_, _>>()?;
            grad_fns::shape::expand(&base, &new_shape)?
        }
        "slice_step" => {
            // No public step-slice helper; emulate via index_select over the
            // strided indices. Limited to 1-D inputs (matches probe spec).
            if base.shape().len() != 1 {
                return Err("slice_step probe only supports 1-D tensors".into());
            }
            let ta = targs.ok_or("slice_step needs transform_args")?;
            let start = ta.get("start").and_then(Value::as_i64).unwrap_or(0) as usize;
            let stop = ta
                .get("stop")
                .and_then(Value::as_i64)
                .unwrap_or(base.shape()[0] as i64) as usize;
            let step = ta.get("step").and_then(Value::as_i64).unwrap_or(1) as usize;
            let raw = base.data_vec()?;
            let mut out = Vec::new();
            let mut i = start;
            while i < stop && i < raw.len() {
                out.push(raw[i]);
                i += step;
            }
            from_vec(out.clone(), &[out.len()])?
        }
        other => return Err(format!("unknown transform: {other}").into()),
    };

    Ok(Some(transformed))
}

/// Run one adversarial probe through ferrotorch's `add`/`add_scaled` (and the
/// in-place / out= variants when requested). Returns the resulting tensor or
/// an error string suitable for embedding in the findings file.
fn run_probe_ferrotorch(spec: &Value) -> Result<Option<Tensor<f32>>, String> {
    let args_spec = spec
        .get("args_spec")
        .and_then(Value::as_array)
        .ok_or_else(|| "probe missing args_spec".to_string())?;
    let kwargs = spec
        .get("kwargs")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    if args_spec.len() < 2 {
        return Err(format!("add probe needs 2 args, got {}", args_spec.len()));
    }

    // ALIAS_A: second arg referencing first tensor (self-add probes).
    let a = match materialize_ferrotorch_tensor(&args_spec[0]) {
        Ok(Some(t)) => t,
        Ok(None) => return Ok(None), // dtype-skip
        Err(e) => return Err(format!("materialize arg 0: {e}")),
    };
    let b = match &args_spec[1] {
        Value::String(s) if s == "ALIAS_A" => a.clone(),
        other => match materialize_ferrotorch_tensor(other) {
            Ok(Some(t)) => t,
            Ok(None) => return Ok(None),
            Err(e) => return Err(format!("materialize arg 1: {e}")),
        },
    };

    // requires_grad on each input (autograd probes).
    let rg = kwargs.get("requires_grad").and_then(Value::as_array);
    let (a, b) = if let Some(rg) = rg {
        let want_a = rg.first().and_then(Value::as_bool).unwrap_or(false);
        let want_b = rg.get(1).and_then(Value::as_bool).unwrap_or(false);
        let a = if want_a { a.requires_grad_(true) } else { a };
        let b = if want_b { b.requires_grad_(true) } else { b };
        (a, b)
    } else {
        (a, b)
    };

    let alpha = match kwargs.get("alpha") {
        None => 1.0f64,
        Some(v) => match v {
            Value::Number(n) => n.as_f64().ok_or_else(|| format!("alpha not f64: {n}"))?,
            Value::String(s) => match s.as_str() {
                "NaN" => f64::NAN,
                "+Inf" => f64::INFINITY,
                "-Inf" => f64::NEG_INFINITY,
                "+0" => 0.0,
                "-0" => -0.0,
                other => return Err(format!("unknown alpha token: {other}")),
            },
            other => return Err(format!("alpha not number/string: {other}")),
        },
    };

    let inplace = kwargs
        .get("inplace")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let out_spec = kwargs.get("out_spec");

    // In-place: ferrotorch's `add_scaled_` mirrors torch's
    // `add_(other, *, alpha=1)` — supports broadcasting + alpha. `add_` is
    // the `alpha == 1.0` convenience alias.
    if inplace {
        a.add_scaled_(&b, alpha)
            .map_err(|e| format!("add_scaled_: {e}"))?;
        return Ok(Some(a));
    }

    // `out=` kwarg: ferrotorch exposes `add_scaled_out(&out, &a, &b, alpha)`
    // which writes the result into a caller-allocated tensor (matching torch's
    // `torch.add(a, b, *, out=out)` semantics). We materialize the `out`
    // tensor from its envelope (preserving its fill — e.g. NaN for the
    // "must not leak NaN" probe), run the op, and return that tensor.
    if let Some(out_spec_v) = out_spec {
        let out_tensor = match materialize_ferrotorch_tensor(out_spec_v) {
            Ok(Some(t)) => t,
            Ok(None) => return Ok(None), // dtype-skip for the out= envelope
            Err(e) => return Err(format!("materialize out_spec: {e}")),
        };
        grad_fns::arithmetic::add_scaled_out(&out_tensor, &a, &b, alpha)
            .map_err(|e| format!("add_scaled_out: {e}"))?;
        return Ok(Some(out_tensor));
    }

    let result =
        grad_fns::arithmetic::add_scaled(&a, &b, alpha).map_err(|e| format!("add_scaled: {e}"))?;
    Ok(Some(result))
}

/// Run the backward pass for an autograd probe and return the gradients.
fn run_probe_ferrotorch_grads(spec: &Value) -> Result<Vec<Option<Tensor<f32>>>, String> {
    let args_spec = spec
        .get("args_spec")
        .and_then(Value::as_array)
        .ok_or_else(|| "probe missing args_spec".to_string())?;
    let kwargs = spec
        .get("kwargs")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let rg = kwargs
        .get("requires_grad")
        .and_then(Value::as_array)
        .ok_or_else(|| "autograd_check requires kwargs.requires_grad".to_string())?;

    let a = materialize_ferrotorch_tensor(&args_spec[0])
        .map_err(|e| format!("materialize arg 0: {e}"))?
        .ok_or_else(|| "dtype skip".to_string())?;
    let b = match &args_spec[1] {
        Value::String(s) if s == "ALIAS_A" => a.clone(),
        other => materialize_ferrotorch_tensor(other)
            .map_err(|e| format!("materialize arg 1: {e}"))?
            .ok_or_else(|| "dtype skip".to_string())?,
    };

    let want_a = rg.first().and_then(Value::as_bool).unwrap_or(false);
    let want_b = rg.get(1).and_then(Value::as_bool).unwrap_or(false);
    let a = if want_a { a.requires_grad_(true) } else { a };
    let b = if want_b { b.requires_grad_(true) } else { b };

    let alpha = kwargs.get("alpha").and_then(Value::as_f64).unwrap_or(1.0);

    let out = grad_fns::arithmetic::add_scaled(&a, &b, alpha)
        .map_err(|e| format!("add_scaled fwd: {e}"))?;
    let scalar = grad_fns::reduction::sum(&out).map_err(|e| format!("sum: {e}"))?;
    ferrotorch_core::autograd::graph::backward(&scalar).map_err(|e| format!("backward: {e}"))?;

    let ga = if want_a {
        a.grad().map_err(|e| format!("a.grad: {e}"))?
    } else {
        None
    };
    let gb = if want_b {
        b.grad().map_err(|e| format!("b.grad: {e}"))?
    } else {
        None
    };
    Ok(vec![ga, gb])
}

/// NaN-aware bit-similar comparison for probe outputs. Stricter than the
/// op_db sweep: every NaN position must match exactly, ±0 must match exactly,
/// ±Inf must match exactly. Finite values compared with relaxed tolerance.
fn compare_probe_outputs(
    actual: &[f32],
    expected: &[f32],
    actual_shape: &[usize],
    expected_shape: &[usize],
) -> Option<String> {
    if actual_shape != expected_shape {
        return Some(format!(
            "shape mismatch: ferrotorch {actual_shape:?} vs torch {expected_shape:?}"
        ));
    }
    if actual.len() != expected.len() {
        return Some(format!(
            "len mismatch: ferrotorch {} vs torch {}",
            actual.len(),
            expected.len()
        ));
    }
    let (rtol, atol) = (1e-5_f32, 1e-7_f32);
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() != e.is_nan() {
            return Some(format!(
                "NaN mismatch at index {i}: ferrotorch={a} vs torch={e}"
            ));
        }
        if a.is_nan() && e.is_nan() {
            continue;
        }
        if a.is_infinite() || e.is_infinite() {
            if a.is_infinite() != e.is_infinite() || a.signum() != e.signum() {
                return Some(format!(
                    "Inf mismatch at index {i}: ferrotorch={a} vs torch={e}"
                ));
            }
            continue;
        }
        // ±0 distinction.
        if a == 0.0 && e == 0.0 && a.is_sign_negative() != e.is_sign_negative() {
            return Some(format!(
                "signed-zero mismatch at index {i}: ferrotorch={a:+} vs torch={e:+}"
            ));
        }
        let diff = (a - e).abs();
        let bound = atol + rtol * e.abs();
        if diff > bound {
            return Some(format!(
                "value mismatch at index {i}: ferrotorch={a} vs torch={e} (diff={diff})"
            ));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Comparison (per-dtype tolerance, matches torch.testing.assert_close defaults)
// ---------------------------------------------------------------------------

fn tol_f32() -> (f32, f32) {
    // torch.testing.assert_close defaults for float32: rtol=1.3e-6, atol=1e-5.
    // Loosened slightly for cross-impl numerical drift.
    (1e-5, 1e-7)
}

fn assert_close_f32(actual: &Tensor<f32>, expected_wire: &WireTensor) -> Result<(), String> {
    let expected = expected_wire
        .to_f32()
        .map_err(|e| format!("decode expected: {e}"))?;
    if actual.shape() != expected.shape() {
        return Err(format!(
            "shape mismatch: ferrotorch {:?} vs torch {:?}",
            actual.shape(),
            expected.shape()
        ));
    }
    let actual_data = actual
        .data()
        .map_err(|e| format!("ferrotorch tensor.data() failed: {e}"))?;
    let expected_data = expected
        .data()
        .map_err(|e| format!("expected tensor.data() failed: {e}"))?;
    let (rtol, atol) = tol_f32();
    let mut worst: Option<(usize, f32, f32, f32)> = None;
    for (i, (&a, &e)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        // NaN handling: torch treats NaN == NaN as failure unless equal_nan=True.
        // Default: NaN positions must match (both NaN or neither).
        if a.is_nan() || e.is_nan() {
            if a.is_nan() != e.is_nan() {
                return Err(format!(
                    "NaN mismatch at index {i}: ferrotorch={a} vs torch={e}"
                ));
            }
            continue;
        }
        let diff = (a - e).abs();
        let bound = atol + rtol * e.abs();
        if diff > bound && worst.is_none_or(|(_, _, _, w)| diff > w) {
            worst = Some((i, a, e, diff));
        }
    }
    if let Some((i, a, e, diff)) = worst {
        return Err(format!(
            "value mismatch at index {i}: ferrotorch={a} vs torch={e} (diff={diff})"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sweep
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct SweepReport {
    op: String,
    samples_attempted: usize,
    samples_passed: usize,
    samples_skipped: usize,
    failures: Vec<String>,
}

impl SweepReport {
    fn print(&self) {
        println!(
            "\n[{op}] {pass}/{attempt} passed ({skip} skipped, {fail} failed)",
            op = self.op,
            pass = self.samples_passed,
            attempt = self.samples_attempted,
            skip = self.samples_skipped,
            fail = self.failures.len(),
        );
        for f in &self.failures {
            println!("  FAIL: {f}");
        }
    }
}

fn sweep(
    oracle: &mut Oracle,
    op: &str,
    seeds: u64,
) -> Result<SweepReport, Box<dyn std::error::Error>> {
    sweep_with_cap(oracle, op, seeds, 1024)
}

fn sweep_with_cap(
    oracle: &mut Oracle,
    op: &str,
    seeds: u64,
    max_samples_per_seed: usize,
) -> Result<SweepReport, Box<dyn std::error::Error>> {
    let mut report = SweepReport {
        op: op.to_string(),
        ..Default::default()
    };
    for seed in 0..seeds {
        // op_db's sample_inputs yields a fixed list per (op, seed, dtype). We
        // walk it index-by-index until the oracle reports we've exhausted it
        // or we hit max_samples_per_seed (so sweep-all stays bounded).
        for i in 0..max_samples_per_seed {
            let resp = oracle.sample(op, seed, i);
            let resp = match resp {
                Ok(r) => r,
                Err(e) => {
                    let s = e.to_string();
                    if s.contains(">= ") && s.contains("samples for") {
                        break; // exhausted this seed
                    }
                    report
                        .failures
                        .push(format!("seed={seed} i={i} oracle: {s}"));
                    break;
                }
            };

            let args = resp
                .get("args")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let empty = serde_json::Map::new();
            let kwargs = resp
                .get("kwargs")
                .and_then(Value::as_object)
                .unwrap_or(&empty);
            let expected_v = resp.get("output").cloned().unwrap_or(Value::Null);
            let expected = match unwrap_tensor_arg(&expected_v) {
                Some(t) => t,
                None => {
                    report.samples_skipped += 1;
                    continue;
                }
            };

            report.samples_attempted += 1;
            let dispatched = dispatch_f32(op, &args, kwargs);
            match dispatched {
                Ok(None) => {
                    report.samples_skipped += 1;
                }
                Ok(Some(actual)) => match assert_close_f32(&actual, &expected) {
                    Ok(()) => report.samples_passed += 1,
                    Err(e) => report
                        .failures
                        .push(format!("seed={seed} i={i} shape={:?}: {e}", expected.shape)),
                },
                Err(e) => report
                    .failures
                    .push(format!("seed={seed} i={i} ferrotorch raised: {e}")),
            }
        }
    }
    Ok(report)
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn usage() -> ! {
    eprintln!(
        "usage:\n  \
         parity-sweep list-ops            # list ops the torch oracle exposes\n  \
         parity-sweep dispatch            # list ops the Rust dispatch table covers\n  \
         parity-sweep sweep --op <name> [--seeds N]\n  \
         parity-sweep sweep-all [--seeds N] [--limit N]   # sweep every op_db op\n  \
         parity-sweep probe --op <name> --probes <jsonl> --out <findings.json>\n  \
                                          # discriminator: per-line probe spec, diffs torch vs ferrotorch"
    );
    std::process::exit(2);
}

#[derive(Serialize)]
struct OpCoverage {
    op: String,
    status: &'static str,
    attempted: usize,
    passed: usize,
    failed: usize,
    skipped: usize,
    first_failure: Option<String>,
}

fn sweep_all(
    oracle: &mut Oracle,
    seeds: u64,
    limit: Option<usize>,
    max_samples_per_seed: usize,
    checkpoint_path: &std::path::Path,
) -> Result<Vec<OpCoverage>, Box<dyn std::error::Error>> {
    let mut ops = oracle.list_ops()?;
    if let Some(n) = limit {
        ops.truncate(n);
    }
    let total = ops.len();
    let mut results = Vec::with_capacity(total);
    for (idx, op) in ops.iter().enumerate() {
        eprint!("\r[{:>4}/{total}] {op:30}", idx + 1);
        use std::io::Write as _;
        let _ = std::io::stderr().flush();

        let report = match sweep_with_cap(oracle, op, seeds, max_samples_per_seed) {
            Ok(r) => r,
            Err(e) => {
                results.push(OpCoverage {
                    op: op.clone(),
                    status: "oracle_error",
                    attempted: 0,
                    passed: 0,
                    failed: 0,
                    skipped: 0,
                    first_failure: Some(e.to_string()),
                });
                continue;
            }
        };
        // "executed" = samples ferrotorch actually ran. Anything else (oracle
        // gave us nothing, or dispatch returned Ok(None)) is NOT verification.
        let executed = report.samples_passed + report.failures.len();
        // Failures fall into two buckets that look the same in `failures`: the
        // oracle could not produce a sample (torch threw / unsupported encoding)
        // vs ferrotorch disagreed. Distinguish them: oracle errors leave
        // samples_attempted == 0 since the increment happens after a successful
        // oracle response.
        let status = if !report.failures.is_empty() && report.samples_attempted == 0 {
            "oracle_error"
        } else if !report.failures.is_empty() {
            "diverges"
        } else if report.samples_attempted == 0 && report.samples_skipped == 0 {
            "torch_no_samples"
        } else if executed == 0 {
            // Got samples from torch, but dispatch returned Ok(None) for all of
            // them. Op exists in op_db; ferrotorch dispatch doesn't know it yet.
            "no_dispatch"
        } else {
            // At least one sample passed and zero failed. NOTE: at the default
            // --max-samples=4 this is a triage signal, not a deep verification —
            // re-run `sweep --op <name> --seeds N` for confidence.
            "passes_quick"
        };
        results.push(OpCoverage {
            op: op.clone(),
            status,
            attempted: report.samples_attempted,
            passed: report.samples_passed,
            failed: report.failures.len(),
            skipped: report.samples_skipped,
            first_failure: report.failures.first().cloned(),
        });

        // Checkpoint every 25 ops so a kill/timeout leaves recoverable output.
        if ((idx + 1) % 25 == 0 || idx + 1 == total)
            && let Ok(json) = serde_json::to_string_pretty(&results)
        {
            let _ = std::fs::write(checkpoint_path, json);
        }
    }
    eprintln!();
    Ok(results)
}

fn print_coverage_summary(results: &[OpCoverage]) {
    let mut by_status: std::collections::BTreeMap<&str, usize> = Default::default();
    for r in results {
        *by_status.entry(r.status).or_insert(0) += 1;
    }
    println!(
        "\n=== coverage summary ({} ops in op_db) ===",
        results.len()
    );
    println!("  status               count   meaning");
    println!(
        "  passes_quick         {:>5}   ferrotorch matched torch on every executed sample at this sweep depth",
        by_status.get("passes_quick").unwrap_or(&0)
    );
    println!(
        "  diverges             {:>5}   at least one sample disagreed",
        by_status.get("diverges").unwrap_or(&0)
    );
    println!(
        "  no_dispatch          {:>5}   exists in op_db; ferrotorch Rust dispatch returns None",
        by_status.get("no_dispatch").unwrap_or(&0)
    );
    println!(
        "  torch_no_samples     {:>5}   op_db produced 0 samples (op needs special invocation)",
        by_status.get("torch_no_samples").unwrap_or(&0)
    );
    println!(
        "  oracle_error         {:>5}   oracle couldn't encode args (e.g. torch.memory_format)",
        by_status.get("oracle_error").unwrap_or(&0)
    );
    println!("\n=== ops with divergences ===");
    for r in results.iter().filter(|r| r.status == "diverges") {
        println!(
            "  {:30} {:>4}/{:<4} passed   first: {}",
            r.op,
            r.passed,
            r.attempted,
            r.first_failure.as_deref().unwrap_or("?")
        );
    }
    println!("\nNOTE: passes_quick at low --max-samples is a TRIAGE signal, not");
    println!("      deep verification. Re-run `sweep --op <name> --seeds N` to confirm.");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
    }
    match args[1].as_str() {
        "dispatch" => {
            for op in dispatch_ops() {
                println!("{op}");
            }
            Ok(())
        }
        "list-ops" => {
            let mut oracle = Oracle::spawn()?;
            let (ver, n) = oracle.ready()?;
            eprintln!("torch {ver} ({n} ops in op_db)");
            for op in oracle.list_ops()? {
                println!("{op}");
            }
            oracle.shutdown();
            Ok(())
        }
        "sweep-all" => {
            let mut seeds: u64 = 1;
            let mut limit: Option<usize> = None;
            let mut max_samples: usize = 4;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--seeds" => {
                        seeds = args.get(i + 1).ok_or("--seeds needs a value")?.parse()?;
                        i += 2;
                    }
                    "--limit" => {
                        limit = Some(args.get(i + 1).ok_or("--limit needs a value")?.parse()?);
                        i += 2;
                    }
                    "--max-samples" => {
                        max_samples = args
                            .get(i + 1)
                            .ok_or("--max-samples needs a value")?
                            .parse()?;
                        i += 2;
                    }
                    other => return Err(format!("unknown arg: {other}").into()),
                }
            }
            let json_out = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .ok_or("no parent")?
                .join("runs")
                .join("_all_coverage.json");
            if let Some(parent) = json_out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut oracle = Oracle::spawn()?;
            let (ver, n) = oracle.ready()?;
            eprintln!(
                "torch {ver} ({n} ops) — sweep-all seeds={seeds} max_samples_per_seed={max_samples} \
                 (checkpoint every 25 ops → {})",
                json_out.display()
            );
            let results = sweep_all(&mut oracle, seeds, limit, max_samples, &json_out)?;
            oracle.shutdown();
            std::fs::write(&json_out, serde_json::to_string_pretty(&results)?)?;
            eprintln!("wrote {}", json_out.display());
            print_coverage_summary(&results);
            let any_diverges = results.iter().any(|r| r.status == "diverges");
            if any_diverges {
                std::process::exit(1);
            }
            Ok(())
        }
        "probe" => {
            let mut op: Option<String> = None;
            let mut probes_path: Option<String> = None;
            let mut out_path: Option<String> = None;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--op" => {
                        op = Some(args.get(i + 1).cloned().ok_or("--op needs a value")?);
                        i += 2;
                    }
                    "--probes" => {
                        probes_path =
                            Some(args.get(i + 1).cloned().ok_or("--probes needs a value")?);
                        i += 2;
                    }
                    "--out" => {
                        out_path = Some(args.get(i + 1).cloned().ok_or("--out needs a value")?);
                        i += 2;
                    }
                    other => return Err(format!("unknown arg: {other}").into()),
                }
            }
            let op = op.ok_or("probe requires --op")?;
            let probes_path = probes_path.ok_or("probe requires --probes <jsonl>")?;
            let out_path = out_path.ok_or("probe requires --out <findings.json>")?;

            let probes_text = std::fs::read_to_string(&probes_path)?;
            let probes: Vec<Value> = probes_text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(serde_json::from_str::<Value>)
                .collect::<Result<_, _>>()?;
            let total = probes.len();

            let mut oracle = Oracle::spawn()?;
            let (ver, n) = oracle.ready()?;
            eprintln!("torch {ver} ({n} ops) — probe op={op} ({total} probes from {probes_path})");

            let mut findings: Vec<Value> = Vec::new();
            let mut by_cat_total: std::collections::BTreeMap<String, usize> = Default::default();
            let mut by_cat_div: std::collections::BTreeMap<String, usize> = Default::default();
            let mut by_cat_skip: std::collections::BTreeMap<String, usize> = Default::default();
            let mut by_cat_deferred: std::collections::BTreeMap<String, usize> = Default::default();

            // Categories whose probes exercise an API surface ferrotorch has
            // explicitly deferred to a tracking issue (i.e. not a parity bug
            // — a scheduled feature gap). Findings in these categories are
            // recorded as "deferred" instead of counted toward the divergence
            // total. Update this map when a deferred feature lands.
            //
            // (Empty: `out_kwarg` was implemented inline via
            // `grad_fns::arithmetic::add_scaled_out`; #1190 closed. If a
            // future category genuinely needs to be deferred to a tracking
            // issue, add it here as `("category_name", "#issue_number")`.)
            let deferred_categories: std::collections::BTreeMap<&str, &str> =
                std::collections::BTreeMap::new();

            for (idx, probe_spec) in probes.iter().enumerate() {
                let category = probe_spec
                    .get("category")
                    .and_then(Value::as_str)
                    .unwrap_or("?")
                    .to_string();
                let rationale = probe_spec
                    .get("rationale")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let probe_id = probe_spec
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("?")
                    .to_string();
                *by_cat_total.entry(category.clone()).or_insert(0) += 1;

                // 1. Ask the oracle what torch produces.
                let torch_resp = oracle.probe(&op, probe_spec)?;
                let torch_ok = torch_resp
                    .get("ok")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);

                // 2. Run ferrotorch.
                let ferr_result = run_probe_ferrotorch(probe_spec);

                let mut divergence: Option<String> = None;
                let mut ferr_repr: Value = Value::Null;
                // torch_repr is assigned in every arm of the match below;
                // the initial Null is just the typed slot.
                #[allow(unused_assignments)]
                let mut torch_repr: Value = Value::Null;

                match (&torch_resp, &ferr_result) {
                    // torch errored, ferrotorch errored: agreement on rejection.
                    (resp, Err(ferr_msg)) if !torch_ok => {
                        let terr = resp.get("err").and_then(Value::as_str).unwrap_or("?");
                        torch_repr = json!({"ERROR": terr});
                        ferr_repr = json!({"ERROR": ferr_msg});
                        // Both rejected — not a divergence. We still log as a
                        // finding labelled "both_reject" so the human can audit.
                        divergence = Some(format!(
                            "both rejected (torch: {terr}; ferrotorch: {ferr_msg})"
                        ));
                    }
                    // torch errored but ferrotorch succeeded -> divergence.
                    (resp, Ok(Some(t))) if !torch_ok => {
                        let terr = resp.get("err").and_then(Value::as_str).unwrap_or("?");
                        torch_repr = json!({"ERROR": terr});
                        let data = t.data_vec().ok();
                        ferr_repr = json!({
                            "shape": t.shape(),
                            "data": data,
                        });
                        divergence = Some(format!(
                            "torch rejected ({terr}) but ferrotorch returned a tensor"
                        ));
                    }
                    // torch errored, ferrotorch dispatched-skip — log as a skip.
                    (resp, Ok(None)) if !torch_ok => {
                        let terr = resp.get("err").and_then(Value::as_str).unwrap_or("?");
                        torch_repr = json!({"ERROR": terr});
                        ferr_repr = json!({"SKIP": "ferrotorch dispatch declines this dtype"});
                        *by_cat_skip.entry(category.clone()).or_insert(0) += 1;
                    }
                    // torch ok, ferrotorch errored -> divergence, UNLESS the
                    // category is on the deferred-features list (then it's a
                    // tracked scheduled gap, not a parity bug).
                    (resp, Err(ferr_msg)) => {
                        torch_repr = resp.get("output").cloned().unwrap_or(Value::Null);
                        ferr_repr = json!({"ERROR": ferr_msg});
                        if let Some(tracking_ref) = deferred_categories.get(category.as_str()) {
                            *by_cat_deferred.entry(category.clone()).or_insert(0) += 1;
                            ferr_repr = json!({
                                "DEFERRED": format!(
                                    "tracked in {tracking_ref}: {ferr_msg}"
                                ),
                            });
                            // No `divergence = Some(...)` here — the finding
                            // is recorded with DEFERRED ferr_repr but not
                            // counted as a divergence in the summary.
                        } else {
                            divergence = Some(format!("ferrotorch raised: {ferr_msg}"));
                        }
                    }
                    // torch ok, ferrotorch declined this dtype.
                    (resp, Ok(None)) => {
                        torch_repr = resp.get("output").cloned().unwrap_or(Value::Null);
                        ferr_repr = json!({"SKIP": "ferrotorch dispatch declines this dtype"});
                        *by_cat_skip.entry(category.clone()).or_insert(0) += 1;
                    }
                    // both ok -> diff the outputs.
                    (resp, Ok(Some(ferr_tensor))) => {
                        let torch_output_v = resp.get("output").cloned().unwrap_or(Value::Null);
                        let torch_wire: Option<WireTensor> = unwrap_tensor_arg(&torch_output_v);
                        let ferr_data = match ferr_tensor.data_vec() {
                            Ok(d) => d,
                            Err(e) => {
                                divergence = Some(format!("ferrotorch.data_vec: {e}"));
                                ferr_repr = json!({"ERROR": format!("{e}")});
                                vec![]
                            }
                        };
                        if let Some(wire) = torch_wire {
                            torch_repr = json!({
                                "shape": wire.shape,
                                "dtype": wire.dtype,
                            });
                            ferr_repr = json!({
                                "shape": ferr_tensor.shape(),
                                "dtype": "float32",
                            });
                            if wire.dtype == "float32" {
                                match wire.to_f32() {
                                    Ok(t_tensor) => {
                                        let t_data = t_tensor.data_vec().unwrap_or_default();
                                        if let Some(msg) = compare_probe_outputs(
                                            &ferr_data,
                                            &t_data,
                                            ferr_tensor.shape(),
                                            &wire.shape,
                                        ) {
                                            divergence = Some(msg);
                                            torch_repr = json!({
                                                "shape": wire.shape,
                                                "dtype": wire.dtype,
                                                "data": t_data,
                                            });
                                            ferr_repr = json!({
                                                "shape": ferr_tensor.shape(),
                                                "dtype": "float32",
                                                "data": ferr_data,
                                            });
                                        }
                                    }
                                    Err(e) => {
                                        divergence = Some(format!("decode torch f32 output: {e}"));
                                    }
                                }
                            } else {
                                divergence = Some(format!(
                                    "torch output dtype {} but ferrotorch dispatch is f32 only",
                                    wire.dtype
                                ));
                            }
                        } else {
                            divergence = Some("torch output is not a tensor envelope".to_string());
                            torch_repr = torch_output_v;
                        }

                        // 3. Autograd-check: compare grads if requested AND
                        // forward matched.
                        let autograd_check = probe_spec
                            .get("autograd_check")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);
                        if autograd_check && divergence.is_none() {
                            let torch_grads_v = resp.get("grads").cloned().unwrap_or(Value::Null);
                            match run_probe_ferrotorch_grads(probe_spec) {
                                Ok(ferr_grads) => {
                                    if let Some(arr) = torch_grads_v.as_array() {
                                        for (i_grad, t_g) in arr.iter().enumerate() {
                                            let torch_wire = unwrap_tensor_arg(t_g);
                                            let ferr_g = ferr_grads.get(i_grad);
                                            match (torch_wire, ferr_g) {
                                                (None, Some(None)) => {} // both None, ok
                                                (Some(tw), Some(Some(fg))) => {
                                                    let t_data = tw
                                                        .to_f32()
                                                        .and_then(|t| Ok(t.data_vec()?))
                                                        .unwrap_or_default();
                                                    let f_data = fg.data_vec().unwrap_or_default();
                                                    if let Some(msg) = compare_probe_outputs(
                                                        &f_data,
                                                        &t_data,
                                                        fg.shape(),
                                                        &tw.shape,
                                                    ) {
                                                        divergence =
                                                            Some(format!("grad[{i_grad}]: {msg}"));
                                                        break;
                                                    }
                                                }
                                                (Some(_), Some(None)) => {
                                                    divergence = Some(format!(
                                                        "grad[{i_grad}]: torch produced grad, ferrotorch has None"
                                                    ));
                                                    break;
                                                }
                                                (None, Some(Some(_))) => {
                                                    divergence = Some(format!(
                                                        "grad[{i_grad}]: torch grad is None, ferrotorch produced one"
                                                    ));
                                                    break;
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    divergence = Some(format!("ferrotorch backward raised: {e}"));
                                }
                            }
                        }
                    }
                }

                if let Some(msg) = &divergence {
                    *by_cat_div.entry(category.clone()).or_insert(0) += 1;
                    findings.push(json!({
                        "id": probe_id,
                        "category": category,
                        "rationale": rationale,
                        "args_spec": probe_spec.get("args_spec").cloned().unwrap_or(Value::Null),
                        "kwargs": probe_spec.get("kwargs").cloned().unwrap_or(json!({})),
                        "torch_output": torch_repr,
                        "ferrotorch_output": ferr_repr,
                        "divergence": msg,
                    }));
                }
                if idx % 10 == 0 || idx + 1 == total {
                    eprint!(
                        "\r[{:>3}/{total}] probed; {} divergences so far",
                        idx + 1,
                        findings.len()
                    );
                    use std::io::Write as _;
                    let _ = std::io::stderr().flush();
                }
            }
            eprintln!();
            oracle.shutdown();

            // Real-divergence count excludes "both_reject" (informational) and
            // skips. A both-reject entry was recorded so the audit shows the
            // input/output but it isn't an actual disagreement.
            let real_divergences = findings
                .iter()
                .filter(|f| {
                    f.get("divergence")
                        .and_then(Value::as_str)
                        .map(|s| !s.starts_with("both rejected"))
                        .unwrap_or(false)
                })
                .count();

            let total_deferred: usize = by_cat_deferred.values().sum();
            let both_reject = findings
                .iter()
                .filter(|f| {
                    f.get("divergence")
                        .and_then(Value::as_str)
                        .map(|s| s.starts_with("both rejected"))
                        .unwrap_or(false)
                })
                .count();
            std::fs::write(&out_path, serde_json::to_string_pretty(&findings)?)?;
            eprintln!("\nWrote {} findings to {out_path}", findings.len());
            println!("\n=== probe summary (op={op}) ===");
            println!("  total probes:      {total}");
            println!("  real divergences:  {real_divergences}");
            println!("  deferred:          {total_deferred}");
            println!("  both-reject logs:  {both_reject}");
            println!("\n  category                  total  diverged  deferred  skipped");
            for cat in by_cat_total.keys() {
                println!(
                    "  {:25} {:>5}  {:>8}  {:>8}  {:>7}",
                    cat,
                    by_cat_total.get(cat).copied().unwrap_or(0),
                    by_cat_div.get(cat).copied().unwrap_or(0),
                    by_cat_deferred.get(cat).copied().unwrap_or(0),
                    by_cat_skip.get(cat).copied().unwrap_or(0),
                );
            }
            Ok(())
        }
        "sweep" => {
            let mut op: Option<String> = None;
            let mut seeds: u64 = 8;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--op" => {
                        op = Some(args.get(i + 1).cloned().ok_or("--op needs a value")?);
                        i += 2;
                    }
                    "--seeds" => {
                        seeds = args.get(i + 1).ok_or("--seeds needs a value")?.parse()?;
                        i += 2;
                    }
                    other => return Err(format!("unknown arg: {other}").into()),
                }
            }
            let op = op.ok_or("sweep requires --op <name>")?;
            let mut oracle = Oracle::spawn()?;
            let (ver, n) = oracle.ready()?;
            eprintln!("torch {ver} ({n} ops) — sweeping {op} with seeds 0..{seeds}");
            let report = sweep(&mut oracle, &op, seeds)?;
            oracle.shutdown();
            report.print();
            if !report.failures.is_empty() {
                std::process::exit(1);
            }
            Ok(())
        }
        _ => usage(),
    }
}
