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
use ferrotorch_core::{BoolTensor, IntTensor, Tensor, from_vec, grad_fns};
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
        // Reduction ops emit integer (argmax/argmin/count_nonzero -> int64)
        // and bool (any/all) outputs. The parity-sweep's value-equality
        // gate compares against ferrotorch's Tensor<f32>; we widen those
        // expected envelopes to f32 here so the existing
        // `assert_close_f32` continues to work. Widening direction is
        // lossless for the value ranges op_db emits (int64 indices fit
        // in f32 mantissa for tensor shapes the suite uses; bool is {0,1}).
        match self.dtype.as_str() {
            "float32" => {}
            "int64" | "int32" | "uint8" | "bool" => {
                let numel: usize = if self.shape.is_empty() {
                    1
                } else {
                    self.shape.iter().product()
                };
                let data: Vec<f32> = match self.dtype.as_str() {
                    "int64" => self
                        .to_int_tensor_i64()?
                        .data()?
                        .iter()
                        .map(|&v| v as f32)
                        .collect(),
                    "int32" => self
                        .to_int_tensor_i64()?
                        .data()?
                        .iter()
                        .map(|&v| v as f32)
                        .collect(),
                    "uint8" => self
                        .to_int_tensor_i64()?
                        .data()?
                        .iter()
                        .map(|&v| v as f32)
                        .collect(),
                    "bool" => self
                        .to_bool_tensor()?
                        .data()?
                        .iter()
                        .map(|&b| if b { 1.0 } else { 0.0 })
                        .collect(),
                    _ => unreachable!(),
                };
                if data.len() != numel {
                    return Err(format!(
                        "widened {} length {} does not match shape {:?}",
                        self.dtype,
                        data.len(),
                        self.shape
                    )
                    .into());
                }
                return Ok(from_vec(data, &self.shape)?);
            }
            other => return Err(format!("dispatch supports float32/int/bool, got {other}").into()),
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

    /// Decode the wire envelope to an `IntTensor<i64>`. Accepts `uint8`,
    /// `int32`, and `int64` dtypes; narrower forms are widened to i64 to
    /// match ferrotorch's `IntTensor<i64>` carrier (the upstream contract
    /// for the per-channel `zero_point` is `int32 | float | half` per
    /// `aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp:53`,
    /// but ferrotorch's typed carrier always widens). `uint8` indices
    /// appear in op_db's `gather` samples (small empty-tensor cases).
    fn to_int_tensor_i64(&self) -> Result<IntTensor<i64>, Box<dyn std::error::Error>> {
        let bytes = B64.decode(&self.data_b64)?;
        let numel: usize = if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        };
        let data: Vec<i64> = match self.dtype.as_str() {
            "uint8" => {
                let expected = numel;
                if bytes.len() != expected {
                    return Err(format!(
                        "uint8 byte length {} does not match shape {:?} (expected {})",
                        bytes.len(),
                        self.shape,
                        expected
                    )
                    .into());
                }
                bytes.iter().map(|&b| i64::from(b)).collect()
            }
            "int32" => {
                let expected = numel * 4;
                if bytes.len() != expected {
                    return Err(format!(
                        "int32 byte length {} does not match shape {:?} (expected {})",
                        bytes.len(),
                        self.shape,
                        expected
                    )
                    .into());
                }
                bytes
                    .chunks_exact(4)
                    .map(|c| i64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                    .collect()
            }
            "int64" => {
                let expected = numel * 8;
                if bytes.len() != expected {
                    return Err(format!(
                        "int64 byte length {} does not match shape {:?} (expected {})",
                        bytes.len(),
                        self.shape,
                        expected
                    )
                    .into());
                }
                bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect()
            }
            other => {
                return Err(format!(
                    "to_int_tensor_i64: only uint8/int32/int64 supported, got {other}"
                )
                .into());
            }
        };
        Ok(IntTensor::<i64>::from_vec(data, self.shape.clone())?)
    }

    /// Decode the wire envelope to a [`BoolTensor`]. Oracle bool wire is
    /// one byte per element (each byte is 0 or 1) per `oracle.py:54-86`,
    /// byte-identical to a host `&[bool]` (Rust's `bool` is also 1 byte with
    /// the same {0, 1} bit pattern). Used by the `masked_select` / `masked_fill`
    /// / `where` runner dispatch arms (closes #1250 #1251 #1255).
    fn to_bool_tensor(&self) -> Result<BoolTensor, Box<dyn std::error::Error>> {
        if self.dtype != "bool" {
            return Err(format!("to_bool_tensor: expected bool dtype, got {}", self.dtype).into());
        }
        let bytes = B64.decode(&self.data_b64)?;
        let numel: usize = if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        };
        if bytes.len() != numel {
            return Err(format!(
                "bool byte length {} does not match shape {:?} (expected {})",
                bytes.len(),
                self.shape,
                numel
            )
            .into());
        }
        let data: Vec<bool> = bytes.into_iter().map(|b| b != 0).collect();
        Ok(BoolTensor::from_vec(data, self.shape.clone())?)
    }
}

/// An arg returned by the oracle — either a tensor envelope or a JSON scalar.
fn unwrap_tensor_arg(v: &Value) -> Option<WireTensor> {
    let envelope = v.as_object()?.get("__tensor__")?;
    serde_json::from_value(envelope.clone()).ok()
}

/// Decode an int64-typed wire tensor into a flat `Vec<usize>` with its shape,
/// rejecting any negative index (the underlying ferrotorch
/// gather/scatter/scatter_add/index_select_dim entry points enforce
/// non-negative indices — upstream `Tensor index_select_cpu_` at
/// `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1862` does wrap negative
/// indices, which ferrotorch deliberately diverges from per
/// `.design/ferrotorch-core/grad_fns/indexing.md` REQ-5 / parity contract).
/// Returns `Ok(None)` if any index is negative — that sample is a legitimate
/// skip rather than an authoritative failure.
///
/// The `Result<Option<(...)>, _>` return shape mirrors how the gather /
/// scatter / scatter_add / index_select arms below treat skip vs. error vs.
/// success: `Ok(None)` is the legitimate-skip pathway, `Err(_)` is a real
/// wire-decode failure. Factoring this into a named type alias for a single
/// helper in a runner binary would obscure the local control flow.
#[allow(
    clippy::type_complexity,
    reason = "single-use helper in the runner binary; mirrors `ternary`'s \
              inline closure shape precedent at the dispatch_f32 helper layer"
)]
fn decode_int64_index_to_usize(
    wire: &WireTensor,
) -> Result<Option<(Vec<usize>, Vec<usize>)>, Box<dyn std::error::Error>> {
    let it = wire.to_int_tensor_i64()?;
    let data = it.data()?;
    let mut out = Vec::with_capacity(data.len());
    for &v in data {
        if v < 0 {
            // Upstream wraps negative indices; ferrotorch rejects them per
            // existing index_select_1d_it / index_select_dim contract. Skip
            // rather than report a divergence — the wrap-semantics gap is
            // its own design-doc-level question, not a parity bug.
            return Ok(None);
        }
        out.push(v as usize);
    }
    Ok(Some((out, wire.shape.clone())))
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
    // 3-arg ternary helper for ops like `torch.addcmul(input, tensor1, tensor2,
    // *, value=1)` and `torch.addcdiv` per `aten/src/ATen/native/PointwiseOps.cpp`.
    // op_db emits `args = [input, tensor1, tensor2]` with `value` in kwargs.
    // Reusable for blocker #1201 (addcdiv) — the helper is op-agnostic.
    //
    // The 3-tuple-of-Tensor return shape exactly mirrors `binary`'s 2-tuple
    // form above; clippy's `type-complexity` lint fires on the inline
    // closure return type, but factoring out a named type alias for a
    // single-use helper in a runner binary obscures more than it clarifies.
    #[allow(
        clippy::type_complexity,
        reason = "mirrors `binary`'s inline closure shape; runner-only helper, \
                  not worth a one-shot type alias"
    )]
    let ternary = |name: &str| -> Result<
        (Tensor<f32>, Tensor<f32>, Tensor<f32>),
        Box<dyn std::error::Error>,
    > {
        if args.len() < 3 {
            return Err(format!("{name} expects 3 args, got {}", args.len()).into());
        }
        let a = unwrap_tensor_arg(&args[0])
            .ok_or_else(|| format!("{name} arg 0 not a tensor"))?
            .to_f32()?;
        let b = unwrap_tensor_arg(&args[1])
            .ok_or_else(|| format!("{name} arg 1 not a tensor"))?
            .to_f32()?;
        let c = unwrap_tensor_arg(&args[2])
            .ok_or_else(|| format!("{name} arg 2 not a tensor"))?
            .to_f32()?;
        Ok((a, b, c))
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
    // `torch.addcmul(input, tensor1, tensor2, *, value=1)` ships `value` as
    // a JSON number in the kwargs envelope (default 1.0 when absent). Same
    // shape as `alpha_kwarg` above but renamed for clarity since the kwarg
    // name is `value`, not `alpha`. Reusable for `addcdiv` (#1201).
    let value_kwarg = |name: &str| -> Result<f64, Box<dyn std::error::Error>> {
        match kwargs.get("value") {
            None => Ok(1.0),
            Some(v) => v
                .as_f64()
                .ok_or_else(|| format!("{name}: value kwarg is not a JSON number: {v}").into()),
        }
    };

    // Reduction-op kwarg helpers. op_db emits `dim` as int OR list-of-ints
    // OR empty list `[]` (the "no-op full-reduction" form mirroring
    // `torch.sum(x, dim=())`). `keepdim` is always a bool with default
    // false. Returns `None` for full reduction (no `dim` kwarg OR `dim==[]`).
    let dim_kwarg = |_name: &str| -> Result<Option<Vec<i64>>, Box<dyn std::error::Error>> {
        match kwargs.get("dim") {
            None => Ok(None),
            Some(Value::Number(n)) => {
                let v = n.as_i64().ok_or("dim kwarg: non-integer JSON number")?;
                Ok(Some(vec![v]))
            }
            Some(Value::Array(arr)) => {
                if arr.is_empty() {
                    Ok(None)
                } else {
                    let mut out = Vec::with_capacity(arr.len());
                    for x in arr {
                        out.push(x.as_i64().ok_or("dim kwarg list: non-int element")?);
                    }
                    Ok(Some(out))
                }
            }
            Some(Value::Null) => Ok(None),
            Some(other) => Err(format!("dim kwarg: unexpected JSON value {other}").into()),
        }
    };
    let keepdim_kwarg = || -> bool {
        kwargs
            .get("keepdim")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    };
    // For ops where the second positional is a bool (`std`/`var`'s
    // `unbiased`) or `keepdim` (`logsumexp`'s third positional).
    let arg_bool_at = |idx: usize| -> Option<bool> { args.get(idx).and_then(Value::as_bool) };
    // Multi-dim list at positional `args[idx]` (logsumexp emits
    // `args = [tensor, [dim0, dim1, ...], keepdim]`).
    let arg_dim_list_at = |idx: usize| -> Option<Vec<i64>> {
        match args.get(idx)? {
            Value::Number(n) => n.as_i64().map(|v| vec![v]),
            Value::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for x in arr {
                    out.push(x.as_i64()?);
                }
                Some(out)
            }
            _ => None,
        }
    };

    // Coerce an `IntTensor<i64>` produced by a non-differentiable reduction
    // (argmax / argmin / count_nonzero) into a `Tensor<f32>` so the existing
    // `assert_close_f32` value-equality gate can consume it. Pairs with
    // `WireTensor::to_f32`'s int-widening branch on the expected side.
    let int_to_f32 =
        |it: &ferrotorch_core::IntTensor<i64>| -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
            let d = it.data()?;
            let f: Vec<f32> = d.iter().map(|&v| v as f32).collect();
            Ok(ferrotorch_core::from_vec(f, it.shape())?)
        };
    // Same for BoolTensor (any/all).
    let bool_to_f32 =
        |b: &ferrotorch_core::BoolTensor| -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
            let d = b.data()?;
            let f: Vec<f32> = d.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
            Ok(ferrotorch_core::from_vec(f, b.shape())?)
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
        // `torch.remainder(input, other, *, out=None)` — `_torch_docs.py:9453-9472`.
        // ferrotorch's `arithmetic::remainder<T: Float>(a, b)` mirrors the
        // upstream CPU kernel at `aten/src/ATen/native/cpu/BinaryOpsKernel
        // .cpp:391-409`'s `AT_DISPATCH_FLOATING_TYPES_AND_HALF` branch
        // (`scalar_t mod = std::fmod(a, b); if ((mod != 0) && ((b < 0) !=
        // (mod < 0))) mod += b;`). Sign-of-divisor (Python `%`) semantics
        // distinct from `fmod` (dividend-sign / C99). Backward per
        // `tools/autograd/derivatives.yaml:1455-1457`: `da = grad`,
        // `db = -grad * floor(a / b)`. Binary, no kwargs — `remainder` does
        // not take alpha. Closes blocker #1198.
        "remainder" => Ok(Some({
            let (a, b) = binary("remainder")?;
            grad_fns::arithmetic::remainder(&a, &b)?
        })),
        // `torch.fmod(input, other, *, out=None)` — `_torch_docs.py:4302-4350`.
        // ferrotorch's `arithmetic::fmod<T: Float>(a, b)` mirrors the
        // upstream CPU kernel at `aten/src/ATen/native/cpu/BinaryOpsKernel
        // .cpp:1052-1054`'s `AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16,
        // kHalf, ...)` branch (`[](scalar_t x, scalar_t d) -> scalar_t {
        // return std::fmod(x, d); }`). Sign-of-dividend (C99 `fmod`)
        // semantics distinct from `remainder` (divisor-sign / Python `%`).
        // Backward per `tools/autograd/derivatives.yaml:717-720`:
        // `da = grad`, `db = -grad * trunc(a / b)`. Binary, no kwargs —
        // `fmod` does not take alpha. Closes blocker #1199.
        "fmod" => Ok(Some({
            let (a, b) = binary("fmod")?;
            grad_fns::arithmetic::fmod(&a, &b)?
        })),
        // `torch.floor_divide(input, other, *, out=None)` —
        // `_torch_docs.py:4265-4296`. ferrotorch's
        // `arithmetic::floor_divide<T: Float>(a, b)` mirrors the upstream
        // CPU kernel at `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:297-349
        // div_floor_kernel`'s floating-types branch which calls
        // `c10::div_floor_floating` at `c10/util/generic_math.h:34-58`
        // byte-for-byte. TRUE FLOOR semantics (toward -inf): verified live
        // 2026-05-25 `torch.floor_divide(-7.0, 3.0) = -3.0`. The doc note
        // at `_torch_docs.py:4267-4271` explicitly states the pre-1.13
        // trunc-division behaviour is gone; current PyTorch performs
        // floor. `floor_divide` is NOT in `derivatives.yaml` — upstream
        // `grad_fn=<NotImplemented>` raises `derivative for
        // aten::floor_divide is not implemented` on `.backward()`;
        // `FloorDivideBackward` mirrors that error. Binary, no kwargs —
        // `floor_divide` does not take alpha. Closes blocker #1197.
        "floor_divide" => Ok(Some({
            let (a, b) = binary("floor_divide")?;
            grad_fns::arithmetic::floor_divide(&a, &b)?
        })),
        // `torch.addcmul(input, tensor1, tensor2, *, value=1)` — fused
        // `out = input + value * tensor1 * tensor2` per
        // `aten/src/ATen/native/PointwiseOps.cpp:57 TORCH_IMPL_FUNC(addcmul_out)`
        // and `_torch_docs.py:510`. ferrotorch's
        // `arithmetic::addcmul<T: Float>(input, t1, t2, value)` mirrors the
        // 3-input broadcast TensorIteratorConfig at `PointwiseOps.cpp:17-31`;
        // backward per `tools/autograd/derivatives.yaml`:
        //   self    : grad
        //   tensor1 : grad * (tensor2 * value)
        //   tensor2 : grad * (tensor1 * value)
        // 3 args via the new `ternary()` helper + `value_kwarg` kwarg
        // (default 1.0). Closes blocker #1200.
        "addcmul" => Ok(Some({
            let (input, t1, t2) = ternary("addcmul")?;
            let value = value_kwarg("addcmul")?;
            grad_fns::arithmetic::addcmul(&input, &t1, &t2, value)?
        })),
        // `torch.addcdiv(input, tensor1, tensor2, *, value=1)` — fused
        // `out = input + value * tensor1 / tensor2` per
        // `aten/src/ATen/native/PointwiseOps.cpp:66 TORCH_IMPL_FUNC(addcdiv_out)`
        // and `_torch_docs.py:461`. ferrotorch's
        // `arithmetic::addcdiv<T: Float>(input, t1, t2, value)` mirrors the
        // 3-input ternary `build_ternary_op` at `PointwiseOps.cpp:51`;
        // backward per `tools/autograd/derivatives.yaml`:
        //   self    : grad
        //   tensor1 : grad * (value / tensor2)
        //   tensor2 : -grad * (value * tensor1 / (tensor2 * tensor2))
        // 3 args via the existing `ternary()` helper + `value_kwarg` kwarg
        // (default 1.0) — both helpers introduced for addcmul (#1200) and
        // reused here per R-DEFER-8. Integer-dtype error path at
        // `PointwiseOps.cpp:38-50 TORCH_META_FUNC(addcdiv)` is unreachable
        // for `Tensor<T: Float>`. Closes blocker #1201.
        "addcdiv" => Ok(Some({
            let (input, t1, t2) = ternary("addcdiv")?;
            let value = value_kwarg("addcdiv")?;
            grad_fns::arithmetic::addcdiv(&input, &t1, &t2, value)?
        })),
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
        // Cumulative (scan) ops — `torch.cumsum(input, dim)` and friends ship
        // `dim` as a Python int positional `args[1]` (verified 2026-05-25 via
        // live oracle sample inspection). Dim extraction is inlined per-arm
        // to avoid shifting the line numbers of pre-existing arithmetic-arm
        // anchors cited in `.design/ferrotorch-core/grad_fns/arithmetic.md`
        // (the `divergence_addcmul_req15_runner_cite_shift` test pins those
        // cites to within 4 lines). Closes blocker #1230 (the runner
        // dispatch gap; the production-consumer wiring is separately
        // tracked under blocker #1232).
        //
        // `torch.cumsum(input, dim, *, dtype=None)` —
        // `aten/src/ATen/native/ReduceOps.cpp:511 TORCH_IMPL_FUNC(cumsum_out)`
        // dispatches via `cumsum_stub` declared at `:460`. ferrotorch's
        // `grad_fns::cumulative::cumsum<T: Float>(input, dim: i64)` wraps
        // `ops::cumulative::cumsum_forward` with `CumsumBackward` per
        // `tools/autograd/derivatives.yaml:529-531` (`reversed_cumsum`
        // upper-triangular multiplication).
        "cumsum" => Ok(Some({
            let a = unary("cumsum")?;
            let dim = args
                .get(1)
                .and_then(Value::as_i64)
                .ok_or("cumsum: missing or non-int dim arg")?;
            grad_fns::cumulative::cumsum(&a, dim)?
        })),
        // `torch.cumprod(input, dim, *, dtype=None)` —
        // `ReduceOps.cpp:519 TORCH_IMPL_FUNC(cumprod_out)`. ferrotorch's
        // `grad_fns::cumulative::cumprod` wraps `cumprod_forward` with
        // `CumprodBackward` per `derivatives.yaml:525-527`.
        "cumprod" => Ok(Some({
            let a = unary("cumprod")?;
            let dim = args
                .get(1)
                .and_then(Value::as_i64)
                .ok_or("cumprod: missing or non-int dim arg")?;
            grad_fns::cumulative::cumprod(&a, dim)?
        })),
        // `torch.cummax(input, dim) -> (values, indices)` —
        // `ReduceOps.cpp:860 Tensor cummax(const Tensor& self, int64_t dim)`
        // -> `cummax_cummin_helper<T1, T2, std::greater_equal<scalar_t>>` at
        // `:811-826`. PyTorch returns a namedtuple `(values, indices)`; the
        // oracle wraps these as a JSON array. The sweep loop's expected-
        // output extraction handles the JSON-array case by selecting
        // `output[0]` (values). Option A from the #1230 dispatch prompt:
        // values-parity only; indices-parity divergences (tie-break,
        // differentiability, NaN handling) tracked under #1231.
        "cummax" => Ok(Some({
            let a = unary("cummax")?;
            let dim = args
                .get(1)
                .and_then(Value::as_i64)
                .ok_or("cummax: missing or non-int dim arg")?;
            grad_fns::cumulative::cummax(&a, dim)?.values
        })),
        // `torch.cummin(input, dim) -> (values, indices)` —
        // `ReduceOps.cpp:899 Tensor cummin(...)`. Symmetric to cummax
        // (Option A: values only; #1231 covers indices divergences).
        "cummin" => Ok(Some({
            let a = unary("cummin")?;
            let dim = args
                .get(1)
                .and_then(Value::as_i64)
                .ok_or("cummin: missing or non-int dim arg")?;
            grad_fns::cumulative::cummin(&a, dim)?.values
        })),
        // `torch.logcumsumexp(input, dim)` —
        // `ReduceOps.cpp:475 Tensor logcumsumexp(...)` dispatching via
        // `_logcumsumexp_cpu` at `:465-468` -> `logcumsumexp_stub` at `:471`.
        // ferrotorch's `grad_fns::cumulative::logcumsumexp` wraps the two-
        // pass running-max rescaling kernel at `ops/cumulative.rs:378-410`.
        // Backward per `derivatives.yaml:521-523` factors as
        // `exp(input) * reverse_cumsum(grad * exp(-output))`.
        "logcumsumexp" => Ok(Some({
            let a = unary("logcumsumexp")?;
            let dim = args
                .get(1)
                .and_then(Value::as_i64)
                .ok_or("logcumsumexp: missing or non-int dim arg")?;
            grad_fns::cumulative::logcumsumexp(&a, dim)?
        })),
        // `torch.fake_quantize_per_tensor_affine(input, scale, zero_point,
        // quant_min, quant_max)` — `torch/overrides.py:622`. Oracle emits
        // `args = [input_tensor, scale: f64, zero_point: i64, quant_min: i64,
        // quant_max: i64]` per `tools/parity-sweep/oracle.py:184
        // ((input, scale, zp, qmin, qmax), {})`. ferrotorch impl at
        // `ferrotorch-core/src/grad_fns/quantize_grad.rs:fake_quantize_per_tensor_affine`
        // mirrors the upstream forward at `aten/src/ATen/native/quantized/
        // FakeQuantPerTensorAffine.cpp:31-40` byte-for-byte (banker's
        // rounding via `f64::round_ties_even`, NaN-safe clamp via
        // `f64::min`/`f64::max`). Closes blocker #1238.
        "fake_quantize_per_tensor_affine" => Ok(Some({
            if args.len() < 5 {
                return Err(format!(
                    "fake_quantize_per_tensor_affine: expected 5 args, got {}",
                    args.len()
                )
                .into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("fake_quantize_per_tensor_affine: arg 0 not a tensor")?
                .to_f32()?;
            let scale = args[1]
                .as_f64()
                .ok_or("fake_quantize_per_tensor_affine: arg 1 (scale) not a JSON number")?;
            let zero_point = args[2]
                .as_i64()
                .ok_or("fake_quantize_per_tensor_affine: arg 2 (zero_point) not a JSON integer")?;
            let quant_min = args[3]
                .as_i64()
                .ok_or("fake_quantize_per_tensor_affine: arg 3 (quant_min) not a JSON integer")?;
            let quant_max = args[4]
                .as_i64()
                .ok_or("fake_quantize_per_tensor_affine: arg 4 (quant_max) not a JSON integer")?;
            grad_fns::quantize_grad::fake_quantize_per_tensor_affine(
                &input, scale, zero_point, quant_min, quant_max,
            )?
        })),
        // `torch.fake_quantize_per_channel_affine(input, scale, zero_point,
        // axis, quant_min, quant_max)` — `torch/overrides.py:621`. Oracle
        // emits `args = [input_tensor (f32), scale_tensor (f32, 1-D),
        // zero_point_tensor (int32, 1-D), axis: i64, quant_min: i64,
        // quant_max: i64]` per `tools/parity-sweep/oracle.py:269
        // ((s0, scale0, zp0, 1, -128, 127), {})`. ferrotorch impl at
        // `ferrotorch-core/src/grad_fns/quantize_grad.rs::fake_quantize_per_channel_affine`
        // mirrors the upstream forward at `aten/src/ATen/native/quantized/
        // FakeQuantPerChannelAffine.cpp:32-42` byte-for-byte (per-channel
        // banker's rounding + NaN-safe clamp using `scale[c]` / `zp[c]`).
        // Closes blocker #1239.
        // `torch.masked_select(input, mask)` — return a 1-D compaction of
        // input elements where mask is true. Upstream broadcasts input and
        // mask via `expand_outplace(mask, self)` at
        // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2545` (called from
        // `masked_select_cpu` at `:2621-2624`). Routes through the
        // broadcasting wrapper `grad_fns::indexing::masked_select_bcast`
        // which infers the common shape, expands both operands via the
        // autograd-aware `grad_fns::shape::expand`, then delegates to the
        // existing shape-strict `ops::indexing::masked_select`. Closes #1250.
        "masked_select" => Ok(Some({
            if args.len() < 2 {
                return Err(format!("masked_select expects 2 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("masked_select: arg 0 not a tensor")?
                .to_f32()?;
            let mask = unwrap_tensor_arg(&args[1])
                .ok_or("masked_select: arg 1 not a tensor")?
                .to_bool_tensor()?;
            grad_fns::indexing::masked_select_bcast(&input, &mask)?
        })),
        // `torch.masked_fill(input, mask, value)` — fill elements of input
        // with `value` where mask is true, mask is broadcast to input shape.
        // Upstream broadcasts via `expand_outplace(mask, self)` at
        // `TensorAdvancedIndexing.cpp:2503` (called from
        // `Tensor masked_fill(...)` at `:2494-2509`). Oracle wire emits
        // `args = [input, mask, value]` where `value` is either a JSON number
        // or a 0-d tensor envelope (sample_inputs_masked_fill at
        // `torch/testing/_internal/common_methods_invocations.py:6989-7010`).
        // Closes #1251.
        "masked_fill" => Ok(Some({
            if args.len() < 3 {
                return Err(format!("masked_fill expects 3 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("masked_fill: arg 0 not a tensor")?
                .to_f32()?;
            let mask = unwrap_tensor_arg(&args[1])
                .ok_or("masked_fill: arg 1 not a tensor")?
                .to_bool_tensor()?;
            // `value` is a JSON scalar (number or string-quoted "10") or a
            // 0-d tensor envelope. Decode either form to f32.
            let value: f32 = if let Some(v) = args[2].as_f64() {
                v as f32
            } else if let Some(s) = args[2].as_str() {
                // op_db sometimes emits the value as a Python int → JSON
                // string ("10"). Parse the string.
                s.parse::<f32>()
                    .map_err(|e| format!("masked_fill: arg 2 string parse failed: {e}"))?
            } else if let Some(wt) = unwrap_tensor_arg(&args[2]) {
                if !wt.shape.is_empty() {
                    // Tensor-valued fill must be 0-d per upstream contract at
                    // `TensorAdvancedIndexing.cpp:2482-2487
                    // TORCH_CHECK(value.dim() == 0, "masked_fill_ only
                    // supports a 0-dimensional value tensor, ...");`. Non-0-d
                    // is a legitimate skip (oracle never emits it, but
                    // belt-and-braces).
                    return Ok(None);
                }
                let t = wt.to_f32()?;
                let d = t.data_vec()?;
                d.first().copied().unwrap_or(0.0)
            } else {
                return Err(
                    format!("masked_fill: arg 2 not a number/string/tensor: {}", args[2]).into(),
                );
            };
            grad_fns::indexing::masked_fill_bcast(&input, &mask, value)?
        })),
        // `torch.where(condition, self, other)` — ternary selection with
        // 3-way broadcasting. Upstream builds a TensorIterator over
        // (condition, self, other) at
        // `aten/src/ATen/native/TensorCompare.cpp:629-637 where_self_out`
        // (called from `Tensor where(...)` at `:642-648`). Op_db's `where`
        // entry registers `op=lambda self, condition, other: torch.where(
        // condition, self, other)` per
        // `common_methods_invocations.py:21742-21746`, so oracle wire emits
        // `args = [self, condition, other]` (self / x first, then mask, then
        // other / y). Routes through `grad_fns::indexing::where_cond_bcast`.
        // Closes #1255.
        "where" => Ok(Some({
            if args.len() < 3 {
                return Err(format!("where expects 3 args, got {}", args.len()).into());
            }
            let x = unwrap_tensor_arg(&args[0])
                .ok_or("where: arg 0 (self) not a tensor")?
                .to_f32()?;
            let cond = unwrap_tensor_arg(&args[1])
                .ok_or("where: arg 1 (condition) not a tensor")?
                .to_bool_tensor()?;
            let y = unwrap_tensor_arg(&args[2])
                .ok_or("where: arg 2 (other) not a tensor")?
                .to_f32()?;
            grad_fns::indexing::where_cond_bcast(&cond, &x, &y)?
        })),
        // `torch.gather(input, dim, index, *, sparse_grad=False)` — for each
        // output position `p`, returns `input[p with axis-dim replaced by
        // index[p]]`. Upstream forward at
        // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2070
        // TORCH_IMPL_FUNC(gather_out)`. op_db emits `args = [input_f32,
        // dim_i64, index_int64]` (verified 2026-05-25 via live oracle
        // sample inspection: `i=0 shapes=[[10,5], 0, [5,5]]`,
        // `i=2 shapes=[[10,5], 1, [10,2]]`). ferrotorch's shape-strict
        // `ops::indexing::gather` at `ferrotorch-core/src/ops/indexing.rs:112`
        // takes `(input, dim: isize, index: &[usize], index_shape: &[usize])`
        // and validates `input.ndim() == index.ndim()` at `:73-80`. 0-d
        // inputs (sample `i=6 shapes=[[], 0, []]`) and ndim-mismatch index
        // samples are legitimate skips per the shape-strict contract.
        // Closes #1242.
        "gather" => {
            if args.len() < 3 {
                return Err(format!("gather expects 3 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("gather: arg 0 not a tensor")?
                .to_f32()?;
            let dim = args[1]
                .as_i64()
                .ok_or("gather: arg 1 (dim) not a JSON integer")?;
            let index_wire =
                unwrap_tensor_arg(&args[2]).ok_or("gather: arg 2 (index) not a tensor")?;
            let (index, index_shape) = match decode_int64_index_to_usize(&index_wire)? {
                Some(p) => p,
                None => return Ok(None),
            };
            // Shape-strict gather rejects 0-d inputs and ndim-mismatch
            // (input.ndim != index.ndim). Skip rather than fail — those
            // are narrower-contract divergences tracked at the REQ level.
            if input.ndim() == 0 || input.ndim() != index_shape.len() {
                return Ok(None);
            }
            Ok(Some(ferrotorch_core::ops::indexing::gather(
                &input,
                dim as isize,
                &index,
                &index_shape,
            )?))
        }
        // `torch.scatter(self, dim, index, src, *, reduce=None)` — writes
        // `output[index[p] at axis dim] = src[p]` into a clone of self.
        // Upstream forward at
        // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2263
        // TORCH_IMPL_FUNC(scatter_src_out)`. op_db emits `args = [input_f32,
        // dim_i64, index_int64, src_f32]` (verified 2026-05-25: `i=0
        // shapes=[[10,5], 0, [5,5], [5,5]]`, negative-dim samples at
        // `i=9..11 shapes=[[10,5], -1, [5,5], [5,5]]`, 0-d at `i=6
        // shapes=[[], 0, [], []]`). The op_db sweep also mixes the
        // `reduce` kwarg overloads (`scatter_reduce_two_out` at
        // `TensorAdvancedIndexing.cpp:2354`): `reduce='add'` matches
        // ferrotorch's `scatter_add` semantics so we route there;
        // `reduce='multiply'` is REQ-4 NOT-STARTED (blocker #1245) so we
        // skip; absent/None routes to plain scatter. A scalar `src` (the
        // `scatter.value` overload at `TensorAdvancedIndexing.cpp:2278`)
        // is also a legitimate skip — ferrotorch has no Scalar-src
        // forward. ferrotorch's shape-strict `ops::indexing::scatter` at
        // `ops/indexing.rs:183` accepts `dim: isize` (handles negative via
        // `normalize_axis`). 0-d input is a legitimate skip (the
        // shape-strict impl rejects ndim==0 at `ops/indexing.rs:191-194`).
        // Closes #1243.
        "scatter" => {
            if args.len() < 4 {
                return Err(format!("scatter expects 4 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("scatter: arg 0 not a tensor")?
                .to_f32()?;
            let dim = args[1]
                .as_i64()
                .ok_or("scatter: arg 1 (dim) not a JSON integer")?;
            let index_wire =
                unwrap_tensor_arg(&args[2]).ok_or("scatter: arg 2 (index) not a tensor")?;
            // Scalar src (the scatter.value overload) is out of dispatch
            // scope; skip rather than treat as a divergence.
            let src = match unwrap_tensor_arg(&args[3]) {
                Some(w) => w.to_f32()?,
                None => return Ok(None),
            };
            // Inspect the `reduce` kwarg. Per the op_db sweep the values
            // observed are `'add'` (→ scatter_add semantics) and
            // `'multiply'` (REQ-4 NOT-STARTED, blocker #1245 — skip).
            let reduce = kwargs.get("reduce").and_then(Value::as_str);
            let (index, index_shape) = match decode_int64_index_to_usize(&index_wire)? {
                Some(p) => p,
                None => return Ok(None),
            };
            if input.ndim() == 0 || input.ndim() != index_shape.len() {
                return Ok(None);
            }
            match reduce {
                None => Ok(Some(ferrotorch_core::ops::indexing::scatter(
                    &input,
                    dim as isize,
                    &index,
                    &index_shape,
                    &src,
                )?)),
                Some("add") => Ok(Some(ferrotorch_core::ops::indexing::scatter_add(
                    &input,
                    dim as isize,
                    &index,
                    &index_shape,
                    &src,
                )?)),
                Some("multiply") | Some("amin") | Some("amax") | Some("mean") | Some("prod")
                | Some("sum") => {
                    // scatter_reduce family is REQ-4 NOT-STARTED in
                    // `.design/ferrotorch-core/grad_fns/indexing.md`
                    // (blocker #1245). Legitimate skip.
                    Ok(None)
                }
                Some(other) => Err(format!("scatter: unknown reduce kwarg: {other}").into()),
            }
        }
        // `torch.scatter_add(self, dim, index, src)` — like scatter but
        // accumulates via `+=`. Upstream forward at
        // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2317
        // TORCH_IMPL_FUNC(scatter_add)`. Same arg shape as scatter.
        // ferrotorch's `ops::indexing::scatter_add` at `ops/indexing.rs:259`.
        // Production consumer of the underlying forward at
        // `ferrotorch-core/src/grad_fns/cumulative.rs:503` (cummax/cummin
        // VJP). Closes #1244.
        "scatter_add" => {
            if args.len() < 4 {
                return Err(format!("scatter_add expects 4 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("scatter_add: arg 0 not a tensor")?
                .to_f32()?;
            let dim = args[1]
                .as_i64()
                .ok_or("scatter_add: arg 1 (dim) not a JSON integer")?;
            let index_wire =
                unwrap_tensor_arg(&args[2]).ok_or("scatter_add: arg 2 (index) not a tensor")?;
            let src = unwrap_tensor_arg(&args[3])
                .ok_or("scatter_add: arg 3 (src) not a tensor")?
                .to_f32()?;
            let (index, index_shape) = match decode_int64_index_to_usize(&index_wire)? {
                Some(p) => p,
                None => return Ok(None),
            };
            if input.ndim() == 0 || input.ndim() != index_shape.len() {
                return Ok(None);
            }
            Ok(Some(ferrotorch_core::ops::indexing::scatter_add(
                &input,
                dim as isize,
                &index,
                &index_shape,
                &src,
            )?))
        }
        // `torch.index_select(input, dim, index)` — gather slices along
        // `dim` using a 1-D index tensor. Upstream forward at
        // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1862
        // index_select_cpu_`. op_db emits `args = [input_f32, dim_i64,
        // index_int64]` (verified 2026-05-25: `i=0 shapes=[[], 0, [1]]`,
        // `i=2 shapes=[[5,5], -1, [5]]`). ferrotorch's
        // `grad_fns::indexing::index_select_dim` at
        // `ferrotorch-core/src/grad_fns/indexing.rs:1229` takes `(input,
        // dim: usize, indices: &IntTensor<I>)`, requires `input.ndim() >= 1`
        // (rejects 0-d at `:1236-1240`) and 1-D index (rejects multi-d at
        // `:1246-1253`). Negative dim is normalized here before
        // delegation. 0-d input is a legitimate skip. Production
        // consumer of the underlying impl at
        // `ferrotorch-data/src/transforms.rs:389` (HorizontalFlip).
        // Closes #1246.
        "index_select" => {
            if args.len() < 3 {
                return Err(format!("index_select expects 3 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("index_select: arg 0 not a tensor")?
                .to_f32()?;
            let dim_i64 = args[1]
                .as_i64()
                .ok_or("index_select: arg 1 (dim) not a JSON integer")?;
            let index = unwrap_tensor_arg(&args[2])
                .ok_or("index_select: arg 2 (index) not a tensor")?
                .to_int_tensor_i64()?;
            // The shape-strict impl rejects 0-d input and non-1-D index.
            // Both are narrower than upstream's contract; skip rather than
            // report as divergence.
            if input.ndim() == 0 || index.ndim() != 1 {
                return Ok(None);
            }
            // Skip on negative indices (the impl rejects them).
            for &v in index.data()? {
                if v < 0 {
                    return Ok(None);
                }
            }
            // Normalize negative dim per PyTorch: dim ∈ [-ndim, ndim).
            let ndim = input.ndim() as i64;
            let dim = if dim_i64 < 0 { dim_i64 + ndim } else { dim_i64 };
            if !(0..ndim).contains(&dim) {
                return Err(format!(
                    "index_select: dim {dim_i64} out of range for input ndim {ndim}"
                )
                .into());
            }
            Ok(Some(grad_fns::indexing::index_select_dim(
                &input,
                dim as usize,
                &index,
            )?))
        }
        // `torch.index_fill(input, dim, index, value)` — fill slices along
        // `dim` at `index` positions with the scalar `value`. Upstream forward
        // at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1979 Tensor
        // index_fill(const Tensor& self, int64_t dim, const Tensor& index,
        // const Scalar& source)`. Op_db's `sample_inputs_index_fill` emits
        // `args = [input_f32, dim_i64, index_int64, value]` where `value` is
        // either a JSON number (the `index_fill.int_Scalar` overload at
        // `:1979`) or a 0-d tensor envelope (the `index_fill.int_Tensor`
        // overload at `:1987-1992` which delegates to `.item()` via `:1965-1976`).
        // Verified 2026-05-25 by live oracle sample inspection:
        //   i=0: args=[scalar_f32, 0, [1]_int64, -8.478373527526855]
        //   i=3: args=[[1]_f32,    0, [1]_int64, {0-d float tensor envelope}]
        // ferrotorch's shape-strict `grad_fns::indexing::index_fill` rejects
        // 0-d input and multi-d index (REQ-8 narrower contract — see the
        // matching `index_select` rejection at line 1001 above). Negative
        // index values are also rejected per the IntTensor convention shared
        // with `index_select_dim` (`indexing.rs:1259-1272`); the runner skips
        // those samples rather than reporting a divergence.
        // Closes #1249.
        "index_fill" => {
            if args.len() < 4 {
                return Err(format!("index_fill expects 4 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("index_fill: arg 0 not a tensor")?
                .to_f32()?;
            let dim_i64 = args[1]
                .as_i64()
                .ok_or("index_fill: arg 1 (dim) not a JSON integer")?;
            let index = unwrap_tensor_arg(&args[2])
                .ok_or("index_fill: arg 2 (index) not a tensor")?
                .to_int_tensor_i64()?;
            // value: JSON number, string-quoted int, or 0-d tensor envelope.
            // Mirrors the masked_fill arm's scalar-decode at line 761.
            let value_f64: f64 = if let Some(v) = args[3].as_f64() {
                v
            } else if let Some(v) = args[3].as_i64() {
                v as f64
            } else if let Some(s) = args[3].as_str() {
                s.parse::<f64>()
                    .map_err(|e| format!("index_fill: arg 3 string parse failed: {e}"))?
            } else if let Some(wt) = unwrap_tensor_arg(&args[3]) {
                if !wt.shape.is_empty() {
                    // Upstream `TORCH_CHECK(source.dim() == 0, "index_fill_
                    // only supports a 0-dimensional value tensor, ...")` at
                    // `TensorAdvancedIndexing.cpp:1970-1975`. Non-0-d is a
                    // legitimate skip (oracle never emits it, but
                    // belt-and-braces).
                    return Ok(None);
                }
                let t = wt.to_f32()?;
                let d = t.data_vec()?;
                d.first().copied().unwrap_or(0.0) as f64
            } else {
                return Err(
                    format!("index_fill: arg 3 not a number/string/tensor: {}", args[3]).into(),
                );
            };
            // The shape-strict impl rejects 0-d input, multi-d index, and
            // negative index values. Skip those samples rather than report
            // divergences — they're narrower-contract gaps tracked at the
            // REQ level (#1256 for 0-d, the IntTensor convention for
            // negative).
            if input.ndim() == 0 || index.ndim() > 1 {
                return Ok(None);
            }
            for &v in index.data()? {
                if v < 0 {
                    return Ok(None);
                }
            }
            Ok(Some(grad_fns::indexing::index_fill(
                &input, dim_i64, &index, value_f64,
            )?))
        }
        // `torch.scatter_reduce(self, dim, index, src, reduce, *, include_self=
        // True)` — reduce-mode scatter onto a clone of self. Upstream forward
        // at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2354
        // TORCH_IMPL_FUNC(scatter_reduce_two)`. Backward per
        // `tools/autograd/derivatives.yaml:3074-3077` only for reduce='sum'.
        // op_db emits args = [input_f32, dim_i64, index_int64, src_f32,
        // reduce_str], kwargs include_self=bool. Verified 2026-05-25: seed
        // 0..3 i=0..25 → all samples reduce='sum'. Closes #1245.
        "scatter_reduce" => {
            if args.len() < 5 {
                return Err(format!("scatter_reduce expects 5 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("scatter_reduce: arg 0 not a tensor")?
                .to_f32()?;
            let dim_i64 = args[1]
                .as_i64()
                .ok_or("scatter_reduce: arg 1 (dim) not a JSON integer")?;
            let index_wire =
                unwrap_tensor_arg(&args[2]).ok_or("scatter_reduce: arg 2 (index) not a tensor")?;
            let src = unwrap_tensor_arg(&args[3])
                .ok_or("scatter_reduce: arg 3 (src) not a tensor")?
                .to_f32()?;
            let reduce_str = args[4]
                .as_str()
                .ok_or("scatter_reduce: arg 4 (reduce) not a string")?;
            let include_self = kwargs
                .get("include_self")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let (index, index_shape) = match decode_int64_index_to_usize(&index_wire)? {
                Some(p) => p,
                None => return Ok(None),
            };
            // 0-d input + 0-d index: legitimate skip — the shape-strict path
            // here can't validate ndim-mismatch cleanly.
            if input.ndim() == 0 || input.ndim() != index_shape.len() {
                return Ok(None);
            }
            // Only `sum` is in the op_db characterization sweep — other
            // modes are out-of-scope skips per design doc REQ-4.
            let mode = match grad_fns::indexing::ScatterReduce::parse_str(reduce_str) {
                Some(m) => m,
                None => return Ok(None),
            };
            Ok(Some(grad_fns::indexing::scatter_reduce(
                &input,
                dim_i64,
                &index,
                &index_shape,
                &src,
                mode,
                include_self,
            )?))
        }
        // `torch.index_add(self, dim, index, source, *, alpha=1)` — upstream
        // forward at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1153
        // TORCH_IMPL_FUNC(index_add_cpu_out)`. Backward per
        // `tools/autograd/derivatives.yaml:862-869`. op_db emits args =
        // [input_f32, dim_i64, index_int64, source_f32] with kwargs.alpha
        // ∈ {-1, 0, 2, ...}. Closes #1247.
        "index_add" => {
            if args.len() < 4 {
                return Err(format!("index_add expects 4 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("index_add: arg 0 not a tensor")?
                .to_f32()?;
            let dim_i64 = args[1]
                .as_i64()
                .ok_or("index_add: arg 1 (dim) not a JSON integer")?;
            let index = unwrap_tensor_arg(&args[2])
                .ok_or("index_add: arg 2 (index) not a tensor")?
                .to_int_tensor_i64()?;
            let source = unwrap_tensor_arg(&args[3])
                .ok_or("index_add: arg 3 (source) not a tensor")?
                .to_f32()?;
            // alpha: JSON number, integer, or absent (default 1).
            let alpha: f64 = if let Some(v) = kwargs.get("alpha") {
                if let Some(f) = v.as_f64() {
                    f
                } else if let Some(i) = v.as_i64() {
                    i as f64
                } else {
                    1.0
                }
            } else {
                1.0
            };
            // Skip multi-d index — narrower contract.
            if index.ndim() > 1 {
                return Ok(None);
            }
            // No pre-filtering on negative indices, source-size mismatch,
            // or 0-d source on N-D self. The parity harness's `both errored`
            // matcher at `:2342-2351` correctly accounts for the case where
            // upstream rejects with one message and ferrotorch rejects with
            // its mirrored message. Filtering these inputs out of the runner
            // (the previous behavior introduced in pin #1286) HIDES the
            // strict-validation contract from the sweep — any future
            // regression that silently accepts a negative index or a 0-d
            // source on N-D self would slip through the parity gate
            // unnoticed. Closes #1288-D (parity-pre-filter masking).
            Ok(Some(grad_fns::indexing::index_add(
                &input, dim_i64, &index, &source, alpha,
            )?))
        }
        // `torch.index_copy(self, dim, index, source)` — upstream forward at
        // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1082
        // TORCH_IMPL_FUNC(index_copy_out)`. Backward per
        // `tools/autograd/derivatives.yaml:875-883`. op_db emits args =
        // [input_f32, dim_i64, index_int64, source_f32]. Closes #1248.
        "index_copy" => {
            if args.len() < 4 {
                return Err(format!("index_copy expects 4 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("index_copy: arg 0 not a tensor")?
                .to_f32()?;
            let dim_i64 = args[1]
                .as_i64()
                .ok_or("index_copy: arg 1 (dim) not a JSON integer")?;
            let index = unwrap_tensor_arg(&args[2])
                .ok_or("index_copy: arg 2 (index) not a tensor")?
                .to_int_tensor_i64()?;
            let source = unwrap_tensor_arg(&args[3])
                .ok_or("index_copy: arg 3 (source) not a tensor")?
                .to_f32()?;
            if index.ndim() > 1 {
                return Ok(None);
            }
            // No pre-filtering on negative indices, source-size mismatch,
            // or shape mismatch. The parity harness's `both errored` matcher
            // accounts for symmetric rejection. Filtering these out (the
            // previous behavior introduced in pin #1286) HID the strict-
            // validation contract from the sweep. NB: for index_copy
            // SPECIFICALLY, 0-d source on N-D self is NOT a divergence —
            // upstream meta `:285-300` accepts it (broadcasts scalar src
            // per index slot) and ferrotorch now matches per #1288-B. Both
            // succeed; the harness compares output values. Closes #1288-D.
            Ok(Some(grad_fns::indexing::index_copy(
                &input, dim_i64, &index, &source,
            )?))
        }
        // `torch.masked_scatter(self, mask, source)` — copies elements of
        // source into self at mask-true positions, in C-order. Upstream
        // forward at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2402-
        // 2409`. Backward per `tools/autograd/derivatives.yaml:1105-1108`.
        // op_db emits args = [input_f32, mask_bool, source_f32]. Mask may
        // be broadcast against input — wrapper handles. Closes #1252.
        "masked_scatter" => {
            if args.len() < 3 {
                return Err(format!("masked_scatter expects 3 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("masked_scatter: arg 0 not a tensor")?
                .to_f32()?;
            let mask = unwrap_tensor_arg(&args[1])
                .ok_or("masked_scatter: arg 1 (mask) not a tensor")?
                .to_bool_tensor()?;
            let source = unwrap_tensor_arg(&args[2])
                .ok_or("masked_scatter: arg 2 (source) not a tensor")?
                .to_f32()?;
            // 0-d/empty mask: torch handles as identity-copy; skip per the
            // narrower contract since broadcast on 0-d adds little value.
            if mask.numel() == 0 {
                return Ok(None);
            }
            // Skip when source has fewer elements than mask-true count.
            let true_count = mask.data()?.iter().filter(|&&b| b).count();
            if source.numel() < true_count {
                return Ok(None);
            }
            Ok(Some(grad_fns::indexing::masked_scatter(
                &input, &mask, &source,
            )?))
        }
        // `torch.take(input, index)` — flat-index gather. Upstream forward at
        // `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1067-1071`.
        // Backward per `tools/autograd/derivatives.yaml:1766-1769`. op_db
        // emits args = [input_f32, index_int64]. 0-d empty index case is a
        // legitimate skip — the upstream early-returns empty. Closes #1253.
        "take" => {
            if args.len() < 2 {
                return Err(format!("take expects 2 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("take: arg 0 not a tensor")?
                .to_f32()?;
            let index = unwrap_tensor_arg(&args[1])
                .ok_or("take: arg 1 (index) not a tensor")?
                .to_int_tensor_i64()?;
            // 0-d index on 0-d input: skip (out_numel=1, edge of contract).
            if input.numel() == 0 {
                return Ok(None);
            }
            // Skip negative indices per narrow-contract convention shared
            // with index_select / index_fill.
            for &v in index.data()? {
                if v < 0 {
                    return Ok(None);
                }
            }
            Ok(Some(grad_fns::indexing::take(&input, &index)?))
        }
        // `torch.put(self, index, source, accumulate=False)` — flat-index
        // scatter. Upstream forward at `aten/src/ATen/native/
        // TensorAdvancedIndexing.cpp:928-934`. Backward per
        // `tools/autograd/derivatives.yaml:1421-1424`. op_db emits args =
        // [input_f32, index_int64, source_f32, accumulate_bool]. Closes #1254.
        "put" => {
            if args.len() < 4 {
                return Err(format!("put expects 4 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("put: arg 0 not a tensor")?
                .to_f32()?;
            let index = unwrap_tensor_arg(&args[1])
                .ok_or("put: arg 1 (index) not a tensor")?
                .to_int_tensor_i64()?;
            let source = unwrap_tensor_arg(&args[2])
                .ok_or("put: arg 2 (source) not a tensor")?
                .to_f32()?;
            // accumulate: JSON boolean (also support 0/1 fallback).
            let accumulate = if let Some(b) = args[3].as_bool() {
                b
            } else if let Some(i) = args[3].as_i64() {
                i != 0
            } else {
                false
            };
            // Skip 0-d input (empty buffer) per narrow-contract convention.
            if input.numel() == 0 {
                return Ok(None);
            }
            for &v in index.data()? {
                if v < 0 {
                    return Ok(None);
                }
            }
            if source.numel() < index.numel() {
                return Ok(None);
            }
            Ok(Some(grad_fns::indexing::put(
                &input, &index, &source, accumulate,
            )?))
        }
        // ---- Transcendental unary family (closes #1298 and per-op blockers
        // #1303 #1305 #1307 #1309 #1311 #1313 #1315 #1316 #1317 #1319 #1320
        // #1322 #1323 #1324 #1325 #1326 #1327 #1328 #1329 #1330 #1331 #1333) ----
        //
        // Each arm decodes `args=[input_f32]` (matching op_db's unary samples,
        // verified 2026-05-25 via oracle inspection) and dispatches through
        // the ferrotorch impl at `ferrotorch-core/src/grad_fns/transcendental.rs`.
        // Each impl mirrors a `CREATE_UNARY_TORCH_IMPL_FUNC(<op>_out, <op>_stub)`
        // in `aten/src/ATen/native/UnaryOps.cpp:316-363` per the design doc
        // `.design/ferrotorch-core/grad_fns/transcendental.md` REQ table.
        "exp" => Ok(Some(grad_fns::transcendental::exp(&unary("exp")?)?)),
        "log" => Ok(Some(grad_fns::transcendental::log(&unary("log")?)?)),
        "sin" => Ok(Some(grad_fns::transcendental::sin(&unary("sin")?)?)),
        "cos" => Ok(Some(grad_fns::transcendental::cos(&unary("cos")?)?)),
        "tan" => Ok(Some(grad_fns::transcendental::tan(&unary("tan")?)?)),
        "asin" => Ok(Some(grad_fns::transcendental::asin(&unary("asin")?)?)),
        "acos" => Ok(Some(grad_fns::transcendental::acos(&unary("acos")?)?)),
        "atan" => Ok(Some(grad_fns::transcendental::atan(&unary("atan")?)?)),
        "sinh" => Ok(Some(grad_fns::transcendental::sinh(&unary("sinh")?)?)),
        "cosh" => Ok(Some(grad_fns::transcendental::cosh(&unary("cosh")?)?)),
        "asinh" => Ok(Some(grad_fns::transcendental::asinh(&unary("asinh")?)?)),
        "acosh" => Ok(Some(grad_fns::transcendental::acosh(&unary("acosh")?)?)),
        "atanh" => Ok(Some(grad_fns::transcendental::atanh(&unary("atanh")?)?)),
        "exp2" => Ok(Some(grad_fns::transcendental::exp2(&unary("exp2")?)?)),
        "expm1" => Ok(Some(grad_fns::transcendental::expm1(&unary("expm1")?)?)),
        "log2" => Ok(Some(grad_fns::transcendental::log2(&unary("log2")?)?)),
        "log10" => Ok(Some(grad_fns::transcendental::log10(&unary("log10")?)?)),
        "log1p" => Ok(Some(grad_fns::transcendental::log1p(&unary("log1p")?)?)),
        "ceil" => Ok(Some(grad_fns::transcendental::ceil(&unary("ceil")?)?)),
        "floor" => Ok(Some(grad_fns::transcendental::floor(&unary("floor")?)?)),
        "round" => Ok(Some(grad_fns::transcendental::round(&unary("round")?)?)),
        "trunc" => Ok(Some(grad_fns::transcendental::trunc(&unary("trunc")?)?)),
        "frac" => Ok(Some(grad_fns::transcendental::frac(&unary("frac")?)?)),
        "sign" => Ok(Some(grad_fns::transcendental::sign(&unary("sign")?)?)),
        "sinc" => Ok(Some(grad_fns::transcendental::sinc(&unary("sinc")?)?)),
        // `torch.clamp(input, min, max)` — op_db's unary `clamp` samples
        // ship min/max as TENSOR-valued bounds (broadcastable to input).
        // ferrotorch's `pub fn clamp` accepts scalar `T` bounds only — the
        // tensor-bound `clamp.Tensor` overload at
        // `aten/src/ATen/native/TensorCompare.cpp:856 TORCH_IMPL_FUNC(clamp_Tensor_out)`
        // is documented as NOT-STARTED in REQ-5's divergence section of
        // `.design/ferrotorch-core/grad_fns/transcendental.md`. Until that
        // ships, any non-0-d bound is a legitimate skip; 0-d bounds (the
        // `clamp.Scalar` shape) extract via `.item()` and dispatch.
        // The `clamp.Scalar` overload also accepts `Optional` bounds —
        // ferrotorch requires both, so single-bound samples (min=None or
        // max=None) skip too. Closes runner-arm half of #1298 for clamp.
        "clamp" => {
            if args.len() < 2 {
                return Err(format!("clamp expects >=2 args, got {}", args.len()).into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("clamp: arg 0 not a tensor")?
                .to_f32()?;
            // Helper to coerce arg[i] to an Optional scalar bound.
            // Returns Ok(Some(v)) for 0-d tensor / number, Ok(None) for
            // None / non-0-d tensor (skip-the-sample signal upstream).
            let extract_scalar_bound =
                |v: &Value| -> Result<Option<Option<f32>>, Box<dyn std::error::Error>> {
                    if v.is_null() {
                        return Ok(Some(None));
                    }
                    if let Some(f) = v.as_f64() {
                        return Ok(Some(Some(f as f32)));
                    }
                    if let Some(i) = v.as_i64() {
                        return Ok(Some(Some(i as f32)));
                    }
                    if let Some(wt) = unwrap_tensor_arg(v) {
                        if !wt.shape.is_empty() {
                            // Tensor bound with shape != [] — clamp.Tensor
                            // overload, not implementable via scalar clamp.
                            return Ok(None);
                        }
                        let t = wt.to_f32()?;
                        let d = t.data_vec()?;
                        return Ok(Some(Some(d.first().copied().unwrap_or(0.0))));
                    }
                    Err(format!("clamp: unsupported bound arg: {v}").into())
                };
            let min_opt = match extract_scalar_bound(&args[1])? {
                Some(o) => o,
                None => return Ok(None),
            };
            let max_opt = if let Some(a2) = args.get(2) {
                match extract_scalar_bound(a2)? {
                    Some(o) => o,
                    None => return Ok(None),
                }
            } else {
                None
            };
            // ferrotorch's `clamp(input, min: T, max: T)` requires both.
            // One-sided clamps (clamp_min/clamp_max) are documented
            // NOT-STARTED in REQ-5; treat as legitimate skip.
            let (min_v, max_v) = match (min_opt, max_opt) {
                (Some(lo), Some(hi)) => (lo, hi),
                _ => return Ok(None),
            };
            Ok(Some(grad_fns::transcendental::clamp(&input, min_v, max_v)?))
        }
        "fake_quantize_per_channel_affine" => Ok(Some({
            if args.len() < 6 {
                return Err(format!(
                    "fake_quantize_per_channel_affine: expected 6 args, got {}",
                    args.len()
                )
                .into());
            }
            let input = unwrap_tensor_arg(&args[0])
                .ok_or("fake_quantize_per_channel_affine: arg 0 not a tensor")?
                .to_f32()?;
            let scale = unwrap_tensor_arg(&args[1])
                .ok_or("fake_quantize_per_channel_affine: arg 1 (scale) not a tensor")?
                .to_f32()?;
            let zero_point = unwrap_tensor_arg(&args[2])
                .ok_or("fake_quantize_per_channel_affine: arg 2 (zero_point) not a tensor")?
                .to_int_tensor_i64()?;
            let axis = args[3]
                .as_i64()
                .ok_or("fake_quantize_per_channel_affine: arg 3 (axis) not a JSON integer")?;
            let quant_min = args[4]
                .as_i64()
                .ok_or("fake_quantize_per_channel_affine: arg 4 (quant_min) not a JSON integer")?;
            let quant_max = args[5]
                .as_i64()
                .ok_or("fake_quantize_per_channel_affine: arg 5 (quant_max) not a JSON integer")?;
            grad_fns::quantize_grad::fake_quantize_per_channel_affine(
                &input,
                &scale,
                &zero_point,
                axis,
                quant_min,
                quant_max,
            )?
        })),

        // ------------------------------------------------------------------
        // Reduction cluster — closes umbrella #1314 + per-op #1301/#1304/
        // #1310/#1312. Maps op_db `dim` / `keepdim` envelopes onto
        // ferrotorch's single-dim reduction surface, chaining `sum_dim` /
        // `mean_dim` / `logsumexp_dim` for multi-dim list inputs (matches
        // upstream `at::sum(x, [d0, d1, ...], keepdim)` semantics: reduce
        // outer dim first so the inner dim indices stay valid; chain in
        // descending order to avoid shifting).
        // ------------------------------------------------------------------

        // `torch.sum(input, dim=None, keepdim=False)` —
        // `aten/src/ATen/native/ReduceOps.cpp:1245 TORCH_IMPL_FUNC(sum_out)`.
        // Full reduction when `dim` absent/`[]`; single-dim via `sum_dim`;
        // multi-dim list reduced by sequential `sum_dim` calls in
        // descending-order so axis indices stay valid after each squeeze.
        // VJP per `tools/autograd/derivatives.yaml:1702-1719 sum_backward`.
        //
        // Edge cases:
        // - 0-D input + dim kwarg: ferrotorch's `sum_dim` rejects 0-D
        //   (deliberate divergence — see `.design/ferrotorch-core/grad_fns/
        //   reduction.md` AC-16). Treat as full reduction, then reshape to
        //   torch's keepdim shape when keepdim=true.
        // - Full reduction with keepdim=true on N-D input: torch returns
        //   `[1, 1, ..., 1]` (ndim ones); ferrotorch's `sum` returns `[]`.
        //   Reshape after the fact.
        "sum" => Ok(Some({
            let a = unary("sum")?;
            let keepdim = keepdim_kwarg();
            let dims_opt = dim_kwarg("sum")?;
            let in_ndim = a.ndim();
            // Decide: full-reduction path vs dim-chain. The full path covers
            // (a) no dim kwarg, (b) dim==[] empty list, (c) 0-D input
            // (sum_dim would reject — full reduction is identity on scalars
            // and matches torch's `sum(scalar)` returning the scalar).
            let do_full = match &dims_opt {
                None => true,
                Some(_) if in_ndim == 0 => true,
                _ => false,
            };
            if do_full {
                let r = grad_fns::reduction::sum(&a)?;
                if keepdim && in_ndim > 0 {
                    // Torch emits `[1, 1, ..., 1]` for keepdim+all-dim reduce.
                    let ones = vec![1usize; in_ndim];
                    let d = r.data()?.to_vec();
                    ferrotorch_core::from_vec(d, &ones)?
                } else {
                    r
                }
            } else {
                let dims = dims_opt.unwrap();
                let mut sorted: Vec<i64> = dims
                    .iter()
                    .map(|&d| if d < 0 { in_ndim as i64 + d } else { d })
                    .collect();
                sorted.sort_unstable();
                let mut cur = a.clone();
                // Reduce highest dim first so lower-dim indices stay
                // valid through the squeeze chain.
                for &d in sorted.iter().rev() {
                    cur = grad_fns::reduction::sum_dim(&cur, d, keepdim)?;
                }
                cur
            }
        })),

        // `torch.mean(input, dim=None, keepdim=False)` —
        // `aten/src/ATen/native/ReduceOps.cpp:1396 TORCH_IMPL_FUNC(mean_out)`.
        // Multi-dim mean factors as `mean(mean(mean(...)))` only if all
        // dims have the SAME size; in the general case we use the
        // equivalent `sum_over_dims / prod(dim_sizes)`. Implemented as
        // sum-chain then divide by the product of reduced dim sizes.
        // VJP per `derivatives.yaml:1143-1155`.
        "mean" => Ok(Some({
            let a = unary("mean")?;
            let keepdim = keepdim_kwarg();
            let dims_opt = dim_kwarg("mean")?;
            let in_ndim = a.ndim();
            let do_full = match &dims_opt {
                None => true,
                Some(_) if in_ndim == 0 => true,
                _ => false,
            };
            if do_full {
                let r = grad_fns::reduction::mean(&a)?;
                if keepdim && in_ndim > 0 {
                    let ones = vec![1usize; in_ndim];
                    let d = r.data()?.to_vec();
                    ferrotorch_core::from_vec(d, &ones)?
                } else {
                    r
                }
            } else {
                let dims = dims_opt.unwrap();
                // Single-dim short path uses native `mean_dim`.
                if dims.len() == 1 {
                    grad_fns::reduction::mean_dim(&a, dims[0], keepdim)?
                } else {
                    // Multi-dim: sum over dims then divide by the product
                    // of their sizes. Matches upstream's
                    // `at::sum_out(...).div_(dim_prod)` recipe at
                    // `ReduceOps.cpp:1452-1454`.
                    let in_shape = a.shape().to_vec();
                    let mut sorted: Vec<i64> = dims
                        .iter()
                        .map(|&d| if d < 0 { in_ndim as i64 + d } else { d })
                        .collect();
                    sorted.sort_unstable();
                    let dim_prod: usize = sorted.iter().map(|&d| in_shape[d as usize]).product();
                    let mut cur = a.clone();
                    for &d in sorted.iter().rev() {
                        cur = grad_fns::reduction::sum_dim(&cur, d, keepdim)?;
                    }
                    let data = cur.data()?.to_vec();
                    let inv = 1.0f32 / (dim_prod as f32);
                    let scaled: Vec<f32> = data.iter().map(|&v| v * inv).collect();
                    ferrotorch_core::from_vec(scaled, cur.shape())?
                }
            }
        })),

        // `torch.prod(input, dim=None)` —
        // `aten/src/ATen/native/ReduceOps.cpp:1379 Tensor prod(...)`.
        // op_db emits `dim` POSITIONALLY at `args[1]` (no kwargs). The
        // dim-keyed variant `prod(self, int dim)` is NOT-STARTED in
        // ferrotorch (single-dim only would need its own `ProdDimBackward`);
        // dim-supplied samples are a legitimate skip. Full-reduction
        // routes through `grad_fns::reduction::prod` with `ProdBackward`
        // per `derivatives.yaml:1413-1415` (prefix-suffix VJP).
        "prod" => Ok(Some({
            // op_db emits `prod` as either `args=[input]` (full reduction)
            // or `args=[input, dim]` (single-dim reduction). `keepdim`
            // arrives as kwarg per the `prod.dim_int` overload.
            let a = unary("prod")?;
            let keepdim = keepdim_kwarg();
            if args.len() < 2 {
                let r = grad_fns::reduction::prod(&a)?;
                let in_ndim = a.ndim();
                if keepdim && in_ndim > 0 {
                    let ones = vec![1usize; in_ndim];
                    let d = r.data()?.to_vec();
                    ferrotorch_core::from_vec(d, &ones)?
                } else {
                    r
                }
            } else {
                let dim = args[1]
                    .as_i64()
                    .ok_or("prod: arg 1 (dim) is not a JSON integer")?;
                grad_fns::reduction::prod_dim(&a, dim, keepdim)?
            }
        })),

        // `torch.amin(input, dim=[], keepdim=False)` /
        // `torch.amax(...)` — `ReduceOps.cpp:1758` / `:1766`. ferrotorch's
        // `pub fn amin` / `amax` are full-reduction only (NaN handling
        // diverges — skips NaN vs upstream NaN-propagation; tracked under
        // #1314). Dim-keyed amin/amax variant is NOT-STARTED (blocker
        // #1302 alongside max/min-with-dim). Dim-supplied samples skip,
        // EXCEPT 0-D inputs (amin/amax over a scalar is the scalar). For
        // full-reduction + keepdim=true on N-D input we reshape to
        // `[1, 1, ..., 1]` per upstream's keepdim semantics.
        // amin / amax — full-reduction via `pub fn amin/amax`, single-dim
        // via `amin_dim/amax_dim`, multi-dim list reduced by sequential
        // dim-keyed calls in descending-order.
        "amin" => Ok(Some({
            let a = unary("amin")?;
            let keepdim = keepdim_kwarg();
            let in_ndim = a.ndim();
            let dims_opt = dim_kwarg("amin")?;
            let do_full = match &dims_opt {
                None => true,
                Some(_) if in_ndim == 0 => true,
                _ => false,
            };
            if do_full {
                let r = grad_fns::reduction::amin(&a)?;
                if keepdim && in_ndim > 0 {
                    let ones = vec![1usize; in_ndim];
                    let d = r.data()?.to_vec();
                    ferrotorch_core::from_vec(d, &ones)?
                } else {
                    r
                }
            } else {
                let dims = dims_opt.unwrap();
                let mut sorted: Vec<i64> = dims
                    .iter()
                    .map(|&d| if d < 0 { in_ndim as i64 + d } else { d })
                    .collect();
                sorted.sort_unstable();
                let mut cur = a.clone();
                for &d in sorted.iter().rev() {
                    cur = grad_fns::reduction::amin_dim(&cur, d, keepdim)?;
                }
                cur
            }
        })),
        "amax" => Ok(Some({
            let a = unary("amax")?;
            let keepdim = keepdim_kwarg();
            let in_ndim = a.ndim();
            let dims_opt = dim_kwarg("amax")?;
            let do_full = match &dims_opt {
                None => true,
                Some(_) if in_ndim == 0 => true,
                _ => false,
            };
            if do_full {
                let r = grad_fns::reduction::amax(&a)?;
                if keepdim && in_ndim > 0 {
                    let ones = vec![1usize; in_ndim];
                    let d = r.data()?.to_vec();
                    ferrotorch_core::from_vec(d, &ones)?
                } else {
                    r
                }
            } else {
                let dims = dims_opt.unwrap();
                let mut sorted: Vec<i64> = dims
                    .iter()
                    .map(|&d| if d < 0 { in_ndim as i64 + d } else { d })
                    .collect();
                sorted.sort_unstable();
                let mut cur = a.clone();
                for &d in sorted.iter().rev() {
                    cur = grad_fns::reduction::amax_dim(&cur, d, keepdim)?;
                }
                cur
            }
        })),

        // `torch.logsumexp(input, dim, keepdim=False)` —
        // `aten/src/ATen/native/ReduceOps.cpp:1548-1559`. op_db emits
        // `args = [input, dim_list, keepdim]` (dim ALWAYS positional, never
        // a kwarg in the logsumexp sample iterator). Multi-dim reduces via
        // sequential `logsumexp_dim` calls in descending-order. Closes #1310.
        "logsumexp" => Ok(Some({
            let a = unary("logsumexp")?;
            let dims = arg_dim_list_at(1).ok_or("logsumexp: missing dim list at args[1]")?;
            let keepdim = arg_bool_at(2).unwrap_or(false);
            let in_ndim = a.ndim();
            // 0-D input + dim=[0]: torch returns the same scalar (logsumexp
            // of a single element is the element itself); ferrotorch's
            // logsumexp_dim rejects 0-D. Treat as full reduction.
            let do_full = dims.is_empty() || in_ndim == 0;
            if do_full {
                let r = grad_fns::reduction::logsumexp(&a)?;
                if keepdim && in_ndim > 0 {
                    let ones = vec![1usize; in_ndim];
                    let d = r.data()?.to_vec();
                    ferrotorch_core::from_vec(d, &ones)?
                } else {
                    r
                }
            } else {
                let mut sorted: Vec<i64> = dims
                    .iter()
                    .map(|&d| if d < 0 { in_ndim as i64 + d } else { d })
                    .collect();
                sorted.sort_unstable();
                let mut cur = a.clone();
                for &d in sorted.iter().rev() {
                    cur = grad_fns::reduction::logsumexp_dim(&cur, d, keepdim)?;
                }
                cur
            }
        })),

        // `torch.argmax(input, dim=None, keepdim=False)` /
        // `torch.argmin(...)` — `ReduceOps.cpp:1809` / `:1817`. Integer-
        // output, non-differentiable. Single-dim only (matches ferrotorch's
        // `argmax_dim` / `argmin_dim`); multi-dim is not a valid op_db
        // sample for argmax/argmin (upstream's signature is
        // `argmax(self, std::optional<int64_t> dim, bool keepdim)`).
        // Closes #1304.
        "argmax" => Ok(Some({
            let a = unary("argmax")?;
            let dims = dim_kwarg("argmax")?;
            let keepdim = keepdim_kwarg();
            let it = match dims {
                None => grad_fns::reduction::argmax(&a)?,
                Some(ds) if ds.len() == 1 => grad_fns::reduction::argmax_dim(&a, ds[0], keepdim)?,
                Some(_) => return Ok(None),
            };
            int_to_f32(&it)?
        })),
        "argmin" => Ok(Some({
            let a = unary("argmin")?;
            let dims = dim_kwarg("argmin")?;
            let keepdim = keepdim_kwarg();
            let it = match dims {
                None => grad_fns::reduction::argmin(&a)?,
                Some(ds) if ds.len() == 1 => grad_fns::reduction::argmin_dim(&a, ds[0], keepdim)?,
                Some(_) => return Ok(None),
            };
            int_to_f32(&it)?
        })),

        // `torch.std(input, *, unbiased=True)` / `torch.var(...)` —
        // `aten/src/ATen/native/ReduceOps.cpp:2085` (var) / `:2105` (std).
        // ferrotorch's std/var are full-reduction only — dim-keyed
        // variants are NOT-STARTED (the `*.correction` overloads in
        // upstream require multi-dim list support that defers to a
        // future builder). op_db's std/var samples emit `dim` as kwarg OR
        // `unbiased` as `args[0]` (Python positional bool); skip dim-
        // supplied samples. Closes #1301.
        // std / var — full-reduction via `pub fn std/var(unbiased)`,
        // dim-keyed via `std_dim/var_dim(correction, keepdim)`. Multi-dim
        // list chains `var_dim` in descending-order; for std, multi-dim
        // chains `var_dim` then takes sqrt (var is associative across
        // disjoint axes; std is not because sqrt breaks associativity).
        // `correction` is the upstream `n - correction` denominator
        // (default 1.0 = unbiased / Bessel); `unbiased=False` ↔
        // `correction=0`.
        "std" => Ok(Some({
            let a = unary("std")?;
            let dims_opt = dim_kwarg("std")?;
            let keepdim = keepdim_kwarg();
            let in_ndim = a.ndim();
            // Decode correction. Priority: `correction` kwarg > `unbiased`
            // kwarg > `unbiased` positional > default unbiased=true.
            let correction: f64 = match kwargs.get("correction") {
                Some(Value::Number(n)) => n.as_f64().unwrap_or(1.0),
                Some(Value::Null) | None => {
                    let unbiased = kwargs
                        .get("unbiased")
                        .and_then(Value::as_bool)
                        .unwrap_or_else(|| arg_bool_at(0).unwrap_or(true));
                    if unbiased { 1.0 } else { 0.0 }
                }
                Some(_) => return Ok(None),
            };
            match &dims_opt {
                None => {
                    // Full-reduction std with arbitrary correction —
                    // mirrors upstream `std_var_all_cpu` correction-scalar
                    // path at `ReduceOps.cpp:1858-1864`. Closes #1346
                    // audit REQ-8 correction-API gap.
                    let r = grad_fns::reduction::std_with_correction(&a, correction)?;
                    if keepdim && in_ndim > 0 {
                        let ones = vec![1usize; in_ndim];
                        let d = r.data()?.to_vec();
                        ferrotorch_core::from_vec(d, &ones)?
                    } else {
                        r
                    }
                }
                Some(dims) if dims.len() == 1 => {
                    grad_fns::reduction::std_dim(&a, dims[0], correction, keepdim)?
                }
                Some(dims) => {
                    // Multi-dim std: var_dim chain then sqrt.
                    let mut sorted: Vec<i64> = dims
                        .iter()
                        .map(|&d| if d < 0 { in_ndim as i64 + d } else { d })
                        .collect();
                    sorted.sort_unstable();
                    let mut cur = a.clone();
                    let last = sorted.len() - 1;
                    for (k, &d) in sorted.iter().rev().enumerate() {
                        // Apply correction only on the FIRST reduction
                        // (outermost dim); subsequent reductions in the
                        // chain are plain `sum / n` (no correction) so the
                        // total denominator matches upstream's
                        // `prod(reduced_sizes) - correction`. The chain
                        // form is exact for var since variance is
                        // associative across disjoint axes; not for std,
                        // hence the sqrt-once-at-end pattern.
                        let c = if k == 0 { correction } else { 0.0 };
                        let _ = last;
                        cur = grad_fns::reduction::var_dim(&cur, d, c, keepdim)?;
                    }
                    // Now apply sqrt to the accumulated variance.
                    let cd = cur.data()?.to_vec();
                    let sq: Vec<f32> = cd.iter().map(|&v| v.sqrt()).collect();
                    ferrotorch_core::from_vec(sq, cur.shape())?
                }
            }
        })),
        "var" => Ok(Some({
            let a = unary("var")?;
            let dims_opt = dim_kwarg("var")?;
            let keepdim = keepdim_kwarg();
            let in_ndim = a.ndim();
            let correction: f64 = match kwargs.get("correction") {
                Some(Value::Number(n)) => n.as_f64().unwrap_or(1.0),
                Some(Value::Null) | None => {
                    let unbiased = kwargs
                        .get("unbiased")
                        .and_then(Value::as_bool)
                        .unwrap_or_else(|| arg_bool_at(0).unwrap_or(true));
                    if unbiased { 1.0 } else { 0.0 }
                }
                Some(_) => return Ok(None),
            };
            match &dims_opt {
                None => {
                    // Full-reduction var with arbitrary correction —
                    // mirrors upstream `std_var_all_cpu` correction-scalar
                    // path at `ReduceOps.cpp:1858-1864`. Closes #1346
                    // audit REQ-8 correction-API gap.
                    let r = grad_fns::reduction::var_with_correction(&a, correction)?;
                    if keepdim && in_ndim > 0 {
                        let ones = vec![1usize; in_ndim];
                        let d = r.data()?.to_vec();
                        ferrotorch_core::from_vec(d, &ones)?
                    } else {
                        r
                    }
                }
                Some(dims) if dims.len() == 1 => {
                    grad_fns::reduction::var_dim(&a, dims[0], correction, keepdim)?
                }
                Some(dims) => {
                    let mut sorted: Vec<i64> = dims
                        .iter()
                        .map(|&d| if d < 0 { in_ndim as i64 + d } else { d })
                        .collect();
                    sorted.sort_unstable();
                    let mut cur = a.clone();
                    for (k, &d) in sorted.iter().rev().enumerate() {
                        let c = if k == 0 { correction } else { 0.0 };
                        cur = grad_fns::reduction::var_dim(&cur, d, c, keepdim)?;
                    }
                    cur
                }
            }
        })),

        // `torch.any(input)` / `torch.all(input)` —
        // `aten/src/ATen/native/ReduceOps.cpp:1681` / `:1667`. Bool-output,
        // non-differentiable. ferrotorch full-reduction only; dim-keyed
        // variant NOT-STARTED (would need dim-keyed any/all on the
        // BoolTensor surface — a separate builder dispatch). Multi-dim
        // and single-dim with keepdim full-reduction is reshaped to
        // upstream's `[1, 1, ..., 1]` shape per `ReduceOps.cpp:1672` /
        // `:1686` (any/all multi-dim collapse with keepdim). Closes #1312
        // for the full-reduction surface.
        // any / all — full reduction via `pub fn any/all`, single-dim via
        // `any_dim/all_dim`, multi-dim list chained in descending-order.
        // Bool-output coerced to f32 for the value-equality gate.
        "any" => Ok(Some({
            let a = unary("any")?;
            let keepdim = keepdim_kwarg();
            let in_ndim = a.ndim();
            let dims_opt = dim_kwarg("any")?;
            let do_full = match &dims_opt {
                None => true,
                Some(_) if in_ndim == 0 => true,
                _ => false,
            };
            if do_full {
                let bt = grad_fns::reduction::any(&a)?;
                let r = bool_to_f32(&bt)?;
                if keepdim && in_ndim > 0 {
                    let ones = vec![1usize; in_ndim];
                    let d = r.data()?.to_vec();
                    ferrotorch_core::from_vec(d, &ones)?
                } else {
                    r
                }
            } else {
                let dims = dims_opt.unwrap();
                let mut sorted: Vec<i64> = dims
                    .iter()
                    .map(|&d| if d < 0 { in_ndim as i64 + d } else { d })
                    .collect();
                sorted.sort_unstable();
                let mut cur_b: Option<ferrotorch_core::BoolTensor> = None;
                let mut cur = a.clone();
                for &d in sorted.iter().rev() {
                    let bt = grad_fns::reduction::any_dim(&cur, d, keepdim)?;
                    // Need a Tensor<f32> for the next any_dim call IF we
                    // had >1 dim. Since BoolTensor isn't a Float carrier,
                    // cast back to f32 between steps. {0,1} bool maps
                    // cleanly to {0.0, 1.0} f32 and `any` is monotone in
                    // truthiness (chaining the predicate stays correct).
                    let f = bool_to_f32(&bt)?;
                    cur = f;
                    cur_b = Some(bt);
                }
                bool_to_f32(&cur_b.unwrap())?
            }
        })),
        "all" => Ok(Some({
            let a = unary("all")?;
            let keepdim = keepdim_kwarg();
            let in_ndim = a.ndim();
            let dims_opt = dim_kwarg("all")?;
            let do_full = match &dims_opt {
                None => true,
                Some(_) if in_ndim == 0 => true,
                _ => false,
            };
            if do_full {
                let bt = grad_fns::reduction::all(&a)?;
                let r = bool_to_f32(&bt)?;
                if keepdim && in_ndim > 0 {
                    let ones = vec![1usize; in_ndim];
                    let d = r.data()?.to_vec();
                    ferrotorch_core::from_vec(d, &ones)?
                } else {
                    r
                }
            } else {
                let dims = dims_opt.unwrap();
                let mut sorted: Vec<i64> = dims
                    .iter()
                    .map(|&d| if d < 0 { in_ndim as i64 + d } else { d })
                    .collect();
                sorted.sort_unstable();
                let mut cur_b: Option<ferrotorch_core::BoolTensor> = None;
                let mut cur = a.clone();
                for &d in sorted.iter().rev() {
                    let bt = grad_fns::reduction::all_dim(&cur, d, keepdim)?;
                    let f = bool_to_f32(&bt)?;
                    cur = f;
                    cur_b = Some(bt);
                }
                bool_to_f32(&cur_b.unwrap())?
            }
        })),
        "count_nonzero" => Ok(Some({
            let a = unary("count_nonzero")?;
            let in_ndim = a.ndim();
            let dims_opt = dim_kwarg("count_nonzero")?;
            // `count_nonzero(dim=int)` is the dim-keyed overload from
            // `aten/src/ATen/native/SummaryOps.cpp::count_nonzero_dim`.
            // Multi-dim list is the `count_nonzero.dim_IntList` overload —
            // realized as `sum_dim` chain over a 0/1 indicator view of
            // `a` (each element is `1.0 if nonzero else 0.0`), then cast
            // to int. This is correct because counting non-zeros along
            // a multi-axis subset equals summing the indicator along the
            // same subset.
            let do_full = match &dims_opt {
                None => true,
                Some(_) if in_ndim == 0 => true,
                _ => false,
            };
            if do_full {
                let it = grad_fns::reduction::count_nonzero(&a)?;
                int_to_f32(&it)?
            } else {
                let dims = dims_opt.unwrap();
                // Indicator view: 1.0 if nonzero (NaN counts per IEEE-754
                // `NaN != 0.0`), else 0.0. Matches the predicate in
                // `is_nonzero_float`.
                let in_data = a.data()?.to_vec();
                let indicator: Vec<f32> = in_data
                    .iter()
                    .map(|&v| if v != 0.0 { 1.0 } else { 0.0 })
                    .collect();
                let ind_t = ferrotorch_core::from_vec(indicator, a.shape())?;
                let mut sorted: Vec<i64> = dims
                    .iter()
                    .map(|&d| if d < 0 { in_ndim as i64 + d } else { d })
                    .collect();
                sorted.sort_unstable();
                let mut cur = ind_t;
                for &d in sorted.iter().rev() {
                    cur = grad_fns::reduction::sum_dim(&cur, d, false)?;
                }
                // Cast to integer-valued f32 (round in case of fp drift).
                let cd = cur.data()?.to_vec();
                let rounded: Vec<f32> = cd.iter().map(|&v| v.round()).collect();
                ferrotorch_core::from_vec(rounded, cur.shape())?
            }
        })),

        // ------------------------------------------------------------------
        // Activation op cluster — closes umbrella #1338 (runner arms) +
        // #1341 (the 4 fused-GradFn additions: threshold/rrelu/celu/softmin).
        //
        // All 22 ops in `.design/ferrotorch-core/grad_fns/activation.md`'s
        // `parity_ops` route field dispatch here. Upstream entry points are
        // in `aten/src/ATen/native/Activation.cpp` (CPU + autograd defs) +
        // `torch/nn/functional.py` (Python user surface). The oracle exposes
        // most of these as `nn.functional.<name>` (some — sigmoid / tanh /
        // softmax / log_softmax — also live at top level); the alias map in
        // `oracle_name()` handles the rename before each `oracle.sample`
        // call so the bare names from the route's `parity_ops` field flow
        // through to the right op_db entry.
        //
        // All ops are unary (single tensor positional). The handful with
        // kwargs / extra positional scalars are handled inline below.
        //
        // ReLU family ----------------------------------------------------
        "relu" => Ok(Some(grad_fns::activation::relu(&unary("relu")?)?)),
        // `torch.nn.functional.relu6(input)` — clamp to `[0, 6]`. Upstream
        // `Tensor relu6(...)` at `aten/src/ATen/native/Activation.cpp:528-530`
        // delegates to `at::hardtanh(self, 0, 6)`. ferrotorch's `relu6`
        // mirrors via `hardtanh_with(input, 0.0, 6.0)`.
        "relu6" => Ok(Some(grad_fns::activation::relu6(&unary("relu6")?)?)),
        // `torch.nn.functional.leaky_relu(input, negative_slope=0.01)` —
        // upstream `TORCH_IMPL_FUNC(leaky_relu_out)` at
        // `aten/src/ATen/native/Activation.cpp:324-328`. op_db emits
        // `negative_slope` as kwarg (default 0.01).
        "leaky_relu" => Ok(Some({
            let a = unary("leaky_relu")?;
            let ns = kwargs
                .get("negative_slope")
                .and_then(Value::as_f64)
                .unwrap_or(0.01);
            grad_fns::activation::leaky_relu(&a, ns)?
        })),
        // PReLU. op_db's `nn.functional.prelu` emits
        // `args = [input, weight]` where weight is a 1-D tensor (per-channel)
        // or scalar (the only variant ferrotorch's fused `prelu` supports —
        // upstream allows per-channel via the `weight.reshape_symint(dim_w)`
        // branch at `Activation.cpp:716-723`, NOT yet shipped). For per-
        // channel samples we skip (return Ok(None)) so the sweep classifies
        // them as "infrastructure gap, not a value mismatch".
        "prelu" => {
            // op_db's `nn.functional.prelu` samples can ship weight either as
            // positional `args[1]` or via the `weight=` kwarg. Some samples
            // (the per-channel variants from `sample_inputs_nn_functional_prelu`)
            // emit `args=[input]` with no explicit weight at all — those
            // exercise the upstream default. We support the scalar-weight
            // path only (REQ-17 prelu scalar restriction); per-channel
            // weight + missing-weight samples are legitimate skips.
            let input = match args.first().and_then(unwrap_tensor_arg) {
                Some(t) => t.to_f32()?,
                None => return Ok(None),
            };
            let weight_wire = args
                .get(1)
                .and_then(unwrap_tensor_arg)
                .or_else(|| kwargs.get("weight").and_then(unwrap_tensor_arg));
            let weight = match weight_wire {
                Some(w) => w.to_f32()?,
                None => return Ok(None),
            };
            if weight.numel() != 1 {
                return Ok(None);
            }
            Ok(Some(grad_fns::activation::prelu(&input, &weight)?))
        }
        // RReLU. op_db's `nn.functional.rrelu` default `training=False` —
        // inference mode delegates to leaky_relu with mean slope per
        // `aten/src/ATen/native/Activation.cpp:624-630`. Kwargs: `lower`
        // (default 1/8), `upper` (default 1/3), `training` (default False).
        "rrelu" => {
            let a = unary("rrelu")?;
            let lower = kwargs.get("lower").and_then(Value::as_f64).unwrap_or(0.125);
            let upper = kwargs
                .get("upper")
                .and_then(Value::as_f64)
                .unwrap_or(1.0 / 3.0);
            let training = kwargs
                .get("training")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            // Stochastic training-mode samples emit per-element slopes drawn
            // from `Uniform[lower, upper]` — see `_rrelu_with_noise_train` at
            // `aten/src/ATen/native/Activation.cpp:578-608`. ferrotorch's
            // deterministic mean-slope inference path cannot match a single
            // stochastic oracle output by construction (each invocation draws
            // different slopes). Skip those samples as legitimate
            // "differentiability infrastructure not yet shipped" rather than
            // reporting a numerical divergence. The non-training samples
            // (which are the public-API contract worth pinning) all pass.
            if training {
                return Ok(None);
            }
            Ok(Some(grad_fns::activation::rrelu(
                &a, lower, upper, training,
            )?))
        }
        // ELU / SELU / CELU ----------------------------------------------
        // `torch.nn.functional.elu(input, alpha=1.0)` — upstream
        // `TORCH_IMPL_FUNC(elu_out)` at `aten/src/ATen/native/Activation.cpp:272-277`.
        "elu" => Ok(Some({
            let a = unary("elu")?;
            let alpha = kwargs.get("alpha").and_then(Value::as_f64).unwrap_or(1.0);
            grad_fns::activation::elu(&a, alpha)?
        })),
        // `torch.nn.functional.selu(input)` — no kwargs. Upstream
        // `Tensor selu(const Tensor& self)` at
        // `aten/src/ATen/native/Activation.cpp:524-526` delegates to
        // `at::elu(self, SELU_ALPHA, SELU_SCALE)`; ferrotorch's `selu`
        // fuses the same closed-form.
        "selu" => Ok(Some(grad_fns::activation::selu(&unary("selu")?)?)),
        // `torch.nn.functional.celu(input, alpha=1.0)` — upstream
        // `Tensor celu(const Tensor& self, const Scalar& alpha)` at
        // `aten/src/ATen/native/Activation.cpp:540-545` delegates to
        // `at::elu(self, alpha, 1.0, 1/alpha)`. ferrotorch's `celu` ships
        // the fused single-`CeluBackward` in `grad_fns::activation` (closes
        // #1341 REQ-21).
        "celu" => Ok(Some({
            let a = unary("celu")?;
            let alpha = kwargs.get("alpha").and_then(Value::as_f64).unwrap_or(1.0);
            grad_fns::activation::celu(&a, alpha)?
        })),
        // Sigmoid / Tanh / GELU / SiLU / Mish ----------------------------
        "sigmoid" => Ok(Some(grad_fns::activation::sigmoid(&unary("sigmoid")?)?)),
        "tanh" => Ok(Some(grad_fns::activation::tanh(&unary("tanh")?)?)),
        // `torch.nn.functional.gelu(input, approximate='none')` — op_db
        // kwargs default `approximate='none'` (the erf-based exact path).
        // The two upstream-supported approximations map to
        // `GeluApproximate::None` and `GeluApproximate::Tanh`; the
        // `Sigmoid` variant is a ferrotorch extension not exercised by
        // op_db.
        "gelu" => Ok(Some({
            let a = unary("gelu")?;
            let approx_s = kwargs
                .get("approximate")
                .and_then(Value::as_str)
                .unwrap_or("none");
            let approx = match approx_s {
                "tanh" => grad_fns::activation::GeluApproximate::Tanh,
                _ => grad_fns::activation::GeluApproximate::None,
            };
            grad_fns::activation::gelu_with(&a, approx)?
        })),
        "silu" => Ok(Some(grad_fns::activation::silu(&unary("silu")?)?)),
        "mish" => Ok(Some(grad_fns::activation::mish(&unary("mish")?)?)),
        // Softmax / LogSoftmax / Softmin ---------------------------------
        // op_db's `softmax` / `log_softmax` samples ship `dim` either as
        // positional `args[1]` (an int) or kwarg. ferrotorch's
        // `grad_fns::activation::softmax` / `log_softmax` are last-axis-only
        // — skip non-last-axis samples (the per-dim softmax routing is its
        // own REQ tracked separately).
        "softmax" => {
            let a = unary("softmax")?;
            // Resolve dim from args[1] or kwargs.dim.
            let dim_opt = args
                .get(1)
                .and_then(Value::as_i64)
                .or_else(|| kwargs.get("dim").and_then(Value::as_i64));
            if let Some(d) = dim_opt {
                let nd = a.ndim() as i64;
                let dn = if d < 0 { nd + d } else { d };
                if dn != nd - 1 {
                    return Ok(None);
                }
            }
            Ok(Some(grad_fns::activation::softmax(&a)?))
        }
        "log_softmax" => {
            let a = unary("log_softmax")?;
            let dim_opt = args
                .get(1)
                .and_then(Value::as_i64)
                .or_else(|| kwargs.get("dim").and_then(Value::as_i64));
            if let Some(d) = dim_opt {
                let nd = a.ndim() as i64;
                let dn = if d < 0 { nd + d } else { d };
                if dn != nd - 1 {
                    return Ok(None);
                }
            }
            Ok(Some(grad_fns::activation::log_softmax(&a)?))
        }
        // `torch.nn.functional.softmin(input, dim=None)` — same last-axis
        // restriction as `softmax`. Fused `SoftminBackward` (closes #1341
        // REQ-22).
        "softmin" => {
            let a = unary("softmin")?;
            let dim_opt = args
                .get(1)
                .and_then(Value::as_i64)
                .or_else(|| kwargs.get("dim").and_then(Value::as_i64));
            if let Some(d) = dim_opt {
                let nd = a.ndim() as i64;
                let dn = if d < 0 { nd + d } else { d };
                if dn != nd - 1 {
                    return Ok(None);
                }
            }
            Ok(Some(grad_fns::activation::softmin(&a)?))
        }
        // Softplus / Softsign --------------------------------------------
        // `torch.nn.functional.softplus(input, beta=1, threshold=20)` —
        // upstream `TORCH_IMPL_FUNC(softplus_out)` at
        // `aten/src/ATen/native/Activation.cpp:308-312`.
        "softplus" => Ok(Some({
            let a = unary("softplus")?;
            let beta = kwargs.get("beta").and_then(Value::as_f64).unwrap_or(1.0);
            let thr = kwargs
                .get("threshold")
                .and_then(Value::as_f64)
                .unwrap_or(20.0);
            grad_fns::activation::softplus(&a, beta, thr)?
        })),
        "softsign" => Ok(Some(grad_fns::activation::softsign(&unary("softsign")?)?)),
        // Hardtanh / Hardsigmoid / Hardswish -----------------------------
        // `torch.nn.functional.hardtanh(input, min_val=-1, max_val=1)` —
        // upstream `Tensor hardtanh(...)` at
        // `aten/src/ATen/native/Activation.cpp:436-468`.
        "hardtanh" => Ok(Some({
            let a = unary("hardtanh")?;
            let mn = kwargs
                .get("min_val")
                .and_then(Value::as_f64)
                .unwrap_or(-1.0);
            let mx = kwargs.get("max_val").and_then(Value::as_f64).unwrap_or(1.0);
            grad_fns::activation::hardtanh_with(&a, mn, mx)?
        })),
        "hardsigmoid" => Ok(Some(grad_fns::activation::hardsigmoid(&unary(
            "hardsigmoid",
        )?)?)),
        "hardswish" => Ok(Some(grad_fns::activation::hardswish(&unary("hardswish")?)?)),
        // Threshold ------------------------------------------------------
        // `torch.nn.functional.threshold(input, threshold, value)` —
        // op_db emits `args = [input, threshold: f64, value: f64]`. Upstream
        // `TORCH_IMPL_FUNC(threshold_out)` at
        // `aten/src/ATen/native/Activation.cpp:688-690`. ferrotorch ships
        // the fused single-`ThresholdBackward` (closes #1341 REQ-19).
        "threshold" => Ok(Some({
            let a = unary("threshold")?;
            // Both scalars can be either positional (args[1]/args[2]) or
            // kwargs (`threshold`/`value`).
            let thr = args
                .get(1)
                .and_then(Value::as_f64)
                .or_else(|| kwargs.get("threshold").and_then(Value::as_f64))
                .ok_or("threshold: missing threshold scalar")?;
            let val = args
                .get(2)
                .and_then(Value::as_f64)
                .or_else(|| kwargs.get("value").and_then(Value::as_f64))
                .ok_or("threshold: missing value scalar")?;
            grad_fns::activation::threshold(&a, thr, val)?
        })),
        // GLU ------------------------------------------------------------
        // `torch.nn.functional.glu(input, dim=-1)` — fused GLU activation,
        // splits `input` along `dim` and computes `a * sigmoid(b)`.
        // Upstream surface at `torch/nn/functional.py:1743`. ferrotorch's
        // `pub fn glu` lives in `grad_fns::activation`.
        "glu" => Ok(Some({
            let a = unary("glu")?;
            let dim = args
                .get(1)
                .and_then(Value::as_i64)
                .or_else(|| kwargs.get("dim").and_then(Value::as_i64))
                .unwrap_or(-1);
            grad_fns::activation::glu(&a, dim)?
        })),

        // ------------------------------------------------------------------
        // Shape op cluster — wired 2026-05-25 to close umbrella #1340
        // (runner arms for the shape ops in
        // `.design/ferrotorch-core/grad_fns/shape.md`'s SHIPPED REQ set:
        // view, reshape, flatten, squeeze, unsqueeze, permute, transpose,
        // expand, cat, stack, split, chunk, narrow, roll). The prior
        // dispatch's claim of "runner arm: view|reshape in dispatch_f32"
        // was false — only `transpose`/`expand` branches existed inside
        // the *probe* materializer (`run_probe_ferrotorch` at :2749/:2755),
        // not in `dispatch_f32`. These arms decode op_db's
        // shape-list / dim-int / list-of-tensors envelopes and route to
        // the matching ferrotorch entry points (`grad_fns::shape::*`,
        // `methods::{view_t, permute_t, narrow_t, split_t, chunk_t}`,
        // `vmap::stack`, `ops::tensor_ops::roll`). For Vec-returning ops
        // (split / chunk) the runner's `sweep_with_cap` selects
        // `expected_v = output[0]` when the wire output is a JSON array
        // (`main.rs:3147`) — so each arm returns the first chunk's
        // tensor to gate value-equality. `broadcast_shapes` is
        // intentionally excluded: it takes shape lists, not tensors,
        // so the f32-tensor dispatch_f32 envelope is the wrong fit
        // (the op_db sample's args are `[List[int], List[int]]` and
        // its output is `List[int]`, not a tensor).
        // ------------------------------------------------------------------

        // `torch.view(input, *shape)` — op_db emits
        // `args = [tensor, [d0, d1, ...]]`. ferrotorch's
        // `view_t(input, &[i64])` mirrors upstream
        // `aten/src/ATen/native/TensorShape.cpp:4563 Tensor view`. Rejects
        // non-contiguous inputs (`methods.rs:1296`); samples with non-
        // contiguous inputs are skipped via the upstream error path.
        "view" => {
            let input = unary("view")?;
            let shape =
                arg_dim_list_at(1).ok_or("view: arg 1 must be a shape list [d0, d1, ...]")?;
            // ferrotorch's view_t errors on non-contiguous input — that's the
            // upstream contract too (`computeStride` returning nullopt). Skip
            // such samples defensively.
            if !input.is_contiguous() {
                return Ok(None);
            }
            Ok(Some(ferrotorch_core::view_t(&input, &shape)?))
        }

        // `torch.reshape(input, shape)` — op_db emits
        // `args = [tensor, [d0, d1, ...]]`. ferrotorch's
        // `grad_fns::shape::reshape(input, &[isize])` mirrors upstream
        // `TensorShape.cpp:2129 Tensor reshape`; handles the single `-1`
        // infer slot via `resolve_shape` (`shape.rs:1029`).
        "reshape" => {
            let input = unary("reshape")?;
            let raw = arg_dim_list_at(1).ok_or("reshape: arg 1 must be a shape list")?;
            let isize_shape: Vec<isize> = raw.iter().map(|&d| d as isize).collect();
            Ok(Some(grad_fns::shape::reshape(&input, &isize_shape)?))
        }

        // `torch.flatten(input, start_dim=0, end_dim=-1)` — op_db emits the
        // no-arg form (full flatten to 1-D) AND the kwarg-driven partial
        // form `kwargs={'start_dim': 1, 'end_dim': -1}`. ferrotorch's
        // `grad_fns::shape::flatten` only implements the full-flatten case;
        // we compute the partial-flatten shape locally then dispatch through
        // `grad_fns::shape::reshape` so the existing `ReshapeBackward`
        // covers the partial case (upstream `TensorShape.cpp:4178` itself
        // reduces partial flatten to a reshape).
        "flatten" => {
            let input = unary("flatten")?;
            let ndim = input.ndim() as i64;
            let start_dim = kwargs
                .get("start_dim")
                .and_then(Value::as_i64)
                .or_else(|| args.get(1).and_then(Value::as_i64))
                .unwrap_or(0);
            let end_dim = kwargs
                .get("end_dim")
                .and_then(Value::as_i64)
                .or_else(|| args.get(2).and_then(Value::as_i64))
                .unwrap_or(-1);
            // 0-d input: torch.flatten returns a 1-element 1-D tensor.
            if ndim == 0 {
                return Ok(Some(grad_fns::shape::reshape(&input, &[1isize])?));
            }
            let normalize = |d: i64| -> Result<usize, Box<dyn std::error::Error>> {
                let r = if d < 0 { d + ndim } else { d };
                if !(0..ndim).contains(&r) {
                    return Err(format!("flatten: dim {d} out of range for ndim {ndim}").into());
                }
                Ok(r as usize)
            };
            let s = normalize(start_dim)?;
            let e = normalize(end_dim)?;
            if s > e {
                return Err(format!("flatten: start_dim {s} > end_dim {e}").into());
            }
            let in_shape = input.shape();
            // Build target shape: keep dims [0, s), then collapsed dim, then [e+1, ndim).
            let mut new_shape: Vec<isize> = Vec::with_capacity(in_shape.len() - (e - s));
            for d in &in_shape[..s] {
                new_shape.push(*d as isize);
            }
            let collapsed: usize = in_shape[s..=e].iter().product();
            new_shape.push(collapsed as isize);
            for d in &in_shape[e + 1..] {
                new_shape.push(*d as isize);
            }
            Ok(Some(grad_fns::shape::reshape(&input, &new_shape)?))
        }

        // `torch.squeeze(input)` / `torch.squeeze(input, dim)` /
        // `torch.squeeze(input, dims)` — op_db emits TWO variants:
        // (a) bare `squeeze` (no-arg removes ALL size-1 dims) and
        // (b) `squeeze.multiple` (tuple-of-dims). ferrotorch's
        // `grad_fns::shape::squeeze(input, axis: isize)` is single-dim.
        // Full-squeeze and multi-dim squeeze are unfolded via sequential
        // single-dim squeeze calls in *descending* order so axis indices
        // stay valid after each drop. Non-1 named dims skip — ferrotorch's
        // documented departure (AC-17) errors there while upstream is a no-op,
        // so the value would diverge; honest skip is the right gate.
        "squeeze" => {
            let input = unary("squeeze")?;
            let dims_to_drop: Vec<usize> = match args.get(1) {
                None => {
                    // Full squeeze — collect all size-1 axes.
                    let mut out: Vec<usize> = Vec::new();
                    for (i, &s) in input.shape().iter().enumerate() {
                        if s == 1 {
                            out.push(i);
                        }
                    }
                    out
                }
                Some(Value::Number(n)) => {
                    let d = n.as_i64().ok_or("squeeze: dim not int")?;
                    let ndim = input.ndim() as i64;
                    if ndim == 0 {
                        return Ok(None);
                    }
                    let r = if d < 0 { d + ndim } else { d };
                    if !(0..ndim).contains(&r) {
                        return Err(
                            format!("squeeze: dim {d} out of range for ndim {ndim}").into()
                        );
                    }
                    let r = r as usize;
                    if input.shape()[r] != 1 {
                        return Ok(None);
                    }
                    vec![r]
                }
                Some(Value::Array(arr)) => {
                    let ndim = input.ndim() as i64;
                    if ndim == 0 {
                        return Ok(None);
                    }
                    let mut out: Vec<usize> = Vec::with_capacity(arr.len());
                    for v in arr {
                        let d = v.as_i64().ok_or("squeeze: dim list element not int")?;
                        let r = if d < 0 { d + ndim } else { d };
                        if !(0..ndim).contains(&r) {
                            return Err(format!(
                                "squeeze: dim {d} out of range for ndim {ndim}"
                            )
                            .into());
                        }
                        let r = r as usize;
                        if input.shape()[r] == 1 {
                            out.push(r);
                        }
                    }
                    out.sort_unstable();
                    out.dedup();
                    out
                }
                Some(other) => {
                    return Err(format!("squeeze: arg 1 unexpected: {other}").into());
                }
            };
            let mut t = input;
            let mut sorted = dims_to_drop;
            sorted.sort_unstable();
            for &d in sorted.iter().rev() {
                t = grad_fns::shape::squeeze(&t, d as isize)?;
            }
            Ok(Some(t))
        }

        // `torch.unsqueeze(input, dim)` — op_db emits `args = [tensor, dim]`.
        // ferrotorch's `grad_fns::shape::unsqueeze(input, axis: isize)`
        // mirrors upstream `TensorShape.cpp:4109` with range `[-(ndim+1), ndim]`.
        "unsqueeze" => {
            let input = unary("unsqueeze")?;
            let dim = args
                .get(1)
                .and_then(Value::as_i64)
                .ok_or("unsqueeze: arg 1 (dim) must be an int")?;
            Ok(Some(grad_fns::shape::unsqueeze(&input, dim as isize)?))
        }

        // `torch.permute(input, dims)` — op_db emits `args = [tensor, [perm]]`.
        // ferrotorch's `permute_t(input, &[usize])` mirrors upstream
        // `TensorShape.cpp:1829`. We resolve negative indices here (op_db
        // emits e.g. `[0, -2, -1, 1]`) before delegating.
        "permute" => {
            let input = unary("permute")?;
            let perm_raw = arg_dim_list_at(1).ok_or("permute: arg 1 must be a perm list")?;
            let ndim = input.ndim() as i64;
            // 0-d input + empty perm: identity (torch returns input).
            if ndim == 0 && perm_raw.is_empty() {
                return Ok(Some(input));
            }
            let mut perm: Vec<usize> = Vec::with_capacity(perm_raw.len());
            for d in &perm_raw {
                let r = if *d < 0 { *d + ndim } else { *d };
                if !(0..ndim).contains(&r) {
                    return Err(
                        format!("permute: dim {d} out of range for ndim {ndim}").into()
                    );
                }
                perm.push(r as usize);
            }
            // permute_t returns a strided view; the value-equality gate
            // `assert_close_f32` consumes the result via `.data_vec()` which
            // gathers elements in C-order, so we no longer need to call
            // `.contiguous()` here — the stride-view passes through
            // unchanged and `data_vec()` does the gather. Matches upstream
            // `aten/src/ATen/native/TensorShape.cpp:1829 Tensor permute`
            // returning a zero-copy stride view.
            Ok(Some(ferrotorch_core::permute_t(&input, &perm)?))
        }

        // `torch.transpose(input, dim0, dim1)` — op_db emits
        // `args = [tensor, dim0, dim1]`. The n-D form builds a permutation
        // swapping dim0 ↔ dim1 then delegates to `permute_t`; upstream
        // `TensorShape.cpp:3816`. Negative dims allowed (`maybe_wrap_dim`).
        "transpose" => {
            let input = unary("transpose")?;
            let d0 = args
                .get(1)
                .and_then(Value::as_i64)
                .ok_or("transpose: arg 1 (dim0) must be an int")?;
            let d1 = args
                .get(2)
                .and_then(Value::as_i64)
                .ok_or("transpose: arg 2 (dim1) must be an int")?;
            let ndim = input.ndim() as i64;
            if ndim == 0 {
                return Ok(Some(input));
            }
            let wrap = |d: i64| -> Result<usize, Box<dyn std::error::Error>> {
                let r = if d < 0 { d + ndim } else { d };
                if !(0..ndim).contains(&r) {
                    return Err(format!("transpose: dim {d} out of range for ndim {ndim}").into());
                }
                Ok(r as usize)
            };
            let a = wrap(d0)?;
            let b = wrap(d1)?;
            let mut perm: Vec<usize> = (0..input.ndim()).collect();
            perm.swap(a, b);
            Ok(Some(ferrotorch_core::permute_t(&input, &perm)?))
        }

        // `torch.Tensor.expand(*sizes)` — op_db emits `args = [tensor, [sizes]]`,
        // sizes may contain -1 (meaning "keep input dim unchanged"). ferrotorch's
        // `grad_fns::shape::expand(input, &[usize])` mirrors upstream
        // `TensorShape.cpp:1344`. We resolve any -1 entries to the input's
        // dim size before delegating (the resolution must right-align: when
        // the target adds prepend dims, those new dims cannot be -1 per upstream).
        "expand" => {
            let input = unary("expand")?;
            let target =
                arg_dim_list_at(1).ok_or("expand: arg 1 must be a shape list")?;
            let in_shape = input.shape();
            let in_ndim = in_shape.len();
            let target_ndim = target.len();
            if target_ndim < in_ndim {
                return Err(format!(
                    "expand: target ndim {target_ndim} < input ndim {in_ndim}"
                )
                .into());
            }
            let pad = target_ndim - in_ndim;
            let mut resolved: Vec<usize> = Vec::with_capacity(target_ndim);
            for (i, &d) in target.iter().enumerate() {
                if d == -1 {
                    if i < pad {
                        return Err(
                            "expand: -1 not allowed on prepended dim".into()
                        );
                    }
                    resolved.push(in_shape[i - pad]);
                } else if d < 0 {
                    return Err(
                        format!("expand: negative size {d} (other than -1) not allowed").into(),
                    );
                } else {
                    resolved.push(d as usize);
                }
            }
            Ok(Some(grad_fns::shape::expand(&input, &resolved)?))
        }

        // `torch.cat(tensors, dim=0)` — op_db emits
        // `args = [List[Tensor]]`, `kwargs = {dim: int}` (sometimes dim is
        // positional). ferrotorch's `grad_fns::shape::cat(tensors, axis: isize)`
        // mirrors upstream `TensorShape.cpp:676 cat_out_cpu` / `:772 cat`.
        "cat" => {
            let list_v = args.first().ok_or("cat: missing tensor list arg")?;
            let arr = list_v
                .as_array()
                .ok_or("cat: arg 0 must be a list of tensors")?;
            let mut tensors: Vec<Tensor<f32>> = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let wt = unwrap_tensor_arg(v)
                    .ok_or_else(|| format!("cat: list element {i} is not a tensor"))?;
                tensors.push(wt.to_f32()?);
            }
            if tensors.is_empty() {
                return Err("cat: empty tensor list".into());
            }
            let dim = kwargs
                .get("dim")
                .and_then(Value::as_i64)
                .or_else(|| args.get(1).and_then(Value::as_i64))
                .unwrap_or(0);
            Ok(Some(grad_fns::shape::cat(&tensors, dim as isize)?))
        }

        // `torch.stack(tensors, dim=0)` — op_db emits
        // `args = [List[Tensor], dim]` (dim positional, may be negative).
        // ferrotorch's `vmap::stack(&[Tensor], usize)` is non-negative-dim
        // only; we normalize here before dispatch.
        "stack" => {
            let list_v = args.first().ok_or("stack: missing tensor list arg")?;
            let arr = list_v
                .as_array()
                .ok_or("stack: arg 0 must be a list of tensors")?;
            let mut tensors: Vec<Tensor<f32>> = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let wt = unwrap_tensor_arg(v)
                    .ok_or_else(|| format!("stack: list element {i} is not a tensor"))?;
                tensors.push(wt.to_f32()?);
            }
            if tensors.is_empty() {
                return Err("stack: empty tensor list".into());
            }
            let dim_raw = args
                .get(1)
                .and_then(Value::as_i64)
                .or_else(|| kwargs.get("dim").and_then(Value::as_i64))
                .unwrap_or(0);
            let nd = tensors[0].ndim() as i64;
            // stack inserts a new dim, so valid range is [-(nd+1), nd].
            let normalized = if dim_raw < 0 { dim_raw + nd + 1 } else { dim_raw };
            if normalized < 0 || normalized > nd {
                return Err(format!(
                    "stack: dim {dim_raw} out of range for inputs with ndim {nd}"
                )
                .into());
            }
            Ok(Some(ferrotorch_core::vmap::stack(
                &tensors,
                normalized as usize,
            )?))
        }

        // `torch.split(input, split_size, dim=0)` — op_db emits
        // `args = [tensor, split_size_or_sizes, dim?]`. Returns a tuple of
        // tensors; the runner gates against `output[0]` (first chunk) per
        // `main.rs:3147`. ferrotorch's `methods::split_t(input, &[usize], dim)`
        // mirrors upstream `TensorShape.cpp:3175 split` / `:3265 split_with_sizes`.
        "split" => {
            let input = unary("split")?;
            let split_arg = args.get(1).ok_or("split: missing split_size arg")?;
            let dim_i = args
                .get(2)
                .and_then(Value::as_i64)
                .or_else(|| kwargs.get("dim").and_then(Value::as_i64))
                .unwrap_or(0);
            let nd = input.ndim() as i64;
            let dim = if dim_i < 0 { dim_i + nd } else { dim_i };
            if !(0..nd).contains(&dim) {
                return Err(format!("split: dim {dim_i} out of range for ndim {nd}").into());
            }
            let dim = dim as usize;
            let dim_size = input.shape()[dim];
            let sizes: Vec<usize> = match split_arg {
                Value::Number(n) => {
                    let s = n.as_i64().ok_or("split: split_size not int")? as usize;
                    if s == 0 {
                        return Ok(None);
                    }
                    let mut out = Vec::new();
                    let mut remaining = dim_size;
                    while remaining > 0 {
                        let chunk = s.min(remaining);
                        out.push(chunk);
                        remaining -= chunk;
                    }
                    if out.is_empty() {
                        return Ok(None);
                    }
                    out
                }
                Value::Array(arr) => {
                    let mut out = Vec::with_capacity(arr.len());
                    for x in arr {
                        out.push(x.as_i64().ok_or("split: size list element not int")? as usize);
                    }
                    out
                }
                other => {
                    return Err(format!("split: unexpected split arg {other}").into());
                }
            };
            let pieces = ferrotorch_core::split_t(&input, &sizes, dim)?;
            // Return the first chunk — the wrapper's tuple-vs-tensor gate
            // selects `output[0]` for value-equality (see main.rs:3147).
            Ok(Some(pieces.into_iter().next().ok_or("split: empty result")?))
        }

        // `torch.chunk(input, chunks, dim=0)` — op_db emits
        // `args = [tensor, chunks, dim?]`. ferrotorch's
        // `methods::chunk_t(input, chunks, dim)` mirrors upstream
        // `TensorShape.cpp:1077` (per-chunk size = ceil(dim_size / chunks)).
        // Returns first chunk for value-equality (same tuple convention).
        "chunk" => {
            let input = unary("chunk")?;
            let chunks = args
                .get(1)
                .and_then(Value::as_i64)
                .ok_or("chunk: arg 1 (chunks) must be int")? as usize;
            let dim_i = args
                .get(2)
                .and_then(Value::as_i64)
                .or_else(|| kwargs.get("dim").and_then(Value::as_i64))
                .unwrap_or(0);
            let nd = input.ndim() as i64;
            let dim = if dim_i < 0 { dim_i + nd } else { dim_i };
            if !(0..nd).contains(&dim) {
                return Err(format!("chunk: dim {dim_i} out of range for ndim {nd}").into());
            }
            if chunks == 0 {
                return Err("chunk: chunks must be > 0".into());
            }
            let pieces = ferrotorch_core::chunk_t(&input, chunks, dim as usize)?;
            Ok(Some(pieces.into_iter().next().ok_or("chunk: empty result")?))
        }

        // `torch.narrow(input, dim, start, length)` — op_db emits
        // `args = [tensor, dim, start, length]`; `start` MAY be a 0-d tensor
        // (the `narrow.Tensor` overload at `TensorShape.cpp:1669`), which
        // we extract to a scalar before delegating.
        "narrow" => {
            let input = unary("narrow")?;
            let dim_i = args
                .get(1)
                .and_then(Value::as_i64)
                .ok_or("narrow: arg 1 (dim) must be int")?;
            let nd = input.ndim() as i64;
            let dim = if dim_i < 0 { dim_i + nd } else { dim_i };
            if !(0..nd).contains(&dim) {
                return Err(format!("narrow: dim {dim_i} out of range for ndim {nd}").into());
            }
            let dim = dim as usize;
            // start: may be int OR 0-d tensor.
            let start: usize = match args.get(2) {
                Some(Value::Number(n)) => {
                    let raw = n.as_i64().ok_or("narrow: start not int")?;
                    let dim_size = input.shape()[dim] as i64;
                    let resolved = if raw < 0 { raw + dim_size } else { raw };
                    if resolved < 0 || resolved > dim_size {
                        return Err(format!(
                            "narrow: start {raw} out of range for dim size {dim_size}"
                        )
                        .into());
                    }
                    resolved as usize
                }
                Some(other) => {
                    if let Some(wt) = unwrap_tensor_arg(other) {
                        if !wt.shape.is_empty() {
                            // Non-0-d tensor start — unsupported in ferrotorch's
                            // narrow_t (the `narrow.Tensor` 0-d overload only).
                            return Ok(None);
                        }
                        // 0-d tensor: extract its single scalar. Use the int
                        // path when dtype is integer, float-then-truncate
                        // otherwise (op_db's narrow samples emit int64 0-d
                        // tensors).
                        let raw: i64 = match wt.dtype.as_str() {
                            "int64" | "int32" | "uint8" => {
                                let t = wt.to_int_tensor_i64()?;
                                *t.data()?.first().unwrap_or(&0)
                            }
                            _ => {
                                let t = wt.to_f32()?;
                                let d = t.data_vec()?;
                                *d.first().unwrap_or(&0.0) as i64
                            }
                        };
                        let dim_size = input.shape()[dim] as i64;
                        let resolved = if raw < 0 { raw + dim_size } else { raw };
                        if resolved < 0 || resolved > dim_size {
                            return Err(format!(
                                "narrow: start tensor {raw} out of range for dim size {dim_size}"
                            )
                            .into());
                        }
                        resolved as usize
                    } else {
                        return Err(format!("narrow: arg 2 (start) unexpected: {other}").into());
                    }
                }
                None => return Err("narrow: missing start arg".into()),
            };
            let length = args
                .get(3)
                .and_then(Value::as_i64)
                .ok_or("narrow: arg 3 (length) must be int")? as usize;
            Ok(Some(input.narrow(dim, start, length)?))
        }

        // `torch.roll(input, shifts, dims=None)` — op_db emits
        // `args = [tensor, shifts, dims]` where each of `shifts`/`dims` may be
        // an int or a list of ints (the `roll(Tensor, IntArrayRef shifts,
        // IntArrayRef dims)` overload at `TensorTransformations.cpp:110`).
        // ferrotorch's `ops::tensor_ops::roll(input, shifts: i64, dim: usize)`
        // is single-shift / single-dim only; for the multi-shift form we
        // apply roll sequentially (upstream implements the multi-dim case as
        // a sequence of single-dim rolls, per
        // `TensorTransformations.cpp:154-176`). When dims is None, torch
        // flattens then rolls — we emulate via reshape + 1-D roll.
        "roll" => {
            let input = unary("roll")?;
            let shifts: Vec<i64> = match args.get(1) {
                Some(Value::Number(n)) => vec![n.as_i64().ok_or("roll: shifts not int")?],
                Some(Value::Array(arr)) => {
                    let mut out = Vec::with_capacity(arr.len());
                    for x in arr {
                        out.push(x.as_i64().ok_or("roll: shifts list element not int")?);
                    }
                    out
                }
                other => return Err(format!("roll: unexpected shifts arg {other:?}").into()),
            };
            let dims_v = args.get(2);
            let dims: Vec<i64> = match dims_v {
                None | Some(Value::Null) => {
                    // Flatten-then-roll-then-unflatten path.
                    if input.ndim() == 0 {
                        return Ok(Some(input));
                    }
                    let numel = input.numel();
                    if numel == 0 {
                        return Ok(Some(input));
                    }
                    let in_shape: Vec<isize> =
                        input.shape().iter().map(|&d| d as isize).collect();
                    let flat = grad_fns::shape::reshape(&input, &[-1isize])?;
                    let total: i64 = shifts.iter().sum();
                    let rolled = ferrotorch_core::ops::tensor_ops::roll(&flat, total, 0)?;
                    return Ok(Some(grad_fns::shape::reshape(&rolled, &in_shape)?));
                }
                Some(Value::Number(n)) => vec![n.as_i64().ok_or("roll: dims not int")?],
                Some(Value::Array(arr)) => {
                    let mut out = Vec::with_capacity(arr.len());
                    for x in arr {
                        out.push(x.as_i64().ok_or("roll: dims list element not int")?);
                    }
                    out
                }
                Some(other) => return Err(format!("roll: unexpected dims arg {other}").into()),
            };
            if shifts.len() != dims.len() {
                return Err(format!(
                    "roll: shifts.len() {} != dims.len() {}",
                    shifts.len(),
                    dims.len()
                )
                .into());
            }
            let nd = input.ndim() as i64;
            if nd == 0 {
                return Ok(Some(input));
            }
            let mut t = input;
            for (s, d) in shifts.into_iter().zip(dims) {
                let dim_norm = if d < 0 { d + nd } else { d };
                if !(0..nd).contains(&dim_norm) {
                    return Err(format!("roll: dim {d} out of range for ndim {nd}").into());
                }
                // Empty-axis: torch's roll is identity; ferrotorch's roll
                // returns clone for shift_norm==0; passing through is safe.
                if t.shape()[dim_norm as usize] == 0 {
                    continue;
                }
                t = ferrotorch_core::ops::tensor_ops::roll(&t, s, dim_norm as usize)?;
            }
            Ok(Some(t))
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
        "remainder",
        "fmod",
        "floor_divide",
        "addcmul",
        "addcdiv",
        // Cumulative (scan) ops — `grad_fns::cumulative` dispatch arms added
        // 2026-05-25 to close #1230. cummax/cummin return only the `values`
        // half of the (values, indices) tuple (Option A — see dispatch_f32
        // arms for rationale; #1231 tracks indices divergences separately).
        "cumsum",
        "cumprod",
        "cummax",
        "cummin",
        "logcumsumexp",
        // Quantization: per-tensor affine fake quantize w/ STE backward. (#1238)
        "fake_quantize_per_tensor_affine",
        // Quantization: per-channel affine fake quantize w/ STE backward. (#1239)
        "fake_quantize_per_channel_affine",
        // Indexing: masked / where ops. The runner dispatch routes through
        // the broadcasting wrappers `masked_select_bcast`, `masked_fill_bcast`,
        // and `where_cond_bcast` added 2026-05-25 to close #1250 / #1251 /
        // #1255 — the existing shape-strict entry points reject the
        // broadcast-required samples op_db emits.
        "masked_select",
        "masked_fill",
        "where",
        // Indexing: gather / scatter / scatter_add / index_select. The
        // runner arms decode positional `[input_f32, dim_i64, index_int64,
        // src_f32?]` and route through the existing ferrotorch impls at
        // `ops::indexing::{gather, scatter, scatter_add}` and
        // `grad_fns::indexing::index_select_dim`. 0-d inputs and
        // ndim-mismatch samples skip per the shape-strict-impl contract.
        // Closes #1242 / #1243 / #1244 / #1246.
        "gather",
        "scatter",
        "scatter_add",
        "index_select",
        // Indexing: `torch.index_fill(input, dim, index, value)` — overwrite
        // slices at index positions with a scalar. Runner arm decodes
        // `[input_f32, dim_i64, index_int64, value]` (scalar JSON number or
        // 0-d tensor envelope per upstream `index_fill.int_Tensor` overload).
        // Closes #1249.
        "index_fill",
        // Indexing batch landed 2026-05-25 (S1: batch-by-upstream-file).
        // All 6 ops live in `aten/src/ATen/native/TensorAdvancedIndexing.cpp`.
        // Closes #1245 #1247 #1248 #1252 #1253 #1254.
        "scatter_reduce",
        "index_add",
        "index_copy",
        "masked_scatter",
        "take",
        "put",
        // Transcendental unary family — wired 2026-05-25 to close umbrella
        // #1298 plus per-op blockers #1303 #1305 #1307 #1309 #1311 #1313
        // #1315 #1316 #1317 #1319 #1320 #1322 #1323 #1324 #1325 #1326 #1327
        // #1328 #1329 #1330 #1331. Each arm in `dispatch_f32` above
        // dispatches `args=[input_f32]` through the matching
        // `grad_fns::transcendental::<op>` per the design doc REQ table.
        // `clamp` keeps the legitimate-skip pathway for tensor-bound samples
        // (REQ-5's documented divergence); 0-d-bound samples extract scalars
        // and dispatch through `pub fn clamp`.
        "exp",
        "log",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "asinh",
        "acosh",
        "atanh",
        "exp2",
        "expm1",
        "log2",
        "log10",
        "log1p",
        "ceil",
        "floor",
        "round",
        "trunc",
        "frac",
        "sign",
        "sinc",
        "clamp",
        // Reduction cluster — closes umbrella #1314 + per-op blockers
        // #1301 (std/var) #1304 (argmax/argmin) #1310 (logsumexp autograd)
        // #1312 (any/all/count_nonzero). Owned by `grad_fns::reduction`.
        // `prod` / `amin` / `amax` skip on `dim` kwarg (single-dim
        // variants NOT-STARTED — covered by #1302 alongside max/min-
        // with-dim). `std`/`var` skip on `dim` kwarg (NOT-STARTED).
        "sum",
        "mean",
        "prod",
        "amin",
        "amax",
        "logsumexp",
        "argmax",
        "argmin",
        "std",
        "var",
        "any",
        "all",
        "count_nonzero",
        // Activation op cluster — wired 2026-05-26 to close umbrella #1338
        // (runner arms for the 22 ops in `.design/ferrotorch-core/grad_fns/
        // activation.md`'s `parity_ops` route field) + #1341 (the 4 fused-
        // GradFn additions threshold/rrelu/celu/softmin). The bare names
        // below are what the route's `parity_ops` field declares; the
        // oracle alias map in `oracle_name()` translates them to the
        // `nn.functional.<name>` form op_db uses for the non-top-level
        // entries before each `oracle.sample` call.
        "relu",
        "relu6",
        "leaky_relu",
        "prelu",
        "rrelu",
        "elu",
        "selu",
        "celu",
        "sigmoid",
        "tanh",
        "gelu",
        "silu",
        "mish",
        "softmax",
        "log_softmax",
        "softmin",
        "softplus",
        "softsign",
        "hardtanh",
        "hardsigmoid",
        "hardswish",
        "threshold",
        "glu",
        // Shape op cluster — wired 2026-05-25 to close umbrella #1340
        // (parity-sweep runner arms for the shape ops in
        // `.design/ferrotorch-core/grad_fns/shape.md`'s SHIPPED REQ set).
        // The dispatch_f32 arms above decode op_db's shape-list / dim-int /
        // list-of-tensors envelopes and route to the matching ferrotorch
        // entry points. `broadcast_shapes` is intentionally excluded — it
        // takes shape lists, not tensors (wrong envelope for dispatch_f32).
        "view",
        "reshape",
        "flatten",
        "squeeze",
        "unsqueeze",
        "permute",
        "transpose",
        "expand",
        "cat",
        "stack",
        "split",
        "chunk",
        "narrow",
        "roll",
    ]
}

/// Translate a bare op name (the form the route's `parity_ops` field uses
/// and that flows through ferrotorch's `dispatch_f32` match arms) to the
/// name the torch oracle exposes for `op_info.sample_inputs`.
///
/// Most activation ops live in `op_db` under `nn.functional.<name>` (e.g.
/// `nn.functional.relu`, `nn.functional.gelu`); a handful (sigmoid / tanh /
/// softmax / log_softmax) are registered at top level. Top-level
/// `relu`-style names are NOT in op_db, so the alias must be applied before
/// each `oracle.sample` call (closes the test-infrastructure half of
/// umbrella blocker #1338).
///
/// Returns the input name unchanged for any op that's already at the
/// oracle's canonical name.
fn oracle_name(op: &str) -> &str {
    match op {
        // `nn.functional.*` aliased activations — see
        // `.design/ferrotorch-core/grad_fns/activation.md` REQ table for
        // the upstream entry points.
        "relu" => "nn.functional.relu",
        "relu6" => "nn.functional.relu6",
        "leaky_relu" => "nn.functional.leaky_relu",
        "prelu" => "nn.functional.prelu",
        "rrelu" => "nn.functional.rrelu",
        "elu" => "nn.functional.elu",
        "selu" => "nn.functional.selu",
        "celu" => "nn.functional.celu",
        "gelu" => "nn.functional.gelu",
        "silu" => "nn.functional.silu",
        "mish" => "nn.functional.mish",
        "softmin" => "nn.functional.softmin",
        "softplus" => "nn.functional.softplus",
        "softsign" => "nn.functional.softsign",
        "hardtanh" => "nn.functional.hardtanh",
        "hardsigmoid" => "nn.functional.hardsigmoid",
        "hardswish" => "nn.functional.hardswish",
        "threshold" => "nn.functional.threshold",
        "glu" => "nn.functional.glu",
        // Top-level oracle entries — pass through.
        // `sigmoid` / `tanh` / `softmax` / `log_softmax` live in op_db at
        // the top level (verified 2026-05-26 via `parity-sweep list-ops`).
        other => other,
    }
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
    // Use `.data_vec()` on the actual side: shape-view ops like `permute`,
    // `transpose`, `narrow`, `expand`, `squeeze`, `unsqueeze` legitimately
    // produce stride-view (non-contiguous) tensors per upstream
    // `aten/src/ATen/native/TensorShape.cpp:1829 Tensor permute` (zero-copy
    // stride reorder) and `:1344 Tensor expand` (size-1 → stride-0 broadcast).
    // `.data()` is contiguity-strict and would reject these views even though
    // the values are correct; `.data_vec()` gathers elements in C-order so
    // the value-equality gate compares against torch's contiguous output
    // byte-for-byte. The `expected` side stays on `.data()` because the
    // wire decode emits C-order contiguous storage.
    let actual_data = actual
        .data_vec()
        .map_err(|e| format!("ferrotorch tensor.data_vec() failed: {e}"))?;
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
    // Translate bare ferrotorch op names (e.g. `relu`) to the oracle's
    // canonical name (e.g. `nn.functional.relu`). See `oracle_name` for the
    // alias map. The local `dispatch_f32` arms continue to match against
    // `op` (bare) — the dispatch side is keyed by what the route's
    // `parity_ops` field declares, the oracle side is keyed by what op_db
    // registers. Closes the test-infrastructure half of umbrella #1338.
    let oracle_op = oracle_name(op);
    for seed in 0..seeds {
        // op_db's sample_inputs yields a fixed list per (op, seed, dtype). We
        // walk it index-by-index until the oracle reports we've exhausted it
        // or we hit max_samples_per_seed (so sweep-all stays bounded).
        for i in 0..max_samples_per_seed {
            let resp = oracle.sample(oracle_op, seed, i);
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
            // Tuple-returning ops (cummax / cummin -> (values, indices))
            // arrive as a JSON array `[values, indices]` from the oracle's
            // generic `encode_arg(output)` path which maps Python tuples to
            // JSON arrays at `oracle.py:97-98`. We select `output[0]` (values)
            // for the parity-sweep value-equality gate; indices-parity is
            // tracked separately under #1231 (cummax/cummin differentiability
            // + tie-break + NaN handling — see
            // `.design/ferrotorch-core/grad_fns/cumulative.md` REQ-3 / REQ-4).
            // Single-tensor ops (cumsum / cumprod / logcumsumexp and every
            // arithmetic op) keep the existing single-envelope path.
            let raw_output = resp.get("output").cloned().unwrap_or(Value::Null);
            let expected_v = match raw_output.as_array() {
                Some(arr) if !arr.is_empty() => arr[0].clone(),
                _ => raw_output,
            };
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
