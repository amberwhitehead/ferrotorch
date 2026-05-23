//! Autograd backward parity dump binary for the ferrotorch-core
//! real-artifact parity harness (Phase G.5, #1171).
//!
//! Companion to `scripts/verify_autograd_inference.py` and
//! `scripts/pin_pretrained_autograd_fixtures.py`. The pin script
//! emits, for each canonical op + config in `ferrotorch/autograd-parity-v1`,
//! a directory layout
//!
//! ```text
//! <fixture-dir>/<op>/<config>/
//!     params.json
//!     forward_out.bin
//!     inputs/<name>.bin
//!     grads/<name>.bin       (only when that input had requires_grad)
//! ```
//!
//! This example takes a single `--op` + `--config` selection, replays
//! the forward through `ferrotorch_core`'s differentiable surface,
//! sums the output to a scalar, calls `.backward()`, then dumps:
//!
//! ```text
//! <output-dir>/forward_out.bin
//! <output-dir>/grads/<name>.bin    (one per requires_grad input)
//! ```
//!
//! Binary format (matches the python pin script's `dump_f32`):
//!
//! ```text
//! [u32 ndim][u32 × ndim shape][f32 le data]
//! ```
//!
//! Usage:
//! ```text
//! cargo run -p ferrotorch-core --release --example autograd_dump -- \
//!     --op matmul_2d --config 8x16_16x4 \
//!     --fixture-dir /tmp/.../fixtures \
//!     --output-dir /tmp/dump
//! ```

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferrotorch_core::grad_fns::activation::{
    gelu, log_softmax, relu, sigmoid, silu, softmax, tanh,
};
use ferrotorch_core::grad_fns::arithmetic::{add, div, mul, pow, sub};
use ferrotorch_core::grad_fns::indexing::index_select_dim;
use ferrotorch_core::grad_fns::linalg::{bmm_differentiable, linear_fused, matmul_differentiable};
use ferrotorch_core::grad_fns::reduction::{mean_dim, sum_dim};
use ferrotorch_core::grad_fns::shape::{cat, reshape};
use ferrotorch_core::grad_fns::transcendental::{exp, log};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{IntTensor, Tensor};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    op: String,
    config: String,
    fixture_dir: PathBuf,
    output_dir: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut op: Option<String> = None;
    let mut config: Option<String> = None;
    let mut fixture_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--op" => {
                op = Some(argv.get(i + 1).ok_or("--op needs a value")?.clone());
                i += 2;
            }
            "--config" => {
                config = Some(argv.get(i + 1).ok_or("--config needs a value")?.clone());
                i += 2;
            }
            "--fixture-dir" => {
                fixture_dir = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--fixture-dir needs a value")?,
                ));
                i += 2;
            }
            "--output-dir" => {
                output_dir = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--output-dir needs a value")?,
                ));
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        op: op.ok_or("--op is required")?,
        config: config.ok_or("--config is required")?,
        fixture_dir: fixture_dir.ok_or("--fixture-dir is required")?,
        output_dir: output_dir.ok_or("--output-dir is required")?,
    })
}

// ---------------------------------------------------------------------------
// Binary I/O — single-tensor `[u32 ndim][u32 × ndim shape][f32 data]`
// (matches the python pin script's `dump_f32`).
// ---------------------------------------------------------------------------

fn read_f32_tensor(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)
        .map_err(|e| format!("read ndim from {}: {e}", path.display()))?;
    let ndim = u32::from_le_bytes(buf4) as usize;
    let mut shape = Vec::with_capacity(ndim);
    for di in 0..ndim {
        f.read_exact(&mut buf4)
            .map_err(|e| format!("read shape[{di}] from {}: {e}", path.display()))?;
        shape.push(u32::from_le_bytes(buf4) as usize);
    }
    let numel: usize = shape.iter().product();
    let mut data_bytes = vec![0u8; numel * 4];
    f.read_exact(&mut data_bytes)
        .map_err(|e| format!("read data from {}: {e}", path.display()))?;
    let mut data = Vec::with_capacity(numel);
    for chunk in data_bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((shape, data))
}

fn write_f32_tensor(path: &Path, shape: &[usize], data: &[f32]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    let mut f = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    f.write_all(&(shape.len() as u32).to_le_bytes())
        .map_err(|e| format!("write ndim to {}: {e}", path.display()))?;
    for d in shape {
        f.write_all(&(*d as u32).to_le_bytes())
            .map_err(|e| format!("write shape to {}: {e}", path.display()))?;
    }
    let mut buf: Vec<u8> = Vec::with_capacity(data.len() * 4);
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)
        .map_err(|e| format!("write data to {}: {e}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_leaf(
    fixture_dir: &Path,
    op: &str,
    config: &str,
    name: &str,
    requires_grad: bool,
) -> Result<Tensor<f32>, String> {
    let p = fixture_dir
        .join(op)
        .join(config)
        .join("inputs")
        .join(format!("{name}.bin"));
    let (shape, data) = read_f32_tensor(&p)?;
    Tensor::from_storage(TensorStorage::cpu(data), shape, requires_grad)
        .map_err(|e| format!("from_storage {name}: {e}"))
}

fn dump_tensor(out_dir: &Path, rel: &str, t: &Tensor<f32>) -> Result<(), String> {
    let shape = t.shape().to_vec();
    let data = t.data().map_err(|e| format!("data() {rel}: {e}"))?.to_vec();
    write_f32_tensor(&out_dir.join(rel), &shape, &data)
}

fn dump_grad(out_dir: &Path, name: &str, leaf: &Tensor<f32>) -> Result<(), String> {
    let g = leaf
        .grad()
        .map_err(|e| format!("grad() {name}: {e}"))?
        .ok_or_else(|| format!("leaf {name} produced no gradient after backward()"))?;
    dump_tensor(out_dir, &format!("grads/{name}.bin"), &g)
}

// ---------------------------------------------------------------------------
// Per-op dispatch
// ---------------------------------------------------------------------------

fn run_op(args: &Args) -> Result<(), String> {
    let fd = &args.fixture_dir;
    let op = args.op.as_str();
    let cfg = args.config.as_str();
    let od = &args.output_dir;
    std::fs::create_dir_all(od).map_err(|e| format!("mkdir output: {e}"))?;

    match op {
        // -------- matmul family --------
        "matmul_2d" => {
            let a = load_leaf(fd, op, cfg, "a", true)?;
            let b = load_leaf(fd, op, cfg, "b", true)?;
            let out = matmul_differentiable(&a, &b).map_err(|e| format!("matmul forward: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "a", &a)?;
            dump_grad(od, "b", &b)?;
        }
        "bmm" => {
            let a = load_leaf(fd, op, cfg, "a", true)?;
            let b = load_leaf(fd, op, cfg, "b", true)?;
            let out = bmm_differentiable(&a, &b).map_err(|e| format!("bmm forward: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "a", &a)?;
            dump_grad(od, "b", &b)?;
        }
        "linear" => {
            // linear_fused expects input [B, in], weight [out, in], bias [out]
            // (matches torch.nn.Linear convention and our pin script).
            let x = load_leaf(fd, op, cfg, "input", true)?;
            let w = load_leaf(fd, op, cfg, "weight", true)?;
            let b = load_leaf(fd, op, cfg, "bias", true)?;
            let out =
                linear_fused(&x, &w, Some(&b)).map_err(|e| format!("linear_fused forward: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "input", &x)?;
            dump_grad(od, "weight", &w)?;
            dump_grad(od, "bias", &b)?;
        }

        // -------- activations (single input `x`) --------
        "relu" | "gelu" | "silu" | "sigmoid" | "tanh" => {
            let x = load_leaf(fd, op, cfg, "x", true)?;
            let out = match op {
                "relu" => relu(&x),
                "gelu" => gelu(&x),
                "silu" => silu(&x),
                "sigmoid" => sigmoid(&x),
                "tanh" => tanh(&x),
                _ => unreachable!(),
            }
            .map_err(|e| format!("{op} forward: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "x", &x)?;
        }

        // -------- log_softmax (last axis) — plain `sum()` reduction.
        // `sum(log_softmax(x))` is *not* constant in x (unlike
        // `sum(softmax(x))`), so the plain-sum reduction gives a
        // non-degenerate gradient. --------
        "log_softmax" => {
            let x = load_leaf(fd, op, cfg, "x", true)?;
            let out = log_softmax(&x).map_err(|e| format!("{op} forward: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "x", &x)?;
        }

        // -------- softmax (last axis) — weighted reduction.
        // `sum(softmax(x))` is constant (each row sums to 1) and its
        // VJP is identically zero, which would not exercise the
        // softmax backward Jacobian. Mirror the pin script: multiply
        // the softmax output element-wise by a fixed no-grad "target"
        // tensor before summing, so the VJP threads a non-trivial
        // gradient through `SoftmaxBackward`. --------
        "softmax" => {
            let x = load_leaf(fd, op, cfg, "x", true)?;
            let target = load_leaf(fd, op, cfg, "target", false)?;
            let sm = softmax(&x).map_err(|e| format!("softmax forward: {e}"))?;
            let out = mul(&sm, &target).map_err(|e| format!("softmax * target: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "x", &x)?;
        }

        // -------- reductions with dim --------
        "sum_dim" => {
            let x = load_leaf(fd, op, cfg, "x", true)?;
            let (dim, keepdim) = parse_sum_dim_config(cfg)?;
            let out = sum_dim(&x, dim, keepdim).map_err(|e| format!("sum_dim forward: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "x", &x)?;
        }
        "mean_dim" => {
            let x = load_leaf(fd, op, cfg, "x", true)?;
            // The single pinned config is `3x5x7_dim1_nokeep` — dim=1, keepdim=false.
            let out = mean_dim(&x, 1, false).map_err(|e| format!("mean_dim forward: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "x", &x)?;
        }

        // -------- binary element-wise --------
        "add" | "mul" | "sub" | "div" => {
            let a = load_leaf(fd, op, cfg, "a", true)?;
            let b = load_leaf(fd, op, cfg, "b", true)?;
            let out = match op {
                "add" => add(&a, &b),
                "mul" => mul(&a, &b),
                "sub" => sub(&a, &b),
                "div" => div(&a, &b),
                _ => unreachable!(),
            }
            .map_err(|e| format!("{op} forward: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "a", &a)?;
            dump_grad(od, "b", &b)?;
        }

        // -------- transcendental unary --------
        "log" | "exp" => {
            let x = load_leaf(fd, op, cfg, "x", true)?;
            let out = if op == "log" { log(&x) } else { exp(&x) }
                .map_err(|e| format!("{op} forward: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "x", &x)?;
        }

        // -------- pow(x, c) --------
        "pow" => {
            let x = load_leaf(fd, op, cfg, "x", true)?;
            // Single pinned config: exponent = 2.5
            let out = pow(&x, 2.5).map_err(|e| format!("pow forward: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "x", &x)?;
        }

        // -------- shape ops --------
        "reshape" => {
            let x = load_leaf(fd, op, cfg, "x", true)?;
            // The single config reshapes [2,3,4] -> [6,4].
            let out = reshape(&x, &[6, 4]).map_err(|e| format!("reshape forward: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "x", &x)?;
        }
        "transpose" => {
            let x = load_leaf(fd, op, cfg, "x", true)?;
            // The single config swaps (0,1) on a 2-D [4,6] -> [6,4]. We
            // call .contiguous() to make backward shape-stable (matches
            // the torch reference, which also calls .contiguous() so
            // .sum() reduces over a materialized layout).
            let t = x
                .transpose(0, 1)
                .map_err(|e| format!("transpose forward: {e}"))?;
            let out = t
                .contiguous()
                .map_err(|e| format!("transpose.contiguous: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "x", &x)?;
        }
        "cat" => {
            let a = load_leaf(fd, op, cfg, "a", true)?;
            let b = load_leaf(fd, op, cfg, "b", true)?;
            let c = load_leaf(fd, op, cfg, "c", true)?;
            let out = cat(&[a.clone(), b.clone(), c.clone()], 0)
                .map_err(|e| format!("cat forward: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "a", &a)?;
            dump_grad(od, "b", &b)?;
            dump_grad(od, "c", &c)?;
        }

        // -------- embedding (index_select_dim along axis 0) --------
        "embedding" => {
            // weight: [vocab, emb], indices: f32-encoded long indices.
            let weight = load_leaf(fd, op, cfg, "weight", true)?;
            let (idx_shape, idx_data) =
                read_f32_tensor(&fd.join(op).join(cfg).join("inputs").join("indices.bin"))?;
            if idx_shape.len() != 1 {
                return Err(format!("embedding indices must be 1-D, got {idx_shape:?}"));
            }
            let indices_i64: Vec<i64> = idx_data.iter().map(|&f| f as i64).collect();
            let idx_tensor = IntTensor::<i64>::from_slice(&indices_i64, &idx_shape)
                .map_err(|e| format!("IntTensor::from_slice: {e}"))?;
            let out = index_select_dim(&weight, 0, &idx_tensor)
                .map_err(|e| format!("index_select_dim forward: {e}"))?;
            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "weight", &weight)?;
        }

        // -------- attention: Q @ K^T * scale -> softmax -> @ V --------
        "attention" => {
            // The pin's single config is [B=2, T=3, d=4].
            let q = load_leaf(fd, op, cfg, "q", true)?;
            let k = load_leaf(fd, op, cfg, "k", true)?;
            let v = load_leaf(fd, op, cfg, "v", true)?;
            let d = q.shape().last().copied().unwrap_or(1);
            let scale: f32 = 1.0 / (d as f32).sqrt();

            // K^T over the last two dims: transpose(1, 2). Materialize
            // because bmm requires contiguous inputs in our impl.
            let k_t = k
                .transpose(1, 2)
                .map_err(|e| format!("attention K^T: {e}"))?
                .contiguous()
                .map_err(|e| format!("attention K^T.contiguous: {e}"))?;

            let scores =
                bmm_differentiable(&q, &k_t).map_err(|e| format!("attention Q @ K^T: {e}"))?;
            // scores * scale via mul with a same-shape constant tensor.
            let scale_t = Tensor::from_storage(
                TensorStorage::cpu(vec![scale; scores.numel()]),
                scores.shape().to_vec(),
                false,
            )
            .map_err(|e| format!("attention scale tensor: {e}"))?;
            let scaled = mul(&scores, &scale_t).map_err(|e| format!("attention scale mul: {e}"))?;
            let attn = softmax(&scaled).map_err(|e| format!("attention softmax: {e}"))?;
            let out =
                bmm_differentiable(&attn, &v).map_err(|e| format!("attention attn @ V: {e}"))?;

            dump_tensor(od, "forward_out.bin", &out)?;
            out.sum_to_scalar()?
                .backward()
                .map_err(|e| format!("backward: {e}"))?;
            dump_grad(od, "q", &q)?;
            dump_grad(od, "k", &k)?;
            dump_grad(od, "v", &v)?;
        }

        other => return Err(format!("unknown op {other:?}")),
    }
    Ok(())
}

/// Parse the `sum_dim` config string into `(dim, keepdim)`. The pin
/// emits two configs: `3x5x7_dim1_nokeep` and `3x5x7_dim2_keep`.
fn parse_sum_dim_config(cfg: &str) -> Result<(i64, bool), String> {
    let keepdim = cfg.contains("_keep");
    // Pull the digit immediately after `_dim`.
    let dim = cfg
        .split('_')
        .find_map(|tok| tok.strip_prefix("dim"))
        .and_then(|s| s.parse::<i64>().ok())
        .ok_or_else(|| format!("cannot parse sum_dim config {cfg:?}"))?;
    Ok((dim, keepdim))
}

// ---------------------------------------------------------------------------
// `sum_to_scalar` is a tiny convenience on top of `ferrotorch_core::sum_dim`
// chained until the tensor is 0-D. Calling `.sum()` on a Tensor isn't
// available as a free function in the public surface, but iterating
// `sum_dim` until ndim==0 is the established pattern (see
// `grad_fns/reduction.rs::sum` which is the same semantically).
// ---------------------------------------------------------------------------

trait SumToScalar {
    fn sum_to_scalar(&self) -> Result<Tensor<f32>, String>;
}

impl SumToScalar for Tensor<f32> {
    fn sum_to_scalar(&self) -> Result<Tensor<f32>, String> {
        // `grad_fns::reduction::sum` already produces a 0-D scalar by
        // reducing over every axis; it's the canonical "total sum" op.
        ferrotorch_core::grad_fns::reduction::sum(self).map_err(|e| format!("sum_to_scalar: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), String> {
    let args = parse_args()?;
    eprintln!(
        "[autograd_dump] op={} config={} fixture-dir={} output-dir={}",
        args.op,
        args.config,
        args.fixture_dir.display(),
        args.output_dir.display(),
    );
    run_op(&args)?;
    eprintln!("[autograd_dump] DONE");
    Ok(())
}
