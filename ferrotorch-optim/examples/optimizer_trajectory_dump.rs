//! Optimizer-trajectory dump binary for the ferrotorch-optim real-artifact
//! parity harness (Phase C.2, #1155).
//!
//! Companion to `scripts/verify_optimizer_inference.py` and the pin
//! script `scripts/pin_pretrained_optimizer_trajectories.py`. Given a
//! fixture directory containing `initial_params.bin`,
//! `gradients_step_K.bin` for `K in 0..N`, and a config naming the
//! optimizer to instantiate, this example:
//!
//!   1. Builds a 3-layer MLP whose Parameter tensors are loaded from
//!      `initial_params.bin`.
//!   2. Instantiates the named optimizer (SGD / Adam / AdamW / RMSprop /
//!      Adagrad) with the supplied hyperparameters.
//!   3. For each of `N` steps loads the pre-computed gradient sequence
//!      into each parameter's `.grad` and calls `optimizer.step()`.
//!   4. Dumps the final parameter state to `--output` in the same
//!      multi-tensor f32 little-endian format the pin script produces.
//!
//! All fixture data is `f32`. The optimizer is also instantiated in
//! `f32` so the comparison is f32-vs-f32 (matching `torch.optim`'s
//! default). With **frozen gradients** the per-step round-trip is
//! purely the optimizer math — no autograd is invoked.
//!
//! Multi-tensor binary format (little-endian):
//!
//! ```text
//! [u32 num_tensors]
//! per tensor:
//!   [u32 ndim] [u32 * ndim shape] [f32 * prod(shape)]
//! ```
//!
//! Tensor *order* is the contract: the example expects the 6
//! parameters of the canonical MLP in the order
//!   layer0.weight, layer0.bias,
//!   layer1.weight, layer1.bias,
//!   layer2.weight, layer2.bias.
//!
//! Usage:
//! ```text
//! cargo run -p ferrotorch-optim --release --example optimizer_trajectory_dump -- \
//!   --fixture-dir /tmp/ferrotorch_optimizer_trajectories/adam_default \
//!   --optimizer Adam --config-name adam_default \
//!   --output /tmp/rust_final.bin
//! ```
//!
//! The optimizer hyperparameters are read from `meta.json` inside
//! `--fixture-dir`, so the same example handles every config without a
//! per-config rebuild.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_nn::Parameter;
use ferrotorch_optim::{
    Adagrad, AdagradConfig, Adam, AdamConfig, AdamW, AdamWConfig, Optimizer, ParamGroup, Rmsprop,
    RmspropConfig, Sgd, SgdConfig,
};

const NUM_STEPS: usize = 10;
const NUM_PARAMS: usize = 6;

const PARAM_NAMES: [&str; NUM_PARAMS] = [
    "layer0.weight",
    "layer0.bias",
    "layer1.weight",
    "layer1.bias",
    "layer2.weight",
    "layer2.bias",
];

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    fixture_dir: PathBuf,
    optimizer: String,
    config_name: String,
    output: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut fixture_dir: Option<PathBuf> = None;
    let mut optimizer: Option<String> = None;
    let mut config_name: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--fixture-dir" => {
                fixture_dir = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--fixture-dir needs a value")?,
                ));
                i += 2;
            }
            "--optimizer" => {
                optimizer = Some(argv.get(i + 1).ok_or("--optimizer needs a value")?.clone());
                i += 2;
            }
            "--config-name" => {
                config_name = Some(
                    argv.get(i + 1)
                        .ok_or("--config-name needs a value")?
                        .clone(),
                );
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--output needs a value")?,
                ));
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        fixture_dir: fixture_dir.ok_or("--fixture-dir is required")?,
        optimizer: optimizer.ok_or("--optimizer is required")?,
        config_name: config_name.ok_or("--config-name is required")?,
        output: output.ok_or("--output is required")?,
    })
}

// ---------------------------------------------------------------------------
// Multi-tensor binary format (mirrors the Python pin script).
// ---------------------------------------------------------------------------

/// One `(shape, flat row-major f32 data)` tensor as read from / written
/// to a multi-tensor `.bin` fixture file. The pair carries no name —
/// fixture file order is the contract (see `PARAM_NAMES`).
type DumpedTensor = (Vec<usize>, Vec<f32>);

fn read_multi_tensor_f32(path: &Path) -> Result<Vec<DumpedTensor>, String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)
        .map_err(|e| format!("read num_tensors from {}: {e}", path.display()))?;
    let n = u32::from_le_bytes(buf) as usize;
    let mut out = Vec::with_capacity(n);
    for ti in 0..n {
        f.read_exact(&mut buf)
            .map_err(|e| format!("read ndim[{ti}] from {}: {e}", path.display()))?;
        let ndim = u32::from_le_bytes(buf) as usize;
        let mut shape = Vec::with_capacity(ndim);
        for di in 0..ndim {
            f.read_exact(&mut buf)
                .map_err(|e| format!("read shape[{ti}][{di}] from {}: {e}", path.display()))?;
            shape.push(u32::from_le_bytes(buf) as usize);
        }
        let numel: usize = shape.iter().product();
        let mut data_bytes = vec![0u8; numel * 4];
        f.read_exact(&mut data_bytes)
            .map_err(|e| format!("read data[{ti}] from {}: {e}", path.display()))?;
        let mut data = Vec::with_capacity(numel);
        for chunk in data_bytes.chunks_exact(4) {
            data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        out.push((shape, data));
    }
    Ok(out)
}

fn write_multi_tensor_f32(path: &Path, tensors: &[(Vec<usize>, Vec<f32>)]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    for (shape, data) in tensors {
        let expect: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expect,
            "tensor data {} disagrees with shape product {}",
            data.len(),
            expect
        );
        f.write_all(&(shape.len() as u32).to_le_bytes())?;
        for &d in shape {
            f.write_all(&(d as u32).to_le_bytes())?;
        }
        let mut buf = Vec::with_capacity(data.len() * 4);
        for &v in data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&buf)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Hyperparameter parsing — read meta.json without pulling serde in as
// a hard dep. The script-side meta.json is small and predictable;
// parse the handful of fields we care about by hand.
// ---------------------------------------------------------------------------

/// Hyperparameter set extracted from `meta.json["config"]` for the
/// 5 optimizer families used in the dispatch. Every field is optional;
/// the optimizer factory below applies the same defaults torch.optim
/// uses when a key is absent.
#[derive(Debug, Default, Clone)]
struct HParams {
    lr: Option<f64>,
    momentum: Option<f64>,
    nesterov: Option<bool>,
    weight_decay: Option<f64>,
    betas: Option<(f64, f64)>,
    eps: Option<f64>,
    alpha: Option<f64>,
    lr_decay: Option<f64>,
}

fn read_meta_hparams(meta_path: &Path) -> Result<HParams, String> {
    let raw = std::fs::read_to_string(meta_path)
        .map_err(|e| format!("read {}: {e}", meta_path.display()))?;
    // Crude but bounded JSON tokenizer: we only need primitive number,
    // bool, and 2-tuple values out of the "config" object. Bringing in
    // a JSON parser dep would force a workspace touch outside this
    // crate's scope, which the dispatch forbids.
    let config_obj = extract_object(&raw, "config")
        .ok_or_else(|| "meta.json missing 'config' object".to_string())?;
    let hp = HParams {
        lr: parse_number(config_obj, "lr"),
        momentum: parse_number(config_obj, "momentum"),
        weight_decay: parse_number(config_obj, "weight_decay"),
        eps: parse_number(config_obj, "eps"),
        alpha: parse_number(config_obj, "alpha"),
        lr_decay: parse_number(config_obj, "lr_decay"),
        nesterov: parse_bool(config_obj, "nesterov"),
        betas: parse_pair(config_obj, "betas"),
    };
    Ok(hp)
}

/// Extract the `{...}` body of a `"<key>": { ... }` object from a JSON
/// text. Returns `None` if not found. Brace-counter is enough since
/// the pin script writes one-level-deep config dicts.
fn extract_object<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{key}\":");
    let start = text.find(&pat)?;
    let after = &text[start + pat.len()..];
    let open = after.find('{')?;
    let mut depth = 0i32;
    let bytes = after.as_bytes();
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&after[open + 1..i]);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_number(text: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\":");
    let pos = text.find(&pat)?;
    let after = text[pos + pat.len()..].trim_start();
    // Stop at the first delimiter — `,`, `}`, or whitespace after the
    // value. Numbers in the meta.json may be `1e-08` or `0.001` or `10`.
    let end = after.find([',', '}', '\n']).unwrap_or(after.len());
    after[..end].trim().parse::<f64>().ok()
}

fn parse_bool(text: &str, key: &str) -> Option<bool> {
    let pat = format!("\"{key}\":");
    let pos = text.find(&pat)?;
    let after = text[pos + pat.len()..].trim_start();
    if after.starts_with("true") {
        Some(true)
    } else if after.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn parse_pair(text: &str, key: &str) -> Option<(f64, f64)> {
    let pat = format!("\"{key}\":");
    let pos = text.find(&pat)?;
    let after = text[pos + pat.len()..].trim_start();
    // Tuples serialize as JSON arrays `[a, b]`.
    if !after.starts_with('[') {
        return None;
    }
    let close = after.find(']')?;
    let inner = &after[1..close];
    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
    if parts.len() != 2 {
        return None;
    }
    let a = parts[0].parse::<f64>().ok()?;
    let b = parts[1].parse::<f64>().ok()?;
    Some((a, b))
}

// ---------------------------------------------------------------------------
// Optimizer trampoline — the Optimizer trait is generic, so a single
// `Box<dyn Optimizer<f32>>` dispatch is the cleanest way to drive
// the 5 families through one loop.
// ---------------------------------------------------------------------------

fn build_optimizer(
    name: &str,
    hp: &HParams,
    params: Vec<Parameter<f32>>,
) -> Result<Box<dyn Optimizer<f32>>, String> {
    let lr = hp.lr.ok_or("config missing lr")?;
    match name {
        "SGD" => {
            let mut cfg = SgdConfig::new(lr);
            if let Some(m) = hp.momentum {
                cfg = cfg.momentum(m);
            }
            if let Some(n) = hp.nesterov {
                cfg = cfg.nesterov(n);
            }
            if let Some(wd) = hp.weight_decay {
                cfg = cfg.weight_decay(wd);
            }
            Ok(Box::new(Sgd::new(params, cfg)))
        }
        "Adam" => {
            let mut cfg = AdamConfig::default().with_lr(lr);
            if let Some(b) = hp.betas {
                cfg = cfg.with_betas(b);
            }
            if let Some(e) = hp.eps {
                cfg = cfg.with_eps(e);
            }
            if let Some(wd) = hp.weight_decay {
                cfg = cfg.with_weight_decay(wd);
            }
            Ok(Box::new(Adam::new(params, cfg)))
        }
        "AdamW" => {
            let mut cfg = AdamWConfig::default().with_lr(lr);
            if let Some(b) = hp.betas {
                cfg = cfg.with_betas(b);
            }
            if let Some(e) = hp.eps {
                cfg = cfg.with_eps(e);
            }
            if let Some(wd) = hp.weight_decay {
                cfg = cfg.with_weight_decay(wd);
            }
            Ok(Box::new(AdamW::new(params, cfg)))
        }
        "RMSprop" => {
            let mut cfg = RmspropConfig::default().with_lr(lr);
            if let Some(a) = hp.alpha {
                cfg = cfg.with_alpha(a);
            }
            if let Some(e) = hp.eps {
                cfg = cfg.with_eps(e);
            }
            if let Some(m) = hp.momentum {
                cfg = cfg.with_momentum(m);
            }
            if let Some(wd) = hp.weight_decay {
                cfg = cfg.with_weight_decay(wd);
            }
            Ok(Box::new(Rmsprop::new(params, cfg)))
        }
        "Adagrad" => {
            let mut cfg = AdagradConfig::default().with_lr(lr);
            if let Some(d) = hp.lr_decay {
                cfg = cfg.with_lr_decay(d);
            }
            if let Some(e) = hp.eps {
                cfg = cfg.with_eps(e);
            }
            if let Some(wd) = hp.weight_decay {
                cfg = cfg.with_weight_decay(wd);
            }
            // Adagrad's `new` takes pre-built groups, not a flat param list.
            let group =
                ParamGroup::new(params, lr).with_weight_decay(hp.weight_decay.unwrap_or(0.0));
            Ok(Box::new(Adagrad::new(vec![group], cfg)))
        }
        other => Err(format!("unknown optimizer {other}")),
    }
}

// ---------------------------------------------------------------------------
// Main flow.
// ---------------------------------------------------------------------------

fn run() -> FerrotorchResult<()> {
    let args = parse_args().map_err(|m| FerrotorchError::InvalidArgument { message: m })?;
    eprintln!(
        "[optimizer_trajectory_dump] fixture_dir={} optimizer={} config={}",
        args.fixture_dir.display(),
        args.optimizer,
        args.config_name,
    );

    // -- 1. Read meta + initial params. ---------------------------------
    let meta_path = args.fixture_dir.join("meta.json");
    let hp = read_meta_hparams(&meta_path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("hparam parse failed: {e}"),
    })?;
    eprintln!("[optimizer_trajectory_dump] hparams = {hp:?}");

    let init_path = args.fixture_dir.join("initial_params.bin");
    let init = read_multi_tensor_f32(&init_path).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("read initial_params.bin: {e}"),
    })?;
    if init.len() != NUM_PARAMS {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "initial_params.bin contains {} tensors, expected {NUM_PARAMS}",
                init.len()
            ),
        });
    }

    // -- 2. Build parameters. Order matches PARAM_NAMES. ----------------
    let mut params: Vec<Parameter<f32>> = Vec::with_capacity(NUM_PARAMS);
    let mut shapes: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, (shape, data)) in init.into_iter().enumerate() {
        let name = PARAM_NAMES[idx];
        let p = Parameter::from_slice(&data, &shape)?;
        shapes.insert(name, shape);
        params.push(p);
    }

    // Stash clones for grad-setting (Parameter clones share the underlying
    // Arc, so set_grad on the clone updates the same tensor the optimizer
    // sees).
    let param_clones: Vec<Parameter<f32>> = params.to_vec();

    // -- 3. Build the optimizer. ----------------------------------------
    let mut opt = build_optimizer(&args.optimizer, &hp, params).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("build_optimizer: {e}"),
        }
    })?;

    // -- 4. Drive the 10 steps. -----------------------------------------
    for step in 0..NUM_STEPS {
        let grad_path = args.fixture_dir.join(format!("gradients_step_{step}.bin"));
        let grads =
            read_multi_tensor_f32(&grad_path).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("read {}: {e}", grad_path.display()),
            })?;
        if grads.len() != NUM_PARAMS {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "step {step}: gradient file has {} tensors, expected {NUM_PARAMS}",
                    grads.len(),
                ),
            });
        }

        // Set the .grad on each parameter for this step.
        for (pi, (shape, data)) in grads.into_iter().enumerate() {
            let expected = shapes
                .get(PARAM_NAMES[pi])
                .expect("param shape recorded at init");
            if &shape != expected {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "step {step}: grad[{pi}] shape {shape:?} != param shape {expected:?}"
                    ),
                });
            }
            let g = Tensor::from_storage(TensorStorage::cpu(data), shape, false)?;
            param_clones[pi].set_grad(Some(g))?;
        }

        opt.step()?;
    }

    // -- 5. Dump final params in the same canonical order. --------------
    let mut final_dump: Vec<(Vec<usize>, Vec<f32>)> = Vec::with_capacity(NUM_PARAMS);
    let groups = opt.param_groups();
    // The optimizer might own params in 1 or more groups; SGD/Adam/AdamW/
    // RMSprop place all params in group 0, Adagrad uses a single group too
    // (see `build_optimizer`). Flatten in declaration order.
    let mut flat: Vec<&Parameter<f32>> = Vec::with_capacity(NUM_PARAMS);
    for g in groups {
        for p in g.params() {
            flat.push(p);
        }
    }
    if flat.len() != NUM_PARAMS {
        return Err(FerrotorchError::Internal {
            message: format!(
                "optimizer exposes {} params, expected {NUM_PARAMS}",
                flat.len()
            ),
        });
    }
    for p in &flat {
        let shape = p.shape().to_vec();
        let data: Vec<f32> = p.data()?.to_vec();
        final_dump.push((shape, data));
    }
    write_multi_tensor_f32(&args.output, &final_dump).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("write {}: {e}", args.output.display()),
        }
    })?;
    eprintln!(
        "[optimizer_trajectory_dump] wrote {} ({} bytes)",
        args.output.display(),
        std::fs::metadata(&args.output)
            .map(|m| m.len())
            .unwrap_or(0)
    );

    // -- 6. JSON verdict line so the Python harness can parse the run. --
    let mut s = String::new();
    s.push('{');
    s.push_str(&format!("\"optimizer\":\"{}\",", args.optimizer));
    s.push_str(&format!("\"config_name\":\"{}\",", args.config_name));
    s.push_str(&format!("\"num_steps\":{NUM_STEPS},"));
    s.push_str(&format!("\"num_params\":{NUM_PARAMS}"));
    s.push('}');
    println!("{s}");

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[optimizer_trajectory_dump] error: {e}");
        std::process::exit(1);
    }
}
