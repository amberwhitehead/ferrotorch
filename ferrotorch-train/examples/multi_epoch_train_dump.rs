//! Multi-epoch training-trajectory dump binary for the ferrotorch-train
//! real-artifact parity harness (Phase E, #1161).
//!
//! Companion to `scripts/verify_training_trajectory.py` and the pin
//! script `scripts/pin_pretrained_training_trajectory.py`. Loads the
//! initial parameter state + the full deterministic dataset from a
//! local fixture directory (typically populated from
//! `ferrotorch/training-trajectory-v1` by the verify harness), and
//! runs the full training loop:
//!
//!   1. Build a 3-layer MLP whose Parameters are loaded from
//!      `initial_state.safetensors`.
//!   2. Build the same `Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8)`
//!      that the pin script used.
//!   3. For each of 5 epochs:
//!        - Iterate the deterministic dataset in sequential batches of
//!          size 4 (drop_last=False — 25 batches per epoch). This
//!          matches the pin script's `for i in range(0, N, BATCH)` and
//!          is semantically identical to a `DataLoader(shuffle=False,
//!          drop_last=False)` (which is already covered by the
//!          dedicated DataLoader-parity harness in #1156).
//!        - `opt.zero_grad()`, forward, `mse_loss(reduction='mean')`,
//!          `loss.backward()`, `opt.step()`.
//!        - Snapshot the post-epoch state_dict to
//!          `<output-dir>/epoch_{K+1}_state.safetensors`.
//!
//! This exercises the *combined* behavior of forward (linear + relu),
//! loss (MSE mean), backward (live autograd — gradients are computed
//! by ferrotorch, not replayed from torch), optimizer (Adam state
//! initialization + 125 update steps), and sequential batch iteration.
//! A divergence anywhere in this stack shows up as state_dict drift.
//!
//! ## Multi-tensor binary format
//!
//! `X_full.bin` / `y_full.bin` are little-endian:
//!
//! ```text
//! [u32 num_tensors=1]
//! [u32 ndim] [u32 * ndim shape] [f32 * prod(shape)]
//! ```
//!
//! ## Usage
//!
//! ```text
//! cargo run -p ferrotorch-train --release --example multi_epoch_train_dump -- \
//!   --fixture-dir /tmp/ferrotorch_training_trajectory \
//!   --output-dir  /tmp/ferrotorch_training_trajectory/rust_dump
//! ```
//!
//! All hyperparameters are baked into this binary — they match the pin
//! script's constants exactly. Diverging from them would silently make
//! every harness verdict invalid, so no CLI knob is exposed.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use ferrotorch_core::autograd::no_grad::no_grad;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor};
use ferrotorch_nn::functional::{mse_loss, relu};
use ferrotorch_nn::{Linear, Module, Parameter, StateDict};
use ferrotorch_optim::{Adam, AdamConfig, Optimizer};
use ferrotorch_serialize::{load_safetensors, save_safetensors};

// ---------------------------------------------------------------------------
// Hyperparameters — must match `scripts/pin_pretrained_training_trajectory.py`.
// ---------------------------------------------------------------------------

const D_IN: usize = 64;
const D_HID1: usize = 32;
const D_HID2: usize = 16;
const D_OUT: usize = 8;
const N: usize = 100;
const BATCH: usize = 4;
const EPOCHS: usize = 5;
const LR: f64 = 1e-3;

const PARAM_KEYS: [&str; 6] = [
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias",
];

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    fixture_dir: PathBuf,
    output_dir: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut fixture_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
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
        fixture_dir: fixture_dir.ok_or("--fixture-dir is required")?,
        output_dir: output_dir.ok_or("--output-dir is required")?,
    })
}

// ---------------------------------------------------------------------------
// Multi-tensor f32 binary reader — mirror of the Python pin script's
// `dump_f32_tensor` and the Rust optimizer-trajectory example's
// `read_multi_tensor_f32`.
// ---------------------------------------------------------------------------

fn read_single_tensor_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)
        .map_err(|e| format!("read num_tensors from {}: {e}", path.display()))?;
    let n = u32::from_le_bytes(buf) as usize;
    if n != 1 {
        return Err(format!(
            "{}: expected num_tensors=1 (single-tensor format), got {n}",
            path.display()
        ));
    }
    f.read_exact(&mut buf)
        .map_err(|e| format!("read ndim from {}: {e}", path.display()))?;
    let ndim = u32::from_le_bytes(buf) as usize;
    let mut shape = Vec::with_capacity(ndim);
    for di in 0..ndim {
        f.read_exact(&mut buf)
            .map_err(|e| format!("read shape[{di}] from {}: {e}", path.display()))?;
        shape.push(u32::from_le_bytes(buf) as usize);
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

// ---------------------------------------------------------------------------
// MLP — keyed `fc{1,2,3}.{weight,bias}` to match the pin's
// `torch.nn.Linear` attribute names verbatim. We do not use
// `nn::Sequential` here because Sequential's named_parameters use
// numeric indices (`0.weight`, `2.weight`), which would force a key
// rename on either the pin or the load path; a custom struct keeps
// the contract literal.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Mlp {
    fc1: Linear<f32>,
    fc2: Linear<f32>,
    fc3: Linear<f32>,
    training: bool,
}

impl Mlp {
    fn new() -> FerrotorchResult<Self> {
        Ok(Self {
            fc1: Linear::<f32>::new(D_IN, D_HID1, true)?,
            fc2: Linear::<f32>::new(D_HID1, D_HID2, true)?,
            fc3: Linear::<f32>::new(D_HID2, D_OUT, true)?,
            training: true,
        })
    }
}

impl Module<f32> for Mlp {
    fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        let h1 = relu(&self.fc1.forward(input)?)?;
        let h2 = relu(&self.fc2.forward(&h1)?)?;
        self.fc3.forward(&h2)
    }

    fn parameters(&self) -> Vec<&Parameter<f32>> {
        let mut out = Vec::with_capacity(6);
        out.extend(self.fc1.parameters());
        out.extend(self.fc2.parameters());
        out.extend(self.fc3.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
        let mut out: Vec<&mut Parameter<f32>> = Vec::with_capacity(6);
        out.extend(self.fc1.parameters_mut());
        out.extend(self.fc2.parameters_mut());
        out.extend(self.fc3.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
        let mut out = Vec::with_capacity(6);
        for (sub, p) in self.fc1.named_parameters() {
            out.push((format!("fc1.{sub}"), p));
        }
        for (sub, p) in self.fc2.named_parameters() {
            out.push((format!("fc2.{sub}"), p));
        }
        for (sub, p) in self.fc3.named_parameters() {
            out.push((format!("fc3.{sub}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.fc1.train();
        self.fc2.train();
        self.fc3.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.fc1.eval();
        self.fc2.eval();
        self.fc3.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Load `initial_state.safetensors` into the model's parameters in the
/// canonical `fc{1,2,3}.{weight,bias}` order, shape-checking each entry.
/// Uses `Module::load_state_dict` so missing / extra keys are caught
/// strictly.
fn load_initial_state(model: &mut Mlp, path: &Path) -> FerrotorchResult<()> {
    let state: StateDict<f32> = load_safetensors::<f32>(path)?;
    // strict=true: every PARAM_KEYS entry must be present, no extras.
    model.load_state_dict(&state, true)?;
    // Sanity-check the shape vector after load — `load_state_dict`
    // already verifies shapes, but a missing key + strict=true
    // returns Err; this guards against future loader changes that
    // could silently skip keys.
    let by_name: HashMap<String, &Parameter<f32>> =
        model.named_parameters().into_iter().collect();
    for k in PARAM_KEYS {
        if !by_name.contains_key(k) {
            return Err(FerrotorchError::Internal {
                message: format!("model is missing expected param {k} after load_state_dict"),
            });
        }
    }
    Ok(())
}

/// Materialize the dataset tensors from the multi-tensor f32 binaries
/// produced by the pin script. The pin emits two single-tensor files:
/// `X_full.bin` is `[N, D_IN]` and `y_full.bin` is `[N, D_OUT]`.
fn load_dataset(fixture_dir: &Path) -> FerrotorchResult<(Tensor<f32>, Tensor<f32>)> {
    let x_path = fixture_dir.join("X_full.bin");
    let y_path = fixture_dir.join("y_full.bin");
    let (x_shape, x_data) =
        read_single_tensor_f32(&x_path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("load X_full.bin: {e}"),
        })?;
    let (y_shape, y_data) =
        read_single_tensor_f32(&y_path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("load y_full.bin: {e}"),
        })?;
    if x_shape != [N, D_IN] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("X_full.bin shape {x_shape:?} != expected [{N}, {D_IN}]"),
        });
    }
    if y_shape != [N, D_OUT] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("y_full.bin shape {y_shape:?} != expected [{N}, {D_OUT}]"),
        });
    }
    let x = ferrotorch_core::from_vec(x_data, &x_shape)?;
    let y = ferrotorch_core::from_vec(y_data, &y_shape)?;
    Ok((x, y))
}

/// Build the per-epoch `StateDict<f32>` snapshot. Identical to
/// `Module::state_dict()` semantically, but materializes each tensor
/// via `data_vec()` + `from_vec()` so the snapshot is detached from
/// the live (about-to-mutate-again) autograd graph and the optimizer's
/// internal parameter storage. Without this detach a subsequent
/// `optimizer.step()` would mutate the saved tensors in-place because
/// `Parameter::set_data` rewires the underlying Arc.
fn snapshot_state(model: &Mlp) -> FerrotorchResult<StateDict<f32>> {
    let mut out: StateDict<f32> = HashMap::with_capacity(PARAM_KEYS.len());
    for (name, param) in model.named_parameters() {
        let shape = param.shape().to_vec();
        let data = param.data_vec()?;
        let t = ferrotorch_core::from_vec(data, &shape)?;
        out.insert(name, t);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Training loop.
// ---------------------------------------------------------------------------

fn run() -> FerrotorchResult<()> {
    let args = parse_args().map_err(|m| FerrotorchError::InvalidArgument { message: m })?;
    eprintln!(
        "[multi_epoch_train_dump] fixture_dir={} output_dir={}",
        args.fixture_dir.display(),
        args.output_dir.display(),
    );

    std::fs::create_dir_all(&args.output_dir).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!(
            "create output_dir {}: {e}",
            args.output_dir.display()
        ),
    })?;

    // -- Build & load model. ---------------------------------------------
    let mut model = Mlp::new()?;
    load_initial_state(&mut model, &args.fixture_dir.join("initial_state.safetensors"))?;
    model.train();
    eprintln!(
        "[multi_epoch_train_dump] loaded initial_state ({} params)",
        model.named_parameters().len(),
    );

    // -- Build optimizer. ------------------------------------------------
    // The order of params handed to Adam fixes the per-parameter
    // (m, v) state-tensor identity used internally. Using the same
    // `named_parameters()` order the pin script uses (fc1.weight,
    // fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias) keeps
    // the Adam state initialization deterministic across the two
    // runtimes — Adam state itself is keyed by parameter identity,
    // not name, so cross-run determinism is irrelevant in this
    // direction, but it's a useful invariant for debugging.
    let params: Vec<Parameter<f32>> = model
        .named_parameters()
        .iter()
        .map(|(_, p)| (*p).clone())
        .collect();
    let cfg = AdamConfig::default()
        .with_lr(LR)
        .with_betas((0.9, 0.999))
        .with_eps(1e-8);
    let mut opt = Adam::new(params, cfg);

    // -- Load dataset. ---------------------------------------------------
    let (x_full, y_full) = load_dataset(&args.fixture_dir)?;
    eprintln!(
        "[multi_epoch_train_dump] dataset loaded: X={:?} y={:?}",
        x_full.shape(),
        y_full.shape()
    );

    // -- Snapshot initial state (sanity — should match epoch_0). --------
    // The harness only checks epochs 1..=EPOCHS, but writing epoch_0
    // here too lets debugging compare load fidelity directly.
    let initial = snapshot_state(&model)?;
    save_safetensors(&initial, args.output_dir.join("epoch_0_state.safetensors"))?;

    // -- Per-epoch loop. -------------------------------------------------
    let n_batches = N / BATCH;
    // The dataset size and batch size are compile-time constants in
    // this example (they must match the pin script exactly), so this
    // is a `const` assertion rather than a runtime guard. Moving it
    // out of the hot path also keeps the inner loop free of dead
    // branches.
    const _: () = assert!(N % BATCH == 0, "expected N % BATCH == 0 for drop_last semantics");
    let mut epoch_losses: Vec<f64> = Vec::with_capacity(EPOCHS);

    for epoch in 0..EPOCHS {
        let mut epoch_loss_sum: f64 = 0.0;
        for bi in 0..n_batches {
            let start = bi * BATCH;
            // narrow is zero-copy and view-based: x_batch / y_batch
            // are non-leaf views with requires_grad=false (the
            // dataset is loaded outside an autograd context). The
            // model's Parameters are the only requires_grad=true
            // tensors involved, so backward() will populate exactly
            // their .grad slots.
            let x_batch = x_full.narrow(0, start, BATCH)?.contiguous()?;
            let y_batch = y_full.narrow(0, start, BATCH)?.contiguous()?;

            opt.zero_grad()?;
            let pred = model.forward(&x_batch)?;
            let loss = mse_loss(&pred, &y_batch)?;
            loss.backward()?;

            // Accumulate the scalar loss for the per-epoch mean.
            // `loss.item()` would require a scalar; mse_loss with
            // reduction=mean returns a [] tensor.
            let loss_val = no_grad(|| {
                let v = loss.data_vec()?;
                if v.len() != 1 {
                    return Err(FerrotorchError::Internal {
                        message: format!(
                            "expected scalar mse loss, got numel={}",
                            v.len()
                        ),
                    });
                }
                Ok(f64::from(v[0]))
            })?;
            epoch_loss_sum += loss_val;

            opt.step()?;
        }

        let mean_loss = epoch_loss_sum / n_batches as f64;
        epoch_losses.push(mean_loss);
        let snap = snapshot_state(&model)?;
        let out_path = args
            .output_dir
            .join(format!("epoch_{}_state.safetensors", epoch + 1));
        save_safetensors(&snap, &out_path)?;
        eprintln!(
            "[multi_epoch_train_dump] epoch {} loss={:.6}  wrote {}",
            epoch + 1,
            mean_loss,
            out_path.display()
        );
    }

    // -- Verdict JSON for the Python harness. ----------------------------
    let mut s = String::new();
    s.push('{');
    s.push_str(&format!("\"epochs\":{EPOCHS},"));
    s.push_str(&format!("\"batch_size\":{BATCH},"));
    s.push_str(&format!("\"n_samples\":{N},"));
    s.push_str(&format!("\"n_batches_per_epoch\":{n_batches},"));
    s.push_str("\"epoch_losses\":[");
    for (i, l) in epoch_losses.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{l:.10}"));
    }
    s.push(']');
    s.push('}');
    println!("{s}");

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[multi_epoch_train_dump] error: {e}");
        std::process::exit(1);
    }
}
