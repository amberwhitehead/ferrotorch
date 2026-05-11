//! GCN-on-Cora inference-dump binary for the graph real-artifact harness.
//!
//! Companion to `scripts/verify_gnn_inference.py`. The harness writes
//! the graph inputs (`x: [N, F]` f32, `edge_index: [2, E]` i64) to
//! local files in the same `[u32 ndim][u32 × ndim shape][<dtype> data]`
//! format the vision / causal-LM / text-embedding dumps use, then
//! invokes this binary to:
//!
//!   1. Pull `model.safetensors` for `ferrotorch/<model>` into the
//!      ferrotorch hub cache.
//!   2. Read `--x-bin` and `--edge-index-bin` from disk.
//!   3. Build the `GcnNet` with the right dims (`--in-features`,
//!      `--hidden`, `--num-classes`) and load the pinned weights.
//!   4. Forward → dump logits `[N, num_classes]` to `--output`.
//!   5. Print one JSON verdict line to stdout for the harness.
//!
//! Usage:
//! ```text
//! cargo run -p ferrotorch-graph --release --example gcn_inference_dump -- \
//!     --model gcn-cora \
//!     --x-bin /tmp/cora_x.bin \
//!     --edge-index-bin /tmp/cora_edge_index.bin \
//!     --in-features 1433 --hidden 16 --num-classes 7 \
//!     --output /tmp/rust_logits.bin
//! ```

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_graph::load_gcn_net;
use ferrotorch_hub::{HubCache, hf_download_model};

#[derive(Debug)]
struct Args {
    model: String,
    x_bin: PathBuf,
    edge_index_bin: PathBuf,
    output: PathBuf,
    in_features: usize,
    hidden: usize,
    num_classes: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut x_bin: Option<PathBuf> = None;
    let mut edge_index_bin: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut in_features: Option<usize> = None;
    let mut hidden: Option<usize> = None;
    let mut num_classes: Option<usize> = None;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(argv.get(i + 1).ok_or("--model needs a value")?.clone());
                i += 2;
            }
            "--x-bin" => {
                x_bin = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--x-bin needs a value")?,
                ));
                i += 2;
            }
            "--edge-index-bin" => {
                edge_index_bin = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--edge-index-bin needs a value")?,
                ));
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--output needs a value")?,
                ));
                i += 2;
            }
            "--in-features" => {
                in_features = Some(
                    argv.get(i + 1)
                        .ok_or("--in-features needs a value")?
                        .parse()
                        .map_err(|e| format!("--in-features parse: {e}"))?,
                );
                i += 2;
            }
            "--hidden" => {
                hidden = Some(
                    argv.get(i + 1)
                        .ok_or("--hidden needs a value")?
                        .parse()
                        .map_err(|e| format!("--hidden parse: {e}"))?,
                );
                i += 2;
            }
            "--num-classes" => {
                num_classes = Some(
                    argv.get(i + 1)
                        .ok_or("--num-classes needs a value")?
                        .parse()
                        .map_err(|e| format!("--num-classes parse: {e}"))?,
                );
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        x_bin: x_bin.ok_or("--x-bin is required")?,
        edge_index_bin: edge_index_bin.ok_or("--edge-index-bin is required")?,
        output: output.ok_or("--output is required")?,
        in_features: in_features.ok_or("--in-features is required")?,
        hidden: hidden.ok_or("--hidden is required")?,
        num_classes: num_classes.ok_or("--num-classes is required")?,
    })
}

/// Read a `[u32 ndim][u32 × ndim shape][f32 × prod(shape)]` LE blob.
fn read_dump_f32(path: &Path) -> std::io::Result<(Vec<usize>, Vec<f32>)> {
    let mut f = File::open(path)?;
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)?;
    let ndim = u32::from_le_bytes(buf4) as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        f.read_exact(&mut buf4)?;
        shape.push(u32::from_le_bytes(buf4) as usize);
    }
    let n: usize = shape.iter().product();
    let mut data = vec![0.0_f32; n];
    let mut bytes = vec![0u8; n * 4];
    f.read_exact(&mut bytes)?;
    for (i, slot) in data.iter_mut().enumerate() {
        let b = &bytes[i * 4..(i + 1) * 4];
        *slot = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    Ok((shape, data))
}

/// Read a `[u32 ndim][u32 × ndim shape][i64 × prod(shape)]` LE blob.
fn read_dump_i64(path: &Path) -> std::io::Result<(Vec<usize>, Vec<i64>)> {
    let mut f = File::open(path)?;
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)?;
    let ndim = u32::from_le_bytes(buf4) as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        f.read_exact(&mut buf4)?;
        shape.push(u32::from_le_bytes(buf4) as usize);
    }
    let n: usize = shape.iter().product();
    let mut data = vec![0_i64; n];
    let mut bytes = vec![0u8; n * 8];
    f.read_exact(&mut bytes)?;
    for (i, slot) in data.iter_mut().enumerate() {
        let b = &bytes[i * 8..(i + 1) * 8];
        *slot = i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]);
    }
    Ok((shape, data))
}

fn write_dump_f32(path: &Path, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
    let expected: usize = shape.iter().product();
    assert_eq!(
        data.len(),
        expected,
        "data len {} != shape product {}",
        data.len(),
        expected
    );
    let mut f = File::create(path)?;
    f.write_all(&(shape.len() as u32).to_le_bytes())?;
    for &d in shape {
        f.write_all(&(d as u32).to_le_bytes())?;
    }
    let mut buf = Vec::with_capacity(data.len() * 4);
    for &v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)
}

fn run() -> FerrotorchResult<()> {
    let args = parse_args().map_err(|m| FerrotorchError::InvalidArgument { message: m })?;

    // -- 1. Hub-cache the safetensors. --------------------------------------
    let repo = format!("ferrotorch/{}", args.model);
    eprintln!("[gcn_inference_dump] repo = {repo}");
    let cache = HubCache::with_default_dir();
    let repo_dir = hf_download_model(&repo, "main", &cache)?;
    eprintln!(
        "[gcn_inference_dump] cached at {} ({} files)",
        repo_dir.display(),
        std::fs::read_dir(&repo_dir)
            .map(|r| r.count())
            .unwrap_or(0)
    );

    // -- 2. Read x and edge_index from local files. -------------------------
    let (x_shape, x_data) =
        read_dump_f32(&args.x_bin).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed reading {}: {e}", args.x_bin.display()),
        })?;
    if x_shape.len() != 2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("--x-bin must be 2-D [N, F], got {x_shape:?}"),
        });
    }
    let n = x_shape[0];
    let f_in = x_shape[1];
    if f_in != args.in_features {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "--x-bin features dim {f_in} != --in-features {}",
                args.in_features
            ),
        });
    }
    let x = Tensor::<f32>::from_storage(TensorStorage::cpu(x_data), x_shape.clone(), false)?;
    eprintln!("[gcn_inference_dump] x shape = {x_shape:?}");

    let (ei_shape, edge_index) =
        read_dump_i64(&args.edge_index_bin).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed reading {}: {e}", args.edge_index_bin.display()),
        })?;
    if ei_shape.len() != 2 || ei_shape[0] != 2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("--edge-index-bin must be 2-D [2, E], got {ei_shape:?}"),
        });
    }
    let num_edges = ei_shape[1];
    eprintln!(
        "[gcn_inference_dump] edge_index shape = {ei_shape:?} (E={num_edges})"
    );

    // -- 3. Build GcnNet and load pinned weights. ---------------------------
    let weights_path = repo_dir.join("model.safetensors");
    let (net, report) = load_gcn_net(
        &weights_path,
        args.in_features,
        args.hidden,
        args.num_classes,
        /* strict = */ true,
    )?;
    eprintln!(
        "[gcn_inference_dump] loaded weights: unmapped={:?}",
        report.unmapped
    );

    // -- 4. Forward. --------------------------------------------------------
    let logits = net.forward(&x, &edge_index)?;
    let shape = logits.shape().to_vec();
    let data = logits.data_vec()?;
    if shape != vec![n, args.num_classes] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "GcnNet logits shape {shape:?} != [N={n}, num_classes={}]",
                args.num_classes
            ),
        });
    }

    // -- 5. Dump + verdict line. -------------------------------------------
    write_dump_f32(&args.output, &shape, &data).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("failed writing {}: {e}", args.output.display()),
        }
    })?;
    eprintln!(
        "[gcn_inference_dump] wrote {} ({} bytes, shape={shape:?})",
        args.output.display(),
        std::fs::metadata(&args.output)
            .map(|m| m.len())
            .unwrap_or(0)
    );

    let mut out = String::new();
    out.push('{');
    out.push_str(&format!("\"shape\":[{},{}],", shape[0], shape[1]));
    out.push_str(&format!("\"num_nodes\":{},", n));
    out.push_str(&format!("\"num_edges\":{},", num_edges));
    out.push_str(&format!("\"in_features\":{},", args.in_features));
    out.push_str(&format!("\"hidden\":{},", args.hidden));
    out.push_str(&format!("\"num_classes\":{},", args.num_classes));
    out.push_str(&format!("\"unmapped\":{}", report.unmapped.len()));
    out.push('}');
    println!("{out}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[gcn_inference_dump] error: {e}");
        std::process::exit(1);
    }
}
