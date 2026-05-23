//! DataLoader iteration-dump binary for the ferrotorch-data real-artifact
//! parity harness (Phase C.3, #1156).
//!
//! Companion to `scripts/verify_dataloader_inference.py` and the pin
//! script `scripts/pin_pretrained_dataloader_batches.py`. Builds the
//! reference 10-item fixed dict dataset in Rust, constructs a
//! `DataLoader` matching one of the 5 configs in the matrix, iterates it
//! fully, and dumps each batch in the same `[u32 num_tensors=2]` +
//! per-tensor `[u32 ndim][u32*ndim shape][f32 data]` multi-tensor binary
//! format the pin script produces.
//!
//! ## Equality semantics
//!
//! Rust's `rand` crate and torch's `torch.Generator` are *different*
//! PRNGs — they cannot byte-match shuffle permutations. The harness
//! handles this with Option B (the principled default):
//!
//!   * sequential configs (`shuffle=False`) → ORDER-equality on items.
//!   * shuffled configs (`shuffle=True`)    → SET-equality on items.
//!
//! For shuffled configs Rust uses the same `seed=42` it received via
//! `meta.json` for *its own* RNG so the rust run is reproducible, even
//! though it doesn't byte-match torch's permutation.
//!
//! ## Sample type
//!
//! ```text
//! struct Sample {
//!     features: [f32; 8],   // arange(8) + i * 0.1
//!     label: i32,           // i % 3
//! }
//! ```
//!
//! Stored in a `VecDataset<Sample>`; the Rust default DataLoader path
//! does not collate (it yields `Vec<Sample>`), so the example flattens
//! each batch into a `[B, 8]` features tensor + `[B]` labels tensor
//! manually before writing.
//!
//! ## Usage
//!
//! ```text
//! cargo run -p ferrotorch-data --release --example dataloader_iterate_dump -- \
//!   --config shuffled_seeded \
//!   --seed 42 \
//!   --output-dir /tmp/rust_dl_shuffled_seeded
//! ```
//!
//! The `--config` flag selects one of the 5 configs hard-coded below.
//! Seed is optional; when present it forces the ferrotorch DataLoader
//! seed (only meaningful for shuffled configs).

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferrotorch_data::{DataLoader, Dataset, VecDataset};

const NUM_ITEMS: usize = 10;
const FEATURE_DIM: usize = 8;
const NUM_LABELS: i32 = 3;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    config: String,
    seed: Option<u64>,
    output_dir: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut config: Option<String> = None;
    let mut seed: Option<u64> = None;
    let mut output_dir: Option<PathBuf> = None;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--config" => {
                config = Some(argv.get(i + 1).ok_or("--config needs a value")?.clone());
                i += 2;
            }
            "--seed" => {
                let s = argv.get(i + 1).ok_or("--seed needs a value")?;
                seed = Some(s.parse::<u64>().map_err(|e| format!("--seed: {e}"))?);
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
        config: config.ok_or("--config is required")?,
        seed,
        output_dir: output_dir.ok_or("--output-dir is required")?,
    })
}

// ---------------------------------------------------------------------------
// Sample type — mirror of the Python pin script.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Sample {
    features: [f32; FEATURE_DIM],
    label: i32,
}

fn build_dataset() -> VecDataset<Sample> {
    let mut items: Vec<Sample> = Vec::with_capacity(NUM_ITEMS);
    for i in 0..NUM_ITEMS {
        // features[i] = arange(8) + i * 0.1 (as f32). We compute the
        // arange-plus-shift in f32 so bit-for-bit matches the Python
        // pin script's `torch.arange(8, dtype=torch.float32) + i * 0.1`.
        // Concretely: torch.arange returns 0.0, 1.0, ..., 7.0 in f32,
        // then promotes the python float `i * 0.1` to f32 and broadcasts.
        let shift: f32 = (i as f32) * 0.1f32;
        let mut features = [0.0f32; FEATURE_DIM];
        for (j, slot) in features.iter_mut().enumerate() {
            *slot = (j as f32) + shift;
        }
        let label: i32 = (i as i32).rem_euclid(NUM_LABELS);
        items.push(Sample { features, label });
    }
    VecDataset::new(items)
}

// ---------------------------------------------------------------------------
// Config table — mirror of the Python pin script's SPECS list.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LoaderSpec {
    name: &'static str,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

const SPECS: &[LoaderSpec] = &[
    LoaderSpec {
        name: "sequential",
        batch_size: 4,
        shuffle: false,
        drop_last: false,
    },
    LoaderSpec {
        name: "sequential_droplast",
        batch_size: 4,
        shuffle: false,
        drop_last: true,
    },
    LoaderSpec {
        name: "shuffled_seeded",
        batch_size: 4,
        shuffle: true,
        drop_last: false,
    },
    LoaderSpec {
        name: "shuffled_droplast",
        batch_size: 4,
        shuffle: true,
        drop_last: true,
    },
    LoaderSpec {
        name: "batch_size_3",
        batch_size: 3,
        shuffle: false,
        drop_last: false,
    },
];

fn lookup_spec(name: &str) -> Result<&'static LoaderSpec, String> {
    SPECS.iter().find(|s| s.name == name).ok_or_else(|| {
        format!(
            "unknown config {name:?}; known: {:?}",
            SPECS.iter().map(|s| s.name).collect::<Vec<_>>()
        )
    })
}

// ---------------------------------------------------------------------------
// Multi-tensor binary format (mirrors the Python pin script).
// ---------------------------------------------------------------------------

fn write_multi_tensor_f32(path: &Path, tensors: &[(Vec<usize>, Vec<f32>)]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(
        &u32::try_from(tensors.len())
            .expect("num_tensors fits u32")
            .to_le_bytes(),
    )?;
    for (shape, data) in tensors {
        let expect: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expect,
            "tensor data {} disagrees with shape product {}",
            data.len(),
            expect
        );
        f.write_all(
            &u32::try_from(shape.len())
                .expect("ndim fits u32")
                .to_le_bytes(),
        )?;
        for &d in shape {
            f.write_all(&u32::try_from(d).expect("dim fits u32").to_le_bytes())?;
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
// Main flow.
// ---------------------------------------------------------------------------

fn run() -> Result<(), String> {
    let args = parse_args()?;
    eprintln!(
        "[dataloader_iterate_dump] config={} seed={:?} output_dir={}",
        args.config,
        args.seed,
        args.output_dir.display(),
    );

    let spec = lookup_spec(&args.config)?;
    std::fs::create_dir_all(&args.output_dir)
        .map_err(|e| format!("create_dir_all {}: {e}", args.output_dir.display()))?;

    // -- 1. Build the dataset. -----------------------------------------
    let dataset = build_dataset();
    if dataset.len() != NUM_ITEMS {
        return Err(format!(
            "dataset has {} items, expected {NUM_ITEMS}",
            dataset.len()
        ));
    }
    let ds = Arc::new(dataset);

    // -- 2. Build the DataLoader. --------------------------------------
    // prefetch_factor(0) forces the synchronous code path so the
    // iteration order is deterministic and not subject to any
    // prefetch-thread / reorder-buffer interaction. This matches the
    // pattern used throughout `conformance_data_loader.rs`.
    let mut loader = DataLoader::new(Arc::clone(&ds), spec.batch_size)
        .map_err(|e| format!("DataLoader::new failed: {e}"))?
        .shuffle(spec.shuffle)
        .drop_last(spec.drop_last)
        .prefetch_factor(0);
    if let Some(s) = args.seed {
        loader = loader.seed(s);
    }

    // -- 3. Iterate and dump every batch. ------------------------------
    let mut batch_sizes: Vec<usize> = Vec::new();
    let mut batch_count: usize = 0;
    for (bi, batch_result) in loader.iter(0).enumerate() {
        let batch = batch_result.map_err(|e| format!("loader.iter().next() batch {bi}: {e}"))?;
        let b = batch.len();
        if b == 0 {
            return Err(format!("batch {bi}: zero items"));
        }
        batch_sizes.push(b);
        batch_count += 1;

        // Flatten into [B, 8] features + [B] labels.
        let mut features: Vec<f32> = Vec::with_capacity(b * FEATURE_DIM);
        let mut labels: Vec<f32> = Vec::with_capacity(b);
        let mut label_summary: Vec<i32> = Vec::with_capacity(b);
        for s in &batch {
            features.extend_from_slice(&s.features);
            labels.push(s.label as f32);
            label_summary.push(s.label);
        }
        let bin_path = args.output_dir.join(format!("batch_{bi:04}.bin"));
        write_multi_tensor_f32(
            &bin_path,
            &[(vec![b, FEATURE_DIM], features), (vec![b], labels)],
        )
        .map_err(|e| format!("write {}: {e}", bin_path.display()))?;
        eprintln!("[dataloader_iterate_dump]   batch {bi}: size={b}  labels={label_summary:?}");
    }

    if batch_count == 0 {
        return Err("loader produced zero batches".to_string());
    }

    // -- 4. Validate drop_last expectations against the spec. ----------
    let expected: usize = if spec.drop_last {
        NUM_ITEMS / spec.batch_size
    } else {
        NUM_ITEMS.div_ceil(spec.batch_size)
    };
    if batch_count != expected {
        return Err(format!(
            "{}: produced {batch_count} batches but expected {expected} for drop_last={}",
            spec.name, spec.drop_last,
        ));
    }

    // -- 5. JSON verdict line so the Python harness can parse the run. -
    let mut s = String::new();
    s.push('{');
    s.push_str(&format!("\"config\":\"{}\",", spec.name));
    s.push_str(&format!("\"batch_size\":{},", spec.batch_size));
    s.push_str(&format!("\"shuffle\":{},", spec.shuffle));
    s.push_str(&format!("\"drop_last\":{},", spec.drop_last));
    s.push_str(&format!("\"num_batches\":{batch_count},"));
    s.push_str("\"batch_sizes\":[");
    for (i, sz) in batch_sizes.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&sz.to_string());
    }
    s.push_str("],");
    s.push_str(&format!(
        "\"output_dir\":\"{}\"",
        args.output_dir.display().to_string().replace('"', "\\\"")
    ));
    s.push('}');
    println!("{s}");

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[dataloader_iterate_dump] error: {e}");
        std::process::exit(1);
    }
}
