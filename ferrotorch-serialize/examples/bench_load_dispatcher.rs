//! Synthetic benchmark for the `load_safetensors` dtype-aware
//! dispatcher (ferrotorch#1178).
//!
//! Writes a multi-tensor BF16 safetensors file (~512 MB on disk), then
//! times both code paths against the same artifact:
//!
//!   1. Default dispatch (parallel for BF16, this hardware).
//!   2. Force-serial via `FERROTORCH_FORCE_SERIAL_LOAD=1`.
//!
//! Run:
//!
//! ```bash
//! cargo run --release --example bench_load_dispatcher -p ferrotorch-serialize
//! ```
//!
//! Prints per-path wall-clock and the parallel/serial ratio. Not a
//! committed regression test (timing depends on hardware); kept as
//! a manual operator-driven measurement so the next dispatcher
//! tuning starts from real numbers.

use std::path::PathBuf;
use std::time::Instant;

use std::error::Error;

use ferrotorch_nn::StateDict;
use ferrotorch_serialize::load_safetensors;
use safetensors::tensor::{Dtype, TensorView};

const N_TENSORS: usize = 600;
const ELEMS_PER_TENSOR: usize = 256 * 1024; // 256K bf16 → 512 KB → ~300 MB total

fn main() -> Result<(), Box<dyn Error>> {
    // Generate ~300 MB of BF16 data spread across 600 tensors —
    // representative of a real Llama / Mistral checkpoint's tensor
    // count and per-tensor size distribution. Values are arbitrary;
    // we only care about wall-clock decode time.
    let mut tensors: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::with_capacity(N_TENSORS);
    let mut buf: Vec<u8> = Vec::with_capacity(ELEMS_PER_TENSOR * 2);
    for i in 0..N_TENSORS {
        buf.clear();
        for j in 0..ELEMS_PER_TENSOR {
            // Spread BF16 patterns across the value range so the upcast
            // exercises every code path in `half::bf16::to_f32`.
            let bits: u16 = ((i.wrapping_mul(0xa39c) + j) as u16) | 0x3F00;
            buf.extend_from_slice(&bits.to_le_bytes());
        }
        tensors.push((format!("t{i:04}"), buf.clone(), vec![ELEMS_PER_TENSOR]));
    }

    let tmp = tempfile::tempdir()?;
    let path: PathBuf = tmp.path().join("bench.safetensors");

    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, &None, &path)?;
    let on_disk = std::fs::metadata(&path)?.len();
    println!(
        "wrote {} ({} tensors, {:.1} MB on disk)",
        path.display(),
        N_TENSORS,
        on_disk as f64 / (1024.0 * 1024.0),
    );

    // Path A: default dispatch (parallel for BF16).
    // SAFETY: single-threaded benchmark, the env var manipulation
    // is local to this process.
    unsafe {
        std::env::remove_var("FERROTORCH_FORCE_SERIAL_LOAD");
    }
    let t0 = Instant::now();
    let parallel: StateDict<f32> = load_safetensors(&path)?;
    let parallel_elapsed = t0.elapsed();
    assert_eq!(parallel.len(), N_TENSORS);
    println!("parallel: {:.3} s", parallel_elapsed.as_secs_f64());
    drop(parallel);

    // Path B: force-serial.
    unsafe {
        std::env::set_var("FERROTORCH_FORCE_SERIAL_LOAD", "1");
    }
    let t0 = Instant::now();
    let serial: StateDict<f32> = load_safetensors(&path)?;
    let serial_elapsed = t0.elapsed();
    assert_eq!(serial.len(), N_TENSORS);
    unsafe {
        std::env::remove_var("FERROTORCH_FORCE_SERIAL_LOAD");
    }
    println!("serial:   {:.3} s", serial_elapsed.as_secs_f64());

    let speedup = serial_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64();
    println!("speedup:  {speedup:.2}× (parallel wins iff >1, regresses iff <1)");
    Ok(())
}
