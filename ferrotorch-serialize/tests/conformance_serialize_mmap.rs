//! Conformance tests for mmap load variants — ferrotorch-serialize #850.
//!
//! Covers `load_safetensors_mmap`, `load_safetensors_sharded_mmap`, and
//! `load_pytorch_state_dict_mmap`. Each test verifies that the mmap-backed
//! loader produces identical tensor values to its standard (heap-read)
//! counterpart, and that error paths behave correctly.
//!
//! ## Why these tests live separately
//!
//! The mmap variants share the same deserialization logic as their non-mmap
//! counterparts; the only divergence is the I/O layer (memmap2::Mmap vs
//! `read_to_end`). The unit tests in `safetensors_io.rs` already verify mmap
//! parity, but those live in the source module and cannot drive the surface
//! coverage gate. These conformance tests close the #850 gate entry.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::uninlined_format_args
)]

use std::collections::HashMap;

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_nn::StateDict;
use ferrotorch_serialize::{
    load_pytorch_state_dict, load_pytorch_state_dict_mmap, load_safetensors, load_safetensors_mmap,
    load_safetensors_sharded, load_safetensors_sharded_mmap, save_pytorch, save_safetensors,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    let storage = TensorStorage::cpu(data);
    Tensor::from_storage(storage, shape, false).unwrap()
}

fn make_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
    let storage = TensorStorage::cpu(data);
    Tensor::from_storage(storage, shape, false).unwrap()
}

/// Build a minimal `model.safetensors.index.json` + two shard files in `dir`.
///
/// Returns the path to the index file.
fn write_two_shard_fixture(dir: &std::path::Path) -> std::path::PathBuf {
    // Shard A: model.layers.0.weight
    let mut shard_a: StateDict<f32> = HashMap::new();
    shard_a.insert(
        "model.layers.0.weight".to_string(),
        make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
    );
    let shard_a_path = dir.join("model-00001-of-00002.safetensors");
    save_safetensors(&shard_a, &shard_a_path).unwrap();

    // Shard B: model.layers.1.weight
    let mut shard_b: StateDict<f32> = HashMap::new();
    shard_b.insert(
        "model.layers.1.weight".to_string(),
        make_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
    );
    let shard_b_path = dir.join("model-00002-of-00002.safetensors");
    save_safetensors(&shard_b, &shard_b_path).unwrap();

    // Total bytes: 2 * (2*2 tensors * 4 bytes) = 32
    let index_json = r#"{
        "metadata": {"total_size": 32},
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors"
        }
    }"#;
    let index_path = dir.join("model.safetensors.index.json");
    std::fs::write(&index_path, index_json).unwrap();

    index_path
}

// ---------------------------------------------------------------------------
// load_safetensors_mmap — single-file round-trip matches heap load
// ---------------------------------------------------------------------------

/// mmap single-file load matches heap load for an f32 state dict.
#[test]
fn mmap_single_file_f32_matches_heap_load() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.safetensors");

    let mut sd: StateDict<f32> = HashMap::new();
    sd.insert(
        "weight".to_string(),
        make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
    );
    sd.insert("bias".to_string(), make_f32(vec![0.1, 0.2, 0.3], vec![3]));
    save_safetensors(&sd, &path).unwrap();

    let from_heap: StateDict<f32> = load_safetensors(&path).unwrap();
    let from_mmap: StateDict<f32> = load_safetensors_mmap(&path).unwrap();

    assert_eq!(from_heap.len(), from_mmap.len(), "tensor count must match");
    for (name, heap_tensor) in &from_heap {
        let mmap_tensor = from_mmap
            .get(name)
            .unwrap_or_else(|| panic!("mmap result missing key {name:?}"));
        assert_eq!(
            heap_tensor.shape(),
            mmap_tensor.shape(),
            "[{name}] shape mismatch"
        );
        assert_eq!(
            heap_tensor.data().unwrap(),
            mmap_tensor.data().unwrap(),
            "[{name}] values mismatch between heap and mmap load"
        );
    }
}

/// mmap single-file load matches heap load for an f64 state dict.
#[test]
#[allow(clippy::approx_constant)] // 3.14 is an arbitrary round-trip value, not π.
fn mmap_single_file_f64_matches_heap_load() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model_f64.safetensors");

    let mut sd: StateDict<f64> = HashMap::new();
    sd.insert(
        "param".to_string(),
        make_f64(vec![1.0, -2.5, 3.14, 0.0], vec![2, 2]),
    );
    save_safetensors(&sd, &path).unwrap();

    let from_heap: StateDict<f64> = load_safetensors(&path).unwrap();
    let from_mmap: StateDict<f64> = load_safetensors_mmap(&path).unwrap();

    let heap_data = from_heap["param"].data().unwrap();
    let mmap_data = from_mmap["param"].data().unwrap();
    assert_eq!(heap_data.len(), mmap_data.len(), "element count mismatch");
    for (i, (h, m)) in heap_data.iter().zip(mmap_data.iter()).enumerate() {
        assert!((h - m).abs() < 1e-12, "param[{i}]: heap={h}, mmap={m}");
    }
}

/// mmap load of an empty state dict succeeds and returns an empty map.
#[test]
fn mmap_single_file_empty_state_dict() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("empty.safetensors");

    let sd: StateDict<f32> = HashMap::new();
    save_safetensors(&sd, &path).unwrap();

    let from_mmap: StateDict<f32> = load_safetensors_mmap(&path).unwrap();
    assert!(
        from_mmap.is_empty(),
        "mmap load of empty state dict must be empty"
    );
}

/// mmap load returns an error for a nonexistent file.
#[test]
fn mmap_single_file_missing_file_error() {
    let result: Result<StateDict<f32>, _> =
        load_safetensors_mmap("/nonexistent/path/model.safetensors");
    assert!(result.is_err(), "expected error for nonexistent file");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("failed to open"),
        "error message should mention 'failed to open', got: {msg}"
    );
}

/// mmap load returns owned data: after overwriting the underlying file
/// the already-loaded tensors keep their original values.
#[test]
fn mmap_single_file_returns_owned_data() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("owned.safetensors");

    let mut sd: StateDict<f32> = HashMap::new();
    sd.insert(
        "w".to_string(),
        make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
    );
    save_safetensors(&sd, &path).unwrap();

    let loaded = load_safetensors_mmap::<f32>(&path).unwrap();
    let original = loaded["w"].data().unwrap().to_vec();

    // Overwrite the file with garbage — the loaded data must remain valid.
    std::fs::write(&path, b"not a safetensors file at all").unwrap();
    let after_overwrite = loaded["w"].data().unwrap().to_vec();

    assert_eq!(
        original, after_overwrite,
        "mmap-loaded tensor must hold owned data independent of the file"
    );
}

/// mmap load of a multi-tensor file preserves correct shapes.
#[test]
fn mmap_single_file_preserves_shapes() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("shapes.safetensors");

    let mut sd: StateDict<f32> = HashMap::new();
    sd.insert(
        "conv.weight".to_string(),
        make_f32(vec![0.0; 24], vec![2, 3, 2, 2]),
    );
    sd.insert("fc.bias".to_string(), make_f32(vec![1.0; 4], vec![4]));
    save_safetensors(&sd, &path).unwrap();

    let from_mmap: StateDict<f32> = load_safetensors_mmap(&path).unwrap();
    assert_eq!(from_mmap["conv.weight"].shape(), &[2, 3, 2, 2]);
    assert_eq!(from_mmap["fc.bias"].shape(), &[4]);
}

// ---------------------------------------------------------------------------
// load_safetensors_sharded_mmap — sharded round-trip matches heap load
// ---------------------------------------------------------------------------

/// mmap sharded loader produces identical results to the heap sharded loader.
#[test]
fn mmap_sharded_matches_heap_sharded() {
    let tmp = tempfile::tempdir().unwrap();
    let index_path = write_two_shard_fixture(tmp.path());

    let from_heap: StateDict<f32> = load_safetensors_sharded(&index_path).unwrap();
    let from_mmap: StateDict<f32> = load_safetensors_sharded_mmap(&index_path).unwrap();

    assert_eq!(from_heap.len(), from_mmap.len(), "shard count mismatch");
    for (name, heap_tensor) in &from_heap {
        let mmap_tensor = from_mmap
            .get(name)
            .unwrap_or_else(|| panic!("mmap result missing key {name:?}"));
        assert_eq!(
            heap_tensor.data().unwrap(),
            mmap_tensor.data().unwrap(),
            "[{name}] values differ between heap and mmap sharded load"
        );
    }
}

/// mmap sharded loader returns an error for a missing shard file.
#[test]
fn mmap_sharded_missing_shard_error() {
    let tmp = tempfile::tempdir().unwrap();
    let index_json = r#"{
        "metadata": {"total_size": 16},
        "weight_map": {"x": "nonexistent_shard.safetensors"}
    }"#;
    let index_path = tmp.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, index_json).unwrap();

    let result: Result<StateDict<f32>, _> = load_safetensors_sharded_mmap(&index_path);
    assert!(result.is_err(), "expected error for missing shard file");
}

/// mmap sharded loader returns an error for a malformed index file.
#[test]
fn mmap_sharded_malformed_index_error() {
    let tmp = tempfile::tempdir().unwrap();
    let index_path = tmp.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, b"{ not valid json at all }").unwrap();

    let result: Result<StateDict<f32>, _> = load_safetensors_sharded_mmap(&index_path);
    assert!(result.is_err(), "expected error for malformed index");
}

/// mmap sharded loader returns an error when the index references a tensor
/// that is not in the shard file.
#[test]
fn mmap_sharded_index_tensor_not_in_shard_error() {
    let tmp = tempfile::tempdir().unwrap();

    // Write a shard with only "present"
    let mut shard: StateDict<f32> = HashMap::new();
    shard.insert("present".to_string(), make_f32(vec![1.0], vec![1]));
    let shard_path = tmp.path().join("shard.safetensors");
    save_safetensors(&shard, &shard_path).unwrap();

    // Index claims both "present" and "missing" are in the shard
    let index_json = r#"{
        "metadata": {"total_size": 4},
        "weight_map": {
            "present": "shard.safetensors",
            "missing": "shard.safetensors"
        }
    }"#;
    let index_path = tmp.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, index_json).unwrap();

    let result: Result<StateDict<f32>, _> = load_safetensors_sharded_mmap(&index_path);
    assert!(
        result.is_err(),
        "expected error when index tensor is absent from shard"
    );
}

// ---------------------------------------------------------------------------
// load_pytorch_state_dict_mmap — mmap variant matches heap load
// ---------------------------------------------------------------------------

/// mmap pytorch state dict load produces identical tensors to the heap load.
#[test]
fn mmap_pytorch_matches_heap_pytorch_load() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.pt");

    let mut sd: StateDict<f32> = HashMap::new();
    sd.insert(
        "layer.weight".to_string(),
        make_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
    );
    sd.insert("layer.bias".to_string(), make_f32(vec![0.1, 0.2], vec![2]));
    save_pytorch(&sd, &path).unwrap();

    let from_heap: StateDict<f32> = load_pytorch_state_dict(&path).unwrap();
    let from_mmap: StateDict<f32> = load_pytorch_state_dict_mmap(&path).unwrap();

    assert_eq!(from_heap.len(), from_mmap.len(), "tensor count must match");
    for (name, heap_tensor) in &from_heap {
        let mmap_tensor = from_mmap
            .get(name)
            .unwrap_or_else(|| panic!("mmap pytorch result missing key {name:?}"));
        assert_eq!(
            heap_tensor.shape(),
            mmap_tensor.shape(),
            "[{name}] shape mismatch"
        );
        let heap_data = heap_tensor.data().unwrap();
        let mmap_data = mmap_tensor.data().unwrap();
        for (i, (h, m)) in heap_data.iter().zip(mmap_data.iter()).enumerate() {
            assert!((h - m).abs() < 1e-7, "[{name}][{i}]: heap={h}, mmap={m}");
        }
    }
}

/// mmap pytorch load returns an error for a nonexistent file.
#[test]
fn mmap_pytorch_missing_file_error() {
    let result: Result<StateDict<f32>, _> = load_pytorch_state_dict_mmap("/nonexistent/model.pt");
    assert!(
        result.is_err(),
        "expected error for nonexistent pytorch mmap file"
    );
}

/// mmap pytorch load returns owned data — overwriting the file does not
/// corrupt already-loaded tensors.
#[test]
fn mmap_pytorch_returns_owned_data() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("owned.pt");

    let mut sd: StateDict<f32> = HashMap::new();
    sd.insert("p".to_string(), make_f32(vec![10.0, 20.0, 30.0], vec![3]));
    save_pytorch(&sd, &path).unwrap();

    let loaded = load_pytorch_state_dict_mmap::<f32>(&path).unwrap();
    let original = loaded["p"].data().unwrap().to_vec();

    std::fs::write(&path, b"garbage").unwrap();
    let after_overwrite = loaded["p"].data().unwrap().to_vec();

    assert_eq!(
        original, after_overwrite,
        "mmap-loaded pytorch tensor must hold owned data"
    );
}

// ---------------------------------------------------------------------------
// Surface anchors for the coverage gate
// ---------------------------------------------------------------------------

/// Surface anchors: string literals scanned by the Layer 4 coverage gate.
#[test]
fn surface_anchors_mmap() {
    let _ = [
        "load_safetensors_mmap",
        "load_safetensors_sharded_mmap",
        "load_pytorch_state_dict_mmap",
    ];
}
