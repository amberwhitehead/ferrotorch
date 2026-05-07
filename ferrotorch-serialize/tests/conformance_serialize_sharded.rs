//! Conformance tests for sharded safetensors loaders — ferrotorch-serialize #851.
//!
//! Covers `load_safetensors_sharded`, `load_safetensors_sharded_filtered`,
//! `load_safetensors_sharded_with_progress`, `load_safetensors_auto` (sharded
//! dispatch), and `ShardProgress` field inspection.
//!
//! ## Fixture strategy
//!
//! Rather than committing binary fixtures to the repo, each test constructs a
//! temporary directory containing two or three shard `.safetensors` files and
//! a corresponding `model.safetensors.index.json`. This matches the HuggingFace
//! layout used in production and is self-contained (no external file dependency).

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
    ShardProgress, load_safetensors, load_safetensors_auto, load_safetensors_sharded,
    load_safetensors_sharded_filtered, load_safetensors_sharded_with_progress, save_safetensors,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    let storage = TensorStorage::cpu(data);
    Tensor::from_storage(storage, shape, false).unwrap()
}

/// Write a safetensors shard to `dir/filename`.
fn write_shard(dir: &std::path::Path, filename: &str, state: &StateDict<f32>) -> std::path::PathBuf {
    let path = dir.join(filename);
    save_safetensors(state, &path).unwrap();
    path
}

/// Write a minimal `model.safetensors.index.json` to `dir/index_name`.
///
/// `shards` is a slice of `(shard_filename, state_dict)` pairs.  Every tensor
/// key in each state dict is mapped to its shard filename in `weight_map`.
fn write_index(
    dir: &std::path::Path,
    index_name: &str,
    shards: &[(&str, &StateDict<f32>)],
) -> std::path::PathBuf {
    use std::fmt::Write as _;

    let mut weight_entries: Vec<(String, String)> = Vec::new();
    let mut total_bytes: u64 = 0;

    for (shard_file, sd) in shards {
        for (key, tensor) in *sd {
            weight_entries.push((key.clone(), (*shard_file).to_string()));
            total_bytes += (tensor.numel() * std::mem::size_of::<f32>()) as u64;
        }
    }

    let mut json = String::from("{\"metadata\":{\"total_size\":");
    json.push_str(&total_bytes.to_string());
    json.push_str("},\"weight_map\":{");
    for (i, (k, v)) in weight_entries.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        write!(json, "\"{k}\":\"{v}\"").unwrap();
    }
    json.push_str("}}");

    let path = dir.join(index_name);
    std::fs::write(&path, json).unwrap();
    path
}

/// Build a standard 3-shard fixture used by several tests.
///
/// Returns `(index_path, shard_a, shard_b, shard_c)` where each shard dict
/// contains its own set of tensors.
fn make_three_shard_fixture(
    dir: &std::path::Path,
) -> (
    std::path::PathBuf,
    StateDict<f32>,
    StateDict<f32>,
    StateDict<f32>,
) {
    let mut shard_a: StateDict<f32> = HashMap::new();
    shard_a.insert(
        "model.embed_tokens.weight".to_string(),
        make_f32(vec![0.1; 12], vec![4, 3]),
    );
    shard_a.insert(
        "model.layers.0.q_proj.weight".to_string(),
        make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
    );

    let mut shard_b: StateDict<f32> = HashMap::new();
    shard_b.insert(
        "model.layers.1.q_proj.weight".to_string(),
        make_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
    );
    shard_b.insert(
        "model.layers.1.k_proj.weight".to_string(),
        make_f32(vec![0.5; 4], vec![2, 2]),
    );

    let mut shard_c: StateDict<f32> = HashMap::new();
    shard_c.insert(
        "lm_head.weight".to_string(),
        make_f32(vec![0.2; 6], vec![2, 3]),
    );

    write_shard(dir, "model-00001-of-00003.safetensors", &shard_a);
    write_shard(dir, "model-00002-of-00003.safetensors", &shard_b);
    write_shard(dir, "model-00003-of-00003.safetensors", &shard_c);

    let index_path = write_index(
        dir,
        "model.safetensors.index.json",
        &[
            ("model-00001-of-00003.safetensors", &shard_a),
            ("model-00002-of-00003.safetensors", &shard_b),
            ("model-00003-of-00003.safetensors", &shard_c),
        ],
    );

    (index_path, shard_a, shard_b, shard_c)
}

// ---------------------------------------------------------------------------
// load_safetensors_sharded — basic round-trip
// ---------------------------------------------------------------------------

/// load_safetensors_sharded merges all shards into one StateDict.
#[test]
fn sharded_load_merges_all_tensors() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, shard_a, shard_b, shard_c) = make_three_shard_fixture(tmp.path());

    let merged: StateDict<f32> = load_safetensors_sharded(&index_path).unwrap();

    let expected_count = shard_a.len() + shard_b.len() + shard_c.len();
    assert_eq!(
        merged.len(),
        expected_count,
        "merged state dict should have {expected_count} tensors"
    );
}

/// load_safetensors_sharded preserves tensor values across shards.
#[test]
fn sharded_load_preserves_values() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, _, _, _) = make_three_shard_fixture(tmp.path());

    let merged: StateDict<f32> = load_safetensors_sharded(&index_path).unwrap();

    // Spot-check values from each shard.
    assert_eq!(
        merged["model.layers.0.q_proj.weight"].data().unwrap(),
        &[1.0_f32, 2.0, 3.0, 4.0],
        "shard A tensor value mismatch"
    );
    assert_eq!(
        merged["model.layers.1.q_proj.weight"].data().unwrap(),
        &[5.0_f32, 6.0, 7.0, 8.0],
        "shard B tensor value mismatch"
    );
    assert_eq!(
        merged["lm_head.weight"].shape(),
        &[2, 3],
        "shard C tensor shape mismatch"
    );
}

/// load_safetensors_sharded preserves tensor shapes.
#[test]
fn sharded_load_preserves_shapes() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, _, _, _) = make_three_shard_fixture(tmp.path());

    let merged: StateDict<f32> = load_safetensors_sharded(&index_path).unwrap();

    assert_eq!(merged["model.embed_tokens.weight"].shape(), &[4, 3]);
    assert_eq!(merged["model.layers.0.q_proj.weight"].shape(), &[2, 2]);
    assert_eq!(merged["model.layers.1.k_proj.weight"].shape(), &[2, 2]);
    assert_eq!(merged["lm_head.weight"].shape(), &[2, 3]);
}

/// load_safetensors_sharded ignores tensors in a shard not listed in the index.
#[test]
fn sharded_load_ignores_extra_tensors_in_shard() {
    let tmp = tempfile::tempdir().unwrap();

    let mut shard: StateDict<f32> = HashMap::new();
    shard.insert("listed".to_string(), make_f32(vec![1.0], vec![1]));
    shard.insert("unlisted".to_string(), make_f32(vec![9.9], vec![1]));
    write_shard(tmp.path(), "shard.safetensors", &shard);

    // Index only lists "listed".
    let index_json = r#"{"metadata":{"total_size":4},"weight_map":{"listed":"shard.safetensors"}}"#;
    let index_path = tmp.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, index_json).unwrap();

    let merged: StateDict<f32> = load_safetensors_sharded(&index_path).unwrap();
    assert_eq!(merged.len(), 1, "only indexed tensors should be loaded");
    assert!(merged.contains_key("listed"));
    assert!(!merged.contains_key("unlisted"), "'unlisted' must be ignored");
}

/// load_safetensors_sharded returns an error when the index names a tensor
/// that is not present in the shard file.
#[test]
fn sharded_load_error_on_missing_index_tensor() {
    let tmp = tempfile::tempdir().unwrap();

    let mut shard: StateDict<f32> = HashMap::new();
    shard.insert("present".to_string(), make_f32(vec![1.0], vec![1]));
    write_shard(tmp.path(), "shard.safetensors", &shard);

    let index_json = r#"{
        "metadata":{"total_size":4},
        "weight_map":{"present":"shard.safetensors","absent":"shard.safetensors"}
    }"#;
    let index_path = tmp.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, index_json).unwrap();

    let result: Result<StateDict<f32>, _> = load_safetensors_sharded(&index_path);
    assert!(result.is_err(), "expected error for tensor absent from shard");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("absent") || msg.contains("missing"),
        "error should name the missing tensor, got: {msg}"
    );
}

/// load_safetensors_sharded returns an error when a shard file is missing.
#[test]
fn sharded_load_error_on_missing_shard_file() {
    let tmp = tempfile::tempdir().unwrap();

    let index_json = r#"{
        "metadata":{"total_size":4},
        "weight_map":{"x":"nonexistent.safetensors"}
    }"#;
    let index_path = tmp.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, index_json).unwrap();

    assert!(
        load_safetensors_sharded::<f32>(&index_path).is_err(),
        "expected error for missing shard file"
    );
}

/// load_safetensors_sharded returns an error for a malformed index JSON.
#[test]
fn sharded_load_error_on_malformed_index() {
    let tmp = tempfile::tempdir().unwrap();
    let index_path = tmp.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, b"{ this is not json").unwrap();

    assert!(
        load_safetensors_sharded::<f32>(&index_path).is_err(),
        "expected error for malformed index JSON"
    );
}

// ---------------------------------------------------------------------------
// load_safetensors_sharded_filtered
// ---------------------------------------------------------------------------

/// load_safetensors_sharded_filtered returns only tensors passing the predicate.
#[test]
fn sharded_filtered_keeps_only_matching_keys() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, _, _, _) = make_three_shard_fixture(tmp.path());

    // Only load tensors whose name contains ".q_proj."
    let filtered: StateDict<f32> =
        load_safetensors_sharded_filtered(&index_path, |k| k.contains(".q_proj.")).unwrap();

    assert_eq!(filtered.len(), 2, "only the two q_proj tensors should be loaded");
    assert!(filtered.contains_key("model.layers.0.q_proj.weight"));
    assert!(filtered.contains_key("model.layers.1.q_proj.weight"));
    assert!(!filtered.contains_key("model.embed_tokens.weight"), "embed must be excluded");
    assert!(!filtered.contains_key("lm_head.weight"), "lm_head must be excluded");
}

/// load_safetensors_sharded_filtered with a predicate matching nothing returns
/// an empty state dict (no error).
#[test]
fn sharded_filtered_no_matches_returns_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, _, _, _) = make_three_shard_fixture(tmp.path());

    let filtered: StateDict<f32> =
        load_safetensors_sharded_filtered(&index_path, |_| false).unwrap();
    assert!(
        filtered.is_empty(),
        "predicate matching nothing must produce empty dict"
    );
}

/// load_safetensors_sharded_filtered with a predicate matching all keys loads
/// the full model.
#[test]
fn sharded_filtered_all_matches_loads_full_model() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, shard_a, shard_b, shard_c) = make_three_shard_fixture(tmp.path());

    let filtered: StateDict<f32> =
        load_safetensors_sharded_filtered(&index_path, |_| true).unwrap();
    let expected = shard_a.len() + shard_b.len() + shard_c.len();
    assert_eq!(
        filtered.len(),
        expected,
        "all-true predicate must return all {expected} tensors"
    );
}

/// load_safetensors_sharded_filtered preserves correct values for matched tensors.
#[test]
fn sharded_filtered_values_match_unfiltered() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, _, _, _) = make_three_shard_fixture(tmp.path());

    let full: StateDict<f32> = load_safetensors_sharded(&index_path).unwrap();
    let filtered: StateDict<f32> =
        load_safetensors_sharded_filtered(&index_path, |k| k.contains("lm_head")).unwrap();

    assert_eq!(filtered.len(), 1);
    assert_eq!(
        filtered["lm_head.weight"].data().unwrap(),
        full["lm_head.weight"].data().unwrap(),
        "filtered tensor data must match full load"
    );
}

// ---------------------------------------------------------------------------
// load_safetensors_sharded_with_progress — ShardProgress inspection
// ---------------------------------------------------------------------------

/// load_safetensors_sharded_with_progress fires the callback once per shard.
#[test]
fn sharded_progress_fires_once_per_shard() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, _, _, _) = make_three_shard_fixture(tmp.path());

    let mut call_count = 0usize;
    let _: StateDict<f32> =
        load_safetensors_sharded_with_progress(&index_path, |_p| {
            call_count += 1;
        })
        .unwrap();

    assert_eq!(call_count, 3, "callback must fire exactly once per shard");
}

/// ShardProgress.shard_index is 0-based and advances correctly.
#[test]
fn sharded_progress_shard_index_is_0_based() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, _, _, _) = make_three_shard_fixture(tmp.path());

    let mut indices: Vec<usize> = Vec::new();
    let _: StateDict<f32> =
        load_safetensors_sharded_with_progress(&index_path, |p| {
            indices.push(p.shard_index);
        })
        .unwrap();

    assert_eq!(indices, vec![0, 1, 2], "shard_index must be 0, 1, 2");
}

/// ShardProgress.shard_count equals the total number of shards.
#[test]
fn sharded_progress_shard_count_is_correct() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, _, _, _) = make_three_shard_fixture(tmp.path());

    let mut counts: Vec<usize> = Vec::new();
    let _: StateDict<f32> =
        load_safetensors_sharded_with_progress(&index_path, |p| {
            counts.push(p.shard_count);
        })
        .unwrap();

    // Every callback must report shard_count = 3.
    assert!(
        counts.iter().all(|&c| c == 3),
        "shard_count must be 3 for every callback, got: {counts:?}"
    );
}

/// ShardProgress.total_tensors equals the total tensor count across all shards.
#[test]
fn sharded_progress_total_tensors_is_correct() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, shard_a, shard_b, shard_c) = make_three_shard_fixture(tmp.path());
    let expected_total = shard_a.len() + shard_b.len() + shard_c.len();

    let mut reported_totals: Vec<usize> = Vec::new();
    let _: StateDict<f32> =
        load_safetensors_sharded_with_progress(&index_path, |p| {
            reported_totals.push(p.total_tensors);
        })
        .unwrap();

    assert!(
        reported_totals.iter().all(|&t| t == expected_total),
        "total_tensors must be {expected_total} for all callbacks, got: {reported_totals:?}"
    );
}

/// ShardProgress.tensors_loaded_so_far is 0 before the first shard and
/// increases monotonically thereafter.
#[test]
fn sharded_progress_tensors_loaded_so_far_is_monotonic() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, _, _, _) = make_three_shard_fixture(tmp.path());

    let mut loaded_so_far: Vec<usize> = Vec::new();
    let _: StateDict<f32> =
        load_safetensors_sharded_with_progress(&index_path, |p| {
            loaded_so_far.push(p.tensors_loaded_so_far);
        })
        .unwrap();

    // First callback fires before any shard is loaded.
    assert_eq!(loaded_so_far[0], 0, "no tensors loaded before first shard");

    // Subsequent callbacks must be non-decreasing.
    for i in 1..loaded_so_far.len() {
        assert!(
            loaded_so_far[i] >= loaded_so_far[i - 1],
            "tensors_loaded_so_far must be non-decreasing: {:?}",
            loaded_so_far
        );
    }
}

/// ShardProgress.shard_file is non-empty for every callback.
#[test]
fn sharded_progress_shard_file_is_nonempty() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, _, _, _) = make_three_shard_fixture(tmp.path());

    let mut files: Vec<String> = Vec::new();
    let _: StateDict<f32> =
        load_safetensors_sharded_with_progress(&index_path, |p| {
            files.push(p.shard_file.to_string());
        })
        .unwrap();

    for f in &files {
        assert!(!f.is_empty(), "shard_file must not be empty");
        assert!(
            f.ends_with(".safetensors"),
            "shard_file must end with .safetensors, got: {f}"
        );
    }
}

/// The result of load_safetensors_sharded_with_progress matches the result
/// of load_safetensors_sharded (same tensors, same values).
#[test]
fn sharded_progress_result_matches_plain_sharded_load() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, _, _, _) = make_three_shard_fixture(tmp.path());

    let plain: StateDict<f32> = load_safetensors_sharded(&index_path).unwrap();
    let with_progress: StateDict<f32> =
        load_safetensors_sharded_with_progress(&index_path, |_| {}).unwrap();

    assert_eq!(plain.len(), with_progress.len(), "tensor count must match");
    for (name, plain_tensor) in &plain {
        let prog_tensor = with_progress
            .get(name)
            .unwrap_or_else(|| panic!("progress result missing key {name:?}"));
        assert_eq!(
            plain_tensor.data().unwrap(),
            prog_tensor.data().unwrap(),
            "[{name}] values differ between plain and progress-callback load"
        );
    }
}

/// ShardProgress exposes tensors_in_shard > 0 for every shard.
#[test]
fn sharded_progress_tensors_in_shard_is_positive() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, _, _, _) = make_three_shard_fixture(tmp.path());

    let mut tensors_in_shard: Vec<usize> = Vec::new();
    let _: StateDict<f32> =
        load_safetensors_sharded_with_progress(&index_path, |p: ShardProgress<'_>| {
            tensors_in_shard.push(p.tensors_in_shard);
        })
        .unwrap();

    for (i, &count) in tensors_in_shard.iter().enumerate() {
        assert!(
            count > 0,
            "shard {i} must have at least one tensor, got tensors_in_shard=0"
        );
    }
}

// ---------------------------------------------------------------------------
// load_safetensors_auto — dispatch on filename
// ---------------------------------------------------------------------------

/// load_safetensors_auto dispatches to the single-file loader for a .safetensors
/// path and returns the correct tensors.
#[test]
fn auto_single_file_dispatches_correctly() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.safetensors");

    let mut sd: StateDict<f32> = HashMap::new();
    sd.insert("w".to_string(), make_f32(vec![1.0, 2.0, 3.0], vec![3]));
    save_safetensors(&sd, &path).unwrap();

    let from_auto: StateDict<f32> = load_safetensors_auto(&path).unwrap();
    let from_direct: StateDict<f32> = load_safetensors(&path).unwrap();

    assert_eq!(from_auto.len(), from_direct.len(), "auto single-file tensor count mismatch");
    assert_eq!(
        from_auto["w"].data().unwrap(),
        from_direct["w"].data().unwrap(),
        "auto single-file values mismatch"
    );
}

/// load_safetensors_auto dispatches to the sharded loader for a .index.json
/// path and returns the full merged state dict.
#[test]
fn auto_sharded_dispatches_correctly() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, shard_a, shard_b, shard_c) = make_three_shard_fixture(tmp.path());

    let from_auto: StateDict<f32> = load_safetensors_auto(&index_path).unwrap();
    let expected_count = shard_a.len() + shard_b.len() + shard_c.len();

    assert_eq!(
        from_auto.len(),
        expected_count,
        "auto sharded dispatch must load all {expected_count} tensors"
    );
}

/// load_safetensors_auto sharded result matches load_safetensors_sharded result.
#[test]
fn auto_sharded_matches_explicit_sharded_load() {
    let tmp = tempfile::tempdir().unwrap();
    let (index_path, _, _, _) = make_three_shard_fixture(tmp.path());

    let from_auto: StateDict<f32> = load_safetensors_auto(&index_path).unwrap();
    let from_explicit: StateDict<f32> = load_safetensors_sharded(&index_path).unwrap();

    assert_eq!(from_auto.len(), from_explicit.len());
    for (name, auto_tensor) in &from_auto {
        let explicit_tensor = from_explicit
            .get(name)
            .unwrap_or_else(|| panic!("explicit load missing key {name:?}"));
        assert_eq!(
            auto_tensor.data().unwrap(),
            explicit_tensor.data().unwrap(),
            "[{name}] auto vs explicit sharded mismatch"
        );
    }
}

/// load_safetensors_auto returns an error for a missing path.
#[test]
fn auto_missing_file_error() {
    let result: Result<StateDict<f32>, _> =
        load_safetensors_auto("/nonexistent/model.safetensors");
    assert!(result.is_err(), "expected error for nonexistent file");
}

// ---------------------------------------------------------------------------
// Surface anchors for the coverage gate
// ---------------------------------------------------------------------------

/// Surface anchors: string literals scanned by the Layer 4 coverage gate.
#[test]
fn surface_anchors_sharded() {
    let _ = [
        "load_safetensors_sharded",
        "load_safetensors_sharded_filtered",
        "load_safetensors_sharded_with_progress",
        "load_safetensors_auto",
        "ShardProgress",
    ];
}
