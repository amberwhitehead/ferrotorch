#!/usr/bin/env python3
"""Verify ferrotorch-data's `DataLoader::iter()` against torch.utils.data
batch trajectories, using the deterministic fixtures pinned at
`ferrotorch/dataloader-batches-v1`.

Phase C.3 of real-artifact-driven development (#1156). Companion to:
  * `scripts/pin_pretrained_dataloader_batches.py` (the pin)
  * `ferrotorch-data/examples/dataloader_iterate_dump.rs`
  * `ferrotorch-data/tests/conformance_dataloader_iteration.rs`

For each (config) tuple this script:

  1. Downloads the per-config subfolder
     (`<config>/meta.json` + `<config>/batch_XXXX.bin` for every batch)
     from the HF mirror via `huggingface_hub.hf_hub_download`.
  2. Invokes the matching Rust example:
       `cargo run -p ferrotorch-data --release --example
        dataloader_iterate_dump -- --config <name>
        --seed 42 --output-dir <tmp>`
  3. Reads the Rust-side per-batch `.bin` dumps and the reference
     per-batch `.bin` dumps shipped by the mirror.
  4. Applies the PASS gate (per Phase C.3 spec):
       - same number of batches
       - sequential configs: ORDER-equality on items
                             (per-item features max_abs <= 1e-6,
                              labels exact)
       - shuffled configs:   SET-equality on items
                             (sorted (label, features-tuple) lists
                              match exactly)

Usage:
  python3 scripts/verify_dataloader_inference.py \
      [--configs sequential,shuffled_seeded,...]
      [--quiet]
      [--self-test]

The Rust example is pre-built by this script on first invocation.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path("/tmp/ferrotorch_verify_dataloader")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_REPO_ID = "ferrotorch/dataloader-batches-v1"

NUM_ITEMS = 10
FEATURE_DIM = 8

# PASS gate for per-item features. Both sides are f32 and the rust side
# rebuilds the dataset deterministically with the same arange-plus-shift
# math the pin script uses, so they are bit-identical. We keep a tiny
# epsilon for cross-platform f32 representation tolerance (in practice
# observed diffs are always exactly 0).
FEATURES_MAX_ABS_CAP = 1e-6

# Matrix of config names. Must match the specs declared in
# pin_pretrained_dataloader_batches.py.
#
# Equality mode per config:
#   ORDER  — rust and torch must produce items in identical order.
#   SET    — rust's multiset of items == torch's multiset of items.
#   SUBSET — rust's items are a no-duplicate subset of the FULL 10-item
#            reference dataset of the expected length. Used only for
#            shuffled+drop_last, where rust and torch drop *different*
#            trailing items because their PRNGs differ.
CONFIGS: list[tuple[str, str]] = [
    # (config_name, equality_mode)
    ("sequential",          "ORDER"),
    ("sequential_droplast", "ORDER"),
    ("shuffled_seeded",     "SET"),
    ("shuffled_droplast",   "SUBSET"),
    ("batch_size_3",        "ORDER"),
]


# ---------------------------------------------------------------------------
# Multi-tensor binary format — mirror of the Rust + Python encoders.
# ---------------------------------------------------------------------------


def read_multi_tensor_f32(path: Path) -> list[np.ndarray]:
    """Inverse of `dump_multi_tensor_f32` in pin script."""
    raw = path.read_bytes()
    off = 0
    if len(raw) < 4:
        raise ValueError(f"{path}: file too short for num_tensors header")
    (n,) = struct.unpack_from("<I", raw, off)
    off += 4
    out: list[np.ndarray] = []
    for ti in range(n):
        if len(raw) < off + 4:
            raise ValueError(f"{path}: truncated reading ndim[{ti}]")
        (ndim,) = struct.unpack_from("<I", raw, off)
        off += 4
        if len(raw) < off + 4 * ndim:
            raise ValueError(f"{path}: truncated reading shape[{ti}]")
        shape = struct.unpack_from(f"<{ndim}I", raw, off)
        off += 4 * ndim
        numel = 1
        for s in shape:
            numel *= int(s)
        if len(raw) < off + 4 * numel:
            raise ValueError(f"{path}: truncated reading data[{ti}]")
        arr = np.frombuffer(raw, dtype="<f4", count=numel, offset=off).reshape(shape)
        off += 4 * numel
        out.append(arr.astype(np.float32, copy=True))
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes after {n} tensors")
    return out


# ---------------------------------------------------------------------------
# Fixture download — pull each config's full file list directly.
# ---------------------------------------------------------------------------


def fetch_fixture(config_name: str) -> tuple[Path, dict[str, Any]]:
    """Download every file the verifier needs into the HF cache and
    return (snapshot_dir, meta_dict). See the docstring on
    `fetch_fixture` in verify_optimizer_inference.py for why we don't
    call `.resolve()`."""
    # meta.json comes first so we can read batch_sizes / num_batches and
    # ask for the right number of batch_XXXX.bin files.
    meta_local = hf_hub_download(repo_id=HF_REPO_ID, filename=f"{config_name}/meta.json")
    meta_path = Path(meta_local).absolute()
    meta = json.loads(meta_path.read_text())
    num_batches = int(meta["num_batches"])

    needed_bins = [f"batch_{i:04d}.bin" for i in range(num_batches)]
    parent: Path = meta_path.parent
    for fn in needed_bins:
        local = hf_hub_download(repo_id=HF_REPO_ID, filename=f"{config_name}/{fn}")
        p = Path(local).absolute()
        if p.parent != parent:
            raise RuntimeError(
                f"{config_name}: HF cached files for the same fixture into "
                f"distinct dirs ({parent} vs {p.parent}); this should not "
                "happen — hf_hub_download materialises per-repo subfolders."
            )
    return parent, meta


# ---------------------------------------------------------------------------
# Cargo example dispatch.
# ---------------------------------------------------------------------------


def build_rust_example_once() -> None:
    """Pre-build the example so per-config invocations don't repeatedly
    invoke cargo's build path."""
    cmd = [
        "cargo", "build", "-p", "ferrotorch-data", "--release",
        "--example", "dataloader_iterate_dump",
    ]
    print(f"  building Rust example once: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"cargo build failed ({proc.returncode})")


def run_rust_dump(
    config_name: str,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        # Clean any stale batch_*.bin files from a previous run so we
        # don't accidentally compare against last invocation's output.
        for p in output_dir.glob("batch_*.bin"):
            p.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cargo", "run", "-q", "-p", "ferrotorch-data", "--release",
        "--example", "dataloader_iterate_dump", "--",
        "--config", config_name,
        "--seed", str(seed),
        "--output-dir", str(output_dir),
    ]
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"rust dump failed for {config_name} ({proc.returncode})")
    json_line: str | None = None
    for line in proc.stdout.splitlines():
        t = line.strip()
        if t.startswith("{") and t.endswith("}"):
            json_line = t
    if json_line is None:
        sys.stderr.write(proc.stdout)
        raise RuntimeError(f"{config_name}: rust dump did not print a JSON verdict")
    return json.loads(json_line)


# ---------------------------------------------------------------------------
# Per-config verdict.
# ---------------------------------------------------------------------------


@dataclass
class ConfigVerdict:
    name: str
    equality_mode: str
    passed: bool
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)


def _flatten_batches(
    batches: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[int, tuple[float, ...]]]:
    """Flatten a list of (features[B, 8], labels[B]) batches into a flat
    list of (label, features) items in iteration order."""
    out: list[tuple[int, tuple[float, ...]]] = []
    for features, labels in batches:
        for i in range(features.shape[0]):
            label = int(labels[i])
            feats = tuple(float(v) for v in features[i].tolist())
            out.append((label, feats))
    return out


def _read_batches(snapshot_dir: Path, num_batches: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load every batch_XXXX.bin and return [(features, labels), ...]."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(num_batches):
        bin_path = snapshot_dir / f"batch_{i:04d}.bin"
        tensors = read_multi_tensor_f32(bin_path)
        if len(tensors) != 2:
            raise ValueError(
                f"{bin_path}: expected 2 tensors (features+labels), got {len(tensors)}"
            )
        features, labels = tensors
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"{bin_path}: features shape {features.shape} != [B, 8]")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"{bin_path}: labels shape {labels.shape} disagrees with features {features.shape}"
            )
        out.append((features, labels))
    return out


def _compare_order(
    ref_items: list[tuple[int, tuple[float, ...]]],
    rust_items: list[tuple[int, tuple[float, ...]]],
) -> tuple[bool, list[str], float]:
    """ORDER-equality comparison. Returns (passed, failure_list, worst_max_abs)."""
    failures: list[str] = []
    worst_max_abs = 0.0
    if len(ref_items) != len(rust_items):
        failures.append(
            f"item count rust={len(rust_items)} != ref={len(ref_items)}"
        )
        return (False, failures, worst_max_abs)
    for idx, ((rlbl, rfeats), (xlbl, xfeats)) in enumerate(zip(ref_items, rust_items)):
        if rlbl != xlbl:
            failures.append(f"item {idx}: label rust={xlbl} != ref={rlbl}")
            continue
        if len(rfeats) != len(xfeats):
            failures.append(
                f"item {idx}: feature dim rust={len(xfeats)} != ref={len(rfeats)}"
            )
            continue
        diffs = [abs(a - b) for a, b in zip(rfeats, xfeats)]
        max_abs = max(diffs) if diffs else 0.0
        if max_abs > worst_max_abs:
            worst_max_abs = max_abs
        if max_abs > FEATURES_MAX_ABS_CAP:
            failures.append(
                f"item {idx}: features max_abs={max_abs:.3e} > {FEATURES_MAX_ABS_CAP:.0e}"
            )
    return (not failures, failures, worst_max_abs)


def _compare_subset(
    rust_items: list[tuple[int, tuple[float, ...]]],
    full_dataset: list[tuple[int, tuple[float, ...]]],
    expected_count: int,
) -> tuple[bool, list[str], float]:
    """SUBSET-validity check used for shuffled+drop_last configs.

    Rust and torch's PRNGs differ, so they drop *different* items under
    drop_last. Instead of comparing rust vs torch's kept items, verify
    the semantic invariant:
      1. rust produced exactly `expected_count` items
      2. each rust item exists in `full_dataset` (label match + features
         within FEATURES_MAX_ABS_CAP)
      3. no duplicates among rust items (each maps to a *distinct* index
         in `full_dataset`)

    Returns (passed, failure_list, worst_max_abs).
    """
    failures: list[str] = []
    worst_max_abs = 0.0

    if len(rust_items) != expected_count:
        failures.append(
            f"item count rust={len(rust_items)} != expected={expected_count}"
        )
        return (False, failures, worst_max_abs)

    consumed: set[int] = set()
    for ri, (rust_label, rust_feats) in enumerate(rust_items):
        # Find a matching unclaimed item in the full dataset.
        match_idx: int | None = None
        match_max_abs = float("inf")
        for di, (full_label, full_feats) in enumerate(full_dataset):
            if di in consumed:
                continue
            if full_label != rust_label:
                continue
            if len(full_feats) != len(rust_feats):
                continue
            diffs = [abs(a - b) for a, b in zip(full_feats, rust_feats)]
            max_abs = max(diffs) if diffs else 0.0
            if max_abs <= FEATURES_MAX_ABS_CAP:
                match_idx = di
                match_max_abs = max_abs
                break
        if match_idx is None:
            failures.append(
                f"rust item {ri} (label={rust_label}) does not appear in full dataset "
                f"(or is a duplicate of an already-consumed item)"
            )
            continue
        consumed.add(match_idx)
        if match_max_abs > worst_max_abs:
            worst_max_abs = match_max_abs

    return (not failures, failures, worst_max_abs)


def _compare_set(
    ref_items: list[tuple[int, tuple[float, ...]]],
    rust_items: list[tuple[int, tuple[float, ...]]],
) -> tuple[bool, list[str], float]:
    """SET-equality comparison. Sorts both lists by (label, features) and
    compares pointwise. Honors the f32 max_abs tolerance.

    Returns (passed, failure_list, worst_max_abs)."""
    failures: list[str] = []
    worst_max_abs = 0.0
    if len(ref_items) != len(rust_items):
        failures.append(
            f"item count rust={len(rust_items)} != ref={len(ref_items)}"
        )
        return (False, failures, worst_max_abs)
    ref_sorted = sorted(ref_items)
    rust_sorted = sorted(rust_items)
    for idx, ((rlbl, rfeats), (xlbl, xfeats)) in enumerate(zip(ref_sorted, rust_sorted)):
        if rlbl != xlbl:
            failures.append(
                f"sorted item {idx}: label rust={xlbl} != ref={rlbl}"
            )
            continue
        diffs = [abs(a - b) for a, b in zip(rfeats, xfeats)]
        max_abs = max(diffs) if diffs else 0.0
        if max_abs > worst_max_abs:
            worst_max_abs = max_abs
        if max_abs > FEATURES_MAX_ABS_CAP:
            failures.append(
                f"sorted item {idx}: features max_abs={max_abs:.3e} > {FEATURES_MAX_ABS_CAP:.0e}"
            )
    return (not failures, failures, worst_max_abs)


def verify_one(config_name: str, equality_mode: str, quiet: bool) -> ConfigVerdict:
    print(f"\n=== {config_name} (equality={equality_mode}) ===", flush=True)

    # -- 1. Fetch fixture. -------------------------------------------------
    fixture_dir, meta = fetch_fixture(config_name)
    expected_num = int(meta["num_batches"])
    expected_sizes = list(meta["batch_sizes"])
    print(f"  fixture: {fixture_dir}")
    print(f"  expected: {expected_num} batches, sizes={expected_sizes}")

    # Sanity-check the meta's declared equality mode matches our table.
    fixture_mode = meta.get("equality_mode")
    if fixture_mode and fixture_mode != equality_mode:
        return ConfigVerdict(
            name=config_name, equality_mode=equality_mode, passed=False,
            summary=(
                f"meta.equality_mode={fixture_mode} disagrees with "
                f"harness-declared {equality_mode}"
            ),
        )

    # -- 2. Run ferrotorch. -----------------------------------------------
    output_dir = CACHE_DIR / f"rust_dl_{config_name}"
    verdict = run_rust_dump(config_name, output_dir, seed=42)
    rust_num = int(verdict["num_batches"])
    rust_sizes = list(verdict["batch_sizes"])
    print(f"  rust:     {rust_num} batches, sizes={rust_sizes}")

    # -- 3. Same-number-of-batches gate. ----------------------------------
    if rust_num != expected_num:
        return ConfigVerdict(
            name=config_name, equality_mode=equality_mode, passed=False,
            summary=f"num_batches rust={rust_num} != ref={expected_num}",
            detail={"rust_verdict": verdict, "expected_num": expected_num},
        )
    # For sequential / non-shuffled configs the per-batch sizes must
    # match exactly. For shuffled configs the sizes also match — torch
    # and rust agree on the *partitioning* of N items into batches; only
    # the *identity* of which item lands in which slot differs. So we
    # check sizes unconditionally.
    if rust_sizes != expected_sizes:
        return ConfigVerdict(
            name=config_name, equality_mode=equality_mode, passed=False,
            summary=f"batch_sizes rust={rust_sizes} != ref={expected_sizes}",
            detail={"rust_verdict": verdict},
        )

    # -- 4. Read both sides' batches. -------------------------------------
    ref_batches = _read_batches(fixture_dir, expected_num)
    rust_batches = _read_batches(output_dir, expected_num)

    ref_items = _flatten_batches(ref_batches)
    rust_items = _flatten_batches(rust_batches)

    # -- 5. Comparison. ---------------------------------------------------
    if equality_mode == "ORDER":
        passed, failures, worst_max_abs = _compare_order(ref_items, rust_items)
    elif equality_mode == "SET":
        passed, failures, worst_max_abs = _compare_set(ref_items, rust_items)
    elif equality_mode == "SUBSET":
        # Build the full reference dataset from meta.json. Required for
        # shuffled+drop_last where torch and rust drop different items.
        full_feats_raw = meta.get("full_dataset_features")
        full_labels_raw = meta.get("full_dataset_labels")
        if full_feats_raw is None or full_labels_raw is None:
            return ConfigVerdict(
                name=config_name, equality_mode=equality_mode, passed=False,
                summary=(
                    "meta.json missing full_dataset_features / full_dataset_labels "
                    "required for SUBSET comparison (regenerate the fixture pin)"
                ),
            )
        full_dataset: list[tuple[int, tuple[float, ...]]] = [
            (int(full_labels_raw[i]), tuple(float(v) for v in full_feats_raw[i]))
            for i in range(len(full_labels_raw))
        ]
        expected_count = sum(expected_sizes)
        passed, failures, worst_max_abs = _compare_subset(
            rust_items, full_dataset, expected_count
        )
    else:
        return ConfigVerdict(
            name=config_name, equality_mode=equality_mode, passed=False,
            summary=f"unknown equality_mode {equality_mode!r}",
        )

    summary = (
        f"{rust_num} batches, sizes={rust_sizes}, "
        f"worst features max_abs={worst_max_abs:.3e}"
    )
    if failures:
        summary += "  — FAIL: " + "; ".join(failures[:3])
        if len(failures) > 3:
            summary += f"  (+{len(failures) - 3} more)"
    if not quiet:
        for bi, ((ref_f, ref_l), (rust_f, rust_l)) in enumerate(zip(ref_batches, rust_batches)):
            print(
                f"  batch {bi}: rust_labels={[int(v) for v in rust_l]} "
                f"ref_labels={[int(v) for v in ref_l]}"
            )
        print(f"  {summary}")

    return ConfigVerdict(
        name=config_name, equality_mode=equality_mode, passed=passed,
        summary=summary,
        detail={
            "rust_verdict": verdict,
            "worst_max_abs": worst_max_abs,
            "num_failures": len(failures),
            "failures": failures[:20],
            "ref_num_batches": expected_num,
            "rust_num_batches": rust_num,
            "ref_batch_sizes": expected_sizes,
            "rust_batch_sizes": rust_sizes,
        },
    )


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--configs", default=",".join(c[0] for c in CONFIGS),
        help="Comma-separated subset of config names to verify.",
    )
    p.add_argument("--quiet", action="store_true",
                   help="Only print the final per-config verdict line.")
    args = p.parse_args()

    requested = [c.strip() for c in args.configs.split(",") if c.strip()]
    by_name = {n: m for (n, m) in CONFIGS}
    for r in requested:
        if r not in by_name:
            print(f"unknown config {r!r}. Known: {list(by_name)}", file=sys.stderr)
            return 2

    build_rust_example_once()

    verdicts: list[ConfigVerdict] = []
    for config_name in requested:
        try:
            v = verify_one(config_name, by_name[config_name], quiet=args.quiet)
        except Exception as e:  # noqa: BLE001
            v = ConfigVerdict(
                name=config_name, equality_mode=by_name[config_name],
                passed=False, summary=f"exception: {e!r}",
                detail={"exception": repr(e)},
            )
        verdicts.append(v)

    print("\n=== VERDICTS ===")
    any_fail = False
    for v in verdicts:
        tag = "PASS" if v.passed else "FAIL"
        if not v.passed:
            any_fail = True
        print(f"{v.name}: {tag} — {v.summary}")

    report = {
        v.name: {
            "equality_mode": v.equality_mode,
            "passed": v.passed,
            "summary": v.summary,
            "detail": v.detail,
        }
        for v in verdicts
    }
    report_path = CACHE_DIR / "verify_dataloader_inference_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    if not args.quiet:
        print(f"\nDetailed report: {report_path}")
    return 1 if any_fail else 0


# ---------------------------------------------------------------------------
# Self-test (small unit tests for the metric helpers + binary roundtrip).
# ---------------------------------------------------------------------------


def _test_read_multi_tensor_f32(tmp: Path) -> None:
    path = tmp / "_self_test_multi.bin"
    tensors_in = [
        np.arange(8, dtype="<f4").reshape(1, 8),
        np.array([0.0], dtype="<f4"),
    ]
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(tensors_in)))
        for arr in tensors_in:
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<I", int(d)))
            f.write(arr.tobytes(order="C"))
    got = read_multi_tensor_f32(path)
    assert len(got) == 2
    assert got[0].shape == (1, 8)
    assert np.allclose(got[0], tensors_in[0])
    assert got[1].shape == (1,)
    print("_test_read_multi_tensor_f32: ok")


def _test_set_vs_order() -> None:
    a = [(0, (0.0, 1.0)), (1, (2.0, 3.0))]
    b = [(1, (2.0, 3.0)), (0, (0.0, 1.0))]
    # Order: a vs b should fail (different first item)
    passed, fails, _ = _compare_order(a, b)
    assert not passed and fails
    # Set: a vs b should pass (same multiset of items)
    passed, fails, _ = _compare_set(a, b)
    assert passed and not fails, f"set should pass but got {fails}"
    # Set: a vs differing items should fail
    c = [(2, (2.0, 3.0)), (0, (0.0, 1.0))]
    passed, fails, _ = _compare_set(a, c)
    assert not passed and fails
    print("_test_set_vs_order: ok")


def _test_subset() -> None:
    full = [
        (0, (0.0, 1.0)),
        (1, (2.0, 3.0)),
        (2, (4.0, 5.0)),
        (0, (6.0, 7.0)),
    ]
    # rust kept 3 items from the full set of 4 — must PASS.
    rust_a = [(2, (4.0, 5.0)), (0, (6.0, 7.0)), (1, (2.0, 3.0))]
    passed, fails, _ = _compare_subset(rust_a, full, expected_count=3)
    assert passed and not fails, f"subset valid case should pass but got {fails}"

    # Wrong count — must FAIL.
    passed, fails, _ = _compare_subset(rust_a, full, expected_count=4)
    assert not passed and fails, "subset wrong-count should fail"

    # Duplicate — rust yields the same item twice; must FAIL.
    rust_dup = [(0, (0.0, 1.0)), (0, (0.0, 1.0))]
    passed, fails, _ = _compare_subset(rust_dup, full, expected_count=2)
    assert not passed and fails, "subset duplicate should fail"

    # Item not in full dataset — must FAIL.
    rust_bad = [(9, (99.0, 99.0))]
    passed, fails, _ = _compare_subset(rust_bad, full, expected_count=1)
    assert not passed and fails, "subset alien-item should fail"
    print("_test_subset: ok")


def _self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _test_read_multi_tensor_f32(Path(td))
    _test_set_vs_order()
    _test_subset()
    print("self-test: all assertions passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(_self_test())
    sys.exit(main())
