#!/usr/bin/env python3
"""Pin a deterministic torch.utils.data.DataLoader batch sequence to the
`ferrotorch/dataloader-batches-v1` HF mirror.

Phase C.3 of real-artifact-driven development (#1156): build a reference
batch sequence for every (DataLoader config) tuple in the matrix, freeze
the exact items per batch, and ship one .bin per batch + a per-config
index manifest. ferrotorch-data's `DataLoader` can then be byte-compared
against torch.utils.data's iteration without re-running the upstream
iterator at verification time.

Why **batch-by-batch pinning** instead of one big tensor:

* Each batch is a distinct .bin so the harness can quickly diff
  per-batch features+labels; partial-final-batch (drop_last=False) shows
  up as a smaller last file.
* Per-config manifest records batch count, batch sizes, and which
  config produced it — drop_last regressions surface immediately.
* PRNG difference between rust `rand` and torch is honest: for shuffled
  configs the harness compares SET-equality (sorted lists of items) and
  not ORDER-equality. For shuffled+drop_last specifically, even
  SET-equality cannot hold (each side drops a *different* trailing
  partial batch under its own PRNG), so the harness instead asserts
  SUBSET-validity against the full 10-item ground truth.

Dataset (fixed; 10 items, dict-style):
    features[i] = arange(8) + i * 0.1       (f32, shape [8])
    label[i]    = i % 3                     (i32)

Configurations (5 total):
    sequential          batch_size=4 shuffle=False drop_last=False
    sequential_droplast batch_size=4 shuffle=False drop_last=True
    shuffled_seeded     batch_size=4 shuffle=True  drop_last=False  seed=42
    shuffled_droplast   batch_size=4 shuffle=True  drop_last=True   seed=42
    batch_size_3        batch_size=3 shuffle=False drop_last=False

Per config the pin emits one subfolder:
  * meta.json              — config + num_batches + per-batch sizes
  * batch_0000.bin         — items in batch 0
  * batch_0001.bin         — items in batch 1
  * ...

Multi-tensor binary layout per batch (little-endian):
  [u32 num_tensors=2]
  tensor 0: features  ndim=2  shape=[B, 8]   f32 data
  tensor 1: labels    ndim=1  shape=[B]      f32 data (label-as-f32)

Then everything is bundled into a single `bundle.tar` artifact (one HF
subfolder per config) and uploaded to `ferrotorch/dataloader-batches-v1`.

Usage:
  python3 scripts/pin_pretrained_dataloader_batches.py \
      [--out-dir /tmp/ferrotorch_dataloader_batches] \
      [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import tarfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import HfApi
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Configuration matrix — 5 configs total.
# ---------------------------------------------------------------------------


@dataclass
class LoaderSpec:
    """One DataLoader config to pin."""

    name: str
    batch_size: int
    shuffle: bool
    drop_last: bool
    seed: int | None  # None = no manual_seed call; int = torch.manual_seed before iter


SPECS: list[LoaderSpec] = [
    LoaderSpec("sequential",          batch_size=4, shuffle=False, drop_last=False, seed=None),
    LoaderSpec("sequential_droplast", batch_size=4, shuffle=False, drop_last=True,  seed=None),
    LoaderSpec("shuffled_seeded",     batch_size=4, shuffle=True,  drop_last=False, seed=42),
    LoaderSpec("shuffled_droplast",   batch_size=4, shuffle=True,  drop_last=True,  seed=42),
    LoaderSpec("batch_size_3",        batch_size=3, shuffle=False, drop_last=False, seed=None),
]

NUM_ITEMS = 10
FEATURE_DIM = 8
NUM_LABELS = 3  # label = i % 3
HF_REPO_ID = "ferrotorch/dataloader-batches-v1"


# ---------------------------------------------------------------------------
# Fixed dataset — deterministic.
# ---------------------------------------------------------------------------


class FixedDictDataset(Dataset):
    """The reference 10-item dict-style dataset.

    Sample i = {
        'features': torch.tensor([0, 1, ..., 7], dtype=f32) + i * 0.1
        'label': int(i % 3)
    }

    The rust side replicates these exact f32 values bitwise.
    """

    def __len__(self) -> int:
        return NUM_ITEMS

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not 0 <= idx < NUM_ITEMS:
            raise IndexError(idx)
        features = torch.arange(FEATURE_DIM, dtype=torch.float32) + float(idx) * 0.1
        label = int(idx % NUM_LABELS)
        return {"features": features, "label": label}


# ---------------------------------------------------------------------------
# Multi-tensor binary format — mirrors pin_pretrained_optimizer_trajectories.py.
# ---------------------------------------------------------------------------


def dump_multi_tensor_f32(path: Path, tensors: list[np.ndarray]) -> None:
    """Write a `[u32 num_tensors]` + per-tensor `[u32 ndim][u32 shape][f32]`
    little-endian dump."""
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(tensors)))
        for arr in tensors:
            arr32 = np.ascontiguousarray(arr, dtype="<f4")
            shape = list(arr32.shape)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", int(d)))
            f.write(arr32.tobytes(order="C"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Batch dump per spec.
# ---------------------------------------------------------------------------


def iterate_and_dump(spec: LoaderSpec, out_dir: Path) -> dict[str, Any]:
    """Build a torch DataLoader matching `spec`, iterate fully, and dump
    each batch as a 2-tensor (features, labels) .bin file. Returns
    metadata describing the pinned trajectory."""
    print(f"\n=== {spec.name} (batch_size={spec.batch_size}, "
          f"shuffle={spec.shuffle}, drop_last={spec.drop_last}, "
          f"seed={spec.seed}) ===", flush=True)

    dataset = FixedDictDataset()

    # If shuffled, seed the global torch RNG; torch.utils.data.DataLoader
    # uses the default generator (`torch.Generator()` with global seed)
    # when shuffle=True and no explicit generator is supplied. Setting
    # torch.manual_seed(42) immediately before constructing the loader
    # makes the permutation reproducible.
    if spec.seed is not None:
        torch.manual_seed(spec.seed)

    loader = DataLoader(
        dataset,
        batch_size=spec.batch_size,
        shuffle=spec.shuffle,
        drop_last=spec.drop_last,
        num_workers=0,
    )

    batch_sizes: list[int] = []
    item_features: list[list[list[float]]] = []  # per-batch [B][8]
    item_labels: list[list[int]] = []            # per-batch [B]

    for bi, batch in enumerate(loader):
        # default_collate stacks features into shape [B, 8] and labels
        # into shape [B] (int64). Convert labels to f32 (lossless for
        # values 0..NUM_LABELS-1).
        features = batch["features"].detach().cpu().numpy().astype(np.float32, copy=True)
        labels = batch["label"].detach().cpu().numpy().astype(np.float32, copy=True)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise RuntimeError(
                f"{spec.name}: batch {bi} features have unexpected shape {features.shape}"
            )
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise RuntimeError(
                f"{spec.name}: batch {bi} labels shape {labels.shape} "
                f"disagrees with features {features.shape}"
            )

        bin_path = out_dir / f"batch_{bi:04d}.bin"
        dump_multi_tensor_f32(bin_path, [features, labels])
        batch_sizes.append(int(features.shape[0]))
        item_features.append(features.tolist())
        item_labels.append([int(v) for v in labels])
        print(f"  batch {bi}: size={features.shape[0]}  "
              f"labels={[int(v) for v in labels]}  "
              f"features[:, 0]={features[:, 0].tolist()}")

    if not batch_sizes:
        raise RuntimeError(f"{spec.name}: loader produced zero batches")

    # Validate the expected drop_last shape.
    expected_num_batches: int
    if spec.drop_last:
        expected_num_batches = NUM_ITEMS // spec.batch_size
    else:
        expected_num_batches = (NUM_ITEMS + spec.batch_size - 1) // spec.batch_size
    if len(batch_sizes) != expected_num_batches:
        raise RuntimeError(
            f"{spec.name}: produced {len(batch_sizes)} batches but expected "
            f"{expected_num_batches} for drop_last={spec.drop_last}"
        )

    # Equality semantics:
    #   * sequential (no shuffle):    ORDER (rust must match torch order)
    #   * shuffled, no drop_last:     SET   (rust's items == torch's items, any order)
    #   * shuffled + drop_last:       SUBSET (rust's items are a no-duplicate
    #                                         subset of the FULL 10-item
    #                                         dataset, of the expected
    #                                         length — torch and rust drop
    #                                         different items because their
    #                                         PRNGs differ)
    if not spec.shuffle:
        equality_mode = "ORDER"
    elif spec.drop_last:
        equality_mode = "SUBSET"
    else:
        equality_mode = "SET"

    # Reference dataset: the full 10-item ground truth, so the verifier
    # can do SUBSET checks for shuffled+drop_last configs without needing
    # to reconstruct the dataset itself.
    full_dataset_features: list[list[float]] = []
    full_dataset_labels: list[int] = []
    for i in range(NUM_ITEMS):
        sample = dataset[i]
        full_dataset_features.append(
            sample["features"].detach().cpu().numpy().astype(np.float32).tolist()
        )
        full_dataset_labels.append(int(sample["label"]))

    meta = {
        "name": spec.name,
        "batch_size": spec.batch_size,
        "shuffle": spec.shuffle,
        "drop_last": spec.drop_last,
        "seed": spec.seed,
        "num_items": NUM_ITEMS,
        "feature_dim": FEATURE_DIM,
        "num_batches": len(batch_sizes),
        "batch_sizes": batch_sizes,
        "expected_num_batches": expected_num_batches,
        # Pinned ground truth — used by the verifier to compute SET
        # vs. ORDER (vs. SUBSET) equality against rust's output.
        "item_features": item_features,
        "item_labels": item_labels,
        # Full 10-item reference dataset; needed for SUBSET checks where
        # torch and rust drop different items.
        "full_dataset_features": full_dataset_features,
        "full_dataset_labels": full_dataset_labels,
        "dtype": "float32",
        "torch_version": torch.__version__,
        "format": (
            "Each batch_XXXX.bin file is `[u32 num_tensors=2]` followed by "
            "two per-tensor records `[u32 ndim][u32 shape][f32 data]` "
            "(little-endian). Tensor 0: features [B, 8]. Tensor 1: labels "
            "[B] (label-as-f32). batch_sizes[] records each batch's B."
        ),
        "equality_mode": equality_mode,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# Bundle + upload.
# ---------------------------------------------------------------------------


def write_readme(out_root: Path, metas: list[dict[str, Any]]) -> None:
    """Write the bundle-level README.md describing the artifact set."""
    config_lines = []
    for m in metas:
        config_lines.append(
            f"  * `{m['name']}` — batch_size={m['batch_size']} "
            f"shuffle={m['shuffle']} drop_last={m['drop_last']} "
            f"seed={m['seed']} → {m['num_batches']} batches "
            f"(equality_mode={m['equality_mode']})"
        )
    readme = textwrap.dedent(f"""
        ---
        license: apache-2.0
        tags:
        - test-fixtures
        - dataloader
        - pytorch
        ---

        # ferrotorch / dataloader-batches-v1

        DataLoader-iteration parity fixtures for ferrotorch's
        `DataLoader::iter()` implementation, generated by iterating
        `torch.utils.data.DataLoader` over a deterministic 10-item dict
        dataset and snapshotting every batch as a `.bin` file.

        Phase C.3 of real-artifact-driven development (#1156). Companion to:
          * `scripts/pin_pretrained_dataloader_batches.py` (this pin)
          * `scripts/verify_dataloader_inference.py` (the harness)
          * `ferrotorch-data/examples/dataloader_iterate_dump.rs`
          * `ferrotorch-data/tests/conformance_dataloader_iteration.rs`

        ## Dataset

        Fixed, deterministic, 10 items:

        ```
        item[i] = {{
            "features": arange(8, dtype=f32) + i * 0.1   # shape [8]
            "label": i % 3                                # int
        }}
        ```

        ## Configurations

        {chr(10).join(config_lines)}

        ## Layout

        One subfolder per configuration:

        ```
        <config_name>/
          meta.json
          batch_0000.bin
          batch_0001.bin
          batch_NNNN.bin    # one file per batch, count recorded in meta.json
        ```

        ## Binary format

        Each `.bin` file is a little-endian multi-tensor dump:

        ```
        [u32 num_tensors=2]
        tensor 0 (features):
          [u32 ndim=2] [u32 B] [u32 8] [f32 * B*8]
        tensor 1 (labels):
          [u32 ndim=1] [u32 B] [f32 * B]   # label-as-f32
        ```

        ## Equality semantics

        * Sequential (`shuffle=False`) configs: ORDER-equality. Rust and
          torch must yield items in identical order.
        * Shuffled (`shuffle=True`), `drop_last=False`: SET-equality. Rust's
          `rand` crate and torch's `torch.Generator` are different PRNGs,
          so the shuffle permutations cannot byte-match. The verifier
          requires that the multiset of items is identical.
        * Shuffled + `drop_last=True`: SUBSET-equality. With drop_last each
          side drops the trailing partial batch; because torch and rust
          permute differently the *kept* items differ as well, so the
          verifier checks that rust's kept items are a no-duplicate subset
          of the full 10-item dataset (encoded in `meta.json` as
          `full_dataset_features` / `full_dataset_labels`) of the
          expected length.

        ## License

        Apache 2.0. Synthetic fixtures generated by this repo's pin
        script; no upstream weights / data.
    """).strip()
    (out_root / "README.md").write_text(readme)


def hf_upload(out_root: Path) -> None:
    api = HfApi()
    print(f"\nuploading to https://huggingface.co/{HF_REPO_ID} ...", flush=True)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(out_root),
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="feat: pin DataLoader iteration fixtures v1 (#1156)",
    )
    print("upload complete.", flush=True)


def build_bundle(out_root: Path) -> Path:
    """Write a single `bundle.tar` so registry.rs can checksum one
    artifact. The verify script downloads individual files via
    `hf_hub_download` and does not consume this tar."""
    tar_path = out_root / "bundle.tar"
    with tarfile.open(tar_path, "w") as tar:
        for sub in sorted(out_root.iterdir()):
            if sub.is_dir():
                tar.add(sub, arcname=sub.name)
    return tar_path


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default="/tmp/ferrotorch_dataloader_batches",
        help="Staging directory.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Stage everything locally but do not upload to HF.",
    )
    p.add_argument(
        "--only", default="",
        help="Comma-separated subset of config names to regenerate (debug).",
    )
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    specs = [s for s in SPECS if not only or s.name in only]
    if not specs:
        print("no specs match --only filter", file=sys.stderr)
        return 2

    metas: list[dict[str, Any]] = []
    for spec in specs:
        sub = out_root / spec.name
        sub.mkdir(parents=True, exist_ok=True)
        metas.append(iterate_and_dump(spec, sub))

    write_readme(out_root, metas)
    bundle_path = build_bundle(out_root)
    bundle_sha = sha256_of(bundle_path)

    if not args.dry_run:
        hf_upload(out_root)

    print("\n=== SUMMARY ===")
    for m in metas:
        print(f"  {m['name']:24s} batches={m['num_batches']}  "
              f"sizes={m['batch_sizes']}  equality={m['equality_mode']}")
    print(f"\nlocal stage:   {out_root}")
    print(f"bundle:        {bundle_path}")
    print(f"bundle sha256: {bundle_sha}")
    print(f"hf:            https://huggingface.co/{HF_REPO_ID}")

    print("\n=== Drop-in registry pin (for ferrotorch-hub/src/registry.rs) ===")
    print('  ModelInfo {')
    print('      name: "dataloader-batches-v1",')
    print('      description: "DataLoader iteration parity fixtures: 5 torch.utils.data configs (sequential, sequential_droplast, shuffled_seeded, shuffled_droplast, batch_size_3) over a fixed 10-item dict dataset — Phase C.3 real-artifact harness baseline (#1156).",')
    print(f'      weights_url: "https://huggingface.co/{HF_REPO_ID}/resolve/main/bundle.tar",')
    print(f'      weights_sha256: "{bundle_sha}",')
    print('      format: WeightsFormat::FerrotorchStateDict,')
    print('      num_parameters: 0,')
    print('  },')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
