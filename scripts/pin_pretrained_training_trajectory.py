#!/usr/bin/env python3
"""Pin a multi-epoch training-trajectory fixture to the
`ferrotorch/training-trajectory-v1` HF mirror.

Phase E of real-artifact-driven development (#1161): exercise the full
training stack — forward (linear + relu) + loss (MSE mean) + backward
(autograd) + optimizer (Adam) + DataLoader iteration (sequential,
batch_size=4, drop_last=False) over multiple epochs — against a fixed
deterministic dataset, and snapshot the per-epoch state_dict so
ferrotorch can byte-compare its own end-to-end training trajectory.

Reuses the same MLP architecture as #1155 (the optimizer-trajectory
pin) for maximum code reuse: Linear(64 -> 32) -> ReLU
-> Linear(32 -> 16) -> ReLU -> Linear(16 -> 8).

Per-epoch outputs (epoch index 0 = initial state, 1..5 = post-epoch):
  * epoch_0_state.safetensors    — initial parameters, before training
  * epoch_1_state.safetensors    — parameters after epoch 1
  * ...
  * epoch_5_state.safetensors    — parameters after epoch 5
  * initial_state.safetensors    — alias of epoch_0_state (loader
                                   convenience for the Rust example)
  * X_full.bin / y_full.bin      — full dataset, multi-tensor f32
  * meta.json                    — hyperparameters + per-epoch losses

Multi-tensor binary format (little-endian) for X_full / y_full:
  [u32 num_tensors=1]
  [u32 ndim] [u32 * ndim shape] [f32 * prod(shape)]

The state-dict files use HuggingFace's `safetensors` format so the
ferrotorch-serialize `load_safetensors` loader handles them directly
without a custom decoder.

Usage:
  python3 scripts/pin_pretrained_training_trajectory.py \
      [--out-dir /tmp/ferrotorch_training_trajectory] \
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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi
from safetensors.torch import save_file


HF_REPO_ID = "ferrotorch/training-trajectory-v1"
BATCH = 4
EPOCHS = 5
N = 100
D_IN = 64
D_OUT = 8
LR = 1e-3


# ---------------------------------------------------------------------------
# Model & dataset — kept byte-deterministic via manual_seed(42).
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """3-layer MLP matching the architecture pinned in #1155.

    The state_dict produced here uses the natural `nn.Module`
    attribute names — `fc1.weight`, `fc1.bias`, `fc2.weight`, ... —
    rather than the `layer{i}.{weight,bias}` aliases used by the
    optimizer-trajectory bundle. This keeps the safetensors loader
    contract familiar (one key per `nn.Linear` weight / bias).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(D_IN, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, D_OUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def build_model_and_data() -> tuple[MLP, torch.Tensor, torch.Tensor]:
    """Construct the model and dataset under a single seed so the
    initial state and the dataset are jointly deterministic across
    pytorch / numpy versions."""
    torch.manual_seed(42)
    model = MLP()
    # Drawing X / y AFTER constructing the model means the model's
    # Kaiming-init state is captured at construction time, then the
    # RNG advances for the dataset draws — same shape as the dispatch.
    X = torch.randn(N, D_IN)
    y = torch.randn(N, D_OUT)
    return model, X, y


# ---------------------------------------------------------------------------
# Multi-tensor f32 binary format.
# ---------------------------------------------------------------------------


def dump_f32_tensor(path: Path, tensor: torch.Tensor) -> None:
    """Write a single-tensor `[u32 num_tensors=1]` + per-tensor
    `[u32 ndim][u32 shape][f32]` little-endian file. Identical layout
    to the optimizer-trajectory / dataloader-batches pins so the Rust
    side can reuse the same reader."""
    arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    with path.open("wb") as f:
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<I", int(d)))
        f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes(order="C"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Training trajectory.
# ---------------------------------------------------------------------------


def state_dict_for_save(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone the state dict to detach from autograd / shared storage,
    matching what `model.state_dict()` returns but safe to mutate later
    without affecting the model."""
    return {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}


def train(out_dir: Path) -> dict[str, Any]:
    print(f"[pin] building model + dataset (N={N}, D_in={D_IN}, D_out={D_OUT})")
    model, X_full, y_full = build_model_and_data()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # -- Snapshot initial state. ---------------------------------------
    initial = state_dict_for_save(model)
    save_file(initial, str(out_dir / "epoch_0_state.safetensors"))
    save_file(initial, str(out_dir / "initial_state.safetensors"))
    print(f"[pin]   epoch_0_state.safetensors  ({len(initial)} tensors)")

    # -- Dataset on disk. ----------------------------------------------
    dump_f32_tensor(out_dir / "X_full.bin", X_full)
    dump_f32_tensor(out_dir / "y_full.bin", y_full)
    print(f"[pin]   X_full.bin / y_full.bin")

    # -- Train EPOCHS epochs, sequential batch_size=4, drop_last=False.
    # Per the dispatch: 100 / 4 = 25 batches per epoch, 125 steps total.
    epoch_losses: list[float] = []
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, BATCH):
            x_batch = X_full[i : i + BATCH]
            y_batch = y_full[i : i + BATCH]
            opt.zero_grad()
            pred = model(x_batch)
            loss = F.mse_loss(pred, y_batch, reduction="mean")
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        mean_loss = epoch_loss / n_batches
        epoch_losses.append(mean_loss)
        post = state_dict_for_save(model)
        save_file(post, str(out_dir / f"epoch_{epoch + 1}_state.safetensors"))
        # First-tensor norm so the pin log is informative without
        # dumping every parameter.
        nrm0 = float(torch.linalg.vector_norm(post["fc1.weight"]).item())
        print(
            f"[pin]   epoch {epoch + 1}: loss={mean_loss:.6f}  "
            f"||fc1.weight||={nrm0:.4f}"
        )

    meta = {
        "n_samples": N,
        "batch_size": BATCH,
        "epochs": EPOCHS,
        "n_batches_per_epoch": N // BATCH,
        "lr": LR,
        "epoch_losses": epoch_losses,
        "model_arch": (
            f"MLP(Linear({D_IN},32) ReLU Linear(32,16) ReLU Linear(16,{D_OUT}))"
        ),
        "optimizer": "Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8)",
        "loss_fn": "F.mse_loss(reduction='mean')",
        "dataloader": (
            "sequential, batch_size=4, drop_last=False (manual sequential "
            "iteration so the harness does not bake in a particular "
            "DataLoader shuffle semantics)"
        ),
        "param_names": list(initial.keys()),
        "param_shapes": {k: list(v.shape) for k, v in initial.items()},
        "torch_version": torch.__version__,
        "seed": 42,
        "format": (
            "epoch_K_state.safetensors are HuggingFace safetensors files "
            "containing the state_dict at the start (K=0) or after each "
            "epoch (K=1..5). X_full.bin / y_full.bin are multi-tensor f32 "
            "files: [u32 num_tensors=1] + [u32 ndim][u32 * ndim shape]"
            "[f32 * prod(shape)]."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[pin]   meta.json  (epoch_losses={[f'{l:.4f}' for l in epoch_losses]})")
    return meta


# ---------------------------------------------------------------------------
# Bundle + upload.
# ---------------------------------------------------------------------------


def write_readme(out_root: Path, meta: dict[str, Any]) -> None:
    losses_str = "\n".join(
        f"  * epoch {ep + 1}: `{meta['epoch_losses'][ep]:.6f}`"
        for ep in range(EPOCHS)
    )
    readme = textwrap.dedent(f"""
        ---
        license: apache-2.0
        tags:
        - test-fixtures
        - training
        - autograd
        - optimizer
        - dataloader
        - pytorch
        ---

        # ferrotorch / training-trajectory-v1

        Multi-epoch training-trajectory parity fixtures for ferrotorch's
        full training stack — autograd + loss + optimizer + DataLoader.
        Phase E of real-artifact-driven development (#1161).

        Generated by running `torch.optim.Adam` on a fixed 3-layer MLP
        against a fixed deterministic regression dataset for 5 epochs of
        sequential iteration (`batch_size=4`, `drop_last=False`,
        no shuffling) and snapshotting the state_dict after each epoch.
        125 optimizer steps total.

        Companion to:
          * `scripts/pin_pretrained_training_trajectory.py` (this pin)
          * `scripts/verify_training_trajectory.py` (the harness)
          * `ferrotorch-train/examples/multi_epoch_train_dump.rs`
          * `ferrotorch-train/tests/conformance_multi_epoch_training.rs`

        ## Why live autograd

        Phase C.2 (#1155) verified the *optimizer step math* with
        **frozen** gradients (snapshotted from torch, re-applied on the
        ferrotorch side) to isolate one suspect at a time. This pin
        verifies the *full training loop* with **live** autograd — the
        ferrotorch side has to re-derive the gradients itself. If
        anything in the stack diverges (linear backward, relu backward,
        mse backward, Adam state, sequential dataloader iteration order)
        the harness will catch it as a per-epoch state_dict drift.

        ## Architecture

        ```
        MLP(
          Linear({D_IN} -> 32) -> ReLU
          Linear(32 -> 16)      -> ReLU
          Linear(16 -> {D_OUT})
        )
        ```

        ## Dataset

        * `X_full.bin`  — `torch.randn({N}, {D_IN})` with seed 42
        * `y_full.bin`  — `torch.randn({N}, {D_OUT})` with seed 42
        * Loss target: `F.mse_loss(pred, y, reduction='mean')`

        ## Training

        * Optimizer: `Adam(lr={LR}, betas=(0.9, 0.999), eps=1e-8)`
        * Batch size: `{BATCH}`
        * Iteration: sequential (`for i in range(0, N, BATCH)` —
          equivalent to `DataLoader(shuffle=False, drop_last=False)`)
        * Epochs: `{EPOCHS}`
        * Per-epoch losses (mean over 25 batches):
        {losses_str}

        ## Layout

        ```
        epoch_0_state.safetensors    # initial state (alias: initial_state.safetensors)
        epoch_1_state.safetensors    # after epoch 1
        epoch_2_state.safetensors    # after epoch 2
        epoch_3_state.safetensors    # after epoch 3
        epoch_4_state.safetensors    # after epoch 4
        epoch_5_state.safetensors    # after epoch 5
        X_full.bin                   # full dataset features
        y_full.bin                   # full dataset targets
        meta.json                    # hyperparameters + per-epoch losses
        bundle.tar                   # convenience archive (registry pin checksum)
        ```

        State-dict keys: `fc1.weight`, `fc1.bias`, `fc2.weight`,
        `fc2.bias`, `fc3.weight`, `fc3.bias`.

        ## Tolerance

        The harness gate is `max_abs <= 1e-4` and `cosine_sim >= 0.9999`
        per tensor for every epoch — autograd noise budget for 125 steps
        of accumulated f32 noise across two independent runtimes.

        ## License

        Apache 2.0. Synthetic fixtures generated by this repo's pin
        script; no upstream weights / data.
    """).strip()
    (out_root / "README.md").write_text(readme)


def build_bundle(out_root: Path) -> Path:
    """Tar the artifacts into a single file the registry can pin with
    one SHA. Mirrors the optimizer-trajectories / dataloader-batches /
    ml-sklearn-parity bundle pattern."""
    tar_path = out_root / "bundle.tar"
    entries = [
        "initial_state.safetensors",
        *[f"epoch_{ep}_state.safetensors" for ep in range(EPOCHS + 1)],
        "X_full.bin",
        "y_full.bin",
        "meta.json",
    ]
    with tarfile.open(tar_path, "w") as tar:
        for name in entries:
            p = out_root / name
            if not p.is_file():
                raise FileNotFoundError(f"bundle entry missing: {p}")
            tar.add(p, arcname=name)
    return tar_path


def hf_upload(out_root: Path) -> None:
    api = HfApi()
    print(f"\n[pin] uploading to https://huggingface.co/{HF_REPO_ID} ...")
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(out_root),
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="feat: pin training-trajectory fixtures v1 (#1161)",
    )
    print("[pin] upload complete.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default="/tmp/ferrotorch_training_trajectory",
        help="Staging directory.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage everything locally but do not upload to HF.",
    )
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    meta = train(out_root)
    write_readme(out_root, meta)
    bundle_path = build_bundle(out_root)
    bundle_sha = sha256_of(bundle_path)

    if not args.dry_run:
        hf_upload(out_root)

    print("\n=== SUMMARY ===")
    print(f"  epochs                    {EPOCHS}")
    print(f"  steps/epoch               {N // BATCH}")
    print(f"  total optimizer steps     {EPOCHS * (N // BATCH)}")
    print(f"  initial loss              {meta['epoch_losses'][0]:.6f}")
    print(f"  final   loss (epoch {EPOCHS})    {meta['epoch_losses'][-1]:.6f}")
    print(f"\nlocal stage:    {out_root}")
    print(f"bundle:         {bundle_path}")
    print(f"bundle sha256:  {bundle_sha}")
    print(f"hf:             https://huggingface.co/{HF_REPO_ID}")

    print("\n=== Drop-in registry pin (for ferrotorch-hub/src/registry.rs) ===")
    print('  ModelInfo {')
    print('      name: "training-trajectory-v1",')
    print(
        '      description: "Phase E multi-epoch training-trajectory '
        'fixtures: 3-layer MLP (64-32-16-8) trained with Adam(lr=1e-3) + '
        'MSE on a deterministic 100-sample dataset for 5 epochs (25 '
        'batches/epoch, batch_size=4, sequential). Reference state_dicts '
        'from torch + autograd for verifying ferrotorch\'s full training '
        'stack (forward + loss + backward + optimizer + dataloader) '
        'against torch. Apache 2.0; real-artifact baseline for end-to-end '
        'training parity vs torch (#1161).",'
    )
    print(
        f'      weights_url: "https://huggingface.co/{HF_REPO_ID}/resolve/main/bundle.tar",'
    )
    print(f'      weights_sha256: "{bundle_sha}",')
    print('      format: WeightsFormat::FerrotorchStateDict,')
    print('      num_parameters: 0,')
    print('  },')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
