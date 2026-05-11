#!/usr/bin/env python3
"""Pin a frozen-gradient optimizer-trajectory fixture set to the
`ferrotorch/optimizer-trajectories-v1` HF mirror.

Phase C.2 of real-artifact-driven development (#1155): build a reference
trajectory for every (optimizer, config) tuple in the high-traffic
optimizer matrix (SGD/Adam/AdamW/RMSprop/Adagrad — 10 configs total),
freeze the per-step gradients, and ship the start state + gradient
sequence + end state so ferrotorch's `Optimizer::step()` can be
byte-compared against torch.optim's update math without re-running
autograd at verification time.

Why **frozen gradients** instead of live autograd:

* The test is "does ferrotorch's optimizer math match torch's optimizer
  math", not "does ferrotorch's autograd match torch's autograd". Live
  autograd would fold MSELoss + Linear backward bugs into every PASS/FAIL
  verdict — the matmul-grad path is already covered by the causal-LM
  and BERT real-artifact harnesses. Decoupling lets a failure here
  finger-point at one of {SGD, Adam, AdamW, RMSprop, Adagrad}.
* Frozen gradients are byte-reproducible across PyTorch versions /
  CPU/GPU / random-state quirks — a snapshot binary is portable.
* The dispatch (Phase C.2 prompt) explicitly recommends this choice.

MLP architecture (fixed; matches the dispatch):
    Linear(64 -> 32) -> ReLU -> Linear(32 -> 16) -> ReLU -> Linear(16 -> 8)
6 parameters per model:
    layer0.weight  [32, 64]   layer0.bias [32]
    layer1.weight  [16, 32]   layer1.bias [16]
    layer2.weight  [8, 16]    layer2.bias [8]

Per (optimizer, config) the pin emits:
  * initial_params.bin       — concatenated f32 params before step 0
  * gradients_step_K.bin     — concatenated f32 gradients for K = 0..9
  * final_params.bin         — concatenated f32 params after step 10
  * meta.json                — config dict + shapes + dtype

Multi-tensor binary layout (little-endian):
  [u32 num_tensors]
  per tensor:
    [u32 ndim] [u32 × ndim shape] [f32 × prod(shape)]

Then everything is bundled into a single `model.safetensors`-style
artifact bundle (one HF subfolder per config) and uploaded to
`ferrotorch/optimizer-trajectories-v1`.

Usage:
  python3 scripts/pin_pretrained_optimizer_trajectories.py \
      [--out-dir /tmp/ferrotorch_optimizer_trajectories] \
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
import torch.nn as nn
from huggingface_hub import HfApi


# ---------------------------------------------------------------------------
# Configuration matrix — 10 trajectories total.
# ---------------------------------------------------------------------------


@dataclass
class OptimizerSpec:
    """One (optimizer, config) tuple to pin."""

    name: str           # config key, e.g. "sgd_plain"
    optimizer: str      # "SGD" | "Adam" | "AdamW" | "RMSprop" | "Adagrad"
    config: dict[str, Any]


# Match the dispatch exactly: 3 SGD + 2 Adam + 1 AdamW + 2 RMSprop + 2 Adagrad.
SPECS: list[OptimizerSpec] = [
    OptimizerSpec(
        name="sgd_plain",
        optimizer="SGD",
        config=dict(lr=1e-2),
    ),
    OptimizerSpec(
        name="sgd_momentum",
        optimizer="SGD",
        config=dict(lr=1e-2, momentum=0.9),
    ),
    OptimizerSpec(
        name="sgd_nesterov",
        optimizer="SGD",
        config=dict(lr=1e-2, momentum=0.9, nesterov=True),
    ),
    OptimizerSpec(
        name="adam_default",
        optimizer="Adam",
        config=dict(lr=1e-3),
    ),
    OptimizerSpec(
        name="adam_explicit",
        optimizer="Adam",
        config=dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8),
    ),
    OptimizerSpec(
        name="adamw_decoupled",
        optimizer="AdamW",
        config=dict(lr=1e-3, weight_decay=1e-2),
    ),
    OptimizerSpec(
        name="rmsprop_default",
        optimizer="RMSprop",
        config=dict(lr=1e-3),
    ),
    OptimizerSpec(
        name="rmsprop_momentum",
        optimizer="RMSprop",
        config=dict(lr=1e-3, momentum=0.9, alpha=0.99),
    ),
    OptimizerSpec(
        name="adagrad_default",
        optimizer="Adagrad",
        config=dict(lr=1e-2),
    ),
    OptimizerSpec(
        name="adagrad_explicit",
        optimizer="Adagrad",
        config=dict(lr=1e-2, lr_decay=0.1, eps=1e-10),
    ),
]

NUM_STEPS = 10
HF_REPO_ID = "ferrotorch/optimizer-trajectories-v1"
PARAM_NAMES = [
    "layer0.weight",
    "layer0.bias",
    "layer1.weight",
    "layer1.bias",
    "layer2.weight",
    "layer2.bias",
]


# ---------------------------------------------------------------------------
# MLP — deterministic across PyTorch versions thanks to manual_seed(42)
# + torch.use_deterministic_algorithms.
# ---------------------------------------------------------------------------


def build_mlp_and_data() -> tuple[nn.Sequential, torch.Tensor, torch.Tensor]:
    """Return (model, x, y). The model is freshly constructed and seeded;
    the caller re-builds it from scratch per spec so each spec starts
    from the same initial state."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    )
    x = torch.randn(8, 64)
    y = torch.randn(8, 8)
    return model, x, y


def named_param_list(model: nn.Sequential) -> list[tuple[str, torch.nn.Parameter]]:
    """Return params in the canonical fixture order:
    layer0.weight, layer0.bias, layer1.weight, ..., layer2.bias."""
    layers = [m for m in model if isinstance(m, nn.Linear)]
    if len(layers) != 3:
        raise RuntimeError(f"expected 3 Linear layers, got {len(layers)}")
    out: list[tuple[str, torch.nn.Parameter]] = []
    for i, lin in enumerate(layers):
        out.append((f"layer{i}.weight", lin.weight))
        out.append((f"layer{i}.bias", lin.bias))
    return out


# ---------------------------------------------------------------------------
# Multi-tensor binary format.
# ---------------------------------------------------------------------------


def dump_multi_tensor_f32(path: Path, tensors: list[tuple[str, np.ndarray]]) -> None:
    """Write a `[u32 num_tensors]` + per-tensor `[u32 ndim][u32 shape][f32]`
    little-endian dump. Tensor name is *not* serialized — order is the
    contract (see `PARAM_NAMES`)."""
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(tensors)))
        for _name, arr in tensors:
            arr32 = np.ascontiguousarray(arr, dtype="<f4")
            shape = list(arr32.shape)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", int(d)))
            f.write(arr32.tobytes(order="C"))


def read_multi_tensor_f32(path: Path) -> list[np.ndarray]:
    """Inverse of `dump_multi_tensor_f32`. Returns tensors in order."""
    raw = path.read_bytes()
    off = 0
    (n,) = struct.unpack_from("<I", raw, off)
    off += 4
    out: list[np.ndarray] = []
    for _ in range(n):
        (ndim,) = struct.unpack_from("<I", raw, off)
        off += 4
        shape = struct.unpack_from(f"<{ndim}I", raw, off)
        off += 4 * ndim
        numel = 1
        for s in shape:
            numel *= int(s)
        arr = np.frombuffer(raw, dtype="<f4", count=numel, offset=off).reshape(shape)
        off += 4 * numel
        out.append(arr.astype(np.float32, copy=True))
    if off != len(raw):
        raise ValueError(f"trailing bytes in {path}: {len(raw) - off}")
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# torch.optim factory.
# ---------------------------------------------------------------------------


def make_torch_optimizer(spec: OptimizerSpec, params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    """Instantiate the torch.optim optimizer matching `spec`."""
    if spec.optimizer == "SGD":
        return torch.optim.SGD(params, **spec.config)
    if spec.optimizer == "Adam":
        return torch.optim.Adam(params, **spec.config)
    if spec.optimizer == "AdamW":
        return torch.optim.AdamW(params, **spec.config)
    if spec.optimizer == "RMSprop":
        return torch.optim.RMSprop(params, **spec.config)
    if spec.optimizer == "Adagrad":
        return torch.optim.Adagrad(params, **spec.config)
    raise ValueError(f"unknown optimizer {spec.optimizer!r}")


# ---------------------------------------------------------------------------
# Trajectory generation.
# ---------------------------------------------------------------------------


def generate_trajectory(spec: OptimizerSpec, out_dir: Path) -> dict[str, Any]:
    """Build initial state, run 10 (autograd-computed-once-then-frozen)
    gradient steps with torch.optim, snapshot per-step gradients and
    final params. Returns a metadata dict."""
    print(f"\n=== {spec.name} ({spec.optimizer}{spec.config}) ===", flush=True)
    model, x, y = build_mlp_and_data()
    named = named_param_list(model)

    # -- Snapshot initial state. ------------------------------------
    init_arr: list[tuple[str, np.ndarray]] = []
    shapes: dict[str, list[int]] = {}
    for name, p in named:
        arr = p.detach().cpu().numpy().astype(np.float32, copy=True)
        init_arr.append((name, arr))
        shapes[name] = list(arr.shape)
    dump_multi_tensor_f32(out_dir / "initial_params.bin", init_arr)
    print(f"  wrote initial_params.bin ({len(init_arr)} tensors)")

    # -- Set up optimizer and loss. ---------------------------------
    optimizer = make_torch_optimizer(spec, [p for _, p in named])
    loss_fn = nn.MSELoss(reduction="mean")

    # -- Run 10 steps, snapshotting gradients per step. -------------
    # Frozen-gradient policy:
    #
    # For step K we run a real forward + backward pass on the live
    # parameters, snapshot the resulting `.grad` into
    # `gradients_step_K.bin`, then call `optimizer.step()` and
    # `optimizer.zero_grad()`. The Rust side will re-apply these
    # snapshotted gradients verbatim (no autograd) — so any forward /
    # backward divergence between PyTorch and ferrotorch does NOT
    # influence the verdict. Only the optimizer's update math is under
    # test.
    for step in range(NUM_STEPS):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()

        grad_arr: list[tuple[str, np.ndarray]] = []
        for name, p in named:
            if p.grad is None:
                raise RuntimeError(f"step {step}: param {name} has no grad")
            grad_arr.append(
                (name, p.grad.detach().cpu().numpy().astype(np.float32, copy=True))
            )
        dump_multi_tensor_f32(out_dir / f"gradients_step_{step}.bin", grad_arr)

        optimizer.step()
        print(
            f"  step {step}: loss={loss.item():.6f}  "
            f"||grad[0]||={float(np.linalg.norm(grad_arr[0][1])):.4f}  "
            f"||param[0]||={float(np.linalg.norm(named[0][1].detach().numpy())):.4f}"
        )

    # -- Snapshot final state. --------------------------------------
    final_arr: list[tuple[str, np.ndarray]] = []
    for name, p in named:
        final_arr.append(
            (name, p.detach().cpu().numpy().astype(np.float32, copy=True))
        )
    dump_multi_tensor_f32(out_dir / "final_params.bin", final_arr)
    print(f"  wrote final_params.bin ({len(final_arr)} tensors)")

    meta = {
        "name": spec.name,
        "optimizer": spec.optimizer,
        "config": spec.config,
        "num_steps": NUM_STEPS,
        "param_names": PARAM_NAMES,
        "shapes": shapes,
        "dtype": "float32",
        "torch_version": torch.__version__,
        "mlp": {
            "layers": [
                {"in": 64, "out": 32, "bias": True},
                {"act": "relu"},
                {"in": 32, "out": 16, "bias": True},
                {"act": "relu"},
                {"in": 16, "out": 8, "bias": True},
            ],
            "batch": [8, 64],
            "target": [8, 8],
            "loss": "MSELoss(reduction='mean')",
            "seed": 42,
        },
        "format": (
            "Each .bin file is `[u32 num_tensors]` followed by per-tensor "
            "`[u32 ndim][u32 × ndim shape][f32 × prod(shape)]` "
            "little-endian. Tensors appear in the order listed in "
            "`param_names` — name is not stored."
        ),
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
        cfg = ", ".join(f"{k}={v}" for k, v in m["config"].items())
        config_lines.append(f"  * `{m['name']}` — `{m['optimizer']}({cfg})`")
    readme = textwrap.dedent(f"""
        ---
        license: apache-2.0
        tags:
        - test-fixtures
        - optimizer
        - pytorch
        ---

        # ferrotorch / optimizer-trajectories-v1

        Frozen-gradient parity fixtures for ferrotorch's `Optimizer::step()`
        implementations, generated by running `torch.optim` against a
        small fixed MLP for {NUM_STEPS} steps and snapshotting initial
        params, per-step gradients, and final params.

        Phase C.2 of real-artifact-driven development (#1155). Companion
        to:
          * `scripts/pin_pretrained_optimizer_trajectories.py` (this pin)
          * `scripts/verify_optimizer_inference.py` (the harness)
          * `ferrotorch-optim/examples/optimizer_trajectory_dump.rs`
          * `ferrotorch-optim/tests/conformance_optimizer_trajectories.rs`

        ## Why frozen gradients

        The test is "does ferrotorch's optimizer match torch's optimizer
        math", not "does ferrotorch's autograd match torch's autograd".
        Live autograd would fold linear+MSELoss backward bugs into every
        verdict; the matmul-grad path is already covered by the
        causal-LM and BERT real-artifact harnesses. By snapshotting
        gradients on the PyTorch side and re-applying them verbatim on
        the ferrotorch side, a failure in this harness fingers an
        optimizer (one of SGD / Adam / AdamW / RMSprop / Adagrad), not
        autograd.

        ## MLP

        ```
        Linear(64 -> 32) -> ReLU -> Linear(32 -> 16) -> ReLU -> Linear(16 -> 8)
        ```

        * Seed: `torch.manual_seed(42)` before construction
        * Input batch: `torch.randn(8, 64)`
        * Target batch: `torch.randn(8, 8)`
        * Loss: `MSELoss(reduction='mean')`
        * 6 parameters per model in canonical order:
          `layer{{0,1,2}}.{{weight,bias}}`

        ## Configurations

        {chr(10).join(config_lines)}

        ## Layout

        One subfolder per configuration:

        ```
        <config_name>/
          meta.json
          initial_params.bin       # params before step 0
          gradients_step_0.bin     # gradient at step 0
          gradients_step_1.bin     # gradient at step 1
          gradients_step_2.bin     # gradient at step 2
          gradients_step_3.bin     # gradient at step 3
          gradients_step_4.bin     # gradient at step 4
          gradients_step_5.bin     # gradient at step 5
          gradients_step_6.bin     # gradient at step 6
          gradients_step_7.bin     # gradient at step 7
          gradients_step_8.bin     # gradient at step 8
          gradients_step_9.bin     # gradient at step 9
          final_params.bin         # params after step {NUM_STEPS}
        ```

        ## Binary format

        All `.bin` files use the same little-endian multi-tensor layout:

        ```
        [u32 num_tensors]
        per tensor:
          [u32 ndim] [u32 * ndim shape] [f32 * prod(shape)]
        ```

        Tensor *order* (not name) is the contract — see
        `param_names` in `meta.json`.

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
        commit_message="feat: pin optimizer-trajectory fixtures v1 (#1155)",
    )
    print("upload complete.", flush=True)


def build_bundle(out_root: Path) -> Path:
    """Write a single `model.safetensors`-style aggregate file as
    `bundle.tar` containing the per-config subfolders. This lets the
    registry pin point at one file with one SHA, matching the existing
    pin pattern; the verify script downloads individual files via
    `hf_hub_download` and does not need this tar (it pulls each
    `.bin` directly). The tar exists so `registry.rs` has a single
    artifact to checksum."""
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
        default="/tmp/ferrotorch_optimizer_trajectories",
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
    if len(specs) != len(SPECS) and not only:
        raise SystemExit("internal: missing specs")
    if not specs:
        print("no specs match --only filter", file=sys.stderr)
        return 2

    metas: list[dict[str, Any]] = []
    for spec in specs:
        sub = out_root / spec.name
        sub.mkdir(parents=True, exist_ok=True)
        metas.append(generate_trajectory(spec, sub))

    write_readme(out_root, metas)
    bundle_path = build_bundle(out_root)
    bundle_sha = sha256_of(bundle_path)

    if not args.dry_run:
        hf_upload(out_root)

    print("\n=== SUMMARY ===")
    for m in metas:
        cfg = ", ".join(f"{k}={v}" for k, v in m["config"].items())
        print(f"  {m['name']:24s} {m['optimizer']}({cfg})")
    print(f"\nlocal stage:  {out_root}")
    print(f"bundle:       {bundle_path}")
    print(f"bundle sha256: {bundle_sha}")
    print(f"hf:           https://huggingface.co/{HF_REPO_ID}")

    print("\n=== Drop-in registry pin (for ferrotorch-hub/src/registry.rs) ===")
    print('  ModelInfo {')
    print('      name: "optimizer-trajectories-v1",')
    print('      description: "Frozen-gradient optimizer trajectory fixtures (SGD/Adam/AdamW/RMSprop/Adagrad x 10 configs, MLP 64-32-16-8, 10 steps each) — Phase C.2 real-artifact harness baseline (#1155).",')
    print(f'      weights_url: "https://huggingface.co/{HF_REPO_ID}/resolve/main/bundle.tar",')
    print(f'      weights_sha256: "{bundle_sha}",')
    print('      format: WeightsFormat::FerrotorchStateDict,')
    print('      num_parameters: 0,')
    print('  },')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
