#!/usr/bin/env python3
"""Pin the ferrotorch-core autograd backward parity fixtures (#1171, Phase G.5).

Generates a deterministic torch.autograd reference for each canonical
op exposed by `ferrotorch_core`, then ships the inputs + per-input
backward gradients to ``ferrotorch/autograd-parity-v1`` on the
Hugging Face Hub.

Op inventory (24 ops, all native to ``ferrotorch-core``):

  * matmul family: ``matmul_2d`` (mm), ``bmm``, ``linear``
  * activations:   ``relu``, ``gelu``, ``silu``, ``sigmoid``, ``tanh``
  * reductions:    ``softmax``, ``log_softmax``, ``sum_dim``, ``mean_dim``
  * element-wise:  ``add``, ``mul``, ``sub``, ``div``, ``log``, ``exp``,
                   ``pow``
  * shape ops:     ``reshape``, ``transpose``, ``cat``
  * indexing:      ``embedding`` (index_select_dim)
  * attention:     ``attention`` (Q @ K.T / sqrt(d) -> softmax -> @V)

Skipped / not core-native:

  * Conv1d / Conv2d / LayerNorm / BatchNorm / GroupNorm / RmsNorm /
    scaled_dot_product_attention all live in ``ferrotorch-nn``, not
    ``ferrotorch-core``. They are nn::Module layers (or
    nn::functional::* wrappers around those modules), not ops exposed
    by core's ``grad_fns::*`` taxonomy. Their backward parity is a
    separate harness on ``ferrotorch-nn`` (out of scope for #1171).
    ``attention`` is included here in its composed form
    (matmul + softmax + matmul) because that exact chain exercises
    the core autograd graph and is what ``ferrotorch-nn::flash_attention``
    lowers to.

For each fixture the script writes:

    fixtures/<op>/<config>/
        params.json        — op kwargs + tensor metadata + which gradients exist
        forward_out.bin    — torch forward output (sanity)
        inputs/<name>.bin  — each input tensor used in the forward
        grads/<name>.bin   — torch.autograd reference gradient for <name>
                              (only present when that input had requires_grad)

Binary format (matches the ferrotorch ``read_f32_tensor`` helper):

    [u32 ndim][u32 × ndim shape][f32 le data]

Run via:
    python3 scripts/pin_pretrained_autograd_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

WORK_DIR = Path("/tmp/ferrotorch_pin_autograd_parity_v1")
WORK_DIR.mkdir(parents=True, exist_ok=True)
FIX_DIR = WORK_DIR / "fixtures"
FIX_DIR.mkdir(parents=True, exist_ok=True)

HF_REPO_ID = "ferrotorch/autograd-parity-v1"
SEED = 42


# ---------------------------------------------------------------------------
# Binary I/O.
# ---------------------------------------------------------------------------


def dump_f32(path: Path, t: torch.Tensor) -> None:
    arr = t.detach().cpu().contiguous().to(torch.float32).numpy()
    arr = np.ascontiguousarray(arr, dtype="<f4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<I", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<I", int(d)))
        f.write(arr.tobytes(order="C"))


# ---------------------------------------------------------------------------
# Fixture writer.
# ---------------------------------------------------------------------------


def run_fixture(
    op: str,
    config: str,
    inputs: dict[str, torch.Tensor],
    forward: Callable[..., torch.Tensor],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run torch forward+backward and emit fixture files. Returns metadata."""
    torch.manual_seed(SEED)
    out_dir = FIX_DIR / op / config
    inputs_dir = out_dir / "inputs"
    grads_dir = out_dir / "grads"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    grads_dir.mkdir(parents=True, exist_ok=True)

    # Dump inputs.
    grad_names: list[str] = []
    for name, t in inputs.items():
        dump_f32(inputs_dir / f"{name}.bin", t)
        if t.requires_grad:
            grad_names.append(name)

    # Forward + sum reduction so backward is well-defined.
    out = forward(**inputs)
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"{op}/{config}: forward must return Tensor, got {type(out)}")
    loss = out.sum()
    loss.backward()

    dump_f32(out_dir / "forward_out.bin", out)

    # Dump gradients.
    for name in grad_names:
        g = inputs[name].grad
        if g is None:
            raise RuntimeError(f"{op}/{config}: torch did not produce a grad for {name}")
        dump_f32(grads_dir / f"{name}.bin", g)

    # Cosmetic stats for the metadata.
    out_np = out.detach().cpu().to(torch.float32).numpy()
    meta = {
        "op": op,
        "config": config,
        "params": params,
        "inputs": {
            name: {
                "shape": list(t.shape),
                "requires_grad": bool(t.requires_grad),
            }
            for name, t in inputs.items()
        },
        "grad_inputs": grad_names,
        "forward_shape": list(out.shape),
        "forward_sample": [float(x) for x in out_np.ravel()[:4].tolist()],
        "loss_value": float(loss.detach().item()),
    }
    (out_dir / "params.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[pin] {op}/{config:32s} loss={meta['loss_value']:+.6f} "
        f"grad_inputs={grad_names}",
        flush=True,
    )
    return meta


def leaf(shape: tuple[int, ...], *, requires_grad: bool = True, scale: float = 1.0,
         rng: np.random.RandomState | None = None,
         positive: bool = False) -> torch.Tensor:
    """Build a deterministic input tensor of shape `shape`.

    The default draw is N(0,1) scaled by `scale`. When `positive=True`
    the draw is shifted+clamped to (0.1, inf) so that domain-restricted
    ops (log, pow with fractional exponent, div) have a defined VJP.
    """
    if rng is None:
        rng = np.random.RandomState(SEED)
    data = rng.standard_normal(size=shape).astype(np.float32) * scale
    if positive:
        data = np.abs(data) + 0.1
    t = torch.from_numpy(data).to(torch.float32)
    t.requires_grad_(requires_grad)
    return t


# ---------------------------------------------------------------------------
# Op generators — one function per op. Each adds 1-2 configs to capture
# both the canonical shape and a non-trivial broadcast / batched case.
# ---------------------------------------------------------------------------


def pin_matmul_2d() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    a = leaf((8, 16), rng=rng)
    b = leaf((16, 4), rng=rng)
    metas.append(run_fixture(
        "matmul_2d", "8x16_16x4",
        {"a": a, "b": b},
        lambda a, b: a @ b,
        {"shape_a": [8, 16], "shape_b": [16, 4]},
    ))
    return metas


def pin_bmm() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    a = leaf((3, 4, 5), rng=rng)
    b = leaf((3, 5, 6), rng=rng)
    metas.append(run_fixture(
        "bmm", "3x4x5_3x5x6",
        {"a": a, "b": b},
        lambda a, b: torch.bmm(a, b),
        {"shape_a": [3, 4, 5], "shape_b": [3, 5, 6]},
    ))
    return metas


def pin_linear() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    # input [B=4, in=6], weight [out=3, in=6], bias [out=3]
    x = leaf((4, 6), rng=rng)
    w = leaf((3, 6), rng=rng)
    b = leaf((3,), rng=rng)
    metas.append(run_fixture(
        "linear", "4x6_out3_bias",
        {"input": x, "weight": w, "bias": b},
        lambda input, weight, bias: F.linear(input, weight, bias),
        {"in_features": 6, "out_features": 3, "has_bias": True},
    ))
    return metas


def _activation(name: str, fn: Callable[[torch.Tensor], torch.Tensor]) -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    x = leaf((4, 8), rng=rng)
    metas.append(run_fixture(
        name, "4x8",
        {"x": x},
        lambda x: fn(x),
        {"shape": [4, 8]},
    ))
    return metas


def pin_relu() -> list[dict[str, Any]]:
    return _activation("relu", F.relu)


def pin_gelu() -> list[dict[str, Any]]:
    # torch's default gelu uses the exact erf-based formulation, which
    # matches `ferrotorch_core::gelu` (the wrapper for the "Exact"
    # variant — see `GeluApproximate::None` in grad_fns/activation.rs).
    return _activation("gelu", lambda x: F.gelu(x, approximate="none"))


def pin_silu() -> list[dict[str, Any]]:
    return _activation("silu", F.silu)


def pin_sigmoid() -> list[dict[str, Any]]:
    return _activation("sigmoid", torch.sigmoid)


def pin_tanh() -> list[dict[str, Any]]:
    return _activation("tanh", torch.tanh)


def pin_softmax() -> list[dict[str, Any]]:
    # ferrotorch's softmax is hardcoded to the LAST axis (see
    # `grad_fns/activation.rs::softmax_inner`). Mirror that here.
    #
    # NOTE: `sum(softmax(x))` is mathematically constant (each row sums
    # to 1, so the total sums to `n_rows`) — its VJP is identically zero
    # and the test would not exercise the softmax backward Jacobian at
    # all. To get a non-degenerate gradient we multiply the softmax
    # output element-wise by a fixed no-grad "target" tensor before
    # summing. The rust dump example mirrors this exact reduction.
    metas = []
    rng = np.random.RandomState(SEED)
    x = leaf((4, 8), rng=rng)
    # Frozen target — no requires_grad, dumped as an input so the rust
    # side can load it identically.
    target_rng = np.random.RandomState(SEED + 100)
    target = torch.from_numpy(
        target_rng.standard_normal(size=(4, 8)).astype(np.float32)
    )  # requires_grad = False (default)
    metas.append(run_fixture(
        "softmax", "4x8_lastdim",
        {"x": x, "target": target},
        lambda x, target: F.softmax(x, dim=-1) * target,
        {"shape": [4, 8], "dim": -1,
         "reduction": "(softmax(x) * target).sum()"},
    ))
    return metas


def pin_log_softmax() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    x = leaf((4, 8), rng=rng)
    metas.append(run_fixture(
        "log_softmax", "4x8_lastdim",
        {"x": x},
        lambda x: F.log_softmax(x, dim=-1),
        {"shape": [4, 8], "dim": -1},
    ))
    return metas


def pin_sum_dim() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    x = leaf((3, 5, 7), rng=rng)
    # Multiple configs — middle dim, last dim, keepdim variations.
    metas.append(run_fixture(
        "sum_dim", "3x5x7_dim1_nokeep",
        {"x": x},
        lambda x: x.sum(dim=1, keepdim=False),
        {"dim": 1, "keepdim": False},
    ))
    # Fresh leaf for the second config (grad accumulator wouldn't be reset).
    x2 = leaf((3, 5, 7), rng=np.random.RandomState(SEED + 1))
    metas.append(run_fixture(
        "sum_dim", "3x5x7_dim2_keep",
        {"x": x2},
        lambda x: x.sum(dim=2, keepdim=True),
        {"dim": 2, "keepdim": True},
    ))
    return metas


def pin_mean_dim() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    x = leaf((3, 5, 7), rng=rng)
    metas.append(run_fixture(
        "mean_dim", "3x5x7_dim1_nokeep",
        {"x": x},
        lambda x: x.mean(dim=1, keepdim=False),
        {"dim": 1, "keepdim": False},
    ))
    return metas


def pin_add() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    a = leaf((4, 5), rng=rng)
    b = leaf((4, 5), rng=rng)
    metas.append(run_fixture(
        "add", "4x5_same",
        {"a": a, "b": b},
        lambda a, b: a + b,
        {"shape": [4, 5]},
    ))
    return metas


def pin_mul() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    a = leaf((4, 5), rng=rng)
    b = leaf((4, 5), rng=rng)
    metas.append(run_fixture(
        "mul", "4x5_same",
        {"a": a, "b": b},
        lambda a, b: a * b,
        {"shape": [4, 5]},
    ))
    return metas


def pin_sub() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    a = leaf((4, 5), rng=rng)
    b = leaf((4, 5), rng=rng)
    metas.append(run_fixture(
        "sub", "4x5_same",
        {"a": a, "b": b},
        lambda a, b: a - b,
        {"shape": [4, 5]},
    ))
    return metas


def pin_div() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    a = leaf((4, 5), rng=rng)
    b = leaf((4, 5), rng=rng, positive=True)  # avoid /0
    metas.append(run_fixture(
        "div", "4x5_same",
        {"a": a, "b": b},
        lambda a, b: a / b,
        {"shape": [4, 5]},
    ))
    return metas


def pin_log() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    x = leaf((4, 5), rng=rng, positive=True)
    metas.append(run_fixture(
        "log", "4x5_positive",
        {"x": x},
        torch.log,
        {"shape": [4, 5]},
    ))
    return metas


def pin_exp() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    # exp(x) blows up for x >> 0; clip to keep things in f32 range and
    # keep grad magnitudes small enough that 1e-4 max_abs is meaningful.
    x = leaf((4, 5), rng=rng, scale=0.5)
    metas.append(run_fixture(
        "exp", "4x5",
        {"x": x},
        torch.exp,
        {"shape": [4, 5]},
    ))
    return metas


def pin_pow() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    x = leaf((4, 5), rng=rng, positive=True)  # avoid pow(neg, frac)
    metas.append(run_fixture(
        "pow", "4x5_exp_2_5",
        {"x": x},
        lambda x: torch.pow(x, 2.5),
        {"exponent": 2.5},
    ))
    return metas


def pin_reshape() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    x = leaf((2, 3, 4), rng=rng)
    metas.append(run_fixture(
        "reshape", "2x3x4_to_6x4",
        {"x": x},
        lambda x: x.reshape(6, 4),
        {"new_shape": [6, 4]},
    ))
    return metas


def pin_transpose() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    x = leaf((4, 6), rng=rng)
    metas.append(run_fixture(
        "transpose", "4x6_swap01",
        {"x": x},
        lambda x: x.transpose(0, 1).contiguous(),
        {"dim0": 0, "dim1": 1},
    ))
    return metas


def pin_cat() -> list[dict[str, Any]]:
    metas = []
    rng = np.random.RandomState(SEED)
    a = leaf((2, 3), rng=rng)
    b = leaf((4, 3), rng=np.random.RandomState(SEED + 1))
    c = leaf((1, 3), rng=np.random.RandomState(SEED + 2))
    metas.append(run_fixture(
        "cat", "axis0_2_4_1_x3",
        {"a": a, "b": b, "c": c},
        lambda a, b, c: torch.cat([a, b, c], dim=0),
        {"axis": 0},
    ))
    return metas


def pin_embedding() -> list[dict[str, Any]]:
    """`embedding` here means *the gather along a chosen axis* in core's
    parlance — i.e. `index_select_dim`, the backward primitive both
    ``nn::Embedding`` and ``index_select`` lower to. ``nn.Embedding``
    is a thin wrapper that calls ``index_select_dim(weight, 0,
    indices)`` (see ``ferrotorch-nn/src/embedding.rs``).
    """
    metas = []
    rng = np.random.RandomState(SEED)
    # weight: [vocab=10, emb=4], indices: shape [3] (duplicates exercise
    # the scatter-add path in backward).
    weight = leaf((10, 4), rng=rng)
    indices = torch.tensor([2, 5, 2, 7], dtype=torch.long)
    metas.append(run_fixture(
        "embedding", "vocab10_emb4_idx_2_5_2_7",
        {"weight": weight, "indices": indices.float()},  # store as f32 for binary; rust casts back
        lambda weight, indices: F.embedding(indices.long(), weight),
        {"vocab": 10, "embedding_dim": 4, "indices": [2, 5, 2, 7]},
    ))
    return metas


def pin_attention() -> list[dict[str, Any]]:
    """Composed scaled-dot-product attention: Q@K^T / sqrt(d) -> softmax -> @V.

    Verifies the autograd graph through the canonical attention chain,
    which is the load-bearing path for transformer blocks. Q/K/V each
    have requires_grad, so backward produces three gradients.
    """
    metas = []
    rng = np.random.RandomState(SEED)
    # [B=2, T=3, d=4]
    q = leaf((2, 3, 4), rng=rng)
    k = leaf((2, 3, 4), rng=np.random.RandomState(SEED + 1))
    v = leaf((2, 3, 4), rng=np.random.RandomState(SEED + 2))
    scale = 1.0 / np.sqrt(4)

    def fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Q @ K^T  -> [B, T, T]
        scores = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)
        return out

    metas.append(run_fixture(
        "attention", "B2_T3_d4_unmasked",
        {"q": q, "k": k, "v": v},
        fwd,
        {"batch": 2, "seq_len": 3, "head_dim": 4, "scale": float(scale)},
    ))
    return metas


# ---------------------------------------------------------------------------
# Main: regenerate every fixture, then bundle + upload.
# ---------------------------------------------------------------------------


GENERATORS: list[Callable[[], list[dict[str, Any]]]] = [
    pin_matmul_2d,
    pin_bmm,
    pin_linear,
    pin_relu,
    pin_gelu,
    pin_silu,
    pin_sigmoid,
    pin_tanh,
    pin_softmax,
    pin_log_softmax,
    pin_sum_dim,
    pin_mean_dim,
    pin_add,
    pin_mul,
    pin_sub,
    pin_div,
    pin_log,
    pin_exp,
    pin_pow,
    pin_reshape,
    pin_transpose,
    pin_cat,
    pin_embedding,
    pin_attention,
]


def build_bundle_tar() -> tuple[Path, str]:
    """Pack every fixture under FIX_DIR into bundle.tar; return (path, sha256)."""
    bundle = WORK_DIR / "bundle.tar"
    with tarfile.open(bundle, "w") as tar:
        tar.add(FIX_DIR, arcname="fixtures")
    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    return bundle, sha


def upload_bundle(bundle: Path, fixtures: list[dict[str, Any]]) -> bool:
    if os.environ.get("FERROTORCH_PIN_UPLOAD", "1") == "0":
        print("[pin] FERROTORCH_PIN_UPLOAD=0, skipping upload", flush=True)
        return True
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        print(
            "[pin] huggingface_hub not installed — skipping upload "
            "(set FERROTORCH_PIN_UPLOAD=0 to silence)",
            file=sys.stderr,
            flush=True,
        )
        return True
    try:
        create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
        # Write the index.json + bundle.tar into WORK_DIR alongside the
        # fixtures/ tree, then upload the whole folder in ONE commit to
        # avoid the HF 128-commits/hour rate limit (originally hit when
        # this script did one upload_file per fixture file).
        idx_path = WORK_DIR / "index.json"
        idx_path.write_text(json.dumps({"fixtures": fixtures}, indent=2))
        # bundle is already in WORK_DIR.
        upload_folder(
            folder_path=str(WORK_DIR),
            repo_id=HF_REPO_ID,
            repo_type="model",
            allow_patterns=[
                "bundle.tar",
                "index.json",
                "fixtures/**",
            ],
        )
        api = HfApi()
        files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="model")
        print(f"[pin] uploaded to {HF_REPO_ID}. Repo files ({len(files)}):", flush=True)
        for fname in sorted(files)[:20]:
            print(f"[pin]   - {fname}", flush=True)
        if len(files) > 20:
            print(f"[pin]   ... and {len(files) - 20} more", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[pin] HF upload failed: {exc!r}", file=sys.stderr, flush=True)
        return False


def main() -> int:
    print(f"[pin] torch.__version__ = {torch.__version__}", flush=True)
    print(f"[pin] WORK_DIR = {WORK_DIR}", flush=True)

    # Reset the fixture tree so re-runs are clean.
    if FIX_DIR.exists():
        for root, _, files in os.walk(FIX_DIR, topdown=False):
            for fname in files:
                (Path(root) / fname).unlink()

    fixtures: list[dict[str, Any]] = []
    for gen in GENERATORS:
        fixtures.extend(gen())

    print(f"\n[pin] {len(fixtures)} fixtures generated across "
          f"{len(GENERATORS)} ops", flush=True)

    bundle, sha = build_bundle_tar()
    print(f"[pin] wrote {bundle} ({bundle.stat().st_size} bytes)", flush=True)
    print(f"[pin] SHA-256 (registry pin): {sha}", flush=True)

    if not upload_bundle(bundle, fixtures):
        return 2

    print(f"\n[pin] DONE. SHA-256 for registry.rs pin: {sha}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
