#!/usr/bin/env python3
"""Verify ferrotorch-core autograd backward parity vs torch.autograd
(#1171, Phase G.5).

Iterates every (op, config) fixture in ``ferrotorch/autograd-parity-v1``
and, for each one:

  1. Drives the `cargo run --example autograd_dump` binary to replay
     the forward + backward through ferrotorch-core's differentiable
     surface.
  2. Compares the rust-dumped gradient(s) against the torch reference
     gradient(s).
  3. Compares the rust forward output against the torch reference
     forward (sanity floor).

Tolerances are gradcheck-style: every comparison must satisfy
``max_abs <= 1e-4`` AND ``cosine_sim >= 0.9999``.

Exit code:
  * 0 — every fixture PASSES
  * 1 — any fixture FAILS or the rust dump command errors out

Usage:
  python3 scripts/verify_autograd_inference.py
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Tolerances.
# ---------------------------------------------------------------------------

MAX_ABS = 1e-4
COS_MIN = 0.9999

REPO_ROOT = Path(__file__).resolve().parent.parent
PIN_FIXTURE_DIR = Path("/tmp/ferrotorch_pin_autograd_parity_v1") / "fixtures"
HF_REPO_ID = "ferrotorch/autograd-parity-v1"

# Order matters for legibility of the per-op verdict block.
OPS_AND_CONFIGS: list[tuple[str, str]] = [
    ("matmul_2d", "8x16_16x4"),
    ("bmm",       "3x4x5_3x5x6"),
    ("linear",    "4x6_out3_bias"),
    ("relu",      "4x8"),
    ("gelu",      "4x8"),
    ("silu",      "4x8"),
    ("sigmoid",   "4x8"),
    ("tanh",      "4x8"),
    ("softmax",   "4x8_lastdim"),
    ("log_softmax","4x8_lastdim"),
    ("sum_dim",   "3x5x7_dim1_nokeep"),
    ("sum_dim",   "3x5x7_dim2_keep"),
    ("mean_dim",  "3x5x7_dim1_nokeep"),
    ("add",       "4x5_same"),
    ("mul",       "4x5_same"),
    ("sub",       "4x5_same"),
    ("div",       "4x5_same"),
    ("log",       "4x5_positive"),
    ("exp",       "4x5"),
    ("pow",       "4x5_exp_2_5"),
    ("reshape",   "2x3x4_to_6x4"),
    ("transpose", "4x6_swap01"),
    ("cat",       "axis0_2_4_1_x3"),
    ("embedding", "vocab10_emb4_idx_2_5_2_7"),
    ("attention", "B2_T3_d4_unmasked"),
]


# ---------------------------------------------------------------------------
# Binary I/O.
# ---------------------------------------------------------------------------


def read_f32_tensor(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    with path.open("rb") as f:
        ndim = struct.unpack("<I", f.read(4))[0]
        shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))
        numel = 1
        for d in shape:
            numel *= d
        data = np.frombuffer(f.read(numel * 4), dtype="<f4")
    return shape, data.reshape(shape).copy() if numel > 0 else data.reshape(shape)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    na = float(np.linalg.norm(af))
    nb = float(np.linalg.norm(bf))
    if na == 0.0 or nb == 0.0:
        # Cosine similarity is undefined when either tensor is the zero
        # vector. Adopt the convention that "both effectively zero" → 1.0
        # (degenerate-but-equal), and only flag mismatch when one side is
        # zero while the other is materially non-zero. The "materially
        # non-zero" threshold is MAX_ABS, matching the absolute-difference
        # tolerance — this covers cases like
        # `sum(softmax(x))` whose VJP is identically zero (the sum of the
        # softmax outputs is constant in x), where torch produces ~1e-8
        # floating-point noise and ferrotorch produces exact zeros. Both
        # are correct; the cosine_sim numeric fallback is what bridges
        # the gap.
        return 1.0 if max(na, nb) <= MAX_ABS else 0.0
    return float(np.dot(af, bf) / (na * nb))


# ---------------------------------------------------------------------------
# Fixture availability.
# ---------------------------------------------------------------------------


def ensure_fixtures() -> Path:
    """Return a directory holding the full fixture tree.

    Prefer the local pin script's WORK_DIR if it exists; otherwise
    download every file under `fixtures/` from the HF mirror.
    """
    if PIN_FIXTURE_DIR.exists() and any(PIN_FIXTURE_DIR.iterdir()):
        return PIN_FIXTURE_DIR
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        print(f"[verify] huggingface_hub unavailable ({exc!r}); cannot fetch fixtures")
        sys.exit(1)
    api = HfApi()
    files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="model")
    target = Path(tempfile.mkdtemp(prefix="ferrotorch_autograd_fixtures_"))
    for f in files:
        if not f.startswith("fixtures/"):
            continue
        local = hf_hub_download(repo_id=HF_REPO_ID, filename=f, repo_type="model")
        # mirror layout under <target>/fixtures/... — strip the "fixtures/" prefix.
        rel = Path(f).relative_to("fixtures")
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, dest)
    return target


# ---------------------------------------------------------------------------
# Rust dump driver.
# ---------------------------------------------------------------------------


def run_rust_dump(op: str, config: str, fixture_dir: Path, dump_dir: Path) -> int:
    cmd = [
        "cargo",
        "run",
        "-p",
        "ferrotorch-core",
        "--release",
        "--example",
        "autograd_dump",
        "--",
        "--op", op,
        "--config", config,
        "--fixture-dir", str(fixture_dir),
        "--output-dir", str(dump_dir),
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[verify] rust dump for {op}/{config} failed:")
        print(res.stdout)
        print(res.stderr)
    return res.returncode


# ---------------------------------------------------------------------------
# Per-fixture verification.
# ---------------------------------------------------------------------------


def compare_one(label: str, rust: np.ndarray, ref: np.ndarray) -> bool:
    if rust.shape != ref.shape:
        print(f"        {label:18s} FAIL shape rust={rust.shape} ref={ref.shape}")
        return False
    ma = max_abs(rust, ref)
    cs = cosine_sim(rust, ref)
    ok = (ma <= MAX_ABS) and (cs >= COS_MIN)
    verdict = "PASS" if ok else "FAIL"
    print(
        f"        {label:18s} {verdict} max_abs={ma:.3e} cosine_sim={cs:.6f}"
    )
    return ok


def verify_fixture(
    op: str,
    config: str,
    fixture_dir: Path,
    rust_dump_dir: Path,
) -> bool:
    fdir = fixture_dir / op / config
    params = json.loads((fdir / "params.json").read_text())
    grad_inputs: list[str] = params["grad_inputs"]

    # Forward output parity.
    rust_fwd_path = rust_dump_dir / "forward_out.bin"
    if not rust_fwd_path.exists():
        print(f"  [{op}/{config}] FAIL — rust did not emit forward_out.bin")
        return False
    _, rust_fwd = read_f32_tensor(rust_fwd_path)
    _, ref_fwd = read_f32_tensor(fdir / "forward_out.bin")
    print(f"  [{op}/{config}]")
    ok_fwd = compare_one("forward_out", rust_fwd, ref_fwd)

    # Per-gradient parity.
    grad_results: list[bool] = []
    for name in grad_inputs:
        rust_g_path = rust_dump_dir / "grads" / f"{name}.bin"
        ref_g_path = fdir / "grads" / f"{name}.bin"
        if not rust_g_path.exists():
            print(f"        grad/{name:12s} FAIL rust dump missing")
            grad_results.append(False)
            continue
        _, rust_g = read_f32_tensor(rust_g_path)
        _, ref_g = read_f32_tensor(ref_g_path)
        grad_results.append(compare_one(f"grad/{name}", rust_g, ref_g))

    return ok_fwd and all(grad_results)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    fixture_dir = ensure_fixtures()
    print(f"[verify] fixture-dir = {fixture_dir}")

    with tempfile.TemporaryDirectory(prefix="ferrotorch_autograd_verify_") as tmp:
        tmp_root = Path(tmp)
        passes: list[tuple[str, str, bool]] = []
        for op, config in OPS_AND_CONFIGS:
            dump_dir = tmp_root / op / config
            dump_dir.mkdir(parents=True, exist_ok=True)
            rc = run_rust_dump(op, config, fixture_dir, dump_dir)
            if rc != 0:
                print(f"  [{op}/{config}] FAIL — rust dump exited rc={rc}")
                passes.append((op, config, False))
                continue
            ok = verify_fixture(op, config, fixture_dir, dump_dir)
            passes.append((op, config, ok))

        print("\n[verify] Per-fixture verdict:")
        for op, config, ok in passes:
            print(f"  {op:12s} {config:32s} {'PASS' if ok else 'FAIL'}")
        all_pass = all(ok for _, _, ok in passes)
        n_pass = sum(1 for _, _, ok in passes if ok)
        print(
            f"\n[verify] OVERALL: {'PASS' if all_pass else 'FAIL'} "
            f"({n_pass}/{len(passes)} fixtures passed)"
        )
        return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
