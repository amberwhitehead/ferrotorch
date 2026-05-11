#!/usr/bin/env python3
"""Verify ferrotorch's `Optimizer::step()` math against torch.optim
trajectories, using the frozen-gradient fixtures pinned at
`ferrotorch/optimizer-trajectories-v1`.

Phase C.2 of real-artifact-driven development (#1155). Companion to:
  * `scripts/pin_pretrained_optimizer_trajectories.py` (the pin)
  * `ferrotorch-optim/examples/optimizer_trajectory_dump.rs`
  * `ferrotorch-optim/tests/conformance_optimizer_trajectories.rs`

For each (optimizer, config) tuple in the matrix this script:

  1. Downloads the per-config subfolder
     (`<config>/initial_params.bin`, `<config>/gradients_step_K.bin` for
     `K in 0..9`, `<config>/final_params.bin`, `<config>/meta.json`) from
     the HF mirror via `huggingface_hub.hf_hub_download`.
  2. Invokes the matching Rust example:
       `cargo run -p ferrotorch-optim --release --example
        optimizer_trajectory_dump -- --fixture-dir <local> ...`
  3. Reads the Rust-side `final_params.bin` dump and the reference
     `final_params.bin` shipped by the mirror.
  4. Compares per parameter:
       - `cosine_sim` — `(rust @ ref) / (||rust|| * ||ref||)`
       - `max_abs`    — `max(abs(rust - ref))`
       - `rel_err`    — `||rust - ref|| / ||ref||`
  5. Applies the PASS gate (per Phase C.2 spec):
       - max_abs    <= 1e-5
       - cosine_sim >= 0.99999

Usage:
  python3 scripts/verify_optimizer_inference.py [--configs sgd_plain,adam_default,...]
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
CACHE_DIR = Path("/tmp/ferrotorch_verify_optimizer")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_REPO_ID = "ferrotorch/optimizer-trajectories-v1"
NUM_STEPS = 10
NUM_PARAMS = 6

# PASS gate per the Phase C.2 dispatch (#1155). Identical-gradient
# f32-vs-f32 round-trip should be tight enough that any divergence is a
# real algorithm bug; tolerance loosening below these floors is
# explicitly forbidden.
COSINE_MIN = 0.99999
MAX_ABS_CAP = 1e-5

# Matrix of (config_name, optimizer_family) tuples. Must match the
# specs declared in pin_pretrained_optimizer_trajectories.py.
CONFIGS: list[tuple[str, str]] = [
    ("sgd_plain", "SGD"),
    ("sgd_momentum", "SGD"),
    ("sgd_nesterov", "SGD"),
    ("adam_default", "Adam"),
    ("adam_explicit", "Adam"),
    ("adamw_decoupled", "AdamW"),
    ("rmsprop_default", "RMSprop"),
    ("rmsprop_momentum", "RMSprop"),
    ("adagrad_default", "Adagrad"),
    ("adagrad_explicit", "Adagrad"),
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


def fetch_fixture(config_name: str) -> Path:
    """Download every file the Rust example needs into the HF cache and
    return the absolute path to the per-config snapshot directory.

    `hf_hub_download` materialises files as symlinks under
    `<cache>/models--<repo>/snapshots/<rev>/<config_name>/...` pointing
    at content-addressed blobs in `<cache>/.../blobs/<sha>`.
    `Path.resolve()` would follow the symlink to the blob and give us
    `blobs/`, whose siblings are unrelated blobs from other configs —
    not the per-config snapshot dir the Rust example expects. We keep
    the symlink path (no `resolve()`) so `parent` is the snapshot
    subfolder."""
    needed = ["meta.json", "initial_params.bin", "final_params.bin"]
    needed += [f"gradients_step_{k}.bin" for k in range(NUM_STEPS)]
    parent: Path | None = None
    for fn in needed:
        local = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"{config_name}/{fn}",
        )
        # Intentionally NOT `.resolve()` — see docstring.
        p = Path(local).absolute()
        if parent is None:
            parent = p.parent
        elif p.parent != parent:
            raise RuntimeError(
                f"{config_name}: HF cached files for the same fixture into "
                f"distinct dirs ({parent} vs {p.parent}); this should not "
                "happen — hf_hub_download materialises per-repo subfolders."
            )
    if parent is None:
        raise RuntimeError(f"{config_name}: hf_hub_download yielded no files")
    return parent


# ---------------------------------------------------------------------------
# Cargo example dispatch.
# ---------------------------------------------------------------------------


def build_rust_example_once() -> None:
    """Pre-build the example so per-config invocations don't repeatedly
    invoke cargo's build path."""
    cmd = [
        "cargo", "build", "-p", "ferrotorch-optim", "--release",
        "--example", "optimizer_trajectory_dump",
    ]
    print(f"  building Rust example once: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"cargo build failed ({proc.returncode})")


def run_rust_dump(
    config_name: str,
    optimizer: str,
    fixture_dir: Path,
    output_bin: Path,
) -> dict[str, Any]:
    cmd = [
        "cargo", "run", "-q", "-p", "ferrotorch-optim", "--release",
        "--example", "optimizer_trajectory_dump", "--",
        "--fixture-dir", str(fixture_dir),
        "--optimizer", optimizer,
        "--config-name", config_name,
        "--output", str(output_bin),
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
# Metric computations + verdict.
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class PerTensorMetric:
    name: str
    shape: tuple[int, ...]
    max_abs: float
    cosine_sim: float
    rel_err: float
    rust_norm: float
    ref_norm: float


@dataclass
class ConfigVerdict:
    name: str
    optimizer: str
    passed: bool
    summary: str
    per_tensor: list[PerTensorMetric] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


PARAM_NAMES = [
    "layer0.weight",
    "layer0.bias",
    "layer1.weight",
    "layer1.bias",
    "layer2.weight",
    "layer2.bias",
]


def verify_one(config_name: str, optimizer: str, quiet: bool) -> ConfigVerdict:
    print(f"\n=== {config_name} ({optimizer}) ===", flush=True)

    # -- 1. Fetch the fixture. --------------------------------------------
    fixture_dir = fetch_fixture(config_name)
    print(f"  fixture: {fixture_dir}")

    # -- 2. Read reference final params. ----------------------------------
    ref_tensors = read_multi_tensor_f32(fixture_dir / "final_params.bin")
    if len(ref_tensors) != NUM_PARAMS:
        return ConfigVerdict(
            name=config_name, optimizer=optimizer, passed=False,
            summary=f"reference has {len(ref_tensors)} tensors, expected {NUM_PARAMS}",
        )

    # -- 3. Run ferrotorch. -----------------------------------------------
    output_bin = CACHE_DIR / f"{config_name}_rust.bin"
    verdict = run_rust_dump(config_name, optimizer, fixture_dir, output_bin)
    rust_tensors = read_multi_tensor_f32(output_bin)
    if len(rust_tensors) != NUM_PARAMS:
        return ConfigVerdict(
            name=config_name, optimizer=optimizer, passed=False,
            summary=f"rust dump has {len(rust_tensors)} tensors, expected {NUM_PARAMS}",
            detail={"rust_dump_verdict": verdict},
        )

    # -- 4. Per-tensor metrics. -------------------------------------------
    metrics: list[PerTensorMetric] = []
    failures: list[str] = []
    worst_max_abs = 0.0
    worst_cos = 1.0
    for i, (rust, ref) in enumerate(zip(rust_tensors, ref_tensors)):
        name = PARAM_NAMES[i]
        if rust.shape != ref.shape:
            failures.append(
                f"{name}: shape rust={rust.shape} != ref={ref.shape}"
            )
            continue
        diff = rust - ref
        max_abs = float(np.abs(diff).max())
        cos = cosine_similarity(rust, ref)
        rust_norm = float(np.linalg.norm(rust))
        ref_norm = float(np.linalg.norm(ref))
        rel_err = (
            float(np.linalg.norm(diff)) / ref_norm if ref_norm > 0 else float("inf")
        )
        metrics.append(
            PerTensorMetric(
                name=name, shape=tuple(int(s) for s in rust.shape),
                max_abs=max_abs, cosine_sim=cos, rel_err=rel_err,
                rust_norm=rust_norm, ref_norm=ref_norm,
            )
        )
        if max_abs > worst_max_abs:
            worst_max_abs = max_abs
        if cos < worst_cos:
            worst_cos = cos
        if max_abs > MAX_ABS_CAP:
            failures.append(f"{name}: max_abs={max_abs:.3e} > {MAX_ABS_CAP:.0e}")
        if cos < COSINE_MIN:
            failures.append(f"{name}: cosine_sim={cos:.6f} < {COSINE_MIN}")

    passed = not failures
    summary = (
        f"worst max_abs={worst_max_abs:.3e}  worst cosine_sim={worst_cos:.7f}"
    )
    if failures:
        summary += "  — FAIL: " + "; ".join(failures)
    if not quiet:
        for m in metrics:
            print(
                f"  {m.name:<14} shape={list(m.shape)} "
                f"max_abs={m.max_abs:.3e} cosine={m.cosine_sim:.7f} "
                f"rel_err={m.rel_err:.3e}"
            )
        print(f"  {summary}")

    return ConfigVerdict(
        name=config_name, optimizer=optimizer, passed=passed,
        summary=summary, per_tensor=metrics,
        detail={
            "rust_dump_verdict": verdict,
            "worst_max_abs": worst_max_abs,
            "worst_cosine_sim": worst_cos,
            "failures": failures,
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
    by_name = {n: o for (n, o) in CONFIGS}
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
                name=config_name, optimizer=by_name[config_name],
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
            "optimizer": v.optimizer,
            "passed": v.passed,
            "summary": v.summary,
            "per_tensor": [
                {
                    "name": m.name,
                    "shape": list(m.shape),
                    "max_abs": m.max_abs,
                    "cosine_sim": m.cosine_sim,
                    "rel_err": m.rel_err,
                    "rust_norm": m.rust_norm,
                    "ref_norm": m.ref_norm,
                }
                for m in v.per_tensor
            ],
            "detail": v.detail,
        }
        for v in verdicts
    }
    report_path = CACHE_DIR / "verify_optimizer_inference_report.json"
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
        ("a", np.arange(6, dtype="<f4").reshape(2, 3)),
        ("b", np.array([1.5], dtype="<f4")),
    ]
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(tensors_in)))
        for _name, arr in tensors_in:
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<I", int(d)))
            f.write(arr.tobytes(order="C"))
    got = read_multi_tensor_f32(path)
    assert len(got) == 2
    assert got[0].shape == (2, 3)
    assert np.allclose(got[0], tensors_in[0][1])
    assert got[1].shape == (1,)
    assert np.allclose(got[1], tensors_in[1][1])
    print("_test_read_multi_tensor_f32: ok")


def _test_cosine() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-9
    c = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, c)) < 1e-9
    print("_test_cosine: ok")


def _self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _test_read_multi_tensor_f32(Path(td))
    _test_cosine()
    print("self-test: all assertions passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(_self_test())
    sys.exit(main())
