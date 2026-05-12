#!/usr/bin/env python3
"""Verify ferrotorch's full training stack (forward + MSE loss + autograd
backward + Adam optimizer + sequential batching) against torch by
comparing per-epoch state_dicts and per-epoch mean losses against the
pinned `ferrotorch/training-trajectory-v1` mirror.

Phase E of real-artifact-driven development (#1161). Companion to:
  * `scripts/pin_pretrained_training_trajectory.py` (the pin)
  * `ferrotorch-train/examples/multi_epoch_train_dump.rs`
  * `ferrotorch-train/tests/conformance_multi_epoch_training.rs`

For each epoch K in 1..=5 the script:

  1. Downloads `initial_state.safetensors`, `X_full.bin`, `y_full.bin`,
     `meta.json`, and `epoch_K_state.safetensors` from the HF mirror.
  2. Invokes the Rust example once (per --models invocation) and reads
     each `epoch_K_state.safetensors` Rust dump back.
  3. Loads the reference safetensors via `safetensors.numpy.load_file`.
  4. For each of the 6 named parameters
     (fc{1,2,3}.{weight,bias}) computes:
       - `cosine_sim` — `(rust @ ref) / (||rust|| * ||ref||)`
       - `max_abs`    — `max(abs(rust - ref))`
       - `rel_err`    — `||rust - ref|| / ||ref||`
  5. Applies the PASS gate (per the #1161 dispatch):
       - max_abs    <= 1e-4
       - cosine_sim >= 0.9999
     for every parameter, plus the per-epoch mean loss reported by the
     Rust example must match `meta.json["epoch_losses"][K-1]` within
     `max_abs <= 1e-4`.

Usage:
  python3 scripts/verify_training_trajectory.py --models training-trajectory-v1
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
from safetensors.numpy import load_file as safe_load


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path("/tmp/ferrotorch_verify_training_trajectory")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# PASS gate per #1161. Identical-input f32-vs-f32 training should land
# inside these bounds for every parameter; loosening below them is
# explicitly forbidden by the dispatch constraints.
COSINE_MIN = 0.9999
MAX_ABS_CAP = 1e-4

EPOCHS = 5
PARAM_NAMES = [
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias",
]

# Registered models. The HF repo for each is `ferrotorch/<name>`.
MODELS: dict[str, str] = {
    "training-trajectory-v1": "ferrotorch/training-trajectory-v1",
}


# ---------------------------------------------------------------------------
# Fixture download — pull the full file list into the HF cache and
# return the absolute path to the snapshot directory the Rust example
# expects.
# ---------------------------------------------------------------------------


def fetch_fixture(repo_id: str) -> Path:
    needed = ["meta.json", "initial_state.safetensors", "X_full.bin", "y_full.bin"]
    needed += [f"epoch_{k}_state.safetensors" for k in range(EPOCHS + 1)]
    parent: Path | None = None
    for fn in needed:
        local = hf_hub_download(repo_id=repo_id, filename=fn)
        # Intentionally NOT `.resolve()` — see the analogous comment in
        # verify_optimizer_inference.py; HF caches files as symlinks
        # under `snapshots/<rev>/` pointing at content-addressed blobs.
        p = Path(local).absolute()
        if parent is None:
            parent = p.parent
        elif p.parent != parent:
            raise RuntimeError(
                f"{repo_id}: HF cached files into distinct dirs "
                f"({parent} vs {p.parent}); this should not happen."
            )
    if parent is None:
        raise RuntimeError(f"{repo_id}: hf_hub_download yielded no files")
    return parent


# ---------------------------------------------------------------------------
# Cargo example dispatch — build once, run once per model.
# ---------------------------------------------------------------------------


def build_rust_example_once() -> None:
    cmd = [
        "cargo", "build", "-p", "ferrotorch-train", "--release",
        "--example", "multi_epoch_train_dump",
    ]
    print(f"  building Rust example once: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"cargo build failed ({proc.returncode})")


def run_rust_dump(fixture_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cargo", "run", "-q", "-p", "ferrotorch-train", "--release",
        "--example", "multi_epoch_train_dump", "--",
        "--fixture-dir", str(fixture_dir),
        "--output-dir", str(output_dir),
    ]
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"rust dump failed ({proc.returncode})")
    json_line: str | None = None
    for line in proc.stdout.splitlines():
        t = line.strip()
        if t.startswith("{") and t.endswith("}"):
            json_line = t
    if json_line is None:
        sys.stderr.write(proc.stdout)
        raise RuntimeError("rust dump did not print a JSON verdict line")
    return json.loads(json_line)


# ---------------------------------------------------------------------------
# Metric helpers.
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
class TensorMetric:
    name: str
    shape: tuple[int, ...]
    max_abs: float
    cosine_sim: float
    rel_err: float


@dataclass
class EpochVerdict:
    epoch: int
    passed: bool
    summary: str
    per_tensor: list[TensorMetric] = field(default_factory=list)
    loss_rust: float | None = None
    loss_ref: float | None = None
    loss_abs_err: float | None = None
    failures: list[str] = field(default_factory=list)


@dataclass
class ModelVerdict:
    name: str
    repo_id: str
    passed: bool
    summary: str
    epochs: list[EpochVerdict] = field(default_factory=list)


def compare_state_dicts(
    rust_path: Path, ref_path: Path, epoch: int,
) -> tuple[list[TensorMetric], list[str]]:
    rust_sd = safe_load(str(rust_path))
    ref_sd = safe_load(str(ref_path))
    missing = [k for k in PARAM_NAMES if k not in rust_sd]
    if missing:
        return [], [f"epoch {epoch}: rust state_dict missing {missing}"]
    missing_ref = [k for k in PARAM_NAMES if k not in ref_sd]
    if missing_ref:
        return [], [f"epoch {epoch}: ref state_dict missing {missing_ref}"]

    metrics: list[TensorMetric] = []
    failures: list[str] = []
    for name in PARAM_NAMES:
        rust = np.asarray(rust_sd[name], dtype=np.float32)
        ref = np.asarray(ref_sd[name], dtype=np.float32)
        if rust.shape != ref.shape:
            failures.append(
                f"epoch {epoch} {name}: shape rust={rust.shape} != ref={ref.shape}"
            )
            continue
        diff = rust - ref
        max_abs = float(np.abs(diff).max())
        cos = cosine_similarity(rust, ref)
        ref_norm = float(np.linalg.norm(ref))
        rel_err = (
            float(np.linalg.norm(diff)) / ref_norm if ref_norm > 0 else float("inf")
        )
        metrics.append(
            TensorMetric(
                name=name,
                shape=tuple(int(s) for s in rust.shape),
                max_abs=max_abs,
                cosine_sim=cos,
                rel_err=rel_err,
            )
        )
        if max_abs > MAX_ABS_CAP:
            failures.append(
                f"epoch {epoch} {name}: max_abs={max_abs:.3e} > {MAX_ABS_CAP:.0e}"
            )
        if cos < COSINE_MIN:
            failures.append(
                f"epoch {epoch} {name}: cosine_sim={cos:.6f} < {COSINE_MIN}"
            )
    return metrics, failures


# ---------------------------------------------------------------------------
# Per-model verification.
# ---------------------------------------------------------------------------


def verify_one(name: str, repo_id: str, quiet: bool) -> ModelVerdict:
    print(f"\n=== {name} ({repo_id}) ===", flush=True)
    fixture_dir = fetch_fixture(repo_id)
    print(f"  fixture: {fixture_dir}")

    meta = json.loads((fixture_dir / "meta.json").read_text())
    ref_losses: list[float] = list(meta["epoch_losses"])
    if len(ref_losses) != EPOCHS:
        return ModelVerdict(
            name=name, repo_id=repo_id, passed=False,
            summary=f"meta.json has {len(ref_losses)} epoch_losses, expected {EPOCHS}",
        )

    output_dir = CACHE_DIR / name / "rust_dump"
    verdict = run_rust_dump(fixture_dir, output_dir)
    rust_losses: list[float] = list(verdict.get("epoch_losses") or [])
    if len(rust_losses) != EPOCHS:
        return ModelVerdict(
            name=name, repo_id=repo_id, passed=False,
            summary=(
                f"rust dump reported {len(rust_losses)} losses, expected {EPOCHS}"
            ),
        )

    epoch_verdicts: list[EpochVerdict] = []
    overall_failures: list[str] = []
    for k in range(1, EPOCHS + 1):
        rust_path = output_dir / f"epoch_{k}_state.safetensors"
        ref_path = fixture_dir / f"epoch_{k}_state.safetensors"
        metrics, failures = compare_state_dicts(rust_path, ref_path, epoch=k)

        loss_rust = rust_losses[k - 1]
        loss_ref = ref_losses[k - 1]
        loss_abs_err = abs(loss_rust - loss_ref)
        if loss_abs_err > MAX_ABS_CAP:
            failures.append(
                f"epoch {k} loss: rust={loss_rust:.10f} ref={loss_ref:.10f} "
                f"abs_err={loss_abs_err:.3e} > {MAX_ABS_CAP:.0e}"
            )

        worst_max_abs = max((m.max_abs for m in metrics), default=0.0)
        worst_cos = min((m.cosine_sim for m in metrics), default=1.0)
        ev = EpochVerdict(
            epoch=k,
            passed=not failures,
            summary=(
                f"worst max_abs={worst_max_abs:.3e} "
                f"worst cosine_sim={worst_cos:.7f} "
                f"loss_abs_err={loss_abs_err:.3e}"
            ),
            per_tensor=metrics,
            loss_rust=loss_rust,
            loss_ref=loss_ref,
            loss_abs_err=loss_abs_err,
            failures=failures,
        )
        epoch_verdicts.append(ev)
        overall_failures.extend(failures)

        if not quiet:
            print(f"  -- epoch {k} --")
            for m in metrics:
                print(
                    f"    {m.name:<14} shape={list(m.shape)} "
                    f"max_abs={m.max_abs:.3e} cosine={m.cosine_sim:.7f} "
                    f"rel_err={m.rel_err:.3e}"
                )
            print(
                f"    loss rust={loss_rust:.10f} ref={loss_ref:.10f} "
                f"abs_err={loss_abs_err:.3e}"
            )
            tag = "PASS" if ev.passed else "FAIL"
            print(f"    epoch {k}: {tag} — {ev.summary}")

    passed = not overall_failures
    summary = (
        f"{sum(1 for e in epoch_verdicts if e.passed)}/{EPOCHS} epochs PASS"
    )
    if overall_failures:
        summary += " — FAIL: " + "; ".join(overall_failures[:6])
        if len(overall_failures) > 6:
            summary += f" (+{len(overall_failures) - 6} more)"
    return ModelVerdict(
        name=name, repo_id=repo_id, passed=passed,
        summary=summary, epochs=epoch_verdicts,
    )


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models", default=",".join(MODELS),
        help="Comma-separated subset of model names to verify.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Only print the per-model summary line.",
    )
    args = p.parse_args()

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in requested:
        if m not in MODELS:
            print(f"unknown model {m!r}. Known: {list(MODELS)}", file=sys.stderr)
            return 2

    build_rust_example_once()

    verdicts: list[ModelVerdict] = []
    for name in requested:
        try:
            v = verify_one(name, MODELS[name], quiet=args.quiet)
        except Exception as e:  # noqa: BLE001
            v = ModelVerdict(
                name=name, repo_id=MODELS[name], passed=False,
                summary=f"exception: {e!r}",
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
            "repo_id": v.repo_id,
            "passed": v.passed,
            "summary": v.summary,
            "epochs": [
                {
                    "epoch": e.epoch,
                    "passed": e.passed,
                    "summary": e.summary,
                    "loss_rust": e.loss_rust,
                    "loss_ref": e.loss_ref,
                    "loss_abs_err": e.loss_abs_err,
                    "failures": e.failures,
                    "per_tensor": [
                        {
                            "name": m.name,
                            "shape": list(m.shape),
                            "max_abs": m.max_abs,
                            "cosine_sim": m.cosine_sim,
                            "rel_err": m.rel_err,
                        }
                        for m in e.per_tensor
                    ],
                }
                for e in v.epochs
            ],
        }
        for v in verdicts
    }
    report_path = CACHE_DIR / "verify_training_trajectory_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    if not args.quiet:
        print(f"\nDetailed report: {report_path}")
    return 1 if any_fail else 0


# ---------------------------------------------------------------------------
# Minimal self-tests for the metric helpers.
# ---------------------------------------------------------------------------


def _test_cosine() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-9
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b)) < 1e-9
    print("_test_cosine: ok")


def _self_test() -> int:
    _test_cosine()
    # Round-trip a fake state_dict to confirm safetensors+numpy linkage.
    from safetensors.numpy import save_file as safe_save
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "x.safetensors"
        safe_save({"fc1.weight": np.zeros((2, 3), dtype=np.float32)}, str(path))
        got = safe_load(str(path))
        assert got["fc1.weight"].shape == (2, 3)
    print("_self_test: ok")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(_self_test())
    sys.exit(main())
