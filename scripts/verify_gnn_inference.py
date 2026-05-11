#!/usr/bin/env python3
"""Verify ferrotorch pretrained GCN-on-Cora inference against the
`torch_geometric` reference (#1157).

For each pinned graph model in the `ferrotorch/*` HF org this script:

  1. Loads the Cora dataset via `torch_geometric.datasets.Planetoid`.
  2. Builds the upstream model (`GCNConv(1433, 16) -> ReLU -> GCNConv(16, 7)`)
     and loads the pinned `model.safetensors` from
     `huggingface.co/ferrotorch/gcn-cora` into it (the pin script
     wrote PyG's exact `state_dict()` layout, so this is a 1:1 load
     with no key remap).
  3. Runs one eval-mode full-graph forward to obtain reference logits
     `[N, num_classes]`.
  4. Writes the input fixtures (`x: [N, F]` f32, `edge_index: [2, E]` i64)
     to local files in the `[u32 ndim][u32 × ndim shape][<dtype> data]`
     little-endian format the Rust example expects.
  5. Invokes the Rust binary
     (`cargo run -p ferrotorch-graph --release --example gcn_inference_dump`)
     with the local input paths and reads back the dumped `[N, C]` logits.
  6. Computes:
       - `cosine_sim`   — flat cosine over all `N * C` entries
       - `max_abs`      — max absolute element-wise diff
       - `argmax_agree` — `(rust.argmax(-1) == ref.argmax(-1)).mean()`
       - `test_acc_rust` / `test_acc_ref` — accuracies on Cora test_mask
     and compares each against the per-model tolerance in `TOL`.
  7. Prints a one-line verdict per model and writes a JSON report.

The tolerances are intentionally loose enough to absorb f32 accumulation
noise on a 2708-node graph (Σ-over-neighbours sums hundreds of terms)
but tight enough that a real bug — wrong normalization, missing self-
loop, transposed weight, etc. — will fail loudly.

Usage:
  python3 scripts/verify_gnn_inference.py [--models gcn-cora,...] [--quiet]
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
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path("/tmp/ferrotorch_verify_gnn")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Per-model tolerances. The cosine-sim floor and `argmax_agree` floor
# are the real correctness signal — `max_abs` only catches whole-graph
# drift, but with N=2708 nodes and accumulating up to ~deg(v) terms per
# node, a small per-edge f32 epsilon can pile up to a noticeable
# absolute number even when the model is bitwise correct.
TOL: dict[str, dict[str, Any]] = {
    "gcn-cora": dict(
        cosine_sim_min=0.999,
        max_abs=0.5,
        argmax_agree_pct=99.0,
        in_features=1433,
        hidden=16,
        num_classes=7,
    ),
}


# ---------------------------------------------------------------------------
# Reference model — identical class definition to the pin script.
# ---------------------------------------------------------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_features: int, hidden: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# ---------------------------------------------------------------------------
# Binary dump helpers.
# ---------------------------------------------------------------------------
def dump_f32(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    with path.open("wb") as f:
        f.write(struct.pack("<I", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<I", int(d)))
        f.write(arr.tobytes(order="C"))


def dump_i64(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr.astype(np.int64, copy=False))
    with path.open("wb") as f:
        f.write(struct.pack("<I", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<I", int(d)))
        f.write(arr.tobytes(order="C"))


def read_dump_f32(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    (ndim,) = struct.unpack_from("<I", raw, 0)
    off = 4
    shape = struct.unpack_from(f"<{ndim}I", raw, off)
    off += 4 * ndim
    n = 1
    for s in shape:
        n *= int(s)
    expect = off + 4 * n
    if len(raw) != expect:
        raise ValueError(
            f"dump {path}: header claims shape={shape} -> {expect} bytes "
            f"but file is {len(raw)} bytes"
        )
    flat = np.frombuffer(raw, dtype="<f4", count=n, offset=off)
    return flat.reshape([int(s) for s in shape]).astype(np.float32, copy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Hub fetch (best-effort fallback to local cache directory).
# ---------------------------------------------------------------------------
def fetch_safetensors(model_name: str) -> Path:
    """Resolve `model.safetensors` for `ferrotorch/<model_name>`.

    Prefers `huggingface_hub.hf_hub_download` (mirrors ferrotorch's
    runtime cache); falls back to the local `/tmp/ferrotorch_pin_*`
    dir if HF is unreachable.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None  # type: ignore

    repo_id = f"ferrotorch/{model_name}"
    if hf_hub_download is not None:
        try:
            return Path(hf_hub_download(repo_id, "model.safetensors"))
        except Exception as e:  # noqa: BLE001
            print(f"  HF download failed: {e!r}; trying local cache", flush=True)
    local = Path(f"/tmp/ferrotorch_pin_{model_name.replace('-', '_')}/model.safetensors")
    if local.is_file():
        return local
    raise RuntimeError(f"could not locate model.safetensors for {repo_id}")


# ---------------------------------------------------------------------------
# Rust binary invocation.
# ---------------------------------------------------------------------------
def run_rust_dump(
    model_name: str,
    output_bin: Path,
    x_bin: Path,
    ei_bin: Path,
    in_features: int,
    hidden: int,
    num_classes: int,
) -> dict[str, Any]:
    cmd = [
        "cargo", "run", "-p", "ferrotorch-graph", "--release",
        "--example", "gcn_inference_dump", "--",
        "--model", model_name,
        "--x-bin", str(x_bin),
        "--edge-index-bin", str(ei_bin),
        "--in-features", str(in_features),
        "--hidden", str(hidden),
        "--num-classes", str(num_classes),
        "--output", str(output_bin),
    ]
    print(f"  running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"rust dump failed ({proc.returncode}); stderr above")
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
# Per-model evaluation.
# ---------------------------------------------------------------------------
@dataclass
class ModelVerdict:
    name: str
    passed: bool
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)


def verify_one(name: str, quiet: bool) -> ModelVerdict:
    print(f"\n=== {name} ===", flush=True)
    tol = TOL[name]

    # -- 1. Cora dataset. ---------------------------------------------------
    dataset = Planetoid(root=str(CACHE_DIR / "cora"), name="Cora")
    data = dataset[0]
    if data.num_features != tol["in_features"]:
        return ModelVerdict(
            name=name, passed=False,
            summary=f"Cora features {data.num_features} != expected {tol['in_features']}",
        )

    # -- 2. Reference forward. ----------------------------------------------
    print("  loading pinned safetensors…", flush=True)
    weights_path = fetch_safetensors(name)
    state = load_file(str(weights_path))
    model = GCN(tol["in_features"], tol["hidden"], tol["num_classes"])
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        ref_logits = model(data.x, data.edge_index)
    ref_pred = ref_logits.argmax(dim=1)
    ref_test_acc = (
        (ref_pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    )
    print(
        f"  ref logits: shape={list(ref_logits.shape)} "
        f"test_acc={ref_test_acc:.4f}",
        flush=True,
    )

    # -- 3. Write inputs for the Rust binary. -------------------------------
    x_bin = CACHE_DIR / f"{name}_x.bin"
    ei_bin = CACHE_DIR / f"{name}_edge_index.bin"
    dump_f32(x_bin, data.x.numpy())
    dump_i64(ei_bin, data.edge_index.numpy())

    # -- 4. Run the Rust binary. --------------------------------------------
    output_bin = CACHE_DIR / f"{name}_rust_logits.bin"
    verdict = run_rust_dump(
        name, output_bin, x_bin, ei_bin,
        tol["in_features"], tol["hidden"], tol["num_classes"],
    )
    rust_logits = read_dump_f32(output_bin)
    if list(rust_logits.shape) != list(ref_logits.shape):
        return ModelVerdict(
            name=name, passed=False,
            summary=f"shape mismatch: rust={list(rust_logits.shape)} "
                    f"ref={list(ref_logits.shape)}",
        )

    # -- 5. Metrics. ---------------------------------------------------------
    ref_np = ref_logits.detach().cpu().numpy().astype(np.float32)
    diff = rust_logits - ref_np
    max_abs = float(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())
    cos = cosine_similarity(rust_logits, ref_np)
    rust_pred = rust_logits.argmax(axis=1)
    argmax_agree_pct = float((rust_pred == ref_pred.numpy()).mean() * 100.0)
    test_mask = data.test_mask.numpy()
    test_acc_rust = float(
        (rust_pred[test_mask] == data.y.numpy()[test_mask]).mean()
    )

    failures: list[str] = []
    if cos < tol["cosine_sim_min"]:
        failures.append(f"cosine_sim={cos:.6f} < {tol['cosine_sim_min']}")
    if max_abs > tol["max_abs"]:
        failures.append(f"max_abs={max_abs:.6f} > {tol['max_abs']}")
    if argmax_agree_pct < tol["argmax_agree_pct"]:
        failures.append(
            f"argmax_agree={argmax_agree_pct:.2f}% < {tol['argmax_agree_pct']}%"
        )

    passed = not failures
    summary = (
        f"cosine_sim={cos:.6f}, max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, "
        f"argmax_agree={argmax_agree_pct:.2f}%, "
        f"test_acc_rust={test_acc_rust:.4f} (ref {ref_test_acc:.4f})"
    )
    if failures:
        summary += " — FAIL: " + "; ".join(failures)
    if not quiet:
        print(f"  metrics: {summary}", flush=True)

    return ModelVerdict(
        name=name, passed=passed, summary=summary,
        detail=dict(
            shape=list(rust_logits.shape),
            cosine_sim=cos,
            max_abs=max_abs,
            mean_abs=mean_abs,
            argmax_agree_pct=argmax_agree_pct,
            test_acc_rust=test_acc_rust,
            test_acc_ref=ref_test_acc,
            rust_verdict=verdict,
            failures=failures,
        ),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models", default=",".join(TOL.keys()),
        help="Comma-separated subset of model names to verify.",
    )
    p.add_argument("--quiet", action="store_true",
                   help="Only print the final per-model verdict line.")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in TOL:
            print(f"unknown model {m!r}. Known: {list(TOL)}", file=sys.stderr)
            return 2

    verdicts: list[ModelVerdict] = []
    for m in models:
        try:
            v = verify_one(m, quiet=args.quiet)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            v = ModelVerdict(
                name=m, passed=False, summary=f"exception: {e!r}",
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
        v.name: {"passed": v.passed, "summary": v.summary, "detail": v.detail}
        for v in verdicts
    }
    report_path = CACHE_DIR / "verify_gnn_inference_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    if not args.quiet:
        print(f"\nDetailed report: {report_path}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
