#!/usr/bin/env python3
"""Verify ferrotorch pretrained text-embedding inference against
sentence-transformers reference.

Companion to `scripts/verify_causal_lm_inference.py` but for BERT-family
encoder-only sentence-embedding models. For each pinned model in the
`ferrotorch/*` HF org this script:

  1. Loads the upstream model via
     `sentence_transformers.SentenceTransformer(<repo>)`.
  2. Encodes a frozen sentence (`PARITY_SENTENCE` — same one the pin
     script froze into `_value_parity_input.txt`) with
     `normalize_embeddings=True`.
  3. Invokes the Rust binary
     (`cargo run -p ferrotorch-bert --release --example text_embedding_dump`)
     against the same sentence and reads the dumped `[1, hidden]` f32
     tensor.
  4. Computes:
       - `cosine_sim` — `(rust @ tv) / (||rust|| * ||tv||)`
       - `max_abs`    — `max(abs(rust - tv))`
     and compares each against the per-model tolerance in `TOL`.
  5. Prints a one-line verdict per model and a JSON report.

Tolerances are intentionally tight: at f32 the only divergence between
ferrotorch and sentence-transformers (which share weights byte-for-byte)
is f32 accumulation noise from a different op-order in attention / FFN.
We require `cosine_sim >= 0.999` and `max_abs <= 0.01`.

Usage:
  python3 scripts/verify_text_embedding_inference.py [--models all-MiniLM-L6-v2,...]
                                                     [--quiet]
                                                     [--self-test]

The Rust example must be pre-built (this script will also build it on
first invocation):
  cargo build -p ferrotorch-bert --release --example text_embedding_dump
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
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path("/tmp/ferrotorch_verify_text_embedding")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Match the pin script's frozen sentence.
PARITY_SENTENCE = "The quick brown fox jumps over the lazy dog."


# Per-model tolerances. Tight on purpose — the ferrotorch path consumes
# the same upstream safetensors byte-for-byte, so any drift larger than
# f32 accumulation noise is a bug.
TOL: dict[str, dict[str, Any]] = {
    "all-MiniLM-L6-v2": dict(
        cosine_sim_min=0.999,
        max_abs=0.01,
        embedding_dim=384,
    ),
}

# Upstream HF repo per ferrotorch mirror.
UPSTREAM_REPO: dict[str, str] = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_dump_f32(path: Path) -> np.ndarray:
    """Read `[u32 ndim][u32 × ndim shape][f32 × prod(shape)]` little-endian."""
    raw = path.read_bytes()
    if len(raw) < 4:
        raise ValueError(f"dump {path} truncated (< 4 bytes)")
    (ndim,) = struct.unpack_from("<I", raw, 0)
    off = 4
    if len(raw) < off + 4 * ndim:
        raise ValueError(
            f"dump {path}: header claims ndim={ndim} but only {len(raw)} bytes total"
        )
    shape = struct.unpack_from(f"<{ndim}I", raw, off)
    off += 4 * ndim
    n = 1
    for s in shape:
        n *= int(s)
    expect = off + 4 * n
    if len(raw) != expect:
        raise ValueError(
            f"dump {path}: header claims shape={shape} (expects {expect} bytes) "
            f"but file is {len(raw)} bytes"
        )
    flat = np.frombuffer(raw, dtype="<f4", count=n, offset=off)
    return flat.reshape([int(s) for s in shape]).astype(np.float32, copy=True)


def run_rust_dump(model_name: str, output_bin: Path, sentence: str) -> dict[str, Any]:
    """Invoke the Rust example and parse its stdout JSON verdict line."""
    cmd = [
        "cargo", "run", "-p", "ferrotorch-bert", "--release",
        "--example", "text_embedding_dump", "--",
        "--model", model_name,
        "--output", str(output_bin),
        "--sentence", sentence,
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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors."""
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Per-model evaluation
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
    upstream = UPSTREAM_REPO[name]

    # -- 1. Load reference (sentence-transformers). --------------------------
    print(f"  loading upstream sentence-transformers model {upstream!r}…", flush=True)
    st = SentenceTransformer(upstream)
    enc = st.tokenizer(
        PARITY_SENTENCE, return_tensors="np", add_special_tokens=True,
        padding=False, truncation=False,
    )
    py_token_ids = enc["input_ids"][0].tolist()
    print(
        f"  sentence: {PARITY_SENTENCE!r} -> {len(py_token_ids)} tokens: {py_token_ids}",
        flush=True,
    )
    tv_emb = st.encode([PARITY_SENTENCE], normalize_embeddings=True)
    if tv_emb.ndim != 2 or tv_emb.shape[0] != 1:
        return ModelVerdict(
            name=name, passed=False,
            summary=f"upstream emb shape {tv_emb.shape!r} not [1, H]",
        )
    tv_emb = tv_emb.astype(np.float32, copy=False)
    expected_dim = tol["embedding_dim"]
    if tv_emb.shape[1] != expected_dim:
        return ModelVerdict(
            name=name, passed=False,
            summary=f"upstream dim {tv_emb.shape[1]} != expected {expected_dim}",
        )
    print(
        f"  tv embedding: shape={list(tv_emb.shape)} "
        f"||emb||={float(np.linalg.norm(tv_emb)):.6f}",
        flush=True,
    )

    # Free upstream model.
    del st

    # -- 2. Run ferrotorch. -------------------------------------------------
    output_bin = CACHE_DIR / f"{name}_rust_dump.bin"
    verdict = run_rust_dump(name, output_bin, PARITY_SENTENCE)
    rust_emb = read_dump_f32(output_bin)
    if rust_emb.shape != tv_emb.shape:
        return ModelVerdict(
            name=name, passed=False,
            summary=f"shape mismatch: rust={list(rust_emb.shape)} vs tv={list(tv_emb.shape)}",
        )
    rust_token_ids = list(verdict.get("token_ids", []))
    if rust_token_ids != py_token_ids:
        return ModelVerdict(
            name=name, passed=False,
            summary=(
                f"tokenizer disagreement: rust={rust_token_ids} vs py={py_token_ids}"
            ),
        )

    # -- 3. Compute metrics. ------------------------------------------------
    diff = rust_emb - tv_emb
    max_abs = float(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())
    rust_norm = float(np.linalg.norm(rust_emb))
    cos = cosine_similarity(rust_emb, tv_emb)

    # -- 4. Apply tolerances. -----------------------------------------------
    failures: list[str] = []
    if cos < tol["cosine_sim_min"]:
        failures.append(f"cosine_sim={cos:.6f} < {tol['cosine_sim_min']}")
    if max_abs > tol["max_abs"]:
        failures.append(f"max_abs={max_abs:.6f} > {tol['max_abs']}")

    passed = not failures
    summary = (
        f"cosine_sim={cos:.6f}, max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, "
        f"||rust||={rust_norm:.6f}"
    )
    if failures:
        summary += " — FAIL: " + "; ".join(failures)
    if not quiet:
        print(f"  metrics: {summary}")

    return ModelVerdict(
        name=name, passed=passed, summary=summary,
        detail=dict(
            shape=list(rust_emb.shape),
            cosine_sim=cos,
            max_abs=max_abs,
            mean_abs=mean_abs,
            rust_norm=rust_norm,
            tv_norm=float(np.linalg.norm(tv_emb)),
            token_ids=py_token_ids,
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
        v.name: {
            "passed": v.passed,
            "summary": v.summary,
            "detail": v.detail,
        }
        for v in verdicts
    }
    report_path = CACHE_DIR / "verify_text_embedding_inference_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    if not args.quiet:
        print(f"\nDetailed report: {report_path}")
    return 1 if any_fail else 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _test_read_dump_f32(tmp: Path) -> None:
    path = tmp / "_self_test_dump.bin"
    shape = (1, 4)
    data = np.arange(4, dtype="<f4").reshape(shape)
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(shape)))
        for d in shape:
            f.write(struct.pack("<I", d))
        f.write(data.tobytes(order="C"))
    got = read_dump_f32(path)
    assert got.shape == shape, (got.shape, shape)
    assert np.allclose(got, data), (got, data)
    print("_test_read_dump_f32: ok")


def _test_cosine() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-9
    c = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, c)) < 1e-9
    d = -a
    assert abs(cosine_similarity(a, d) + 1.0) < 1e-9
    print("_test_cosine: ok")


def _self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _test_read_dump_f32(Path(td))
    _test_cosine()
    print("self-test: all assertions passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(_self_test())
    sys.exit(main())
