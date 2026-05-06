#!/usr/bin/env python3
"""Regenerate ferrotorch-tokenize conformance fixtures.

This script downloads a Llama-3-family `tokenizer.json` from HuggingFace,
runs the canonical `tokenizers.Tokenizer` against a curated corpus, and
writes the encode/decode reference outputs to a JSON fixture the Rust
conformance test compares against.

Phase 1 of the workspace conformance proof (issue #758):
ferrotorch-tokenize wraps the same `tokenizers` library that
`transformers.AutoTokenizer` uses, so bit-identical output between the
Rust wrapper and the Python reference is the conformance proof.

# Usage from WSL (preferred — Linux-native after #777):
#   python3 /home/doll/ferrotorch/scripts/regenerate_tokenize_fixtures.py
#
# Required Python deps (installed in WSL via `pip install --user` per #777):
#   tokenizers>=0.22  (must match the Rust workspace's `tokenizers` crate)
#   huggingface_hub>=1.0
#
# Fallback usage via the Windows host Python (only if WSL install is
# unavailable; this was the original Path-2 workflow before #777):
#   /mnt/c/Users/texas/AppData/Local/Programs/Python/Python312/python.exe \
#     /home/doll/ferrotorch/scripts/regenerate_tokenize_fixtures.py
#
# The script writes:
#   ferrotorch-tokenize/tests/conformance/assets/llama3_tokenizer.json
#       (the raw HF tokenizer.json — committed)
#   ferrotorch-tokenize/tests/conformance/fixtures/llama3.json
#       (the reference encode/decode outputs — committed)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path

# Repo root resolution: the script lives at <repo>/scripts/, so the
# parent of __file__'s parent is the repo root regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOKENIZE_DIR = _REPO_ROOT / "ferrotorch-tokenize"
_CONF_DIR = _TOKENIZE_DIR / "tests" / "conformance"
_ASSETS_DIR = _CONF_DIR / "assets"
_FIXTURES_DIR = _CONF_DIR / "fixtures"

# Fallback chain: try the gated full-fat Llama-3 first, then the ungated
# 3.2-1B variant (same tokenizer.json structure, no auth required).
_REPO_CANDIDATES = (
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "unsloth/Llama-3.2-1B-Instruct",  # ungated mirror, identical tokenizer
)


def _import_tokenizers():
    try:
        from tokenizers import Tokenizer  # type: ignore  # noqa: PLC0415
    except ImportError as e:
        sys.stderr.write(
            "ERROR: `tokenizers` Python package not importable. Install with:\n"
            "  pip install tokenizers huggingface_hub\n"
            f"Original error: {e}\n"
        )
        sys.exit(2)
    return Tokenizer


def _import_hf_hub():
    try:
        from huggingface_hub import hf_hub_download  # type: ignore  # noqa: PLC0415
    except ImportError as e:
        sys.stderr.write(
            "ERROR: `huggingface_hub` Python package not importable.\n"
            f"Original error: {e}\n"
        )
        sys.exit(2)
    return hf_hub_download


def _download_tokenizer_json(asset_path: Path) -> tuple[str, str]:
    """Download tokenizer.json, return (repo_used, local_path).

    Tries each candidate repo in order; falls back to the next on
    GatedRepoError / RepositoryNotFoundError. The downloaded file is
    copied to `asset_path` (committed to the repo).
    """
    hf_hub_download = _import_hf_hub()
    last_err: Exception | None = None
    for repo in _REPO_CANDIDATES:
        try:
            print(f"[regen] trying tokenizer.json from {repo}", file=sys.stderr)
            local = hf_hub_download(repo_id=repo, filename="tokenizer.json")
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy by reading + writing (avoids cross-device-link failures
            # when HF cache is on a different mount than the repo).
            asset_path.write_bytes(Path(local).read_bytes())
            print(f"[regen] using tokenizer from {repo} ({asset_path})", file=sys.stderr)
            return repo, str(asset_path)
        except Exception as e:  # noqa: BLE001 — we want to fall through on any HF error
            last_err = e
            print(f"[regen]   -> failed: {type(e).__name__}: {e}", file=sys.stderr)
            continue
    sys.stderr.write(
        f"ERROR: could not download tokenizer.json from any candidate repo "
        f"({_REPO_CANDIDATES}). Last error: {last_err!r}\n"
    )
    sys.exit(3)


# Curated test corpus. Each entry exercises a distinct property of the
# tokenizer (whitespace, multi-byte, RTL, ZWJ joiners, special tokens, etc.).
# Comments document what each entry probes; do not reorder without updating
# the surface index of any test that pins a specific case.
_TEST_INPUTS: list[str] = [
    # Empty
    "",
    # ASCII baseline
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    # Multi-line
    "line one\nline two\nline three",
    # CJK
    "你好世界",  # 你好世界
    "日本語のテキスト",  # 日本語のテキスト
    "한국어 텍스트",  # 한국어 텍스트
    # Arabic / RTL
    "مرحبا بالعالم",  # مرحبا بالعالم
    # Cyrillic
    "Привет, мир",  # Привет, мир
    # Emoji
    "\U0001f980 + \U0001f40d = ❤️",  # 🦀 + 🐍 = ❤️
    "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466",  # 👨‍👩‍👧‍👦 (ZWJ family)
    # Mixed scripts
    "Rust \U0001f980 Python \U0001f40d 中文",
    # Invisible / boundary characters
    "﻿hello",  # BOM
    "non breaking",  # NBSP
    # Long string (length stress)
    "abcdefghij" * 1000,
    # Numbers / scientific
    "3.14159 1e-10 -42",
    # Code-shaped
    'fn main() { println!("hello"); }',
    # Llama-3 special tokens (in the vocab as added tokens)
    "<|begin_of_text|>user message<|eot_id|>",
    # Whitespace edges
    "   leading",
    "trailing   ",
    "\t\ttabs\t\t",
]


def _generate_fixture(tokenizer, repo_used: str, asset_relpath: str) -> dict:
    """Build the fixture dict from `tokenizer` against `_TEST_INPUTS`."""
    import tokenizers as _tok_pkg  # type: ignore  # noqa: PLC0415

    tokenizers_version = getattr(_tok_pkg, "__version__", "unknown")
    # transformers is optional — we record its version as metadata when
    # importable, but the conformance suite uses `tokenizers` directly so
    # transformers is not load-bearing. Record the absence explicitly
    # rather than swallowing the import error silently.
    try:
        import transformers as _tr_pkg  # type: ignore  # noqa: PLC0415

        transformers_version = getattr(_tr_pkg, "__version__", "unknown")
    except ImportError as _imp_err:
        transformers_version = f"not-installed ({_imp_err.__class__.__name__})"

    cases = []
    for s in _TEST_INPUTS:
        enc_special = tokenizer.encode(s, add_special_tokens=True).ids
        enc_no_special = tokenizer.encode(s, add_special_tokens=False).ids
        # Decode the with-special encoding two ways: keeping or skipping
        # special tokens. The Rust conformance test asserts both.
        dec_with_special_keep = tokenizer.decode(
            enc_special, skip_special_tokens=False
        )
        dec_with_special_skip = tokenizer.decode(
            enc_special, skip_special_tokens=True
        )
        # Sanity round-trip on the no-special encoding so the fixture
        # records the canonical "decode the bare ids" answer.
        dec_no_special = tokenizer.decode(enc_no_special, skip_special_tokens=False)
        cases.append(
            {
                "input": s,
                "encode_with_special": enc_special,
                "encode_no_special": enc_no_special,
                "decode_with_special_keep": dec_with_special_keep,
                "decode_with_special_skip": dec_with_special_skip,
                "decode_no_special": dec_no_special,
            }
        )

    fixture = {
        "metadata": {
            "tokenizer_repo": repo_used,
            "tokenizer_path": asset_relpath,
            "tokenizers_version": tokenizers_version,
            "transformers_version": transformers_version,
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "num_test_cases": len(cases),
            "schema_version": 1,
        },
        "vocab_size_with_added_tokens": tokenizer.get_vocab_size(with_added_tokens=True),
        "vocab_size_no_added_tokens": tokenizer.get_vocab_size(with_added_tokens=False),
        "test_cases": cases,
    }
    return fixture


def main() -> int:
    Tokenizer = _import_tokenizers()

    _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    asset_path = _ASSETS_DIR / "llama3_tokenizer.json"
    repo_used, local = _download_tokenizer_json(asset_path)

    print(f"[regen] loading tokenizer from {local}", file=sys.stderr)
    tokenizer = Tokenizer.from_file(str(asset_path))

    asset_relpath = os.path.relpath(asset_path, _TOKENIZE_DIR).replace("\\", "/")
    fixture = _generate_fixture(tokenizer, repo_used, asset_relpath)

    out_path = _FIXTURES_DIR / "llama3.json"
    out_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[regen] wrote {out_path} ({fixture['metadata']['num_test_cases']} cases, "
        f"vocab={fixture['vocab_size_with_added_tokens']})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
