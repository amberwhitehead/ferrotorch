#!/usr/bin/env python3
"""Pin a pretrained text-embedding checkpoint to the `ferrotorch/*` HF org.

Phase B.1 of real-artifact-driven development (closes issue #1148).

Mirrors `scripts/pin_pretrained_llm_weights.py` but for BERT-family
encoder-only sentence-embedding models. For the chosen model this
script:

1. Downloads the upstream HF safetensors + tokenizer + config plus the
   sentence-transformers pooling config (`1_Pooling/config.json`).
2. Verifies the safetensors key list matches the layout
   `ferrotorch_bert::BertModel::load_hf_state_dict` consumes. Every key
   must either map onto a parameter or appear in the documented
   drop list (`embeddings.position_ids`, `pooler.*`) — the FPN-bias
   silent-drop bug (#1141) burned us once, so every key is accounted
   for.
3. Generates a fixed parity probe:
     - `_value_parity_input.txt`: the verbatim sentence the harness will
       encode.
     - `_value_parity_token_ids.json`: the upstream WordPiece encode
       (with `add_special_tokens=True`).
     - `_value_parity_output.bin`: float32 sentence embedding
       `[1, hidden]` from a fresh
       `SentenceTransformer(<repo>).encode(<sentence>, normalize_embeddings=True)`
       call, dumped in the standard `[u32 ndim][u32 shape][f32 data]`
       little-endian format.
4. Uploads `model.safetensors`, `config.json`, `tokenizer.json`,
   `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`,
   `1_Pooling/config.json`, the parity probe files, and a README to
   `huggingface.co/ferrotorch/<name>`.
5. Hashes the uploaded `model.safetensors` with SHA-256 and prints a
   registry-ready snippet for `ferrotorch-hub/src/registry.rs`.

Usage:
    python3 scripts/pin_pretrained_text_weights.py \
        [--model all-MiniLM-L6-v2] \
        [--dry-run] \
        [--skip-upload] \
        [--out-dir /tmp/ferrotorch_pretrained_text_weights]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Apache 2.0 LICENSE text (verbatim) — included in the uploaded README
# because the upstream model is released under Apache 2.0 and we
# redistribute the weights byte-for-byte.
# ---------------------------------------------------------------------------
APACHE_2_0_LICENSE_NOTICE = """\
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# The sentence the harness will re-encode. Same in pin + verify scripts.
PARITY_SENTENCE = "The quick brown fox jumps over the lazy dog."


@dataclass
class TextModelInfo:
    """One pinnable text-embedding model entry."""

    name: str
    upstream_repo: str
    description: str
    license: str
    param_count: int
    # True if the sentence-transformers pipeline ends with `2_Normalize`.
    normalize: bool


MODELS: dict[str, TextModelInfo] = {
    "all-MiniLM-L6-v2": TextModelInfo(
        name="all-MiniLM-L6-v2",
        upstream_repo="sentence-transformers/all-MiniLM-L6-v2",
        description=(
            "all-MiniLM-L6-v2 (sentence-transformers/all-MiniLM-L6-v2). "
            "BERT-family encoder-only sentence-embedding model, 22M "
            "parameters, 6 layers, hidden=384, intermediate=1536, "
            "num_attention_heads=12, vocab=30522, type_vocab_size=2, "
            "max_position_embeddings=512, post-norm residual, GELU FFN. "
            "Sentence pipeline = mean-pool over attention mask + L2 "
            "normalize. Apache 2.0 license. Pinned as the real-artifact "
            "baseline for sentence-embedding parity vs "
            "`sentence_transformers==5.4.1` (issue #1148)."
        ),
        license="apache-2.0",
        # Real upstream parameter count: 22_713_216
        # = embed(30522*384) + pos(512*384) + type(2*384) + LN(2*384)
        #   + 6 * (
        #       attn(query+key+value each 384*384 + 384 bias, output 384*384 + 384 bias)
        #     + attn_output_LN(2*384)
        #     + intermediate(1536*384 + 1536 bias)
        #     + output(384*1536 + 384 bias) + output_LN(2*384)
        #     )
        # = 11720448 + 196608 + 768 + 768
        #   + 6 * ( 3*(384*384 + 384) + 384*384 + 384 + 2*384
        #           + 1536*384 + 1536 + 384*1536 + 384 + 2*384 )
        # = 11918592 + 6 * ( 442368 + 1152 + 147840 + 768
        #                    + 589824 + 1536 + 589824 + 384 + 768 )
        # = 11918592 + 6 * 1774464
        # = 11918592 + 10646784
        # = 22_565_376
        param_count=22_565_376,
        normalize=True,
    ),
}


# ---------------------------------------------------------------------------
# Expected ferrotorch-bert state-dict key set, parameterised by config.
# Mirrors `BertModel::named_parameters()` exactly.
# ---------------------------------------------------------------------------

def expected_keys_and_shapes(cfg: dict) -> dict[str, list[int]]:
    """Per-parameter shape pin. Refuses any checkpoint whose layout
    diverges from what the loader will consume."""
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    type_vocab = cfg.get("type_vocab_size", 2)
    max_pos = cfg["max_position_embeddings"]
    n_layers = cfg["num_hidden_layers"]

    shapes: dict[str, list[int]] = {
        "embeddings.word_embeddings.weight": [vocab, hidden],
        "embeddings.position_embeddings.weight": [max_pos, hidden],
        "embeddings.token_type_embeddings.weight": [type_vocab, hidden],
        "embeddings.LayerNorm.weight": [hidden],
        "embeddings.LayerNorm.bias": [hidden],
    }
    for i in range(n_layers):
        p = f"encoder.layer.{i}"
        shapes[f"{p}.attention.self.query.weight"] = [hidden, hidden]
        shapes[f"{p}.attention.self.query.bias"] = [hidden]
        shapes[f"{p}.attention.self.key.weight"] = [hidden, hidden]
        shapes[f"{p}.attention.self.key.bias"] = [hidden]
        shapes[f"{p}.attention.self.value.weight"] = [hidden, hidden]
        shapes[f"{p}.attention.self.value.bias"] = [hidden]
        shapes[f"{p}.attention.output.dense.weight"] = [hidden, hidden]
        shapes[f"{p}.attention.output.dense.bias"] = [hidden]
        shapes[f"{p}.attention.output.LayerNorm.weight"] = [hidden]
        shapes[f"{p}.attention.output.LayerNorm.bias"] = [hidden]
        shapes[f"{p}.intermediate.dense.weight"] = [inter, hidden]
        shapes[f"{p}.intermediate.dense.bias"] = [inter]
        shapes[f"{p}.output.dense.weight"] = [hidden, inter]
        shapes[f"{p}.output.dense.bias"] = [hidden]
        shapes[f"{p}.output.LayerNorm.weight"] = [hidden]
        shapes[f"{p}.output.LayerNorm.bias"] = [hidden]
    return shapes


# Keys we intentionally do not consume. The loader drops these; the pin
# script asserts these are the ONLY upstream keys not mapped, so a
# silent state-dict drop (cf. FPN-bias bug #1141) cannot recur.
EXPECTED_DROPPED = {
    "embeddings.position_ids",
    "pooler.dense.weight",
    "pooler.dense.bias",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_f32_2d(t, path: Path) -> None:
    """Dump a 2-D float32 ndarray-like `[1, hidden]` in the same
    little-endian format the Rust example writes."""
    arr = t.reshape(-1).astype("<f4", copy=False)
    shape = list(t.shape)
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(shape)))
        for d in shape:
            f.write(struct.pack("<I", int(d)))
        f.write(arr.tobytes(order="C"))


def convert_one(info: TextModelInfo, out_root: Path) -> tuple[str, Path]:
    """Download, verify, write parity probe. Returns (sha256, model_dir)."""
    print(f"\n=== {info.name} <- {info.upstream_repo} ===", flush=True)

    out_dir = out_root / info.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "1_Pooling").mkdir(parents=True, exist_ok=True)

    upstream_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "vocab.txt",
        "1_Pooling/config.json",
        "modules.json",
        "sentence_bert_config.json",
    ]
    local_paths: dict[str, Path] = {}
    for fn in upstream_files:
        try:
            p = hf_hub_download(repo_id=info.upstream_repo, filename=fn)
        except Exception as e:
            raise SystemExit(
                f"{info.name}: failed to download upstream {fn} from "
                f"{info.upstream_repo}: {e}"
            )
        target = out_dir / fn
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(p).read_bytes())
        local_paths[fn] = target
        print(f"  fetched {fn} -> {target}", flush=True)

    cfg = json.loads(local_paths["config.json"].read_text())
    arch = cfg.get("architectures", [])
    if "BertModel" not in arch:
        raise SystemExit(
            f"{info.name}: upstream architecture {arch!r} is not BertModel "
            f"— ferrotorch-bert cannot load it."
        )
    print(
        f"  config: arch={arch} hidden={cfg['hidden_size']} "
        f"layers={cfg['num_hidden_layers']} heads={cfg['num_attention_heads']} "
        f"vocab={cfg['vocab_size']} pos={cfg['max_position_embeddings']} "
        f"type_vocab={cfg.get('type_vocab_size', 2)} "
        f"hidden_act={cfg.get('hidden_act')}",
        flush=True,
    )

    # ---- Verify safetensors layout. ---------------------------------
    expected_shapes = expected_keys_and_shapes(cfg)
    with safe_open(local_paths["model.safetensors"], framework="pt") as f:
        actual_keys = set(f.keys())
        actual_shapes: dict[str, list[int]] = {
            k: list(f.get_slice(k).get_shape()) for k in actual_keys
        }

    missing = set(expected_shapes) - actual_keys
    if missing:
        raise SystemExit(
            f"{info.name}: ferrotorch-bert expects {len(missing)} keys "
            f"absent from the upstream safetensors. Sample: "
            f"{sorted(missing)[:5]}"
        )
    extra = actual_keys - set(expected_shapes)
    unexpected = extra - EXPECTED_DROPPED
    if unexpected:
        raise SystemExit(
            f"{info.name}: upstream safetensors has {len(unexpected)} keys "
            f"ferrotorch-bert does NOT consume AND are not in the "
            f"documented drop list. Refusing to pin (we will not silently "
            f"drop parameters — see #1141). Sample: {sorted(unexpected)[:5]}"
        )
    for k, exp in expected_shapes.items():
        got = actual_shapes.get(k)
        if got != exp:
            raise SystemExit(
                f"{info.name}: shape mismatch for '{k}': upstream {got} vs "
                f"ferrotorch expects {exp}. Refusing to pin."
            )
    dropped_actually_present = sorted(extra & EXPECTED_DROPPED)
    print(
        f"  state-dict cross-check OK: "
        f"{len(expected_shapes)}/{len(expected_shapes)} keys mapped, "
        f"intentionally dropped: {dropped_actually_present}.",
        flush=True,
    )

    # ---- Cross-check pooling config matches our `normalize` flag. ---
    pooling_cfg = json.loads(local_paths["1_Pooling/config.json"].read_text())
    if not pooling_cfg.get("pooling_mode_mean_tokens", False):
        raise SystemExit(
            f"{info.name}: pooling_mode_mean_tokens is not true in "
            f"1_Pooling/config.json — ferrotorch-bert only implements "
            f"mean pooling. Refusing to pin."
        )
    if pooling_cfg.get("pooling_mode_cls_token", False) or pooling_cfg.get(
        "pooling_mode_max_tokens", False
    ):
        raise SystemExit(
            f"{info.name}: pooling_mode_cls_token / pooling_mode_max_tokens "
            f"is set in 1_Pooling/config.json — ferrotorch-bert only "
            f"implements mean pooling. Refusing to pin."
        )

    # ---- Generate parity probe. --------------------------------------
    print("  generating value-parity probe…", flush=True)
    st = SentenceTransformer(info.upstream_repo)
    # The harness needs to know exactly what tokens went in. Use the
    # tokenizer directly (it's the same one shipped in `tokenizer.json`).
    enc = st.tokenizer(
        PARITY_SENTENCE,
        return_tensors="np",
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    py_token_ids = enc["input_ids"][0].tolist()
    print(f"  sentence: {PARITY_SENTENCE!r} -> {len(py_token_ids)} tokens: {py_token_ids}",
          flush=True)
    emb = st.encode([PARITY_SENTENCE], normalize_embeddings=info.normalize)
    if emb.ndim != 2 or emb.shape[0] != 1:
        raise SystemExit(
            f"{info.name}: sentence_transformers returned shape "
            f"{emb.shape}, expected [1, hidden]"
        )
    hidden = emb.shape[1]
    print(
        f"  reference embedding: shape={list(emb.shape)} "
        f"||emb||={float((emb ** 2).sum() ** 0.5):.6f}",
        flush=True,
    )

    parity_in = out_dir / "_value_parity_input.txt"
    parity_in.write_text(PARITY_SENTENCE + "\n")
    parity_out = out_dir / "_value_parity_output.bin"
    dump_f32_2d(emb, parity_out)
    parity_ids = out_dir / "_value_parity_token_ids.json"
    parity_ids.write_text(json.dumps(py_token_ids))
    print(
        f"  wrote {parity_in.name}, {parity_out.name} "
        f"({parity_out.stat().st_size} bytes), {parity_ids.name}",
        flush=True,
    )

    # ---- SHA. We pin the upstream-equivalent file byte-for-byte. ----
    sha = sha256_of(local_paths["model.safetensors"])
    print(f"  model.safetensors SHA-256: {sha}", flush=True)

    # ---- README. ----------------------------------------------------
    readme_path = out_dir / "README.md"
    readme_path.write_text(render_readme(info, cfg, sha, hidden))
    print(f"  wrote {readme_path}", flush=True)

    return sha, out_dir


def render_readme(info: TextModelInfo, cfg: dict, sha: str, hidden: int) -> str:
    return textwrap.dedent(f"""\
        ---
        license: {info.license}
        tags:
          - sentence-similarity
          - feature-extraction
          - bert
          - ferrotorch
        ---

        # `ferrotorch/{info.name}`

        {info.description}

        ## Provenance

        * Upstream: `{info.upstream_repo}` ({info.license}).
        * Conversion script: [`ferrotorch/scripts/pin_pretrained_text_weights.py`](https://github.com/dollspace/ferrotorch/blob/main/scripts/pin_pretrained_text_weights.py).
        * Ferrotorch issue: <https://github.com/dollspace/ferrotorch/issues/1148>.
        * SHA-256 of `model.safetensors` (this file is pinned in
          `ferrotorch-hub/src/registry.rs`): `{sha}`.
        * Number of trainable parameters: **{info.param_count:,}**.
        * Embedding dimension: **{hidden}**.
        * Config snapshot: hidden={cfg['hidden_size']}, layers={cfg['num_hidden_layers']},
          heads={cfg['num_attention_heads']}, intermediate={cfg['intermediate_size']},
          vocab={cfg['vocab_size']}, max_position_embeddings={cfg['max_position_embeddings']},
          type_vocab_size={cfg.get('type_vocab_size', 2)},
          hidden_act={cfg.get('hidden_act', 'gelu')},
          layer_norm_eps={cfg.get('layer_norm_eps', 1e-12)}.

        ## Value-parity probe

        Three extra files are uploaded so the ferrotorch-side harness can
        reproduce the parity verdict without re-running the upstream
        sentence-transformers model:

        * `_value_parity_input.txt` — verbatim sentence (`"{PARITY_SENTENCE}"`).
        * `_value_parity_token_ids.json` — upstream `tokenizer(...)` output
          for that sentence with `add_special_tokens=True`.
        * `_value_parity_output.bin` — float32 sentence embedding
          dumped from `SentenceTransformer.encode(..., normalize_embeddings=True)`.
          Format: `[u32 ndim][u32 × ndim shape][f32 × prod(shape) data]`
          little-endian (matches the vision / causal-LM dumps).

        ## How to load

        ```rust
        use ferrotorch_bert::{{BertConfig, HfBertConfig, load_sentence_transformer}};
        use ferrotorch_hub::{{HubCache, hf_download_model}};

        let cache = HubCache::with_default_dir();
        let repo_dir = hf_download_model("ferrotorch/{info.name}", "main", &cache)?;
        let hf_cfg = HfBertConfig::from_file(repo_dir.join("config.json"))?;
        let cfg = BertConfig::from_hf(&hf_cfg)?;
        let (st, _report) = load_sentence_transformer::<f32>(
            &repo_dir.join("model.safetensors"),
            cfg,
            /* normalize = */ {str(info.normalize).lower()},
            /* strict   = */ false,  // upstream has pooler.* + position_ids
        )?;
        ```

        ## Upstream license

        ```
{textwrap.indent(APACHE_2_0_LICENSE_NOTICE, '        ')}
        ```
    """)


def hf_upload(info: TextModelInfo, out_dir: Path) -> None:
    api = HfApi()
    repo_id = f"ferrotorch/{info.name}"
    print(f"  uploading to https://huggingface.co/{repo_id}", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "model.safetensors",
        "modules.json",
        "sentence_bert_config.json",
        "1_Pooling/config.json",
        "_value_parity_input.txt",
        "_value_parity_token_ids.json",
        "_value_parity_output.bin",
        "README.md",
    ]
    for fname in files:
        p = out_dir / fname
        if not p.exists():
            print(f"    skip (missing locally): {fname}", flush=True)
            continue
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"feat: pin text-embedding artifact for {info.name} (#1148)",
        )
        print(f"    uploaded {fname}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Which model to pin (key in MODELS). Default: all-MiniLM-L6-v2.",
    )
    p.add_argument(
        "--out-dir", default="/tmp/ferrotorch_pretrained_text_weights",
        help="Staging directory.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Stage everything locally but do not upload.")
    p.add_argument("--skip-upload", action="store_true",
                   help="Alias for --dry-run.")
    args = p.parse_args()

    if args.model not in MODELS:
        print(f"unknown model '{args.model}'. Known: {list(MODELS)}",
              file=sys.stderr)
        return 2

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    info = MODELS[args.model]
    sha, out_dir = convert_one(info, out_root)
    if not (args.dry_run or args.skip_upload):
        hf_upload(info, out_dir)

    print("\n=== SUMMARY ===")
    print(f"  {info.name:24s}  sha256={sha}")
    print(f"  hf:   https://huggingface.co/ferrotorch/{info.name}")
    print(f"  dir:  {out_dir}")
    print("\n=== Drop-in registry pin (for ferrotorch-hub/src/registry.rs) ===")
    print(f"  // {info.name}: {info.upstream_repo}")
    print(f'  weights_url: "https://huggingface.co/ferrotorch/{info.name}/resolve/main/model.safetensors",')
    print(f'  weights_sha256: "{sha}",')
    print(f"  num_parameters: {info.param_count},")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
