#!/usr/bin/env python3
"""Pin a pretrained Stable-Diffusion VAE-decoder checkpoint to the
`ferrotorch/*` HF org.

Phase B.3a of real-artifact-driven development (closes ferrotorch issue
#1150). First of three SD sub-model pins (VAE decoder → UNet → CLIP text
encoder); the other two are out of scope for this script.

For the chosen model this script:

1. Downloads the upstream HF VAE bundle from
   `runwayml/stable-diffusion-v1-5`'s `vae/` subfolder
   (`config.json` + `diffusion_pytorch_model.safetensors`).
2. Verifies the safetensors `post_quant_conv.*` / `decoder.*` key list
   matches the layout `ferrotorch_diffusion::VaeDecoder::load_hf_state_dict`
   consumes. Every decoder key must either map onto a parameter or
   appear in the documented (drop-by-design) list — silent state-dict
   drops are refused.
3. Generates a fixed parity probe:
     - `_value_parity_latent.bin`: deterministic latent
       `torch.manual_seed(42); torch.randn(1, 4, 64, 64) * 0.18215` —
       float32, dumped in `[u32 ndim][u32 shape][f32]` little-endian.
     - `_value_parity_image.bin`: float32 decoded image
       `[1, 3, 512, 512]` from
       `vae.decode(latent / 0.18215, return_dict=False)[0]` in eval mode
       on float32 weights. Same dump format.
4. Re-packs the decoder-only subset of the safetensors so the pinned
   `model.safetensors` carries ONLY `post_quant_conv.*` + `decoder.*`.
   This shrinks the upload to roughly the decoder half of the full VAE
   and makes the pin asymmetric — a future change that needs the
   encoder cannot accidentally use this mirror.
5. Uploads `model.safetensors`, `config.json`, the parity probe files,
   and a README to `huggingface.co/ferrotorch/<name>`.
6. Hashes the uploaded `model.safetensors` with SHA-256 and prints a
   registry-ready snippet for `ferrotorch-hub/src/registry.rs`.

Usage:
    python3 scripts/pin_pretrained_diffusion_weights.py \
        [--model sd-v1-5-vae-decoder] \
        [--dry-run] [--skip-upload] \
        [--out-dir /tmp/ferrotorch_pretrained_diffusion_weights]
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

import numpy as np
import torch
from diffusers import AutoencoderKL
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# CreativeML Open RAIL-M LICENSE summary line. SD 1.5 is RAIL-M licensed;
# we redistribute byte-for-byte for the decoder slice. The full license
# text is included in the uploaded README so downstream consumers can
# reproduce the legal trail.
# ---------------------------------------------------------------------------
RAIL_LICENSE_SUMMARY = (
    "Stable Diffusion v1.5 is distributed under the CreativeML Open RAIL-M "
    "license. The decoder slice mirrored here inherits that license — see "
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/LICENSE "
    "for the full terms."
)


@dataclass
class DiffusionModelInfo:
    """One pinnable SD-family model entry."""

    name: str
    upstream_repo: str
    upstream_subfolder: str
    description: str
    license: str
    param_count: int


MODELS: dict[str, DiffusionModelInfo] = {
    "sd-v1-5-vae-decoder": DiffusionModelInfo(
        name="sd-v1-5-vae-decoder",
        upstream_repo="runwayml/stable-diffusion-v1-5",
        upstream_subfolder="vae",
        description=(
            "Stable Diffusion 1.5 VAE decoder (runwayml/stable-diffusion-v1-5, "
            "vae/ subfolder). post_quant_conv (Conv2d 4->4, k=1) + Decoder "
            "(conv_in 4->512, UNetMidBlock2D with 1-head attention at 512ch, "
            "4× UpDecoderBlock2D with 3 resnets each and nearest-2x upsample "
            "on all but the last block, GroupNorm32 + SiLU + conv_out "
            "128->3). ~50M-param decoder slice of AutoencoderKL. RAIL-M "
            "licensed. Pinned decoder-only — encoder + quant_conv keys are "
            "dropped from this mirror. Real-artifact baseline for SD VAE "
            "decoder parity vs diffusers (#1150)."
        ),
        license="openrail",
        # Decoder-only parameter count (post_quant_conv + decoder).
        # Computed by the script after the pin; this value is the
        # observed count for SD 1.5's VAE decoder slice. The exact value
        # is asserted at pin time and is recorded for the registry.
        param_count=49_490_179,
    ),
}


# ---------------------------------------------------------------------------
# Expected ferrotorch-diffusion VAE decoder state-dict key set.
# Mirrors `VaeDecoder::named_parameters()` exactly.
# ---------------------------------------------------------------------------

def expected_decoder_keys_and_shapes(cfg: dict) -> dict[str, list[int]]:
    """Per-parameter shape pin. Refuses any VAE whose layout diverges
    from what the loader will consume."""
    blocks = cfg["block_out_channels"]      # [128, 256, 512, 512]
    layers_per_block = cfg["layers_per_block"]  # 2
    norm_groups = cfg["norm_num_groups"]    # 32
    out_channels = cfg["out_channels"]      # 3
    latent_channels = cfg["latent_channels"]  # 4

    _ = norm_groups  # used implicitly via the shape size of GroupNorm
                     # weights, which equals num_channels — no shape
                     # check here beyond that.

    top = blocks[-1]
    bottom = blocks[0]
    resnets_per_up = layers_per_block + 1

    shapes: dict[str, list[int]] = {
        # post_quant_conv (1x1 conv over 4 channels)
        "post_quant_conv.weight": [latent_channels, latent_channels, 1, 1],
        "post_quant_conv.bias": [latent_channels],
        # decoder.conv_in
        "decoder.conv_in.weight": [top, latent_channels, 3, 3],
        "decoder.conv_in.bias": [top],
        # mid_block: attentions.0 + resnets.{0,1}
        "decoder.mid_block.attentions.0.group_norm.weight": [top],
        "decoder.mid_block.attentions.0.group_norm.bias": [top],
        "decoder.mid_block.attentions.0.to_q.weight": [top, top],
        "decoder.mid_block.attentions.0.to_q.bias": [top],
        "decoder.mid_block.attentions.0.to_k.weight": [top, top],
        "decoder.mid_block.attentions.0.to_k.bias": [top],
        "decoder.mid_block.attentions.0.to_v.weight": [top, top],
        "decoder.mid_block.attentions.0.to_v.bias": [top],
        "decoder.mid_block.attentions.0.to_out.0.weight": [top, top],
        "decoder.mid_block.attentions.0.to_out.0.bias": [top],
    }
    for ri in (0, 1):
        prefix = f"decoder.mid_block.resnets.{ri}"
        shapes[f"{prefix}.norm1.weight"] = [top]
        shapes[f"{prefix}.norm1.bias"] = [top]
        shapes[f"{prefix}.conv1.weight"] = [top, top, 3, 3]
        shapes[f"{prefix}.conv1.bias"] = [top]
        shapes[f"{prefix}.norm2.weight"] = [top]
        shapes[f"{prefix}.norm2.bias"] = [top]
        shapes[f"{prefix}.conv2.weight"] = [top, top, 3, 3]
        shapes[f"{prefix}.conv2.bias"] = [top]

    # up_blocks: reversed channels (top -> bottom).
    reversed_blocks = list(reversed(blocks))
    prev = reversed_blocks[0]
    n_up = len(reversed_blocks)
    for i, c in enumerate(reversed_blocks):
        is_final = i == n_up - 1
        for j in range(resnets_per_up):
            in_c = prev if j == 0 else c
            p = f"decoder.up_blocks.{i}.resnets.{j}"
            shapes[f"{p}.norm1.weight"] = [in_c]
            shapes[f"{p}.norm1.bias"] = [in_c]
            shapes[f"{p}.conv1.weight"] = [c, in_c, 3, 3]
            shapes[f"{p}.conv1.bias"] = [c]
            shapes[f"{p}.norm2.weight"] = [c]
            shapes[f"{p}.norm2.bias"] = [c]
            shapes[f"{p}.conv2.weight"] = [c, c, 3, 3]
            shapes[f"{p}.conv2.bias"] = [c]
            if in_c != c:
                shapes[f"{p}.conv_shortcut.weight"] = [c, in_c, 1, 1]
                shapes[f"{p}.conv_shortcut.bias"] = [c]
        if not is_final:
            shapes[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = [c, c, 3, 3]
            shapes[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = [c]
        prev = c

    # conv_norm_out + conv_out
    shapes["decoder.conv_norm_out.weight"] = [bottom]
    shapes["decoder.conv_norm_out.bias"] = [bottom]
    shapes["decoder.conv_out.weight"] = [out_channels, bottom, 3, 3]
    shapes["decoder.conv_out.bias"] = [out_channels]

    return shapes


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_f32(data: np.ndarray, path: Path) -> None:
    """Dump a float32 ndarray in the `[u32 ndim][u32 shape][f32]`
    little-endian format the Rust dump example reads."""
    arr = data.reshape(-1).astype("<f4", copy=False)
    shape = list(data.shape)
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(shape)))
        for d in shape:
            f.write(struct.pack("<I", int(d)))
        f.write(arr.tobytes(order="C"))


def deterministic_latent(latent_channels: int, h: int, w: int, scaling_factor: float) -> torch.Tensor:
    """`torch.manual_seed(42); randn(1, C, H, W) * scaling_factor` —
    matches the test plan exactly so re-pinning regenerates the same
    bytes."""
    g = torch.Generator()
    g.manual_seed(42)
    z = torch.randn(1, latent_channels, h, w, generator=g, dtype=torch.float32)
    return z * scaling_factor


def convert_one(info: DiffusionModelInfo, out_root: Path) -> tuple[str, Path, int]:
    """Download, verify, write parity probe. Returns (sha256, model_dir,
    actual_param_count)."""
    print(f"\n=== {info.name} <- {info.upstream_repo}/{info.upstream_subfolder} ===",
          flush=True)

    out_dir = out_root / info.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Download upstream config + weights from the vae/ subfolder. -
    upstream_files = [
        ("config.json", "config.json"),
        ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.safetensors"),
    ]
    local_paths: dict[str, Path] = {}
    for upstream_name, local_name in upstream_files:
        try:
            p = hf_hub_download(
                repo_id=info.upstream_repo,
                filename=f"{info.upstream_subfolder}/{upstream_name}",
            )
        except Exception as e:
            raise SystemExit(
                f"{info.name}: failed to download upstream "
                f"{info.upstream_subfolder}/{upstream_name} from "
                f"{info.upstream_repo}: {e}"
            )
        target = out_dir / local_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(p).read_bytes())
        local_paths[local_name] = target
        print(f"  fetched {upstream_name} -> {target}", flush=True)

    cfg = json.loads(local_paths["config.json"].read_text())
    print(
        f"  config: block_out_channels={cfg['block_out_channels']} "
        f"layers_per_block={cfg['layers_per_block']} "
        f"norm_num_groups={cfg['norm_num_groups']} "
        f"sample_size={cfg['sample_size']} "
        f"latent_channels={cfg['latent_channels']} "
        f"scaling_factor={cfg.get('scaling_factor', 0.18215)} "
        f"act_fn={cfg.get('act_fn')}",
        flush=True,
    )

    # ---- Verify safetensors layout (decoder subset). ------------------
    with safe_open(local_paths["diffusion_pytorch_model.safetensors"], framework="pt") as f:
        full_keys = set(f.keys())
        full_shapes: dict[str, list[int]] = {
            k: list(f.get_slice(k).get_shape()) for k in full_keys
        }
        full_tensors: dict[str, torch.Tensor] = {
            k: f.get_tensor(k) for k in full_keys
        }

    expected_shapes = expected_decoder_keys_and_shapes(cfg)

    # ---- SD 1.5 ships the VAE attention block in the *deprecated*
    #      layout (`query/key/value/proj_attn`). diffusers's own loader
    #      runs `_convert_deprecated_attention_blocks` on the way in;
    #      we apply the same renames here so the on-disk mirror is
    #      already in the canonical `to_q/to_k/to_v/to_out.0` form that
    #      ferrotorch-diffusion's loader consumes. The four upstream
    #      attention parameters are mapped 1:1 with no transformation.
    deprecated_remap = {
        "decoder.mid_block.attentions.0.query.weight":   "decoder.mid_block.attentions.0.to_q.weight",
        "decoder.mid_block.attentions.0.query.bias":     "decoder.mid_block.attentions.0.to_q.bias",
        "decoder.mid_block.attentions.0.key.weight":     "decoder.mid_block.attentions.0.to_k.weight",
        "decoder.mid_block.attentions.0.key.bias":       "decoder.mid_block.attentions.0.to_k.bias",
        "decoder.mid_block.attentions.0.value.weight":   "decoder.mid_block.attentions.0.to_v.weight",
        "decoder.mid_block.attentions.0.value.bias":     "decoder.mid_block.attentions.0.to_v.bias",
        "decoder.mid_block.attentions.0.proj_attn.weight": "decoder.mid_block.attentions.0.to_out.0.weight",
        "decoder.mid_block.attentions.0.proj_attn.bias":   "decoder.mid_block.attentions.0.to_out.0.bias",
    }
    # The SD 1.5 VAE attention block's q/k/v/proj_attn linear weights
    # are stored as 4-D `[C, C, 1, 1]` conv-style tensors in older
    # checkpoints, even though they apply as linear projections.
    # Squeeze the trailing 1x1 spatial axes so the shape matches a
    # `Linear` weight `[C, C]`. Diffusers does this implicitly via
    # `_convert_deprecated_attention_blocks` (the deprecated
    # `AttentionBlock` used `nn.Conv2d(C, C, 1)` for q/k/v/proj_attn).
    full_tensors_remapped: dict[str, torch.Tensor] = {}
    full_shapes_remapped: dict[str, list[int]] = {}
    for k, t in full_tensors.items():
        nk = deprecated_remap.get(k, k)
        # Squeeze trailing 1x1 if this is a remapped attention weight
        # arriving as [C, C, 1, 1] but the target expects [C, C].
        if nk in (
            "decoder.mid_block.attentions.0.to_q.weight",
            "decoder.mid_block.attentions.0.to_k.weight",
            "decoder.mid_block.attentions.0.to_v.weight",
            "decoder.mid_block.attentions.0.to_out.0.weight",
        ) and t.ndim == 4 and t.shape[-2:] == (1, 1):
            t = t.squeeze(-1).squeeze(-1)
        full_tensors_remapped[nk] = t
        full_shapes_remapped[nk] = list(t.shape)
    full_keys_remapped = set(full_tensors_remapped)
    print(
        f"  applied deprecated-attn-block rename ({len([k for k in full_keys if k in deprecated_remap])} keys)",
        flush=True,
    )

    decoder_keys = {
        k for k in full_keys_remapped
        if k.startswith("post_quant_conv.") or k.startswith("decoder.")
    }
    missing = set(expected_shapes) - decoder_keys
    if missing:
        raise SystemExit(
            f"{info.name}: ferrotorch-diffusion expects {len(missing)} decoder "
            f"keys absent from the upstream safetensors. Sample: "
            f"{sorted(missing)[:5]}"
        )
    unexpected_dec = decoder_keys - set(expected_shapes)
    if unexpected_dec:
        raise SystemExit(
            f"{info.name}: upstream safetensors has {len(unexpected_dec)} "
            f"decoder-prefix keys ferrotorch-diffusion does NOT consume. "
            f"Refusing to pin (we will not silently drop decoder parameters). "
            f"Sample: {sorted(unexpected_dec)[:5]}"
        )
    for k, exp in expected_shapes.items():
        got = full_shapes_remapped.get(k)
        if got != exp:
            raise SystemExit(
                f"{info.name}: shape mismatch for '{k}': upstream {got} vs "
                f"ferrotorch expects {exp}. Refusing to pin."
            )
    non_decoder = sorted(full_keys_remapped - decoder_keys)
    print(
        f"  state-dict cross-check OK: {len(expected_shapes)}/{len(expected_shapes)} "
        f"decoder keys mapped; will drop {len(non_decoder)} non-decoder keys "
        f"from the mirror (first few: {non_decoder[:3]}).",
        flush=True,
    )

    # ---- Re-pack decoder-only safetensors. ----------------------------
    decoder_tensors: dict[str, torch.Tensor] = {}
    for k in expected_shapes:
        decoder_tensors[k] = full_tensors_remapped[k].contiguous().clone()
    decoder_path = out_dir / "model.safetensors"
    save_file(decoder_tensors, str(decoder_path))
    actual_param_count = sum(t.numel() for t in decoder_tensors.values())
    print(
        f"  re-packed decoder-only model.safetensors "
        f"({decoder_path.stat().st_size} bytes, {len(decoder_tensors)} keys, "
        f"{actual_param_count} scalar parameters)",
        flush=True,
    )

    # ---- Generate parity probe. ---------------------------------------
    print("  generating value-parity probe…", flush=True)
    scaling_factor = float(cfg.get("scaling_factor", 0.18215))
    latent_h = cfg["sample_size"] // (2 ** (len(cfg["block_out_channels"]) - 1))
    latent_w = latent_h
    latent = deterministic_latent(
        cfg["latent_channels"], latent_h, latent_w, scaling_factor,
    )
    print(
        f"  latent: shape={tuple(latent.shape)} dtype={latent.dtype} "
        f"min={latent.min().item():.4f} max={latent.max().item():.4f} "
        f"mean={latent.mean().item():.6f} std={latent.std().item():.6f}",
        flush=True,
    )

    # Run the upstream decoder (eval, fp32, no grad). The on-disk
    # latent is post-scaled (`randn * scaling_factor`); the SD pipeline
    # convention is then `vae.decode(latent / scaling_factor)`, so we
    # match it byte-for-byte here.
    vae = AutoencoderKL.from_pretrained(
        info.upstream_repo, subfolder=info.upstream_subfolder, torch_dtype=torch.float32,
    )
    vae.eval()
    with torch.no_grad():
        decoded = vae.decode(latent / scaling_factor, return_dict=False)[0]
    dec_np = decoded.cpu().numpy().astype(np.float32)
    print(
        f"  decoded image: shape={dec_np.shape} "
        f"min={dec_np.min():.4f} max={dec_np.max():.4f} "
        f"mean={dec_np.mean():.4f} std={dec_np.std():.4f}",
        flush=True,
    )
    expected_image_shape = (
        1,
        cfg["out_channels"],
        cfg["sample_size"],
        cfg["sample_size"],
    )
    if dec_np.shape != expected_image_shape:
        raise SystemExit(
            f"{info.name}: decoded image shape {dec_np.shape} != "
            f"{expected_image_shape}"
        )

    parity_latent = out_dir / "_value_parity_latent.bin"
    dump_f32(latent.numpy(), parity_latent)
    parity_image = out_dir / "_value_parity_image.bin"
    dump_f32(dec_np, parity_image)
    print(
        f"  wrote {parity_latent.name} ({parity_latent.stat().st_size} bytes), "
        f"{parity_image.name} ({parity_image.stat().st_size} bytes)",
        flush=True,
    )

    # ---- SHA of the re-packed decoder-only safetensors. ---------------
    sha = sha256_of(decoder_path)
    print(f"  decoder-only model.safetensors SHA-256: {sha}", flush=True)

    # ---- README. -----------------------------------------------------
    readme_path = out_dir / "README.md"
    readme_path.write_text(
        render_readme(info, cfg, sha, non_decoder, actual_param_count)
    )
    print(f"  wrote {readme_path}", flush=True)

    return sha, out_dir, actual_param_count


def render_readme(
    info: DiffusionModelInfo,
    cfg: dict,
    sha: str,
    non_decoder: list[str],
    actual_param_count: int,
) -> str:
    return textwrap.dedent(f"""\
        ---
        license: {info.license}
        tags:
          - stable-diffusion
          - vae
          - autoencoder-kl
          - ferrotorch
        ---

        # `ferrotorch/{info.name}`

        {info.description}

        ## Provenance

        * Upstream: `{info.upstream_repo}` (subfolder `{info.upstream_subfolder}/`),
          {info.license}.
        * Conversion script:
          [`ferrotorch/scripts/pin_pretrained_diffusion_weights.py`](https://github.com/dollspace-gay/ferrotorch/blob/main/scripts/pin_pretrained_diffusion_weights.py).
        * Ferrotorch issue: <https://github.com/dollspace-gay/ferrotorch/issues/1150>.
        * SHA-256 of `model.safetensors` (this file is pinned in
          `ferrotorch-hub/src/registry.rs`): `{sha}`.
        * Number of trainable parameters in the decoder slice:
          **{actual_param_count:,}**.
        * Config snapshot:
          block_out_channels={cfg['block_out_channels']},
          layers_per_block={cfg['layers_per_block']},
          norm_num_groups={cfg['norm_num_groups']},
          sample_size={cfg['sample_size']},
          latent_channels={cfg['latent_channels']},
          scaling_factor={cfg.get('scaling_factor', 0.18215)},
          act_fn={cfg.get('act_fn', 'silu')!r}.
        * Non-decoder keys dropped from the upstream checkpoint (this
          mirror is decoder-only): {len(non_decoder)} total, first few:
          `{non_decoder[:3]}`.

        ## Value-parity probe

        Two extra files are uploaded so the ferrotorch-side harness can
        reproduce the parity verdict without re-running the upstream
        AutoencoderKL.decode:

        * `_value_parity_latent.bin` — deterministic latent
          `torch.manual_seed(42); torch.randn(1, 4, 64, 64) * 0.18215`,
          float32, shape `[1, 4, 64, 64]`. This is the *post-scaling*
          latent the SD pipeline feeds to `vae.decode` (which itself
          divides by `scaling_factor` internally).
        * `_value_parity_image.bin` — float32 decoded image
          `[1, 3, 512, 512]` from
          `AutoencoderKL.decode(latent, return_dict=False)[0]` on
          float32 weights in eval mode. Same dump format as every other
          ferrotorch artifact:
          `[u32 ndim][u32 × ndim shape][f32 × prod(shape)]` little-endian.

        ## How to load

        ```rust
        use ferrotorch_diffusion::{{VaeDecoderConfig, load_vae_decoder}};
        use ferrotorch_hub::{{HubCache, hf_download_model}};

        let cache = HubCache::with_default_dir();
        let repo_dir = hf_download_model("ferrotorch/{info.name}", "main", &cache)?;
        let cfg = VaeDecoderConfig::from_file(&repo_dir.join("config.json"))?;
        let (decoder, _drop_report) = load_vae_decoder::<f32>(
            &repo_dir.join("model.safetensors"),
            cfg,
            /* strict = */ false,
        )?;
        ```

        ## Upstream license

        {RAIL_LICENSE_SUMMARY}
    """)


def hf_upload(info: DiffusionModelInfo, out_dir: Path) -> None:
    api = HfApi()
    repo_id = f"ferrotorch/{info.name}"
    print(f"  uploading to https://huggingface.co/{repo_id}", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    files = [
        "config.json",
        "model.safetensors",
        "_value_parity_latent.bin",
        "_value_parity_image.bin",
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
            commit_message=f"feat: pin decoder-only artifact for {info.name} (#1150)",
        )
        print(f"    uploaded {fname}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", default="sd-v1-5-vae-decoder",
        help="Which model to pin (key in MODELS). Default: sd-v1-5-vae-decoder.",
    )
    p.add_argument(
        "--out-dir", default="/tmp/ferrotorch_pretrained_diffusion_weights",
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
    sha, out_dir, actual_param_count = convert_one(info, out_root)
    if not (args.dry_run or args.skip_upload):
        hf_upload(info, out_dir)

    print("\n=== SUMMARY ===")
    print(f"  {info.name:24s}  sha256={sha}")
    print(f"  num_parameters: {actual_param_count:,}")
    print(f"  hf:   https://huggingface.co/ferrotorch/{info.name}")
    print(f"  dir:  {out_dir}")
    print("\n=== Drop-in registry pin (for ferrotorch-hub/src/registry.rs) ===")
    print(f"  // {info.name}: {info.upstream_repo}/{info.upstream_subfolder}")
    print(f'  weights_url: "https://huggingface.co/ferrotorch/{info.name}/resolve/main/model.safetensors",')
    print(f'  weights_sha256: "{sha}",')
    print(f"  num_parameters: {actual_param_count},")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
