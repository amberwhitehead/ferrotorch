#!/usr/bin/env python3
"""Pin the end-to-end Stable-Diffusion 1.5 text-to-image generation
trajectory to `ferrotorch/sd-v1-5-generation-trajectory` on HuggingFace.

Phase F of real-artifact-driven development (#1163). Composes the three
already-pinned SD sub-models (`sd-v1-5-clip-text-encoder` from #1152,
`sd-v1-5-unet` from #1151, `sd-v1-5-vae-decoder` from #1150) plus
`diffusers.schedulers.DDIMScheduler` into a single fixed-seed
generation pipeline and dumps every intermediate stage as a parity
probe.

Pipeline:

  1. Tokenize `PROMPT` ("a photograph of an astronaut riding a horse")
     and `NEG_PROMPT` ("") with `CLIPTokenizer` from the SD-1.5
     `tokenizer/` subfolder.
  2. Encode both with `CLIPTextModel.text_model` to get
     `cond_embeds [1, 77, 768]` and `uncond_embeds [1, 77, 768]`.
  3. Seed `torch.Generator(device='cpu').manual_seed(SEED=42)` and draw
     `init_latent = randn(1, 4, 64, 64)`. (This is the reference noise;
     the rust side reads it back from the mirror because rust's PRNG
     does not match torch's.)
  4. Configure `DDIMScheduler.from_config(pipe.scheduler.config)` with
     `set_timesteps(NUM_STEPS=4)`. The scheduler emits 4 timesteps in
     descending order; for SD-1.5 leading spacing + steps_offset=1 this
     is exactly [751, 501, 251, 1].
  5. Run the denoising loop with classifier-free guidance
     (`guidance_scale=7.5`). At each step persist
     `noise_pred_uncond`, `noise_pred_cond`, the CFG-guided noise, and
     the latent AFTER the scheduler step.
  6. Decode the final latent via `pipe.vae.decode(latent / 0.18215)`.
  7. Dump every f32 tensor in the standard `[u32 ndim][u32 shape][f32]`
     little-endian format, plus a `meta.json` recording the prompt,
     seed, step count, guidance scale, and the exact timestep list.
  8. SHA-256 the convenience tar bundle and upload everything to
     `huggingface.co/ferrotorch/sd-v1-5-generation-trajectory`.

The pin is run-once and the SHA is recorded in
`ferrotorch-hub/src/registry.rs`. Subsequent verification runs pull the
mirror via `hf_hub_download` (no rerun of diffusers required).

Usage:
    python3 scripts/pin_pretrained_sd_pipeline.py
    python3 scripts/pin_pretrained_sd_pipeline.py --dry-run     # stage only
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import struct
import sys
import tarfile
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from huggingface_hub import HfApi


REPO_ID = "ferrotorch/sd-v1-5-generation-trajectory"
UPSTREAM_REPO = "runwayml/stable-diffusion-v1-5"

PROMPT = "a photograph of an astronaut riding a horse"
NEG_PROMPT = ""
SEED = 42
NUM_STEPS = 4
GUIDANCE_SCALE = 7.5

RAIL_LICENSE_SUMMARY = (
    "Stable Diffusion v1.5 is distributed under the CreativeML Open RAIL-M "
    "license. This pipeline-trajectory bundle inherits that license — see "
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/LICENSE "
    "for the full terms."
)


def dump_f32(data: np.ndarray, path: Path) -> None:
    """Dump a float32 ndarray in `[u32 ndim][u32 shape][f32]` little-endian."""
    arr = data.reshape(-1).astype("<f4", copy=False)
    shape = list(data.shape)
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(shape)))
        for d in shape:
            f.write(struct.pack("<I", int(d)))
        f.write(arr.tobytes(order="C"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def render_readme(sha: str, timesteps: list[int]) -> str:
    return textwrap.dedent(
        f"""\
        ---
        license: openrail
        tags:
          - stable-diffusion
          - sd-1.5
          - diffusers
          - ferrotorch
          - real-artifact
        ---

        # `ferrotorch/sd-v1-5-generation-trajectory`

        End-to-end SD-1.5 text-to-image generation trajectory pinned for the
        ferrotorch real-artifact parity harness (Phase F, #1163).

        ## Provenance

        * Upstream model: `{UPSTREAM_REPO}` (`StableDiffusionPipeline`
          composed from `text_encoder/`, `unet/`, `vae/` subfolders, plus
          `DDIMScheduler.from_config(pipe.scheduler.config)`).
        * Conversion script:
          [`scripts/pin_pretrained_sd_pipeline.py`](https://github.com/dollspace-gay/ferrotorch/blob/main/scripts/pin_pretrained_sd_pipeline.py).
        * Ferrotorch issue: <https://github.com/dollspace-gay/ferrotorch/issues/1163>.
        * SHA-256 of `bundle.tar` (pinned in
          `ferrotorch-hub/src/registry.rs`): `{sha}`.

        ## Files

        * `cond_embeds.bin` — `[1, 77, 768]` f32 CLIP text embedding of
          `PROMPT = "{PROMPT}"`.
        * `uncond_embeds.bin` — `[1, 77, 768]` f32 CLIP text embedding of
          the empty negative prompt.
        * `init_latent.bin` — `[1, 4, 64, 64]` f32 Gaussian noise drawn
          via `torch.Generator(device='cpu').manual_seed({SEED}).randn`.
          The rust pipeline reads this file directly because the rust
          PRNG (`rand::StdRng`) does not match `torch.Generator`.
        * `final_image.bin` — `[1, 3, 512, 512]` f32 decoded image in
          `[-1, 1]` from `pipe.vae.decode(latent / 0.18215).sample`.
        * `step_K_noise_pred_uncond.bin` — `[1, 4, 64, 64]` f32 UNet
          forward pass with the unconditional embedding, for `K=0..{NUM_STEPS - 1}`.
        * `step_K_noise_pred_cond.bin` — same but with the conditional
          embedding.
        * `step_K_guided_noise.bin` — `noise_uncond + {GUIDANCE_SCALE} *
          (noise_cond - noise_uncond)`.
        * `step_K_latent_after.bin` — latent after the scheduler step,
          i.e. the input to step `K+1` (or the VAE for the final step).
        * `meta.json` — prompt, negative prompt, seed, step count,
          guidance scale, and the exact timestep list.
        * `bundle.tar` — single-file convenience archive carrying every
          fixture above (so the registry pin has one SHA-256 to track).

        ## Settings

        * `prompt              = "{PROMPT}"`
        * `negative_prompt     = "{NEG_PROMPT}"`
        * `seed                = {SEED}`
        * `num_inference_steps = {NUM_STEPS}`
        * `guidance_scale      = {GUIDANCE_SCALE}`
        * `scheduler           = DDIMScheduler` (scaled_linear,
          beta_start=0.00085, beta_end=0.012,
          clip_sample=False, set_alpha_to_one=False,
          prediction_type="epsilon", timestep_spacing="leading",
          steps_offset=1)
        * `timesteps           = {timesteps}`

        ## How the rust side consumes this

        The rust dump example
        [`ferrotorch-diffusion/examples/sd_pipeline_dump.rs`](https://github.com/dollspace-gay/ferrotorch/blob/main/ferrotorch-diffusion/examples/sd_pipeline_dump.rs)
        loads the three sub-models from `ferrotorch/sd-v1-5-{{clip-text-encoder,unet,vae-decoder}}`,
        loads `init_latent.bin` and the two text embeddings from this
        mirror (so the rust↔torch PRNG mismatch and tokenizer absence
        are routed around), runs the same 4-step CFG loop with a rust
        DDIMScheduler whose constants mirror diffusers byte-for-byte,
        and dumps the equivalent intermediates. The python harness
        [`scripts/verify_sd_pipeline_inference.py`](https://github.com/dollspace-gay/ferrotorch/blob/main/scripts/verify_sd_pipeline_inference.py)
        then compares each rust intermediate against the corresponding
        file shipped here, per-stage tolerances.

        ## Upstream license

        {RAIL_LICENSE_SUMMARY}
        """
    )


def build_pipeline() -> StableDiffusionPipeline:
    """Load SD-1.5 with the DDIM scheduler swapped in."""
    pipe = StableDiffusionPipeline.from_pretrained(
        UPSTREAM_REPO,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe


def generate_trajectory(pipe: StableDiffusionPipeline, out_dir: Path) -> dict:
    """Run the SD-1.5 fixed-seed generation and dump every fixture."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Pipeline: {pipe.__class__.__name__}; scheduler: {pipe.scheduler.__class__.__name__}",
          flush=True)

    # ---- Text encoding -----------------------------------------------------
    text_input = pipe.tokenizer(
        PROMPT,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_input = pipe.tokenizer(
        NEG_PROMPT,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    assert text_input.input_ids.shape == (1, 77), \
        f"unexpected tokenizer max_length: {text_input.input_ids.shape}"

    with torch.no_grad():
        cond_embeds = pipe.text_encoder(text_input.input_ids)[0]
        uncond_embeds = pipe.text_encoder(uncond_input.input_ids)[0]
    assert cond_embeds.shape == (1, 77, 768), \
        f"unexpected cond_embeds shape: {cond_embeds.shape}"

    print(f"  cond_embeds:   shape={tuple(cond_embeds.shape)} "
          f"min={cond_embeds.min():.4f} max={cond_embeds.max():.4f} "
          f"mean={cond_embeds.mean():.4f}", flush=True)
    print(f"  uncond_embeds: shape={tuple(uncond_embeds.shape)} "
          f"min={uncond_embeds.min():.4f} max={uncond_embeds.max():.4f} "
          f"mean={uncond_embeds.mean():.4f}", flush=True)

    # Save the tokenizer ids alongside the embeddings so the rust
    # consumer can hand them to its own CLIP encoder if it wants to.
    dump_f32(
        text_input.input_ids.cpu().numpy().astype(np.int64).astype(np.float32),
        out_dir / "prompt_input_ids.bin",
    )
    dump_f32(
        uncond_input.input_ids.cpu().numpy().astype(np.int64).astype(np.float32),
        out_dir / "uncond_input_ids.bin",
    )
    dump_f32(cond_embeds.cpu().numpy().astype(np.float32), out_dir / "cond_embeds.bin")
    dump_f32(uncond_embeds.cpu().numpy().astype(np.float32), out_dir / "uncond_embeds.bin")

    # ---- Initial noise -----------------------------------------------------
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    init_latent = torch.randn((1, 4, 64, 64), generator=generator, dtype=torch.float32)
    dump_f32(init_latent.cpu().numpy().astype(np.float32), out_dir / "init_latent.bin")
    print(f"  init_latent:  shape={tuple(init_latent.shape)} "
          f"min={init_latent.min():.4f} max={init_latent.max():.4f} "
          f"mean={init_latent.mean():.4f}", flush=True)

    # ---- Configure scheduler ----------------------------------------------
    pipe.scheduler.set_timesteps(NUM_STEPS)
    timesteps = [int(t) for t in pipe.scheduler.timesteps.tolist()]
    print(f"  timesteps:    {timesteps}", flush=True)

    # ---- Diffusion loop ----------------------------------------------------
    latent = init_latent * pipe.scheduler.init_noise_sigma
    per_step_meta = []
    t_loop_start = time.time()
    with torch.no_grad():
        for i, t in enumerate(pipe.scheduler.timesteps):
            # Two-pass CFG (we keep them separate so the dumps can verify
            # each forward pass independently). Functionally equivalent to
            # the `torch.cat([uncond, cond])` batched variant up to f32
            # accumulation noise; the separate-pass shape is the easier
            # one for the rust harness to mirror.
            t_scalar = t.to(dtype=torch.int64)
            t_in = t_scalar.expand(1)
            latent_in = pipe.scheduler.scale_model_input(latent, t)
            noise_uncond = pipe.unet(latent_in, t_in, encoder_hidden_states=uncond_embeds).sample
            noise_cond = pipe.unet(latent_in, t_in, encoder_hidden_states=cond_embeds).sample
            guided = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)
            latent = pipe.scheduler.step(guided, t, latent).prev_sample
            print(
                f"    step {i} (t={int(t)}): "
                f"|uncond|={noise_uncond.norm():.3f} "
                f"|cond|={noise_cond.norm():.3f} "
                f"|guided|={guided.norm():.3f} "
                f"|latent|={latent.norm():.3f}",
                flush=True,
            )
            dump_f32(noise_uncond.cpu().numpy().astype(np.float32),
                     out_dir / f"step_{i}_noise_pred_uncond.bin")
            dump_f32(noise_cond.cpu().numpy().astype(np.float32),
                     out_dir / f"step_{i}_noise_pred_cond.bin")
            dump_f32(guided.cpu().numpy().astype(np.float32),
                     out_dir / f"step_{i}_guided_noise.bin")
            dump_f32(latent.cpu().numpy().astype(np.float32),
                     out_dir / f"step_{i}_latent_after.bin")
            per_step_meta.append({"step": i, "timestep": int(t)})
    print(f"  diffusion loop: {time.time() - t_loop_start:.1f}s", flush=True)

    # ---- VAE decode --------------------------------------------------------
    with torch.no_grad():
        image = pipe.vae.decode(latent / 0.18215).sample
    assert image.shape == (1, 3, 512, 512), f"unexpected image shape: {image.shape}"
    dump_f32(image.cpu().numpy().astype(np.float32), out_dir / "final_image.bin")
    print(f"  final_image:  shape={tuple(image.shape)} "
          f"min={image.min():.4f} max={image.max():.4f} "
          f"mean={image.mean():.4f}", flush=True)

    meta = {
        "prompt": PROMPT,
        "negative_prompt": NEG_PROMPT,
        "seed": SEED,
        "num_inference_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "timesteps": timesteps,
        "scheduler": (
            "DDIMScheduler (scaled_linear, beta_start=0.00085, beta_end=0.012, "
            "clip_sample=false, set_alpha_to_one=false, prediction_type=epsilon, "
            "timestep_spacing=leading, steps_offset=1)"
        ),
        "per_step": per_step_meta,
        "upstream_repo": UPSTREAM_REPO,
        "diffusers_version": __import__("diffusers").__version__,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote meta.json: {len(json.dumps(meta))} bytes", flush=True)
    return meta


def make_bundle(out_dir: Path) -> tuple[Path, str]:
    """Bundle every fixture into a single tar so the registry pin has one
    SHA-256 to track. Returns (bundle_path, sha256)."""
    bundle = out_dir / "bundle.tar"
    with tarfile.open(bundle, "w") as tar:
        for f in sorted(out_dir.glob("*")):
            if f.name == "bundle.tar":
                continue
            tar.add(f, arcname=f.name)
    sha = sha256_of(bundle)
    print(f"  bundle.tar: {bundle.stat().st_size} bytes, SHA-256 {sha}", flush=True)
    return bundle, sha


def upload(out_dir: Path, sha: str, timesteps: list[int]) -> None:
    """Create the HF mirror and upload every fixture."""
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    readme_path = out_dir / "README.md"
    readme_path.write_text(render_readme(sha, timesteps))
    upload_paths = sorted(out_dir.glob("*"))
    print(f"  uploading {len(upload_paths)} files to https://huggingface.co/{REPO_ID}",
          flush=True)
    for p in upload_paths:
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"feat: pin SD-1.5 end-to-end generation trajectory (#1163)",
        )
        print(f"    uploaded {p.name} ({p.stat().st_size} bytes)", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir", default="/tmp/sd_gen",
        help="Local staging directory. Default: /tmp/sd_gen.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Stage every fixture locally but skip the HF upload.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ferrotorch SD-1.5 pipeline pin ({REPO_ID}) ===")
    pipe = build_pipeline()
    print(f"  loaded {UPSTREAM_REPO} on CPU, dtype={pipe.unet.dtype}", flush=True)

    meta = generate_trajectory(pipe, out_dir)
    _bundle, sha = make_bundle(out_dir)

    if not args.dry_run:
        upload(out_dir, sha, meta["timesteps"])
    else:
        # Still write README locally so a reviewer can sanity-check it.
        (out_dir / "README.md").write_text(render_readme(sha, meta["timesteps"]))
        print("  dry-run: skipped HF upload", flush=True)

    print("\n=== SUMMARY ===")
    print(f"  repo:           https://huggingface.co/{REPO_ID}")
    print(f"  bundle.tar SHA: {sha}")
    print(f"  num files:      {len(list(out_dir.glob('*')))}")
    print(f"  out_dir:        {out_dir}")
    print("\n=== Drop-in registry pin (for ferrotorch-hub/src/registry.rs) ===")
    print(f'  weights_url: "https://huggingface.co/{REPO_ID}/resolve/main/bundle.tar",')
    print(f'  weights_sha256: "{sha}",')
    print(f"  num_parameters: 0,")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
