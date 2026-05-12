#!/usr/bin/env python3
"""Focused probe for the #1163 step-3 divergence.

Differentiates Hypothesis A (real bug — rust GPU UNet diverges from
diffusers UNet at high-noise step) from Hypothesis B (genuine f32
accumulation drift propagating into step 3).

Approach:
  1. Read both trajectories — the pinned diffusers reference trajectory
     and rust's per-step dumps from the latest GPU run.
  2. Take rust's step-2 latent_after (= step-3 latent INPUT) and feed it
     into the diffusers UNet with the SAME text_embed + timestep that
     rust used. The resulting noise prediction is what diffusers WOULD
     have produced given rust's drifted input.
  3. Compare:
       (a) diffusers UNet on rust-drifted input  vs  rust UNet on rust-drifted input
           → architectural correctness probe. If close, rust UNet
             is faithful at step 3; the divergence at step 3 vs the
             pinned reference is purely propagated drift.
       (b) diffusers UNet on rust-drifted input  vs  pinned reference noise_pred at step 3
           → quantifies how much divergence the drifted input ALONE
             causes (without any rust UNet contribution).

If (a) is tight (cos >= 0.9999, max_abs small) and (b) is roughly the
size of the failing comparison, Hypothesis B is confirmed — the rust
GPU UNet is bit-faithful within numerical limits and step 3's "FAIL"
is just propagated f32 noise. The verify harness criterion needs to
allow for step-scaled tolerance, NOT a real bug fix.

If (a) DIVERGES more than expected (rust UNet on same drifted input is
nowhere near diffusers UNet on same drifted input), Hypothesis A is
confirmed — there's an architectural bug in rust GpuUNet that's
masked at low-noise steps and only surfaces at high-noise. Pursue
fix.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from huggingface_hub import hf_hub_download


TRAJ_REPO = "ferrotorch/sd-v1-5-generation-trajectory"
UPSTREAM_REPO = "runwayml/stable-diffusion-v1-5"
NUM_STEPS = 4
GUIDANCE = 7.5
PROBE_STEP = 3  # the failing step
PROBE_TIMESTEP = 1  # SD-1.5 4-step DDIM leading: timesteps = [751, 501, 251, 1]


def read_dump_f32(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    (ndim,) = struct.unpack_from("<I", raw, 0)
    off = 4
    shape = struct.unpack_from(f"<{ndim}I", raw, off)
    off += 4 * ndim
    n = 1
    for s in shape:
        n *= int(s)
    flat = np.frombuffer(raw, dtype="<f4", count=n, offset=off)
    return flat.reshape([int(s) for s in shape]).astype(np.float32, copy=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def report(label: str, a: np.ndarray, b: np.ndarray) -> None:
    diff = a - b
    print(
        f"  {label}: cos={cosine(a, b):.6f} "
        f"max_abs={float(np.abs(diff).max()):.6f} "
        f"mean_abs={float(np.abs(diff).mean()):.6e} "
        f"shape={list(a.shape)}"
    )


def fetch(name: str) -> Path:
    return Path(hf_hub_download(repo_id=TRAJ_REPO, filename=name))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rust-dump-dir",
        type=Path,
        default=Path("/tmp/ferrotorch_verify_sd_pipeline/rust_dumps"),
        help="Directory holding rust's per-step dumps from the latest run.",
    )
    args = p.parse_args()

    if not args.rust_dump_dir.is_dir():
        print(
            f"error: rust dump dir not found: {args.rust_dump_dir}\n"
            f"       run `python3 scripts/verify_sd_pipeline_inference.py --device gpu --keep-dumps` first.",
            file=sys.stderr,
        )
        return 2

    print(f"=== #1163 step-{PROBE_STEP} probe (t={PROBE_TIMESTEP}) ===")
    print(f"rust dumps: {args.rust_dump_dir}")

    # ---- 1. Load rust step-2 latent_after (= step-3 latent input). --------
    rust_step2_latent = read_dump_f32(args.rust_dump_dir / "step_2_latent_after.bin")
    # The pinned reference's step-2 latent (for sanity).
    ref_step2_latent = read_dump_f32(fetch("step_2_latent_after.bin"))
    # Rust's step-3 noise predictions (the failing ones).
    rust_step3_uncond = read_dump_f32(args.rust_dump_dir / "step_3_noise_pred_uncond.bin")
    rust_step3_cond = read_dump_f32(args.rust_dump_dir / "step_3_noise_pred_cond.bin")
    # Reference noise_preds.
    ref_step3_uncond = read_dump_f32(fetch("step_3_noise_pred_uncond.bin"))
    ref_step3_cond = read_dump_f32(fetch("step_3_noise_pred_cond.bin"))

    # Text embeds (rust's, since the latent trajectory diverged from
    # them — rust step 3 used rust embeds).
    rust_cond_embeds = read_dump_f32(args.rust_dump_dir / "cond_embeds.bin")
    rust_uncond_embeds = read_dump_f32(args.rust_dump_dir / "uncond_embeds.bin")
    ref_cond_embeds = read_dump_f32(fetch("cond_embeds.bin"))
    ref_uncond_embeds = read_dump_f32(fetch("uncond_embeds.bin"))

    print()
    print("--- baseline: rust step-2 latent_after vs reference ---")
    report("latent_input_to_step3", rust_step2_latent, ref_step2_latent)
    print()
    print("--- baseline: text embeds (rust vs reference, F.2 inputs to UNet) ---")
    report("cond_embeds", rust_cond_embeds, ref_cond_embeds)
    report("uncond_embeds", rust_uncond_embeds, ref_uncond_embeds)

    # ---- 2. Load diffusers SD-1.5 UNet on CPU f32. ------------------------
    print()
    print("--- loading diffusers SD-1.5 UNet (CPU f32) ---")
    pipe = StableDiffusionPipeline.from_pretrained(
        UPSTREAM_REPO,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(NUM_STEPS)
    pipe = pipe.to("cpu")
    pipe.unet.eval()

    # ---- 3. Run diffusers UNet on RUST's drifted step-3 input. ------------
    print()
    print("--- diffusers UNet on rust's drifted step-3 input ---")
    print("    (rust step-2 latent_after + rust text embeds + t=1)")
    rust_latent_t = torch.from_numpy(rust_step2_latent.copy())
    rust_cond_t = torch.from_numpy(rust_cond_embeds.copy())
    rust_uncond_t = torch.from_numpy(rust_uncond_embeds.copy())
    t_in = torch.tensor([PROBE_TIMESTEP], dtype=torch.int64)
    with torch.no_grad():
        latent_scaled = pipe.scheduler.scale_model_input(rust_latent_t, PROBE_TIMESTEP)
        diff_uncond_on_rust = pipe.unet(
            latent_scaled, t_in, encoder_hidden_states=rust_uncond_t
        ).sample.cpu().numpy().astype(np.float32)
        diff_cond_on_rust = pipe.unet(
            latent_scaled, t_in, encoder_hidden_states=rust_cond_t
        ).sample.cpu().numpy().astype(np.float32)

    # ---- 4. Comparisons. --------------------------------------------------
    print()
    print("=== Hypothesis A: rust UNet vs diffusers UNet (SAME inputs) ===")
    print("If rust GpuUNet is bit-faithful to diffusers, these should be cos>=0.99999")
    print("(the only differences should be cuBLAS-level f32 reordering).")
    report("uncond (rust GPU vs diffusers, same drifted input)",
           rust_step3_uncond, diff_uncond_on_rust)
    report("cond   (rust GPU vs diffusers, same drifted input)",
           rust_step3_cond, diff_cond_on_rust)

    print()
    print("=== Hypothesis B: diffusers UNet (drifted input) vs ref noise_pred ===")
    print("Quantifies how much divergence the drifted input ALONE causes.")
    report("uncond (diffusers on drifted input vs pinned ref noise_pred)",
           diff_uncond_on_rust, ref_step3_uncond)
    report("cond   (diffusers on drifted input vs pinned ref noise_pred)",
           diff_cond_on_rust, ref_step3_cond)

    print()
    print("=== For reference: failing comparison (rust GPU vs pinned ref) ===")
    report("uncond (rust GPU vs pinned ref — the failing comparison)",
           rust_step3_uncond, ref_step3_uncond)
    report("cond   (rust GPU vs pinned ref — the failing comparison)",
           rust_step3_cond, ref_step3_cond)

    print()
    print("=== Verdict logic ===")
    cos_arch_uncond = cosine(rust_step3_uncond, diff_uncond_on_rust)
    cos_arch_cond = cosine(rust_step3_cond, diff_cond_on_rust)
    cos_drift_uncond = cosine(diff_uncond_on_rust, ref_step3_uncond)
    cos_drift_cond = cosine(diff_cond_on_rust, ref_step3_cond)
    cos_total_uncond = cosine(rust_step3_uncond, ref_step3_uncond)
    cos_total_cond = cosine(rust_step3_cond, ref_step3_cond)

    print(f"  rust UNet vs diffusers UNet (same input):     cos uncond={cos_arch_uncond:.6f} cond={cos_arch_cond:.6f}")
    print(f"  diffusers on drifted input vs ref noise_pred: cos uncond={cos_drift_uncond:.6f} cond={cos_drift_cond:.6f}")
    print(f"  rust GPU vs ref noise_pred (failing):         cos uncond={cos_total_uncond:.6f} cond={cos_total_cond:.6f}")

    # If architectural cos is very close to 1.0 AND drift cos is close to total,
    # the divergence is purely propagated f32 drift (Hypothesis B).
    arch_tight = min(cos_arch_uncond, cos_arch_cond) >= 0.9999
    drift_explains = (
        abs(cos_drift_uncond - cos_total_uncond) < 5e-4
        and abs(cos_drift_cond - cos_total_cond) < 5e-4
    )

    print()
    if arch_tight and drift_explains:
        print("VERDICT: Hypothesis B confirmed.")
        print("  rust GPU UNet is bit-faithful to diffusers UNet given the same input.")
        print("  The step-3 failure vs the pinned reference is purely propagated f32 drift")
        print("  from earlier steps (which themselves PASS within tolerance).")
        print("  Fix: scale the per-step noise-pred cosine tolerance with step index.")
        return 0
    print("VERDICT: Hypothesis A — investigate further.")
    print("  rust GPU UNet does NOT match diffusers UNet on the same drifted input")
    print("  closely enough to attribute the step-3 failure to drift alone.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
