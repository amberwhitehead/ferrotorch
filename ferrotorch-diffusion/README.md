# ferrotorch-diffusion

Stable Diffusion family model composition for ferrotorch.

## Status — Phase B.3a (`#1150`)

This crate currently implements **only the VAE decoder** of Stable
Diffusion 1.5 (the `Decoder` half of `diffusers.AutoencoderKL`):

```
post_quant_conv (Conv2d 4 -> 4, kernel=1)
+ Decoder
  ├── conv_in     (Conv2d 4 -> 512, k=3, pad=1)
  ├── mid_block   (UNetMidBlock2D: resnet + 1-head attn + resnet @ 512ch)
  ├── up_blocks   (4 × UpDecoderBlock2D — 3 resnets + nearest-2x upsample;
  │                no upsample on the last block)
  ├── conv_norm_out (GroupNorm 32, num_channels=128, eps=1e-6)
  ├── SiLU
  └── conv_out    (Conv2d 128 -> 3, k=3, pad=1)
```

Latent `[B, 4, 64, 64]` → image `[B, 3, 512, 512]`. The decoder accepts
the post-`scaling_factor` latent (`z / 0.18215`, mirroring
`AutoencoderKL.decode`).

The VAE encoder, the UNet, the CLIP text encoder, and the scheduler are
out of scope for Phase B.3a (each tracked under follow-up dispatches).

## Real-artifact harness

`scripts/verify_diffusion_inference.py` compares this crate's output for
the pinned `ferrotorch/sd-v1-5-vae-decoder` checkpoint against a frozen
`diffusers==0.38.0` reference forward pass on a deterministic latent
(`torch.manual_seed(42); torch.randn(1, 4, 64, 64) * 0.18215`). The PASS
floor is `cosine_sim >= 0.999, max_abs <= 0.5` — same baseline as the
other Phase-B real-artifact harnesses.
