// Crate-level lint baseline. Mirrors the ferrotorch-whisper / ferrotorch-bert
// posture: deny correctness / idiom / Debug / docs problems; warn pedantic
// stylistic issues. Specific pedantic lints are allowed crate-wide where
// the lint is consistently wrong for ML/numeric kernel code.

#![deny(unsafe_code)]
#![deny(rust_2018_idioms)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// Casts: dimension math (`as usize`, `as f32`, `as u32`) is intrinsic
// to tensor indexing — every kernel call would otherwise need a
// per-call allow.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
// Builder-style accessors don't all need `#[must_use]`.
#![allow(clippy::must_use_candidate)]
// Identifiers like `bf16`, `f32`, `VAE`, `SD`, `SiLU` are flagged as
// missing backticks even when they appear in code-fenced text.
#![allow(clippy::doc_markdown)]
// `needless_pass_by_value` would force `&VaeDecoderConfig` signatures
// throughout, hiding intent in the API.
#![allow(clippy::needless_pass_by_value)]
// `unnecessary_wraps` flags `Result`-returning helpers that today
// always succeed but are part of an extensible API surface.
#![allow(clippy::unnecessary_wraps)]
// `uninlined_format_args` flags `format!("x={}", x)` vs
// `format!("x={x}")`. Both are equally clear; the fixup churn is high.
#![allow(clippy::uninlined_format_args)]
// `many_single_char_names` flags conventional ML kernel locals
// (`q`, `k`, `v`, `h`).
#![allow(clippy::many_single_char_names)]
// `similar_names` flags variable pairs that are intentionally similar
// (e.g. `q2` / `q_h`).
#![allow(clippy::similar_names)]
// `module_name_repetitions`: every type starts with `Vae` (matching the
// HF / diffusers naming) — the lint would force renames that lose the
// upstream-1:1 mapping.
#![allow(clippy::module_name_repetitions)]
// `too_many_lines`: the decoder forward is one cohesive sequence of ops
// mirroring the diffusers reference; splitting it hurts cross-reading.
#![allow(clippy::too_many_lines)]

//! Stable-Diffusion VAE decoder composition for ferrotorch.
//!
//! Phase B.3a of real-artifact-driven development. This crate currently
//! implements the **decoder half** of `diffusers.AutoencoderKL` —
//! enough to invert a latent `[B, 4, 64, 64]` into an image
//! `[B, 3, 512, 512]`. The encoder, the UNet, the CLIP text encoder, and
//! the scheduler are out of scope and tracked under follow-up
//! dispatches.
//!
//! The architecture matches `runwayml/stable-diffusion-v1-5` /
//! `vae/config.json`:
//!
//! ```text
//! VaeDecoder
//! ├── post_quant_conv (Conv2d 4 -> 4, k=1)
//! └── Decoder
//!     ├── conv_in        (Conv2d 4 -> 512, k=3, pad=1)
//!     ├── mid_block      (UNetMidBlock2D)
//!     │   ├── resnets[0] (ResnetBlock2D 512 -> 512)
//!     │   ├── attentions[0] (single-head spatial attention, residual)
//!     │   └── resnets[1] (ResnetBlock2D 512 -> 512)
//!     ├── up_blocks      (4 × UpDecoderBlock2D, reversed channels:
//!     │                    [512, 512, 256, 128])
//!     ├── conv_norm_out  (GroupNorm 32, num_channels=128, eps=1e-6)
//!     ├── SiLU
//!     └── conv_out       (Conv2d 128 -> 3, k=3, pad=1)
//! ```
//!
//! ResnetBlock2D (no time embedding for VAE):
//!
//! ```text
//! h = norm1(x); h = silu(h); h = conv1(h)
//! h = norm2(h); h = silu(h); h = dropout(h); h = conv2(h)
//! out = h + (x if in==out else conv_shortcut(x))
//! ```
//!
//! AttnBlock2D (mid-block self-attention — single head, residual):
//!
//! ```text
//! r = x;                                  // [B, C, H, W]
//! h = x.view(B, C, H*W).transpose(1, 2);  // [B, HW, C]
//! h = group_norm(h.transpose(1, 2)).transpose(1, 2);
//! q = to_q(h); k = to_k(h); v = to_v(h);  // each [B, HW, C]
//! a = softmax(q @ k^T * scale, dim=-1) @ v;
//! h = to_out[0](a);                       // Linear C -> C
//! out = (h.transpose(1, 2).view(B, C, H, W)) + r;
//! ```
//!
//! Upsample2D: nearest-neighbour 2x interpolation + Conv2d(C, C, k=3,
//! pad=1).

pub mod blocks;
pub mod config;
pub mod safetensors_loader;
pub mod vae;

pub use blocks::{AttnBlock2D, ResnetBlock2D, UNetMidBlock2D, UpDecoderBlock2D, Upsample2D};
pub use config::VaeDecoderConfig;
pub use safetensors_loader::{load_vae_decoder, DropReport};
pub use vae::{Decoder, VaeDecoder};
