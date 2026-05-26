//! Stable-Diffusion VAE decoder composition.
//!
//! The forward path mirrors `diffusers.AutoencoderKL.decode(z).sample`
//! for `runwayml/stable-diffusion-v1-5`:
//!
//! ```text
//! z (pre-divided by scaling_factor)
//!   -> post_quant_conv
//!   -> Decoder.conv_in
//!   -> Decoder.mid_block
//!   -> Decoder.up_blocks[0..N]
//!   -> Decoder.conv_norm_out -> SiLU -> Decoder.conv_out
//! ```
//!
//! The `decode_with_scaling` helper applies `z / scaling_factor` first,
//! matching `AutoencoderKL.decode(z).sample`. `forward(z)` is the
//! post-scaling path and accepts an already-divided latent.
//!
//! ## REQ status (per `.design/ferrotorch-diffusion/vae.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `Decoder<T>` at `vae.rs:30..50` and `Decoder::new` at `vae.rs:51..120`; consumer: `VaeDecoder::new` at `vae.rs:281` builds it; itself consumed by `safetensors_loader.rs:330` `load_vae_decoder` |
//! | REQ-2 | SHIPPED | `VaeDecoder<T>` at `vae.rs:254..263` and `VaeDecoder::new` at `vae.rs:265..288`; consumer: `safetensors_loader.rs:330` `load_vae_decoder` instantiates it; `pipeline.rs:75` carries it as a pipeline field |
//! | REQ-3 | SHIPPED | `Module<T>::forward` at `vae.rs:314..327`; consumer: `pipeline.rs:227` `vae.decode_with_scaling(...)` (which calls `forward` internally) |
//! | REQ-4 | SHIPPED | `decode_with_scaling` at `vae.rs:297..308`; consumer: `pipeline.rs:227` and `examples/vae_decode_dump.rs` invoke it for SD-1.5 decoding |
//! | REQ-5 | SHIPPED | `Module<T>::load_state_dict` at `vae.rs:366..389`; consumer: `safetensors_loader.rs:89` `VaeDecoder::load_hf_state_dict` calls `self.load_state_dict(&remapped, strict)` after stripping the `vae.` prefix |

use std::collections::HashMap;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Conv2d, GroupNorm, SiLU};

use crate::blocks::{UNetMidBlock2D, UpDecoderBlock2D};
use crate::config::VaeDecoderConfig;

/// The bare `Decoder` half — matches `diffusers.models.autoencoders.vae.Decoder`.
#[derive(Debug)]
pub struct Decoder<T: Float> {
    /// First conv: `latent_channels -> block_out_channels[-1]` (k=3, pad=1).
    pub conv_in: Conv2d<T>,
    /// VAE mid-block at `block_out_channels[-1]` channels.
    pub mid_block: UNetMidBlock2D<T>,
    /// Up-blocks in *decoder order* — block 0 operates at the highest
    /// channel count and lowest spatial resolution.
    pub up_blocks: Vec<UpDecoderBlock2D<T>>,
    /// Final GroupNorm before the output conv (operates on
    /// `block_out_channels[0]` channels).
    pub conv_norm_out: GroupNorm<T>,
    /// Output activation (SiLU).
    pub conv_act: SiLU,
    /// Output conv: `block_out_channels[0] -> out_channels` (k=3, pad=1).
    pub conv_out: Conv2d<T>,
    /// Frozen copy of the config.
    pub config: VaeDecoderConfig,
    training: bool,
}

impl<T: Float> Decoder<T> {
    /// Build a randomly-initialized `Decoder`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] for any invalid
    /// config field (forwarded from [`VaeDecoderConfig::validate`]). In
    /// particular `block_out_channels` must be non-empty — the `unwrap`
    /// on `.last()` below is preceded by `cfg.validate()?` which checks
    /// exactly that.
    pub fn new(cfg: VaeDecoderConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        let groups = cfg.norm_num_groups;
        let resnet_eps = 1e-6_f64;
        let top_channels =
            *cfg.block_out_channels
                .last()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: "Decoder::new: block_out_channels is empty (should be unreachable \
                              after validate)"
                        .into(),
                })?;

        let conv_in = Conv2d::<T>::new(
            cfg.latent_channels,
            top_channels,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
        )?;

        let mid_block = UNetMidBlock2D::<T>::new(top_channels, groups, resnet_eps)?;

        let reversed: Vec<usize> = cfg.block_out_channels.iter().rev().copied().collect();
        let mut up_blocks = Vec::with_capacity(reversed.len());
        let mut prev_out = reversed[0];
        let num_blocks = reversed.len();
        let resnets = cfg.resnets_per_up_block();
        for (i, &c) in reversed.iter().enumerate() {
            let is_final = i == num_blocks - 1;
            up_blocks.push(UpDecoderBlock2D::<T>::new(
                prev_out, c, resnets, groups, resnet_eps, !is_final,
            )?);
            prev_out = c;
        }

        let bottom_channels = cfg.block_out_channels[0];
        let conv_norm_out = GroupNorm::<T>::new(groups, bottom_channels, resnet_eps, true)?;
        let conv_out = Conv2d::<T>::new(
            bottom_channels,
            cfg.out_channels,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
        )?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_act: SiLU::new(),
            conv_out,
            config: cfg,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for Decoder<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Sanity check: [B, latent_channels, H_lat, W_lat].
        let cfg = &self.config;
        if input.ndim() != 4 || input.shape()[1] != cfg.latent_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Decoder::forward: expected [B, {}, H, W], got {:?}",
                    cfg.latent_channels,
                    input.shape()
                ),
            });
        }
        let mut h = self.conv_in.forward(input)?;
        h = self.mid_block.forward(&h)?;
        for up in &self.up_blocks {
            h = up.forward(&h)?;
        }
        h = self.conv_norm_out.forward(&h)?;
        h = self.conv_act.forward(&h)?;
        self.conv_out.forward(&h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.conv_in.parameters());
        out.extend(self.mid_block.parameters());
        for b in &self.up_blocks {
            out.extend(b.parameters());
        }
        out.extend(self.conv_norm_out.parameters());
        out.extend(self.conv_out.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.conv_in.parameters_mut());
        out.extend(self.mid_block.parameters_mut());
        for b in &mut self.up_blocks {
            out.extend(b.parameters_mut());
        }
        out.extend(self.conv_norm_out.parameters_mut());
        out.extend(self.conv_out.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.conv_in.named_parameters() {
            out.push((format!("conv_in.{n}"), p));
        }
        for (n, p) in self.mid_block.named_parameters() {
            out.push((format!("mid_block.{n}"), p));
        }
        for (i, b) in self.up_blocks.iter().enumerate() {
            for (n, p) in b.named_parameters() {
                out.push((format!("up_blocks.{i}.{n}"), p));
            }
        }
        for (n, p) in self.conv_norm_out.named_parameters() {
            out.push((format!("conv_norm_out.{n}"), p));
        }
        for (n, p) in self.conv_out.named_parameters() {
            out.push((format!("conv_out.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        for b in &mut self.up_blocks {
            b.train();
        }
        self.mid_block.train();
    }
    fn eval(&mut self) {
        self.training = false;
        for b in &mut self.up_blocks {
            b.eval();
        }
        self.mid_block.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }

    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let extract = |prefix: &str| -> StateDict<T> {
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };

        if strict {
            for k in state.keys() {
                let ok = k.starts_with("conv_in.")
                    || k.starts_with("mid_block.")
                    || k.starts_with("up_blocks.")
                    || k.starts_with("conv_norm_out.")
                    || k.starts_with("conv_out.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in Decoder state_dict: \"{k}\""),
                    });
                }
            }
        }

        self.conv_in.load_state_dict(&extract("conv_in"), strict)?;
        self.mid_block
            .load_state_dict(&extract("mid_block"), strict)?;
        for (i, b) in self.up_blocks.iter_mut().enumerate() {
            b.load_state_dict(&extract(&format!("up_blocks.{i}")), strict)?;
        }
        self.conv_norm_out
            .load_state_dict(&extract("conv_norm_out"), strict)?;
        self.conv_out
            .load_state_dict(&extract("conv_out"), strict)?;
        Ok(())
    }
}

/// `AutoencoderKL`-style VAE decoder = `post_quant_conv` + [`Decoder`].
///
/// The decoder pre-divides the latent by `config.scaling_factor` when
/// using [`Self::decode_with_scaling`], matching
/// `AutoencoderKL.decode(z).sample`. [`Module::forward`] expects the
/// latent already pre-divided (this matches the order of operations the
/// SD pipeline performs externally).
#[derive(Debug)]
pub struct VaeDecoder<T: Float> {
    /// 1x1 post-quant projection over the 4 latent channels.
    pub post_quant_conv: Conv2d<T>,
    /// The actual `Decoder` stack.
    pub decoder: Decoder<T>,
    /// Frozen config copy.
    pub config: VaeDecoderConfig,
    training: bool,
}

impl<T: Float> VaeDecoder<T> {
    /// Build a randomly-initialized `VaeDecoder`.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: VaeDecoderConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        let post_quant_conv = Conv2d::<T>::new(
            cfg.latent_channels,
            cfg.latent_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            true,
        )?;
        let decoder = Decoder::<T>::new(cfg.clone())?;
        Ok(Self {
            post_quant_conv,
            decoder,
            config: cfg,
            training: false,
        })
    }

    /// Decode a latent with the SD scaling convention:
    /// `image = decoder(post_quant_conv(z / scaling_factor))`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] when the input is not
    /// `[B, latent_channels, H, W]`. Propagates downstream op errors.
    pub fn decode_with_scaling(&self, latent: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let inv = self.config.scaling_factor.recip();
        let inv_t = T::from(inv).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!(
                "VaeDecoder::decode_with_scaling: cannot cast 1/{} into Float",
                self.config.scaling_factor
            ),
        })?;
        let inv_tensor = ferrotorch_core::scalar::<T>(inv_t)?;
        let scaled = ferrotorch_core::grad_fns::arithmetic::mul(latent, &inv_tensor)?;
        self.forward(&scaled)
    }
}

impl<T: Float> Module<T> for VaeDecoder<T> {
    /// Forward expects the post-scaled latent (the caller has already
    /// divided by `scaling_factor`).
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let cfg = &self.config;
        if input.ndim() != 4 || input.shape()[1] != cfg.latent_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "VaeDecoder::forward: expected [B, {}, H, W], got {:?}",
                    cfg.latent_channels,
                    input.shape()
                ),
            });
        }
        let post = self.post_quant_conv.forward(input)?;
        self.decoder.forward(&post)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.post_quant_conv.parameters());
        out.extend(self.decoder.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.post_quant_conv.parameters_mut());
        out.extend(self.decoder.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.post_quant_conv.named_parameters() {
            out.push((format!("post_quant_conv.{n}"), p));
        }
        for (n, p) in self.decoder.named_parameters() {
            out.push((format!("decoder.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.decoder.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.decoder.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }

    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let extract = |prefix: &str| -> StateDict<T> {
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };
        if strict {
            for k in state.keys() {
                let ok = k.starts_with("post_quant_conv.") || k.starts_with("decoder.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in VaeDecoder state_dict: \"{k}\""),
                    });
                }
            }
        }
        self.post_quant_conv
            .load_state_dict(&extract("post_quant_conv"), strict)?;
        self.decoder.load_state_dict(&extract("decoder"), strict)?;
        let _: HashMap<String, Tensor<T>> = HashMap::new(); // keep HashMap import alive
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;

    /// Tiny config that still exercises every architectural feature
    /// (mid-block attn, 4 up-blocks, channel-changing resnet shortcut)
    /// without making the test slow.
    fn tiny_cfg() -> VaeDecoderConfig {
        VaeDecoderConfig {
            out_channels: 3,
            latent_channels: 4,
            // 4 blocks; channels grow with depth so the decoder's
            // *reversed* sequence is [16, 16, 8, 4] — and the
            // resnet shortcut path is exercised on each transition.
            block_out_channels: vec![4, 8, 16, 16],
            layers_per_block: 1, // => 2 resnets per up-block (faster)
            norm_num_groups: 4,
            sample_size: 8,
            scaling_factor: 0.18215,
        }
    }

    #[test]
    fn decoder_forward_shape() {
        let cfg = tiny_cfg();
        let d = Decoder::<f32>::new(cfg.clone()).unwrap();
        // latent: [1, 4, 1, 1] -> after 3 upsamples => [1, 4, 8, 8].
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 4]),
            vec![1, 4, 1, 1],
            false,
        )
        .unwrap();
        let y = d.forward(&x).unwrap();
        // 1 -> 2 -> 4 -> 8 (3 upsamples, last block has no upsample).
        assert_eq!(y.shape(), &[1, 3, 8, 8]);
        for &v in y.data().unwrap() {
            assert!(v.is_finite(), "decoder output non-finite: {v}");
        }
    }

    #[test]
    fn vae_decoder_named_parameters_include_post_quant_conv() {
        let cfg = tiny_cfg();
        let v = VaeDecoder::<f32>::new(cfg).unwrap();
        let names: Vec<String> = v.named_parameters().into_iter().map(|(n, _)| n).collect();
        for k in [
            "post_quant_conv.weight",
            "post_quant_conv.bias",
            "decoder.conv_in.weight",
            "decoder.mid_block.attentions.0.to_q.weight",
            "decoder.up_blocks.0.resnets.0.norm1.weight",
            "decoder.conv_norm_out.weight",
            "decoder.conv_out.bias",
        ] {
            assert!(names.iter().any(|n| n == k), "missing {k} in {names:?}");
        }
    }

    #[test]
    fn vae_decoder_forward_shape() {
        let cfg = tiny_cfg();
        let v = VaeDecoder::<f32>::new(cfg).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 4]),
            vec![1, 4, 1, 1],
            false,
        )
        .unwrap();
        let y = v.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 3, 8, 8]);
    }

    #[test]
    fn vae_decoder_decode_with_scaling_matches_manual_div() {
        let cfg = tiny_cfg();
        let v = VaeDecoder::<f32>::new(cfg.clone()).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.05f32; 4]),
            vec![1, 4, 1, 1],
            false,
        )
        .unwrap();
        let inv = (1.0 / cfg.scaling_factor) as f32;
        let scaled_data: Vec<f32> = x.data().unwrap().iter().map(|&v| v * inv).collect();
        let scaled =
            Tensor::from_storage(TensorStorage::cpu(scaled_data), vec![1, 4, 1, 1], false).unwrap();
        let a = v.decode_with_scaling(&x).unwrap();
        let b = v.forward(&scaled).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!(
                (x - y).abs() < 1e-4,
                "decode_with_scaling vs manual div differ: {x} vs {y}"
            );
        }
    }

    #[test]
    fn round_trip_state_dict() {
        let cfg = tiny_cfg();
        let src = VaeDecoder::<f32>::new(cfg.clone()).unwrap();
        let sd = src.state_dict();
        let mut dst = VaeDecoder::<f32>::new(cfg.clone()).unwrap();
        dst.load_state_dict(&sd, true).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 4]),
            vec![1, 4, 1, 1],
            false,
        )
        .unwrap();
        let a = src.forward(&x).unwrap();
        let b = dst.forward(&x).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-5, "round-trip differs: {x} vs {y}");
        }
    }
}
