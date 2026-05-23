//! Configuration for the Stable-Diffusion VAE decoder.
//!
//! Matches the public surface of `diffusers.AutoencoderKL.config` for the
//! fields the decoder actually consumes. Encoder-side fields (e.g.
//! `down_block_types`) are not stored — the decoder mirror is
//! decoder-only.

use ferrotorch_core::{FerrotorchError, FerrotorchResult};

/// Frozen config for the Stable-Diffusion VAE decoder.
///
/// Mirrors the decoder-relevant subset of `AutoencoderKL.config` for
/// `runwayml/stable-diffusion-v1-5`. The defaults match SD 1.5 exactly.
#[derive(Debug, Clone)]
pub struct VaeDecoderConfig {
    /// Number of input channels of the image the encoder consumes (and
    /// therefore of the image the decoder produces). For SD 1.5: 3.
    pub out_channels: usize,
    /// Number of latent channels. For SD 1.5: 4.
    pub latent_channels: usize,
    /// Per-block-level output channel counts (in encoder order: from
    /// the highest-resolution block out). For SD 1.5: `[128, 256, 512,
    /// 512]`. The decoder walks these in reverse, so the first block
    /// after `conv_in` has `block_out_channels[-1]` channels (= 512).
    pub block_out_channels: Vec<usize>,
    /// Number of resnet layers in each Encoder / Decoder up- or
    /// down-block. The decoder's `UpDecoderBlock2D` uses
    /// `layers_per_block + 1` resnets (the diffusers convention). For
    /// SD 1.5: 2 (so each up-block has 3 resnets).
    pub layers_per_block: usize,
    /// Number of GroupNorm groups (decoder-internal `norm1` / `norm2` /
    /// `conv_norm_out`). For SD 1.5: 32.
    pub norm_num_groups: usize,
    /// Spatial size the encoder accepts (and the decoder produces).
    /// For SD 1.5: 512.
    pub sample_size: usize,
    /// VAE latent scaling factor. The decoder pre-divides the latent by
    /// this value (matching `AutoencoderKL.decode`). For SD 1.5: 0.18215.
    pub scaling_factor: f64,
}

impl Default for VaeDecoderConfig {
    fn default() -> Self {
        // SD 1.5 VAE config.
        Self {
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            sample_size: 512,
            scaling_factor: 0.18215,
        }
    }
}

impl VaeDecoderConfig {
    /// SD 1.5 VAE decoder config (alias for `Default::default()`).
    pub fn sd_v1_5() -> Self {
        Self::default()
    }

    /// Validate field bounds (positive sizes, channels divisible by
    /// `norm_num_groups`, at least one resolution).
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] for any out-of-bounds
    /// or arithmetic-incompatible field.
    pub fn validate(&self) -> FerrotorchResult<()> {
        if self.block_out_channels.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "VaeDecoderConfig: block_out_channels must be non-empty".into(),
            });
        }
        if self.norm_num_groups == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "VaeDecoderConfig: norm_num_groups must be > 0".into(),
            });
        }
        for &c in &self.block_out_channels {
            if c == 0 || c % self.norm_num_groups != 0 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "VaeDecoderConfig: block_out_channels entry {c} must be > 0 and divisible \
                         by norm_num_groups={}",
                        self.norm_num_groups
                    ),
                });
            }
        }
        if self.latent_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "VaeDecoderConfig: latent_channels must be > 0".into(),
            });
        }
        if self.out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "VaeDecoderConfig: out_channels must be > 0".into(),
            });
        }
        if self.layers_per_block == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "VaeDecoderConfig: layers_per_block must be > 0".into(),
            });
        }
        if self.sample_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "VaeDecoderConfig: sample_size must be > 0".into(),
            });
        }
        if !self.scaling_factor.is_finite() || self.scaling_factor == 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "VaeDecoderConfig: scaling_factor must be finite and non-zero, got {}",
                    self.scaling_factor
                ),
            });
        }
        Ok(())
    }

    /// Number of resnets in each `UpDecoderBlock2D` (the diffusers
    /// convention is `layers_per_block + 1`).
    pub fn resnets_per_up_block(&self) -> usize {
        self.layers_per_block + 1
    }

    /// Number of up-blocks (== number of down-blocks the encoder used,
    /// == `block_out_channels.len()`).
    pub fn num_up_blocks(&self) -> usize {
        self.block_out_channels.len()
    }

    /// Parse a `vae/config.json` document into a [`VaeDecoderConfig`].
    ///
    /// Recognised keys (all optional — anything missing falls back to
    /// the SD-1.5 defaults):
    ///   - `out_channels`, `latent_channels`, `block_out_channels`,
    ///     `layers_per_block`, `norm_num_groups`, `sample_size`,
    ///     `scaling_factor`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] on malformed JSON or
    /// a wrong-type field (e.g. `block_out_channels` not an array of
    /// integers).
    pub fn from_json_str(s: &str) -> FerrotorchResult<Self> {
        let v: serde_json::Value =
            serde_json::from_str(s).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("VaeDecoderConfig::from_json_str: bad JSON: {e}"),
            })?;
        let mut cfg = Self::default();
        if let Some(x) = v.get("out_channels").and_then(serde_json::Value::as_u64) {
            cfg.out_channels = x as usize;
        }
        if let Some(x) = v.get("latent_channels").and_then(serde_json::Value::as_u64) {
            cfg.latent_channels = x as usize;
        }
        if let Some(arr) = v
            .get("block_out_channels")
            .and_then(serde_json::Value::as_array)
        {
            let mut out = Vec::with_capacity(arr.len());
            for e in arr {
                let n = e.as_u64().ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!(
                        "VaeDecoderConfig::from_json_str: block_out_channels entry \
                         must be a non-negative integer, got {e}"
                    ),
                })?;
                out.push(n as usize);
            }
            cfg.block_out_channels = out;
        }
        if let Some(x) = v
            .get("layers_per_block")
            .and_then(serde_json::Value::as_u64)
        {
            cfg.layers_per_block = x as usize;
        }
        if let Some(x) = v.get("norm_num_groups").and_then(serde_json::Value::as_u64) {
            cfg.norm_num_groups = x as usize;
        }
        if let Some(x) = v.get("sample_size").and_then(serde_json::Value::as_u64) {
            cfg.sample_size = x as usize;
        }
        if let Some(x) = v.get("scaling_factor").and_then(serde_json::Value::as_f64) {
            cfg.scaling_factor = x;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse a `vae/config.json` file from disk.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] for I/O or parse
    /// failures (file missing, malformed JSON, wrong-type field).
    pub fn from_file(path: &std::path::Path) -> FerrotorchResult<Self> {
        let s = std::fs::read_to_string(path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "VaeDecoderConfig::from_file: failed to read {}: {e}",
                path.display()
            ),
        })?;
        Self::from_json_str(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_sd_v1_5() {
        let c = VaeDecoderConfig::default();
        assert_eq!(c.block_out_channels, vec![128, 256, 512, 512]);
        assert_eq!(c.layers_per_block, 2);
        assert_eq!(c.latent_channels, 4);
        assert_eq!(c.norm_num_groups, 32);
        assert_eq!(c.sample_size, 512);
        assert_eq!(c.resnets_per_up_block(), 3);
        assert_eq!(c.num_up_blocks(), 4);
        // Match the published `scaling_factor` exactly.
        assert!((c.scaling_factor - 0.18215).abs() < 1e-9);
        c.validate().unwrap();
    }

    #[test]
    fn validate_catches_bad_groups() {
        // 128 not divisible by 33
        let c = VaeDecoderConfig {
            norm_num_groups: 33,
            ..VaeDecoderConfig::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn from_json_str_round_trip() {
        let json = r#"{
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D",
                               "UpDecoderBlock2D", "UpDecoderBlock2D"],
            "block_out_channels": [128, 256, 512, 512],
            "layers_per_block": 2,
            "act_fn": "silu",
            "latent_channels": 4,
            "norm_num_groups": 32,
            "sample_size": 512,
            "scaling_factor": 0.18215
        }"#;
        let c = VaeDecoderConfig::from_json_str(json).unwrap();
        assert_eq!(c.block_out_channels, vec![128, 256, 512, 512]);
        assert_eq!(c.layers_per_block, 2);
        assert_eq!(c.sample_size, 512);
    }
}
