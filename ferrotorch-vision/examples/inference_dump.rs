//! Inference-dump binary for crosslink #1139 verification.
//!
//! Loads one of the 5 pinned pretrained models, runs forward on a single image
//! (with model-appropriate preprocessing), and dumps the output to disk in a
//! deterministic format. The companion Python script in
//! `scripts/verify_pretrained_inference.py` reads these dumps and compares
//! them against torchvision reference outputs.
//!
//! Usage:
//! ```text
//! cargo run -p ferrotorch-vision --release --example inference_dump -- \
//!     --model <name> --image <path.jpg> --output <path.bin>
//! ```
//!
//! Output format (raw little-endian):
//!   [u32: ndim][u32 × ndim: dims][f32 × prod(dims): data]
//!
//! The dumper deliberately uses `vision::get_model("<name>", true, num_classes)`
//! (the architect-mandated path) so we exercise the registry weight-loading
//! pipeline. `Module::forward` returns:
//!   SSD300         → [N_det, num_classes]  (first-image per-anchor class scores)
//!   FasterRCNN     → [N_det, num_classes]  (first-image per-proposal class scores)
//!   MaskRCNN       → [N_det, num_classes, 28, 28]  (first-image mask logits)
//!   DeepLabV3      → [B, num_classes, H, W]  (per-pixel class logits)
//!   FCN            → [B, num_classes, H, W]  (per-pixel class logits)

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor};
use ferrotorch_nn::{InterpolateMode, interpolate};
use ferrotorch_vision::io::read_image_as_tensor;
use ferrotorch_vision::models::get_model;

fn parse_args() -> Result<(String, PathBuf, PathBuf), String> {
    let args: Vec<String> = env::args().collect();
    let mut model: Option<String> = None;
    let mut image: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = Some(
                    args.get(i + 1)
                        .ok_or("--model needs a value")?
                        .clone(),
                );
                i += 2;
            }
            "--image" => {
                image = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--image needs a value")?,
                ));
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--output needs a value")?,
                ));
                i += 2;
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    Ok((
        model.ok_or("--model required")?,
        image.ok_or("--image required")?,
        output.ok_or("--output required")?,
    ))
}

/// Number of classes per model, matching the registered pretrained weights.
fn num_classes_for(model: &str) -> Result<usize, String> {
    match model {
        "ssd300_vgg16" => Ok(91),
        "fasterrcnn_resnet50_fpn" => Ok(91),
        "maskrcnn_resnet50_fpn" => Ok(91),
        "deeplabv3_resnet50" => Ok(21),
        "fcn_resnet50" => Ok(21),
        other => Err(format!("unknown model: {other}")),
    }
}

/// Manually bilinear-resize a `[C, H, W]` tensor's spatial dims to `(out_h, out_w)`.
///
/// Mirrors `torch.nn.functional.interpolate(mode='bilinear', align_corners=False,
/// antialias=False)` exactly — used to avoid the nearest-neighbour ferrotorch
/// `Resize` transform that would diverge from torchvision's resizing policy.
fn bilinear_resize_chw_to_bchw(
    chw: &Tensor<f32>,
    out_h: usize,
    out_w: usize,
) -> FerrotorchResult<Tensor<f32>> {
    let shape = chw.shape().to_vec();
    if shape.len() != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("expected [C, H, W], got {shape:?}"),
        });
    }
    // Promote to [1, C, H, W] for interpolate.
    let data = chw.data_vec()?;
    let bchw = Tensor::from_storage(
        ferrotorch_core::TensorStorage::cpu(data),
        vec![1, shape[0], shape[1], shape[2]],
        false,
    )?;
    interpolate(
        &bchw,
        Some([out_h, out_w]),
        None,
        InterpolateMode::Bilinear,
        false,
    )
}

/// Per-channel normalize a `[1, C, H, W]` tensor in-place semantics with
/// `(x - mean) / std`.
fn normalize_bchw(
    bchw: &Tensor<f32>,
    mean: [f32; 3],
    std: [f32; 3],
) -> FerrotorchResult<Tensor<f32>> {
    let shape = bchw.shape().to_vec();
    let b = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];
    if c != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("normalize_bchw expects C=3, got shape {shape:?}"),
        });
    }
    let mut data = bchw.data_vec()?;
    let plane = h * w;
    for bi in 0..b {
        for ci in 0..3 {
            let base = (bi * c + ci) * plane;
            let m = mean[ci];
            let s = std[ci];
            for i in 0..plane {
                data[base + i] = (data[base + i] - m) / s;
            }
        }
    }
    Tensor::from_storage(
        ferrotorch_core::TensorStorage::cpu(data),
        shape,
        false,
    )
}

/// Build a `[1, 3, H_out, W_out]` input tensor following the model's
/// torchvision preprocessing recipe.
///
/// Detection (SSD/FasterRCNN/MaskRCNN): torchvision `ObjectDetection` transform
///   just rescales u8→f32 in `[0,1]`. The model's internal
///   `GeneralizedRCNNTransform` (FasterRCNN/MaskRCNN) or its anchor layout
///   (SSD) handles further resize/normalize.
///
/// Because ferrotorch's SSD/FasterRCNN/MaskRCNN do NOT include a
/// `GeneralizedRCNNTransform` (they expect already-preprocessed input — see
/// e.g. `Ssd300::forward` doc: `[B, 3, 300, 300] tensor (RGB, normalised to
/// ImageNet stats)`), we reproduce the recipe here:
///
/// - SSD300: bilinear-resize to 300×300, normalize with ImageNet stats.
///   (Note: torchvision SSD300 uses non-ImageNet stats; we follow
///   ferrotorch's documented expectation — this is itself a candidate for a
///   divergence diagnosis.)
/// - FasterRCNN/MaskRCNN: resize so min(H,W)=800 keeping aspect, max(H,W)≤1333,
///   normalize with ImageNet stats. Pad to multiple of 32 (FPN stride).
/// - DeepLabV3/FCN: resize shorter side to 520 keeping aspect, normalize
///   ImageNet stats.
fn preprocess_for_model(model: &str, raw_chw: Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
    let shape = raw_chw.shape().to_vec();
    let h_in = shape[1];
    let w_in = shape[2];

    match model {
        "ssd300_vgg16" => {
            let resized = bilinear_resize_chw_to_bchw(&raw_chw, 300, 300)?;
            normalize_bchw(&resized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        }
        "fasterrcnn_resnet50_fpn" | "maskrcnn_resnet50_fpn" => {
            // torchvision GeneralizedRCNNTransform: scale so min side = 800,
            // max side ≤ 1333; preserve aspect ratio.
            let min_size = 800.0_f64;
            let max_size = 1333.0_f64;
            let h = h_in as f64;
            let w = w_in as f64;
            let s_min = min_size / h.min(w);
            let s_max = max_size / h.max(w);
            let scale = s_min.min(s_max);
            let out_h = (h * scale).round() as usize;
            let out_w = (w * scale).round() as usize;
            let resized = bilinear_resize_chw_to_bchw(&raw_chw, out_h, out_w)?;
            let normed =
                normalize_bchw(&resized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])?;
            // Pad to multiple of 32 (FPN stride).
            let stride: usize = 32;
            let pad_h = out_h.div_ceil(stride) * stride;
            let pad_w = out_w.div_ceil(stride) * stride;
            if pad_h == out_h && pad_w == out_w {
                Ok(normed)
            } else {
                // Zero-pad on the bottom/right.
                let normed_data = normed.data_vec()?;
                let c = 3;
                let mut padded = vec![0.0_f32; c * pad_h * pad_w];
                for ci in 0..c {
                    for r in 0..out_h {
                        let src_base = (ci * out_h + r) * out_w;
                        let dst_base = (ci * pad_h + r) * pad_w;
                        padded[dst_base..dst_base + out_w]
                            .copy_from_slice(&normed_data[src_base..src_base + out_w]);
                    }
                }
                Tensor::from_storage(
                    ferrotorch_core::TensorStorage::cpu(padded),
                    vec![1, c, pad_h, pad_w],
                    false,
                )
            }
        }
        "deeplabv3_resnet50" | "fcn_resnet50" => {
            // torchvision SemanticSegmentation: resize shorter side to 520,
            // preserve aspect ratio.
            let resize_size = 520.0_f64;
            let h = h_in as f64;
            let w = w_in as f64;
            let scale = resize_size / h.min(w);
            let out_h = (h * scale).round() as usize;
            let out_w = (w * scale).round() as usize;
            let resized = bilinear_resize_chw_to_bchw(&raw_chw, out_h, out_w)?;
            normalize_bchw(&resized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        }
        other => Err(FerrotorchError::InvalidArgument {
            message: format!("unknown model: {other}"),
        }),
    }
}

/// Write a `Tensor<f32>` to disk as raw little-endian:
///   `[u32 ndim][u32 × ndim shape][f32 × numel data]`.
fn dump_tensor(t: &Tensor<f32>, path: &PathBuf) -> FerrotorchResult<()> {
    let shape = t.shape().to_vec();
    let data = t.data_vec()?;
    let mut f = File::create(path).map_err(|e| FerrotorchError::Internal {
        message: format!("failed to open output: {e}"),
    })?;
    let ndim = shape.len() as u32;
    f.write_all(&ndim.to_le_bytes())
        .map_err(|e| FerrotorchError::Internal {
            message: format!("write ndim: {e}"),
        })?;
    for d in &shape {
        let d32 = *d as u32;
        f.write_all(&d32.to_le_bytes())
            .map_err(|e| FerrotorchError::Internal {
                message: format!("write dim: {e}"),
            })?;
    }
    for v in &data {
        f.write_all(&v.to_le_bytes())
            .map_err(|e| FerrotorchError::Internal {
                message: format!("write val: {e}"),
            })?;
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let (model_name, image_path, output_path) = parse_args()?;
    let num_classes = num_classes_for(&model_name)?;

    eprintln!("[inference_dump] model={model_name} image={image_path:?} num_classes={num_classes}");

    // Load raw image as [C, H, W] tensor in [0, 1].
    let raw = read_image_as_tensor::<f32>(&image_path)
        .map_err(|e| format!("read_image_as_tensor: {e}"))?;
    eprintln!("[inference_dump] raw image shape: {:?}", raw.shape());

    // Preprocess according to torchvision recipe for this model.
    let input =
        preprocess_for_model(&model_name, raw).map_err(|e| format!("preprocess: {e}"))?;
    eprintln!("[inference_dump] preprocessed shape: {:?}", input.shape());

    // Build model via the architect-mandated registry path; this loads
    // pretrained weights from the local hub cache (pinned in #1130).
    let mut model =
        get_model(&model_name, true, num_classes).map_err(|e| format!("get_model: {e}"))?;
    model.eval();
    eprintln!("[inference_dump] model loaded; running forward...");

    let output = model
        .forward(&input)
        .map_err(|e| format!("forward: {e}"))?;
    eprintln!("[inference_dump] output shape: {:?}", output.shape());

    dump_tensor(&output, &output_path).map_err(|e| format!("dump_tensor: {e}"))?;
    eprintln!("[inference_dump] dumped to {output_path:?}");

    Ok(())
}
