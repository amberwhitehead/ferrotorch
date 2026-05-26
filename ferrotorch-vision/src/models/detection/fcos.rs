//! FCOS anchor-free one-stage detector with ResNet-50 + FPN backbone.
//!
//! Mirrors `torchvision.models.detection.fcos_resnet50_fpn`
//! (`FCOS_ResNet50_FPN_Weights.COCO_V1` checkpoint).
//!
//! ## Architecture
//!
//! ```text
//! image [B, 3, H, W]
//!   └─ ResNet-50 backbone → {layer2, layer3, layer4} (C3-C5)
//!         └─ FPN (P3-P7)  — IDENTICAL to RetinaNet's P3-P7 FPN with
//!                           LastLevelP6P7 extras (`RetinaFpn`).
//!         └─ FCOS classification head — shared 4× (Conv 3×3 + GroupNorm(32)
//!                                       + ReLU), final Conv 3×3 outputting
//!                                       `num_classes=91` channels.
//!         └─ FCOS regression head      — shared 4× (Conv 3×3 + GroupNorm(32)
//!                                       + ReLU), two parallel output convs:
//!                                       `bbox_reg`     Conv 3×3 → 4 channels
//!                                                      (passed through ReLU)
//!                                       `bbox_ctrness` Conv 3×3 → 1 channel
//!                                                      (raw logits)
//!   ↳ AnchorGenerator: ONE anchor per spatial location at stride S of level,
//!                      box = `[-S/2, -S/2, +S/2, +S/2]` rounded, shifted
//!                      to `(col*stride_w, row*stride_h)`.
//!   ↳ postprocess: per-level score = sqrt(sigmoid(cls) * sigmoid(centerness));
//!                  filter score > 0.2; per-level top-K (1000); decode boxes
//!                  via BoxLinearCoder (normalize_by_size=True); clip;
//!                  cross-class batched_nms (IoU 0.6); detections_per_img=100.
//! ```
//!
//! Distinct from RetinaNet (the other ResNet-50 + FPN single-stage detector):
//! - **Anchor-free**: one box per FPN cell, regression predicts `(l, t, r, b)`
//!   distances from cell center to box edges.
//! - **GroupNorm** in the heads (RetinaNet uses no norm).
//! - **Centerness branch**: additional 1-channel head that gates the
//!   classification score during post-processing.
//! - **No focal-loss prior** at the cls_logits bias (we still set
//!   `-log((1-π)/π)` for π=0.01 at construction since torchvision does, but
//!   the parity test loads pretrained weights that overwrite this).
//!
//! ## Reference
//! Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection",
//! ICCV 2019. torchvision 0.21.x `fcos_resnet50_fpn(weights="COCO_V1")`.
//!
//! ## REQ status (per `.design/ferrotorch-vision/models/detection/fcos.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: `pub const FCOS_NUM_ANCHORS_PER_LOC: usize = 1;` in `fcos.rs`; consumer: `Fcos::new` in same file passes it to both head constructors. |
//! | REQ-2 | SHIPPED | impl: `pub const FCOS_NUM_CONVS: usize = 4;` in `fcos.rs`; consumer: `FcosConvGnTrunk::new` in same file uses the constant to size the conv and GroupNorm arrays — invoked via `Fcos::new` through the head constructors. |
//! | REQ-3 | SHIPPED | impl: `pub const FCOS_GN_GROUPS: usize = 32;` in `fcos.rs`; consumer: `FcosConvGnTrunk::new` reads it inside `Fcos::new`. |
//! | REQ-4 | SHIPPED | impl: `pub const FCOS_BASE_SIZES: [f64; 5] = [8, 16, 32, 64, 128];` in `fcos.rs`; consumer: `fcos_anchors_per_level` reads it inside `Fcos::forward`. |
//! | REQ-5 | SHIPPED | impl: `pub const FCOS_SCORE_THRESH` / `_NMS_THRESH` / `_TOPK_CANDIDATES` / `_DETECTIONS_PER_IMG` in `fcos.rs`; consumer: `Fcos::forward` reads each — registered as a constructor via `register_model("fcos_resnet50_fpn", ...)` in `ferrotorch-vision/src/models/registry.rs`. |
//! | REQ-6 | SHIPPED | impl: `FcosConvGnTrunk<T>` (module-private) + `Self::named_parameters` mapping in `fcos.rs` (Sequential-style `conv.{0,3,6,9}` for `Conv2d` and `conv.{1,4,7,10}` for `GroupNorm`); consumer: both head structs in same file own one; named parameters surface through `FcosClassificationHead::named_parameters` (public method) consumed by `Fcos::named_parameters` and reachable from the registry. |
//! | REQ-7 | SHIPPED | impl: `pub struct FcosClassificationHead<T>` + `Self::new` + `Self::forward_level` in `fcos.rs` (trunk + final `cls_logits` conv, no ReLU on logits); consumer: `Fcos::new` in same file calls `FcosClassificationHead::new(FPN_OUT_CHANNELS, FCOS_NUM_ANCHORS_PER_LOC, num_classes)?`. |
//! | REQ-8 | SHIPPED | impl: `pub struct FcosRegressionHead<T>` + `Self::new` + `Self::forward_level` in `fcos.rs` (parallel `bbox_reg` ReLU-gated + `bbox_ctrness` raw); consumer: `Fcos::new` in same file calls `FcosRegressionHead::new(FPN_OUT_CHANNELS, FCOS_NUM_ANCHORS_PER_LOC)?`. |
//! | REQ-9 | SHIPPED | impl: `fn fcos_anchors_per_level` in `fcos.rs` (one anchor per cell at `[-S/2, -S/2, +S/2, +S/2]` rounded, shifted by `(col*stride_w, row*stride_h)`); consumer: `Fcos::forward` calls `fcos_anchors_per_level::<T>(&fm_sizes, (img_h, img_w))?`. |
//! | REQ-10 | SHIPPED | impl: `pub struct Fcos<T>` + `Self::new` in `fcos.rs` (composes `ResNet`, `RetinaFpn`, classification head, regression head); consumer: `register_model("fcos_resnet50_fpn", ...)` in `ferrotorch-vision/src/models/registry.rs`. |
//! | REQ-11 | SHIPPED | impl: `pub fn Fcos::forward` body in `fcos.rs` (`sqrt(sigmoid(cls) * sigmoid(centerness))` → score-thresh → per-level top-K → `BoxLinearCoder(normalize_by_size=True)` decode → clip → cross-class `batched_nms` (IoU 0.6) → top-K 100); consumer: `impl<T> Module<T> for Fcos<T>::forward` invokes it; the registry closure in `ferrotorch-vision/src/models/registry.rs` reaches it via `Module::forward`. |
//! | REQ-12 | SHIPPED | impl: `impl<T> Module<T> for Fcos<T>::forward` in `fcos.rs` returns first-image scores as a 1-D `[N_det]` tensor; consumer: registered as `ModelConstructor<f32>` via `register_model("fcos_resnet50_fpn", ...)` in `ferrotorch-vision/src/models/registry.rs`. |
//! | REQ-13 | SHIPPED | impl: `pub fn fcos_resnet50_fpn` in `fcos.rs`; consumer: `register_model("fcos_resnet50_fpn", ...)` in `ferrotorch-vision/src/models/registry.rs` calls it inside the closure. |

use std::collections::HashMap;

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Conv2d, GroupNorm};

use crate::models::detection::fpn::FPN_OUT_CHANNELS;
use crate::models::detection::retinanet::RetinaFpn;
use crate::models::feature_extractor::IntermediateFeatures;
use crate::models::resnet::{ResNet, resnet50};
use crate::ops::{batched_nms, clip_boxes_to_image};

// ---------------------------------------------------------------------------
// Constants — mirror torchvision FCOS defaults
// ---------------------------------------------------------------------------

/// FCOS has one anchor per spatial location (1.0 aspect ratio, single size
/// equal to the level's stride). See torchvision
/// `fcos.py::FCOS.__init__::anchor_generator` default.
pub const FCOS_NUM_ANCHORS_PER_LOC: usize = 1;

/// Number of conv layers in each head's shared trunk (matches torchvision's
/// `num_convs=4` default).
pub const FCOS_NUM_CONVS: usize = 4;

/// GroupNorm group count in head trunks.
pub const FCOS_GN_GROUPS: usize = 32;

/// Per-cell anchor base sizes for P3..P7 (equal to the level's stride —
/// matches torchvision `anchor_sizes = ((8,), (16,), (32,), (64,), (128,))`).
pub const FCOS_BASE_SIZES: [f64; 5] = [8.0, 16.0, 32.0, 64.0, 128.0];

/// Per-class score gate (matches `FCOS(score_thresh=0.2)`).
pub const FCOS_SCORE_THRESH: f64 = 0.2;

/// NMS IoU threshold (matches `FCOS(nms_thresh=0.6)`).
pub const FCOS_NMS_THRESH: f64 = 0.6;

/// Per-level top-K candidates pre-NMS (matches `FCOS(topk_candidates=1000)`).
pub const FCOS_TOPK_CANDIDATES: usize = 1000;

/// Cross-class detection cap per image (matches `FCOS(detections_per_img=100)`).
pub const FCOS_DETECTIONS_PER_IMG: usize = 100;

// ---------------------------------------------------------------------------
// Head: 4 × (Conv + GN + ReLU) trunk
// ---------------------------------------------------------------------------

/// Shared 4-layer Conv + GroupNorm + ReLU trunk used by both the FCOS
/// classification and regression heads. Mirrors the
/// `Sequential[Conv2d, GroupNorm, ReLU, Conv2d, GroupNorm, ReLU, ...]`
/// layout that torchvision builds inline (no `Conv2dNormActivation`
/// wrapper); ferrotorch named_parameters expose the same `conv.{i}.*`
/// indexing so the pinning script can pass torchvision keys through
/// unchanged.
///
/// Layout (mirrors torchvision's `self.conv = nn.Sequential(*conv)`):
///
/// | index | module          | named params                           |
/// |-------|-----------------|----------------------------------------|
/// | 0     | Conv2d(256,256) | `conv.0.weight`, `conv.0.bias`         |
/// | 1     | GroupNorm(32)   | `conv.1.weight`, `conv.1.bias`         |
/// | 2     | ReLU            | (none)                                 |
/// | 3     | Conv2d(256,256) | `conv.3.weight`, `conv.3.bias`         |
/// | 4     | GroupNorm(32)   | `conv.4.weight`, `conv.4.bias`         |
/// | 5     | ReLU            | (none)                                 |
/// | 6     | Conv2d(256,256) | `conv.6.weight`, `conv.6.bias`         |
/// | 7     | GroupNorm(32)   | `conv.7.weight`, `conv.7.bias`         |
/// | 8     | ReLU            | (none)                                 |
/// | 9     | Conv2d(256,256) | `conv.9.weight`, `conv.9.bias`         |
/// | 10    | GroupNorm(32)   | `conv.10.weight`, `conv.10.bias`       |
/// | 11    | ReLU            | (none)                                 |
struct FcosConvGnTrunk<T: Float> {
    convs: [Conv2d<T>; FCOS_NUM_CONVS],
    gns: [GroupNorm<T>; FCOS_NUM_CONVS],
}

impl<T: Float> FcosConvGnTrunk<T> {
    fn new(in_channels: usize) -> FerrotorchResult<Self> {
        let conv0 = Conv2d::new(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true)?;
        let conv1 = Conv2d::new(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true)?;
        let conv2 = Conv2d::new(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true)?;
        let conv3 = Conv2d::new(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true)?;
        let gn0 = GroupNorm::new(FCOS_GN_GROUPS, in_channels, 1e-5, true)?;
        let gn1 = GroupNorm::new(FCOS_GN_GROUPS, in_channels, 1e-5, true)?;
        let gn2 = GroupNorm::new(FCOS_GN_GROUPS, in_channels, 1e-5, true)?;
        let gn3 = GroupNorm::new(FCOS_GN_GROUPS, in_channels, 1e-5, true)?;
        Ok(Self {
            convs: [conv0, conv1, conv2, conv3],
            gns: [gn0, gn1, gn2, gn3],
        })
    }

    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut h = self.convs[0].forward(x)?;
        h = self.gns[0].forward(&h)?;
        h = relu(&h)?;
        h = self.convs[1].forward(&h)?;
        h = self.gns[1].forward(&h)?;
        h = relu(&h)?;
        h = self.convs[2].forward(&h)?;
        h = self.gns[2].forward(&h)?;
        h = relu(&h)?;
        h = self.convs[3].forward(&h)?;
        h = self.gns[3].forward(&h)?;
        h = relu(&h)?;
        Ok(h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        for i in 0..FCOS_NUM_CONVS {
            p.extend(self.convs[i].parameters());
            p.extend(self.gns[i].parameters());
        }
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        // We need disjoint borrows by interleaving conv[i] and gn[i] for each
        // i without aliasing — iterate the arrays in lockstep, mutably.
        let convs_iter = self.convs.iter_mut();
        let gns_iter = self.gns.iter_mut();
        for (c, g) in convs_iter.zip(gns_iter) {
            p.extend(c.parameters_mut());
            p.extend(g.parameters_mut());
        }
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        // Match torchvision's Sequential indexing: Conv at 0,3,6,9 and GN at
        // 1,4,7,10. ReLU positions 2,5,8,11 contribute no parameters.
        let mut out = Vec::new();
        let conv_idx = [0usize, 3, 6, 9];
        let gn_idx = [1usize, 4, 7, 10];
        for i in 0..FCOS_NUM_CONVS {
            for (n, p) in self.convs[i].named_parameters() {
                out.push((format!("conv.{}.{n}", conv_idx[i]), p));
            }
            for (n, p) in self.gns[i].named_parameters() {
                out.push((format!("conv.{}.{n}", gn_idx[i]), p));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Classification head
// ---------------------------------------------------------------------------

/// Shared classification head for FCOS.
///
/// Trunk: 4 × (Conv 3×3 + GroupNorm(32) + ReLU).
/// Final: Conv 3×3 producing `num_anchors * num_classes` channels.
///
/// torchvision initializes the final-conv bias to `-log((1-π)/π)` for
/// `π=0.01` (focal-loss prior). The COCO_V1 pretrained checkpoint
/// immediately overwrites this, so we follow the same pattern as
/// RetinaNet (#1143) and rely on the pinned weight load for the
/// pretrained inference path. The default `Conv2d::new` init suffices
/// for from-scratch construction in tests.
pub struct FcosClassificationHead<T: Float> {
    trunk: FcosConvGnTrunk<T>,
    cls_logits: Conv2d<T>,
    num_anchors: usize,
    num_classes: usize,
}

impl<T: Float> FcosClassificationHead<T> {
    pub fn new(
        in_channels: usize,
        num_anchors: usize,
        num_classes: usize,
    ) -> FerrotorchResult<Self> {
        let trunk = FcosConvGnTrunk::new(in_channels)?;
        let cls_logits = Conv2d::new(
            in_channels,
            num_anchors * num_classes,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
        )?;
        Ok(Self {
            trunk,
            cls_logits,
            num_anchors,
            num_classes,
        })
    }

    /// Forward on a single feature map `[B, C, H, W]`. Returns
    /// `[B, H*W*num_anchors, num_classes]` logits — same permute/reshape as
    /// `FCOSClassificationHead.forward` (which produces `(N, HWA, K)`).
    pub fn forward_level(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h = self.trunk.forward(x)?;
        let logits = self.cls_logits.forward(&h)?; // [B, A*K, H, W]
        permute_a_k_hw_to_hwa_k(&logits, self.num_anchors, self.num_classes)
    }

    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.trunk.parameters();
        p.extend(self.cls_logits.parameters());
        p
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.trunk.parameters_mut();
        p.extend(self.cls_logits.parameters_mut());
        p
    }

    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = self.trunk.named_parameters();
        for (n, p) in self.cls_logits.named_parameters() {
            out.push((format!("cls_logits.{n}"), p));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Regression head
// ---------------------------------------------------------------------------

/// Shared regression head for FCOS — produces `(l, t, r, b)` distances AND a
/// per-cell centerness logit.
///
/// Trunk: 4 × (Conv 3×3 + GroupNorm(32) + ReLU), shared by both heads.
/// `bbox_reg`: Conv 3×3 → `num_anchors * 4` channels, passed through ReLU.
/// `bbox_ctrness`: Conv 3×3 → `num_anchors * 1` channels (raw logits).
///
/// The ReLU on `bbox_reg` is critical — torchvision applies it on every
/// forward, so the live `(l, t, r, b)` predictions are non-negative even
/// before decode (see `fcos.py::FCOSRegressionHead.forward`, line:
/// `bbox_regression = nn.functional.relu(self.bbox_reg(bbox_feature))`).
pub struct FcosRegressionHead<T: Float> {
    trunk: FcosConvGnTrunk<T>,
    bbox_reg: Conv2d<T>,
    bbox_ctrness: Conv2d<T>,
    num_anchors: usize,
}

impl<T: Float> FcosRegressionHead<T> {
    pub fn new(in_channels: usize, num_anchors: usize) -> FerrotorchResult<Self> {
        let trunk = FcosConvGnTrunk::new(in_channels)?;
        let bbox_reg = Conv2d::new(in_channels, num_anchors * 4, (3, 3), (1, 1), (1, 1), true)?;
        let bbox_ctrness = Conv2d::new(in_channels, num_anchors, (3, 3), (1, 1), (1, 1), true)?;
        Ok(Self {
            trunk,
            bbox_reg,
            bbox_ctrness,
            num_anchors,
        })
    }

    /// Forward on a single feature map. Returns
    /// `(bbox_reg [B, H*W*A, 4], bbox_ctrness [B, H*W*A, 1])`.
    pub fn forward_level(&self, x: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
        let h = self.trunk.forward(x)?;
        // bbox_reg goes through ReLU on every forward.
        let raw_reg = self.bbox_reg.forward(&h)?;
        let reg = relu(&raw_reg)?;
        let ctr = self.bbox_ctrness.forward(&h)?;

        let reg_out = permute_a_k_hw_to_hwa_k(&reg, self.num_anchors, 4)?;
        let ctr_out = permute_a_k_hw_to_hwa_k(&ctr, self.num_anchors, 1)?;
        Ok((reg_out, ctr_out))
    }

    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.trunk.parameters();
        p.extend(self.bbox_reg.parameters());
        p.extend(self.bbox_ctrness.parameters());
        p
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.trunk.parameters_mut();
        p.extend(self.bbox_reg.parameters_mut());
        p.extend(self.bbox_ctrness.parameters_mut());
        p
    }

    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = self.trunk.named_parameters();
        for (n, p) in self.bbox_reg.named_parameters() {
            out.push((format!("bbox_reg.{n}"), p));
        }
        for (n, p) in self.bbox_ctrness.named_parameters() {
            out.push((format!("bbox_ctrness.{n}"), p));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Helper: tensor permute `[B, A*K, H, W]` → `[B, H*W*A, K]`
// ---------------------------------------------------------------------------

/// Mirror of the `view → permute → reshape` block in torchvision's
/// FCOS heads. Layout:
///
/// `(B, A*K, H, W) → (B, A, K, H, W) → (B, H, W, A, K) → (B, H*W*A, K)`.
///
/// Written as an explicit index loop rather than chained tensor ops to avoid
/// allocating intermediate permuted views (same approach RetinaNet uses).
fn permute_a_k_hw_to_hwa_k<T: Float>(
    x: &Tensor<T>,
    num_anchors: usize,
    k: usize,
) -> FerrotorchResult<Tensor<T>> {
    let shape = x.shape();
    let b = shape[0];
    let ak = shape[1];
    let hh = shape[2];
    let ww = shape[3];
    let a = num_anchors;
    debug_assert_eq!(ak, a * k);
    let data = x.data_vec()?;
    let mut out = vec![cast::<f64, T>(0.0)?; b * hh * ww * a * k];
    for bi in 0..b {
        for hi in 0..hh {
            for wi in 0..ww {
                for ai in 0..a {
                    for ki in 0..k {
                        let src = ((bi * a + ai) * k + ki) * hh * ww + hi * ww + wi;
                        let dst = ((bi * hh + hi) * ww + wi) * a * k + ai * k + ki;
                        out[dst] = data[src];
                    }
                }
            }
        }
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![b, hh * ww * a, k], false)
}

// ---------------------------------------------------------------------------
// FCOS anchor generation
// ---------------------------------------------------------------------------

/// Generate FCOS anchors for every cell of every level.
///
/// Per torchvision:
/// - sizes = `((8,), (16,), (32,), (64,), (128,))` (one size per level).
/// - aspect_ratios = `((1.0,),) * 5` (single aspect ratio, square anchors).
/// - cell anchor (zero-centred): `[-s/2, -s/2, +s/2, +s/2]` rounded.
/// - shifts: `shifts_x = arange(grid_w) * stride_w` (NO 0.5 cell-centre
///   offset — this is unusual; ferrotorch's RetinaNet anchor builder also
///   omits it, so we follow suit).
///
/// Per-dim strides come from `image_size // grid_size` (matching torchvision's
/// `AnchorGenerator.forward`).
///
/// Returns `Vec<Tensor<T>>` of length 5 (P3..P7), each `[H*W, 4]` xyxy.
fn fcos_anchors_per_level<T: Float>(
    feature_map_sizes: &[(usize, usize); 5],
    image_size: (usize, usize),
) -> FerrotorchResult<Vec<Tensor<T>>> {
    let mut levels: Vec<Tensor<T>> = Vec::with_capacity(5);
    for (level_idx, &(fh, fw)) in feature_map_sizes.iter().enumerate() {
        let size = FCOS_BASE_SIZES[level_idx];
        // generate_anchors: ws = hs = size; base = [-w/2,-h/2,w/2,h/2].round().
        let half: f64 = (size * 0.5).round();
        let half_t: T = cast(half)?;
        let neg_half_t: T = cast::<f64, T>(0.0)? - half_t;

        let sh = image_size.0.checked_div(fh).unwrap_or(1);
        let sw = image_size.1.checked_div(fw).unwrap_or(1);
        let stride_h_t: T = cast(sh as f64)?;
        let stride_w_t: T = cast(sw as f64)?;

        let mut all: Vec<T> = Vec::with_capacity(fh * fw * 4);
        for fy in 0..fh {
            for fx in 0..fw {
                let cx: T = cast::<usize, T>(fx)? * stride_w_t;
                let cy: T = cast::<usize, T>(fy)? * stride_h_t;
                all.push(cx + neg_half_t);
                all.push(cy + neg_half_t);
                all.push(cx + half_t);
                all.push(cy + half_t);
            }
        }
        let n = all.len() / 4;
        levels.push(Tensor::from_storage(
            TensorStorage::cpu(all),
            vec![n, 4],
            false,
        )?);
    }
    Ok(levels)
}

// ---------------------------------------------------------------------------
// Detection output
// ---------------------------------------------------------------------------

/// Per-image FCOS detection output.
#[derive(Debug, Clone)]
pub struct Detections<T: Float> {
    /// Predicted boxes `[N_det, 4]` in xyxy pixel coords.
    pub boxes: Tensor<T>,
    /// Per-detection score (`sqrt(sigmoid(cls) * sigmoid(centerness))`),
    /// `[N_det]`.
    pub scores: Tensor<T>,
    /// Predicted class label `[N_det]` — 0-indexed over `num_classes`.
    pub labels: Vec<usize>,
}

// ---------------------------------------------------------------------------
// FCOS
// ---------------------------------------------------------------------------

/// FCOS anchor-free single-stage detector.
pub struct Fcos<T: Float> {
    backbone: ResNet<T>,
    fpn: RetinaFpn<T>,
    classification_head: FcosClassificationHead<T>,
    regression_head: FcosRegressionHead<T>,
    num_classes: usize,
    training: bool,
}

impl<T: Float> Fcos<T> {
    /// Build with `num_classes` (matching torchvision's COCO_V1 value of 91).
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        let backbone = resnet50(1)?;
        let fpn = RetinaFpn::new()?;
        let classification_head =
            FcosClassificationHead::new(FPN_OUT_CHANNELS, FCOS_NUM_ANCHORS_PER_LOC, num_classes)?;
        let regression_head = FcosRegressionHead::new(FPN_OUT_CHANNELS, FCOS_NUM_ANCHORS_PER_LOC)?;
        Ok(Self {
            backbone,
            fpn,
            classification_head,
            regression_head,
            num_classes,
            training: false,
        })
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// FPN-level ordering used both for forward and anchor generation.
    const LEVEL_KEYS: [&'static str; 5] = ["p3", "p4", "p5", "p6", "p7"];

    /// End-to-end forward pass. `images` must be `[B, 3, H, W]` (already
    /// preprocessed — ImageNet mean/std normalised + padded to multiple of 32).
    pub fn forward(&self, images: &Tensor<T>) -> FerrotorchResult<Vec<Detections<T>>> {
        if images.ndim() != 4 || images.shape()[1] != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Fcos::forward: expected [B, 3, H, W], got {:?}",
                    images.shape()
                ),
            });
        }
        let batch = images.shape()[0];
        let img_h = images.shape()[2];
        let img_w = images.shape()[3];

        let backbone_features = self.backbone.forward_features(images)?;
        let fpn_features = self.fpn.forward(&backbone_features)?;

        let fm_sizes: [(usize, usize); 5] = [
            {
                let s = fpn_features["p3"].shape();
                (s[2], s[3])
            },
            {
                let s = fpn_features["p4"].shape();
                (s[2], s[3])
            },
            {
                let s = fpn_features["p5"].shape();
                (s[2], s[3])
            },
            {
                let s = fpn_features["p6"].shape();
                (s[2], s[3])
            },
            {
                let s = fpn_features["p7"].shape();
                (s[2], s[3])
            },
        ];
        let anchors_per_level: Vec<Tensor<T>> =
            fcos_anchors_per_level::<T>(&fm_sizes, (img_h, img_w))?;

        // Per-level head outputs.
        let mut cls_per_level: Vec<Tensor<T>> = Vec::with_capacity(5);
        let mut reg_per_level: Vec<Tensor<T>> = Vec::with_capacity(5);
        let mut ctr_per_level: Vec<Tensor<T>> = Vec::with_capacity(5);
        for key in Self::LEVEL_KEYS.iter() {
            let feat = &fpn_features[*key];
            cls_per_level.push(self.classification_head.forward_level(feat)?);
            let (reg, ctr) = self.regression_head.forward_level(feat)?;
            reg_per_level.push(reg);
            ctr_per_level.push(ctr);
        }

        let num_classes = self.num_classes;
        let mut per_image_detections: Vec<Detections<T>> = Vec::with_capacity(batch);

        for b_idx in 0..batch {
            // Per-level postprocess.
            let mut all_boxes: Vec<f64> = Vec::new();
            let mut all_scores: Vec<f64> = Vec::new();
            let mut all_labels: Vec<usize> = Vec::new();

            for lv in 0..5 {
                let cls_t = &cls_per_level[lv];
                let reg_t = &reg_per_level[lv];
                let ctr_t = &ctr_per_level[lv];
                let anc_t = &anchors_per_level[lv];

                let cls_shape = cls_t.shape();
                let hwa = cls_shape[1];
                let k = cls_shape[2];
                debug_assert_eq!(k, num_classes);
                let cls_data = cls_t.data_vec()?;
                let cls_offset = b_idx * hwa * k;

                let reg_data = reg_t.data_vec()?;
                let reg_offset = b_idx * hwa * 4;

                let ctr_data = ctr_t.data_vec()?;
                // ctr layout is [B, HWA, 1] — same HWA as cls.
                let ctr_offset = b_idx * hwa;

                let anc_data = anc_t.data_vec()?;

                // Compute combined score = sqrt(sigmoid(cls) * sigmoid(ctr))
                // and gate by score_thresh.
                // Layout: `flat = anchor_idx * k + class_idx`; torchvision
                // flattens the per-level [HWA, K] tensor and applies topk over
                // ALL (anchor, class) combinations together.
                let mut cand: Vec<(usize, f64)> = Vec::new();
                for anchor_idx in 0..hwa {
                    let ctr_logit = ctr_data[ctr_offset + anchor_idx].to_f64().unwrap_or(0.0);
                    let ctr_sig = 1.0 / (1.0 + (-ctr_logit).exp());
                    for class_idx in 0..k {
                        let cls_logit = cls_data[cls_offset + anchor_idx * k + class_idx]
                            .to_f64()
                            .unwrap_or(0.0);
                        let cls_sig = 1.0 / (1.0 + (-cls_logit).exp());
                        let score = (cls_sig * ctr_sig).sqrt();
                        if score > FCOS_SCORE_THRESH {
                            let flat = anchor_idx * k + class_idx;
                            cand.push((flat, score));
                        }
                    }
                }

                // Per-level top-K (1000), descending sort.
                if cand.len() > FCOS_TOPK_CANDIDATES {
                    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    cand.truncate(FCOS_TOPK_CANDIDATES);
                }

                // Decode boxes for surviving candidates.
                // BoxLinearCoder(normalize_by_size=True):
                //   ctr_x = 0.5 * (a0 + a2);  ctr_y = 0.5 * (a1 + a3)
                //   w = a2 - a0;             h = a3 - a1
                //   pred = [ctr_x - l*w, ctr_y - t*h, ctr_x + r*w, ctr_y + b*h]
                // Cache decoded boxes per anchor_idx — multiple classes per
                // anchor share the same decoded box (just different scores).
                let mut decoded_cache: HashMap<usize, [f64; 4]> = HashMap::new();
                for &(flat, score) in &cand {
                    let anchor_idx = flat / k;
                    let class_idx = flat % k;
                    let dec = if let Some(b) = decoded_cache.get(&anchor_idx) {
                        *b
                    } else {
                        let a0 = anc_data[anchor_idx * 4].to_f64().unwrap_or(0.0);
                        let a1 = anc_data[anchor_idx * 4 + 1].to_f64().unwrap_or(0.0);
                        let a2 = anc_data[anchor_idx * 4 + 2].to_f64().unwrap_or(0.0);
                        let a3 = anc_data[anchor_idx * 4 + 3].to_f64().unwrap_or(0.0);
                        let cx = 0.5 * (a0 + a2);
                        let cy = 0.5 * (a1 + a3);
                        let w = a2 - a0;
                        let h = a3 - a1;
                        let l = reg_data[reg_offset + anchor_idx * 4]
                            .to_f64()
                            .unwrap_or(0.0);
                        let t = reg_data[reg_offset + anchor_idx * 4 + 1]
                            .to_f64()
                            .unwrap_or(0.0);
                        let r = reg_data[reg_offset + anchor_idx * 4 + 2]
                            .to_f64()
                            .unwrap_or(0.0);
                        let bb = reg_data[reg_offset + anchor_idx * 4 + 3]
                            .to_f64()
                            .unwrap_or(0.0);
                        let box_xy = [cx - l * w, cy - t * h, cx + r * w, cy + bb * h];
                        decoded_cache.insert(anchor_idx, box_xy);
                        box_xy
                    };
                    // Clip per-level before concat (matches torchvision).
                    let x1 = dec[0].clamp(0.0, img_w as f64);
                    let y1 = dec[1].clamp(0.0, img_h as f64);
                    let x2 = dec[2].clamp(0.0, img_w as f64);
                    let y2 = dec[3].clamp(0.0, img_h as f64);
                    all_boxes.extend_from_slice(&[x1, y1, x2, y2]);
                    all_scores.push(score);
                    all_labels.push(class_idx);
                }
            }

            if all_scores.is_empty() {
                per_image_detections.push(Detections {
                    boxes: Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0, 4], false)?,
                    scores: Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0usize], false)?,
                    labels: vec![],
                });
                continue;
            }

            // Cross-class batched NMS.
            let n_all = all_scores.len();
            let boxes_f64 =
                Tensor::from_storage(TensorStorage::cpu(all_boxes.clone()), vec![n_all, 4], false)?;
            // Re-clip (no-op for valid boxes).
            let boxes_clipped = clip_boxes_to_image(&boxes_f64, [img_h, img_w])?;
            let scores_f64 =
                Tensor::from_storage(TensorStorage::cpu(all_scores.clone()), vec![n_all], false)?;
            let idxs: Vec<u32> = all_labels.iter().map(|&l| l as u32).collect();
            let keep = batched_nms::<f64>(&boxes_clipped, &scores_f64, &idxs, FCOS_NMS_THRESH)?;

            let post = keep
                .into_iter()
                .take(FCOS_DETECTIONS_PER_IMG)
                .collect::<Vec<_>>();

            let clipped_data = boxes_clipped.data_vec()?;
            let mut out_boxes: Vec<T> = Vec::with_capacity(post.len() * 4);
            let mut out_scores: Vec<T> = Vec::with_capacity(post.len());
            let mut out_labels: Vec<usize> = Vec::with_capacity(post.len());
            for &i in &post {
                out_boxes.push(cast::<f64, T>(clipped_data[i * 4])?);
                out_boxes.push(cast::<f64, T>(clipped_data[i * 4 + 1])?);
                out_boxes.push(cast::<f64, T>(clipped_data[i * 4 + 2])?);
                out_boxes.push(cast::<f64, T>(clipped_data[i * 4 + 3])?);
                out_scores.push(cast::<f64, T>(all_scores[i])?);
                out_labels.push(all_labels[i]);
            }
            let n_out = out_scores.len();
            per_image_detections.push(Detections {
                boxes: Tensor::from_storage(TensorStorage::cpu(out_boxes), vec![n_out, 4], false)?,
                scores: Tensor::from_storage(TensorStorage::cpu(out_scores), vec![n_out], false)?,
                labels: out_labels,
            });
        }

        Ok(per_image_detections)
    }
}

// ---------------------------------------------------------------------------
// Module trait
// ---------------------------------------------------------------------------

impl<T: Float> Module<T> for Fcos<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Module::forward exposes the first-image post-NMS scores as a 1-D
        // `[N_det]` tensor — matching the contract used by FasterRCNN/MaskRCNN/
        // RetinaNet in the #1139 verify harness, so torchvision's
        // `fcos_resnet50_fpn(...)(img)[0]["scores"]` is directly comparable.
        let dets = Fcos::forward(self, input)?;
        if dets.is_empty() || dets[0].scores.shape()[0] == 0 {
            return Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0usize], false);
        }
        Ok(dets[0].scores.clone())
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.backbone.parameters());
        p.extend(self.fpn.parameters());
        p.extend(self.classification_head.parameters());
        p.extend(self.regression_head.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.backbone.parameters_mut());
        p.extend(self.fpn.parameters_mut());
        p.extend(self.classification_head.parameters_mut());
        p.extend(self.regression_head.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        // Key layout matches torchvision's state_dict tree so the pinning
        // script can pass keys through with minimal rewriting:
        //   backbone.<resnet50 names>
        //   fpn.lateral{3,4,5}.{weight,bias} / fpn.output{3,4,5}.{weight,bias}
        //   fpn.p6.{weight,bias} / fpn.p7.{weight,bias}
        //   classification_head.conv.{0,1,3,4,6,7,9,10}.{weight,bias}
        //   classification_head.cls_logits.{weight,bias}
        //   regression_head.conv.{0,1,3,4,6,7,9,10}.{weight,bias}
        //   regression_head.bbox_reg.{weight,bias}
        //   regression_head.bbox_ctrness.{weight,bias}
        let mut out = Vec::new();
        for (n, p) in self.backbone.named_parameters() {
            out.push((format!("backbone.{n}"), p));
        }
        for (n, p) in self.fpn.named_parameters() {
            out.push((format!("fpn.{n}"), p));
        }
        for (n, p) in self.classification_head.named_parameters() {
            out.push((format!("classification_head.{n}"), p));
        }
        for (n, p) in self.regression_head.named_parameters() {
            out.push((format!("regression_head.{n}"), p));
        }
        out
    }

    // BN buffer loader walks the ResNet backbone subtree.
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.backbone]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![("backbone".to_string(), &self.backbone)]
    }

    fn train(&mut self) {
        self.training = true;
        self.backbone.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.backbone.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor
// ---------------------------------------------------------------------------

/// Construct an FCOS detector with ResNet-50 + FPN(P3-P7) backbone for
/// `num_classes` detection classes (COCO default: 91, mirroring torchvision's
/// pretrained `fcos_resnet50_fpn(weights="COCO_V1")` model).
pub fn fcos_resnet50_fpn<T: Float>(num_classes: usize) -> FerrotorchResult<Fcos<T>> {
    Fcos::new(num_classes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{no_grad, randn};

    #[test]
    fn test_fcos_constructs() {
        let m = fcos_resnet50_fpn::<f32>(91).unwrap();
        assert!(m.num_parameters() > 0);
        assert_eq!(m.num_classes(), 91);
    }

    #[test]
    fn test_fcos_param_count_matches_torchvision_plus_bn_affine() {
        // torchvision `FCOS_ResNet50_FPN_Weights.COCO_V1` reports
        // `num_params=32_269_600` in its meta, but that count treats
        // `FrozenBatchNorm2d.weight` / `.bias` as buffers (not
        // Parameters) and also strips the ResNet `.fc` head.
        //
        // ferrotorch's `BatchNorm2d.weight` / `.bias` ARE Parameters and
        // its `resnet50(num_classes=1)` builds a 2049-element fc head.
        // The resulting delta is:
        //   - BN affine (ResNet body):    53_120 extra params
        //   - resnet50 fc head (num_classes=1):  2_049 extra params
        // ⇒ 32_269_600 + 53_120 + 2_049 = 32_324_769.
        //
        // Matching the EXACT ferrotorch count locks in the architecture
        // shape: any future refactor that adds/removes a conv or BN layer
        // will trip this test. The retinanet conformance harness
        // exercises the inference parity path on top of this.
        let m = fcos_resnet50_fpn::<f32>(91).unwrap();
        assert_eq!(
            m.num_parameters(),
            32_324_769,
            "param count drift — see comment for the upstream+BN+fc decomposition"
        );
    }

    #[test]
    fn test_fcos_named_params_prefixes() {
        let m = fcos_resnet50_fpn::<f32>(91).unwrap();
        let names: Vec<String> = m.named_parameters().into_iter().map(|(n, _)| n).collect();
        assert!(names.iter().any(|n| n.starts_with("backbone.")));
        assert!(names.iter().any(|n| n.starts_with("fpn.lateral3.")));
        assert!(names.iter().any(|n| n.starts_with("fpn.lateral5.")));
        assert!(names.iter().any(|n| n.starts_with("fpn.output3.")));
        assert!(names.iter().any(|n| n.starts_with("fpn.p6.")));
        assert!(names.iter().any(|n| n.starts_with("fpn.p7.")));
        // Sequential indexing: Conv at 0,3,6,9 and GroupNorm at 1,4,7,10.
        for idx in [0, 1, 3, 4, 6, 7, 9, 10] {
            let prefix = format!("classification_head.conv.{idx}.");
            assert!(
                names.iter().any(|n| n.starts_with(&prefix)),
                "missing key prefix {prefix}"
            );
        }
        assert!(
            names
                .iter()
                .any(|n| n.starts_with("classification_head.cls_logits."))
        );
        for idx in [0, 1, 3, 4, 6, 7, 9, 10] {
            let prefix = format!("regression_head.conv.{idx}.");
            assert!(
                names.iter().any(|n| n.starts_with(&prefix)),
                "missing key prefix {prefix}"
            );
        }
        assert!(
            names
                .iter()
                .any(|n| n.starts_with("regression_head.bbox_reg."))
        );
        assert!(
            names
                .iter()
                .any(|n| n.starts_with("regression_head.bbox_ctrness."))
        );
    }

    #[test]
    fn test_cls_logits_output_dim() {
        // 1 anchor * 91 classes = 91 channels.
        let head = FcosClassificationHead::<f32>::new(256, 1, 91).unwrap();
        let names: Vec<(String, &Parameter<f32>)> = head.named_parameters();
        let cls_w = names
            .iter()
            .find(|(n, _)| n == "cls_logits.weight")
            .expect("cls_logits.weight missing");
        assert_eq!(cls_w.1.shape(), &[91, 256, 3, 3]);
    }

    #[test]
    fn test_bbox_reg_and_ctrness_output_dims() {
        // 1 anchor * 4 = 4 channels; 1 anchor * 1 = 1 channel.
        let head = FcosRegressionHead::<f32>::new(256, 1).unwrap();
        let names: Vec<(String, &Parameter<f32>)> = head.named_parameters();
        let reg_w = names
            .iter()
            .find(|(n, _)| n == "bbox_reg.weight")
            .expect("bbox_reg.weight missing");
        assert_eq!(reg_w.1.shape(), &[4, 256, 3, 3]);
        let ctr_w = names
            .iter()
            .find(|(n, _)| n == "bbox_ctrness.weight")
            .expect("bbox_ctrness.weight missing");
        assert_eq!(ctr_w.1.shape(), &[1, 256, 3, 3]);
    }

    #[test]
    fn test_cls_head_forward_layout() {
        // Cls head on a 4×4 feature map with 1 anchor/loc → output
        // [B, H*W*A=16, K=91].
        let head = FcosClassificationHead::<f32>::new(256, 1, 91).unwrap();
        let feat = no_grad(|| randn(&[1, 256, 4, 4]).unwrap());
        let out = no_grad(|| head.forward_level(&feat).unwrap());
        assert_eq!(out.shape(), &[1, 16, 91]);
    }

    #[test]
    fn test_reg_head_forward_layout() {
        // 4×4 feature map, 1 anchor/loc → 16 (anchor, loc) pairs.
        let head = FcosRegressionHead::<f32>::new(256, 1).unwrap();
        let feat = no_grad(|| randn(&[1, 256, 4, 4]).unwrap());
        let (reg, ctr) = no_grad(|| head.forward_level(&feat).unwrap());
        assert_eq!(reg.shape(), &[1, 16, 4]);
        assert_eq!(ctr.shape(), &[1, 16, 1]);
    }

    #[test]
    fn test_reg_head_outputs_non_negative_after_relu() {
        // FCOS gates `bbox_reg` through ReLU on every forward — predictions
        // are always non-negative. This is the live formula (not exp).
        let head = FcosRegressionHead::<f32>::new(256, 1).unwrap();
        let feat = no_grad(|| randn(&[2, 256, 3, 3]).unwrap());
        let (reg, _ctr) = no_grad(|| head.forward_level(&feat).unwrap());
        let data = reg.data_vec().unwrap();
        for v in data {
            assert!(
                v >= 0.0,
                "bbox_reg output must be non-negative after ReLU, got {v}"
            );
        }
    }

    #[test]
    fn test_fcos_anchor_box_at_origin_level_0() {
        // Level 0 stride 8, single-cell feature map → anchor centred at
        // (0, 0). cell size = 8, half = round(8/2) = 4 → [-4,-4,4,4].
        let lvls = fcos_anchors_per_level::<f32>(&[(1, 1); 5], (32, 32)).unwrap();
        assert_eq!(lvls.len(), 5);
        let l0 = lvls[0].data_vec().unwrap();
        // Single anchor: 4 floats.
        assert_eq!(l0.len(), 4);
        assert!((l0[0] - -4.0).abs() < 1e-4, "{l0:?}");
        assert!((l0[1] - -4.0).abs() < 1e-4, "{l0:?}");
        assert!((l0[2] - 4.0).abs() < 1e-4, "{l0:?}");
        assert!((l0[3] - 4.0).abs() < 1e-4, "{l0:?}");
    }

    #[test]
    fn test_fcos_anchor_shifts_match_torchvision_convention() {
        // For a 2×2 grid at stride 8 (image 16×16), anchors must shift by
        // (col*8, row*8). Cell anchor base is [-4,-4,4,4].
        // Cells row-major (fy outer, fx inner):
        //   (0,0): [-4,-4,4,4]
        //   (0,1): [4,-4,12,4]   (cx=8, cy=0)
        //   (1,0): [-4,4,4,12]   (cx=0, cy=8)
        //   (1,1): [4,4,12,12]   (cx=8, cy=8)
        let lvls =
            fcos_anchors_per_level::<f32>(&[(2, 2), (1, 1), (1, 1), (1, 1), (1, 1)], (16, 16))
                .unwrap();
        let l0 = lvls[0].data_vec().unwrap();
        assert_eq!(l0.len(), 16);
        let expected: [f32; 16] = [
            -4.0, -4.0, 4.0, 4.0, 4.0, -4.0, 12.0, 4.0, -4.0, 4.0, 4.0, 12.0, 4.0, 4.0, 12.0, 12.0,
        ];
        for i in 0..16 {
            assert!((l0[i] - expected[i]).abs() < 1e-4, "i={i} got {l0:?}");
        }
    }

    #[test]
    fn test_fcos_forward_small_image_returns_per_image_detections() {
        let m = fcos_resnet50_fpn::<f32>(91).unwrap();
        // Use 128×128 — large enough to leave a non-empty P7 (128/128=1 → 1×1).
        let img = no_grad(|| randn(&[1, 3, 128, 128]).unwrap());
        let dets = no_grad(|| Fcos::forward(&m, &img).unwrap());
        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert_eq!(d.boxes.shape().len(), 2);
        assert_eq!(d.boxes.shape()[1], 4);
        assert_eq!(d.scores.shape().len(), 1);
        assert_eq!(d.scores.shape()[0], d.boxes.shape()[0]);
        assert_eq!(d.labels.len(), d.boxes.shape()[0]);
    }

    #[test]
    fn test_fcos_train_eval_toggle() {
        let mut m = fcos_resnet50_fpn::<f32>(91).unwrap();
        assert!(!m.is_training());
        m.train();
        assert!(m.is_training());
        m.eval();
        assert!(!m.is_training());
    }
}
