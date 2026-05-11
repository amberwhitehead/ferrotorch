//! Keypoint R-CNN with ResNet-50 FPN backbone.
//!
//! Mirrors `torchvision.models.detection.keypointrcnn_resnet50_fpn`.
//!
//! ## Architecture
//!
//! ```text
//! image [B, 3, H, W]
//!   └─ ResNet-50 backbone → {layer1..layer4}   (C2–C5 feature maps)
//!         └─ FPN            → {p2..p6}          (256-ch multi-scale)
//!               └─ RPN      → proposals [N, 4]  (xyxy image coords)
//!                     ├─ ROI Align (7×7)  → [N, 256, 7, 7]  (detection)
//!                     │       └─ TwoMlpHead (num_classes=2: bg + person)
//!                     └─ ROI Align (14×14) → [N, 256, 14, 14]  (keypoints)
//!                             └─ KeypointHead (8 × Conv(3×3) + ReLU)
//!                                   └─ KeypointPredictor
//!                                       (deconv → [N, 17, 28, 28]
//!                                        → bilinear 2× → [N, 17, 56, 56])
//!                                         └─ heatmaps_to_keypoints
//!                                               → [N_det, 17, 3] image-space (x, y, 1)
//! ```
//!
//! ## Reference
//! He et al., "Mask R-CNN", ICCV 2017 (the keypoint variant).
//! torchvision 0.21.x `keypointrcnn_resnet50_fpn(weights=None)`.

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Conv2d, ConvTranspose2d, InterpolateMode, interpolate};

use crate::models::detection::faster_rcnn::{FasterRcnn, fasterrcnn_resnet50_fpn};
use crate::ops::roi_align_with_aligned;

/// Number of COCO person keypoints predicted by the pretrained model
/// (nose, eyes ×2, ears ×2, shoulders ×2, elbows ×2, wrists ×2, hips ×2,
/// knees ×2, ankles ×2).
pub const KEYPOINT_RCNN_NUM_KEYPOINTS: usize = 17;

/// Number of classes (background + person) for the box predictor in the
/// COCO pretrained checkpoint.
pub const KEYPOINT_RCNN_NUM_CLASSES: usize = 2;

// ---------------------------------------------------------------------------
// KeypointHead — 8 conv layers, all 3×3 pad=1, 256→512 then 512→512×7.
// ---------------------------------------------------------------------------

/// Eight-layer FCN head applied to keypoint ROI features.
///
/// Mirrors `torchvision.models.detection.keypoint_rcnn.KeypointRCNNHeads`
/// with the default config `(512, 512, 512, 512, 512, 512, 512, 512)`.
///
/// torchvision stores these in `nn.Sequential` with interleaved ReLUs, so
/// the convs live at even indices `0, 2, 4, 6, 8, 10, 12, 14` — we mirror
/// that exact key layout (`conv0`, `conv2`, ..., `conv14`) when emitting
/// `named_parameters`.
pub struct KeypointHead<T: Float> {
    conv0: Conv2d<T>,
    conv2: Conv2d<T>,
    conv4: Conv2d<T>,
    conv6: Conv2d<T>,
    conv8: Conv2d<T>,
    conv10: Conv2d<T>,
    conv12: Conv2d<T>,
    conv14: Conv2d<T>,
}

impl<T: Float> KeypointHead<T> {
    /// Create a new `KeypointHead`.
    ///
    /// `in_channels` — number of input feature channels (256 from FPN).
    pub fn new(in_channels: usize) -> FerrotorchResult<Self> {
        let conv0 = Conv2d::new(in_channels, 512, (3, 3), (1, 1), (1, 1), true)?;
        let conv2 = Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), true)?;
        let conv4 = Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), true)?;
        let conv6 = Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), true)?;
        let conv8 = Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), true)?;
        let conv10 = Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), true)?;
        let conv12 = Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), true)?;
        let conv14 = Conv2d::new(512, 512, (3, 3), (1, 1), (1, 1), true)?;
        Ok(Self {
            conv0,
            conv2,
            conv4,
            conv6,
            conv8,
            conv10,
            conv12,
            conv14,
        })
    }

    /// Forward on `[N, in_channels, H, W]` → `[N, 512, H, W]`.
    pub fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = relu(&self.conv0.forward(x)?)?;
        let x = relu(&self.conv2.forward(&x)?)?;
        let x = relu(&self.conv4.forward(&x)?)?;
        let x = relu(&self.conv6.forward(&x)?)?;
        let x = relu(&self.conv8.forward(&x)?)?;
        let x = relu(&self.conv10.forward(&x)?)?;
        let x = relu(&self.conv12.forward(&x)?)?;
        relu(&self.conv14.forward(&x)?)
    }

    /// Trainable parameters.
    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv0.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.conv4.parameters());
        p.extend(self.conv6.parameters());
        p.extend(self.conv8.parameters());
        p.extend(self.conv10.parameters());
        p.extend(self.conv12.parameters());
        p.extend(self.conv14.parameters());
        p
    }

    /// Mutable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv0.parameters_mut());
        p.extend(self.conv2.parameters_mut());
        p.extend(self.conv4.parameters_mut());
        p.extend(self.conv6.parameters_mut());
        p.extend(self.conv8.parameters_mut());
        p.extend(self.conv10.parameters_mut());
        p.extend(self.conv12.parameters_mut());
        p.extend(self.conv14.parameters_mut());
        p
    }

    /// Named parameters (`conv{0,2,4,6,8,10,12,14}.{weight,bias}`).
    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (i, c) in [
            (0usize, &self.conv0),
            (2, &self.conv2),
            (4, &self.conv4),
            (6, &self.conv6),
            (8, &self.conv8),
            (10, &self.conv10),
            (12, &self.conv12),
            (14, &self.conv14),
        ] {
            for (n, p) in c.named_parameters() {
                out.push((format!("conv{i}.{n}"), p));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// KeypointPredictor — single ConvTranspose2d(512 → 17, k=4, s=2, p=1).
// ---------------------------------------------------------------------------

/// Keypoint predictor: a single transposed convolution that doubles spatial
/// resolution (14×14 → 28×28), followed by a 2× bilinear upsample to 56×56,
/// projecting to 17 keypoint heatmap channels.
///
/// Mirrors `torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor`,
/// whose `state_dict` exposes only `kps_score_lowres.{weight,bias}` and whose
/// `forward` does `F.interpolate(x, scale_factor=2, mode='bilinear',
/// align_corners=False)` after the deconv. The post-deconv 2× upsample is
/// **parameter-free** (no state_dict keys), matching torchvision exactly.
pub struct KeypointPredictor<T: Float> {
    /// `ConvTranspose2d(in_channels → num_keypoints, k=4, s=2, p=1)`.
    kps_score_lowres: ConvTranspose2d<T>,
}

impl<T: Float> KeypointPredictor<T> {
    /// Create a new `KeypointPredictor`.
    ///
    /// `in_channels` is the KeypointHead output channels (512).
    /// `num_keypoints` is typically 17 for COCO person keypoints.
    pub fn new(in_channels: usize, num_keypoints: usize) -> FerrotorchResult<Self> {
        let kps_score_lowres = ConvTranspose2d::new(
            in_channels,
            num_keypoints,
            (4, 4),
            (2, 2),
            (1, 1),
            (0, 0),
            true,
        )?;
        Ok(Self { kps_score_lowres })
    }

    /// Forward on `[N, in_channels, 14, 14]` → `[N, num_keypoints, 56, 56]`.
    ///
    /// Steps (matching torchvision exactly):
    ///   1. `ConvTranspose2d(k=4, s=2, p=1)`     14×14 → 28×28
    ///   2. `F.interpolate(scale_factor=2,        28×28 → 56×56
    ///       mode='bilinear', align_corners=False)`
    pub fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let low = self.kps_score_lowres.forward(x)?;
        let low_shape = low.shape();
        let h2 = low_shape[2] * 2;
        let w2 = low_shape[3] * 2;
        interpolate(
            &low,
            Some([h2, w2]),
            None,
            InterpolateMode::Bilinear,
            false,
        )
    }

    /// Trainable parameters.
    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        self.kps_score_lowres.parameters()
    }

    /// Mutable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.kps_score_lowres.parameters_mut()
    }

    /// Named parameters (`kps_score_lowres.{weight,bias}`).
    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.kps_score_lowres
            .named_parameters()
            .into_iter()
            .map(|(n, p)| (format!("kps_score_lowres.{n}"), p))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Per-image detection result including keypoints.
// ---------------------------------------------------------------------------

/// Per-image detection output from Keypoint R-CNN.
///
/// Mirrors `torchvision.models.detection.KeypointRCNN`'s output dictionary
/// after `keypointrcnn_inference` + `GeneralizedRCNNTransform.postprocess`:
/// `keypoints` is `[N_det, 17, 3]` in image-space pixel coords with the
/// third column always 1.0 (visibility flag), and `keypoint_scores` is the
/// per-keypoint raw heatmap logit at the argmax location, `[N_det, 17]`.
#[derive(Debug, Clone)]
pub struct KeypointDetections<T: Float> {
    /// Predicted boxes `[N_det, 4]` in xyxy pixel coords.
    pub boxes: Tensor<T>,
    /// Per-detection score `[N_det]` (softmax probability of the predicted
    /// class — always the `person` class for the COCO pretrained model).
    pub scores: Tensor<T>,
    /// Predicted class label `[N_det]` (always `>= 1` — background is dropped).
    pub labels: Vec<usize>,
    /// Per-detection keypoint coordinates `[N_det, 17, 3]`.
    ///
    /// The 3 columns are `(x_image, y_image, 1.0)` matching torchvision's
    /// `heatmaps_to_keypoints` post-permute layout.
    pub keypoints: Tensor<T>,
    /// Per-detection per-keypoint raw heatmap logit at the argmax `[N_det, 17]`.
    pub keypoint_scores: Tensor<T>,
}

// ---------------------------------------------------------------------------
// KeypointRcnn
// ---------------------------------------------------------------------------

/// Keypoint R-CNN with ResNet-50 FPN backbone.
///
/// Extends Faster R-CNN by adding a parallel keypoint branch that operates
/// on 14×14 ROI-aligned features and outputs per-keypoint heatmaps decoded
/// to image-space pixel coordinates. The pretrained COCO checkpoint uses
/// `num_classes=2` (background + person).
///
/// **Reuses Sprint C.1 components**: backbone (ResNet-50), FPN, RPN, ROI
/// Align, and the `TwoMlpHead` detection head from `FasterRcnn`. Only the
/// keypoint-specific layers (`KeypointHead` + `KeypointPredictor`) are new.
pub struct KeypointRcnn<T: Float> {
    /// Faster R-CNN sub-model (backbone + FPN + RPN + 2-class detection head).
    ///
    /// All Sprint C.1 components are owned here; no duplication.
    faster_rcnn: FasterRcnn<T>,
    /// 8-layer FCN keypoint head.
    keypoint_head: KeypointHead<T>,
    /// Single-deconv keypoint predictor.
    keypoint_predictor: KeypointPredictor<T>,
    num_classes: usize,
    num_keypoints: usize,
    /// ROI Align spatial size for the keypoint branch (14×14).
    keypoint_roi_size: usize,
    /// Spatial scales per FPN level p2..p6 (1/stride).
    roi_spatial_scales: Vec<f64>,
    training: bool,
}

impl<T: Float> KeypointRcnn<T> {
    /// FPN level names used by the **keypoint** ROI pool, in order (p2..p5).
    ///
    /// torchvision's `MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"],
    /// output_size=14, sampling_ratio=2)` for the keypoint head uses only
    /// the four finer FPN levels — p6 is excluded. The corresponding
    /// LevelMapper also clamps to `[k_min=2, k_max=5]`. (#1145)
    const FPN_LEVEL_KEYS: [&'static str; 4] = ["p2", "p3", "p4", "p5"];

    /// Spatial scales for the keypoint ROI levels p2..p5 (1/stride).
    const FPN_SPATIAL_SCALES: [f64; 4] = [1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0];

    /// Create a new Keypoint R-CNN from scratch.
    ///
    /// `num_classes` includes background at index 0 (default COCO: 2 —
    /// background + person). `num_keypoints` is the number of keypoint
    /// heatmap channels (default COCO: 17).
    pub fn new(num_classes: usize, num_keypoints: usize) -> FerrotorchResult<Self> {
        let faster_rcnn = fasterrcnn_resnet50_fpn::<T>(num_classes)?;
        let keypoint_head = KeypointHead::new(256)?;
        let keypoint_predictor = KeypointPredictor::new(512, num_keypoints)?;

        Ok(Self {
            faster_rcnn,
            keypoint_head,
            keypoint_predictor,
            num_classes,
            num_keypoints,
            keypoint_roi_size: 14,
            roi_spatial_scales: Self::FPN_SPATIAL_SCALES.to_vec(),
            training: false,
        })
    }

    /// End-to-end forward pass.
    ///
    /// `images` — `[B, 3, H, W]` float tensor (RGB, any scale).
    ///
    /// Returns a `Vec<KeypointDetections<T>>` of length `B`.
    pub fn forward(&self, images: &Tensor<T>) -> FerrotorchResult<Vec<KeypointDetections<T>>> {
        if images.ndim() != 4 || images.shape()[1] != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "KeypointRcnn::forward: expected [B, 3, H, W], got {:?}",
                    images.shape()
                ),
            });
        }
        let batch = images.shape()[0];

        // ---- Reuse FasterRcnn detection pipeline ----
        let detections = self.faster_rcnn.forward(images)?;
        let backbone_features = self.faster_rcnn.forward_backbone(images)?;
        let fpn_features = self.faster_rcnn.forward_fpn(&backbone_features)?;

        let mut results: Vec<KeypointDetections<T>> = Vec::with_capacity(batch);

        for (b_idx, det) in detections.into_iter().enumerate() {
            let n_proposals = det.boxes.shape()[0];

            if n_proposals == 0 {
                // No detections → empty keypoint tensors.
                let empty_kp = Tensor::from_storage(
                    TensorStorage::cpu(vec![]),
                    vec![0, self.num_keypoints, 3],
                    false,
                )?;
                let empty_kp_scores = Tensor::from_storage(
                    TensorStorage::cpu(vec![]),
                    vec![0, self.num_keypoints],
                    false,
                )?;
                results.push(KeypointDetections {
                    boxes: det.boxes,
                    scores: det.scores,
                    labels: det.labels,
                    keypoints: empty_kp,
                    keypoint_scores: empty_kp_scores,
                });
                continue;
            }

            // ---- Keypoint ROI Align (14×14) ----
            //
            // torchvision's keypoint `MultiScaleRoIAlign` uses 4 levels
            // (`featmap_names=["0","1","2","3"]` → p2..p5), so the
            // `LevelMapper` clamps to `[k_min=2, k_max=5]`. Any box that
            // would map to p6 by the FPN heuristic instead reuses p5. (#1145)
            let roi_levels = assign_fpn_levels_keypoint(&det.boxes, 4.0, 224.0, 2, 5)?;

            let mut kp_roi_features_all: Vec<Option<Vec<T>>> = vec![None; n_proposals];

            for (level_idx, &level_key) in Self::FPN_LEVEL_KEYS.iter().enumerate() {
                let fpn_level = level_idx + 2;

                let feat_b = &fpn_features[level_key];
                let feat_single = slice_batch_item_kp(feat_b, b_idx)?;

                let indices: Vec<usize> = roi_levels
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &lv)| if lv == fpn_level { Some(i) } else { None })
                    .collect();

                if indices.is_empty() {
                    continue;
                }

                let scale = self.roi_spatial_scales[level_idx];
                let zero: T = cast(0.0f64)?;
                let prop_data = det.boxes.data_vec()?;

                let mut roi_boxes: Vec<T> = Vec::with_capacity(indices.len() * 5);
                for &i in &indices {
                    roi_boxes.push(zero);
                    roi_boxes.push(prop_data[i * 4]);
                    roi_boxes.push(prop_data[i * 4 + 1]);
                    roi_boxes.push(prop_data[i * 4 + 2]);
                    roi_boxes.push(prop_data[i * 4 + 3]);
                }

                let k = indices.len();
                let boxes_t =
                    Tensor::from_storage(TensorStorage::cpu(roi_boxes), vec![k, 5], false)?;

                // torchvision's `MultiScaleRoIAlign` (keypoint head) uses
                // `aligned=false` (legacy) — match pretrained-weight
                // semantics. (#1145)
                let roi_out = roi_align_with_aligned(
                    &feat_single,
                    &boxes_t,
                    (self.keypoint_roi_size, self.keypoint_roi_size),
                    scale,
                    2,
                    false,
                )?;

                let channels = feat_single.shape()[1];
                let per_roi_size = channels * self.keypoint_roi_size * self.keypoint_roi_size;
                let roi_data = roi_out.data_vec()?;

                for (local_idx, &global_idx) in indices.iter().enumerate() {
                    let start = local_idx * per_roi_size;
                    let row: Vec<T> = roi_data[start..start + per_roi_size].to_vec();
                    kp_roi_features_all[global_idx] = Some(row);
                }
            }

            let channels = 256usize;
            let p = self.keypoint_roi_size;
            let per_roi = channels * p * p;
            let mut stacked: Vec<T> = Vec::with_capacity(n_proposals * per_roi);
            for slot in &kp_roi_features_all {
                if let Some(row) = slot {
                    stacked.extend_from_slice(row);
                } else {
                    let zero: T = cast(0.0f64)?;
                    stacked.extend(vec![zero; per_roi]);
                }
            }

            let kp_roi_tensor = Tensor::from_storage(
                TensorStorage::cpu(stacked),
                vec![n_proposals, channels, p, p],
                false,
            )?;

            // ---- Keypoint head + predictor ----
            let kp_features = self.keypoint_head.forward(&kp_roi_tensor)?;
            // Shape: [N_det, 512, 14, 14].
            let kp_heatmaps = self.keypoint_predictor.forward(&kp_features)?;
            // Shape: [N_det, num_keypoints, 28, 28].

            // ---- heatmaps_to_keypoints: per-ROI bicubic upsample → argmax → image coords ----
            let (keypoints, keypoint_scores) = heatmaps_to_keypoints(
                &kp_heatmaps,
                &det.boxes,
                self.num_keypoints,
            )?;

            results.push(KeypointDetections {
                boxes: det.boxes,
                scores: det.scores,
                labels: det.labels,
                keypoints,
                keypoint_scores,
            });
        }

        Ok(results)
    }

    /// Total trainable parameter count.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Number of detection classes including background at index 0.
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Number of keypoint heatmap channels.
    pub fn num_keypoints(&self) -> usize {
        self.num_keypoints
    }
}

// ---------------------------------------------------------------------------
// Module trait implementation
// ---------------------------------------------------------------------------

impl<T: Float> Module<T> for KeypointRcnn<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Module::forward is required for the registry; primary API is
        // `KeypointRcnn::forward` which returns `Vec<KeypointDetections<T>>`.
        //
        // Convention (matches #1139 verification harness for the
        // retinanet/fcos/fasterrcnn detection branch): expose the first-image
        // post-NMS, post-top-K per-detection scores as a 1-D `[N_det]`
        // tensor, matching `torchvision`'s `model(img)[0]["scores"]`.
        // Keypoints + keypoint_scores are reachable via the inherent
        // `KeypointRcnn::forward` API.
        let dets = KeypointRcnn::forward(self, input)?;
        if dets.is_empty() || dets[0].scores.shape()[0] == 0 {
            return Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0usize], false);
        }
        Ok(dets[0].scores.clone())
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.faster_rcnn.parameters());
        p.extend(self.keypoint_head.parameters());
        p.extend(self.keypoint_predictor.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.faster_rcnn.parameters_mut());
        p.extend(self.keypoint_head.parameters_mut());
        p.extend(self.keypoint_predictor.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.faster_rcnn.named_parameters() {
            out.push((format!("faster_rcnn.{n}"), p));
        }
        for (n, p) in self.keypoint_head.named_parameters() {
            out.push((format!("keypoint_head.{n}"), p));
        }
        for (n, p) in self.keypoint_predictor.named_parameters() {
            out.push((format!("keypoint_predictor.{n}"), p));
        }
        out
    }

    // Phase 4 (#995): expose `faster_rcnn` so the BN-buffer loader walks
    // into the wrapped ResNet backbone. `KeypointHead` / `KeypointPredictor`
    // are inherent-method helpers (no BN buffers, no `Module<T>` impl), so
    // we project their inner Conv2d / ConvTranspose2d directly to let
    // `Module<T>::children` enumerate everything for diagnostics.
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![
            &self.faster_rcnn,
            &self.keypoint_head.conv0,
            &self.keypoint_head.conv2,
            &self.keypoint_head.conv4,
            &self.keypoint_head.conv6,
            &self.keypoint_head.conv8,
            &self.keypoint_head.conv10,
            &self.keypoint_head.conv12,
            &self.keypoint_head.conv14,
            &self.keypoint_predictor.kps_score_lowres,
        ]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("faster_rcnn".to_string(), &self.faster_rcnn),
            ("keypoint_head.conv0".to_string(), &self.keypoint_head.conv0),
            ("keypoint_head.conv2".to_string(), &self.keypoint_head.conv2),
            ("keypoint_head.conv4".to_string(), &self.keypoint_head.conv4),
            ("keypoint_head.conv6".to_string(), &self.keypoint_head.conv6),
            ("keypoint_head.conv8".to_string(), &self.keypoint_head.conv8),
            (
                "keypoint_head.conv10".to_string(),
                &self.keypoint_head.conv10,
            ),
            (
                "keypoint_head.conv12".to_string(),
                &self.keypoint_head.conv12,
            ),
            (
                "keypoint_head.conv14".to_string(),
                &self.keypoint_head.conv14,
            ),
            (
                "keypoint_predictor.kps_score_lowres".to_string(),
                &self.keypoint_predictor.kps_score_lowres,
            ),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.faster_rcnn.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.faster_rcnn.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor
// ---------------------------------------------------------------------------

/// Construct a Keypoint R-CNN with ResNet-50 FPN backbone.
///
/// Uses `num_classes=2` (background + person) and `num_keypoints=17` —
/// the COCO defaults matching `torchvision.models.detection.keypointrcnn_resnet50_fpn`.
pub fn keypointrcnn_resnet50_fpn<T: Float>() -> FerrotorchResult<KeypointRcnn<T>> {
    KeypointRcnn::new(KEYPOINT_RCNN_NUM_CLASSES, KEYPOINT_RCNN_NUM_KEYPOINTS)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// FPN level assignment for keypoint ROIs.
///
/// Same formula as Faster R-CNN's `assign_fpn_levels` — mirrors torchvision's
/// `LevelMapper` (including the `eps = 1e-6` numerical nudge). (#1145)
fn assign_fpn_levels_keypoint<T: Float>(
    proposals: &Tensor<T>,
    k0: f64,
    canonical_size: f64,
    min_level: usize,
    max_level: usize,
) -> FerrotorchResult<Vec<usize>> {
    let data = proposals.data_vec()?;
    let n = proposals.shape()[0];
    let mut levels = Vec::with_capacity(n);
    const LEVEL_MAPPER_EPS: f64 = 1e-6;
    for i in 0..n {
        let x1 = data[i * 4].to_f64().unwrap_or(0.0);
        let y1 = data[i * 4 + 1].to_f64().unwrap_or(0.0);
        let x2 = data[i * 4 + 2].to_f64().unwrap_or(0.0);
        let y2 = data[i * 4 + 3].to_f64().unwrap_or(0.0);
        let area = ((x2 - x1) * (y2 - y1)).max(1.0);
        let level = (k0 + (area.sqrt() / canonical_size).log2() + LEVEL_MAPPER_EPS)
            .floor()
            .clamp(min_level as f64, max_level as f64) as usize;
        levels.push(level);
    }
    Ok(levels)
}

/// Extract item `b` from a `[B, C, H, W]` tensor → `[1, C, H, W]`.
fn slice_batch_item_kp<T: Float>(t: &Tensor<T>, b: usize) -> FerrotorchResult<Tensor<T>> {
    let shape = t.shape();
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];
    let stride = c * h * w;
    let data = t.data_vec()?;
    let slice = data[b * stride..(b + 1) * stride].to_vec();
    Tensor::from_storage(TensorStorage::cpu(slice), vec![1, c, h, w], false)
}

/// Decode keypoint heatmaps to image-space coordinates per detection.
///
/// Mirrors `torchvision.models.detection.roi_heads.heatmaps_to_keypoints`
/// (non-tracing path):
///
/// 1. For each ROI, bicubic-upsample its `[num_kp, 28, 28]` heatmap to
///    `(roi_h_ceil, roi_w_ceil)` matching the (post-clamp-to-1) box size.
/// 2. Argmax per keypoint over the upsampled grid → integer `(x_int, y_int)`.
/// 3. Continuous coords (Heckbert 1990 `c = d + 0.5` convention):
///    `x = (x_int + 0.5) * (width / roi_w_ceil) + box_x1`,
///    `y = (y_int + 0.5) * (height / roi_h_ceil) + box_y1`.
/// 4. Per-keypoint score = raw heatmap logit at `(x_int, y_int)`.
///
/// Returns `(keypoints, end_scores)`:
///   keypoints  : `[N, num_keypoints, 3]` with columns `(x, y, 1.0)`.
///   end_scores : `[N, num_keypoints]`.
pub fn heatmaps_to_keypoints<T: Float>(
    heatmaps: &Tensor<T>,
    rois: &Tensor<T>,
    num_keypoints: usize,
) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    let shape = heatmaps.shape();
    if shape.len() != 4 || shape[1] != num_keypoints {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "heatmaps_to_keypoints: expected [N, {num_keypoints}, H, W], got {shape:?}"
            ),
        });
    }
    let n = shape[0];
    let map_h = shape[2];
    let map_w = shape[3];
    let map_data = heatmaps.data_vec()?;

    let roi_shape = rois.shape();
    if roi_shape.len() != 2 || roi_shape[1] != 4 || roi_shape[0] != n {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "heatmaps_to_keypoints: expected rois [{n}, 4], got {roi_shape:?}"
            ),
        });
    }
    let roi_data = rois.data_vec()?;

    let mut keypoints_out: Vec<T> = vec![cast(0.0f64)?; n * num_keypoints * 3];
    let mut scores_out: Vec<T> = vec![cast(0.0f64)?; n * num_keypoints];
    let one: T = cast(1.0f64)?;

    for i in 0..n {
        let x1 = roi_data[i * 4].to_f64().unwrap_or(0.0);
        let y1 = roi_data[i * 4 + 1].to_f64().unwrap_or(0.0);
        let x2 = roi_data[i * 4 + 2].to_f64().unwrap_or(0.0);
        let y2 = roi_data[i * 4 + 3].to_f64().unwrap_or(0.0);
        let width = (x2 - x1).max(1.0);
        let height = (y2 - y1).max(1.0);
        let width_ceil = width.ceil();
        let height_ceil = height.ceil();
        let roi_h = width_ceil.max(1.0) as usize; // guard against pathological zero
        let roi_w_us = width_ceil.max(1.0) as usize;
        let roi_h_us = height_ceil.max(1.0) as usize;
        // Use Heckbert correction factor: width / ceil(width).
        let width_correction = width / (width_ceil.max(1.0));
        let height_correction = height / (height_ceil.max(1.0));
        let _ = roi_h; // silence unused (kept for clarity above)

        // Extract this ROI's [num_kp, map_h, map_w] block.
        let per_n = num_keypoints * map_h * map_w;
        let start = i * per_n;
        let slice: Vec<T> = map_data[start..start + per_n].to_vec();
        let in_tensor = Tensor::from_storage(
            TensorStorage::cpu(slice),
            vec![1, num_keypoints, map_h, map_w],
            false,
        )?;

        // Bicubic upsample to (roi_h_us, roi_w_us). Matches torchvision's
        // `F.interpolate(..., mode='bicubic', align_corners=False)`.
        let upsampled = interpolate(
            &in_tensor,
            Some([roi_h_us, roi_w_us]),
            None,
            InterpolateMode::Bicubic,
            false,
        )?;
        // Shape: [1, num_keypoints, roi_h_us, roi_w_us].
        let up_data = upsampled.data_vec()?;

        // Per-keypoint argmax over the upsampled map.
        let per_kp = roi_h_us * roi_w_us;
        for k in 0..num_keypoints {
            let kp_start = k * per_kp;
            let mut best_idx = 0usize;
            let mut best_val = up_data[kp_start];
            for j in 1..per_kp {
                let v = up_data[kp_start + j];
                if v.partial_cmp(&best_val) == Some(std::cmp::Ordering::Greater) {
                    best_val = v;
                    best_idx = j;
                }
            }
            let x_int = best_idx % roi_w_us;
            let y_int = best_idx / roi_w_us;
            let x_cont = (x_int as f64 + 0.5) * width_correction + x1;
            let y_cont = (y_int as f64 + 0.5) * height_correction + y1;

            keypoints_out[(i * num_keypoints + k) * 3] = cast(x_cont)?;
            keypoints_out[(i * num_keypoints + k) * 3 + 1] = cast(y_cont)?;
            keypoints_out[(i * num_keypoints + k) * 3 + 2] = one;
            scores_out[i * num_keypoints + k] = best_val;
        }
    }

    let keypoints = Tensor::from_storage(
        TensorStorage::cpu(keypoints_out),
        vec![n, num_keypoints, 3],
        false,
    )?;
    let end_scores = Tensor::from_storage(
        TensorStorage::cpu(scores_out),
        vec![n, num_keypoints],
        false,
    )?;
    Ok((keypoints, end_scores))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::no_grad;

    fn make_model() -> KeypointRcnn<f32> {
        keypointrcnn_resnet50_fpn::<f32>().unwrap()
    }

    #[test]
    fn test_keypoint_rcnn_constructs() {
        let model = make_model();
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_keypoint_rcnn_param_count_ballpark() {
        // ResNet-50 (~25.5M) + FPN (~3.3M) + RPN (~1.2M)
        // + TwoMlpHead for 2 classes (~14M)
        // + keypoint head 8×conv (~20M) + predictor (~140K). Total ~64M.
        // Accepted range: 55M–75M for the default config.
        let model = make_model();
        let np = model.num_parameters();
        assert!(np > 55_000_000, "param count too low: {np}");
        assert!(np < 75_000_000, "param count too high: {np}");
    }

    #[test]
    fn test_keypoint_rcnn_named_params_prefixes() {
        let model = make_model();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.iter().any(|n| n.starts_with("faster_rcnn.")));
        assert!(names.iter().any(|n| n.starts_with("keypoint_head.")));
        assert!(names.iter().any(|n| n.starts_with("keypoint_predictor.")));
        // Specific torchvision-key parity for the keypoint subtrees.
        assert!(
            names
                .iter()
                .any(|n| n == "keypoint_head.conv0.weight"),
            "missing keypoint_head.conv0.weight in {names:?}",
        );
        assert!(
            names
                .iter()
                .any(|n| n == "keypoint_head.conv14.weight"),
            "missing keypoint_head.conv14.weight",
        );
        assert!(
            names
                .iter()
                .any(|n| n == "keypoint_predictor.kps_score_lowres.weight"),
            "missing keypoint_predictor.kps_score_lowres.weight",
        );
    }

    #[test]
    fn test_keypoint_head_shapes() {
        let head = KeypointHead::<f32>::new(256).unwrap();
        let x = ferrotorch_core::randn(&[2, 256, 14, 14]).unwrap();
        let out = head.forward(&x).unwrap();
        assert_eq!(
            out.shape(),
            &[2, 512, 14, 14],
            "keypoint head preserves spatial size, projects to 512"
        );
    }

    #[test]
    fn test_keypoint_predictor_shapes() {
        // Predictor steps: deconv(14→28) → bilinear-upsample(28→56).
        let predictor = KeypointPredictor::<f32>::new(512, 17).unwrap();
        let x = ferrotorch_core::randn(&[2, 512, 14, 14]).unwrap();
        let out = predictor.forward(&x).unwrap();
        assert_eq!(
            out.shape(),
            &[2, 17, 56, 56],
            "keypoint predictor 4× spatial (deconv 2× + bilinear 2×) + 17 channels"
        );
    }

    #[test]
    fn test_heatmaps_to_keypoints_argmax_location() {
        // Synthetic: one ROI at [0, 0, 28, 28], 1 keypoint, peak at (5, 7).
        // After bicubic upsample to (ceil(28), ceil(28)) = (28, 28),
        // argmax should be near (5, 7). The Heckbert correction is
        // 28/28 = 1.0, so output x = 5.5, y = 7.5.
        let n = 1usize;
        let num_kp = 1usize;
        let h = 28usize;
        let w = 28usize;
        let mut hm = vec![0.0f32; n * num_kp * h * w];
        hm[7 * w + 5] = 10.0; // peak at (x=5, y=7)
        let heatmaps =
            Tensor::from_storage(TensorStorage::cpu(hm), vec![n, num_kp, h, w], false).unwrap();
        let rois = Tensor::from_storage(
            TensorStorage::cpu(vec![0.0f32, 0.0, 28.0, 28.0]),
            vec![1, 4],
            false,
        )
        .unwrap();
        let (kp, scores) = heatmaps_to_keypoints(&heatmaps, &rois, num_kp).unwrap();
        assert_eq!(kp.shape(), &[1, 1, 3]);
        assert_eq!(scores.shape(), &[1, 1]);
        let kp_data = kp.data_vec().unwrap();
        // x ≈ 5.5, y ≈ 7.5, visibility flag = 1.
        assert!(
            (kp_data[0] - 5.5).abs() < 0.51,
            "x not at 5.5 ± 0.5: {}",
            kp_data[0],
        );
        assert!(
            (kp_data[1] - 7.5).abs() < 0.51,
            "y not at 7.5 ± 0.5: {}",
            kp_data[1],
        );
        assert!((kp_data[2] - 1.0).abs() < 1e-6, "vis flag != 1");
        // Score should be the peak logit.
        let s_data = scores.data_vec().unwrap();
        assert!(s_data[0] > 5.0, "argmax-logit score too low: {}", s_data[0]);
    }

    #[test]
    fn test_keypoint_rcnn_forward_output_structure() {
        let model = make_model();
        let img = no_grad(|| ferrotorch_core::randn(&[1, 3, 64, 64]).unwrap());
        let dets = no_grad(|| model.forward(&img).unwrap());
        assert_eq!(dets.len(), 1, "one detection list per image");
        let d = &dets[0];
        let n = d.boxes.shape()[0];
        assert_eq!(d.boxes.shape().len(), 2);
        assert_eq!(d.boxes.shape()[1], 4);
        assert_eq!(d.scores.shape().len(), 1);
        assert_eq!(d.scores.shape()[0], n);
        assert_eq!(d.labels.len(), n);
        // 2-class model (bg + person): only label 1 may appear (bg dropped).
        assert!(d.labels.iter().all(|&l| l == 1));
        // Keypoints: [N_det, 17, 3].
        assert_eq!(d.keypoints.shape()[0], n);
        assert_eq!(d.keypoints.shape()[1], 17);
        assert_eq!(d.keypoints.shape()[2], 3);
        // Keypoint scores: [N_det, 17].
        assert_eq!(d.keypoint_scores.shape()[0], n);
        assert_eq!(d.keypoint_scores.shape()[1], 17);
    }

    #[test]
    fn test_keypoint_rcnn_module_forward_returns_1d_scores() {
        // Locks the contract that `Module::forward` returns post-NMS scores
        // (1-D `[N_det]`), matching torchvision `model(img)[0]["scores"]`
        // and the retinanet/fcos/fasterrcnn detection harness convention.
        let model = make_model();
        let img = no_grad(|| ferrotorch_core::randn(&[1, 3, 64, 64]).unwrap());
        let out = no_grad(|| <KeypointRcnn<f32> as Module<f32>>::forward(&model, &img).unwrap());
        assert_eq!(out.shape().len(), 1, "Module::forward must be 1-D");
    }

    #[test]
    fn test_keypoint_rcnn_train_eval() {
        let mut model = make_model();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
    }
}
