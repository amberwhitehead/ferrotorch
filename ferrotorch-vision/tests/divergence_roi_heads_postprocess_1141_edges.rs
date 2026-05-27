//! Adversarial edge-case audit of `detection::roi_heads_postprocess` (#1141).
//!
//! The happy-path oracle tests in `roi_heads_postprocess_torchvision_oracle.rs`
//! cover well-separated, high-confidence detections. This file hunts the
//! EDGE cases those miss, against
//! `torchvision/models/detection/roi_heads.py:680 postprocess_detections`
//! (live torchvision 0.26.0+cu130) and `:56 maskrcnn_inference`.
//!
//! Every expected value below was produced by a *live* torchvision call
//! (frozen reproduction: `/tmp/roi_oracle_audit1141.py` + `...1141b.py`), NOT
//! re-derived from the ferrotorch formula (R-CHAR-3).
//!
//! ## Frozen oracle output (torchvision 0.26.0+cu130)
//! ```text
//! case1_all_bg     : n_det=0, labels=[]
//! case2_boundary   : fg_score_exact=0.05000000074505806, n_det=0  (scores > 0.05 strict, f32-demoted thresh)
//! case2_just_above : fg_score=0.05010000616312027,       n_det=1
//! case3_small_box  : n_det=1, boxes=[[60,60,100,100]], labels=[1]  (width-0.005 box dropped)
//! case4_clip       : n_det=1, boxes=[[0,0,100,100]], labels=[1]    (clip BEFORE filter; [-20,-10,150,130]->[0,0,100,100])
//! case5_cross_class: n_det=2, labels=[1,2]  (overlapping diff-class BOTH survive)
//! case5_same_class : n_det=1, labels=[1]    (overlapping same-class, lower suppressed)
//! case6_order      : labels=[1,2,3,4], scores=[1.0,0.9999995,0.9999967,0.9999754]  (global descending-score order)
//! case8_mask_select: shape=[2,1,4,4], det0_val=0.99995458, det1_val=0.99995458
//!                    (channel == label, NO background-offset; off-by-one would give 4.54e-5)
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::unreadable_literal,
    clippy::excessive_precision,
    clippy::identity_op
)]

use ferrotorch_core::from_slice;
use ferrotorch_vision::models::detection::roi_heads_postprocess::{
    postprocess_detections, postprocess_masks,
};

// ---------------------------------------------------------------------------
// Edge case 1 — all proposals score highest on background (class 0).
// torchvision drops class 0 -> returns EMPTY for the image.
// roi_heads.py:710-712 `boxes = boxes[:, 1:]` and :720 `scores > score_thresh`.
// ---------------------------------------------------------------------------
/// Divergence probe: all-background input.
/// Upstream torchvision `postprocess_detections` returns n_det=0, labels=[].
/// (`roi_heads.py:710` drops bg, the remaining fg scores are ~4.5e-5 < 0.05.)
#[test]
fn divergence_case1_all_background_returns_empty() {
    // num_classes=3; every proposal strongly favours class 0 (background).
    let logits = from_slice::<f32>(&[10.0, -5.0, -5.0, 10.0, -5.0, -5.0], &[2, 3]).unwrap();
    let deltas = from_slice::<f32>(&[0.0_f32; 24], &[2, 12]).unwrap();
    let proposals =
        from_slice::<f32>(&[10.0, 10.0, 50.0, 50.0, 60.0, 60.0, 100.0, 100.0], &[2, 4]).unwrap();
    let det = postprocess_detections::<f32>(&logits, &deltas, &proposals, [200, 200]).unwrap();
    // torchvision oracle: n_det == 0.
    assert_eq!(det.boxes.shape(), &[0, 4], "all-bg must yield empty boxes");
    assert_eq!(det.scores.shape(), &[0], "all-bg must yield empty scores");
    assert_eq!(det.labels.len(), 0, "all-bg must yield empty labels");
}

// ---------------------------------------------------------------------------
// Edge case 2 — score exactly at threshold. torchvision uses `scores > thresh`
// with the scalar 0.05 DEMOTED to the tensor's f32 dtype, so a softmax fg value
// that rounds to exactly 0.05f32 fails the strict `>` and is DROPPED.
// roi_heads.py:720 `inds = torch.where(scores > self.score_thresh)[0]`.
// ---------------------------------------------------------------------------
/// Divergence probe: fg score == 0.05 exactly (a = ln(19), b = 0).
/// torchvision oracle: fg softmax = 0.05000000074505806 (f32), n_det = 0
/// because `0.05f32 > 0.05f32` is False (strict, f32-demoted threshold).
#[test]
fn divergence_case2_score_thresh_boundary_exact_dropped() {
    // logits [ln(19), 0] over 2 classes -> fg softmax exactly 0.05f32.
    let a = 19.0_f32.ln();
    let logits = from_slice::<f32>(&[a, 0.0], &[1, 2]).unwrap();
    let deltas = from_slice::<f32>(&[0.0_f32; 8], &[1, 8]).unwrap();
    let proposals = from_slice::<f32>(&[0.0, 0.0, 40.0, 40.0], &[1, 4]).unwrap();
    let det = postprocess_detections::<f32>(&logits, &deltas, &proposals, [200, 200]).unwrap();
    assert_eq!(
        det.boxes.shape(),
        &[0, 4],
        "score exactly at 0.05 must be DROPPED by strict `>`"
    );
}

/// Divergence probe: fg score just above threshold (0.0501).
/// torchvision oracle: n_det = 1.
#[test]
fn divergence_case2_score_thresh_just_above_kept() {
    // a = ln((1-0.0501)/0.0501) -> fg softmax = 0.0501 > 0.05 -> kept.
    let a = ((1.0_f32 - 0.0501) / 0.0501).ln();
    let logits = from_slice::<f32>(&[a, 0.0], &[1, 2]).unwrap();
    let deltas = from_slice::<f32>(&[0.0_f32; 8], &[1, 8]).unwrap();
    let proposals = from_slice::<f32>(&[0.0, 0.0, 40.0, 40.0], &[1, 4]).unwrap();
    let det = postprocess_detections::<f32>(&logits, &deltas, &proposals, [200, 200]).unwrap();
    assert_eq!(
        det.boxes.shape(),
        &[1, 4],
        "score just above 0.05 must be KEPT"
    );
}

// ---------------------------------------------------------------------------
// Edge case 3 — box too small in ONE dim. remove_small_boxes drops if
// EITHER width OR height < min_size (1e-2). boxes.py:144
// `keep = (ws >= min_size) & (hs >= min_size)`.
// ---------------------------------------------------------------------------
/// Divergence probe: box 0 has width 0.005 (< 1e-2), height 40 (ok) -> DROPPED.
/// box 1 is 40x40 -> kept. torchvision oracle: n_det=1, box=[60,60,100,100], label=1.
#[test]
fn divergence_case3_small_box_one_dim_dropped() {
    let logits = from_slice::<f32>(&[-5.0, 5.0, -5.0, 5.0], &[2, 2]).unwrap();
    let deltas = from_slice::<f32>(&[0.0_f32; 16], &[2, 8]).unwrap();
    let proposals = from_slice::<f32>(
        &[10.0, 10.0, 10.005, 50.0, 60.0, 60.0, 100.0, 100.0],
        &[2, 4],
    )
    .unwrap();
    let det = postprocess_detections::<f32>(&logits, &deltas, &proposals, [200, 200]).unwrap();
    assert_eq!(det.boxes.shape(), &[1, 4], "thin box must be removed");
    assert_eq!(det.labels, vec![1usize]);
    let b = det.boxes.data_vec().unwrap();
    assert!((b[0] - 60.0).abs() < 1e-3, "kept box x1={}", b[0]);
    assert!((b[2] - 100.0).abs() < 1e-3, "kept box x2={}", b[2]);
}

// ---------------------------------------------------------------------------
// Edge case 4 — clip_boxes_to_image BEFORE score-filter/NMS. roi_heads.py:703
// runs `clip_boxes_to_image(boxes, image_shape)` before dropping bg / filtering.
// A box past bounds [-20,-10,150,130] in a 100x100 image clips to [0,0,100,100].
// ---------------------------------------------------------------------------
/// Divergence probe: out-of-bounds box clipped to image bounds before NMS.
/// torchvision oracle: n_det=1, box=[0,0,100,100], label=1.
#[test]
fn divergence_case4_clip_boxes_before_filter() {
    let logits = from_slice::<f32>(&[-5.0, 5.0], &[1, 2]).unwrap();
    let deltas = from_slice::<f32>(&[0.0_f32; 8], &[1, 8]).unwrap();
    // proposal extends past [0,100]x[0,100].
    let proposals = from_slice::<f32>(&[-20.0, -10.0, 150.0, 130.0], &[1, 4]).unwrap();
    let det = postprocess_detections::<f32>(&logits, &deltas, &proposals, [100, 100]).unwrap();
    assert_eq!(det.boxes.shape(), &[1, 4]);
    let b = det.boxes.data_vec().unwrap();
    // Clipped to [0,0,W=100,H=100].
    assert!((b[0] - 0.0).abs() < 1e-3, "x1 clip, got {}", b[0]);
    assert!((b[1] - 0.0).abs() < 1e-3, "y1 clip, got {}", b[1]);
    assert!((b[2] - 100.0).abs() < 1e-3, "x2 clip, got {}", b[2]);
    assert!((b[3] - 100.0).abs() < 1e-3, "y2 clip, got {}", b[3]);
    assert_eq!(det.labels, vec![1usize]);
}

// ---------------------------------------------------------------------------
// Edge case 5 — per-class NMS. Two heavily-overlapping boxes of DIFFERENT
// classes must BOTH survive (offset trick keeps them apart). boxes.py:100-102.
// ---------------------------------------------------------------------------
/// Divergence probe: two overlapping boxes, classes 1 and 2 -> BOTH survive.
/// torchvision oracle: n_det=2, labels=[1,2].
#[test]
fn divergence_case5_cross_class_overlap_both_survive() {
    let logits = from_slice::<f32>(&[-5.0, 5.0, -5.0, -5.0, -5.0, 5.0], &[2, 3]).unwrap();
    let deltas = from_slice::<f32>(&[0.0_f32; 24], &[2, 12]).unwrap();
    let proposals =
        from_slice::<f32>(&[10.0, 10.0, 50.0, 50.0, 11.0, 11.0, 51.0, 51.0], &[2, 4]).unwrap();
    let det = postprocess_detections::<f32>(&logits, &deltas, &proposals, [200, 200]).unwrap();
    assert_eq!(
        det.boxes.shape()[0],
        2,
        "cross-class overlap: both boxes must survive per-class NMS"
    );
    let mut labels = det.labels.clone();
    labels.sort_unstable();
    assert_eq!(labels, vec![1, 2]);
}

/// Divergence probe: two overlapping boxes, SAME class 1 -> lower suppressed.
/// torchvision oracle: n_det=1, labels=[1].
#[test]
fn divergence_case5_same_class_overlap_one_suppressed() {
    let logits = from_slice::<f32>(&[-5.0, 5.0, -5.0, -5.0, 4.0, -5.0], &[2, 3]).unwrap();
    let deltas = from_slice::<f32>(&[0.0_f32; 24], &[2, 12]).unwrap();
    let proposals =
        from_slice::<f32>(&[10.0, 10.0, 50.0, 50.0, 11.0, 11.0, 51.0, 51.0], &[2, 4]).unwrap();
    let det = postprocess_detections::<f32>(&logits, &deltas, &proposals, [200, 200]).unwrap();
    assert_eq!(
        det.boxes.shape()[0],
        1,
        "same-class overlap: one suppressed"
    );
    assert_eq!(det.labels, vec![1usize]);
}

// ---------------------------------------------------------------------------
// Edge case 6 — cross-class top-K relies on batched_nms returning indices in
// global DESCENDING-score order (boxes.py:102 -> nms sorts desc), so
// `keep[:detections_per_img]` keeps the highest-scoring. Verify the OUTPUT
// ORDER of the kept detections is global descending score across classes.
// ---------------------------------------------------------------------------
/// Divergence probe: 4 distinct-class detections with strictly decreasing
/// scores must come out in global descending-score order (labels 1,2,3,4).
/// torchvision oracle: labels=[1,2,3,4], scores=[1.0,0.9999995,0.9999967,0.9999754].
#[test]
fn divergence_case6_topk_global_descending_order() {
    let logits = from_slice::<f32>(
        &[
            -10.0, 8.0, -10.0, -10.0, -10.0, // class1 highest score
            -10.0, -10.0, 6.0, -10.0, -10.0, // class2
            -10.0, -10.0, -10.0, 4.0, -10.0, // class3
            -10.0, -10.0, -10.0, -10.0, 2.0, // class4 lowest
        ],
        &[4, 5],
    )
    .unwrap();
    let deltas = from_slice::<f32>(&[0.0_f32; 80], &[4, 20]).unwrap();
    let proposals = from_slice::<f32>(
        &[
            0.0, 0.0, 30.0, 30.0, 100.0, 0.0, 130.0, 30.0, 0.0, 100.0, 30.0, 130.0, 100.0, 100.0,
            130.0, 130.0,
        ],
        &[4, 4],
    )
    .unwrap();
    let det = postprocess_detections::<f32>(&logits, &deltas, &proposals, [500, 500]).unwrap();
    assert_eq!(
        det.boxes.shape()[0],
        4,
        "all 4 distinct-class boxes survive"
    );
    // Must be in global descending-score order (the order keep[:K] truncates).
    assert_eq!(
        det.labels,
        vec![1, 2, 3, 4],
        "kept order must be global descending score"
    );
    let s = det.scores.data_vec().unwrap();
    for w in s.windows(2) {
        assert!(
            w[0] >= w[1],
            "scores must be non-increasing: {} then {}",
            w[0],
            w[1]
        );
    }
    assert!((s[0] - 1.0).abs() < 1e-4, "top score ~1.0, got {}", s[0]);
}

// ---------------------------------------------------------------------------
// Edge case 8 — mask label-index. maskrcnn_inference selects channel == label
// (NO background-offset removal): roi_heads.py:79 `mask_prob[index, labels]`.
// labels here are 1-based (bg already dropped) and the mask head has
// num_classes channels INCLUDING bg, so channel == label directly.
// ---------------------------------------------------------------------------
/// Divergence probe: det0 label=1 -> channel 1; det1 label=2 -> channel 2.
/// Channel L holds +10 logit (sigmoid 0.99995458); other channels -10.
/// torchvision oracle: det0_val=0.99995458, det1_val=0.99995458.
/// An off-by-one (label-1) would select channel 0 -> 4.54e-5 (the trap).
#[test]
fn divergence_case8_mask_selects_label_channel_no_offset() {
    // [N=2, C=3, 4, 4]. det0: ch1=+10; det1: ch2=+10; all other channels -10.
    let mut logits = vec![-10.0_f32; 2 * 3 * 4 * 4];
    let plane = 4 * 4;
    // det 0, channel 1 -> +10
    for i in 0..plane {
        logits[0 * 3 * plane + 1 * plane + i] = 10.0;
    }
    // det 1, channel 2 -> +10
    for i in 0..plane {
        logits[1 * 3 * plane + 2 * plane + i] = 10.0;
    }
    let mask_logits = from_slice::<f32>(&logits, &[2, 3, 4, 4]).unwrap();
    let labels = vec![1usize, 2usize];
    let boxes = from_slice::<f32>(&[0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 4.0, 4.0], &[2, 4]).unwrap();
    let out = postprocess_masks::<f32>(&mask_logits, &labels, &boxes, [16, 16], false).unwrap();
    assert_eq!(out.shape(), &[2, 1, 4, 4]);
    let d = out.data_vec().unwrap();
    // det0 selected channel 1 (=+10 -> sigmoid 0.99995458).
    assert!(
        (d[0] - 0.99995458).abs() < 1e-4,
        "det0 must select channel==label(1); got {} (off-by-one trap value ~4.5e-5)",
        d[0]
    );
    // det1 selected channel 2 (=+10 -> sigmoid 0.99995458).
    assert!(
        (d[plane] - 0.99995458).abs() < 1e-4,
        "det1 must select channel==label(2); got {}",
        d[plane]
    );
}
