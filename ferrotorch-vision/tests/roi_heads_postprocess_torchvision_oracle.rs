//! Oracle-derived parity tests for `detection::roi_heads_postprocess`.
//!
//! Tracking issue: #1141 — FasterRCNN + MaskRCNN RoIHeads postprocess.
//!
//! These tests are NOT hand-constructed: every expected value below was
//! produced by a *live* call into torchvision 0.26.0+cu130
//! (`torchvision.models.detection._utils.BoxCoder.decode`,
//! `torchvision.ops.{clip_boxes_to_image, remove_small_boxes, batched_nms}`,
//! `torchvision.models.detection.roi_heads.maskrcnn_inference`) reproducing
//! `RoIHeads.postprocess_detections`
//! (`/home/doll/.local/lib/python3.13/site-packages/torchvision/models/detection/roi_heads.py:680`)
//! exactly. This satisfies R-CHAR-3 (no tautological tests): the ground truth
//! comes from the upstream system we translate, not from re-deriving the
//! ferrotorch formula.
//!
//! ## Reproduction (frozen oracle)
//!
//! ```python
//! import torch, math
//! import torchvision.ops as box_ops
//! from torchvision.models.detection._utils import BoxCoder
//! from torchvision.models.detection.roi_heads import maskrcnn_inference
//!
//! num_classes = 3
//! weights = (10.0, 10.0, 5.0, 5.0)
//! bbox_xform_clip = math.log(1000.0 / 16)        # roi_heads default
//! score_thresh, nms_thresh, detections_per_img = 0.05, 0.5, 100
//! image_shape = (200, 240)                        # (H, W)
//!
//! proposals = torch.tensor([
//!     [10.0, 10.0, 60.0, 60.0],
//!     [12.0, 11.0, 61.0, 59.0],                   # overlaps proposal 0
//!     [120.0, 80.0, 180.0, 160.0],
//!     [5.0, 5.0, 8.0, 8.0],
//! ])
//! class_logits = torch.tensor([
//!     [-2.0, 4.0, -1.0],
//!     [-2.0, 3.5, -1.0],
//!     [-3.0, -1.0, 5.0],
//!     [ 1.0, 0.2, -0.5],
//! ])
//! box_regression = torch.tensor([
//!     [0,0,0,0,  0.1,0.05,0.02,0.03,  -0.1,0.1,0,0],
//!     [0,0,0,0,  0.05,0,0.01,0,       0,0,0,0],
//!     [0,0,0,0,  0,0,0,0,             0.2,-0.1,0.05,0.05],
//!     [0,0,0,0,  0,0,0,0,             0,0,0,0],
//! ])
//!
//! box_coder = BoxCoder(weights, bbox_xform_clip=bbox_xform_clip)
//! pred_boxes = box_coder.decode(box_regression, [proposals])   # [N, C, 4]
//! pred_scores = torch.softmax(class_logits, -1)
//! boxes = box_ops.clip_boxes_to_image(pred_boxes, image_shape)
//! labels = torch.arange(num_classes).view(1, -1).expand_as(pred_scores)
//! boxes, scores, labels = boxes[:, 1:], pred_scores[:, 1:], labels[:, 1:]
//! boxes, scores, labels = boxes.reshape(-1, 4), scores.reshape(-1), labels.reshape(-1)
//! inds = torch.where(scores > score_thresh)[0]
//! boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
//! keep = box_ops.remove_small_boxes(boxes, 1e-2)
//! boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
//! keep = box_ops.batched_nms(boxes, scores, labels, nms_thresh)[:detections_per_img]
//! boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
//! # -> det_boxes / det_scores / det_labels constants below.
//!
//! mask_logits = torch.randn(2, 3, 4, 4)  # seed torch.manual_seed(99) in oracle
//! masks = maskrcnn_inference(mask_logits, [torch.tensor([1, 2])])[0]  # [2,1,4,4]
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::unreadable_literal,
    clippy::excessive_precision
)]

use ferrotorch_core::from_slice;
use ferrotorch_vision::models::detection::roi_heads_postprocess::{
    PostprocessedDetections, ROI_BBOX_XFORM_CLIP, ROI_BOX_CODER_WEIGHTS, decode_per_class,
    postprocess_detections, postprocess_masks,
};

// ---------------------------------------------------------------------------
// Frozen torchvision oracle ground truth (torchvision 0.26.0+cu130).
// ---------------------------------------------------------------------------

const PROPOSALS: [f32; 16] = [
    10.0, 10.0, 60.0, 60.0, 12.0, 11.0, 61.0, 59.0, 120.0, 80.0, 180.0, 160.0, 5.0, 5.0, 8.0, 8.0,
];

const CLASS_LOGITS: [f32; 12] = [
    -2.0, 4.0, -1.0, -2.0, 3.5, -1.0, -3.0, -1.0, 5.0, 1.0, 0.2, -0.5,
];

const BOX_REGRESSION: [f32; 48] = [
    0.0, 0.0, 0.0, 0.0, 0.1, 0.05, 0.02, 0.03, -0.1, 0.1, 0.0, 0.0, // prop 0
    0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, // prop 1
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.1, 0.05, 0.05, // prop 2
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // prop 3
];

/// `BoxCoder.decode(box_regression, [proposals])` -> `[N=4, C=3, 4]`, flattened.
const ORACLE_DECODE: [f32; 48] = [
    10.0,
    10.0,
    60.0,
    60.0,
    10.399799346923828,
    10.09954833984375,
    60.60020065307617,
    60.40045166015625,
    9.5,
    10.5,
    59.5,
    60.5,
    12.0,
    11.0,
    61.0,
    59.0,
    12.19594955444336,
    11.0,
    61.29404830932617,
    59.0,
    12.0,
    11.0,
    61.0,
    59.0,
    120.0,
    80.0,
    180.0,
    160.0,
    120.0,
    80.0,
    180.0,
    160.0,
    120.89849090576172,
    78.79798889160156,
    181.50149536132812,
    159.6020050048828,
    5.0,
    5.0,
    8.0,
    8.0,
    5.0,
    5.0,
    8.0,
    8.0,
    5.0,
    5.0,
    8.0,
    8.0,
];

/// Final `det_boxes` after the full pipeline -> `[4, 4]`, flattened.
const ORACLE_DET_BOXES: [f32; 16] = [
    120.89849090576172,
    78.79798889160156,
    181.50149536132812,
    159.6020050048828,
    10.399799346923828,
    10.09954833984375,
    60.60020065307617,
    60.40045166015625,
    5.0,
    5.0,
    8.0,
    8.0,
    5.0,
    5.0,
    8.0,
    8.0,
];

/// Final `det_scores` (descending-score order, exactly batched_nms ordering).
const ORACLE_DET_SCORES: [f32; 4] = [
    0.9971936941146851,
    0.9908674955368042,
    0.268663614988327,
    0.13341441750526428,
];

/// Final `det_labels` (1-indexed class ids; never background).
const ORACLE_DET_LABELS: [usize; 4] = [2, 1, 1, 2];

/// `mask_logits` fed to `maskrcnn_inference`: `[2, 3, 4, 4]`, flattened.
const ORACLE_MASK_LOGITS: [f32; 96] = [
    0.29946109652519226,
    -2.6428585052490234,
    0.7233136296272278,
    0.8390920162200928,
    0.32038766145706177,
    1.8011391162872314,
    0.30793166160583496,
    -1.0945725440979004,
    -0.3583469092845917,
    0.1970628798007965,
    -2.3221840858459473,
    2.6447763442993164,
    0.8298611640930176,
    -1.344632625579834,
    -0.09134356677532196,
    -0.16750997304916382,
    -0.9618344902992249,
    -1.990372896194458,
    -1.6587568521499634,
    0.28559017181396484,
    0.13882692158222198,
    0.08840425312519073,
    0.05313143879175186,
    -0.7923781871795654,
    -0.9542126655578613,
    2.240208625793457,
    -0.018217405304312706,
    1.2306914329528809,
    -0.34562915563583374,
    -0.4904598891735077,
    0.612841010093689,
    3.097460985183716,
    0.32161375880241394,
    0.8502882122993469,
    0.6470543146133423,
    1.2965465784072876,
    1.115404486656189,
    -1.0094980001449585,
    0.6205565929412842,
    0.4542126953601837,
    0.8401968479156494,
    -2.018794298171997,
    -0.39639827609062195,
    -0.5167348384857178,
    -2.33622670173645,
    0.7005530595779419,
    -0.2804567515850067,
    1.4555810689926147,
    -2.0927884578704834,
    -0.7141631841659546,
    0.471746563911438,
    1.9486953020095825,
    0.2693808972835541,
    -1.3431543111801147,
    -0.41885071992874146,
    -0.39448168873786926,
    1.8266992568969727,
    0.7912219762802124,
    -1.5332196950912476,
    -0.8693944811820984,
    -0.29861053824424744,
    -0.7647802829742432,
    0.8588030338287354,
    -0.08247260749340057,
    0.20824621617794037,
    -0.15534016489982605,
    -1.3990508317947388,
    -0.36384496092796326,
    0.4769185781478882,
    -0.1941484808921814,
    0.3998847007751465,
    -0.8020793795585632,
    0.07190115749835968,
    -1.3454346656799316,
    1.4525600671768188,
    0.7868196368217468,
    -0.38486865162849426,
    -0.6554520130157471,
    -1.4504566192626953,
    -1.2058416604995728,
    1.6920998096466064,
    1.7633957862854004,
    -0.10354946553707123,
    -1.7408887147903442,
    -0.44142386317253113,
    -0.7707518935203552,
    1.4733405113220215,
    -0.6032773852348328,
    -0.3394666016101837,
    1.1822892427444458,
    -1.5273609161376953,
    -1.2709726095199585,
    -0.8510947227478027,
    0.9289596676826477,
    -1.4185048341751099,
    -0.5887886881828308,
];

/// `maskrcnn_inference(mask_logits, [[1, 2]])[0]` -> `[2, 1, 4, 4]`, flattened.
const ORACLE_MASK_SELECTED: [f32; 32] = [
    0.2765110433101654,
    0.12021742016077042,
    0.15992894768714905,
    0.5709161758422852,
    0.5346511006355286,
    0.5220866799354553,
    0.5132797360420227,
    0.31165826320648193,
    0.2780384123325348,
    0.9038026332855225,
    0.49544575810432434,
    0.7739395499229431,
    0.4144427478313446,
    0.3797852396965027,
    0.6485886573791504,
    0.9567878842353821,
    0.8445001244544983,
    0.8536344766616821,
    0.47413569688796997,
    0.14920009672641754,
    0.3914017379283905,
    0.31631648540496826,
    0.813564658164978,
    0.35359424352645874,
    0.415939062833786,
    0.7653591632843018,
    0.17838014662265778,
    0.2190907895565033,
    0.2992032468318939,
    0.7168641686439514,
    0.19489608705043793,
    0.3569128215312958,
];

const IMAGE_SIZE: [usize; 2] = [200, 240]; // (H, W)

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `decode_per_class` must reproduce torchvision's `BoxCoder.decode` to f32 ULP
/// tolerance for the FasterRCNN box-coder weights `(10, 10, 5, 5)`.
#[test]
fn decode_per_class_matches_torchvision_box_coder() {
    let proposals = from_slice::<f32>(&PROPOSALS, &[4, 4]).unwrap();
    let deltas = from_slice::<f32>(&BOX_REGRESSION, &[4, 12]).unwrap();
    let decoded = decode_per_class::<f32>(
        &proposals,
        &deltas,
        ROI_BOX_CODER_WEIGHTS,
        ROI_BBOX_XFORM_CLIP,
    )
    .unwrap();
    assert_eq!(decoded.shape(), &[4, 3, 4], "decode shape mismatch");
    let got = decoded.data_vec().unwrap();
    assert_eq!(got.len(), ORACLE_DECODE.len());
    for (i, (&g, &e)) in got.iter().zip(ORACLE_DECODE.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "decode[{i}]: ferrotorch={g} torchvision={e} (diff {})",
            (g - e).abs()
        );
    }
}

/// Full `postprocess_detections` pipeline must reproduce torchvision's
/// `RoIHeads.postprocess_detections`: identical surviving boxes, scores, labels
/// AND identical ordering (the batched_nms descending-score order + top-K
/// slice).
#[test]
fn postprocess_detections_matches_torchvision_roiheads() {
    let logits = from_slice::<f32>(&CLASS_LOGITS, &[4, 3]).unwrap();
    let deltas = from_slice::<f32>(&BOX_REGRESSION, &[4, 12]).unwrap();
    let proposals = from_slice::<f32>(&PROPOSALS, &[4, 4]).unwrap();

    let PostprocessedDetections {
        boxes,
        scores,
        labels,
    } = postprocess_detections::<f32>(&logits, &deltas, &proposals, IMAGE_SIZE).unwrap();

    // Count must match exactly — no spurious drops or survivals.
    assert_eq!(
        labels.len(),
        ORACLE_DET_LABELS.len(),
        "detection count mismatch: ferrotorch={:?} torchvision={:?}",
        labels,
        ORACLE_DET_LABELS
    );
    assert_eq!(boxes.shape(), &[4, 4]);
    assert_eq!(scores.shape(), &[4]);

    // Labels must match in order (NMS ordering is part of the contract).
    assert_eq!(
        labels.as_slice(),
        &ORACLE_DET_LABELS,
        "label/order mismatch vs torchvision batched_nms"
    );

    let score_vec = scores.data_vec().unwrap();
    for (i, (&g, &e)) in score_vec.iter().zip(ORACLE_DET_SCORES.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-4,
            "score[{i}]: ferrotorch={g} torchvision={e}"
        );
    }

    let box_vec = boxes.data_vec().unwrap();
    for (i, (&g, &e)) in box_vec.iter().zip(ORACLE_DET_BOXES.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "box[{i}]: ferrotorch={g} torchvision={e}"
        );
    }
}

/// `postprocess_masks(paste=false)` must reproduce torchvision's
/// `maskrcnn_inference`: `mask_prob = x.sigmoid()` then class-select
/// `mask_prob[index, labels]`.
#[test]
fn postprocess_masks_no_paste_matches_maskrcnn_inference() {
    let mask_logits = from_slice::<f32>(&ORACLE_MASK_LOGITS, &[2, 3, 4, 4]).unwrap();
    let labels = [1usize, 2];
    // maskrcnn_inference does not use boxes; pass placeholder boxes for the API.
    let boxes = from_slice::<f32>(&[0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 4.0, 4.0], &[2, 4]).unwrap();

    let out = postprocess_masks::<f32>(&mask_logits, &labels, &boxes, [16, 16], false).unwrap();
    assert_eq!(out.shape(), &[2, 1, 4, 4], "no-paste mask shape mismatch");
    let got = out.data_vec().unwrap();
    assert_eq!(got.len(), ORACLE_MASK_SELECTED.len());
    for (i, (&g, &e)) in got.iter().zip(ORACLE_MASK_SELECTED.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "mask[{i}]: ferrotorch={g} torchvision={e}"
        );
    }
}
