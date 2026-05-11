#!/usr/bin/env python3
"""#1145 per-stage probe for keypointrcnn_resnet50_fpn.

Loads `keypointrcnn_resnet50_fpn(pretrained)`, runs forward with hooks that
capture:

  - per-image post-NMS detection scores / boxes / keypoints
  - the raw 56x56 keypoint heatmaps that feed `heatmaps_to_keypoints`

then compares against the rust dumps in /tmp/ferrotorch_verify_images/
(scores / boxes / keypoints / keypoint_scores) for each COCO probe image.

Goal: locate the first stage where rust and torchvision diverge so we can
fix the actual bug rather than chase symptoms.

Usage:
  python3 scripts/probe_keypointrcnn_stages_1145.py [--image-id 252219]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.image_list import ImageList

COCO_IDS = [37777, 87038, 174482, 252219, 397133]
CACHE_DIR = Path("/tmp/ferrotorch_verify_images")
DEVICE = torch.device("cpu")  # match rust harness


def preprocess_keypointrcnn(chw: torch.Tensor) -> torch.Tensor:
    """Mirror `preprocess_for_model("keypointrcnn_resnet50_fpn")` in rust."""
    _, h, w = chw.shape
    s_min = 800.0 / min(h, w)
    s_max = 1333.0 / max(h, w)
    scale = min(s_min, s_max)
    out_h = round(h * scale)
    out_w = round(w * scale)
    bchw = F.interpolate(
        chw.unsqueeze(0),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normed = (bchw - mean) / std
    stride = 32
    pad_h = ((out_h + stride - 1) // stride) * stride
    pad_w = ((out_w + stride - 1) // stride) * stride
    if pad_h == out_h and pad_w == out_w:
        return normed
    padded = torch.zeros(1, 3, pad_h, pad_w)
    padded[:, :, :out_h, :out_w] = normed
    return padded


def patch_noop_transform(model: torch.nn.Module) -> None:
    """Replace GeneralizedRCNNTransform with no-op so the pre-resized,
    pre-normalized rust input is consumed verbatim."""
    class NoopTransform(GeneralizedRCNNTransform):
        def __init__(self):
            # Skip GeneralizedRCNNTransform.__init__ (we don't use its state)
            # but still initialise nn.Module internals so PyTorch's forward
            # hook plumbing works.
            torch.nn.Module.__init__(self)

        def forward(self, images, targets=None):
            stacked = torch.stack(images, dim=0)
            image_sizes = [(t.shape[-2], t.shape[-1]) for t in images]
            return ImageList(stacked, image_sizes), targets

        def postprocess(self, result, image_shapes, original_image_sizes):
            return result

    model.transform = NoopTransform()


def read_rust_dump(path: Path) -> np.ndarray:
    """Same format as harness `read_dump` (u32 rank + u32 dims + f32 data)."""
    import struct
    with open(path, "rb") as f:
        rank = struct.unpack("<I", f.read(4))[0]
        shape = struct.unpack(f"<{rank}I", f.read(4 * rank))
        numel = int(np.prod(shape)) if shape else 1
        data = np.frombuffer(f.read(4 * numel), dtype=np.float32)
    return data.reshape(shape)


def pair_by_box_iou(rb: np.ndarray, tb: np.ndarray, thresh: float = 0.5):
    if rb.shape[0] == 0 or tb.shape[0] == 0:
        return []
    # IoU computation
    ra = (rb[:, 2] - rb[:, 0]) * (rb[:, 3] - rb[:, 1])
    ta = (tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])
    pairs = []
    for ri in range(rb.shape[0]):
        best = -1.0
        best_ti = -1
        for ti in range(tb.shape[0]):
            x1 = max(rb[ri, 0], tb[ti, 0])
            y1 = max(rb[ri, 1], tb[ti, 1])
            x2 = min(rb[ri, 2], tb[ti, 2])
            y2 = min(rb[ri, 3], tb[ti, 3])
            iw = max(0.0, x2 - x1)
            ih = max(0.0, y2 - y1)
            inter = iw * ih
            union = ra[ri] + ta[ti] - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best and iou >= thresh:
                best = iou
                best_ti = ti
        if best_ti >= 0:
            pairs.append((ri, best_ti, best))
    return pairs


def probe_image(model, image_id: int, image_path: Path):
    print(f"\n=== Image {image_id} ===")
    raw = Image.open(image_path).convert("RGB")
    chw = torch.from_numpy(np.array(raw)).float().permute(2, 0, 1) / 255.0
    bchw = preprocess_keypointrcnn(chw)
    print(f"  preprocessed shape: {tuple(bchw.shape)}")

    with torch.no_grad():
        preds = model([bchw[0]])
    p = preds[0]
    tv_scores = p["scores"].cpu().numpy()
    tv_boxes = p["boxes"].cpu().numpy()
    tv_keypoints = p["keypoints"].cpu().numpy()
    tv_kp_scores = p["keypoints_scores"].cpu().numpy()
    print(f"  tv: n_det={tv_scores.shape[0]}, top-5 scores={tv_scores[:5].tolist()}")

    # Load rust dumps
    base = CACHE_DIR / f"dump_keypointrcnn_resnet50_fpn_{image_id:012d}.bin"
    r_scores = read_rust_dump(base)
    r_boxes = read_rust_dump(Path(str(base) + ".boxes.bin"))
    r_keypoints = read_rust_dump(Path(str(base) + ".keypoints.bin"))
    r_kp_scores = read_rust_dump(Path(str(base) + ".keypoint_scores.bin"))
    print(f"  rust: n_det={r_scores.shape[0]}, top-5 scores={r_scores[:5].tolist()}")

    # ---- Score-side analysis ----
    k = min(5, r_scores.shape[0], tv_scores.shape[0])
    if k > 0:
        rs = np.sort(r_scores)[::-1][:k]
        ts = np.sort(tv_scores)[::-1][:k]
        diff = np.abs(rs - ts)
        print(f"  Top-{k} score abs diff:")
        for i in range(k):
            print(f"    rank {i}: rust={rs[i]:.4f} tv={ts[i]:.4f} |d|={diff[i]:.5f}")
        print(f"  score_max_abs={diff.max():.5f}  (threshold 0.02)")

    # ---- Detection count parity ----
    n_r, n_t = r_scores.shape[0], tv_scores.shape[0]
    denom = max(n_r, n_t)
    ratio = (min(n_r, n_t) / denom) if denom else 1.0
    print(f"  n_det_ratio = {ratio:.3f}  ({n_r}/{n_t})  (floor 0.80)")

    # ---- Keypoint pixel diffs on matched boxes ----
    score_thresh = 0.5
    box_iou_thresh = 0.5
    r_keep = np.where(r_scores > score_thresh)[0]
    t_keep = np.where(tv_scores > score_thresh)[0]
    rb_f = r_boxes[r_keep] if r_keep.size else np.zeros((0, 4), dtype=np.float32)
    tb_f = tv_boxes[t_keep] if t_keep.size else np.zeros((0, 4), dtype=np.float32)
    pairs = pair_by_box_iou(rb_f, tb_f, box_iou_thresh)
    print(f"  above-thresh: rust={r_keep.size} tv={t_keep.size}  matched={len(pairs)}")
    if pairs:
        diffs = []
        for (ri_l, ti_l, iou) in pairs:
            ri = int(r_keep[ri_l])
            ti = int(t_keep[ti_l])
            r_xy = r_keypoints[ri, :, :2]
            t_xy = tv_keypoints[ti, :, :2]
            d = np.sqrt(((r_xy - t_xy) ** 2).sum(axis=1))
            diffs.append((iou, float(d.mean()), float(d.max()), r_keypoints[ri], tv_keypoints[ti], r_boxes[ri], tv_boxes[ti]))
        for j, (iou, mean_d, max_d, rkp, tkp, rb, tb) in enumerate(diffs):
            print(f"    pair {j}: box_iou={iou:.3f} mean_kp_diff={mean_d:.2f}px max={max_d:.2f}px")
            print(f"             rust box: [{rb[0]:.1f}, {rb[1]:.1f}, {rb[2]:.1f}, {rb[3]:.1f}]")
            print(f"             tv box  : [{tb[0]:.1f}, {tb[1]:.1f}, {tb[2]:.1f}, {tb[3]:.1f}]")
            if mean_d > 5.0:
                # Show per-keypoint diff for failing pair
                d = np.sqrt(((rkp[:, :2] - tkp[:, :2]) ** 2).sum(axis=1))
                print(f"             per-kp diffs: {[round(float(x),2) for x in d.tolist()]}")
                print(f"             rust kp[:5]: {[(round(float(rkp[i,0]),1), round(float(rkp[i,1]),1)) for i in range(5)]}")
                print(f"             tv   kp[:5]: {[(round(float(tkp[i,0]),1), round(float(tkp[i,1]),1)) for i in range(5)]}")
        print(f"  mean kp_pixel_diff across pairs: {np.mean([d[1] for d in diffs]):.2f} px  (threshold 5.0)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", type=int, default=None,
                    help="Specific COCO id to probe; default = all 5.")
    args = ap.parse_args()

    weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
    print("loading torchvision keypointrcnn_resnet50_fpn (this can take a moment)...")
    model = keypointrcnn_resnet50_fpn(weights=weights).to(DEVICE).eval()
    patch_noop_transform(model)

    ids = [args.image_id] if args.image_id is not None else COCO_IDS
    for image_id in ids:
        image_path = CACHE_DIR / f"coco_{image_id:012d}.jpg"
        if not image_path.exists():
            print(f"!!! image cache missing: {image_path}; run the verify harness once first")
            continue
        probe_image(model, image_id, image_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
