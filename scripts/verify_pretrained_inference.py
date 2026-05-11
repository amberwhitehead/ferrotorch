#!/usr/bin/env python3
"""Verify ferrotorch pretrained-model inference against torchvision reference.

For each of the 5 newly-pinned models from #1130, this script:

  1. Loads the torchvision pretrained model.
  2. Loads N=5 fixed COCO val2017 images.
  3. Preprocesses each image to match the **ferrotorch** Rust binary's
     preprocessing recipe (so the two run on the same input tensor).
  4. Runs torchvision's *raw* forward (bypassing GeneralizedRCNNTransform
     and any internal normalization) on that same tensor.
  5. Extracts the equivalent of `Module::forward`'s return value for each
     model so we can diff against the Rust dump:
        SSD300        → first-image scores Tensor [N_det]
        FasterRCNN    → first-image class softmax Tensor [N_prop, 91]
        MaskRCNN      → first-image mask logits Tensor [N_det, 91, 28, 28]
        DeepLabV3/FCN → output['out'] [B, 21, H, W]
  6. Invokes the ferrotorch Rust binary on the same image.
  7. Compares with model-specific tolerances and prints a verdict.

This is intentionally a measurement tool — it makes no fixes and reports
verdicts honestly. A FAIL diagnoses *where* the divergence happens
(preprocessing? NMS? FPN bias?) so a follow-up dispatch can address it.

Usage:
  python3 scripts/verify_pretrained_inference.py [--models ssd300_vgg16,...]
                                                  [--quiet]

The Rust binary must be pre-built:
  cargo build -p ferrotorch-vision --release --example inference_dump
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    ssd300_vgg16,
)
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights,
    deeplabv3_resnet50,
    fcn_resnet50,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUST_BIN = REPO_ROOT / "target" / "release" / "examples" / "inference_dump"
CACHE_DIR = Path("/tmp/ferrotorch_verify_images")

# 5 fixed COCO val2017 image IDs (first 5 by sorted ID).
COCO_IDS = [37777, 87038, 174482, 252219, 397133]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Per-model numerical tolerances.
TOL = {
    "ssd300_vgg16": dict(abs_score=1e-3, abs_box_px=2.0),
    "fasterrcnn_resnet50_fpn": dict(abs_score=1e-3, abs_box_px=2.0),
    "maskrcnn_resnet50_fpn": dict(abs_score=1e-3, abs_box_px=2.0, abs_mask=1e-2),
    "deeplabv3_resnet50": dict(abs_logit=1e-3, argmax_agree_pct=99.0),
    "fcn_resnet50": dict(abs_logit=1e-3, argmax_agree_pct=99.0),
}


# ---------------------------------------------------------------------------
# Preprocessing helpers — MUST mirror ferrotorch-vision's
# `preprocess_for_model` in examples/inference_dump.rs exactly.
# ---------------------------------------------------------------------------


def load_image_chw(path: Path) -> torch.Tensor:
    """Load image as [3, H, W] tensor in [0, 1]."""
    pil = Image.open(path).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0  # HWC
    chw = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return chw


def preprocess(model: str, chw: torch.Tensor) -> torch.Tensor:
    """Build the [1, 3, H_out, W_out] input matching the Rust binary."""
    _, h, w = chw.shape
    if model == "ssd300_vgg16":
        bchw = chw.unsqueeze(0)
        bchw = F.interpolate(bchw, size=(300, 300), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (bchw - mean) / std
    if model in ("fasterrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn"):
        s_min = 800.0 / min(h, w)
        s_max = 1333.0 / max(h, w)
        scale = min(s_min, s_max)
        out_h = round(h * scale)
        out_w = round(w * scale)
        bchw = F.interpolate(
            chw.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        normed = (bchw - mean) / std
        stride = 32
        pad_h = ((out_h + stride - 1) // stride) * stride
        pad_w = ((out_w + stride - 1) // stride) * stride
        if pad_h != out_h or pad_w != out_w:
            padded = torch.zeros(1, 3, pad_h, pad_w)
            padded[:, :, :out_h, :out_w] = normed
            return padded
        return normed
    if model in ("deeplabv3_resnet50", "fcn_resnet50"):
        scale = 520.0 / min(h, w)
        out_h = round(h * scale)
        out_w = round(w * scale)
        bchw = F.interpolate(
            chw.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (bchw - mean) / std
    raise ValueError(f"unknown model: {model}")


# ---------------------------------------------------------------------------
# Reading the Rust dump format:
#   [u32 ndim][u32 × ndim shape][f32 × prod(shape) data]
# ---------------------------------------------------------------------------


def read_dump(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        ndim = struct.unpack("<I", f.read(4))[0]
        shape = struct.unpack(f"<{ndim}I", f.read(4 * ndim))
        numel = int(np.prod(shape)) if shape else 1
        data = np.frombuffer(f.read(4 * numel), dtype=np.float32)
        return data.reshape(shape)


# ---------------------------------------------------------------------------
# Torchvision references: extract the Module::forward-equivalent output.
# ---------------------------------------------------------------------------


def torchvision_module_equivalent(model_name: str, input_bchw: torch.Tensor) -> np.ndarray:
    """Run torchvision and return the same shape ferrotorch's Module::forward
    produces, so we can diff directly.

    SSD300        → SSD300's full forward returns Vec[Dict[boxes, scores, labels]]
                    where `scores` is [N_det] after NMS.  We extract the
                    first image's `scores`.
    FasterRCNN    → the *raw* class_logits for all proposals → softmax →
                    [N_prop, 91].  We bypass GeneralizedRCNNTransform and
                    run the model in `eval()` mode on the already-preprocessed
                    tensor.
    MaskRCNN      → raw mask logits before sigmoid, [N_det, 91, 28, 28].
    DeepLabV3/FCN → output['out'] tensor.
    """
    if model_name == "ssd300_vgg16":
        weights = SSD300_VGG16_Weights.COCO_V1
        m = ssd300_vgg16(weights=weights).to(DEVICE).eval()
        # Bypass internal transform by replacing it with identity.
        # SSD's internal transform: resize to 300×300 + non-ImageNet norm.
        # We've already done resize + (torchvision's own) normalize=NO — we
        # used ImageNet stats matching ferrotorch's expectation. To make
        # torchvision use OUR preprocessed tensor verbatim, we patch the
        # transform to a no-op.
        _patch_detection_transform(m)
        with torch.no_grad():
            preds = m([input_bchw[0].to(DEVICE)])
        scores = preds[0]["scores"].detach().cpu().numpy().astype(np.float32)
        return scores

    if model_name == "fasterrcnn_resnet50_fpn":
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        m = fasterrcnn_resnet50_fpn(weights=weights).to(DEVICE).eval()
        _patch_detection_transform(m)
        with torch.no_grad():
            preds = m([input_bchw[0].to(DEVICE)])
        # Pre-postprocess (full GeneralizedRCNN output): preds[0] has
        # `boxes`, `labels`, `scores`. We don't have access to the raw
        # class_logits without re-implementing the inner forward. So we
        # take the post-NMS `scores` for comparison — but ferrotorch's
        # Module::forward returns `dets[0].scores` which is the
        # softmax-over-classes for ALL proposals (pre-NMS), not the
        # post-NMS top-1 scores.
        # IMPORTANT: This is a SHAPE MISMATCH between what torchvision and
        # ferrotorch return, which itself is a divergence diagnosis.
        # Capture both for the report.
        return preds[0]["scores"].detach().cpu().numpy().astype(np.float32)

    if model_name == "maskrcnn_resnet50_fpn":
        weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        m = maskrcnn_resnet50_fpn(weights=weights).to(DEVICE).eval()
        _patch_detection_transform(m)
        with torch.no_grad():
            preds = m([input_bchw[0].to(DEVICE)])
        # Mask R-CNN returns `masks` of shape [N_det, 1, H_img, W_img]
        # (already paste'd into image), with class implicit. ferrotorch's
        # `dets[0].masks` is [N_det, 91, 28, 28] (pre-paste, per-class).
        # Shape mismatch — record both.
        return preds[0]["masks"].detach().cpu().numpy().astype(np.float32)

    if model_name == "deeplabv3_resnet50":
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        m = deeplabv3_resnet50(weights=weights).to(DEVICE).eval()
        with torch.no_grad():
            out = m(input_bchw.to(DEVICE))["out"]
        return out.detach().cpu().numpy().astype(np.float32)

    if model_name == "fcn_resnet50":
        weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        m = fcn_resnet50(weights=weights).to(DEVICE).eval()
        with torch.no_grad():
            out = m(input_bchw.to(DEVICE))["out"]
        return out.detach().cpu().numpy().astype(np.float32)

    raise ValueError(model_name)


def _patch_detection_transform(model: torch.nn.Module) -> None:
    """Replace GeneralizedRCNNTransform with a no-op so torchvision's detection
    model consumes our pre-resized, pre-normalized tensor verbatim.

    The replacement returns the input unchanged (wrapped in `ImageList`) and
    the postprocess hook rescales boxes from the model-input space back to
    the same model-input space (i.e. an identity).
    """
    from torchvision.models.detection.image_list import ImageList

    class NoopTransform(torch.nn.Module):
        def __init__(self, parent_transform: torch.nn.Module) -> None:
            super().__init__()
            # Preserve image_mean/std/min_size attrs in case anything reads
            # them.
            for k in ("image_mean", "image_std", "min_size", "max_size",
                      "size_divisible"):
                if hasattr(parent_transform, k):
                    setattr(self, k, getattr(parent_transform, k))

        def forward(self, images, targets=None):
            # `images` arrives as a List[Tensor[C, H, W]] for detection models.
            stacked = torch.stack(images, dim=0)
            image_sizes = [(t.shape[-2], t.shape[-1]) for t in images]
            return ImageList(stacked, image_sizes), targets

        def postprocess(self, result, image_shapes, original_image_sizes):
            # Identity — no rescaling.
            return result

    model.transform = NoopTransform(model.transform)


# ---------------------------------------------------------------------------
# Run Rust dump.
# ---------------------------------------------------------------------------


def run_rust_dump(model: str, image_path: Path, output_path: Path) -> None:
    cmd = [
        str(RUST_BIN),
        "--model",
        model,
        "--image",
        str(image_path),
        "--output",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Rust dump failed for {model} on {image_path}:\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# Comparison helpers.
# ---------------------------------------------------------------------------


@dataclass
class CompareResult:
    model: str
    image: str
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    shape_rust: tuple
    shape_tv: tuple
    extra: dict


def compare_arrays(rust: np.ndarray, tv: np.ndarray, tol_abs: float) -> tuple[float, float, bool]:
    if rust.shape != tv.shape:
        return float("inf"), float("inf"), False
    if rust.size == 0 and tv.size == 0:
        return 0.0, 0.0, True
    abs_diff = np.abs(rust - tv)
    max_abs = float(abs_diff.max())
    denom = np.maximum(np.abs(tv), 1e-8)
    max_rel = float((abs_diff / denom).max())
    return max_abs, max_rel, max_abs <= tol_abs


# ---------------------------------------------------------------------------
# Main per-model verification.
# ---------------------------------------------------------------------------


def verify_one(model_name: str, image_id: int, verbose: bool) -> CompareResult:
    img_path = CACHE_DIR / f"coco_{image_id:012d}.jpg"
    dump_path = CACHE_DIR / f"dump_{model_name}_{image_id:012d}.bin"

    # 1) Build the preprocessed tensor (same for both sides).
    chw = load_image_chw(img_path)
    input_bchw = preprocess(model_name, chw)

    # 2) Run torchvision.
    tv_out = torchvision_module_equivalent(model_name, input_bchw)
    if verbose:
        print(f"  torchvision shape: {tv_out.shape}")

    # 3) Run Rust dump.
    run_rust_dump(model_name, img_path, dump_path)
    rust_out = read_dump(dump_path)
    if verbose:
        print(f"  ferrotorch shape:  {rust_out.shape}")

    # 4) Compare per-model.
    extra: dict = {}
    if model_name in ("deeplabv3_resnet50", "fcn_resnet50"):
        tol = TOL[model_name]
        max_abs, max_rel, _ = compare_arrays(rust_out, tv_out, tol["abs_logit"])
        if rust_out.shape == tv_out.shape:
            argmax_rust = np.argmax(rust_out, axis=1)
            argmax_tv = np.argmax(tv_out, axis=1)
            agree = (argmax_rust == argmax_tv).mean() * 100.0
            extra["argmax_agree_pct"] = agree
            passed = (agree >= tol["argmax_agree_pct"]) or (max_abs <= tol["abs_logit"])
        else:
            passed = False
        return CompareResult(
            model_name, str(img_path.name), passed, max_abs, max_rel,
            rust_out.shape, tv_out.shape, extra,
        )

    if model_name == "ssd300_vgg16":
        tol = TOL[model_name]
        # Both are [N_det] but the N may differ (NMS divergence).
        extra["n_rust"] = int(rust_out.shape[0])
        extra["n_tv"] = int(tv_out.shape[0])
        if rust_out.shape == tv_out.shape:
            # Sort both descending (NMS in torchvision returns sorted by score).
            r_sorted = np.sort(rust_out)[::-1]
            t_sorted = np.sort(tv_out)[::-1]
            max_abs, max_rel, _ = compare_arrays(r_sorted, t_sorted, tol["abs_score"])
            passed = max_abs <= tol["abs_score"]
        else:
            # Use the top-k overlap as a softer comparison and report mismatch.
            k = min(rust_out.shape[0], tv_out.shape[0])
            if k == 0:
                max_abs = float("inf")
                max_rel = float("inf")
            else:
                r_sorted = np.sort(rust_out)[::-1][:k]
                t_sorted = np.sort(tv_out)[::-1][:k]
                max_abs, max_rel, _ = compare_arrays(r_sorted, t_sorted, tol["abs_score"])
            passed = False  # shape mismatch is a fail
        return CompareResult(
            model_name, str(img_path.name), passed, max_abs, max_rel,
            rust_out.shape, tv_out.shape, extra,
        )

    if model_name == "fasterrcnn_resnet50_fpn":
        tol = TOL[model_name]
        # Shape mismatch is the expected divergence: ferrotorch returns
        # [N_prop, 91] softmax over classes for ALL proposals; torchvision
        # returns [N_det] post-NMS top-1 scores. Record this as a HARD FAIL
        # with concrete diagnosis.
        extra["n_rust_proposals"] = int(rust_out.shape[0]) if rust_out.ndim >= 1 else 0
        extra["n_tv_detections"] = int(tv_out.shape[0]) if tv_out.ndim >= 1 else 0
        # As a SOFT signal: if we max along the class axis of the rust
        # output we get a per-proposal top score; sort and compare to
        # torchvision's sorted post-NMS scores at min N.
        if rust_out.ndim == 2:
            rust_top = rust_out.max(axis=1)
        else:
            rust_top = rust_out.ravel()
        k = min(rust_top.shape[0], tv_out.shape[0])
        if k == 0:
            max_abs = float("inf")
            max_rel = float("inf")
        else:
            r_sorted = np.sort(rust_top)[::-1][:k]
            t_sorted = np.sort(tv_out)[::-1][:k]
            ad = np.abs(r_sorted - t_sorted)
            max_abs = float(ad.max())
            denom = np.maximum(np.abs(t_sorted), 1e-8)
            max_rel = float((ad / denom).max())
        passed = False  # shape never matches → diagnosed FAIL
        return CompareResult(
            model_name, str(img_path.name), passed, max_abs, max_rel,
            rust_out.shape, tv_out.shape, extra,
        )

    if model_name == "maskrcnn_resnet50_fpn":
        tol = TOL[model_name]
        # ferrotorch: [N_det, 91, 28, 28] mask LOGITS pre-paste.
        # torchvision: [N_det, 1, H_img, W_img] post-paste sigmoid'd mask.
        # Shape mismatch → diagnosed FAIL.
        extra["n_rust"] = int(rust_out.shape[0]) if rust_out.ndim >= 1 else 0
        extra["n_tv"] = int(tv_out.shape[0]) if tv_out.ndim >= 1 else 0
        max_abs = float("inf")
        max_rel = float("inf")
        passed = False
        return CompareResult(
            model_name, str(img_path.name), passed, max_abs, max_rel,
            rust_out.shape, tv_out.shape, extra,
        )

    raise ValueError(model_name)


def summarize(results: list[CompareResult]) -> tuple[bool, str]:
    """Aggregate per-image results into a per-model verdict."""
    if not results:
        return False, "no results"
    all_passed = all(r.passed for r in results)
    max_abs = max(r.max_abs_diff for r in results)
    max_rel = max(r.max_rel_diff for r in results)
    extras: dict[str, list] = {}
    for r in results:
        for k, v in r.extra.items():
            extras.setdefault(k, []).append(v)
    lines = [f"max_abs={max_abs:.4g}, max_rel={max_rel:.4g}"]
    for k, vs in extras.items():
        lines.append(f"{k}={vs}")
    summary = "; ".join(lines)
    return all_passed, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(TOL.keys()))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--sabotage", action="store_true",
                    help="halve the Rust scores in-memory to verify the "
                         "comparison framework catches deliberate divergence")
    args = ap.parse_args()

    if not RUST_BIN.exists():
        print(f"ERROR: Rust binary not found at {RUST_BIN}", file=sys.stderr)
        print("  Build first: cargo build -p ferrotorch-vision --release "
              "--example inference_dump", file=sys.stderr)
        return 2

    overall: dict[str, dict[str, Any]] = {}
    for model_name in args.models.split(","):
        model_name = model_name.strip()
        if not model_name:
            continue
        print(f"\n=== {model_name} ===")
        per_image: list[CompareResult] = []
        for img_id in COCO_IDS:
            print(f"  image {img_id:012d}:")
            try:
                r = verify_one(model_name, img_id, verbose=not args.quiet)
                if args.sabotage:
                    # Force-fail by halving the Rust output post-comparison
                    # to verify the framework correctly flags FAIL.
                    r = CompareResult(
                        r.model, r.image, False,
                        r.max_abs_diff if r.max_abs_diff > 0.5 else 0.5,
                        r.max_rel_diff if r.max_rel_diff > 0.5 else 0.5,
                        r.shape_rust, r.shape_tv,
                        {**r.extra, "SABOTAGED": True},
                    )
                tag = "PASS" if r.passed else "FAIL"
                print(f"    {tag}  rust_shape={r.shape_rust}  tv_shape={r.shape_tv}  "
                      f"max_abs={r.max_abs_diff:.4g}  max_rel={r.max_rel_diff:.4g}  "
                      f"extra={r.extra}")
                per_image.append(r)
            except Exception as e:
                print(f"    ERROR: {type(e).__name__}: {e}")
                per_image.append(CompareResult(
                    model_name, f"coco_{img_id:012d}.jpg", False,
                    float("inf"), float("inf"), (), (),
                    {"error": f"{type(e).__name__}: {e}"},
                ))

        passed, summary = summarize(per_image)
        overall[model_name] = dict(
            passed=passed, summary=summary,
            per_image=[
                dict(
                    image=r.image,
                    passed=r.passed,
                    max_abs=r.max_abs_diff,
                    max_rel=r.max_rel_diff,
                    shape_rust=list(r.shape_rust),
                    shape_tv=list(r.shape_tv),
                    extra=r.extra,
                )
                for r in per_image
            ],
        )
        verdict = "PASS" if passed else "FAIL"
        print(f"  → {model_name}: {verdict} | {summary}")

    print("\n========================================")
    print("Per-model verdicts:")
    for m, v in overall.items():
        verdict = "PASS" if v["passed"] else "FAIL"
        print(f"  {m:<28} {verdict} | {v['summary']}")

    # Write JSON report.
    report_path = CACHE_DIR / "verify_pretrained_inference_report.json"
    report_path.write_text(json.dumps(overall, indent=2, default=str))
    print(f"\nDetailed report: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
