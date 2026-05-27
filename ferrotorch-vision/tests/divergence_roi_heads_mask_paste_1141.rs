//! Byte-level audit of `postprocess_masks(paste=true)` against torchvision
//! `paste_masks_in_image` (roi_heads.py:486) + `paste_mask_in_image` (:415)
//! (live torchvision 0.26.0+cu130).
//!
//! The happy-path test only checks shape + a single in-box pixel > 0.9. This
//! exercises the full geometry: expand_masks pad-1, expand_boxes scale,
//! int64 truncation, bilinear (align_corners=False) resize, crop offsets.
//!
//! Frozen oracle (`/tmp/roi_paste_oracle.py`): a 4x4 gradient mask, label=1,
//! box=[2,3,9,11], image 16x20. torchvision pasted-pixel values:
//! ```text
//! (y=7,x=5) = 0.459438   (y=8,x=6) = 0.668857   (y=6,x=4) = 0.255553
//! (y=10,x=8)= 0.787609   (y=0,x=0) = 0.0        (y=15,x=19)= 0.0
//! nonzero count = 99
//! ```
//! Pre-paste selected mask[0,0] (sigmoid of the channel-1 gradient):
//! ```text
//! [[0.047426,0.067547,0.095349,0.132964],
//!  [0.182426,0.245085,0.320821,0.407333],
//!  [0.5,     0.592667,0.679179,0.754915],
//!  [0.817574,0.867036,0.904651,0.932453]]
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::excessive_precision
)]

use ferrotorch_core::from_slice;
use ferrotorch_vision::models::detection::roi_heads_postprocess::postprocess_masks;

/// Divergence probe: full mask paste-back geometry + bilinear resize.
/// Compares specific pasted pixels against torchvision `paste_masks_in_image`.
#[test]
fn divergence_mask_paste_byte_level_vs_torchvision() {
    let m = 4usize;
    // [1, 2, 4, 4]: channel 0 = -3 everywhere, channel 1 = gradient (r*M+c)/16*6-3.
    let mut logits = vec![-3.0_f32; 1 * 2 * m * m];
    let plane = m * m;
    for r in 0..m {
        for c in 0..m {
            logits[plane + r * m + c] = (r * m + c) as f32 / (m * m) as f32 * 6.0 - 3.0;
        }
    }
    let mask_logits = from_slice::<f32>(&logits, &[1, 2, 4, 4]).unwrap();
    let labels = vec![1usize];
    let boxes = from_slice::<f32>(&[2.0, 3.0, 9.0, 11.0], &[1, 4]).unwrap();
    let im = [16usize, 20usize]; // (H, W)

    let pasted = postprocess_masks::<f32>(&mask_logits, &labels, &boxes, im, true).unwrap();
    assert_eq!(pasted.shape(), &[1, 1, 16, 20]);
    let d = pasted.data_vec().unwrap();
    let w = 20usize;
    let at = |y: usize, x: usize| d[y * w + x];

    // torchvision oracle values (align_corners=False bilinear).
    assert!(
        (at(7, 5) - 0.459438).abs() < 2e-3,
        "(7,5) tv=0.459438 got {}",
        at(7, 5)
    );
    assert!(
        (at(8, 6) - 0.668857).abs() < 2e-3,
        "(8,6) tv=0.668857 got {}",
        at(8, 6)
    );
    assert!(
        (at(6, 4) - 0.255553).abs() < 2e-3,
        "(6,4) tv=0.255553 got {}",
        at(6, 4)
    );
    assert!(
        (at(10, 8) - 0.787609).abs() < 2e-3,
        "(10,8) tv=0.787609 got {}",
        at(10, 8)
    );
    assert_eq!(at(0, 0), 0.0, "(0,0) outside box must be 0");
    assert_eq!(at(15, 19), 0.0, "(15,19) outside box must be 0");

    // nonzero count must match torchvision's 99.
    let nonzero = d.iter().filter(|&&v| v > 0.0).count();
    assert_eq!(nonzero, 99, "nonzero pasted pixels: tv=99 got {}", nonzero);
}
