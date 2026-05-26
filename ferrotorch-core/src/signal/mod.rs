//! Signal-processing utilities.
//!
//! Mirrors `torch.signal.*`. Currently exposes the [`windows`] submodule;
//! future work may add filter design, convolution helpers, and other
//! `scipy.signal`-shaped primitives.
//!
//! ## REQ status (per `.design/ferrotorch-core/signal/mod.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl `pub mod windows`; non-test consumer downstream callers reach `ferrotorch_core::signal::windows::hann`. |
//! | REQ-2 | SHIPPED | impl `pub use windows::{...}` 15 names; non-test consumer test at `signal/windows.rs:343-366` exercises all 15 via re-export. |

pub mod windows;

pub use windows::{
    bartlett, blackman, cosine, exponential, gaussian, general_cosine, general_hamming, hamming,
    hann, hanning, kaiser, nuttall, parzen, taylor, tukey,
};
