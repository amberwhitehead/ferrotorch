# ferrotorch-whisper — `audio` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (numerical foundations only; the
    feature extractor is HF's transformers.WhisperFeatureExtractor
    composed on top of transformers.audio_utils.spectrogram in
    huggingface/transformers/src/transformers/audio_utils.py)
-->

## Summary

`ferrotorch-whisper/src/audio.rs` converts 16 kHz mono `f32` PCM audio
into the `[1, 80, 3000]` log-mel spectrogram that the Whisper encoder
consumes. Direct port of `transformers.WhisperFeatureExtractor`'s
NumPy path: pad/trim to 30 s, reflect-pad by `n_fft / 2`, periodic
Hann-windowed STFT (`n_fft=400`, `hop=160`), magnitude-squared,
80-mel filter bank multiplication, `log10` + clip-to-max-minus-8 +
`(x+4)/4` normalisation.

## Requirements

- REQ-1: Constants — `SAMPLE_RATE=16000`, `CHUNK_LENGTH=30`,
  `N_FFT=400`, `HOP_LENGTH=160`, `N_MELS=80`, `N_FRAMES=3000`,
  `N_SAMPLES=480_000`, `N_FREQS=201` — each `pub const` exposing
  the reference values so callers and pin scripts can cross-check
  against the upstream.
- REQ-2: `log_mel_spectrogram(audio: &[f32]) -> Tensor<f32>`
  produces a `[1, 80, 3000]` tensor. Rejects empty input with
  `InvalidArgument`; pads shorter input with zeros; truncates longer
  input.
- REQ-3: Pre-shipped `assets/mel_filters_80x201.bin` is the
  byte-for-byte mel filter bank matching
  `WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny").mel_filters.T`.
  Decoded at runtime as `[80, 201]` row-major little-endian f32.
- REQ-4: Periodic Hann window — `0.5 - 0.5 * cos(2π k / N_FFT)` for
  `k = 0..N_FFT` (matches `np.hanning(N_FFT+1)[:-1]`).
- REQ-5: Reflect-pad by `N_FFT / 2 = 200` on each side (matches
  `np.pad(audio, ((pad, pad),), mode="reflect")` where the boundary
  sample is the axis, not duplicated).
- REQ-6: STFT — one rfft frame at a time, naive `O(N²)` DFT (length
  400, not a power of 2). The 3001 × O(N²) loop is acceptable for
  one-shot correctness; a future optimisation can swap to `rustfft`.
- REQ-7: Magnitude-squared (`power=2.0`) → drop the last frame to
  match `log_spec[:, :-1]` in the reference → mel filter bank
  multiplication with floor `mel_floor = 1e-10` BEFORE the log
  (matches `np.maximum(mel_floor, ...)` in
  `transformers.audio_utils.spectrogram`).
- REQ-8: `log10` → `clip(x, x.max() - 8.0)` → `(x + 4.0) / 4.0`
  (the canonical Whisper normalisation). f64 accumulation; downcast
  to f32 at the very end.

## Acceptance Criteria

- [x] AC-1: `log_mel_spectrogram(&silence)` for `silence = [0; 480000]`
  returns a `[1, 80, 3000]` tensor with finite values.
- [x] AC-2: `log_mel_spectrogram(&[])` returns `InvalidArgument`.
- [x] AC-3: `hann_window()` has length `N_FFT`, `[0]` is exactly
  zero, last sample is strictly between 0 and 1.
- [x] AC-4: `reflect_pad(&[1, 2, 3, 4, 5], 2)` returns
  `[3, 2, 1, 2, 3, 4, 5, 4, 3]`.
- [x] AC-5: `pad_or_trim` pads short audio with zeros; truncates
  long audio.
- [x] AC-6: `mel_filters()` decodes the asset to a non-trivial
  (positive sum) `[80, 201]` filter bank with all-finite values.

## Architecture

`pub fn log_mel_spectrogram` in `audio.rs` is the public entry
point. The pipeline runs entirely in `f64` then downcasts to `f32`
at the end:

1. `pad_or_trim` — produce a `[N_SAMPLES]` `Vec<f64>`.
2. `reflect_pad` — extend by `N_FFT / 2 = 200` samples on each side.
3. `hann_window` — build the periodic Hann window once.
4. Frame loop — for each of 3001 frames, windowed rfft via
   `stft_one_frame` (naive `O(N²)`), then magnitude-squared into
   a `[frames, freqs]` row-major buffer.
5. Transpose `[frames, freqs] → [freqs, frames]` and drop the last
   frame (`power[k * N_FRAMES + t] = power_t[t * N_FREQS + k]`).
6. Mel filter bank multiplication `[80, 201] @ [201, 3000] → [80, 3000]`
   with `mel_floor = 1e-10` applied per-cell BEFORE the log.
7. `log10` → `clip` → `(x + 4) / 4` normalisation.
8. Downcast f64 → f32; promote to `[1, 80, 3000]` via
   `Tensor::from_storage`.

`MEL_FILTERS_BYTES` at the top of `audio.rs` is the embedded asset
(`include_bytes!`). `mel_filters()` decodes it via
`f32::from_le_bytes` on each 4-byte chunk.

`stft_one_frame` in `audio.rs` computes the rfft naively. The
comment documents the future-rustfft swap; the naive variant is
correctness-first.

### Non-test production consumers

- `pub use audio::{N_FRAMES, N_MELS, SAMPLE_RATE,
  log_mel_spectrogram}` at `ferrotorch-whisper/src/lib.rs:106`.
- `log_mel_spectrogram` is the canonical public API for feeding raw
  audio into `WhisperEncoder::forward_from_mel`; it is the
  ferrotorch-whisper API surface consumed by speech-to-text pin
  scripts.

## Parity contract

`parity_ops = []`. The pipeline composes naive DFT primitives that
have no parity-sweep counterpart; the asset is the byte-exact mel
filter bank so any drift is in the STFT/log/normalize pipeline.

Numerical / structural edge cases preserved:

- **Silence input.** All-zero audio produces all-`-3.25` mel values
  (the floor `1e-10` survives `log10` then the clip-to-max-minus-8
  + normalisation collapse). Finite, never NaN.
- **`mel_floor = 1e-10` applied BEFORE log.** Matches HF
  `np.maximum(mel_floor, ...)`. Applying floor AFTER log would
  produce `-Inf` for zero inputs.
- **Frame count = `1 + (N_SAMPLES + 2*pad - N_FFT) / HOP = 3001`,
  truncated to 3000 by dropping the last frame.** Matches the
  reference `log_spec[:, :-1]`.
- **Periodic Hann window.** `0.5 - 0.5 * cos(2π k / N_FFT)` for
  `k = 0..N_FFT` (NOT `k = 0..N_FFT-1` which would be the symmetric
  variant). First sample is exactly 0; last sample is strictly less
  than 1.

## Verification

Tests in `mod tests in audio.rs`:

- `hann_periodic_first_last`
- `mel_filter_shape_and_norm_finite`
- `reflect_pad_basic`
- `pad_or_trim_pads_short`
- `pad_or_trim_trims_long`
- `log_mel_shape_is_1_80_3000`
- `log_mel_rejects_empty`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-whisper --lib audio:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub const SAMPLE_RATE: u32 = 16_000;` (and siblings) at the top of `audio.rs`; non-test consumer: `pub use` at `ferrotorch-whisper/src/lib.rs:106` re-exports `N_FRAMES`, `N_MELS`, `SAMPLE_RATE`. |
| REQ-2 | SHIPPED | impl: `pub fn log_mel_spectrogram` in `audio.rs`; non-test consumer: `pub use` at `audio in ferrotorch-whisper/src/lib.rs` (the canonical public API for the speech-to-text pipeline). |
| REQ-3 | SHIPPED | impl: `MEL_FILTERS_BYTES` `include_bytes!` + `mel_filters` decoder in `audio.rs`; non-test consumer: `log_mel_spectrogram` in `audio.rs` calls `mel_filters()`. |
| REQ-4 | SHIPPED | impl: `hann_window` in `audio.rs`; non-test consumer: `log_mel_spectrogram` in `audio.rs` calls `hann_window()`. |
| REQ-5 | SHIPPED | impl: `reflect_pad` in `audio.rs`; non-test consumer: `log_mel_spectrogram` in `audio.rs` calls `reflect_pad(&padded, N_FFT / 2)`. |
| REQ-6 | SHIPPED | impl: `stft_one_frame` in `audio.rs`; non-test consumer: `log_mel_spectrogram` in `audio.rs` calls it once per frame in the 0..num_frames loop. |
| REQ-7 | SHIPPED | impl: magnitude-squared + transpose + mel-bank loops inside `log_mel_spectrogram` in `audio.rs`; non-test consumer: same call path — `log_mel_spectrogram` is the only entry into this pipeline and is re-exported via the `pub use` at `ferrotorch-whisper/src/lib.rs:106`. |
| REQ-8 | SHIPPED | impl: `log10` → `clip(x, max - 8)` → `(x + 4) / 4` normalisation block inside `log_mel_spectrogram` in `audio.rs`; non-test consumer: same call path as REQ-7. |
