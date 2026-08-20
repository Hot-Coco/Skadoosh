//! Acoustic echo cancellation (AEC) backed by aec_rs (speexdsp).
//!
//! [`AecFilter`] wraps the speexdsp echo canceller behind a simple,
//! allocation-light, f32 frame API so it can be dropped into the capture
//! pipeline later. speexdsp works on `i16` samples with a *fixed* frame size
//! chosen at initialization, so this module:
//!
//! - converts the public f32 frames to/from `i16` at the boundary, and
//! - lazily creates the inner canceller on the first
//!   [`AecFilter::process_frame`] call (the frame size is taken from the length
//!   of that frame), recreating it if a later frame has a different length.
//!
//! The filter is **not** wired into the live audio pipeline yet; this module
//! only provides the type and a clean API. Note for future wiring: the
//! underlying speexdsp state holds raw pointers, so [`AecFilter`] is neither
//! [`Send`] nor [`Sync`] — keep it on a single thread (e.g. the audio thread).

use aec_rs::{Aec, AecConfig};

/// Acoustic echo canceller wrapping speexdsp (via `aec_rs`).
///
/// Construct with [`AecFilter::new`], then feed pairs of frames to
/// [`AecFilter::process_frame`]: `mic_frame` is what the microphone captured
/// (near-end speech *plus* echo of the loudspeaker) and `speaker_frame` is the
/// reference signal sent to the loudspeaker. The returned frame is the
/// microphone signal with the estimated echo subtracted.
///
/// All samples are `f32` in the range `[-1.0, 1.0]`; conversion to speexdsp's
/// native `i16` happens internally. The speexdsp preprocessor (light noise
/// suppression) is enabled by default and runs on the cancelled output.
pub struct AecFilter {
    /// Sample rate in Hz shared by the mic and speaker streams.
    sample_rate: u32,
    /// Requested echo tail length in milliseconds (kept so the canceller can be
    /// recreated if the frame size changes).
    filter_length_ms: f32,
    /// Frame size, in samples, the inner canceller was created with. `0` until
    /// the first [`AecFilter::process_frame`] call sets it.
    frame_size: usize,
    /// The speexdsp-backed echo canceller. `None` until the first frame is
    /// processed, since the frame size is unknown until then.
    aec: Option<Aec>,
    /// Reusable i16 microphone buffer (avoids per-frame allocation).
    rec_buf: Vec<i16>,
    /// Reusable i16 speaker-reference buffer.
    echo_buf: Vec<i16>,
    /// Reusable i16 output buffer.
    out_buf: Vec<i16>,
}

impl AecFilter {
    /// Creates an echo canceller for `sample_rate` Hz with an echo tail of
    /// `filter_length_ms` milliseconds.
    ///
    /// The inner speexdsp state is created lazily on the first
    /// [`process_frame`](Self::process_frame) call, because speexdsp needs a
    /// fixed frame size and [`new`](Self::new) does not take one — the frame
    /// size is inferred from the first frame's length.
    ///
    /// `filter_length_ms` should cover the longest round-trip echo path (room
    /// reverb plus speaker/headset delay). It is rounded up to a whole multiple
    /// of the frame size when the canceller is created.
    pub fn new(sample_rate: usize, filter_length_ms: f32) -> Self {
        assert!(sample_rate > 0, "sample_rate must be non-zero");
        assert!(filter_length_ms > 0.0, "filter_length_ms must be positive");
        Self {
            sample_rate: sample_rate as u32,
            filter_length_ms,
            frame_size: 0,
            aec: None,
            rec_buf: Vec::new(),
            echo_buf: Vec::new(),
            out_buf: Vec::new(),
        }
    }

    /// Configured sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Frame size, in samples, the inner canceller is configured for. Returns
    /// `0` until the first frame has been processed.
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Processes one audio frame through the echo canceller, subtracting the
    /// estimated echo (modeled from `speaker_frame`) from `mic_frame`.
    ///
    /// `mic_frame` and `speaker_frame` should be the same length; if
    /// `speaker_frame` is shorter it is zero-padded, and if longer the extra
    /// samples are ignored. The returned [`Vec<f32>`] has one sample per
    /// `mic_frame` sample.
    ///
    /// Typical frame sizes are powers of two such as 256 or 512. The frame size
    /// is fixed once the canceller is created: changing it on a later call
    /// recreates the inner speexdsp state, which resets the adapted echo
    /// filter.
    pub fn process_frame(&mut self, mic_frame: &[f32], speaker_frame: &[f32]) -> Vec<f32> {
        let frame_size = mic_frame.len();
        if frame_size == 0 {
            return Vec::new();
        }

        // (Re)create the speexdsp state when first needed or when the frame
        // size changes. speexdsp fixes the frame size at init, so a size change
        // requires a fresh state (this resets the adapted echo filter).
        if self.aec.is_none() || self.frame_size != frame_size {
            let filter_samples = (self.sample_rate as f32 * self.filter_length_ms / 1000.0)
                .round()
                .max(1.0) as i32;
            let fs = frame_size as i32;
            // speexdsp requires filter_length to be a multiple of the frame
            // size; round up (and ensure it is at least one frame).
            let filter_length = ((filter_samples + fs - 1) / fs) * fs;

            let config = AecConfig {
                frame_size,
                filter_length,
                sample_rate: self.sample_rate,
                enable_preprocess: true,
            };
            self.aec = Some(Aec::new(&config));
            self.frame_size = frame_size;
            self.rec_buf.resize(frame_size, 0);
            self.echo_buf.resize(frame_size, 0);
            self.out_buf.resize(frame_size, 0);
        }

        // f32 -> i16 at the speexdsp boundary.
        for (i, &s) in mic_frame.iter().enumerate() {
            self.rec_buf[i] = f32_to_i16(s);
        }
        // Speaker reference: copy the overlapping part, zero-fill the rest.
        let n = speaker_frame.len().min(frame_size);
        for (i, &s) in speaker_frame[..n].iter().enumerate() {
            self.echo_buf[i] = f32_to_i16(s);
        }
        for slot in &mut self.echo_buf[n..] {
            *slot = 0;
        }

        let aec = self.aec.as_ref().expect("aec initialized above");
        aec.cancel_echo(&self.rec_buf, &self.echo_buf, &mut self.out_buf);

        // i16 -> f32 back to the pipeline's native format.
        self.out_buf.iter().map(|&s| i16_to_f32(s)).collect()
    }
}

/// Converts a normalized f32 sample (`[-1.0, 1.0]`) to a signed 16-bit sample.
fn f32_to_i16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

/// Converts a signed 16-bit sample to a normalized f32 sample.
fn i16_to_f32(v: i16) -> f32 {
    v as f32 / 32767.0
}
