//! Acoustic echo cancellation via `aec_rs` (speexdsp).

use aec_rs::{Aec, AecConfig};

/// Wraps speexdsp echo cancellation with a simple f32 frame API.
pub struct AecFilter {
    sample_rate: u32,
    filter_length_ms: f32,
    frame_size: usize,
    aec: Option<Aec>,
    rec_buf: Vec<i16>,
    echo_buf: Vec<i16>,
    out_buf: Vec<i16>,
}

impl AecFilter {
    /// Creates an echo canceller for `sample_rate` Hz with echo tail `filter_length_ms` ms.
    pub fn new(sample_rate: usize, filter_length_ms: f32) -> Self {
        assert!(sample_rate > 0);
        assert!(filter_length_ms > 0.0);
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

    /// Frame size the inner canceller was created with (0 before first frame).
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Cancel echo: subtracts estimated echo (from `speaker_frame`) from `mic_frame`.
    pub fn process_frame(&mut self, mic_frame: &[f32], speaker_frame: &[f32]) -> Vec<f32> {
        let frame_size = mic_frame.len();
        if frame_size == 0 {
            return Vec::new();
        }

        if self.aec.is_none() || self.frame_size != frame_size {
            let filter_samples = (self.sample_rate as f32 * self.filter_length_ms / 1000.0)
                .round()
                .max(1.0) as i32;
            let fs = frame_size as i32;
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

        for (i, &s) in mic_frame.iter().enumerate() {
            self.rec_buf[i] = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        }
        let n = speaker_frame.len().min(frame_size);
        for (i, &s) in speaker_frame[..n].iter().enumerate() {
            self.echo_buf[i] = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        }
        for slot in &mut self.echo_buf[n..] {
            *slot = 0;
        }

        let aec = self.aec.as_ref().expect("initialized above");
        aec.cancel_echo(&self.rec_buf, &self.echo_buf, &mut self.out_buf);
        self.out_buf.iter().map(|&s| s as f32 / 32767.0).collect()
    }
}
