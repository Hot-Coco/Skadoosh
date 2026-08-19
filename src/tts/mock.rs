//! Sine-wave mock TTS: zero model downloads, deterministic durations — lets
//! the whole pipeline run (and be tested) with no Kokoro files.

use super::{TtsClip, TtsEngine, TTS_SAMPLE_RATE};
use crate::error::Result;

/// Sine frequency in Hz (a low, audible hum).
const FREQ_HZ: f32 = 220.0;
/// Milliseconds of audio per input character before clamping.
const MS_PER_CHAR: f32 = 55.0;
/// Duration clamp bounds, milliseconds.
const MIN_MS: f32 = 250.0;
const MAX_MS: f32 = 2_500.0;
/// Raised-cosine edge length, milliseconds (click avoidance).
const EDGE_MS: f32 = 5.0;
/// Peak amplitude (well below clipping).
const PEAK: f32 = 0.3;

/// 220 Hz sine mock: duration `clamp(chars × 55 ms, 250 ms..2.5 s)`, 5 ms
/// raised-cosine edges to avoid clicks, peak 0.3 (no clipping).
#[derive(Debug, Default, Clone, Copy)]
pub struct MockTts;

impl MockTts {
    /// Creates the mock engine.
    pub fn new() -> Self {
        Self
    }
}

impl TtsEngine for MockTts {
    fn synthesize(&mut self, text: &str) -> Result<TtsClip> {
        let chars = text.chars().count();
        let ms = (chars as f32 * MS_PER_CHAR).clamp(MIN_MS, MAX_MS);
        let n = (TTS_SAMPLE_RATE as f32 * ms / 1000.0).round() as usize;
        let edge = ((TTS_SAMPLE_RATE as f32 * EDGE_MS / 1000.0) as usize).min(n / 2);
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let phase = 2.0 * std::f32::consts::PI * FREQ_HZ * i as f32 / TTS_SAMPLE_RATE as f32;
            // Raised-cosine (Hann-style half windows) fade in/out.
            let gain = if edge > 0 && i < edge {
                0.5 * (1.0 - (std::f32::consts::PI * i as f32 / edge as f32).cos())
            } else if edge > 0 && i >= n - edge {
                let k = n - 1 - i;
                0.5 * (1.0 - (std::f32::consts::PI * k as f32 / edge as f32).cos())
            } else {
                1.0
            };
            samples.push(PEAK * gain * phase.sin());
        }
        Ok(TtsClip {
            samples,
            sample_rate: TTS_SAMPLE_RATE,
        })
    }
}
