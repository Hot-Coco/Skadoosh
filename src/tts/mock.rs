//! Sine-wave mock TTS: zero model downloads, deterministic durations — lets
//! the whole pipeline run (and be tested) with no Kokoro files.

use super::{TtsClip, TtsEngine};
use crate::error::Result;

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
        let _ = text;
        todo!("task 5.1: sine clip with raised-cosine edges")
    }
}
