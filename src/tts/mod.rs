//! TTS engine trait, clip type, and the engine factory (Kokoro ONNX with a
//! sine-wave mock fallback).

pub mod mock;
pub mod onnx;
pub mod phonemes;

pub use mock::MockTts;
pub use onnx::OnnxTts;

use crate::config::Config;
use crate::error::Result;

/// A synthesized audio clip.
#[derive(Debug, Clone)]
pub struct TtsClip {
    /// PCM samples, f32 mono.
    pub samples: Vec<f32>,
    /// Sample rate in Hz (24 000 for both Kokoro and the mock).
    pub sample_rate: u32,
}

/// Synchronous TTS engine. Sync by design: the TTS task runs it via
/// `spawn_blocking` per clause (or a pinned thread like STT if profiling
/// shows pool churn).
pub trait TtsEngine: Send {
    /// Synthesizes `text` into one clip.
    fn synthesize(&mut self, text: &str) -> Result<TtsClip>;
}

/// Builds the engine: [`OnnxTts`] when `--mock-tts` is unset and the Kokoro
/// model + voices files exist; otherwise [`MockTts`] with a `tracing::warn!`.
pub fn build_engine(cfg: &Config) -> Result<Box<dyn TtsEngine>> {
    let _ = cfg;
    todo!("task 5.1: onnx → mock fallback factory")
}
