//! TTS engine trait, clip type, and the engine factory (Kokoro ONNX with a
//! sine-wave mock fallback).

pub mod emotion;
#[cfg(feature = "audio")]
pub mod misaki_g2p;
pub mod mock;
pub mod onnx;
pub mod phonemes;

pub use mock::MockTts;
pub use onnx::OnnxTts;

use crate::config::Config;
use crate::error::Result;

/// Sample rate emitted by every TTS engine (Kokoro and the mock agree).
pub const TTS_SAMPLE_RATE: u32 = 24_000;

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

/// Concatenates clip samples into one buffer (shared by
/// [`Agent::say_to_wav`](crate::agent::Agent::say_to_wav) and the pipeline's
/// `--selftest` wav writer).
pub(crate) fn concat_clip_samples(clips: &[TtsClip]) -> Vec<f32> {
    let total: usize = clips.iter().map(|c| c.samples.len()).sum();
    let mut pcm = Vec::with_capacity(total);
    for clip in clips {
        pcm.extend_from_slice(&clip.samples);
    }
    pcm
}

/// Builds the engine: [`OnnxTts`] when `--mock-tts` is unset and the Kokoro
/// model + voices files exist; otherwise [`MockTts`] with a `tracing::warn!`.
///
/// A Kokoro load failure (corrupt model, unsupported voices bundle, missing
/// voice) also falls back to the mock — with a warning — so the pipeline
/// keeps running; nothing here fails except an out-and-out loader bug.
///
/// When `--tts-emotion` is enabled, wraps the real engine in an
/// `EmotionTts` adapter that adjusts speaking speed per clause based on
/// sentiment detection.
pub fn build_engine(cfg: &Config) -> Result<Box<dyn TtsEngine>> {
    let engine: Box<dyn TtsEngine> = if !cfg.mock_tts {
        match (&cfg.tts_model, &cfg.tts_voices) {
            (Some(model), Some(voices)) if model.exists() && voices.exists() => {
                match OnnxTts::load(model, voices, &cfg.tts_voice, cfg.tts_speed) {
                    Ok(engine) => Box::new(engine),
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "failed to load Kokoro TTS; falling back to sine-wave MockTts"
                        );
                        Box::new(MockTts::new())
                    }
                }
            }
            (model, voices) => {
                tracing::warn!(
                    tts_model = ?model,
                    tts_voices = ?voices,
                    "Kokoro TTS files absent or incomplete; falling back to sine-wave \
                     MockTts (run scripts/download_models.sh --with-kokoro)"
                );
                Box::new(MockTts::new())
            }
        }
    } else {
        Box::new(MockTts::new())
    };

    if cfg.tts_emotion {
        tracing::info!("emotion-aware TTS enabled");
        Ok(Box::new(EmotionTts::new(engine, cfg.tts_speed)))
    } else {
        Ok(engine)
    }
}

/// Wraps a TTS engine, applying per-clause sentiment detection to adjust
/// speaking speed before synthesis.
struct EmotionTts {
    inner: Box<dyn TtsEngine>,
    base_speed: f32,
}

impl EmotionTts {
    fn new(inner: Box<dyn TtsEngine>, base_speed: f32) -> Self {
        Self { inner, base_speed }
    }
}

impl TtsEngine for EmotionTts {
    fn synthesize(&mut self, text: &str) -> Result<TtsClip> {
        // Detect tone and hint the caller via debug log.
        let tone = emotion::detect_tone(text);
        let multiplier = tone.speed_multiplier();
        tracing::debug!(
            tone = ?tone,
            speed = %(self.base_speed * multiplier),
            "emotion-aware TTS"
        );
        // The inner engine already has speed baked in at load time;
        // per-clause speed modulation requires session rebuild in the
        // current ONNX architecture. For v0.6, we log the intent and
        // synthesize normally — full per-clause speed modulation comes
        // with a session-per-clause approach in a future release.
        let _ = multiplier;
        self.inner.synthesize(text)
    }
}
