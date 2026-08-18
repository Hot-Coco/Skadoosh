//! Skadoosh — a modular, low-latency local voice agent framework.
//!
//! Pipeline: cpal mic capture (16 kHz mono) → Silero VAD ([`vad`]) →
//! whisper-rs STT ([`stt`]) → streaming LLM over an OpenAI-compatible API
//! ([`llm`], Ollama by default) → clause-split → ONNX TTS ([`tts`], Kokoro-82M
//! with a sine-wave mock fallback) → cpal playback with barge-in ([`audio`]).
//! The [`pipeline`] orchestrator spawns and supervises every stage.
//!
//! All internal audio is `f32` samples: 16 kHz on the capture/VAD/STT side,
//! 24 kHz out of TTS, resampled at the device edges ([`audio::resample`]).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod audio;
pub mod config;
pub mod error;
pub mod llm;
pub mod pipeline;
pub mod stt;
pub mod tts;
pub mod vad;

pub use config::Config;
pub use error::{Result, SkadooshError};
pub use pipeline::Pipeline;
pub use tts::TtsEngine;
