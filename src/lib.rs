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
//!
//! # SDK
//!
//! [`Agent`] (see [`agent`]) is the embedding facade: build it from a
//! [`Config`] plus optional engine trait objects
//! ([`stt::SttEngine`], [`llm::LlmBackend`], [`tts::TtsEngine`]), subscribe
//! to [`AgentEvent`]s, and drive the full audio loop ([`Agent::run`]),
//! single text turns ([`Agent::text_turn`], [`Agent::repl`]), or one-shot
//! speech synthesis ([`Agent::say`], [`Agent::say_to_wav`]).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod agent;
pub mod audio;
pub mod config;
pub mod error;
pub mod llm;
pub mod pipeline;
pub mod stt;
pub mod tts;
pub mod vad;

pub use agent::{Agent, AgentBuilder, AgentEvent};
pub use config::{Config, OutputMode};
pub use error::{Result, SkadooshError};
pub use pipeline::{Pipeline, SelftestReport};
pub use tts::TtsEngine;
