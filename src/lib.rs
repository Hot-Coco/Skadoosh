//! Skadoosh — a modular, low-latency local voice agent framework.
//!
//! Pipeline: cpal mic capture (16 kHz mono) → Silero VAD (`vad`) →
//! whisper-rs STT (`stt`) → streaming LLM over an OpenAI-compatible API
//! ([`llm`], Ollama by default) → clause-split → ONNX TTS ([`tts`], Kokoro-82M
//! with a sine-wave mock fallback) → cpal playback with barge-in ([`audio`]).
//! The `pipeline` orchestrator spawns and supervises every stage. The VAD,
//! whisper STT, cpal I/O, echo-cancellation, and orchestrator stages all live
//! behind the `audio` feature (on by default); a `--no-default-features` build
//! drops them and the system audio libraries they need.
//!
//! All internal audio is `f32` samples: 16 kHz on the capture/VAD/STT side,
//! 24 kHz out of TTS, resampled at the device edges ([`audio::resample`]).
//!
//! # SDK
//!
//! [`Agent`] (see [`agent`]) is the embedding facade: build it from a
//! [`Config`] plus optional engine trait objects
//! ([`stt::SttEngine`], [`llm::LlmBackend`], [`tts::TtsEngine`]), subscribe
//! to [`AgentEvent`]s, and drive the full audio loop (`Agent::run`, behind the
//! `audio` feature), single text turns ([`Agent::text_turn`], [`Agent::repl`]),
//! or one-shot speech synthesis (`Agent::say`, behind `audio`;
//! [`Agent::say_to_wav`]).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod agent;
pub mod audio;
pub mod config;
pub mod error;
pub mod forward;
pub(crate) mod gpu;
pub mod llm;
pub mod memory;
pub mod mesh;
#[cfg(feature = "audio")]
pub mod pipeline;
pub mod plugins;
pub mod rag;
pub mod sandbox;
pub mod stt;
pub mod tools;
pub mod tts;
#[cfg(feature = "audio")]
pub mod vad;
pub mod watch;
pub(crate) mod wav;

pub use agent::{Agent, AgentBuilder, AgentEvent};
pub use config::{Config, OutputMode};
pub use error::{Result, SkadooshError};
pub use memory::MemoryStore;
#[cfg(feature = "audio")]
pub use pipeline::{Pipeline, SelftestReport};
pub use plugins::{LoadedPlugin, PluginManager, PluginManifest};
pub use tts::TtsEngine;
