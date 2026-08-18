//! Speech-to-text via whisper-rs on a dedicated blocking std thread, so the
//! synchronous whisper.cpp API stays off the async runtime.

pub mod whisper;

pub use whisper::{SttConfig, WhisperStt};
