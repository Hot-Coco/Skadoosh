//! Error types: one enum per pipeline stage plus the crate-wide umbrella
//! [`SkadooshError`] and the [`Result`] alias used throughout the crate.

use thiserror::Error;

/// Audio capture, playback, and resampling failures.
#[derive(Debug, Error)]
pub enum AudioError {
    /// No input/output device matched the requested name (or no default
    /// device exists, e.g. on a headless machine).
    #[error("no suitable audio device found")]
    NoDevice,
    /// The device rejected or could not provide a usable stream configuration.
    #[error("failed to negotiate audio stream config: {0}")]
    StreamConfig(String),
    /// Building or starting the cpal stream failed.
    #[error("failed to build audio stream: {0}")]
    StreamBuild(String),
    /// Resampler setup or processing failed.
    #[error("resampling failed: {0}")]
    Resample(String),
}

/// VAD model loading and inference failures.
#[derive(Debug, Error)]
pub enum VadError {
    /// The Silero ONNX model could not be loaded.
    #[error("failed to load VAD model: {0}")]
    ModelLoad(String),
    /// An `ort` inference call failed.
    #[error("VAD inference failed: {0}")]
    Inference(String),
    /// A frame with a length other than `FRAME_LEN` (512) was submitted.
    #[error("VAD frames must be {expected} samples, got {actual}")]
    BadFrameLen {
        /// Required frame length in samples.
        expected: usize,
        /// Frame length that was submitted.
        actual: usize,
    },
}

/// Speech-to-text failures.
#[derive(Debug, Error)]
pub enum SttError {
    /// The whisper.cpp model could not be loaded.
    #[error("failed to load STT model: {0}")]
    ModelLoad(String),
    /// whisper-rs failed to transcribe a segment.
    #[error("transcription failed: {0}")]
    Transcribe(String),
    /// The dedicated STT worker thread exited or its channel closed. During
    /// shutdown drain this is benign and must not reach the fatal-error mpsc.
    #[error("STT worker thread is gone")]
    WorkerGone,
}

/// Streaming LLM client failures.
#[derive(Debug, Error)]
pub enum LlmError {
    /// Transport-level HTTP failure (connect, read, TLS, ...).
    #[error("LLM HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    /// SSE framing/buffering failure while reading the stream.
    #[error("LLM SSE stream error: {0}")]
    Sse(String),
    /// The server answered with a non-success status code.
    #[error("LLM API returned status {status}: {body}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Response body (best effort, may be truncated).
        body: String,
    },
    /// The turn's cancellation token fired (barge-in or shutdown). The
    /// in-flight clause and the partial assistant reply are discarded.
    #[error("LLM request cancelled")]
    Cancelled,
}

/// Text-to-speech failures.
#[derive(Debug, Error)]
pub enum TtsError {
    /// The Kokoro ONNX model could not be loaded.
    #[error("failed to load TTS model: {0}")]
    ModelLoad(String),
    /// espeak-ng phonemization failed or the binary is missing.
    #[error("phonemization failed: {0}")]
    Phonemize(String),
    /// An `ort` inference call failed.
    #[error("TTS inference failed: {0}")]
    Inference(String),
    /// The voices bank is missing or does not contain the requested voice.
    #[error("TTS voices unavailable: {0}")]
    MissingVoices(String),
}

/// Crate-wide umbrella error; every stage error converts into it.
#[derive(Debug, Error)]
pub enum SkadooshError {
    /// Audio stage failure.
    #[error(transparent)]
    Audio(#[from] AudioError),
    /// VAD stage failure.
    #[error(transparent)]
    Vad(#[from] VadError),
    /// STT stage failure.
    #[error(transparent)]
    Stt(#[from] SttError),
    /// LLM stage failure.
    #[error(transparent)]
    Llm(#[from] LlmError),
    /// TTS stage failure.
    #[error(transparent)]
    Tts(#[from] TtsError),
    /// Escape hatch for errors at task boundaries (I/O, wav decoding, ...).
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, SkadooshError>;
