//! CLI configuration: clap derive with `SKADOOSH_*` env fallbacks.

use std::path::PathBuf;

use clap::Parser;

use crate::error::Result;

/// Default system prompt seeded into the LLM conversation history.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a voice assistant. Reply in short, \
     spoken-style sentences. Keep answers under three sentences unless asked for detail.";

/// Command-line configuration for `skadoosh`.
///
/// Every flag falls back to a `SKADOOSH_*` environment variable; flags win
/// over env vars, env vars win over defaults.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "skadoosh",
    version,
    about = "Modular, low-latency local voice agent framework",
    long_about = "Skadoosh runs a continuous-listening local voice agent: \
                  mic → Silero VAD → Whisper STT → streaming LLM → ONNX TTS → playback, \
                  with barge-in."
)]
pub struct Config {
    /// Base URL of the OpenAI-compatible LLM API.
    #[arg(
        long,
        env = "SKADOOSH_LLM_URL",
        default_value = "http://localhost:11434/v1"
    )]
    pub llm_url: String,

    /// Model name passed to the chat-completions API.
    #[arg(long, env = "SKADOOSH_LLM_MODEL", default_value = "qwen2.5:0.5b")]
    pub llm_model: String,

    /// System prompt seeded into the LLM conversation history.
    #[arg(long, env = "SKADOOSH_SYSTEM_PROMPT", default_value = DEFAULT_SYSTEM_PROMPT)]
    pub system_prompt: String,

    /// Maximum number of trailing user/assistant turns kept in LLM history.
    #[arg(long, env = "SKADOOSH_MAX_HISTORY_TURNS", default_value_t = 8)]
    pub max_history_turns: usize,

    /// Path to the whisper.cpp ggml model.
    #[arg(
        long,
        env = "SKADOOSH_WHISPER_MODEL",
        default_value = "models/ggml-tiny.en.bin"
    )]
    pub whisper_model: PathBuf,

    /// Path to the Silero VAD ONNX model.
    #[arg(
        long,
        env = "SKADOOSH_VAD_MODEL",
        default_value = "models/silero_vad.onnx"
    )]
    pub vad_model: PathBuf,

    /// Path to the Kokoro TTS ONNX model. When absent (or the file is
    /// missing) the pipeline falls back to the sine-wave MockTts.
    #[arg(long, env = "SKADOOSH_TTS_MODEL")]
    pub tts_model: Option<PathBuf>,

    /// Path to the Kokoro voices bank (`voices.bin`). Required together with
    /// `--tts-model` for the real TTS engine.
    #[arg(long, env = "SKADOOSH_TTS_VOICES")]
    pub tts_voices: Option<PathBuf>,

    /// Speech probability threshold for the VAD, in `[0, 1)`.
    #[arg(long, env = "SKADOOSH_VAD_THRESHOLD", default_value_t = 0.5)]
    pub vad_threshold: f32,

    /// Trailing silence (milliseconds) that closes a speech segment.
    #[arg(long, env = "SKADOOSH_SILENCE_MS", default_value_t = 300)]
    pub silence_ms: u32,

    /// Input (capture) device name; the default input device is used when unset.
    #[arg(long, env = "SKADOOSH_INPUT_DEVICE")]
    pub input_device: Option<String>,

    /// Output (playback) device name; the default output device is used when unset.
    #[arg(long, env = "SKADOOSH_OUTPUT_DEVICE")]
    pub output_device: Option<String>,

    /// List available audio input/output devices and exit.
    #[arg(long)]
    pub list_devices: bool,

    /// Force the sine-wave mock TTS engine even if Kokoro files are present.
    #[arg(long, env = "SKADOOSH_MOCK_TTS")]
    pub mock_tts: bool,

    /// Run the headless self-test from a wav file (16-bit PCM, any rate),
    /// write `selftest_out.wav`, print the latency table, and exit.
    #[arg(long, env = "SKADOOSH_SELFTEST", value_name = "WAV_PATH")]
    pub selftest: Option<PathBuf>,
}

impl Config {
    /// Parses CLI arguments (with `SKADOOSH_*` env fallbacks). Exits the
    /// process with a usage message on parse failure, like clap's `parse`.
    pub fn parse() -> Config {
        <Config as Parser>::parse()
    }

    /// Validates the configuration:
    ///
    /// * `--vad-threshold` must lie in `[0, 1)`.
    /// * Whisper/VAD model files must exist (except for `--list-devices`).
    /// * The `--selftest` wav must exist when given.
    /// * Missing Kokoro model/voices files only produce a warning — the
    ///   pipeline falls back to MockTts (see [`crate::tts::build_engine`]).
    pub fn validate(&self) -> Result<()> {
        if self.list_devices {
            return Ok(());
        }

        if !(0.0..1.0).contains(&self.vad_threshold) {
            return Err(anyhow::anyhow!(
                "--vad-threshold must be in [0, 1), got {}",
                self.vad_threshold
            )
            .into());
        }

        for (flag, path) in [
            ("--whisper-model", &self.whisper_model),
            ("--vad-model", &self.vad_model),
        ] {
            if !path.exists() {
                return Err(anyhow::anyhow!(
                    "{flag} not found: {} (run scripts/download_models.sh)",
                    path.display()
                )
                .into());
            }
        }

        if let Some(wav) = &self.selftest {
            if !wav.exists() {
                return Err(anyhow::anyhow!("--selftest wav not found: {}", wav.display()).into());
            }
        }

        if !self.mock_tts {
            match (&self.tts_model, &self.tts_voices) {
                (Some(model), Some(voices)) if model.exists() && voices.exists() => {}
                (None, None) => {
                    tracing::warn!(
                        "no TTS model configured (--tts-model/--tts-voices); \
                         falling back to sine-wave MockTts"
                    );
                }
                (model, voices) => {
                    tracing::warn!(
                        tts_model = ?model,
                        tts_voices = ?voices,
                        "Kokoro TTS files absent or incomplete; falling back to \
                         sine-wave MockTts (run scripts/download_models.sh --with-kokoro)"
                    );
                }
            }
        }

        Ok(())
    }
}
